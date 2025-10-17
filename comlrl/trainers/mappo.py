import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .magrpo import MAGRPOConfig, MAGRPOTrainer


@dataclass
class MAPPOConfig(MAGRPOConfig):
    """
    Configuration for MAPPO training.

    Inherits all settings from MAGRPOConfig; behavior is identical except that
    the policy advantage is computed with a learned value (critic) baseline
    produced by a separate LLM.

    Additional critic-related hyperparameters are provided via the trainer
    init arguments rather than this dataclass to keep compatibility with
    MAGRPOConfig usage in existing scripts.
    """


class _ValueHead(torch.nn.Module):
    """A small linear head to regress a scalar value from LLM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, H], take last token and cast to proj dtype
        last = hidden_states[:, -1, :]
        if last.dtype != self.proj.weight.dtype:
            last = last.to(self.proj.weight.dtype)
        v = self.proj(last)  # [B, 1]
        return v.squeeze(-1)  # [B]


class MAPPOTrainer(MAGRPOTrainer):
    """
    Multi-Agent Proximal Policy Optimization variant with a learned value baseline.

    Differences from MAGRPOTrainer:
    - Uses a separate LLM as a critic to predict a scalar value V(s) per node/state
      (where state is the formatted prompt for the current turn).
    - Policy advantages are computed as A_g = R_g - V(s) for each generation g.
    - The same advantages are applied uniformly across agents (like MAGRPO).

    Critic details:
    - The critic model is any `AutoModelForCausalLM`-compatible LLM, paired with a
      lightweight value head. By default, only the value head is trained; optionally,
      the base LLM can be unfrozen via `value_train_base=True`.
    """

    def __init__(
        self,
        *args,
        value_model: Optional[Union[str, PreTrainedModel]] = None,
        value_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        value_model_config: Optional[Dict[str, Any]] = None,
        value_learning_rate: float = 1e-4,
        value_weight_decay: float = 0.0,
        value_train_base: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Build value model and tokenizer
        self.value_model_config = value_model_config or {}
        if isinstance(value_model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.value_model = AutoModelForCausalLM.from_pretrained(
                value_model, **self.value_model_config.get("model_kwargs", {})
            )
            if value_tokenizer is None:
                self.value_tokenizer = AutoTokenizer.from_pretrained(
                    value_model, **self.value_model_config.get("tokenizer_kwargs", {})
                )
            else:
                self.value_tokenizer = value_tokenizer
        elif isinstance(value_model, PreTrainedModel):
            self.value_model = value_model
            if value_tokenizer is None:
                raise ValueError(
                    "value_tokenizer must be provided when passing a value_model instance"
                )
            self.value_tokenizer = value_tokenizer
        else:
            raise ValueError(
                "value_model must be a model name (str) or a PreTrainedModel instance"
            )

        # Create value head sized to the LLM embeddings
        hidden_size = None
        try:
            if hasattr(self.value_model.config, "hidden_size"):
                hidden_size = int(self.value_model.config.hidden_size)
            elif hasattr(self.value_model.config, "n_embd"):
                hidden_size = int(self.value_model.config.n_embd)
        except Exception:
            hidden_size = None
        if hidden_size is None:
            try:
                hidden_size = int(self.value_model.get_input_embeddings().embedding_dim)
            except Exception:
                raise ValueError("Cannot infer hidden size for value head.")

        self.value_head = _ValueHead(hidden_size)

        # Freeze base critic LLM if not training base; train only head by default
        if not value_train_base:
            for p in self.value_model.parameters():
                p.requires_grad = False

        # Optimizer for critic (value head + optionally base)
        trainable_params = list(self.value_head.parameters())
        if value_train_base:
            trainable_params += [
                p for p in self.value_model.parameters() if p.requires_grad
            ]
        self.value_optimizer = torch.optim.AdamW(
            trainable_params, lr=value_learning_rate, weight_decay=value_weight_decay
        )

        # Move to device and set modes; match value head dtype to base model
        device = torch.device("cuda")
        self.value_model.to(device)
        try:
            base_dtype = next(self.value_model.parameters()).dtype
        except StopIteration:
            base_dtype = torch.float32
        self.value_head.to(device=device, dtype=base_dtype)
        self.value_model.train()
        self.value_head.train()

    def _predict_state_value(self, prompt_text: str) -> torch.Tensor:
        """Encode the state prompt with the critic LLM and regress a scalar value."""
        device = torch.device("cuda")
        if self.value_tokenizer.pad_token is None:
            self.value_tokenizer.pad_token = self.value_tokenizer.eos_token

        enc = self.value_tokenizer(
            [prompt_text], padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        with torch.set_grad_enabled(
            self.value_head.proj.weight.requires_grad
            or any(p.requires_grad for p in self.value_model.parameters())
        ):
            outputs = self.value_model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                output_hidden_states=True,
            )
            # Prefer last_hidden_state if available; fall back to hidden_states[-1]
            if (
                hasattr(outputs, "last_hidden_state")
                and outputs.last_hidden_state is not None
            ):
                hidden = outputs.last_hidden_state  # [B, T, H]
            elif (
                hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
            ):
                hidden = outputs.hidden_states[-1]
            else:
                raise ValueError("Critic model did not return hidden states.")

            v = self.value_head(hidden)  # [B]
            return v.squeeze(0)  # scalar tensor

    def _compute_loss_with_gradients(self, agent, completions_data, returns):
        """
        Compute policy loss using critic baseline, and update the critic.

        Advantage per generation: A_g = R_g - V(s), where V(s) is predicted from
        the node prompt (state). The same advantages are then used for each agent.
        The critic is updated by minimizing MSE between V(s) and each R_g.
        """
        device = agent.device

        if len(returns) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)

        # State prompt text is stored in completions_data["prompts"][0]
        prompt_text = completions_data["prompts"][0]

        # Predict scalar value for the state
        self.value_model.train()
        self.value_head.train()
        v_s = self._predict_state_value(
            prompt_text
        )  # scalar on CUDA (dtype ~= critic base)

        # Critic loss (fit V(s) to each sibling return) in critic dtype
        critic_target = returns_tensor.to(dtype=v_s.dtype)
        critic_loss = torch.mean((v_s - critic_target) ** 2)

        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()

        # Advantages using critic baseline, cast to float32 for policy loss
        advantages = returns_tensor - v_s.detach().to(dtype=returns_tensor.dtype)

        # Policy gradient loss for the given agent
        agent.train()
        prompt_input_ids = completions_data["prompt_input_ids"]
        completion_input_ids = completions_data["completion_input_ids"]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        prompt_ids = prompt_input_ids[0]
        for seq_idx, completion_tokens in enumerate(completion_input_ids[0]):
            if seq_idx >= len(advantages):
                break
            advantage = advantages[seq_idx]
            if len(completion_tokens) == 0:
                continue

            input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
            target_ids = completion_tokens
            attention_mask = torch.ones(len(input_ids), device=device)

            outputs = agent(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            )
            completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

            log_probs: List[torch.Tensor] = []
            for i, token_id in enumerate(target_ids):
                if i < completion_logits.size(0):
                    token_logits = completion_logits[i]
                    token_log_prob = torch.log_softmax(token_logits, dim=-1)[token_id]
                    log_probs.append(token_log_prob)

            if log_probs:
                sequence_log_prob = torch.stack(log_probs).sum()
                loss = -sequence_log_prob * advantage
                total_loss = total_loss + loss
                num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.1, device=device, requires_grad=True)

        return total_loss
