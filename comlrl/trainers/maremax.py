from dataclasses import dataclass
from typing import List

import torch

from .magrpo import MAGRPOConfig, MAGRPOTrainer


@dataclass
class MAReMaxConfig(MAGRPOConfig):
    """
    Configuration for MAReMax training.

    Inherits all settings from MAGRPOConfig; behavior is identical to MAGRPO
    except for the advantage computation, which uses a max-baseline across
    generations: A_g = R_g - max(R_1..R_G).
    """


class MAReMaxTrainer(MAGRPOTrainer):
    """
    Multi-Agent Return Max-Baseline (MAReMax) Trainer.

    Identical to MAGRPOTrainer except the advantage is computed with a
    max baseline over generations:

        A_g = R_g - max_k R_k

    The resulting advantage per generation is applied uniformly to each agent,
    same as in MAGRPOTrainer.
    """

    def _compute_loss_with_gradients(self, agent, completions_data, returns):
        """
        Compute loss with a max baseline for advantages.

        Args:
            agent: The agent model
            completions_data: The completions data from _generate_completions
            returns: The returns for each completion (not immediate rewards)

        Returns:
            torch.Tensor: The computed loss with gradients attached
        """
        device = agent.device

        if len(returns) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Convert returns to tensor
        returns_tensor = torch.tensor(returns, dtype=torch.float, device=device)

        # Max-baseline advantage: A_g = R_g - max(R)
        max_ret = returns_tensor.max()
        advantages = returns_tensor - max_ret

        # Set agent to train mode to ensure gradients are tracked
        agent.train()

        prompt_input_ids = completions_data["prompt_input_ids"]
        completion_input_ids = completions_data["completion_input_ids"]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        # Single prompt (batch_size=1)
        prompt_ids = prompt_input_ids[0]

        # Process each generated completion for this prompt
        for seq_idx, completion_tokens in enumerate(completion_input_ids[0]):
            if seq_idx >= len(advantages):
                break

            advantage = advantages[seq_idx]

            if len(completion_tokens) > 0:
                input_ids = torch.cat([prompt_ids, completion_tokens[:-1]])
                target_ids = completion_tokens
                attention_mask = torch.ones(len(input_ids), device=device)

                outputs = agent(
                    input_ids=input_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )

                # Get logits for the completion part (excluding prompt)
                completion_logits = outputs.logits[0, prompt_ids.size(0) - 1 : -1, :]

                # Calculate log probabilities over the completion
                log_probs: List[torch.Tensor] = []
                for i, token_id in enumerate(target_ids):
                    if i < completion_logits.size(0):
                        token_logits = completion_logits[i]
                        token_log_prob = torch.log_softmax(token_logits, dim=-1)[
                            token_id
                        ]
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
