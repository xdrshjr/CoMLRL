import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from anthropic import Anthropic
from datasets import Dataset, IterableDataset
from openai import OpenAI
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

RewardFunc = Union[PreTrainedModel, Callable[[List[str]], float]]


@dataclass
class MTMAGRPOConfig:
    """Configuration for Multi-Turn MAGRPO training."""

    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    learning_rate: float = 5e-6  # Reduced for stability
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    num_generations: int = 4
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.01
    num_turns: int = 2  # Number of turns per episode
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    turn_gradient_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0]
    )  # Default: equal weights
    early_termination_weight: float = 2.0  # Default: 2x weight for early termination
    expert_model: str = "claude-3-5-sonnet-20241022"  # Expert model for feedback


def extract_last_json_from_response(response_text: str) -> Dict[str, str]:
    """
    Extract the last valid JSON from Claude's response.
    Returns dict with 'aux' and 'main' fields.
    """
    # Find all potential JSON blocks in the response
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    potential_jsons = re.findall(json_pattern, response_text, re.DOTALL)

    # Try parsing JSONs from last to first
    for json_str in reversed(potential_jsons):
        try:
            parsed = json.loads(json_str)
            # Validate that it has both required fields
            if "aux" in parsed and "main" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # If no valid JSON found, raise error
    raise ValueError("No valid JSON with 'aux' and 'main' fields found in response")


def get_expert_feedback(
    prompt: str,
    test: str,
    combined_code: str,
    best_reward: float,
    aux_completion: str,
    main_completion: str,
    entry_point: str,
    expert_model: str = "claude-3-5-sonnet-20241022",
    max_retries: int = 3,
) -> Tuple[str, str]:
    """
    Get feedback from Claude expert model.

    Args:
        expert_model: The Claude model to use for feedback

    Returns:
        Tuple of (aux_feedback, main_feedback)
    """

    expert_prompt = f"""You are an advisor helping two agents (an auxiliary agent and a main agent) solve the following problem: {prompt} There are some unit tests: {test} The auxiliary agent provides a helper function (aux), while the main agent defines the task-specific logic.
The current combined solution achieved a reward of {best_reward:.4f} / 4.0.
Your task is to review the provided code and return fixed codes. Specifically: 1. If you identify a missing element, such as an undefined aux or missing entry point (main function), you just rewrite one for it. 2. If both not missing, point out and make changes to any critical syntax or logic errors that would prevent the code from passing the given unit tests.
Important instructions: 1. You should focus only on clear errors on the given unit tests. 2. Be conservative and lenient: ignore issues like redundancy, inefficiency, lack of edge case handling, or type annotations unless they cause failure in the given unit tests. 3. If either function independently completes the task correctly, you don't need to specify this error for this function. 4. Return "Perfect! No changes needed!" if logics are sound.
IMPORTANT: Your response MUST contain the JSON format specified below. Always include both 'aux' and 'main' fields in the JSON, even if no changes are needed.
Show your feedback for the following code: {combined_code}
Respond in the following JSON format: {{ "aux": {{aux_func only here}}, "main": {{main_func only here}}, }}"""

    for attempt in range(max_retries):
        try:
            if "claude" in expert_model.lower():
                # Claude API
                client = Anthropic()
                response = client.messages.create(
                    model=expert_model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": expert_prompt}],
                )
                response_text = response.content[0].text

            elif "deepseek" in expert_model.lower():
                # DeepSeek API
                client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url="https://api.deepseek.com",
                )
                deepseek_model = (
                    "deepseek-coder"
                    if expert_model == "deepseek-coder"
                    else expert_model
                )
                response = client.chat.completions.create(
                    model=deepseek_model,
                    messages=[{"role": "user", "content": expert_prompt}],
                    max_tokens=2048,
                    temperature=0.3,  # [upd] Add temperature
                )

                # [upd] Correct way to extract content from DeepSeek/OpenAI response
                response_text = response.choices[0].message.content

            elif "qwen3-coder" in expert_model.lower():
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                )
                qwen_model = (
                    "qwen3-coder" if expert_model == "qwen3-coder" else expert_model
                )
                response = client.chat.completions.create(
                    model=qwen_model,
                    messages=[{"role": "user", "content": expert_prompt}],
                    max_tokens=2048,
                    temperature=0.3,  # [upd] Add temperature
                )

                response_text = response.choices[0].message.content

            else:
                raise ValueError(f"Unsupported expert model: {expert_model}")

            # Extract JSON from response
            feedback_json = extract_last_json_from_response(response_text)

            # Extract aux and main feedback
            aux_feedback = feedback_json.get("aux", aux_completion)
            main_feedback = feedback_json.get("main", main_completion)

            # Print both full response and extracted functions for visibility
            print("\n" + "=" * 60)
            print("EXPERT FEEDBACK")
            print("=" * 60)
            print(f"Best reward from previous turn: {best_reward:.4f}")
            print("\n--- FULL EXPERT RESPONSE ---")
            print(response_text)
            print("\n--- EXTRACTED EXPERT FEEDBACK ---")
            print("AUX FUNCTION:")
            print(aux_feedback)
            print("\nMAIN FUNCTION:")
            print(main_feedback)
            print("=" * 60 + "\n")

            return aux_feedback, main_feedback

        except Exception as e:
            print(f"Expert feedback attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                print("Max retries reached. Using original completions.")
                return aux_completion, main_completion


class MTMAGRPOTrainer:
    """
    Multi-Turn Multi-Agent Group Relative Policy Optimization Trainer (MT-MAGRPO) with Expert Feedback.
    Includes expert model feedback between turns to guide agent improvements.

    Args:
        agents: List of agent models
        num_agents: The number of agents
        reward_funcs: The reward functions for all agents
        reward_weights: The weights for each reward function
        reward_processors: Processors to apply to rewards
        formatters: Formatters to apply to dataset items for each agent
        args: The training arguments
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset
        tokenizer: The tokenizer
        wandb_config: Configuration for Weights & Biases logging
    """

    def __init__(
        self,
        agents: List[PreTrainedModel],
        num_agents: int = 2,
        reward_funcs: Union[RewardFunc, List[RewardFunc]] = None,
        reward_weights: Optional[List[float]] = None,
        reward_processors: Optional[List[Callable]] = None,
        formatters: Optional[Union[Callable, List[Callable]]] = None,
        args: Optional[MTMAGRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        if agents is None:
            raise ValueError("agents must be provided")

        self.agents = agents
        self.num_agents = len(agents)

        if self.num_agents != num_agents:
            raise ValueError(
                f"Length of agents ({self.num_agents}) must match num_agents ({num_agents})"
            )

        self._setup_formatters(formatters, num_agents)
        self._setup_reward_functions(reward_funcs, reward_weights, reward_processors)

        self.args = args if args is not None else MTMAGRPOConfig()

        if self.args.num_turns < 1:
            raise ValueError("num_turns must be at least 1")

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        if hasattr(agents[0], "config") and hasattr(agents[0].config, "model_type"):
            self.model_name = agents[0].config.model_type
        else:
            self.model_name = agents[0].__class__.__name__

        self.optimizers = [
            torch.optim.AdamW(
                agent.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            for agent in self.agents
        ]

        self.wandb_config = wandb_config
        self.wandb_initialized = False
        if self.wandb_config is not None:
            self._init_wandb()

    def _setup_formatters(self, formatters, num_agents):
        """Set up format functions for each agent."""
        default_format_func = lambda x, expert_feedback=None: x.get("prompt", "")

        if formatters is None:
            self.formatters = [default_format_func] * num_agents
        elif callable(formatters) and not isinstance(formatters, list):
            self.formatters = [formatters] * num_agents
        elif isinstance(formatters, list):
            if len(formatters) != num_agents:
                raise ValueError(
                    f"Number of formatters ({len(formatters)}) must match "
                    f"number of agents ({num_agents})"
                )
            self.formatters = formatters
        else:
            raise ValueError(
                f"formatters must be a callable, a list of callables, or None. "
                f"Got {type(formatters)}"
            )

    def _setup_reward_functions(
        self, reward_funcs, reward_weights=None, reward_processors=None
    ):
        """Set up reward functions with weights and processors."""
        if not isinstance(reward_funcs, list):
            self.reward_funcs = [reward_funcs]
        else:
            self.reward_funcs = reward_funcs

        if reward_weights is None:
            self.reward_weights = [1.0 / len(self.reward_funcs)] * len(
                self.reward_funcs
            )
        else:
            if len(reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(reward_weights)}) must match "
                    f"number of reward functions ({len(self.reward_funcs)})"
                )
            total = sum(reward_weights)
            self.reward_weights = [w / total for w in reward_weights]

        if reward_processors is None:
            self.reward_processors = [lambda x: x] * len(self.reward_funcs)
        elif not isinstance(reward_processors, list):
            self.reward_processors = [reward_processors] * len(self.reward_funcs)
        else:
            if len(reward_processors) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward processors ({len(reward_processors)}) must match "
                    f"number of reward functions ({len(self.reward_funcs)})"
                )
            self.reward_processors = []
            for processor in reward_processors:
                if processor is None:
                    self.reward_processors.append(lambda x: x)
                else:
                    self.reward_processors.append(processor)

    def _init_wandb(self):
        """Initialize Weights & Biases for tracking."""
        if not self.wandb_initialized:
            if self.wandb_config is None:
                self.wandb_config = {}

            wandb_project = self.wandb_config.get("project", "trl")
            wandb_entity = self.wandb_config.get("entity", "nu-llpr")
            wandb_name = self.wandb_config.get("name", "test-mt-magrpo")
            wandb_dir = self.wandb_config.get("dir", None)

            config_dict = {
                "model_name": self.model_name,
                "num_agents": self.num_agents,
                "num_turns": self.args.num_turns,
                "turn_gradient_weights": self.args.turn_gradient_weights,
                "early_termination_weight": self.args.early_termination_weight,
                "num_reward_functions": len(self.reward_funcs),
                "reward_weights": self.reward_weights,
                "learning_rate": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "num_train_epochs": self.args.num_train_epochs,
                "per_device_train_batch_size": self.args.per_device_train_batch_size,
                "num_generations": self.args.num_generations,
                "max_new_tokens": self.args.max_new_tokens,
            }

            init_kwargs = {
                "project": wandb_project,
                "entity": wandb_entity,
                "name": wandb_name,
                "config": config_dict,
            }

            if wandb_dir is not None:
                os.makedirs(wandb_dir, exist_ok=True)
                init_kwargs["dir"] = wandb_dir

            wandb.init(**init_kwargs)
            self.wandb_initialized = True

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Returns the evaluation DataLoader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=lambda examples: examples,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def evaluate(self, num_eval_samples: int = 4) -> Dict[str, float]:
        """
        Multi-turn evaluation on a subset of the evaluation dataset.

        Args:
            num_eval_samples: Number of samples to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.eval_dataset is None:
            return {}

        # Import the multi-turn logger
        from experiments.loggers.mt_code_logger import (
            aggregate_mt_humaneval_metrics_for_logging,
            mt_humaneval_logger,
        )

        eval_dataloader = self.get_eval_dataloader()

        # Storage for multi-turn completions
        all_completions1_turns = []  # [sample][turn]
        all_completions2_turns = []  # [sample][turn]
        all_test_cases = []
        all_entry_points = []
        all_prompts = []

        # Evaluate samples
        with torch.no_grad():
            for eval_idx, batch in enumerate(eval_dataloader):
                if eval_idx >= num_eval_samples:
                    break

                for batch_item in batch:
                    sample_completions1 = []
                    sample_completions2 = []

                    # Store sample information
                    all_test_cases.append(batch_item.get("test", ""))
                    all_entry_points.append(batch_item.get("entry_point", ""))
                    all_prompts.append(batch_item.get("prompt", ""))

                    # Store best completions from previous turn for expert feedback
                    previous_best_aux = None
                    previous_best_main = None
                    previous_best_reward = 0.0

                    # Run multi-turn episode
                    for turn_idx in range(self.args.num_turns):
                        # Prepare expert feedback for turns after the first
                        aux_expert_feedback = None
                        main_expert_feedback = None

                        if (
                            turn_idx > 0
                            and previous_best_aux is not None
                            and previous_best_main is not None
                        ):
                            # Get expert feedback based on previous turn's best result
                            from experiments.rewards.code_utils import (
                                concatenate_functions,
                                extract_imports_from_prompt,
                            )

                            imports = extract_imports_from_prompt(
                                batch_item.get("prompt", "")
                            )
                            combined_code = concatenate_functions(
                                previous_best_aux, previous_best_main, imports
                            )

                            aux_expert_feedback, main_expert_feedback = (
                                get_expert_feedback(
                                    prompt=batch_item.get("prompt", ""),
                                    test=batch_item.get("test", ""),
                                    combined_code=combined_code,
                                    best_reward=previous_best_reward,
                                    aux_completion=previous_best_aux,
                                    main_completion=previous_best_main,
                                    entry_point=batch_item.get("entry_point", ""),
                                    expert_model=self.args.expert_model,
                                )
                            )

                        # Generate one completion from each agent for evaluation
                        all_completions = []
                        expert_feedbacks = [aux_expert_feedback, main_expert_feedback]

                        for agent_idx in range(self.num_agents):
                            agent_completions = self._generate_completions(
                                self.agents[agent_idx],
                                [batch_item],
                                agent_idx=agent_idx,
                                num_return_sequences=1,  # Only one for evaluation
                                max_new_tokens=self.args.max_new_tokens,
                                expert_feedback=expert_feedbacks[agent_idx],
                            )
                            all_completions.append(agent_completions)

                        # Extract completions
                        completion1 = all_completions[0]["completions"][0][0]
                        completion2 = all_completions[1]["completions"][0][0]

                        sample_completions1.append(completion1)
                        sample_completions2.append(completion2)

                        # Check for early termination
                        rewards, _ = self._compute_rewards(
                            [all_completions[0]["prompts"][0]],
                            [[completion1], [completion2]],
                            batch_items=[batch_item],
                        )

                        if rewards:
                            previous_best_reward = rewards[0]
                            previous_best_aux = completion1
                            previous_best_main = completion2

                        if rewards[0] == 4.0:
                            # Early termination
                            break

                    all_completions1_turns.append(sample_completions1)
                    all_completions2_turns.append(sample_completions2)

        # Get detailed multi-turn metrics
        detailed_metrics = mt_humaneval_logger(
            all_completions1_turns,
            all_completions2_turns,
            all_test_cases,
            all_entry_points,
            all_prompts,
        )

        # Aggregate metrics
        aggregated_metrics = aggregate_mt_humaneval_metrics_for_logging(
            detailed_metrics, num_turns=self.args.num_turns
        )

        # Add eval prefix to all metrics
        eval_metrics = {}
        for key, value in aggregated_metrics.items():
            eval_metrics[f"eval/{key}"] = value

        # Log evaluation metrics
        if self.wandb_initialized:
            wandb.log(eval_metrics)

        return eval_metrics

    def train(self, **kwargs):
        """
        Train the multi-turn multi-agent model with expert feedback.
        """
        if self.wandb_config is not None and not self.wandb_initialized:
            self._init_wandb()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for agent in self.agents:
            agent.to(device)
            agent.train()

        # Track epoch rewards for conditional saving
        epoch_rewards_history = []

        for epoch in range(int(self.args.num_train_epochs)):
            epoch_loss = 0.0
            epoch_rewards = []
            epoch_turn_rewards = [[] for _ in range(self.args.num_turns)]
            epoch_early_terminations = 0

            for batch_idx, batch in enumerate(self.get_train_dataloader()):
                # Evaluate every 4 batches
                if batch_idx % 4 == 0:
                    eval_metrics = self.evaluate(num_eval_samples=4)
                    wandb.log(eval_metrics)

                # Process each batch item (should be 1 with batch_size=1)
                for item_idx, batch_item in enumerate(batch):
                    # Store turn data for sequential updates
                    turn_data = []
                    early_termination = False

                    # Store best completions from previous turn for expert feedback
                    previous_best_aux = None
                    previous_best_main = None
                    previous_best_reward = 0.0

                    # Execute multi-turn episode
                    for turn_idx in range(self.args.num_turns):
                        # Zero gradients for each agent at the start of each turn
                        for optimizer in self.optimizers:
                            optimizer.zero_grad()

                        # Prepare expert feedback for turns after the first
                        aux_expert_feedback = None
                        main_expert_feedback = None

                        if (
                            turn_idx > 0
                            and previous_best_aux is not None
                            and previous_best_main is not None
                        ):
                            # Get expert feedback based on previous turn's best result
                            from experiments.rewards.code_utils import (
                                concatenate_functions,
                                extract_imports_from_prompt,
                            )

                            imports = extract_imports_from_prompt(
                                batch_item.get("prompt", "")
                            )
                            combined_code = concatenate_functions(
                                previous_best_aux, previous_best_main, imports
                            )

                            aux_expert_feedback, main_expert_feedback = (
                                get_expert_feedback(
                                    prompt=batch_item.get("prompt", ""),
                                    test=batch_item.get("test", ""),
                                    combined_code=combined_code,
                                    best_reward=previous_best_reward,
                                    aux_completion=previous_best_aux,
                                    main_completion=previous_best_main,
                                    entry_point=batch_item.get("entry_point", ""),
                                    expert_model=self.args.expert_model,
                                )
                            )

                        # Generate completions from each agent
                        all_completions = []
                        expert_feedbacks = [aux_expert_feedback, main_expert_feedback]

                        for agent_idx in range(self.num_agents):
                            agent_completions = self._generate_completions(
                                self.agents[agent_idx],
                                [batch_item],
                                agent_idx=agent_idx,
                                num_return_sequences=self.args.num_generations,
                                max_new_tokens=self.args.max_new_tokens,
                                expert_feedback=expert_feedbacks[agent_idx],
                                **kwargs,
                            )
                            all_completions.append(agent_completions)

                        # Extract completions for reward calculation
                        agent_completions_list = []
                        for agent_idx in range(self.num_agents):
                            agent_completions_list.append(
                                all_completions[agent_idx]["completions"][0]
                            )

                        # Get formatted prompt
                        formatted_prompt = all_completions[0]["prompts"][0]

                        # Compute rewards
                        rewards, reward_components = self._compute_rewards(
                            [formatted_prompt],
                            agent_completions_list,
                            batch_items=[batch_item],
                        )

                        # Find the best completion pair (highest reward)
                        if rewards:
                            best_idx = rewards.index(max(rewards))
                            previous_best_aux = agent_completions_list[0][best_idx]
                            previous_best_main = agent_completions_list[1][best_idx]
                            previous_best_reward = rewards[best_idx]

                        # Calculate turn mean reward
                        turn_mean_reward = np.mean(rewards) if rewards else 0
                        epoch_turn_rewards[turn_idx].append(turn_mean_reward)

                        # Store turn data
                        turn_data.append(
                            {
                                "completions": all_completions,
                                "rewards": rewards,
                                "reward_components": reward_components,
                                "mean_reward": turn_mean_reward,
                            }
                        )

                        # Log turn metrics
                        turn_log_data = {
                            f"turn_{turn_idx + 1}/batch_rewards_mean": turn_mean_reward,
                        }

                        # Log reward components
                        for i, component in enumerate(reward_components):
                            component_mean = np.mean(component) if component else 0
                            turn_log_data[
                                f"turn_{turn_idx + 1}/reward_{i + 1}_mean"
                            ] = component_mean

                        # Check for early termination
                        if turn_mean_reward == 4.0:
                            early_termination = True
                            epoch_early_terminations += 1
                            turn_log_data[f"turn_{turn_idx + 1}/early_termination"] = 1
                            wandb.log(turn_log_data)

                            # Fill remaining turns with max values
                            for future_turn in range(turn_idx + 1, self.args.num_turns):
                                # Log perfect metrics for skipped turns
                                perfect_metrics = {
                                    f"turn_{future_turn + 1}/batch_rewards_mean": 4.0,
                                    f"turn_{future_turn + 1}/early_termination_filled": 1,
                                    f"turn_{future_turn + 1}/reward_1_mean": 4.0,  # Assuming single reward function
                                }
                                wandb.log(perfect_metrics)
                                epoch_turn_rewards[future_turn].append(4.0)

                                # Log improvement as 0
                                if future_turn > 0:
                                    wandb.log(
                                        {
                                            f"turn_{future_turn + 1}/improvement_from_turn_{future_turn}": 0.0
                                        }
                                    )

                            break

                        wandb.log(turn_log_data)

                    # Log turn-to-turn improvements
                    if len(turn_data) >= 2:
                        for i in range(1, len(turn_data)):
                            improvement = (
                                turn_data[i]["mean_reward"]
                                - turn_data[i - 1]["mean_reward"]
                            )
                            wandb.log(
                                {f"turn_{i + 1}/improvement_from_turn_{i}": improvement}
                            )
                    elif early_termination and self.args.num_turns >= 2:
                        # If early terminated at turn 1, still log 0 improvement for turn 2
                        wandb.log({f"turn_2/improvement_from_turn_1": 0.0})

                    # Sequential model updates after episode ends
                    batch_loss = 0.0

                    # Use turn-specific gradient weights from config
                    turn_weights = self.args.turn_gradient_weights

                    for turn_idx, turn_info in enumerate(turn_data):
                        # Get turn-specific weight
                        turn_gradient_weight = (
                            turn_weights[turn_idx]
                            if turn_idx < len(turn_weights)
                            else 1.0
                        )

                        # Apply termination weight if needed (for early termination bonus)
                        if early_termination and turn_idx == len(turn_data) - 1:
                            # Combine turn weight with early termination weight
                            final_weight = (
                                turn_gradient_weight
                                * self.args.early_termination_weight
                            )
                        else:
                            final_weight = turn_gradient_weight

                        # Update each agent for this turn
                        turn_loss = 0.0
                        agent_losses = []

                        for agent_idx in range(self.num_agents):
                            # Compute loss with gradients
                            agent_loss = self._compute_loss_with_gradients(
                                self.agents[agent_idx],
                                turn_info["completions"][agent_idx],
                                turn_info["rewards"],
                            )

                            # Apply weight
                            weighted_loss = agent_loss * final_weight

                            # Backward pass and optimization
                            weighted_loss.backward()
                            self.optimizers[agent_idx].step()
                            self.optimizers[agent_idx].zero_grad()

                            turn_loss += agent_loss.detach().item()
                            agent_losses.append(agent_loss.detach().item())

                        batch_loss += turn_loss

                        # Log turn update info
                        wandb.log(
                            {
                                f"turn_{turn_idx + 1}/update_loss": turn_loss,
                                f"turn_{turn_idx + 1}/gradient_weight": turn_gradient_weight,
                                f"turn_{turn_idx + 1}/final_weight": final_weight,
                            }
                        )

                    epoch_loss += batch_loss

                    # Collect all rewards for epoch tracking
                    for turn_info in turn_data:
                        epoch_rewards.extend(turn_info["rewards"])

                    # Log episode summary
                    wandb.log(
                        {
                            "episode_loss": batch_loss,
                            "episode_num_turns": len(turn_data),
                            "episode_early_termination": early_termination,
                        }
                    )

            # Log epoch summary
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            epoch_rewards_history.append(avg_reward)

            epoch_log = {
                "epoch": epoch,
                "epoch_loss": (
                    epoch_loss / len(self.get_train_dataloader()) if epoch_loss else 0
                ),
                "epoch_avg_reward": avg_reward,
                "epoch_early_termination_rate": epoch_early_terminations
                / len(self.get_train_dataloader()),
            }

            # Log average rewards per turn
            for turn_idx in range(self.args.num_turns):
                if epoch_turn_rewards[turn_idx]:
                    epoch_log[f"epoch_turn_{turn_idx + 1}_avg_reward"] = np.mean(
                        epoch_turn_rewards[turn_idx]
                    )

            wandb.log(epoch_log)

    def _generate_completions(
        self,
        agent,
        batch_items,
        agent_idx=0,
        num_return_sequences=1,
        max_new_tokens=128,
        expert_feedback=None,
        **kwargs,
    ):
        """Generate completions from an agent with optional expert feedback."""
        device = agent.device

        format_func = self.formatters[agent_idx]

        # Apply formatter with expert feedback if provided
        prompts = [
            format_func(item, expert_feedback=expert_feedback) for item in batch_items
        ]
        batch_size = len(prompts)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating completions")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt_encodings = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        prompt_input_ids = prompt_encodings.input_ids
        prompt_attention_mask = prompt_encodings.attention_mask

        # Store original model state
        training_mode = agent.training
        original_requires_grad = {}

        for name, param in agent.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        agent.eval()

        generation_output = None
        try:
            generation_kwargs = {
                "input_ids": prompt_input_ids,
                "attention_mask": prompt_attention_mask,
                "max_new_tokens": max_new_tokens,
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            if num_return_sequences > 1:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": self.args.temperature,
                        "top_p": self.args.top_p,
                        "top_k": 50,
                        "num_beams": 1,
                        "num_return_sequences": num_return_sequences,
                    }
                )

            generation_kwargs.update(kwargs)
            generation_output = agent.generate(**generation_kwargs)

        except Exception as e:
            agent.train(training_mode)
            for name, param in agent.named_parameters():
                if name in original_requires_grad:
                    param.requires_grad = original_requires_grad[name]
            raise ValueError(f"Generation failed: {str(e)}")

        # Restore model state
        agent.train(training_mode)
        for name, param in agent.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        # Extract completion tokens
        completion_input_ids = generation_output.sequences

        prompt_lengths = []
        for b in range(batch_size):
            prompt_len = prompt_input_ids[b].shape[0]
            pad_positions = (
                prompt_input_ids[b] == self.tokenizer.pad_token_id
            ).nonzero()
            if pad_positions.shape[0] > 0:
                prompt_len = pad_positions[0].item()
            prompt_lengths.append(prompt_len)

        completions = []
        completion_tokens_list = []

        total_sequences = completion_input_ids.shape[0]

        for b in range(batch_size):
            prompt_len = prompt_lengths[b]
            batch_completions = []
            batch_completion_tokens = []

            start_idx = b * num_return_sequences
            end_idx = min(start_idx + num_return_sequences, total_sequences)

            for s in range(start_idx, end_idx):
                completion_tokens = completion_input_ids[s, prompt_len:]
                batch_completion_tokens.append(completion_tokens)

                completion_text = self.tokenizer.decode(
                    completion_tokens, skip_special_tokens=True
                )
                batch_completions.append(completion_text)

            completions.append(batch_completions)
            completion_tokens_list.append(batch_completion_tokens)

        # Create attention masks
        completion_attention_masks = []
        for batch_tokens in completion_tokens_list:
            batch_masks = []
            for tokens in batch_tokens:
                mask = torch.ones(len(tokens), device=device)
                batch_masks.append(mask)
            completion_attention_masks.append(batch_masks)

        logits = (
            generation_output.scores if hasattr(generation_output, "scores") else []
        )

        return {
            "prompts": prompts,
            "batch_items": batch_items,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "completions": completions,
            "completion_input_ids": completion_tokens_list,
            "completion_attention_mask": completion_attention_masks,
            "logits": logits,
        }

    def _compute_rewards(
        self, prompts, completions_list, batch_items=None
    ) -> Tuple[List[float], List[List[float]]]:
        """Compute rewards (same as original MAGRPO)."""
        all_rewards = []
        all_reward_components = [[] for _ in range(len(self.reward_funcs))]

        if len(prompts) == 1:
            for i in range(self.num_agents):
                if not isinstance(completions_list[i], list):
                    completions_list[i] = [completions_list[i]]

            min_completions = min(
                len(completions_list[i]) for i in range(self.num_agents)
            )

            for completion_idx in range(min_completions):
                agent_completions = [
                    completions_list[agent_idx][completion_idx]
                    for agent_idx in range(self.num_agents)
                ]

                weighted_reward = 0.0
                reward_components = []

                for func_idx, (reward_func, weight, processor) in enumerate(
                    zip(self.reward_funcs, self.reward_weights, self.reward_processors)
                ):
                    try:
                        completion_args = [[comp] for comp in agent_completions]

                        import inspect

                        sig = inspect.signature(reward_func)
                        if "batch_items" in sig.parameters:
                            func_rewards = reward_func(
                                *completion_args, batch_items=batch_items
                            )
                        else:
                            func_rewards = reward_func(*completion_args)
                    except TypeError:
                        func_rewards = reward_func(agent_completions)

                    processed_rewards = [processor(r) for r in func_rewards]
                    reward_components.append(processed_rewards[0])
                    all_reward_components[func_idx].extend(processed_rewards)
                    weighted_reward += weight * processed_rewards[0]

                all_rewards.append(weighted_reward)

            return all_rewards, all_reward_components
        else:
            # Batch processing
            agent_completions_lists = [[] for _ in range(self.num_agents)]

            for prompt_idx in range(len(prompts)):
                for agent_idx in range(self.num_agents):
                    if prompt_idx < len(completions_list[agent_idx]):
                        agent_completion = (
                            completions_list[agent_idx][prompt_idx][0]
                            if isinstance(completions_list[agent_idx][prompt_idx], list)
                            else completions_list[agent_idx][prompt_idx]
                        )
                        agent_completions_lists[agent_idx].append(agent_completion)

            weighted_rewards = [0.0] * len(agent_completions_lists[0])

            for func_idx, (reward_func, weight, processor) in enumerate(
                zip(self.reward_funcs, self.reward_weights, self.reward_processors)
            ):
                try:
                    batch_rewards = reward_func(*agent_completions_lists)
                except TypeError:
                    batch_rewards = reward_func(agent_completions_lists)

                processed_rewards = [processor(r) for r in batch_rewards]
                all_reward_components[func_idx].extend(processed_rewards)

                for i, r in enumerate(processed_rewards):
                    if i < len(weighted_rewards):
                        weighted_rewards[i] += weight * r

            return weighted_rewards, all_reward_components

    def _compute_loss_with_gradients(self, agent, completions_data, rewards):
        """Compute loss with gradients (same as original MAGRPO)."""
        device = agent.device

        if len(rewards) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        rewards_baseline = rewards_tensor.mean()
        advantages = rewards_tensor - rewards_baseline
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        agent.train()

        prompt_input_ids = completions_data["prompt_input_ids"]
        prompt_attention_mask = completions_data["prompt_attention_mask"]
        completion_input_ids = completions_data["completion_input_ids"]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_samples = 0

        for batch_idx in range(len(prompt_input_ids)):
            prompt_ids = prompt_input_ids[batch_idx]
            prompt_mask = prompt_attention_mask[batch_idx]

            for seq_idx, completion_tokens in enumerate(
                completion_input_ids[batch_idx]
            ):
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

                    completion_logits = outputs.logits[
                        0, prompt_ids.size(0) - 1 : -1, :
                    ]

                    log_probs = []
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

    def save_model(self, output_dir):
        """Save the trained models."""
        os.makedirs(output_dir, exist_ok=True)

        for agent_idx, agent in enumerate(self.agents):
            agent_dir = f"{output_dir}/agent_{agent_idx}"
            os.makedirs(agent_dir, exist_ok=True)
            agent.save_pretrained(agent_dir)

            if self.tokenizer:
                self.tokenizer.save_pretrained(agent_dir)

        if self.wandb_initialized:
            wandb.log({"final_model_saved": output_dir})
            wandb.finish()
