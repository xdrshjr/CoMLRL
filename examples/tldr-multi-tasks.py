# Shuo: Tested on an A40, at least 40GB VRAM is required

from functools import partial

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer

from .rewards.texts_comparison import (
    proper_length_ratio_reward,
    vocabulary_richness_reward,
)


def example_usage():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = MAGRPOConfig(
        output_dir="./magrpo_multi_reward_output",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        num_generations=8,
        max_new_tokens=256,
    )

    dataset_name = "trl-lib/tldr"
    dataset_split = "train[:100]"
    train_dataset = load_dataset(dataset_name, split=dataset_split)

    wandb_config = {
        "project": "mlrl",
        "entity": "nu-llpr",
        "name": "qwen-magrpo-multi-reward",
    }

    configured_proper_length_reward = partial(
        proper_length_ratio_reward, target_min=2, target_max=3
    )
    reward_funcs = [
        configured_proper_length_reward,
        vocabulary_richness_reward,
    ]
    reward_weights = [
        0.3,
        0.7,
    ]

    # fmt: off
    agents = []
    use_peft = False
    for _ in range(2):
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_peft:
            lora_config = LoraConfig(
                r=1024,
                lora_alpha=2048,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=["embed_tokens", "lm_head"],
                fan_in_fan_out=False,
                task_type=TaskType.CAUSAL_LM,
            )
            lora_model = get_peft_model(base_model, lora_config)
            lora_model.print_trainable_parameters()
            agents.append(lora_model)
        else:
            agents.append(base_model)

    trainer = MAGRPOTrainer(
        agents=agents,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        wandb_config=wandb_config,
    )

    trainer.train()
    trainer.save_model(f"{config.output_dir}/final_models")
    print("Training complete!")


if __name__ == "__main__":
    example_usage()
