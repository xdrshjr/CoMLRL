from datasets import load_dataset
from transformers import AutoTokenizer
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer

# model_name = "Qwen/Qwen2.5-0.5B"
model_name = "/mnt/mydisk/models/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"

# Load dataset and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("trl-lib/tldr", split="train").select(range(128))

wandb_config = {
    "project": "magnpo-experiment",
    "entity": "xdrshjr",
    "name": "test-magrpo",
    "tags": ["magnpo", "multi-agent"],
    "config_sections": {
        "dataset": {"name": "humaneval", "type": "code"},
        "model": {"name": "Qwen/Qwen2.5-0.5B"},
        "output": {"verbose": True}
    }
}


# Initialize trainer and start training
trainer = MAGRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    num_agents=2,
    tokenizer=tokenizer,
    train_dataset=dataset,
    reward_func=lambda a, b: [abs(max(len(b[0]), 1) / max(len(a[0]), 1) - 3.0)],
    formatters=[lambda example: example["prompt"]] * 2,
    wandb_config=wandb_config,
    args=MAGRPOConfig(
        per_device_train_batch_size=1,
    ),
)
trainer.train()