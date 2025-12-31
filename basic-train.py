import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer
from comlrl.utils import setup_logger

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train MAGRPO model")
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Logging level (default: INFO)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=3,
    help="Number of training epochs (default: 3)",
)
args = parser.parse_args()

# Setup logging with user-specified level
logger = setup_logger("comlrl", level=args.log_level)
logger.info(f"Starting training with log level: {args.log_level}")
logger.info(f"Training will run for {args.epochs} epochs")

# model_name = "Qwen/Qwen2.5-0.5B"
model_name = "/mnt/mydisk/models/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"

# Load dataset and tokenizer
logger.info(f"Loading model and tokenizer from: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
logger.info("Loading dataset: trl-lib/tldr")
dataset = load_dataset("trl-lib/tldr", split="train").select(range(128))
logger.info(f"Dataset loaded with {len(dataset)} samples")

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
logger.info("Initializing MAGRPOTrainer")
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
        num_train_epochs=args.epochs,
    ),
)
logger.info("Starting training")
trainer.train()
logger.info("Training completed successfully")