# Shuo: Tested 2 * Qwen2.5-1.5B model on a A100, at least 70GB VRAM is required

import os
from functools import partial
from typing import Any, Dict

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from examples.rewards.codes_generation import execution_reward
from comlrl.trainers.magrpo import MAGRPOConfig, MAGRPOTrainer


def create_code_dataset():
    """
    Create a small dataset with coding tasks.
    Each entry contains a question, expected input/output examples, and function name.
    """
    data = {
        "question": [
            "Create a function that checks if a given string is a palindrome (reads the same forwards and backwards).",
            "Write a function that returns the nth Fibonacci number in the sequence (where F(0)=0, F(1)=1, and each subsequent number is the sum of the two preceding ones).",
            "Create a function that counts the number of vowels in a string.",
            "Write a function that checks if a number is prime.",
        ],
        "input": [
            "'radar', 'hello', 'lol'",
            "2, 5, 10",
            "'hello', 'world', 'aeiou'",
            "3, 6, 11",
        ],
        "output": [
            "True, False, True",
            "1, 5, 55",
            "2, 1, 5",
            "True, False, True",
        ],
        "function_name": [
            "is_palindrome",
            "fibonacci",
            "count_vowels",
            "is_prime",
        ],
    }
    return Dataset.from_dict(data)


def function_writer_formatter(example: Dict[str, Any]) -> str:
    """
    Format a prompt for the function-writing agent (Agent 1).

    Args:
        example: A dictionary containing the question, function name, and other details

    Returns:
        A formatted prompt string
    """
    question = example.get("question", "")
    function_name = example.get("function_name", "")
    input_example = example.get("input", "")
    output_example = example.get("output", "")

    prompt = (
        "You are asked to implement a Python function based on the following requirement:\n\n"
        f"Question: {question}\n\n"
        f"Function name: {function_name}\n\n"
    )

    if input_example:
        prompt += f"Example inputs: {input_example}\n\n"

    if output_example:
        prompt += f"Expected outputs: {output_example}\n\n"

    prompt += (
        "IMPORTANT REQUIREMENTS:\n\n"
        "1. Write a single, self-contained function with the exact name specified above.\n"
        "2. Include all helper code inside your main function - do not use unknown variables or functions.\n"
        "3. Ensure your function handles data types correctly - if numeric inputs are needed, convert string inputs to the right type.\n"
        "4. Include appropriate docstrings and comments.\n"
        "5. Make sure your function handles edge cases.\n"
        "6. Do not include any usage examples, main function, or explanations outside the function code.\n"
        "7. Do not add any explanatory text before or after the function - only provide the function code itself.\n\n"
        "Your solution must be completely self-contained within a single function."
    )

    return prompt


def example_writer_formatter(
    example: Dict[str, Any], generate_new_examples: bool = True
) -> str:
    """
    Format a prompt for the example-writing agent (Agent 2).

    Args:
        example: A dictionary containing the question, function name, input, output, and other details
        generate_new_examples: A boolean indicating whether to generate new examples or just use the provided ones

    Returns:
        A formatted prompt string
    """
    question = example.get("question", "")
    function_name = example.get("function_name", "")
    input_example = example.get("input", "")
    output_example = example.get("output", "")

    input_type_hint = ""
    if input_example:
        inputs = input_example.split(",")
        if any(i.strip().isdigit() for i in inputs):
            input_type_hint = (
                "\nIMPORTANT: For numeric inputs, use integers without quotes"
            )

    prompt = (
        "You are asked to write usage examples for a Python function:\n\n"
        f"Question: {question}\n\n"
        f"Function name: {function_name}\n\n"
    )

    if input_example:
        prompt += f"Example inputs: {input_example}\n\n"

    if output_example:
        prompt += f"Expected outputs: {output_example}\n\n"

    prompt += (
        "IMPORTANT REQUIREMENTS:\n\n"
        "1. Write code that demonstrates how to use the function with ONLY real test examples.\n"
    )
    if generate_new_examples:
        prompt += "2. The test examples should include 3 cases above, and you need to design 2 edge cases by yourself, so 5 tests in total.\n"
    else:
        prompt += "2. The test examples should be the 3 cases above.\n"

    prompt += (
        "3. Assume the function has already been defined - DO NOT implement the function itself.\n"
        "4. Use actual values, not placeholders like 'example_input', 'input', 'result', etc.\n"
        "5. Include print statements with the function calls and expected output comments.\n"
        "6. Show a variety of inputs that test different cases.\n"
        "7. Expected output comments should contain only the result, no explanations.\n"
        f"{input_type_hint}\n\n"
        "Example format:\n"
        "```python\n"
        "# Example usage\n"
        f"print({function_name}(input))  # Expected output: result\n"
        "```\n\n"
        "REMEMBER: DO NOT redefine the function or include the function code - assume it already exists."
    )

    return prompt


def main():
    """Main function to run the experiment."""
    output_dir = "../../../projects/bepg/sliu30/output"
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = create_code_dataset()

    model_name = "Qwen/Qwen2.5-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = MAGRPOConfig(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_steps=5,
        num_generations=4,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        beta=0.02,
    )

    generate_new_examples = True
    configured_example_writer_formatter = partial(
        example_writer_formatter, generate_new_examples=generate_new_examples
    )

    wandb_config = {
        "project": "code-agents-magrpo",
        "entity": "nu-llpr",
        "name": "3-basic-2-edge" if generate_new_examples else "3-basic",
    }

    trainer = MAGRPOTrainer(
        agents=[AutoModelForCausalLM.from_pretrained(model_name) for _ in range(2)],
        num_agents=2,
        reward_funcs=execution_reward,
        formatters=[function_writer_formatter, configured_example_writer_formatter],
        args=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        wandb_config=wandb_config,
    )
    trainer.train()

    trainer.save_model(f"{output_dir}/final_models")


if __name__ == "__main__":
    main()
