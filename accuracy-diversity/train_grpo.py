import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_PATH = "./train_dataset/grpo_math_data.jsonl"
OUTPUT_DIR = "./qwen-3.5-4b-math-grpo-lora"


def extract_answer(text):
    match = re.search(r"####\s*\[?(-?[\d.,]+)\]?", str(text))
    if match:
        raw = match.group(1)
        normalized = raw.replace(".", "").replace(",", ".")
        return normalized
    return None


def accuracy_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, target in zip(completions, answer):
        response_text = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )

        pred = extract_answer(response_text)
        if pred is not None and target is not None and float(pred) == float(target):
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        response_text = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        pattern = r"<think>.*?</think>\s*.*####"
        if re.search(pattern, str(response_text), re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def main():
    print("Dataset loading...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    print("Tokenizer and model weights transfer into vram...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA Adapter
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        logging_steps=5,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_completion_length=768,
        num_generations=4,  # (G-Size)
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
        # GRPO arguments
        beta=0.1,  # KL divergence katsayısı (Modelin ana yeteneklerini unutmasını engeller)
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        peft_config=peft_config,
        train_dataset=dataset,
    )

    print("Training starting...")
    trainer.train()

    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    print("Training completed and weights saved!")


if __name__ == "__main__":
    main()
