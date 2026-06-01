from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"

merge_configs = [
    {
        "base": BASE_MODEL_ID,
        "lora": "./qwen-3.5-4b-math-sft-lora/final_model",
        "output": "./merged_models/qwen-4b-math-sft-MERGED",
    },
    {
        "base": BASE_MODEL_ID,
        "lora": "./qwen-3.5-4b-multisft-lora/final_model",
        "output": "./merged_models/qwen-4b-multisft-MERGED",
    },
    {
        "base": BASE_MODEL_ID,
        "lora": "./qwen-3.5-4b-math-grpo-lora/final_model",
        "output": "./merged_models/qwen-4b-math-grpo-MERGED",
    },
    {
        "base": "./merged_models/qwen-4b-math-sft-MERGED",
        "lora": "./qwen-3.5-4b-math-grpo-lora/final_model",
        "output": "./merged_models/qwen-4b-math-sft-grpo-MERGED",
    },
    {
        "base": "./merged_models/qwen-4b-multisft-MERGED",
        "lora": "./qwen-3.5-4b-math-grpo-lora/final_model",
        "output": "./merged_models/qwen-4b-math-multisft-grpo-MERGED",
    },
]


for cfg in merge_configs:
    print(f"\n{'='*50}")
    print(f"Merging: {cfg['base']} + {cfg['lora']} -> {cfg['output']}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base"])

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base"], torch_dtype=torch.bfloat16, device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, cfg["lora"])
    merged_model = model.merge_and_unload()  # type: ignore

    merged_model.save_pretrained(cfg["output"])
    tokenizer.save_pretrained(cfg["output"])
    print(f"Saved -> {cfg['output']}")

    del merged_model, model, base_model
    torch.cuda.empty_cache()

print("\nAll merges completed!")

print("Success")
