from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
LORA_DIR = "qwen-3.5-4b-math-grpo-lora/final_model"
OUTPUT_DIR = "./merged_models/qwen-4b-math-grpo-MERGED"

print("Tokenizer laoding...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

print("Base Model transfering into VRAM...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

print("LoRA adapters integrating...")
model = PeftModel.from_pretrained(base_model, LORA_DIR)

print("Merging...")
merged_model = model.merge_and_unload()

print(f"Saving on: {OUTPUT_DIR}")
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Success")
