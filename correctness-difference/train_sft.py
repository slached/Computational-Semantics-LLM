import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_PATH = "train_dataset/sft_train_data_clean.jsonl"
OUTPUT_DIR = "./qwen-3.5-4b-math-sft-lora"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")


def apply_chat_template(row):
    row_text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return {"text": row_text}


dataset = dataset.map(apply_chat_template)


print("Model loading into VRAM...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)


model = prepare_model_for_kbit_training(model)

# LoRA adapter: train only attention
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    max_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,  # type: ignore
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

torch.cuda.empty_cache()

print("Training starting...")
trainer.train()

trainer.save_model(f"{OUTPUT_DIR}/final_model")
print("Training completed and weights saved!")
