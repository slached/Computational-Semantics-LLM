import json
import os
import re
import random
from datasets import load_dataset

OUTPUT_DIR = "./test_dataset/"
os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

PROMPTS = {
    "BASE": "Sen yardımcı bir yapay zeka asistanısın.",
    "SFT": """Sen uzman bir matematik öğretmenisin. 
Sana verilen matematik problemlerini adım adım, mantıksal bir sırayla Türkçe olarak çözmelisin.
Çözümünün en sonunda nihai cevabı SADECE rakam olarak aşağıdaki formatta vermelisin:
Cevap: #### [Sayı]""",
    "GRPO": "Sen mantıksal adımlarla matematik soruları çözen bir yapay zeka asistanısın. Çözümünü her zaman önce <think> ... </think> etiketleri arasında adım adım düşünerek yap, ardından nihai cevabını SADECE 'Cevap: #### [Sayı]' formatında ver.",
    "Multi-SFT-math_sys": "Sen uzman bir matematik öğretmenisin. Problemi adım adım, mantıksal bir sırayla Türkçe çöz. En sonda cevabı SADECE rakam olarak 'Cevap: #### [Sayı]' formatında ver.",
    "Multi-SFT-code_sys": "Sen kıdemli bir Python yazılım mühendisisin. Görevi yerine getiren temiz, okunabilir ve optimize edilmiş kodu yaz. Gereksiz açıklamalardan kaçın.",
    "Multi-SFT-nlp_sys": "Sen yardımsever bir yapay zeka asistanısın. Kullanıcının sorusunu doğal, akıcı ve dilbilgisi kurallarına uygun bir Türkçe ile doğrudan yanıtla.",
    "Multi-SFT-reasoning_sys": "Sen analitik düşünen bir mantık uzmanısın. Senaryoyu dikkatlice analiz et, adım adım mantıksal bir çıkarım yaparak sonuca ulaş.",
}


def parse_grpo_item(item):
    system_msg = next(m["content"] for m in item["prompt"] if m["role"] == "system")
    user_msg = next(m["content"] for m in item["prompt"] if m["role"] == "user")
    return {"system": system_msg, "user": user_msg, "ground_truth": item["answer"]}


def parse_sft_item(item):
    system_msg = next(m["content"] for m in item["messages"] if m["role"] == "system")
    user_msg = next(m["content"] for m in item["messages"] if m["role"] == "user")
    assistant_msg = next(
        m["content"] for m in item["messages"] if m["role"] == "assistant"
    )
    match = re.search(r"Cevap:\s*(?:####\s*)?(\d+)", assistant_msg)
    answer = match.group(1) if match else assistant_msg

    return {"system": system_msg, "user": user_msg, "ground_truth": answer}


def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[{len(data)} question] -> {filename}")


print("Data loading and parsing...")

raw_sft = load_jsonl("train_dataset/sft_train_data_clean.jsonl")  # 1100
raw_grpo = load_jsonl("train_dataset/grpo_math_data.jsonl")  # 8000
raw_multi = load_jsonl("train_dataset/multi_sft_teacher_student.jsonl")  # 4000

parsed_sft = [parse_sft_item(item) for item in raw_sft]
parsed_grpo = [parse_grpo_item(item) for item in raw_grpo]
parsed_multi = [parse_sft_item(item) for item in raw_multi]

print("\n--- TRAIN Set ---")

train_sft = random.sample(parsed_sft, 1000)
train_grpo = random.sample(parsed_grpo, 1000)
train_multi = random.sample(parsed_multi, 1000)

save_jsonl(train_sft, "TRAIN_1000_SFT.jsonl")
save_jsonl(train_grpo, "TRAIN_1000_GRPO.jsonl")
save_jsonl(train_multi, "TRAIN_1000_MultiSFT.jsonl")

print("\n--- TEST Set ---")
# random 500 sample from dataset which never used to train

ds = load_dataset("oztrkoguz/Open_Math_Instruct_Turkish", split="train")
test_samples = random.sample(list(ds), 501)

parsed_test = []
for item in test_samples:
    user_msg = item.get("question", "")  # type: ignore
    assistant_msg = item.get("answer", "")  # type: ignore

    if not user_msg or not assistant_msg:
        continue

    match = re.search(r"(?:####|Cevap:|Sonuç:)\s*(-?\d+)", assistant_msg, re.IGNORECASE)

    if match:
        ground_truth = match.group(1)
    else:
        numbers = re.findall(r"-?\d+", assistant_msg)
        if numbers:
            ground_truth = numbers[-1]
        else:
            continue

    parsed_test.append({"user": user_msg, "ground_truth": ground_truth})

    if len(parsed_test) >= 500:
        break


def inject_prompt(dataset, new_system_prompt):
    return [
        {
            "system": new_system_prompt,
            "user": item["user"],
            "ground_truth": item["ground_truth"],
        }
        for item in dataset
    ]


test_base = inject_prompt(parsed_test, PROMPTS["BASE"])
test_sft = inject_prompt(parsed_test, PROMPTS["SFT"])
test_grpo = inject_prompt(parsed_test, PROMPTS["GRPO"])
test_multi = inject_prompt(parsed_test, PROMPTS["Multi-SFT-math_sys"])

save_jsonl(test_base, "TEST_500_BASE.jsonl")
save_jsonl(test_sft, "TEST_500_SFT.jsonl")
save_jsonl(test_grpo, "TEST_500_GRPO.jsonl")
save_jsonl(test_multi, "TEST_500_MultiSFT.jsonl")


print("\nAll data ready to test.")
