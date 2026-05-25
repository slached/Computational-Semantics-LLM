import json
import time
import os
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

OUTPUT_FILE = "./train_dataset/multi_sft_teacher_student.jsonl"
MODEL_NAME = "microsoft/wizardlm-2-8x22b"
MAX_TOKENS = 500

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("OpenAI api key is not founded or invalid!")


TARGETS = {"math": 1500, "code": 1000, "nlp": 1000, "reasoning": 500}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

def generate_teacher_response(system_prompt, user_prompt, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Low temp to prevent hallucination
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(
                f"\n[!] API Error: {e}. Retrying in 5 seconds... (Attempt {attempt + 1}/{retries})"
            )
            time.sleep(5)
    return None


def append_to_jsonl(system_prompt, user_prompt, assistant_prompt):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        chatml = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prompt},
            ]
        }
        f.write(json.dumps(chatml, ensure_ascii=False) + "\n")


def process_domain(
    domain_name, dataset_path, split_name, target_count, system_prompt, extract_func
):
    print(f"\n--- Generating {domain_name.upper()} Data ({target_count} samples) ---")
    dataset = load_dataset(dataset_path, split=split_name, streaming=True)

    count = 0
    pbar = tqdm(total=target_count)

    for row in dataset:
        if count >= target_count:
            break

        user_prompt = extract_func(row)
        if not user_prompt:
            continue

        teacher_answer = generate_teacher_response(system_prompt, user_prompt)

        if teacher_answer:
            append_to_jsonl(system_prompt, user_prompt, teacher_answer)
            count += 1
            pbar.update(1)
            time.sleep(0.1)

    pbar.close()


# Extraction rules (Grabbing only the questions/instructions, ignoring existing answers)
INVALID_INPUTS = [
    "",
    "null",
    "none",
    "nan",
    "< giriş yok >",
    "<no input>",
    "n/a",
    "yok",
    "bos",
]


def is_valid_input(inp):
    if inp is None:
        return False
    cleaned_inp = str(inp).strip().lower()
    if not cleaned_inp:
        return False
    if cleaned_inp in INVALID_INPUTS:
        return False
    return True


def extract_math(row):
    return row.get("question", "")


def extract_code(row):
    # berhaan/Turkish-CodeAlpaca-20k
    instruction = row.get("instruction", "")
    inp = row.get("input")

    if is_valid_input(inp):
        return f"{instruction}\n\nGirdi:\n{inp}"
    return instruction


def extract_nlp(row):
    # merve/turkish_instructions
    instruction = row.get("talimat", "")
    inp = row.get("giriş")

    if is_valid_input(inp):
        return f"{instruction}\n\nGirdi:\n{inp}"
    return instruction


def extract_reasoning(row):
    # umarigan/openhermes_tr
    instruction = row.get("instruction", "")
    inp = row.get("input")
    if is_valid_input(inp):
        return f"{instruction}\n\nGirdi:\n{inp}"
    return instruction


if __name__ == "__main__":
    print("Starting Autonomous Teacher-Student Distillation Pipeline...")
    # Clear the file if starting from scratch
    open(OUTPUT_FILE, "w").close()

    # System Prompts
    math_sys = "Sen uzman bir matematik öğretmenisin. Problemi adım adım, mantıksal bir sırayla Türkçe çöz. En sonda cevabı SADECE rakam olarak 'Cevap: #### [Sayı]' formatında ver."
    code_sys = "Sen kıdemli bir Python yazılım mühendisisin. Görevi yerine getiren temiz, okunabilir ve optimize edilmiş kodu yaz. Gereksiz açıklamalardan kaçın."
    nlp_sys = "Sen yardımsever bir yapay zeka asistanısın. Kullanıcının sorusunu doğal, akıcı ve dilbilgisi kurallarına uygun bir Türkçe ile doğrudan yanıtla."
    reasoning_sys = "Sen analitik düşünen bir mantık uzmanısın. Senaryoyu dikkatlice analiz et, adım adım mantıksal bir çıkarım yaparak sonuca ulaş."

    process_domain(
        "Math",
        "ytu-ce-cosmos/gsm8k_tr",
        "train",
        TARGETS["math"],
        math_sys,
        extract_math,
    )
    process_domain(
        "Code",
        "berhaan/Turkish-CodeAlpaca-20k",
        "train",
        TARGETS["code"],
        code_sys,
        extract_code,
    )
    process_domain(
        "NLP",
        "merve/turkish_instructions",
        "train",
        TARGETS["nlp"],
        nlp_sys,
        extract_nlp,
    )
    process_domain(
        "Reasoning",
        "umarigan/openhermes_tr",
        "train",
        TARGETS["reasoning"],
        reasoning_sys,
        extract_reasoning,
    )

    print("\n[SUCCESS] All data generated by Teacher model and saved successfully!")
