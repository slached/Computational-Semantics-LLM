import json
import time
import os
from datasets import load_dataset
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = "./train_dataset/sft_train_data.jsonl"

API_KEY = os.getenv("GROQ_API_KEY")

# Eğer .env dosyası bulunamazsa veya eksikse, mühendisi anında uyar:
if not API_KEY:
    raise ValueError("Groq api key is not founded!")

client = Groq(api_key=API_KEY)

MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]

SYSTEM_PROMPT = """Sen uzman bir matematik öğretmenisin. 
Sana verilen matematik problemlerini adım adım, mantıksal bir sırayla Türkçe olarak çözmelisin.
Çözümünün en sonunda nihai cevabı SADECE rakam olarak aşağıdaki formatta vermelisin:
Cevap: #### [Sayı]"""


def generate_sft_data_with_rotation(
    start_idx=1000, total_samples=1150, output_file=OUTPUT_FILE
):
    print(f"System on going... ({start_idx} - {total_samples})")
    dataset = load_dataset("ytu-ce-cosmos/gsm8k_tr", split="train")
    remaining_dataset = dataset.select(range(start_idx, total_samples))

    current_model_idx = 0

    for idx, sample in enumerate(tqdm(remaining_dataset, desc="Teacher Çözüyor")):
        actual_idx = start_idx + idx
        question = sample["question"]  # type: ignore

        while current_model_idx < len(MODELS):
            current_model = MODELS[current_model_idx]

            try:
                completion = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.2,
                    max_tokens=512,
                )

                teacher_solution = completion.choices[0].message.content.strip()  # type: ignore

                chatml_format = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Sen mantıksal adımlarla matematik soruları çözen bir yapay zeka asistanısın.",
                        },
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": teacher_solution},
                    ]
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(chatml_format, ensure_ascii=False) + "\n")

                time.sleep(5)
                break

            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    print(
                        f"\n[Limit Exceeded] {current_model}. skipping to the next model..."
                    )
                    current_model_idx += 1
                    time.sleep(5)
                else:
                    print(f"\nAPI error on Q {actual_idx}: {e}")
                    time.sleep(10)
                    break

        if current_model_idx >= len(MODELS):
            print(f"\nAll models tokens used. Current question is {actual_idx}.")
            break


if __name__ == "__main__":
    generate_sft_data_with_rotation()
