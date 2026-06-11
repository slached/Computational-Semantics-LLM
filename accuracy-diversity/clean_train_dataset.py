import json
import re

INPUT_FILE = "sft_train_data.jsonl"
OUTPUT_FILE = "sft_train_data_clean.jsonl"


def clean_jsonl_dataset():
    valid_count = 0
    invalid_count = 0

    pattern = re.compile(r"Cevap:\s*(?:#*\s*)?\d+", re.IGNORECASE)
    
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
        OUTPUT_FILE, "w", encoding="utf-8"
    ) as outfile:

        for line in infile:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                is_valid = False

                for msg in data.get("messages", []):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if pattern.search(content):
                            is_valid = True
                        break

                if is_valid:
                    outfile.write(line)
                    valid_count += 1
                else:
                    invalid_count += 1

            except json.JSONDecodeError:
                invalid_count += 1
                print("Skipped corrupted data.")

    print(f"Cleaning completed valid: {valid_count} – invalid: {invalid_count}")


if __name__ == "__main__":
    clean_jsonl_dataset()
