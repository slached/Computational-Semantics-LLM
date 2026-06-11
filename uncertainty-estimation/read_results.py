import json


def load_jsonl(file_path):
    true_count, false_count, truncate_count = 0, 0, 0

    with open(file_path, "r", encoding="utf-8") as f:
        responses = [json.loads(line) for line in f]
        for index, response in enumerate(responses):
            if response["uncertainty_data"]["num_tokens"] == 512:
                """print(
                    "=" * 100
                    + f"\n\n{truncate_count+1}: {response['model_generation']}\n\n"
                )"""
                truncate_count += 1
            if response["model_generation"]["is_correct"] == 1:
                true_count += 1
            else:
                false_count += 1

        return responses, true_count, false_count, truncate_count


def clean(file_path, responses, target_per_class=500):
    correct_count = 0
    incorrect_count = 0

    with open(file_path, "w", encoding="utf-8") as f:
        for response in responses:
            if response["uncertainty_data"]["num_tokens"] < 512:
                is_correct = response["model_generation"]["is_correct"]

                if is_correct == 1 and correct_count < target_per_class:
                    f.write(json.dumps(response, ensure_ascii=False) + "\n")
                    correct_count += 1

                elif is_correct == 0 and incorrect_count < target_per_class:
                    f.write(json.dumps(response, ensure_ascii=False) + "\n")
                    incorrect_count += 1

            if (
                correct_count == target_per_class
                and incorrect_count == target_per_class
            ):
                break

"""res, tc, fc, truncate = load_jsonl("train/cosmos_gsm8k_results.jsonl")
print(f"true:{tc}\nfalse:{fc}\ntruncate:{truncate}")
clean("train/cosmos_gsm8k_clean.jsonl", responses=res, target_per_class=500)"""


res, tc, fc, truncate = load_jsonl("test/cosmos_gsm8k_results.jsonl")
print(f"true:{tc}\nfalse:{fc}\ntruncate:{truncate}")
clean("test/cosmos_gsm8k_clean.jsonl", responses=res,target_per_class=50)
