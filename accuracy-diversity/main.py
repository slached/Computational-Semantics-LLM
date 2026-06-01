import re
import itertools
import json
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

dataset_name = "ytu-ce-cosmos/gsm8k_tr"
base_model = "Qwen/Qwen3.5-4B"
embedder_model = "all-MiniLM-L6-v2"

peft_models = [
    {
        "method": "base",
        "directory": "/home/slached/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    },
    {"method": "sft", "directory": "./merged_models/qwen-4b-math-sft-MERGED"},
    {"method": "multisft", "directory": "./merged_models/qwen-4b-multisft-MERGED"},
    {"method": "grpo", "directory": "./merged_models/qwen-4b-math-grpo-MERGED"},
    {"method": "sft+grpo", "directory": "./merged_models/qwen-4b-math-sft-grpo-MERGED"},
    {
        "method": "multisft+grpo",
        "directory": "./merged_models/qwen-4b-math-multisft-grpo-MERGED",
    },
]


def create_answers_batched(test_data, save_dir, model, tokenizer):
    results = []

    for item in tqdm(test_data):
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        inputs_batched = {k: v.repeat(10, 1) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs_batched,
                max_new_tokens=768,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )

        answers = [
            tokenizer.decode(out[inputs["input_ids"].shape[1] :], skip_special_tokens=True)  # type: ignore
            for out in outputs  # type: ignore
        ]

        results.append(
            {
                "user_query": item["user"],
                "ground_truth": item["ground_truth"],
                "answers": answers,
            }
        )

        with open(save_dir, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "user_query": item["user"],
                        "ground_truth": item["ground_truth"],
                        "answers": answers,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return results


def extract_last_number(text):
    if not text:
        return None
    match = re.search(r"####\s*\[?(-?[\d.,]+)\]?", str(text))
    if match:
        raw = match.group(1)
    else:
        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
        if not numbers:
            return None
        raw = numbers[-1]

    if "." in raw and len(raw.split(".")[-1]) == 3:
        raw = raw.replace(".", "")
    raw = raw.replace(",", ".")

    # Sadece nokta veya boş string döndürme
    if raw in (".", "", "-"):
        return None

    try:
        float(raw)
        return raw
    except ValueError:
        return None


def calculate_diversity(text1, text2, embedder):
    embeddings = embedder.encode([text1, text2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]  # type: ignore
    return 1.0 - sim


def create_coordinates(results, embedder):
    all_coordinates = []

    all_texts = []
    for item in results:
        all_texts.extend(item["answers"])

    print(f"Encoding {len(all_texts)} texts...")
    all_embeddings = embedder.encode(all_texts, batch_size=256, show_progress_bar=True)

    idx = 0
    for item in results:
        n = len(item["answers"])
        embs = all_embeddings[idx : idx + n]
        idx += n

        true_answer_val = extract_last_number(item["ground_truth"])
        accuracy_scores = []
        for ans in item["answers"]:
            ans_val = extract_last_number(ans)
            try:
                accuracy_scores.append(
                    1.0 if ans_val and float(ans_val) == float(true_answer_val) else 0.0  # type: ignore
                )
            except:
                accuracy_scores.append(0.0)

        for i, j in itertools.combinations(range(n), 2):
            sim = float(
                np.dot(embs[i], embs[j])
                / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]))
            )
            diversity = 1.0 - sim
            accuracy = (accuracy_scores[i] + accuracy_scores[j]) / 2.0
            all_coordinates.append((diversity, accuracy))

    print(f"Total coordinates: {len(all_coordinates)}")
    return all_coordinates


def coordinate_plot(coordinates, file_name, model_name):
    x_values = [k[0] for k in coordinates]
    y_values = [k[1] for k in coordinates]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        x_values,
        y_values,
        alpha=0.05,
        color="royalblue",
        s=50,
        label="Generated Pairs",
    )

    plt.plot(
        1.0,
        1.0,
        marker="*",
        markersize=20,
        color="gold",
        markeredgecolor="black",
        label="Target (Perfect Spot)",
    )

    plt.title(
        f"{model_name} Model - Diversity vs Accuracy Distribution",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Diversity between Answers (1 - Cosine Similarity)", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{file_name}.png", dpi=300)
    print(f"Plot saved -> {file_name}.png")


def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    test_or_train = {"test": "test", "train": "train"}

    for peft_model in peft_models:
        method = peft_model["method"]
        directory = peft_model["directory"]

        if method != "sft":
            continue

        print(f"\n{'='*100}\nModel: {method} | {directory}\n{'='*100}")

        dataset_map = {
            "base": "test_dataset/TEST_500_BASE.jsonl",
            "sft": "test_dataset/TRAIN_1000_SFT.jsonl",
            "multisft": "test_dataset/TEST_500_MultiSFT.jsonl",
            "grpo": "test_dataset/TEST_500_GRPO.jsonl",
            "sft+grpo": "test_dataset/TEST_500_GRPO.jsonl",
            "multisft+grpo": "test_dataset/TEST_500_MultiSFT.jsonl",
        }
        data_file = dataset_map[method]
        test_data = load_jsonl(data_file)

        save_path = f"./answers/{test_or_train['train']}_results/{method}_{test_or_train['train']}_results.jsonl"

        tokenizer = AutoTokenizer.from_pretrained(directory)
        model = AutoModelForCausalLM.from_pretrained(
            directory,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        results = create_answers_batched(
            model=model,
            test_data=test_data,
            save_dir=save_path,
            tokenizer=tokenizer,
        )

        del model
        torch.cuda.empty_cache()
"""
        embedder = SentenceTransformer(embedder_model)

        coordinates = create_coordinates(results=results, embedder=embedder)

        coordinate_plot(
            coordinates=coordinates,
            file_name=f"./plot_images/{method}_{test_or_train['test']}_plot",
            model_name=method.upper(),
        )"""
