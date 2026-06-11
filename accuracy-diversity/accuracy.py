import os
import json
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def extract_last_number(text):
    if not text:
        return None
    text = str(text)

    match_gsm = re.search(r"####\s*\[?(-?[\d.,]+)\]?", text)
    if match_gsm:
        return _clean_and_float(match_gsm.group(1))

    match_trigger = re.search(
        r"(?:cevap|sonuç|answer|is|:|=)\s*(-?\d+(?:[.,]\d+)?)", text, re.IGNORECASE
    )
    if match_trigger:
        return _clean_and_float(match_trigger.group(1))

    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    if numbers:
        return _clean_and_float(numbers[-1])

    return None


def _clean_and_float(raw_str):
    if "." in raw_str and len(raw_str.split(".")[-1]) == 3:
        raw_str = raw_str.replace(".", "")
    raw_str = raw_str.replace(",", ".")
    if raw_str in (".", "", "-", "-."):
        return None
    try:
        if raw_str.endswith("."):
            raw_str = raw_str[:-1]
        return float(raw_str)
    except ValueError:
        return None


def evaluate_model(jsonl_file, postfix):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    total_questions = len(data)
    if total_questions == 0:
        return None

    pass_10_count = 0
    majority_vote_count = 0
    total_exact_matches = 0
    valid_answers_count = 0

    for item in data:
        gt_val = extract_last_number(item["ground_truth"])
        if gt_val is None:
            total_questions -= 1
            continue

        gt_float = float(gt_val)

        ans_vals = []
        for ans in item["answers"]:
            val = extract_last_number(ans)
            if val is not None:
                ans_vals.append(float(val))

        valid_answers_count += len(ans_vals)

        if not ans_vals:
            continue

        # Average Accuracy
        matches = [1 for val in ans_vals if val == gt_float]
        total_exact_matches += len(matches)

        # Pass@10
        if len(matches) > 0:
            pass_10_count += 1

        # Self-Consistency
        counts = Counter(ans_vals)
        majority_ans, _ = counts.most_common(1)[0]
        if majority_ans == gt_float:
            majority_vote_count += 1

    avg_accuracy = (total_exact_matches / (total_questions * 10)) * 100
    pass_10 = (pass_10_count / total_questions) * 100
    maj_voting = (majority_vote_count / total_questions) * 100

    return {
        "Model": os.path.basename(jsonl_file).replace(postfix, "").upper(),
        "Avg Accuracy (Pass@1) %": round(avg_accuracy, 2),
        "Pass@10 %": round(pass_10, 2),
        "Majority Voting %": round(maj_voting, 2),
        "Extraction Rate %": round(
            (valid_answers_count / (total_questions * 10)) * 100, 2
        ),
    }


def plot_benchmark_results(df, save_dir):
    # Çizilecek metrikler
    metrics = ["Avg Accuracy (Pass@1) %", "Majority Voting %", "Pass@10 %"]

    # X ekseni pozisyonları
    x = np.arange(len(df["Model"]))
    width = 0.25  # Çubuk genişliği

    fig, ax = plt.subplots(figsize=(14, 8))

    # Çubukları oluştur
    rects1 = ax.bar(
        x - width, df[metrics[0]], width, label="Pass@1 (Avg Accuracy)", color="#4C72B0"
    )
    rects2 = ax.bar(
        x,
        df[metrics[1]],
        width,
        label="Majority Voting (Self-Consistency)",
        color="#DD8452",
    )
    rects3 = ax.bar(x + width, df[metrics[2]], width, label="Pass@10", color="#55A868")

    # Eksen ve başlık ayarları
    ax.set_ylabel("Başarım Oranı (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Modellerin GSM8K Performans Karşılaştırması",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Model"], rotation=15, ha="right", fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper left")

    # Y eksenini 0-110 arası yap (Metinler çubukların üstüne sığsın)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Çubukların üzerine değerleri yazdıran yardımcı fonksiyon
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),  # 4 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    # Grafiği kaydet
    save_path = os.path.join(save_dir, "benchmark_comparison_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Benchmark grafiği kaydedildi -> {save_path}")


if __name__ == "__main__":
    answers_dir = "answers/train_results"
    results_list = []

    if os.path.exists(answers_dir):
        for file_name in os.listdir(answers_dir):
            if file_name.endswith(".jsonl"):
                file_path = os.path.join(answers_dir, file_name)
                metrics = evaluate_model(
                    file_path, postfix=f"_{answers_dir.split('/')[1]}.jsonl"
                )
                if metrics:
                    results_list.append(metrics)

        df = pd.DataFrame(results_list)
        df = df.sort_values(by="Majority Voting %", ascending=False).reset_index(
            drop=True
        )

        plot_benchmark_results(df, answers_dir)
        df.to_csv(f"{answers_dir}/benchmark_results.csv", index=False)

        print(f"\nResults saved -> {answers_dir}/benchmark_results.csv")
    else:
        print(f"err: {answers_dir} is not founded.")
