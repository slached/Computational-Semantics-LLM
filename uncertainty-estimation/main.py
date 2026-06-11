from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, re, json
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from scipy.stats import linregress
from huggingface_hub import login
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from matplotlib import pyplot as plt
import seaborn as sns

model_id = "Qwen/Qwen2-7B-Instruct"
dataset_id = "ytu-ce-cosmos/gsm8k_tr"

def get_dataset():
    print("=" * 50 + f"collecting from {dataset_id}" + "=" * 50)
    ds = load_dataset(dataset_id, split="train", streaming=True)
    data = list(ds.take(3000))
    print("=" * 50 + f"data collected!" + "=" * 50)
    return data


def extract_final_number(text):
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

    if raw in (".", "", "-"):
        return None

    try:
        float(raw)
        return raw
    except ValueError:
        return None

    pass


def calculate_features(delta_p_list):
    if not delta_p_list:
        return {}

    delta_array = np.array(delta_p_list)
    n_tokens = len(delta_array)

    if n_tokens > 1:
        slope, _, _, _, _ = linregress(range(n_tokens), delta_array)
    else:
        slope = 0.0

    return {
        "mean_delta": float(np.mean(delta_array)),
        "min_delta": float(np.min(delta_array)),
        "var_delta": float(np.var(delta_array)),
        "drops_below_05": int(np.sum(delta_array < 0.5)),
        "drops_below_02": int(np.sum(delta_array < 0.2)),
        "volatility": (
            float(np.mean(np.abs(np.diff(delta_array)))) if n_tokens > 1 else 0.0
        ),
        "last_10_tokens_mean": float(np.mean(delta_array[-10:])),
        "trend_slope": float(slope),  # type: ignore
    }


def generate_and_log_uncertainty(
    data,
    output_file,
    model_id="ytu-ce-cosmos/Turkish-Gemma-9b-T1",
    max_token=1000,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )

    model.eval()

    with open(output_file, "a", encoding="utf-8") as f:
        for idx, item in enumerate(tqdm(data, desc="Generating training data")):
            question_text = item["question"]
            ground_truth_text = item["answer"]

            messages = [
                {
                    "role": "system",
                    "content": """Matematik sorusunu adım adım düşünerek çöz. Tüm hesaplama adımlarını detaylıca yaz. Çözümünü bitirdikten sonra, en alt satıra nihai cevabını SADECE şu formatta yaz: #### [Sayı]""",
                },
                {"role": "user", "content": question_text},
            ]

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=max_token,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=6,
                    do_sample=False,
                )

            input_length = inputs.input_ids.shape[1]
            generated_token_ids = outputs.sequences[0, input_length:]

            tokens = []
            top1_probs = []
            top2_probs = []
            delta_ps = []

            for i, scores in enumerate(outputs.scores):
                logits = scores[0]

                # softmax to get probality distrubution
                probs = torch.softmax(logits, dim=-1)

                # get two probality
                top_probs, _ = torch.topk(probs, 2)
                t1_prob = top_probs[0].item()
                t2_prob = top_probs[1].item()

                token_str = tokenizer.decode(generated_token_ids[i])

                tokens.append(token_str)
                top1_probs.append(round(t1_prob, 4))
                top2_probs.append(round(t2_prob, 4))
                delta_ps.append(round(t1_prob - t2_prob, 4))

            generated_text = tokenizer.decode(
                generated_token_ids, skip_special_tokens=True
            )

            pred_num = extract_final_number(generated_text)
            true_num = extract_final_number(ground_truth_text)

            is_correct = (
                1 if (pred_num and true_num is not None and pred_num == true_num) else 0
            )

            record = {
                "question_id": f"q_{idx}",
                "question_text": question_text,
                "ground_truth_raw": ground_truth_text,
                "ground_truth_num": true_num,
                "model_generation": {
                    "raw_text": generated_text,
                    "extracted_answer": pred_num,
                    "is_correct": is_correct,
                },
                "uncertainty_data": {
                    "num_tokens": len(tokens),
                    "tokens": tokens,
                    "top1_probs": top1_probs,
                    "top2_probs": top2_probs,
                    "delta_p": delta_ps,
                },
                "features": calculate_features(delta_ps),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def load_ml_data(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            row = record["features"].copy()
            row["is_correct"] = record["model_generation"]["is_correct"]
            data_list.append(row)

    df = pd.DataFrame(data_list)
    X = df.drop(columns=["is_correct"])
    y = df["is_correct"]
    return X, y


def evaluate_and_report_plt(model_name, y_true, y_pred, model_instance, X_columns):

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"--- {model_name} Performans Metrikleri ---")
    print(f"Doğruluk (Accuracy)  : {acc:.4f}")
    print(f"Kesinlik (Precision) : {prec:.4f}")
    print(f"Duyarlılık (Recall)  : {rec:.4f}")
    print(f"F1-Skoru (F1-Score)  : {f1:.4f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    fig.suptitle(f"{model_name} Model Değerlendirmesi", fontsize=16, fontweight="bold")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        cbar=False,
        annot_kws={"size": 14},
    )
    axes[0].set_title("Karmaşıklık Matrisi", fontsize=14)
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label)", fontsize=12)
    axes[0].set_xticklabels(["0", "1"], rotation=0)
    axes[0].set_yticklabels(["0", "1"], rotation=0)

    importances = model_instance.feature_importances_
    feat_imp = pd.Series(importances, index=X_columns).sort_values(ascending=False)

    sns.barplot(
        x=feat_imp.values,
        y=feat_imp.index,
        ax=axes[1],
        hue=feat_imp.index,
        palette="viridis",
        legend=False,
    )
    axes[1].set_title("Öznitelik Önem Dereceleri", fontsize=14)
    axes[1].set_xlabel("Önem Skoru", fontsize=12)
    axes[1].set_ylabel("Öznitelikler", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{model_name}_degerlendirme.png", bbox_inches="tight")


if __name__ == "__main__":
    # data = get_dataset()
    """generate_and_log_uncertainty(
        data[:1500],
        output_file="train/cosmos_gsm8k_results.jsonl",
        model_id=model_id,
        max_token=512,
    )

    generate_and_log_uncertainty(
        data[1601:1701],
        output_file="test/cosmos_gsm8k_results.jsonl",
        model_id=model_id,
        max_token=512,
    )"""

    
    X_train, y_train = load_ml_data("train/cosmos_gsm8k_clean.jsonl")
    X_test, y_test = load_ml_data("test/cosmos_gsm8k_clean.jsonl")

    xgb_model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)

    evaluate_and_report_plt("XGBoost", y_test, xgb_preds, xgb_model, X_train.columns)

    rf_model = RandomForestClassifier(
        random_state=42, n_estimators=100, max_depth=5, min_samples_split=5
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    evaluate_and_report_plt(
        "Random Forest", y_test, rf_preds, rf_model, X_train.columns
    )