import re
import itertools
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

dataset_name = "ytu-ce-cosmos/gsm8k_tr"
base_model = "Qwen/Qwen3.5-4B"
embedder_model = "all-MiniLM-L6-v2"

peft_models = [
    {"method": "sft", "directoy": "merged_models/qwen-4b-math-sft-MERGED"},
    {"method": "multisft", "directoy": ""},
    {"method": "grpo", "directoy": "merged_models/qwen-4b-math-grpo-MERGED"},
    {"method": "sft+grpo", "directoy": ""},
    {"method": "multisft+grpo", "directoy": ""},
]

embedder = SentenceTransformer(embedder_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="auto"
)


def get_dataset(dataset_name: str):
    # fetch dataset from hugging face
    print(f"{dataset_name} Streaming starting...\n")

    dataset_stream = load_dataset(dataset_name, split="train", streaming=True)
    data_1000 = list(dataset_stream.take(1000))
    df = pd.DataFrame(data_1000)
    print(f"{dataset_name} Streaming ended\n")
    # delete empty spaces
    df = df.dropna(subset=["question", "answer"])

    return df


def create_answers(question: str, ground_truth: str):

    messages = [
        {
            "role": "system",
            "content": "Sen uzman bir matematik öğretmenisin. Sana verilen matematik problemlerini adım adım, mantıksal bir sırayla Türkçe olarak çözmelisin. Çözümünün en sonunda nihai cevabı SADECE rakam olarak aşağıdaki formatta vermelisin:\nCevap: #### [Sayı]",
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nQ: {question}")
    print(f"GT: {ground_truth}\n")
    print(f"On {device} answer creation on progress...\n")

    with torch.no_grad():
        outputs = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            num_return_sequences=10,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_responses = []
    input_length = inputs.input_ids.shape[1]

    for i, output in enumerate(outputs):
        response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        generated_responses.append(response)
        # print(f"--- Created answer {i+1}: {response.strip()}\n")

    return generated_responses


def extract_last_number(text):
    numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", ".")
    return None


def calculate_diversity(text1, text2):

    embeddings = embedder.encode([text1, text2])

    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]  # type: ignore
    return 1.0 - sim


def create_coordinates(accuracy_scores, answers):

    print("Getting coordinate information")
    coordinates = []

    for i, j in itertools.combinations(range(10), 2):

        # Y axis correctness
        y_axis_correctness = (accuracy_scores[i] + accuracy_scores[j]) / 2.0

        # X axis difference
        text1 = answers[i]
        text2 = answers[j]
        x_axis_difference = calculate_diversity(text1, text2)

        coordinates.append((x_axis_difference, y_axis_correctness))

    print(f"Total coordinate: {len(coordinates)}")
    return coordinates


def coordinate_plot(coordinates, file_name):
    x_values = [k[0] for k in coordinates]
    y_values = [k[1] for k in coordinates]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        x_values,
        y_values,
        alpha=0.5,
        color="royalblue",
        edgecolors="black",
        s=100,
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
        "Baseline (Base Model) - Diversity vs Accuracy Distribution",
        fontsize=14,
        fontweight="bold",
    )

    plt.xlabel("Diversity between Answers (1 - Cosine Similarity)", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.yticks([0.0, 0.5, 1.0])
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.1, 1.1)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{file_name}.png", dpi=300)
    plt.show()


def model_result(
    data,
    file_name,
    plot_draw=False,
):
    answers = create_answers(
        question=(str)(data["question"]),
        ground_truth=(str)(data["answer"]),
    )

    results = [extract_last_number(answer) for answer in answers]
    true_answer = extract_last_number(data["answer"])
    accuracy_scores = [
        (
            1.0
            if (
                result is not None
                and true_answer is not None
                and float(result) == float(true_answer)
            )
            else 0.0
        )
        for result in results
    ]
    coordinates = create_coordinates(accuracy_scores=accuracy_scores, answers=answers)
    coordinate_plot(coordinates=coordinates, file_name=file_name) if plot_draw else None


if __name__ == "__main__":
    dataset = get_dataset(dataset_name=dataset_name)
    baseline_data = {
        "question": dataset.loc[0, "question"],
        "answer": dataset.loc[0, "answer"],
    }

    model_result(
        data=baseline_data, plot_draw=True, file_name="./plot_images/baseline_plot"
    )
    model_result(
        data=baseline_data,
        plot_draw=True,
        file_name="./plot_images/sft_trained_plot",
    )
    model_result(
        data=baseline_data,
        plot_draw=True,
        file_name="./plot_images/grpo_trained_plot",
    )
