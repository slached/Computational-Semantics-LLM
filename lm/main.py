import os
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from torch import Tensor, topk, cuda
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gc

# connection for higher download speed and access
MY_HF_TOKEN = ""
os.environ["HF_TOKEN"] = MY_HF_TOKEN


def t_SNE(embeddings: Tensor, model_name: str):
    print("Started to draw t_SNE this may take a while")
    embeddings_np = embeddings.cpu().numpy()
    
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne_model.fit_transform(embeddings_np)

    q_coords = embeddings_2d[:1000]
    a_coords = embeddings_2d[1000:]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        q_coords[:, 0], q_coords[:, 1], color="blue", label="Q", alpha=0.6, s=15
    )

    plt.scatter(a_coords[:, 0], a_coords[:, 1], color="red", label="A", alpha=0.6, s=15)
    plt.title(
        f"{model_name} t-SNE Distribution",
        fontsize=14,
    )

    plt.xlabel("t-SNE D-1")
    plt.ylabel("t-SNE D-2")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)

    model_name = (
        model_name.replace("/", "-") if model_name.find("/") != -1 else model_name
    )

    folder = model_name.split("- ")[1]
    os.makedirs(folder, exist_ok=True)
    
    plt.savefig(f"{folder}/{model_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Graphic has been drawn and saved.")


def get_dataset(dataset_name: str):
    # fetch dataset from hugging face
    print(f"{dataset_name} Streaming starting...")

    dataset_stream = load_dataset(dataset_name, split="train", streaming=True)
    data_1000 = list(dataset_stream.take(1000))
    df = pd.DataFrame(data_1000)

    if "answers" in df.columns:
        df["answer"] = df["answers"].apply(
            lambda x: (
                x["text"][0]
                if isinstance(x, dict) and "text" in x and len(x["text"]) > 0
                else None
            )
        )

    if "input" in df.columns and "output" in df.columns:
        df = df.rename(columns={"input": "question", "output": "answer"})

    # delete empty spaces
    df = df.dropna(subset=["question", "answer"])

    return df


def get_embeddings(model, input_texts: list):
    embeddings = model.encode(
        input_texts, convert_to_tensor=True, normalize_embeddings=True
    )
    return embeddings


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def calculate_similarities(similarity_matrix):
    # row quantity
    total_sample = similarity_matrix.shape[0]

    # the most similar 5 answers for each question
    _, top5_index = topk(similarity_matrix, k=5, dim=1)

    top1_true = 0
    top5_true = 0

    for i in range(total_sample):
        # is real answer the most similar answer(top-1) at same time?
        if top5_index[i][0].item() == i:
            top1_true += 1

        # is real answer in top-5
        if i in top5_index[i].tolist():
            top5_true += 1

    top1_percentage = (top1_true / total_sample) * 100
    top5_percentage = (top5_true / total_sample) * 100
    return top1_percentage, top5_percentage


def get_results(model_name: str, sim_matrix: Tensor):
    q_to_a_top1, q_to_a_top5 = calculate_similarities(sim_matrix)
    a_to_q_top1, a_to_q_top5 = calculate_similarities(sim_matrix.T)

    print(f"\n--- {model_name} Final Results ---")
    print(f"Q -> A | Top-1: %{q_to_a_top1:.2f} | Top-5: %{q_to_a_top5:.2f}")
    print(f"A -> Q | Top-1: %{a_to_q_top1:.2f} | Top-5: %{a_to_q_top5:.2f}")


def arrange_document_query(dataset, model_name) -> list:
    task = "Given a Turkish search query, retrieve relevant passages written in Turkish that best answer the query"

    queries = []
    documents = []

    for q, a in zip(dataset["question"], dataset["answer"]):
        (
            queries.append(get_detailed_instruct(task, str(q)))
            # special task text condition
            if model_name == "ytu-ce-cosmos/turkish-e5-large"
            else queries.append(str(q))
        )
        documents.append(str(a))
    # return input text
    return queries + documents


models = [
    "ytu-ce-cosmos/turkish-e5-large",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-m3",
]
#
datasets = {
    "ByLang QA": "bylang/tr_soru_cevap",
    "Turkish QA": "sixfingerdev/turkish-qa-multi-dialog-dataset",
}

# use cuda for better performance
device = "cuda" if cuda.is_available() else "cpu"

for dataset_label, dataset_path in datasets.items():
    print(f"\n=======================================================")
    print(f"{dataset_label} processing...")
    print(f"=======================================================\n")

    df_current = get_dataset(dataset_path)

    for model_name in models:
        print(f"\n--->Current model: {model_name} ({dataset_label})")

        model = SentenceTransformer(model_name, device=device)

        input_texts = arrange_document_query(df_current, model_name=model_name)

        embeddings = get_embeddings(model, input_texts)

        # split embeddings
        q_embeddings = embeddings[:1000]
        a_embeddings = embeddings[1000:]

        sim_matrix = util.cos_sim(q_embeddings, a_embeddings)
        get_results(sim_matrix=sim_matrix, model_name=f"{model_name} - {dataset_label}")

        # draw t-SNE
        t_SNE(embeddings=embeddings, model_name=f"{model_name} - {dataset_label}")

        # delete model from memo (garbage collector)
        del model
        if device == "cuda":
            cuda.empty_cache()
        gc.collect()
