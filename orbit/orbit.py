import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

token_size = 50
num_trajectories = 1000

prompts = []


def plot_trajectory_space(data_path, labels, model_name):

    # Load 3D tensor: (1000, 50, D)
    data = np.load(data_path)
    num_traj, seq_len, hidden_dim = data.shape

    print(f"Applying PCA dimensionality reduction for {model_name}...")

    # Reshape to 2D for PCA fitting: (50000, D)
    flat_data = data.reshape(-1, hidden_dim)

    # Fit PCA and reduce to 2 components
    pca = PCA(n_components=2)
    flat_2d = pca.fit_transform(flat_data)

    # Reshape back to 3D trajectory format: (1000, 50, 2)
    data_2d = flat_2d.reshape(num_traj, seq_len, 2)

    # Initialize plot
    plt.figure(figsize=(12, 8))

    # Filter indices for specific classes to compare (e.g., 5 samples per class)
    # Class 0: World, Class 3: Sci/Tech
    class_0_indices = np.where(labels == 0)[0][:5]
    class_3_indices = np.where(labels == 3)[0][:5]

    # Plot Class 0 trajectories (World News)
    for idx in class_0_indices:
        traj = data_2d[idx]
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            markersize=4,
            alpha=0.6,
            color="blue",
            label="Class 0 (World)" if idx == class_0_indices[0] else "",
        )

        # Mark start points (Cyan) and end points (Dark Blue X)
        plt.scatter(
            traj[0, 0], traj[0, 1], color="cyan", s=60, edgecolors="black", zorder=5
        )
        plt.scatter(
            traj[-1, 0],
            traj[-1, 1],
            color="darkblue",
            s=80,
            marker="X",
            edgecolors="black",
            zorder=5,
        )

    # Plot Class 3 trajectories (Sci/Tech News)
    for idx in class_3_indices:
        traj = data_2d[idx]
        plt.plot(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            markersize=4,
            alpha=0.6,
            color="red",
            label="Class 3 (Sci/Tech)" if idx == class_3_indices[0] else "",
        )

        # Mark start points (Orange) and end points (Dark Red X)
        plt.scatter(
            traj[0, 0], traj[0, 1], color="orange", s=60, edgecolors="black", zorder=5
        )
        plt.scatter(
            traj[-1, 0],
            traj[-1, 1],
            color="darkred",
            s=80,
            marker="X",
            edgecolors="black",
            zorder=5,
        )

    # Plot formatting
    plt.title(f"{model_name} - 2D Trajectory Visualization (PCA)")
    plt.xlabel(
        f"Principal Component 1 (Explains {pca.explained_variance_ratio_[0]:.2%} variance)"
    )
    plt.ylabel(
        f"Principal Component 2 (Explains {pca.explained_variance_ratio_[1]:.2%} variance)"
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def guess(features_m1, features_m2):

    # ---------------------------------------------------------
    # Guess 1: MODEL CLASSIFICATION (Model 1 vs Model 2)
    # ---------------------------------------------------------
    X_task1 = np.vstack((features_m1, features_m2))  # (2000, 7)
    y_task1 = np.array([0] * 1000 + [1] * 1000)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X_task1, y_task1, test_size=0.2, random_state=42
    )

    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf1.fit(X_train1, y_train1)
    y_pred1 = clf1.predict(X_test1)

    print("--- Guess 1: Model Identification Results ---")
    print(classification_report(y_test1, y_pred1))

    # ---------------------------------------------------------
    # GUESS 2: SUBJECT CLASSIFICATION (ag_news classes)
    # ---------------------------------------------------------

    print("Extracting labels from dataset...")
    dataset = load_dataset("ag_news", split="train")
    labels = [dataset[i]["label"] for i in range(1000)]

    y_task2 = np.array(labels)  # (1000,) label vector
    # use either m2 or m1
    X_task2 = features_m1

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_task2, y_task2, test_size=0.2, random_state=42
    )

    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2.fit(X_train2, y_train2)
    y_pred2 = clf2.predict(X_test2)

    print("\n--- Guess 2: Topic Prediction Results (via Model 1 features) ---")

    print(classification_report(y_test2, y_pred2))


def create_prompt(prompt_quantity):
    print("--------------------------------------------------")
    print(f"Stage 1: Creating {prompt_quantity} prompts...")

    # ag_news dataset used for prompts
    dataset = load_dataset("ag_news", split="train")
    for i in range(prompt_quantity):
        data = dataset[i]["text"]
        # get only 5 word from sentence
        prompt = " ".join(data.split()[:5])
        prompts.append(prompt)

    print(f"Success: {len(prompts)} prompts loaded into memory.")
    print("--------------------------------------------------\n")


def get_trajectory(model_name: str, **kwargs):
    print(f"Stage 2: Trajectory extraction started for '{model_name}'.")
    all_trajectories = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] Using device: {device.type.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # move model to gpu
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)  # type: ignore
    print(f"[{model_name}] Model loaded to {device.type.upper()}. Starting inference.")

    with torch.no_grad():
        for i in tqdm(
            range(num_trajectories), desc=f"Processing {model_name}", unit="traj"
        ):

            # move input tensor to gpu
            input_ids = tokenizer.encode(prompts[i], return_tensors="pt").to(device)

            current_trajectory = []

            for _ in range(token_size):
                # output_hidden_states=True
                outputs = model(input_ids, output_hidden_states=True)

                # Trajectory point extraction
                # get last token's hidden state
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]

                # Numpy cannot read directly from GPU VRAM
                current_trajectory.append(
                    last_hidden_state.squeeze().cpu().to(torch.float32).numpy()
                )

                next_token_logits = outputs.logits[:, -1, :]
                # create next token via greedy decoding
                # torch.multinomial is alternative
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                # add new token into end of input_ids
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # 50 stepped trajectory is completed
            current_trajectory_matrix = np.array(current_trajectory)  # size: (50, 768)
            all_trajectories.append(current_trajectory_matrix)

    print(
        f"\n[{model_name}] Extraction complete. Final shape: ({num_trajectories}, {token_size}, {current_trajectory_matrix.shape[-1]})"  # type: ignore
    )

    # Clear GPU memory for the next model to avoid OOM errors
    del model
    torch.cuda.empty_cache()

    return all_trajectories


def extract_trajectory_features(trajectories_path):
    # Load the 3D tensor: (1000, 50, D)
    data = np.load(trajectories_path)

    print(f"Loaded trajectories shape: {data.shape}")

    # FEATURE SET 1 Distance Based Features
    # Calculate difference between consecutive steps: v(t+1) - v(t)
    diffs = data[:, 1:, :] - data[:, :-1, :]
    
    # Calculate Euclidean distance (L2 norm) for each step
    step_distances = np.linalg.norm(diffs, axis=2)

    total_path_length = np.sum(step_distances, axis=1)  # (1000,)
    mean_step_size = np.mean(step_distances, axis=1)  # (1000,)
    std_step_size = np.std(step_distances, axis=1)  # (1000,)

    # Distance between first and last token
    net_diff = data[:, -1, :] - data[:, 0, :]
    net_displacement = np.linalg.norm(net_diff, axis=1)  # (1000,)

    # Stack Set 1 features
    features_set_1 = np.column_stack(
        (total_path_length, mean_step_size, std_step_size, net_displacement)
    )

    # FEATURE SET 2 Angular/Cosine Based Features
    mean_cos_sim = np.zeros(num_trajectories)
    std_cos_sim = np.zeros(num_trajectories)
    start_end_cos_sim = np.zeros(num_trajectories)

    for i in range(num_trajectories):
        traj = data[i]  

        # Start to End Cosine Similarity
        v_start = traj[0]
        v_end = traj[-1]
        start_end_cos_sim[i] = np.dot(v_start, v_end) / (
            np.linalg.norm(v_start) * np.linalg.norm(v_end)
        )

        # Consecutive Cosine Similarities
        v1 = traj[:-1]  
        v2 = traj[1:] 

        # Vectorized dot product for row pairs
        dot_products = np.sum(v1 * v2, axis=1)
        norms_v1 = np.linalg.norm(v1, axis=1)
        norms_v2 = np.linalg.norm(v2, axis=1)

        cos_sims = dot_products / (norms_v1 * norms_v2)

        mean_cos_sim[i] = np.mean(cos_sims)
        std_cos_sim[i] = np.std(cos_sims)

    # Stack Set 2 features
    features_set_2 = np.column_stack((mean_cos_sim, std_cos_sim, start_end_cos_sim))

    # COMBINE ALL FEATURES
    combined_features = np.hstack((features_set_1, features_set_2))

    print(f"Feature Extraction Complete. Final Matrix Shape: {combined_features.shape}")
    return combined_features, features_set_1, features_set_2


if __name__ == "__main__":
    # create_prompt(num_trajectories)

    """# --- MODEL 1 CREATING TRAJECTORY---
    model_1_name = "distilgpt2"
    model_1_data = np.array(get_trajectory(model_1_name))
    print(f"Stage 3: Saving {model_1_name} data to Desktop...")
    np.save("model_1_trajectories.npy", model_1_data)
    print("Success: 'model_1_trajectories.npy' successfully written to disk.\n")

    # --- MODEL 2 CREATING TRAJECTORY---
    model_2_name = "Qwen/Qwen1.5-0.5B"
    model_2_data = np.array(
        get_trajectory(model_2_name, tie_word_embeddings=False, trust_remote_code=True)
    )
    print(f"Stage 3: Saving {model_2_name} data to Desktop...")
    np.save("model_2_trajectories.npy", model_2_data)
    print("Success: 'model_2_trajectories.npy' successfully written to disk.\n")"""

    # --- FEATURE EXTRACTION FOR BOTH MODEL TRAJECTORIES ---
    model_1_all_features, kinematic, angular = extract_trajectory_features(
        "model_1_trajectories.npy"
    )
    model_2_all_features, kinematic, angular = extract_trajectory_features(
        "model_2_trajectories.npy"
    )

    guess(model_1_all_features, model_2_all_features)

    """print("Extracting labels from dataset...")
    dataset = load_dataset("ag_news", split="train")
    labels = np.array([dataset[i]["label"] for i in range(1000)])

    plot_trajectory_space(
        data_path="model_1_trajectories.npy", labels=labels, model_name="DistilGPT2"
    )

    plot_trajectory_space(
        data_path="model_2_trajectories.npy", labels=labels, model_name="Qwen1.5"
    )"""

    print("All tasks completed.")
