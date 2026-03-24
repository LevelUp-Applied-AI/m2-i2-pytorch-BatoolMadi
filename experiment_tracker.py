import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import itertools
import matplotlib.pyplot as plt
from tabulate import tabulate

from train import HousingModel, load_data   # عدلي الاسم إذا مختلف


# =====================================================
# 1. FIXED TRAIN / TEST SPLIT (same for all experiments)
# =====================================================

def create_split(X_tensor, y_tensor):
    torch.manual_seed(42)

    indices = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]

    split = int(0.8 * len(X_tensor))

    X_train = X_shuffled[:split]
    X_test = X_shuffled[split:]

    y_train = y_shuffled[:split]
    y_test = y_shuffled[split:]

    return X_train, X_test, y_train, y_test


# =====================================================
# 2. METRICS
# =====================================================

def compute_metrics(model, X_test, y_test, y_mean, y_std):
    with torch.no_grad():
        preds = model(X_test).detach().numpy().flatten()

    actual = y_test.numpy().flatten()

    # Reverse scaling to get back to original price scale
    preds = preds * y_std + y_mean
    actual = actual * y_std + y_mean

    mae = np.mean(np.abs(actual - preds))

    ss_res = np.sum((actual - preds) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    
    return mae, r2


# =====================================================
# 3. SINGLE EXPERIMENT RUN
# =====================================================

def run_experiment(config, X_train, X_test, y_train, y_test):

    torch.manual_seed(42)
    np.random.seed(42)

    start_time = time.time()

    model = HousingModel(hidden_size=config["hidden_size"])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    epochs = config["epochs"]

    # ---- Training loop ----
    for epoch in range(epochs):
        model.train()

        preds = model(X_train)
        loss = criterion(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = criterion(model(X_train), y_train).item()

    # ---- Test loss ----
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = criterion(test_preds, y_test).item()

    mae, r2 = compute_metrics(
        model,
        X_test,
        y_test,
        y_mean,
        y_std
    )

    elapsed = time.time() - start_time

    result = {
        "learning_rate": config["learning_rate"],
        "hidden_size": config["hidden_size"],
        "epochs": config["epochs"],
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_mae": float(mae),
        "test_r2": float(r2),
        "time_sec": elapsed,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    return result


# =====================================================
# 4. HYPERPARAMETER GRID (30+ experiments)
# =====================================================

learning_rates = [0.0005, 0.001, 0.01]
hidden_sizes = [16, 32, 64]
epochs_list = [50, 100, 150, 200]

grid = list(itertools.product(
    learning_rates,
    hidden_sizes,
    epochs_list
))

if __name__ == "__main__":
    print(f"Total experiments: {len(grid)}")
    
    # =====================================================
    # 5. LOAD DATA
    # =====================================================

    X_tensor, y_tensor, y_mean, y_std = load_data()
    X_train, X_test, y_train, y_test = create_split(X_tensor, y_tensor)


    # =====================================================
    # 6. RUN ALL EXPERIMENTS
    # =====================================================

    results = []

    for i, (lr, hidden, epochs) in enumerate(grid):

        config = {
            "learning_rate": lr,
            "hidden_size": hidden,
            "epochs": epochs
        }

        print(f"\nRunning experiment {i+1}/{len(grid)}")
        print(config)

        result = run_experiment(
            config,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)


    # =====================================================
    # 7. SAVE JSON LOG
    # =====================================================

    with open("experiments.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved experiments.json")


    # =====================================================
    # 8. LEADERBOARD
    # =====================================================

    sorted_results = sorted(results, key=lambda x: x["test_mae"])

    print("\n===== TOP 10 CONFIGURATIONS =====")

    table_data = []

    for rank, r in enumerate(sorted_results[:10], start=1):
        table_data.append([
            rank,
            r["learning_rate"],
            r["hidden_size"],
            r["epochs"],
            f"{r['test_mae']:.2f}",
            f"{r['test_r2']:.3f}",
            f"{r['time_sec']:.2f}s"
        ])

    print(tabulate(
        table_data,
        headers=["Rank", "LR", "Hidden", "Epochs", "MAE", "R2", "Time"],
        tablefmt="grid"
    ))

    best = sorted_results[0]

    print("\n🏆 BEST CONFIG FOUND:")
    print(tabulate([[k, v] for k, v in best.items()],
                headers=["Metric", "Value"],
                tablefmt="grid"))

    # =====================================================
    # 9. VISUALIZATION
    # =====================================================

    plt.figure()

    for hidden in hidden_sizes:
        subset = [r for r in results if r["hidden_size"] == hidden]

        lrs = [r["learning_rate"] for r in subset]
        maes = [r["test_mae"] for r in subset]

        plt.scatter(lrs, maes, label=f"hidden={hidden}")

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test MAE")
    plt.title("Experiment Summary")
    plt.legend()

    plt.savefig("experiment_summary.png")

    print("Saved experiment_summary.png")