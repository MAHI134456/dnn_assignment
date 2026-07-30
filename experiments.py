import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dnn import SimpleDNN

OUT = "outputs"
plt.rcParams["figure.dpi"] = 120

# --------------------------------------------------------------------------
# Data: sklearn's "digits" dataset (8x8 handwritten digit images, 10 classes)
# --------------------------------------------------------------------------
data = load_digits()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
NUM_CLASSES = 10
INPUT_DIM = X.shape[1]

print(f"Dataset: {X.shape[0]} samples, {INPUT_DIM} features, {NUM_CLASSES} classes")
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

EPOCHS = 80
DEEP_ARCH_HIDDEN = [32] * 8  # 8 hidden layers -> deep enough for vanishing gradients to show


# ==========================================================================
# PART 1.1  Sigmoid vs Tanh vs ReLU -- vanishing gradient
# ==========================================================================
def run_part_1_1():
    print("\n=== PART 1.1: Sigmoid vs Tanh vs ReLU (vanishing gradients) ===")
    layer_sizes = [INPUT_DIM] + DEEP_ARCH_HIDDEN + [NUM_CLASSES]
    results = {}
    for act in ["sigmoid", "tanh", "relu"]:
        print(f"Training {act} network with {len(DEEP_ARCH_HIDDEN)} hidden layers...")
        lr = 0.05 if act == "relu" else 0.3  # sigmoid/tanh need a higher LR just to move at all
        net = SimpleDNN(layer_sizes, activation=act, lr=lr, seed=1)
        hist = net.train(X_train, y_train, X_test, y_test, NUM_CLASSES,
                          epochs=EPOCHS, batch_size=32)
        results[act] = hist
        print(f"  final test acc: {hist['val_acc'][-1]:.3f}")

    # ---- Plot 1: training loss curves ----
    plt.figure(figsize=(6, 4))
    for act in results:
        plt.plot(results[act]["train_loss"], label=act)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (cross-entropy)")
    plt.title("Part 1.1: Training loss — Sigmoid vs Tanh vs ReLU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_1_loss_curves.png")
    plt.close()

    # ---- Plot 2: test accuracy curves ----
    plt.figure(figsize=(6, 4))
    for act in results:
        plt.plot(results[act]["val_acc"], label=act)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Part 1.1: Test accuracy — Sigmoid vs Tanh vs ReLU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_1_accuracy_curves.png")
    plt.close()

    # ---- Plot 3: gradient magnitude per layer (THE key vanishing-gradient plot) ----
    # Average the per-layer gradient norm over the first few epochs of training
    # (when the vanishing-gradient effect is strongest / most representative).
    plt.figure(figsize=(6, 4))
    n_layers = len(DEEP_ARCH_HIDDEN) + 1
    for act in results:
        early_epochs = results[act]["grad_norms_per_epoch"][:10]
        avg_norms = np.mean(np.array(early_epochs), axis=0)
        plt.plot(range(1, n_layers + 1), avg_norms, marker="o", label=act)
    plt.yscale("log")
    plt.xlabel("Layer index (1 = closest to input, "
               f"{n_layers} = output layer)")
    plt.ylabel("Mean |gradient| of layer weights (log scale)")
    plt.title("Part 1.1: Gradient magnitude by layer depth\n(avg. over first 10 epochs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_1_gradient_vanishing.png")
    plt.close()

    return results


# ==========================================================================
# PART 1.2  ReLU vs Leaky ReLU vs PReLU
# ==========================================================================
def run_part_1_2():
    print("\n=== PART 1.2: ReLU vs Leaky ReLU vs PReLU ===")
    layer_sizes = [INPUT_DIM] + DEEP_ARCH_HIDDEN + [NUM_CLASSES]
    results = {}
    dead_fracs = {}
    for act in ["relu", "leaky_relu", "prelu"]:
        print(f"Training {act}...")
        net = SimpleDNN(layer_sizes, activation=act, lr=0.05, seed=2)
        hist = net.train(X_train, y_train, X_test, y_test, NUM_CLASSES,
                          epochs=EPOCHS, batch_size=32)
        results[act] = hist
        dead_fracs[act] = net.dead_neuron_fraction(X_train)
        print(f"  final test acc: {hist['val_acc'][-1]:.3f}   "
              f"dead-neuron fraction: {dead_fracs[act]:.3f}")

    plt.figure(figsize=(6, 4))
    for act in results:
        plt.plot(results[act]["val_acc"], label=act)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Part 1.2: ReLU variants — test accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_2_accuracy_curves.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    for act in results:
        plt.plot(results[act]["train_loss"], label=act)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Part 1.2: ReLU variants — training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_2_loss_curves.png")
    plt.close()

    # Bar chart: final accuracy + dead-neuron fraction side by side
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    acts = list(results.keys())
    final_accs = [results[a]["val_acc"][-1] for a in acts]
    axes[0].bar(acts, final_accs, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_ylabel("Final test accuracy")
    axes[0].set_title("Final accuracy")
    axes[0].set_ylim(0, 1)

    dead_vals = [dead_fracs[a] for a in acts]
    axes[1].bar(acts, dead_vals, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_ylabel("Fraction of 'dead' neurons")
    axes[1].set_title("Dead ReLU units (train set)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_2_summary_bars.png")
    plt.close()

    return results, dead_fracs


# ==========================================================================
# PART 2.1  Depth comparison
# ==========================================================================
def run_part_2_1():
    print("\n=== PART 2.1: Depth comparison ===")
    depths = {
        "shallow (1 hidden)": [32],
        "medium (3 hidden)": [32, 32, 32],
        "deep (6 hidden)": [32] * 6,
        "very deep (10 hidden)": [32] * 10,
    }
    results = {}
    for name, hidden in depths.items():
        layer_sizes = [INPUT_DIM] + hidden + [NUM_CLASSES]
        print(f"Training {name} -> {layer_sizes}")
        net = SimpleDNN(layer_sizes, activation="relu", lr=0.05, seed=3)
        hist = net.train(X_train, y_train, X_test, y_test, NUM_CLASSES,
                          epochs=EPOCHS, batch_size=32, track_grads=False)
        results[name] = hist
        print(f"  final test acc: {hist['val_acc'][-1]:.3f}")

    # Loss curves for all depths
    plt.figure(figsize=(6.5, 4.5))
    for name in results:
        plt.plot(results[name]["train_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Part 2.1: Training loss vs epoch, by depth (ReLU)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/2_1_loss_curves.png")
    plt.close()

    # Test accuracy curves
    plt.figure(figsize=(6.5, 4.5))
    for name in results:
        plt.plot(results[name]["val_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.title("Part 2.1: Test accuracy vs epoch, by depth (ReLU)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/2_1_accuracy_curves.png")
    plt.close()

    # Final test accuracy vs number of hidden layers (the key "role of depth" plot)
    n_hidden = [len(v) for v in depths.values()]
    final_acc = [results[name]["val_acc"][-1] for name in depths]
    final_loss = [results[name]["train_loss"][-1] for name in depths]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(n_hidden, final_acc, marker="o", color="#4C72B0", label="Test accuracy")
    ax1.set_xlabel("Number of hidden layers (depth)")
    ax1.set_ylabel("Final test accuracy", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_xticks(n_hidden)

    ax2 = ax1.twinx()
    ax2.plot(n_hidden, final_loss, marker="s", color="#C44E52", label="Train loss")
    ax2.set_ylabel("Final training loss", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    plt.title("Part 2.1: Final accuracy / loss vs depth")
    fig.tight_layout()
    plt.savefig(f"{OUT}/2_1_accuracy_vs_depth.png")
    plt.close()

    return results


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    r11 = run_part_1_1()
    r12 = run_part_1_2()
    r21 = run_part_2_1()
    print("\nAll experiments complete. Plots saved to ./outputs/")
