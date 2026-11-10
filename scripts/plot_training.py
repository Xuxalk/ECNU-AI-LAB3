import csv
from pathlib import Path

import matplotlib.pyplot as plt

history_path = Path(
    "experiments/baseline_cnn_v1/history.csv"
)

output_path = Path(
    "outputs/figures/baseline_training_curves.png"
)

with history_path.open() as f:
    rows = list(csv.DictReader(f))

epochs = [int(r["epoch"]) for r in rows]
train_loss = [float(r["train_loss"]) for r in rows]
val_loss = [float(r["val_loss"]) for r in rows]
train_acc = [float(r["train_accuracy"]) for r in rows]
val_acc = [float(r["val_accuracy"]) for r in rows]

best_acc_idx = max(
    range(len(rows)),
    key=lambda i: val_acc[i]
)

best_loss_idx = min(
    range(len(rows)),
    key=lambda i: val_loss[i]
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(epochs, train_loss, label="Train Loss")
axes[0].plot(epochs, val_loss, label="Validation Loss")
axes[0].axvline(
    epochs[best_loss_idx],
    linestyle="--",
    label=f"Best Val Loss: Epoch {epochs[best_loss_idx]}"
)
axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross Entropy Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, train_acc, label="Train Accuracy")
axes[1].plot(epochs, val_acc, label="Validation Accuracy")
axes[1].axvline(
    epochs[best_acc_idx],
    linestyle="--",
    label=f"Best Val Accuracy: Epoch {epochs[best_acc_idx]}"
)
axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Best validation accuracy:",
      f"{val_acc[best_acc_idx]:.2f}%",
      "at epoch", epochs[best_acc_idx])

print("Best validation loss:",
      f"{val_loss[best_loss_idx]:.4f}",
      "at epoch", epochs[best_loss_idx])

print("Final generalization gap:",
      f"{train_acc[-1] - val_acc[-1]:.2f} percentage points")

print("Saved:", output_path)
