import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

experiments = {
    "A: Baseline": "baseline_cnn_v1",
    "B: Crop + Flip": "cnn_crop_flip_v1",
    "C: Crop + Flip + ColorJitter":
        "cnn_crop_flip_colorjitter_v1",
    "D: Crop + Flip + Cutout":
        "cnn_crop_flip_cutout_v1"
}

results = {}

for label, name in experiments.items():
    folder = Path("experiments") / name

    with (folder / "history.csv").open() as f:
        rows = list(csv.DictReader(f))

    best_acc = max(rows, key=lambda r: float(r["val_accuracy"]))
    best_loss = min(rows, key=lambda r: float(r["val_loss"]))

    summary = json.loads(
        (folder / "summary.json").read_text()
    )

    results[label] = {
        "best_val_accuracy": float(best_acc["val_accuracy"]),
        "best_accuracy_epoch": int(best_acc["epoch"]),
        "best_val_loss": float(best_loss["val_loss"]),
        "best_loss_epoch": int(best_loss["epoch"]),
        "training_seconds": summary["training_seconds"],
        "history": rows
    }

output = {}

for label, result in results.items():
    output[label] = {
        k: v for k, v in result.items()
        if k != "history"
    }

Path("experiments/comparison_A_B_C_D.json").write_text(
    json.dumps(output, indent=4)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for label, result in results.items():
    rows = result["history"]
    epochs = [int(r["epoch"]) for r in rows]

    axes[0].plot(
        epochs,
        [float(r["val_accuracy"]) for r in rows],
        label=label
    )

    axes[1].plot(
        epochs,
        [float(r["val_loss"]) for r in rows],
        label=label
    )

axes[0].set_title("Validation Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cross Entropy Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()

path = Path("outputs/figures/abcd_comparison.png")
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n===== EXPERIMENT COMPARISON =====")

for label, result in results.items():
    print(f"\n{label}")
    print("Best Accuracy:",
          f"{result['best_val_accuracy']:.2f}%")
    print("Best Accuracy Epoch:",
          result["best_accuracy_epoch"])
    print("Best Loss:",
          f"{result['best_val_loss']:.4f}")
    print("Training Time:",
          f"{result['training_seconds'] / 60:.2f} min")

print("\nSaved:", path)
print("Saved: experiments/comparison_A_B_C_D.json")
