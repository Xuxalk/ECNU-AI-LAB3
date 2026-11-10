import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

experiments = {
    "Baseline": "baseline_cnn_v1",
    "Crop + Flip": "cnn_crop_flip_v1"
}

results = {}

for label, name in experiments.items():
    folder = Path("experiments") / name

    with (folder / "history.csv").open() as f:
        rows = list(csv.DictReader(f))

    best_acc = max(rows, key=lambda x: float(x["val_accuracy"]))
    best_loss = min(rows, key=lambda x: float(x["val_loss"]))

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

base = results["Baseline"]
aug = results["Crop + Flip"]

comparison = {
    "baseline": {k: v for k, v in base.items() if k != "history"},
    "crop_flip": {k: v for k, v in aug.items() if k != "history"},
    "accuracy_improvement_pp": (
        aug["best_val_accuracy"] - base["best_val_accuracy"]
    ),
    "loss_reduction_percent": (
        1 - aug["best_val_loss"] / base["best_val_loss"]
    ) * 100
}

Path("experiments/comparison_A_B.json").write_text(
    json.dumps(comparison, indent=4)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

axes[0].set_title("Validation Accuracy Comparison")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_title("Validation Loss Comparison")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cross Entropy Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()

output = Path("outputs/figures/ab_comparison.png")
fig.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps(comparison, indent=4))
print("\nSaved:", output)
