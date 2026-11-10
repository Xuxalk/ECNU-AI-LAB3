"""Compare the 30-epoch fixed-LR and 100-epoch cosine runs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run(name):
    run_dir = Path("experiments") / name
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    with (run_dir / "history.csv").open() as file:
        history = list(csv.DictReader(file))
    return metrics, summary, history


def main():
    runs = {
        "ResNet20 + SE (30 epochs)": "resnet20_se_seed123",
        "ResNet20 + SE + cosine (100 epochs)":
            "resnet20_se_cosine_100_seed123",
    }
    comparison = {}
    histories = {}
    for label, name in runs.items():
        metrics, summary, history = load_run(name)
        comparison[label] = {
            "test_accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "cat_recall": float(metrics["per_class"]["cat"]["recall"]),
            "dog_recall": float(metrics["per_class"]["dog"]["recall"]),
            "best_val_accuracy": float(summary["best_val_accuracy"]),
            "best_epoch": int(summary["best_epoch"]),
            "training_seconds": float(summary["training_seconds"]),
            "parameters": int(summary["parameters"]),
        }
        histories[label] = history

    comparison["delta_100_epoch_minus_30_epoch"] = {
        key: comparison["ResNet20 + SE + cosine (100 epochs)"][key]
        - comparison["ResNet20 + SE (30 epochs)"][key]
        for key in (
            "test_accuracy", "macro_f1", "cat_recall", "dog_recall",
            "best_val_accuracy", "training_seconds",
        )
    }

    output_dir = Path("experiments") / "resnet20_se_cosine_100_seed123"
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2)
    )

    fields = [
        ("test_accuracy", "Test Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("cat_recall", "Cat Recall (%)"),
        ("dog_recall", "Dog Recall (%)"),
        ("best_val_accuracy", "Best Val Accuracy (%)"),
        ("training_seconds", "Training Time (seconds)"),
    ]
    with (output_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric"] + list(runs))
        for key, label in fields:
            writer.writerow([
                label,
                comparison["ResNet20 + SE (30 epochs)"][key],
                comparison["ResNet20 + SE + cosine (100 epochs)"][key],
            ])

    markdown = [
        "| Metric | ResNet20 + SE (30 Epoch) | ResNet20 + SE (100 Epoch) |",
        "|---|---:|---:|",
    ]
    display_labels = {
        "test_accuracy": "Test Accuracy",
        "macro_f1": "Macro F1",
        "cat_recall": "Cat Recall",
        "dog_recall": "Dog Recall",
        "best_val_accuracy": "Best Val Accuracy",
        "training_seconds": "Training Time",
    }
    old = comparison["ResNet20 + SE (30 epochs)"]
    new = comparison["ResNet20 + SE + cosine (100 epochs)"]
    for key, _ in fields:
        if key == "training_seconds":
            old_value = f"{old[key] / 60:.2f} min"
            new_value = f"{new[key] / 60:.2f} min"
        else:
            old_value = f"{old[key]:.2f}%"
            new_value = f"{new[key]:.2f}%"
        markdown.append(
            f"| {display_labels[key]} | {old_value} | {new_value} |"
        )
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n")

    metric_keys = [
        "test_accuracy", "macro_f1", "cat_recall", "dog_recall",
        "best_val_accuracy",
    ]
    labels = [
        "Test accuracy", "Macro F1", "Cat recall", "Dog recall",
        "Best val accuracy",
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, (name, color) in zip(
        (-width / 2, width / 2),
        (("ResNet20 + SE (30 epochs)", "tab:blue"),
         ("ResNet20 + SE + cosine (100 epochs)", "tab:orange")),
    ):
        ax.bar(
            x + offset,
            [comparison[name][key] for key in metric_keys],
            width,
            label=name,
            color=color,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("ResNet20 + SE training strategy comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, history in histories.items():
        ax.plot(
            [int(row["epoch"]) for row in history],
            [float(row["val_accuracy"]) for row in history],
            label=label,
        )
    ax.set(title="Validation accuracy by epoch", xlabel="Epoch", ylabel="Accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "validation_accuracy_comparison.png",
        dpi=220, bbox_inches="tight",
    )
    plt.close(fig)

    print(json.dumps(comparison, indent=2))
    print(f"Saved comparison artifacts under {output_dir}")


if __name__ == "__main__":
    main()
