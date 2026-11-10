"""Compare CutMix with the existing CIFAR-10 experiments."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def macro_metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    diagonal = np.diag(matrix)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(
        diagonal, actual,
        out=np.zeros_like(diagonal), where=actual != 0,
    )
    precision = np.divide(
        diagonal, predicted,
        out=np.zeros_like(diagonal), where=predicted != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) != 0,
    )
    return float(precision.mean() * 100), float(recall.mean() * 100), float(f1.mean() * 100)


def load_summary(name):
    return json.loads((Path("experiments") / name / "summary.json").read_text())


def main():
    historical_results = json.loads(
        Path("outputs/metrics/final_test_results.json").read_text()
    )
    original = next(
        result for result in historical_results["B"]
        if result["run"] == "cnn_crop_flip_seed123"
    )
    original_precision, original_recall, original_f1 = macro_metrics_from_confusion(
        original["confusion_matrix"]
    )

    records = {
        "Original CNN + Crop/Flip": {
            "experiment": "cnn_crop_flip_seed123",
            "test_accuracy": float(original["accuracy"]),
            "macro_precision": original_precision,
            "macro_recall": original_recall,
            "macro_f1": original_f1,
            "cat_recall": float(original["per_class_recall"]["cat"]),
            "dog_recall": float(original["per_class_recall"]["dog"]),
        },
    }

    for label, name in (
        ("ResNet20 + SE (30 Epoch)", "resnet20_se_seed123"),
        ("ResNet20 + SE + Cosine (100 Epoch)", "resnet20_se_cosine_100_seed123"),
        ("ResNet20 + SE + CutMix (100 Epoch)", "resnet20_se_cutmix_100_seed123"),
    ):
        metrics = json.loads(
            (Path("experiments") / name / "test_metrics.json").read_text()
        )
        records[label] = {
            "experiment": name,
            "test_accuracy": float(metrics["accuracy"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_f1": float(metrics["macro_f1"]),
            "cat_recall": float(metrics["per_class"]["cat"]["recall"]),
            "dog_recall": float(metrics["per_class"]["dog"]["recall"]),
        }

    for label, name in (
        ("Original CNN + Crop/Flip", "cnn_crop_flip_seed123"),
        ("ResNet20 + SE (30 Epoch)", "resnet20_se_seed123"),
        ("ResNet20 + SE + Cosine (100 Epoch)", "resnet20_se_cosine_100_seed123"),
        ("ResNet20 + SE + CutMix (100 Epoch)", "resnet20_se_cutmix_100_seed123"),
    ):
        summary = load_summary(name)
        records[label]["parameters"] = int(summary["parameters"])
        records[label]["best_val_accuracy"] = float(summary["best_val_accuracy"])
        records[label]["best_epoch"] = int(summary["best_epoch"])
        records[label]["training_seconds"] = float(summary["training_seconds"])

    output_dir = Path("experiments/resnet20_se_cutmix_100_seed123")
    (output_dir / "comparison.json").write_text(json.dumps(records, indent=2))

    columns = [
        ("test_accuracy", "Test Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("cat_recall", "Cat Recall (%)"),
        ("dog_recall", "Dog Recall (%)"),
        ("best_val_accuracy", "Best Val Accuracy (%)"),
        ("training_seconds", "Training Time (min)"),
    ]
    with (output_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model"] + [label for _, label in columns])
        for model, result in records.items():
            writer.writerow([
                model,
                *[
                    result[key] / 60 if key == "training_seconds" else result[key]
                    for key, _ in columns
                ],
            ])

    markdown = [
        "| Model | Test Accuracy | Macro F1 | Cat Recall | Dog Recall | Best Val Accuracy | Training Time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, result in records.items():
        markdown.append(
            f"| {model} | {result['test_accuracy']:.2f}% "
            f"| {result['macro_f1']:.2f}% | {result['cat_recall']:.2f}% "
            f"| {result['dog_recall']:.2f}% | {result['best_val_accuracy']:.2f}% "
            f"| {result['training_seconds'] / 60:.2f} min |"
        )
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n")

    metric_keys = ["test_accuracy", "macro_f1", "cat_recall", "dog_recall"]
    labels = ["Test accuracy", "Macro F1", "Cat recall", "Dog recall"]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for index, (model, result) in enumerate(records.items()):
        ax.bar(
            x + (index - 1.5) * width,
            [result[key] for key in metric_keys],
            width,
            label=model,
            color=colors[index],
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("CIFAR-10 historical models and CutMix")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(records, indent=2))
    print(f"Saved comparison artifacts under {output_dir}")


if __name__ == "__main__":
    main()
