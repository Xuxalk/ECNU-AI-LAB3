"""Compare the seed-123 crop/flip CNN and ResNet20+SE test results."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def class_statistics(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    diagonal = np.diag(matrix)
    recall = np.divide(
        diagonal, matrix.sum(axis=1),
        out=np.zeros_like(diagonal), where=matrix.sum(axis=1) != 0,
    )
    precision = np.divide(
        diagonal, matrix.sum(axis=0),
        out=np.zeros_like(diagonal), where=matrix.sum(axis=0) != 0,
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(diagonal), where=(precision + recall) != 0,
    )
    return precision, recall, f1


def main():
    baseline_name = "cnn_crop_flip_seed123"
    new_name = "resnet20_se_seed123"
    old_results = json.loads(
        Path("outputs/metrics/final_test_results.json").read_text()
    )
    baseline_result = next(
        item for item in old_results["B"] if item["run"] == baseline_name
    )
    baseline_summary = json.loads(
        (Path("experiments") / baseline_name / "summary.json").read_text()
    )
    new_result = json.loads(
        (Path("experiments") / new_name / "test_metrics.json").read_text()
    )

    _, baseline_recall, baseline_f1 = class_statistics(
        baseline_result["confusion_matrix"]
    )
    models = {
        "Original CNN": {
            "parameters": int(baseline_summary["parameters"]),
            "test_accuracy": float(baseline_result["accuracy"]),
            "macro_f1": float(baseline_f1.mean() * 100),
            "cat_recall": float(baseline_recall[3] * 100),
            "dog_recall": float(baseline_recall[5] * 100),
            "training_seconds": float(baseline_summary["training_seconds"]),
        },
        "ResNet20 + SE": {
            "parameters": int(new_result["parameters"]),
            "test_accuracy": float(new_result["accuracy"]),
            "macro_f1": float(new_result["macro_f1"]),
            "cat_recall": float(new_result["per_class"]["cat"]["recall"]),
            "dog_recall": float(new_result["per_class"]["dog"]["recall"]),
            "training_seconds": float(new_result["training_seconds"]),
        },
    }
    output_dir = Path("experiments") / new_name
    (output_dir / "comparison.json").write_text(json.dumps(models, indent=2))

    columns = [
        ("parameters", "Parameters"),
        ("test_accuracy", "Test Accuracy (%)"),
        ("macro_f1", "Macro F1 (%)"),
        ("cat_recall", "Cat Recall (%)"),
        ("dog_recall", "Dog Recall (%)"),
        ("training_seconds", "Training Time (s)"),
    ]
    with (output_dir / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model"] + [label for _, label in columns])
        for name, result in models.items():
            writer.writerow([name] + [result[key] for key, _ in columns])

    markdown = [
        "| Metric | Original CNN | ResNet20 + SE |",
        "|---|---:|---:|",
    ]
    for key, label in columns:
        old = models["Original CNN"][key]
        new = models["ResNet20 + SE"][key]
        if key == "parameters":
            old_text, new_text = f"{old:,}", f"{new:,}"
        elif key == "training_seconds":
            old_text, new_text = f"{old / 60:.2f} min", f"{new / 60:.2f} min"
        else:
            old_text, new_text = f"{old:.2f}%", f"{new:.2f}%"
        markdown.append(f"| {label} | {old_text} | {new_text} |")
    (output_dir / "comparison.md").write_text("\n".join(markdown) + "\n")

    metric_keys = ["test_accuracy", "macro_f1", "cat_recall", "dog_recall"]
    labels = ["Test accuracy", "Macro F1", "Cat recall", "Dog recall"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        [models["Original CNN"][key] for key in metric_keys],
        width, label="Original CNN",
    )
    ax.bar(
        x + width / 2,
        [models["ResNet20 + SE"][key] for key in metric_keys],
        width, label="ResNet20 + SE",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Seed 123: CIFAR-10 test comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(models, indent=2))
    print(f"Saved comparison artifacts under {output_dir}")


if __name__ == "__main__":
    main()
