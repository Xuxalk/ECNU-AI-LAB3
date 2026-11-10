"""Evaluate an experiment's best checkpoint and generate its reports."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.models import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metrics_from_confusion(matrix):
    true_positive = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    recall = np.divide(
        true_positive, actual, out=np.zeros_like(true_positive), where=actual != 0
    )
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return precision, recall, f1


def plot_confusion(matrix, classes, output_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test set confusion matrix")
    cutoff = matrix.max() * 0.55
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(
                col, row, str(int(matrix[row, col])),
                ha="center", va="center", fontsize=8,
                color="white" if matrix[row, col] > cutoff else "black",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_history(history_path, output_path, best_epoch, cutmix=False):
    with history_path.open() as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Training history is empty: {history_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    val_accuracy = [float(row["val_accuracy"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    train_loss_label = "Train CutMix objective" if cutmix else "Train"
    train_accuracy_label = (
        "Train (lambda-weighted)" if cutmix else "Train"
    )
    axes[0].plot(epochs, train_loss, label=train_loss_label)
    axes[0].plot(epochs, val_loss, label="Validation CE")
    axes[0].set(
        title="CutMix objective and validation loss" if cutmix else "Loss",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axes[1].plot(epochs, train_accuracy, label=train_accuracy_label)
    axes[1].plot(epochs, val_accuracy, label="Validation")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    axes[1].set(
        title="Accuracy (mixed-label train metric)" if cutmix else "Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
    )
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_learning_rate(rows, output_path):
    epochs = [int(row["epoch"]) for row in rows]
    learning_rates = [float(row["learning_rate"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, learning_rates, color="tab:purple")
    ax.set(
        title="Learning rate used during each epoch",
        xlabel="Epoch",
        ylabel="Learning rate",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(run):
    run_dir = Path("experiments") / run
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    checkpoint_path = Path("checkpoints") / f"{run}_best.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )

    dataset = CIFAR10(
        root=config["data_root"], train=False, download=False
    )
    normalize = transforms.Normalize(
        config["normalization"]["mean"], config["normalization"]["std"]
    )
    dataset.transform = transforms.Compose([transforms.ToTensor(), normalize])
    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0
    )

    device = get_device()
    model = build_model(config, num_classes=len(dataset.classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    matrix = np.zeros((len(dataset.classes), len(dataset.classes)), dtype=np.int64)
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
            np.add.at(matrix, (labels.numpy(), predictions), 1)

    precision, recall, f1 = metrics_from_confusion(matrix)
    class_metrics = {
        name: {
            "precision": float(precision[index] * 100),
            "recall": float(recall[index] * 100),
            "f1": float(f1[index] * 100),
        }
        for index, name in enumerate(dataset.classes)
    }
    result = {
        "experiment": run,
        "checkpoint": str(checkpoint_path),
        "split": "test",
        "samples": int(matrix.sum()),
        "accuracy": float(matrix.trace() / matrix.sum() * 100),
        "macro_precision": float(precision.mean() * 100),
        "macro_recall": float(recall.mean() * 100),
        "macro_f1": float(f1.mean() * 100),
        "per_class": class_metrics,
        "confusion_matrix": matrix.tolist(),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": int(summary["best_epoch"]),
        "best_val_accuracy": float(summary["best_val_accuracy"]),
        "training_seconds": float(summary["training_seconds"]),
        "device": str(device),
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))

    with (run_dir / "class_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1"])
        for name, metrics in class_metrics.items():
            writer.writerow([name, metrics["precision"], metrics["recall"], metrics["f1"]])

    plot_confusion(matrix, dataset.classes, run_dir / "confusion_matrix.png")
    history = plot_history(
        run_dir / "history.csv",
        run_dir / "training_curves.png",
        result["best_epoch"],
        cutmix="cutmix" in config,
    )
    if "learning_rate" in history[0]:
        plot_learning_rate(history, run_dir / "learning_rate.png")
    result["final_train_accuracy"] = float(history[-1]["train_accuracy"])
    result["final_val_accuracy"] = float(history[-1]["val_accuracy"])
    result["final_accuracy_gap_pp"] = (
        result["final_train_accuracy"] - result["final_val_accuracy"]
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))

    print(json.dumps({key: result[key] for key in (
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "best_epoch", "best_val_accuracy", "parameters", "training_seconds",
        "device", "final_accuracy_gap_pp",
    )}, indent=2))
    print("Per-class recall:")
    for name, metrics in class_metrics.items():
        print(f"  {name:12s} {metrics['recall']:.2f}%")
    print(f"Saved reports and figures under {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    main(parser.parse_args().run)
