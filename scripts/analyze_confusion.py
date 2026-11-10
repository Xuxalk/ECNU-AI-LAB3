import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.models.baseline_cnn import BaselineCNN


def main(run):
    config = json.loads(
        (Path("experiments") / run / "config.json").read_text()
    )

    checkpoint = Path("checkpoints") / f"{run}_best.pth"

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = BaselineCNN(num_classes=10)

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    norm = config["normalization"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm["mean"], norm["std"])
    ])

    dataset = CIFAR10(
        root=config["data_root"],
        train=True,
        download=False,
        transform=transform
    )

    indices = json.loads(
        Path(config["val_indices"]).read_text()
    )

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    matrix = np.zeros((10, 10), dtype=np.int64)

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(1).cpu().numpy()

            true_labels = labels.numpy()
            np.add.at(matrix, (true_labels, predictions), 1)

    classes = dataset.classes

    per_class = matrix.diagonal() / matrix.sum(axis=1)
    accuracy = matrix.trace() / matrix.sum()

    errors = []

    for i in range(10):
        for j in range(10):
            if i != j and matrix[i, j] > 0:
                errors.append({
                    "true": classes[i],
                    "predicted": classes[j],
                    "count": int(matrix[i, j])
                })

    errors.sort(key=lambda x: x["count"], reverse=True)

    result = {
        "experiment": run,
        "split": "validation",
        "samples": int(matrix.sum()),
        "accuracy": float(accuracy * 100),
        "per_class_accuracy": {
            classes[i]: float(per_class[i] * 100)
            for i in range(10)
        },
        "confusion_matrix": matrix.tolist(),
        "top_confusions": errors[:10]
    }

    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{run}_confusion.json").write_text(
        json.dumps(result, indent=4)
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(10), classes, rotation=45, ha="right")
    ax.set_yticks(range(10), classes)

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Validation Confusion Matrix")

    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center",
                fontsize=8,
                color="white" if matrix[i, j] > 250 else "black"
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = Path("outputs/figures") / f"{run}_confusion.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

    print("\n===== Per-class Accuracy =====")
    for name, value in result["per_class_accuracy"].items():
        print(f"{name:12s}: {value:.2f}%")

    print("\n===== Top Confusions =====")
    for item in errors[:10]:
        print(
            f"{item['true']:12s} -> "
            f"{item['predicted']:12s}: "
            f"{item['count']}"
        )

    print("\nSaved:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    main(args.run)
