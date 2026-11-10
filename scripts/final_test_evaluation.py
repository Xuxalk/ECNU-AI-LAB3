import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from src.models.baseline_cnn import BaselineCNN

config = json.loads(
    Path("configs/crop_flip.json").read_text()
)

dataset = CIFAR10(
    root="data",
    train=False,
    download=False
)

labels = np.asarray(dataset.targets)

images = torch.from_numpy(
    dataset.data
).permute(0, 3, 1, 2).float() / 255.0

mean = torch.tensor(
    config["normalization"]["mean"]
).view(1, 3, 1, 1)

std = torch.tensor(
    config["normalization"]["std"]
).view(1, 3, 1, 1)

images = (images - mean) / std

classes = dataset.classes
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

runs = {
    "A": [
        "baseline_cnn_v1"
    ],
    "B": [
        "cnn_crop_flip_v1",
        "cnn_crop_flip_seed123",
        "cnn_crop_flip_seed2026"
    ],
    "C": [
        "cnn_crop_flip_colorjitter_v1"
    ],
    "D": [
        "cnn_crop_flip_cutout_v1",
        "cnn_crop_flip_cutout_seed123",
        "cnn_crop_flip_cutout_seed2026"
    ]
}
predictions = {}
results = {}

def evaluate(name):
    model = BaselineCNN(num_classes=10)

    checkpoint = torch.load(
        f"checkpoints/{name}_best.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    output = []

    with torch.inference_mode():
        for start in range(0, len(images), 256):
            batch = images[start:start+256].to(device)
            output.extend(
                model(batch).argmax(1).cpu().tolist()
            )

    return np.asarray(output)

for method, names in runs.items():
    results[method] = []

    for name in names:
        pred = evaluate(name)
        predictions[name] = pred

        cm = np.zeros((10, 10), dtype=int)
        np.add.at(cm, (labels, pred), 1)

        recall = {
            classes[i]: round(
                100 * cm[i, i] / cm[i].sum(), 2
            )
            for i in range(10)
        }

        accuracy = 100 * np.mean(pred == labels)

        item = {
            "run": name,
            "accuracy": round(float(accuracy), 3),
            "per_class_recall": recall,
            "confusion_matrix": cm.tolist()
        }

        results[method].append(item)

        print(
            f"\n{method} | {name} | "
            f"Test Accuracy: {accuracy:.2f}%"
        )

        print(
            "Cat Recall:", recall["cat"],
            "| Dog Recall:", recall["dog"]
        )

print("\n===== PAIRED TEST ANALYSIS =====")

for b_name, d_name in zip(runs["B"], runs["D"]):
    b = predictions[b_name]
    d = predictions[d_name]

    print("\n", b_name, "vs", d_name)

    for cls in ["cat", "dog"]:
        idx = classes.index(cls)
        mask = labels == idx

        b_correct = b[mask] == idx
        d_correct = d[mask] == idx

        print(
            cls,
            "B only correct:", int(np.sum(b_correct & ~d_correct)),
            "D only correct:", int(np.sum(~b_correct & d_correct))
        )

output = Path(
    "outputs/metrics/final_test_results.json"
)

output.write_text(json.dumps(results, indent=2))

print("\nSaved:", output)
print("FINAL TEST EVALUATION: PASS")
