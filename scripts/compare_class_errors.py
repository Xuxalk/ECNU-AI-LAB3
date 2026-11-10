import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from src.models.baseline_cnn import BaselineCNN

runs = {
    "B": [
        "cnn_crop_flip_v1",
        "cnn_crop_flip_seed123",
        "cnn_crop_flip_seed2026",
    ],
    "D": [
        "cnn_crop_flip_cutout_v1",
        "cnn_crop_flip_cutout_seed123",
        "cnn_crop_flip_cutout_seed2026",
    ],
}

config = json.loads(
    Path("configs/crop_flip.json").read_text()
)

indices = json.loads(
    Path(config["val_indices"]).read_text()
)

dataset = CIFAR10(
    root="data", train=True, download=False
)

classes = dataset.classes
labels = np.asarray(dataset.targets)[indices]

images = torch.from_numpy(
    dataset.data[np.asarray(indices)]
).permute(0, 3, 1, 2).float() / 255.0

mean = torch.tensor(
    config["normalization"]["mean"]
).view(1, 3, 1, 1)

std = torch.tensor(
    config["normalization"]["std"]
).view(1, 3, 1, 1)

images = (images - mean) / std

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

pairs = [
    ("cat", "dog"),
    ("cat", "bird"),
    ("frog", "bird"),
    ("dog", "cat"),
]

results = {}

for method, names in runs.items():
    results[method] = []

    for name in names:
        model = BaselineCNN(num_classes=10)

        checkpoint = torch.load(
            f"checkpoints/{name}_best.pth",
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model = model.to(device).eval()

        predictions = []

        with torch.inference_mode():
            for start in range(0, len(images), 256):
                batch = images[start:start+256].to(device)
                pred = model(batch).argmax(1)
                predictions.extend(pred.cpu().tolist())

        predictions = np.asarray(predictions)

        per_class = {}
        for i, label in enumerate(classes):
            mask = labels == i
            per_class[label] = float(
                100 * np.mean(predictions[mask] == i)
            )

        directed_errors = {}
        for true, predicted in pairs:
            a = classes.index(true)
            b = classes.index(predicted)
            directed_errors[f"{true}->{predicted}"] = int(
                np.sum((labels == a) & (predictions == b))
            )

        bird = classes.index("bird")
        false_bird = int(np.sum(
            (labels != bird) & (predictions == bird)
        ))

        results[method].append({
            "run": name,
            "accuracy": float(
                100 * np.mean(predictions == labels)
            ),
            "per_class": per_class,
            "directed_errors": directed_errors,
            "false_bird": false_bird,
        })

        print(f"{method} | {name}: complete")

def summarize(values):
    return (
        f"{statistics.mean(values):.2f} "
        f"+/- {statistics.stdev(values):.2f}"
    )

print("\n===== PER-CLASS RECALL (%) =====")

for label in ["cat", "dog", "bird", "frog"]:
    b = [r["per_class"][label] for r in results["B"]]
    d = [r["per_class"][label] for r in results["D"]]

    print(f"{label:8s} B: {summarize(b)} | D: {summarize(d)}")

print("\n===== DIRECTED ERRORS (COUNT) =====")

for true, predicted in pairs:
    key = f"{true}->{predicted}"
    b = [r["directed_errors"][key] for r in results["B"]]
    d = [r["directed_errors"][key] for r in results["D"]]

    print(f"{key:12s} B: {summarize(b)} | D: {summarize(d)}")

print("\n===== FALSE BIRD PREDICTIONS =====")

for method in ["B", "D"]:
    values = [r["false_bird"] for r in results[method]]
    print(method, summarize(values))

output = Path(
    "outputs/metrics/class_error_multiseed.json"
)

output.write_text(json.dumps(results, indent=2))

print("\nSaved:", output)
print("CLASS ERROR ANALYSIS: PASS")
