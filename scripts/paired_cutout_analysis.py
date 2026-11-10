import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torchvision.datasets import CIFAR10

from src.models.baseline_cnn import BaselineCNN

config = json.loads(
    Path("configs/crop_flip.json").read_text()
)

indices = json.loads(
    Path(config["val_indices"]).read_text()
)

dataset = CIFAR10(
    root="data",
    train=True,
    download=False
)

labels = np.array(dataset.targets)[indices]

images = torch.from_numpy(
    dataset.data[np.array(indices)]
).permute(0, 3, 1, 2).float() / 255.0

mean = torch.tensor(
    config["normalization"]["mean"]
).view(1, 3, 1, 1)

std = torch.tensor(
    config["normalization"]["std"]
).view(1, 3, 1, 1)

images = (images - mean) / std

classes = dataset.classes
cat = classes.index("cat")
dog = classes.index("dog")

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

pairs = [
    ("cnn_crop_flip_v1", "cnn_crop_flip_cutout_v1"),
    ("cnn_crop_flip_seed123", "cnn_crop_flip_cutout_seed123"),
    ("cnn_crop_flip_seed2026", "cnn_crop_flip_cutout_seed2026"),
]

def predict(name):
    model = BaselineCNN(num_classes=10)

    checkpoint = torch.load(
        f"checkpoints/{name}_best.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    predictions = []

    with torch.inference_mode():
        for start in range(0, len(images), 256):
            batch = images[start:start+256].to(device)
            pred = model(batch).argmax(dim=1)
            predictions.extend(pred.cpu().tolist())

    return np.array(predictions)


results = {}

for b_name, d_name in pairs:
    b = predict(b_name)
    d = predict(d_name)

    seed = b_name.split("_")[-1] if "seed" in b_name else "42"

    results[seed] = {}

    print(f"\n{'='*55}")
    print(f"SEED: {seed}")

    for cls, class_id in [("cat", cat), ("dog", dog)]:
        mask = labels == class_id

        bp = b[mask]
        dp = d[mask]

        both_correct = np.sum(
            (bp == class_id) & (dp == class_id)
        )

        b_only = np.sum(
            (bp == class_id) & (dp != class_id)
        )

        d_only = np.sum(
            (bp != class_id) & (dp == class_id)
        )

        both_wrong = np.sum(
            (bp != class_id) & (dp != class_id)
        )

        transitions = Counter(
            zip(bp.tolist(), dp.tolist())
        )

        print(f"\nTrue class: {cls}")
        print("Both correct:", both_correct)
        print("B correct, D wrong:", b_only)
        print("B wrong, D correct:", d_only)
        print("Both wrong:", both_wrong)

        print("Prediction transitions:")

        transition_results = []

        for (old, new), count in transitions.most_common():
            if old != new:
                item = {
                    "from": classes[old],
                    "to": classes[new],
                    "count": count
                }
                transition_results.append(item)

                print(
                    f"  {classes[old]} -> {classes[new]}: {count}"
                )

        results[seed][cls] = {
            "both_correct": int(both_correct),
            "b_only_correct": int(b_only),
            "d_only_correct": int(d_only),
            "both_wrong": int(both_wrong),
            "transitions": transition_results
        }

output = Path(
    "outputs/metrics/paired_cutout_analysis.json"
)

output.write_text(json.dumps(results, indent=2))

print("\nSaved:", output)
print("PAIRED ANALYSIS: PASS")
