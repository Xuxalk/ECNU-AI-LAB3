import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

root = Path("outputs/metrics")

train = CIFAR10(root="data", train=True, download=False)

train_indices = json.loads(
    (root / "train_indices.json").read_text()
)
val_indices = json.loads(
    (root / "val_indices.json").read_text()
)

assert len(train_indices) == 45000
assert len(val_indices) == 5000
assert len(set(train_indices)) == 45000
assert len(set(val_indices)) == 5000
assert not set(train_indices) & set(val_indices)
assert sorted(train_indices + val_indices) == list(range(50000))

targets = np.asarray(train.targets)

train_counts = np.bincount(
    targets[train_indices], minlength=10
)
val_counts = np.bincount(
    targets[val_indices], minlength=10
)

assert np.all(train_counts == 4500)
assert np.all(val_counts == 500)

images = train.data[np.asarray(train_indices)]

mean = images.mean(axis=(0, 1, 2)) / 255.0
std = images.std(axis=(0, 1, 2)) / 255.0

result = {
    "train_size": 45000,
    "val_size": 5000,
    "seed": 42,
    "train_class_counts": train_counts.tolist(),
    "val_class_counts": val_counts.tolist(),
    "normalization_mean": mean.tolist(),
    "normalization_std": std.tolist(),
    "overlap": 0,
    "missing_samples": 0,
}

(root / "split_audit.json").write_text(
    json.dumps(result, indent=4)
)

(root / "normalization.json").write_text(
    json.dumps({
        "mean": mean.tolist(),
        "std": std.tolist()
    }, indent=4)
)

print("===== Split Audit =====")
print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))
print("Training class counts:", train_counts.tolist())
print("Validation class counts:", val_counts.tolist())
print("Overlap: 0")
print("Missing samples: 0")
print("Normalization mean:", mean)
print("Normalization std:", std)
print("\nSPLIT AUDIT: PASS")
