import json
import random
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import CIFAR10

SEED = 42
VAL_PER_CLASS = 500

root = Path("data")
out_dir = Path("outputs/metrics")
out_dir.mkdir(parents=True, exist_ok=True)

random.seed(SEED)

train_set = CIFAR10(root=root, train=True, download=False)

class_to_indices = defaultdict(list)
for idx, label in enumerate(train_set.targets):
    class_to_indices[label].append(idx)

train_indices = []
val_indices = []

for label in range(10):
    indices = class_to_indices[label]
    random.shuffle(indices)
    val_part = indices[:VAL_PER_CLASS]
    train_part = indices[VAL_PER_CLASS:]
    val_indices.extend(val_part)
    train_indices.extend(train_part)

random.shuffle(train_indices)
random.shuffle(val_indices)

summary = {
    "seed": SEED,
    "train_size": len(train_indices),
    "val_size": len(val_indices),
    "val_per_class": VAL_PER_CLASS,
    "classes": train_set.classes,
}

with open(out_dir / "split_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

with open(out_dir / "train_indices.json", "w") as f:
    json.dump(train_indices, f)

with open(out_dir / "val_indices.json", "w") as f:
    json.dump(val_indices, f)

print("Split prepared successfully")
print("Train size:", len(train_indices))
print("Val size:", len(val_indices))
print("Saved:")
print("outputs/metrics/split_summary.json")
print("outputs/metrics/train_indices.json")
print("outputs/metrics/val_indices.json")
