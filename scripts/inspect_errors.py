import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.models.baseline_cnn import BaselineCNN

run = "cnn_crop_flip_v1"
config = json.loads(
    (Path("experiments") / run / "config.json").read_text()
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

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

model = BaselineCNN(num_classes=10)
checkpoint = torch.load(
    f"checkpoints/{run}_best.pth",
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device).eval()

classes = dataset.classes

pairs = [
    ("cat", "dog"),
    ("cat", "bird"),
    ("frog", "bird"),
    ("dog", "cat")
]

pair_ids = {
    (classes.index(a), classes.index(b)): (a, b)
    for a, b in pairs
}

errors = {pair: [] for pair in pairs}
offset = 0

with torch.inference_mode():
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = logits.softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)

        for i, (true, pred, conf) in enumerate(zip(
            labels.tolist(),
            predicted.cpu().tolist(),
            confidence.cpu().tolist()
        )):
            key = (true, pred)
            if key in pair_ids:
                pair = pair_ids[key]
                errors[pair].append((indices[offset + i], conf))

        offset += len(labels)

rng = random.Random(42)
fig, axes = plt.subplots(4, 6, figsize=(15, 10))

for row, pair in enumerate(pairs):
    examples = errors[pair]
    chosen = rng.sample(examples, min(6, len(examples)))

    for col, ax in enumerate(axes[row]):
        ax.axis("off")
        if col < len(chosen):
            index, confidence = chosen[col]
            ax.imshow(dataset.data[index])
            ax.set_title(
                f"Confidence {confidence:.0%}",
                fontsize=9
            )

    axes[row, 0].set_ylabel(
        f"{pair[0]} -> {pair[1]}",
        fontsize=11,
        rotation=90,
        labelpad=15
    )

    print(f"{pair[0]} -> {pair[1]}: {len(examples)} errors")

fig.suptitle(
    "CIFAR-10 Validation Misclassification Examples",
    fontsize=15
)
fig.tight_layout()

output = Path(
    "outputs/figures/cnn_crop_flip_v1_error_examples.png"
)
fig.savefig(output, dpi=220, bbox_inches="tight")
plt.close(fig)

print("Saved:", output)
print("ERROR ANALYSIS: PASS")
