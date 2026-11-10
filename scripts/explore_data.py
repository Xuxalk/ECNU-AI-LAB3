import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
from torchvision.datasets import CIFAR10


root = Path("data")
output = Path("outputs")
figures = output / "figures"
metrics = output / "metrics"

figures.mkdir(parents=True, exist_ok=True)
metrics.mkdir(parents=True, exist_ok=True)

print("Downloading CIFAR-10...")

train = CIFAR10(root=root, train=True, download=True)
test = CIFAR10(root=root, train=False, download=True)

classes = train.classes
train_counts = Counter(train.targets)
test_counts = Counter(test.targets)

print("\n===== Dataset Information =====")
print("Training samples:", len(train))
print("Testing samples:", len(test))
print("Image shape:", train.data.shape[1:])
print("Image dtype:", train.data.dtype)
print("Number of classes:", len(classes))

print("\n===== Class Distribution =====")
for i, name in enumerate(classes):
    print(
        f"{name:12s} "
        f"train={train_counts[i]:5d} "
        f"test={test_counts[i]:5d}"
    )

summary = {
    "dataset": "CIFAR-10",
    "train_samples": len(train),
    "test_samples": len(test),
    "image_shape": list(train.data.shape[1:]),
    "num_classes": len(classes),
    "classes": classes,
    "train_class_counts": dict(train_counts),
    "test_class_counts": dict(test_counts),
}

with open(metrics / "data_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

scale = 4
image_size = 32 * scale
panel_width = image_size * 3 + 40
panel_height = image_size + 36

sheet = Image.new(
    "RGB",
    (panel_width * 5, panel_height * 2),
    "white"
)
draw = ImageDraw.Draw(sheet)

for class_id, class_name in enumerate(classes):
    row = class_id // 5
    col = class_id % 5

    x0 = col * panel_width
    y0 = row * panel_height

    draw.text((x0 + 10, y0 + 5), class_name, fill="black")

    indices = [
        i for i, label in enumerate(train.targets)
        if label == class_id
    ][:3]

    for j, idx in enumerate(indices):
        image = Image.fromarray(train.data[idx])
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.NEAREST
        )
        sheet.paste(
            image,
            (x0 + 10 + j * (image_size + 10), y0 + 25)
        )

sheet.save(figures / "cifar10_samples.png")

print("\nSaved:")
print("outputs/metrics/data_summary.json")
print("outputs/figures/cifar10_samples.png")

assert len(train) == 50000
assert len(test) == 10000
assert all(train_counts[i] == 5000 for i in range(10))
assert all(test_counts[i] == 1000 for i in range(10))

print("\nDATASET AUDIT: PASS")
