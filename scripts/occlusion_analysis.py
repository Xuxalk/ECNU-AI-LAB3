import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.models.baseline_cnn import BaselineCNN


RUN = "cnn_crop_flip_v1"
PAIRS = [
    ("cat", "dog"),
    ("cat", "bird"),
    ("frog", "bird"),
    ("dog", "cat"),
]

PATCH = 8
STRIDE = 4

config = json.loads(
    (Path("experiments") / RUN / "config.json").read_text()
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
    f"checkpoints/{RUN}_best.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device).eval()

classes = dataset.classes

pair_ids = {
    (classes.index(a), classes.index(b)): (a, b)
    for a, b in PAIRS
}

# 找出每种错误中置信度最高的样本
selected = {pair: None for pair in PAIRS}
offset = 0

with torch.inference_mode():
    for images, labels in loader:
        logits = model(images.to(device))
        probs = logits.softmax(dim=1)
        confidence, predictions = probs.max(dim=1)

        for i in range(len(labels)):
            true = labels[i].item()
            pred = predictions[i].item()

            key = (true, pred)

            if key not in pair_ids:
                continue

            pair = pair_ids[key]
            conf = confidence[i].item()

            old = selected[pair]

            if old is None or conf > old[0]:
                selected[pair] = (
                    conf,
                    indices[offset + i]
                )

        offset += len(labels)

positions = [
    (y, x)
    for y in range(0, 32 - PATCH + 1, STRIDE)
    for x in range(0, 32 - PATCH + 1, STRIDE)
]

cases = []
all_maps = []

for pair in PAIRS:
    result = selected[pair]
    assert result is not None, f"No sample found: {pair}"

    confidence, index = result
    x, label = dataset[index]
    x = x.unsqueeze(0).to(device)

    true_id = classes.index(pair[0])
    pred_id = classes.index(pair[1])

    with torch.inference_mode():
        baseline = model(x)[0]

        masked_images = []

        for y, xx in positions:
            masked = x.clone()
            masked[:, :, y:y+PATCH, xx:xx+PATCH] = 0.0
            masked_images.append(masked.squeeze(0))

        masked_images = torch.stack(masked_images)
        masked_logits = model(masked_images)

        pred_drops = (
            baseline[pred_id] - masked_logits[:, pred_id]
        ).cpu().numpy()

        true_drops = (
            baseline[true_id] - masked_logits[:, true_id]
        ).cpu().numpy()

    def build_map(drops):
        total = np.zeros((32, 32))
        counts = np.zeros((32, 32))

        for value, (y, xx) in zip(drops, positions):
            total[y:y+PATCH, xx:xx+PATCH] += value
            counts[y:y+PATCH, xx:xx+PATCH] += 1

        return total / np.maximum(counts, 1)

    pred_map = build_map(pred_drops)
    true_map = build_map(true_drops)

    all_maps.extend([pred_map, true_map])

    cases.append({
        "pair": pair,
        "index": index,
        "confidence": confidence,
        "image": dataset.data[index],
        "pred_map": pred_map,
        "true_map": true_map,
        "pred_baseline_logit": baseline[pred_id].item(),
        "true_baseline_logit": baseline[true_id].item()
    })

# 使用统一颜色范围，便于不同图片之间比较
vmax = max(
    max(float(np.abs(m).max()) for m in all_maps),
    1e-6
)

fig, axes = plt.subplots(4, 3, figsize=(12, 13))

for row, case in enumerate(cases):
    image = case["image"]

    axes[row, 0].imshow(image, interpolation="nearest")
    axes[row, 0].set_title(
        f"Original | {case['confidence']:.1%}"
    )

    axes[row, 1].imshow(image, interpolation="nearest")
    axes[row, 1].imshow(
        case["pred_map"],
        cmap="coolwarm",
        alpha=0.55,
        vmin=-vmax,
        vmax=vmax
    )
    axes[row, 1].set_title(
        f"Predicted: {case['pair'][1]}"
    )

    axes[row, 2].imshow(image, interpolation="nearest")
    axes[row, 2].imshow(
        case["true_map"],
        cmap="coolwarm",
        alpha=0.55,
        vmin=-vmax,
        vmax=vmax
    )
    axes[row, 2].set_title(
        f"True: {case['pair'][0]}"
    )

    axes[row, 0].set_ylabel(
        f"{case['pair'][0]} -> {case['pair'][1]}"
    )

    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])

fig.suptitle(
    "Occlusion Sensitivity Analysis",
    fontsize=16
)

fig.text(
    0.5, 0.01,
    "Red: masking reduces the target logit. "
    "Blue: masking increases the target logit. "
    "Patch: 8x8, stride: 4.",
    ha="center",
    fontsize=10
)

fig.tight_layout(rect=[0, 0.04, 1, 0.97])

output = Path(
    "outputs/figures/occlusion_error_analysis.png"
)

fig.savefig(output, dpi=250, bbox_inches="tight")
plt.close(fig)

summary = []

for case in cases:
    summary.append({
        "pair": list(case["pair"]),
        "sample_index": case["index"],
        "confidence": case["confidence"],
        "predicted_logit": case["pred_baseline_logit"],
        "true_logit": case["true_baseline_logit"],
        "predicted_max_positive_drop": float(
            case["pred_map"].max()
        ),
        "true_max_positive_drop": float(
            case["true_map"].max()
        )
    })

Path("outputs/metrics/occlusion_analysis.json").write_text(
    json.dumps(summary, indent=4)
)

print("Device:", device)
print("Selected samples:", len(cases))
print("Patch:", PATCH, "Stride:", STRIDE)
print("Saved:", output)
print("Saved: outputs/metrics/occlusion_analysis.json")
print("OCCLUSION ANALYSIS: PASS")
