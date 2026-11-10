import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
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

config = json.loads(
    (Path("experiments") / RUN / "config.json").read_text()
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
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

errors = {pair: [] for pair in PAIRS}
offset = 0

with torch.inference_mode():
    for images, labels in loader:
        probabilities = model(
            images.to(device)
        ).softmax(dim=1)

        confidence, predicted = probabilities.max(dim=1)

        for i, (true, pred, conf) in enumerate(zip(
            labels.tolist(),
            predicted.cpu().tolist(),
            confidence.cpu().tolist()
        )):
            key = (true, pred)
            if key in pair_ids:
                errors[pair_ids[key]].append(
                    (indices[offset + i], conf)
                )

        offset += len(labels)


activations = []

def save_activation(module, inputs, output):
    activations.append(output)


target_layer = model.features[2].layers[3]
handle = target_layer.register_forward_hook(
    save_activation
)


def make_cam(image_tensor, target_class):
    activations.clear()

    logits = model(image_tensor)
    feature = activations[-1]

    gradients = torch.autograd.grad(
        logits[0, target_class],
        feature
    )[0]

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu(
        (weights * feature).sum(dim=1, keepdim=True)
    )

    cam = F.interpolate(
        cam,
        size=(32, 32),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.detach().float().cpu().numpy()[0, 0]
    cam = cam - cam.min()

    if cam.max() > 1e-8:
        cam = cam / cam.max()

    return cam


def overlay(image, cam):
    rgb = np.asarray(image, dtype=np.float32) / 255
    heatmap = plt.get_cmap("inferno")(cam)[..., :3]

    return np.clip(
        0.55 * rgb + 0.45 * heatmap,
        0, 1
    )


fig, axes = plt.subplots(4, 6, figsize=(16, 10))

for row, pair in enumerate(PAIRS):
    examples = sorted(
        errors[pair],
        key=lambda item: item[1]
    )

    assert examples, f"No errors found for {pair}"

    selected = [
        examples[len(examples) // 2],
        examples[-1]
    ]

    true_id = classes.index(pair[0])
    pred_id = classes.index(pair[1])

    for j, (index, confidence) in enumerate(selected):
        image = Image.fromarray(dataset.data[index])
        x = transform(image).unsqueeze(0).to(device)

        predicted_cam = make_cam(x, pred_id)
        true_cam = make_cam(x, true_id)

        col = j * 3

        axes[row, col].imshow(image, interpolation="nearest")
        axes[row, col + 1].imshow(
            overlay(image, predicted_cam)
        )
        axes[row, col + 2].imshow(
            overlay(image, true_cam)
        )

        axes[row, col].set_title(
            f"Original | {confidence:.1%}",
            fontsize=10
        )
        axes[row, col + 1].set_title(
            f"Predicted: {pair[1]}",
            fontsize=10
        )
        axes[row, col + 2].set_title(
            f"True: {pair[0]}",
            fontsize=10
        )

    axes[row, 0].set_ylabel(
        f"{pair[0]} -> {pair[1]}",
        fontsize=11
    )

    for ax in axes[row]:
        ax.set_xticks([])
        ax.set_yticks([])

    print(
        f"{pair[0]} -> {pair[1]}: "
        f"{len(examples)} candidates"
    )

handle.remove()

fig.suptitle(
    "Grad-CAM Analysis of CIFAR-10 Misclassifications",
    fontsize=16
)
fig.tight_layout()

output = Path(
    "outputs/figures/gradcam_error_examples.png"
)

fig.savefig(output, dpi=220, bbox_inches="tight")
plt.close(fig)

print("Device:", device)
print("Target layer:", target_layer)
print("Saved:", output)
print("GRAD-CAM ANALYSIS: PASS")
