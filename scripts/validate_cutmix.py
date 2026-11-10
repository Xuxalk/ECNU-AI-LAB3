"""Validate CutMix geometry, target mixing, loss, reproducibility and MPS."""

import json

import torch
from torch import nn

from models.resnet20_se import ResNet20SE
from src.cutmix import (
    apply_cutmix_patch,
    cutmix_batch,
    cutmix_loss,
)


def validate_patch_and_lambda():
    images = torch.stack([
        torch.zeros(1, 4, 4),
        torch.ones(1, 4, 4),
    ])
    permutation = torch.tensor([1, 0])
    box = (1, 1, 3, 3)
    mixed = apply_cutmix_patch(images, permutation, box)
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[1:3, 1:3] = True
    assert torch.all(mixed[0, 0][mask] == 1)
    assert torch.all(mixed[0, 0][~mask] == 0)
    expected_lambda = 1 - (mask.sum().item() / 16)
    assert expected_lambda == 0.75

    labels = torch.tensor([4, 9])
    torch.manual_seed(23)
    (mixed, target_a, target_b, lam, applied, permutation, box) = cutmix_batch(
        images, labels, alpha=1.0, probability=1.0
    )
    assert applied
    assert torch.equal(target_a, labels)
    assert torch.equal(target_b, labels[permutation])
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    assert abs(lam - (1 - area / 16)) < 1e-12
    assert torch.equal(
        mixed[:, :, y1:y2, x1:x2],
        images[permutation, :, y1:y2, x1:x2],
    )


def validate_skip_and_loss():
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    result = cutmix_batch(images, labels, alpha=1.0, probability=0.0)
    mixed, target_a, target_b, lam, applied, permutation, box = result
    assert not applied and target_b is None and permutation is None and box is None
    assert lam == 1.0 and mixed is images and torch.equal(target_a, labels)

    logits = torch.tensor([
        [2.0, 0.1, -0.5],
        [0.2, 1.5, -0.1],
    ])
    labels_a = torch.tensor([0, 1])
    labels_b = torch.tensor([2, 0])
    lam = 0.75
    criterion = nn.CrossEntropyLoss()
    expected = (
        lam * criterion(logits, labels_a)
        + (1 - lam) * criterion(logits, labels_b)
    )
    actual = cutmix_loss(criterion, logits, labels_a, labels_b, lam)
    assert torch.allclose(actual, expected)
    assert torch.allclose(
        cutmix_loss(criterion, logits, labels_a, None, 1.0),
        criterion(logits, labels_a),
    )


def validate_reproducibility():
    images = torch.arange(6 * 3 * 8 * 8, dtype=torch.float32).reshape(6, 3, 8, 8)
    labels = torch.arange(6)
    torch.manual_seed(123)
    first = cutmix_batch(images, labels, alpha=1.0, probability=1.0)
    torch.manual_seed(123)
    second = cutmix_batch(images, labels, alpha=1.0, probability=1.0)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[2], second[2])
    assert first[3] == second[3] and first[6] == second[6]
    assert torch.equal(first[5], second[5])


def validate_mps():
    if not torch.backends.mps.is_available():
        return "unavailable"
    device = torch.device("mps")
    images = torch.randn(4, 3, 32, 32, device=device)
    labels = torch.tensor([0, 1, 2, 3], device=device)
    mixed, target_a, target_b, lam, applied, _, _ = cutmix_batch(
        images, labels, alpha=1.0, probability=1.0
    )
    model = ResNet20SE(num_classes=10).to(device)
    logits = model(mixed)
    criterion = nn.CrossEntropyLoss()
    loss = cutmix_loss(criterion, logits, target_a, target_b, lam)
    loss.backward()
    assert applied and mixed.device.type == device.type
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
    return "PASS"


def main():
    validate_patch_and_lambda()
    validate_skip_and_loss()
    validate_reproducibility()
    mps_status = validate_mps()
    print(json.dumps({
        "patch_area_and_lambda": "PASS",
        "two_label_loss_and_skip_path": "PASS",
        "seeded_reproducibility": "PASS",
        "mps_forward_backward": mps_status,
    }, indent=2))


if __name__ == "__main__":
    main()
