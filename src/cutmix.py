"""CutMix helpers for minibatches of normalized CIFAR images."""

import math

import torch


def rand_bbox(width, height, lam):
    """Sample a CutMix rectangle using CPU RNG for resumable randomness."""
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    center_x = int(torch.randint(width, (1,)).item())
    center_y = int(torch.randint(height, (1,)).item())

    x1 = max(center_x - cut_width // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y1 = max(center_y - cut_height // 2, 0)
    y2 = min(center_y + cut_height // 2, height)
    return x1, y1, x2, y2


def apply_cutmix_patch(images, permutation, box):
    """Copy a donor image rectangle into each image in a batch."""
    x1, y1, x2, y2 = box
    mixed_images = images.clone()
    if x1 < x2 and y1 < y2:
        mixed_images[:, :, y1:y2, x1:x2] = images[
            permutation, :, y1:y2, x1:x2
        ]
    return mixed_images


def cutmix_batch(images, labels, alpha=1.0, probability=0.5):
    """Apply CutMix to a batch and return actual-area-adjusted lambda.

    Returns ``(images, labels_a, labels_b, lambda, applied, permutation,
    box)``. When CutMix is skipped, ``labels_b``, ``permutation`` and ``box``
    are ``None`` and lambda is 1.0.
    """
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must have shape [N] and match the batch")
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if labels.device != images.device:
        raise ValueError("images and labels must be on the same device")

    if torch.rand(()).item() >= probability:
        return images, labels, None, 1.0, False, None, None

    initial_lam = torch.distributions.Beta(alpha, alpha).sample().item()
    _, _, height, width = images.shape
    box = rand_bbox(width, height, initial_lam)
    permutation = torch.randperm(images.shape[0]).to(images.device)

    mixed_images = apply_cutmix_patch(images, permutation, box)
    x1, y1, x2, y2 = box
    replaced_area = (x2 - x1) * (y2 - y1)
    actual_lam = 1.0 - replaced_area / (width * height)

    return (
        mixed_images,
        labels,
        labels[permutation],
        actual_lam,
        True,
        permutation,
        box,
    )


def cutmix_loss(criterion, logits, labels_a, labels_b, lam):
    """Blend two cross-entropy losses using the effective retained area."""
    if labels_b is None:
        return criterion(logits, labels_a)
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(
        logits, labels_b
    )
