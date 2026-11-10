"""Model factory for experiment configurations."""

from src.models.baseline_cnn import BaselineCNN
from models.resnet20_se import ResNet20SE


def build_model(config, num_classes):
    """Construct a model from a config, defaulting to the legacy CNN."""
    model_name = config.get("model", "baseline_cnn")

    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if model_name == "resnet20_se":
        return ResNet20SE(
            num_classes=num_classes,
            reduction=config.get("se_reduction", 16),
        )

    raise ValueError(f"Unknown model: {model_name}")
