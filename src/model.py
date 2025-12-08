import torch
import torch.nn as nn
from typing import Dict

try:
    # Newer torchvision
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except Exception:
    from torchvision.models import efficientnet_b0


def load_efficientnet_b0(num_classes: int, pretrained: bool = True):
    try:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
    except Exception:
        model = efficientnet_b0(pretrained=pretrained)

    # Replace classifier
    in_features = model.classifier[1].in_features if hasattr(model, 'classifier') else model.fc.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    return model


def set_parameter_requires_grad(model: torch.nn.Module, feature_extracting: bool):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    m = load_efficientnet_b0(4, pretrained=False)
    print(m)
