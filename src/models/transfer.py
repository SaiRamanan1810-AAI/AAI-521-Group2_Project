"""Utilities to load and prepare pretrained models for fine-tuning."""
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def load_pretrained_model(model_name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """Return a model with the final classifier replaced for `num_classes`.

    Supported `model_name`: 'resnet18', 'mobilenet_v2', 'efficientnet_b0'
    """
    name = model_name.lower()
    if name == "resnet18":
        m = models.resnet18(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(pretrained=pretrained)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=pretrained)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for name, param in m.named_parameters():
            # leave classifier layers trainable
            if 'fc' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return m
