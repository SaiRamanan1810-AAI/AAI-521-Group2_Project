"""Baseline CNN model for multiclass classification using PyTorch.

Provides a small convolutional neural network suitable as a baseline.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Simple CNN for image classification.

    Args:
        num_classes: number of target classes.
        in_channels: input channels (3 for RGB).
    """
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def example():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 10)


if __name__ == "__main__":
    example()
