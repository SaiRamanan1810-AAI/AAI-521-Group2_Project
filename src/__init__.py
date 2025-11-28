"""src package for multi-crop-disease-classifier.

Making `src` an explicit package so CLI scripts can import as `from src import ...`.
"""

__all__ = [
    "data",
    "model",
    "train",
    "inference",
    "eda",
    "visualize",
]
