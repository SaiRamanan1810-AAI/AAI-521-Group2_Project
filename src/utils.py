"""Utility helpers: device, seed, checkpointing, metrics."""
import random
import os
import json
import torch
import numpy as np


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Apply MixUp augmentation to a batch.

    Returns: mixed_x, y_a, y_b, lam
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    if batch_size == 1:
        return x, y, y, 1.0
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def _rand_bbox(size, lam):
    # size: (B, C, H, W)
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation to a batch.

    Returns: mixed_x, y_a, y_b, lam
    """
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    if batch_size == 1:
        return x, y, y, 1.0
    index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
    mixed = x.clone()
    # x: (B, C, H, W) - note slicing [ :, :, y1:y2, x1:x2 ]
    mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exact area
    lam_adjusted = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(2) * x.size(3)))
    y_a, y_b = y, y[index]
    return mixed, y_a, y_b, lam_adjusted


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
