import os
import random
from typing import List, Tuple, Dict, Optional

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torchvision import transforms


def list_images(directory: str, exts=('.jpg', '.jpeg', '.png')):
    paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return paths


class SimpleImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, transform_map: Optional[Dict[int, transforms.Compose]] = None):
        """samples: list of (path, label)
        transform: default transform applied when label not in transform_map
        transform_map: optional dict mapping label -> transform to apply for that class
        """
        self.samples = samples
        self.transform = transform
        self.transform_map = transform_map or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Try to load image; if broken, try another random sample with same label
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img = Image.open(path).convert('RGB')
                t = self.transform_map.get(label, self.transform)
                if t:
                    img = t(img)
                return img, label, path
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    # Find another sample with the same label and retry
                    same_label_indices = [i for i, (_, l) in enumerate(self.samples) if l == label]
                    if same_label_indices:
                        idx = random.choice(same_label_indices)
                        path, label = self.samples[idx]
                    else:
                        raise  # no fallback available
                else:
                    # Last attempt: return a black image as fallback
                    print(f'Warning: Could not load image at {path} after {max_retries} attempts. Using black image.')
                    img = Image.new('RGB', (224, 224), (0, 0, 0))
                    t = self.transform_map.get(label, self.transform)
                    if t:
                        img = t(img)
                    return img, label, path


def make_weighted_sampler(samples: List[Tuple[str, int]]):
    """Create a torch.utils.data.WeightedRandomSampler to oversample minority classes.
    Returns sampler and class_weights (inverse frequency normalized) as torch tensor.
    """
    import torch
    labels = [lab for _, lab in samples]
    classes, counts = np.unique(labels, return_counts=True)
    # weight for class = 1 / count
    class_weights = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    sample_weights = [class_weights[l] for l in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # return tensor of class weights for loss balancing
    cw = np.array([class_weights.get(c, 0.0) for c in range(int(classes.max()) + 1)])
    cw = cw / cw.sum() * len(cw)
    return sampler, torch.tensor(cw, dtype=torch.float)


def make_stratified_split(paths: List[str], labels: List[int], test_size: float, seed: int = 42):
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return (train_paths, train_labels), (test_paths, test_labels)


def prepare_plant_dataset(data_dir: str = 'data/plants', seed: int = 42):
    # Expect data/plants/{SpeciesName}/*images*
    species = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    samples = []
    labels = []
    for idx, sp in enumerate(species):
        folder = os.path.join(data_dir, sp)
        imgs = list_images(folder)
        for p in imgs:
            samples.append(p)
            labels.append(idx)

    (train_p, train_l), (temp_p, temp_l) = make_stratified_split(samples, labels, test_size=0.3, seed=seed)
    (val_p, val_l), (test_p, test_l) = make_stratified_split(temp_p, temp_l, test_size=0.5, seed=seed)

    meta = {'species': species}

    return {
        'train': list(zip(train_p, train_l)),
        'val': list(zip(val_p, val_l)),
        'test': list(zip(test_p, test_l)),
        'meta': meta,
    }


def prepare_disease_dataset(species_name: str, data_dir: str = 'data/diseases', seed: int = 42):
    # Expect data/diseases/{SpeciesName}/{DiseaseClass}/*images*
    base = os.path.join(data_dir, species_name)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Disease data for {species_name} not found at {base}")
    classes = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    samples = []
    labels = []
    for idx, cl in enumerate(classes):
        folder = os.path.join(base, cl)
        imgs = list_images(folder)
        for p in imgs:
            samples.append(p)
            labels.append(idx)

    if len(samples) == 0:
        raise ValueError(f"No images found for species {species_name} in {base}")

    (train_p, train_l), (temp_p, temp_l) = make_stratified_split(samples, labels, test_size=0.3, seed=seed)
    (val_p, val_l), (test_p, test_l) = make_stratified_split(temp_p, temp_l, test_size=0.5, seed=seed)

    return {
        'train': list(zip(train_p, train_l)),
        'val': list(zip(val_p, val_l)),
        'test': list(zip(test_p, test_l)),
        'meta': {'classes': classes},
    }


def get_transforms(task: str = 'plant'):
    if task == 'plant':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.2, 0.1)], p=0.5),
            transforms.RandomRotation(25),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    print('data.py module. Use functions from code.')
