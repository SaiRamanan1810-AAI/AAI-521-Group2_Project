import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

from src.data import prepare_plant_dataset, prepare_disease_dataset, get_transforms, SimpleImageDataset
from torchvision.utils import make_grid
import torch


def plot_class_balance_plant(data_dir='data/plants', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    ds = prepare_plant_dataset(data_dir)
    species = ds['meta']['species']
    counts = [0]*len(species)
    for _, lab in ds['train']:
        counts[lab] += 1
    plt.figure(figsize=(8,5))
    sns.barplot(x=species, y=counts)
    plt.title('Plant class balance (train)')
    plt.ylabel('Count')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plant_class_balance.png'))


def plot_class_balance_diseases(data_dir='data/diseases', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for sp in species_dirs:
        ds = prepare_disease_dataset(sp, data_dir)
        classes = ds['meta']['classes']
        counts = [0]*len(classes)
        for _, lab in ds['train']:
            counts[lab] += 1
        plt.figure(figsize=(8,5))
        sns.barplot(x=classes, y=counts)
        plt.title(f'Disease class balance for {sp} (train)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{sp}_disease_class_balance.png'))


def save_sanity_grid(data_dir='data/plants', out_path='reports/data_samples.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = prepare_plant_dataset(data_dir)
    samples = ds['train']
    random.shuffle(samples)
    subset = samples[:25]
    imgs = []
    labels = []
    tf = get_transforms('plant')
    for p, l in subset:
        img = Image.open(p).convert('RGB')
        imgs.append(tf(img))
        labels.append((os.path.basename(p), l))
    grid = make_grid(imgs, nrow=5, normalize=True)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.axis('off')
    plt.title('Sample grid (train)')
    plt.savefig(out_path)


if __name__ == '__main__':
    print('Run functions: plot_class_balance_plant, plot_class_balance_diseases, save_sanity_grid')
