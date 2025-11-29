import os
import random
import shutil
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
    species = ds['meta']['species']
    random.shuffle(samples)
    subset = samples[:25]
    imgs = []
    labels = []
    tf = get_transforms('plant')
    for p, l in subset:
        img = Image.open(p).convert('RGB')
        imgs.append(tf(img))
        labels.append((species[l], os.path.basename(p)))
    grid = make_grid(imgs, nrow=5, normalize=True)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(grid.permute(1,2,0).numpy())
    ax.axis('off')
    ax.set_title('Plant Sample Grid (train)', fontsize=16, pad=20)
    
    # Add labels below each image
    for i, (sp, fname) in enumerate(labels):
        row = i // 5
        col = i % 5
        x = (col + 0.5) / 5
        y = (row + 1) / 5 - 0.02
        ax.text(x, y, f'{sp}\n{fname[:15]}...', 
                transform=ax.transAxes, 
                fontsize=7, 
                ha='center', 
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_disease_grids(data_dir='data/diseases', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for sp in species_dirs:
        ds = prepare_disease_dataset(sp, data_dir)
        classes = ds['meta']['classes']
        samples = ds['train']
        
        # Get samples per class
        samples_per_class = {i: [] for i in range(len(classes))}
        for p, l in samples:
            samples_per_class[l].append(p)
        
        # Sample up to 4 images per class
        imgs = []
        labels = []
        tf = get_transforms('disease')
        
        for class_idx, class_name in enumerate(classes):
            class_samples = samples_per_class[class_idx]
            random.shuffle(class_samples)
            for p in class_samples[:4]:
                img = Image.open(p).convert('RGB')
                imgs.append(tf(img))
                labels.append((class_name, os.path.basename(p)))
        
        if len(imgs) == 0:
            continue
        
        # Create grid with 4 columns (4 samples per disease)
        grid = make_grid(imgs, nrow=4, normalize=True)
        fig, ax = plt.subplots(figsize=(12, 3 * len(classes)))
        ax.imshow(grid.permute(1,2,0).numpy())
        ax.axis('off')
        ax.set_title(f'{sp} Disease Sample Grid (train)', fontsize=16, pad=20)
        
        # Add labels below each image
        for i, (disease, fname) in enumerate(labels):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4
            y = (row + 1) / len(classes) - 0.01
            ax.text(x, y, f'{disease}\n{fname[:12]}...', 
                    transform=ax.transAxes, 
                    fontsize=7, 
                    ha='center', 
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{sp}_disease_samples.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved disease grid for {sp} to {out_path}')


def check_unreadable_images(data_dir='data', bad_images_dir='data/bad_images'):
    """
    Check all images in data/plants and data/diseases for readability.
    Move unreadable images to a separate folder and log them.
    
    Returns:
        dict: Statistics about checked and unreadable images
    """
    plants_dir = os.path.join(data_dir, 'plants')
    diseases_dir = os.path.join(data_dir, 'diseases')
    
    unreadable = []
    total_checked = 0
    
    # Check both plant and disease directories
    for root_dir in [plants_dir, diseases_dir]:
        if not os.path.exists(root_dir):
            print(f'Skipping {root_dir} (does not exist)')
            continue
            
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                # Skip non-image files
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    continue
                
                img_path = os.path.join(root, fname)
                total_checked += 1
                
                try:
                    # Try to open and verify the image
                    with Image.open(img_path) as img:
                        img.verify()  # Verify it's not corrupted
                    
                    # Re-open to test actual loading (verify() closes the file)
                    with Image.open(img_path) as img:
                        img.load()  # Force load pixel data
                        
                except Exception as e:
                    print(f'Unreadable image: {img_path} - Error: {str(e)}')
                    unreadable.append((img_path, str(e)))
    
    # Move unreadable images
    if unreadable:
        os.makedirs(bad_images_dir, exist_ok=True)
        log_path = os.path.join(bad_images_dir, 'unreadable_log.txt')
        
        with open(log_path, 'w') as log:
            log.write(f'Found {len(unreadable)} unreadable images out of {total_checked} checked\n\n')
            
            for img_path, error in unreadable:
                # Create destination preserving directory structure
                rel_path = os.path.relpath(img_path, data_dir)
                dest_path = os.path.join(bad_images_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Move the file
                shutil.move(img_path, dest_path)
                log.write(f'{img_path} -> {dest_path}\n')
                log.write(f'  Error: {error}\n\n')
        
        print(f'\nMoved {len(unreadable)} unreadable images to {bad_images_dir}')
        print(f'See log at {log_path}')
    else:
        print(f'\nAll {total_checked} images are readable!')
    
    return {
        'total_checked': total_checked,
        'unreadable': len(unreadable),
        'unreadable_list': unreadable
    }


if __name__ == '__main__':
    print('Run functions: plot_class_balance_plant, plot_class_balance_diseases, save_sanity_grid')
