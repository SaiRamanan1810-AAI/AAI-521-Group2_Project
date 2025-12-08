#!/usr/bin/env python3
import os
import sys
import argparse
import json

# Fix macOS OpenMP conflict (conda environment issue)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ensure project root is on path so `src` imports work when running scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from src.data import prepare_plant_dataset, SimpleImageDataset, get_transforms
from src.model import load_efficientnet_b0
from src.train import train_two_step


def build_dataloaders(data_dir=None, splits_dir=None, batch_size=32, num_workers=4):
    # If splits_dir provided, load CSVs instead of scanning folders
    if splits_dir:
        import pandas as _pd
        train_df = _pd.read_csv(os.path.join(splits_dir, 'plants_train.csv'))
        val_df = _pd.read_csv(os.path.join(splits_dir, 'plants_val.csv'))
        test_df = _pd.read_csv(os.path.join(splits_dir, 'plants_test.csv'))
        
        # Extract species names from data_dir (default to data/plants if not provided)
        data_dir = data_dir or 'data/plants'
        species = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        ds = {
            'train': list(zip(train_df['path'].tolist(), train_df['label'].tolist())),
            'val': list(zip(val_df['path'].tolist(), val_df['label'].tolist())),
            'test': list(zip(test_df['path'].tolist(), test_df['label'].tolist())),
            'meta': {'species': species}
        }
    else:
        ds = prepare_plant_dataset(data_dir)
    tf = get_transforms('plant')
    # compute a sampler to oversample minority classes
    from src.data import make_weighted_sampler
    train_sampler, class_weights = make_weighted_sampler(ds['train'])

    # apply heavier augmentations for minority classes
    # detect minority classes as those with count < median
    labels = [lab for _, lab in ds['train']]
    import numpy as _np
    uniques, counts = _np.unique(labels, return_counts=True)
    median = _np.median(counts)
    transform_map = {}
    for cls, cnt in zip(uniques, counts):
        if cnt < median:
            # heavier augment for minority class
            transform_map[int(cls)] = get_transforms('disease')

    train_ds = SimpleImageDataset(ds['train'], transform=tf, transform_map=transform_map)
    val_ds = SimpleImageDataset(ds['val'], transform=tf)
    test_ds = SimpleImageDataset(ds['test'], transform=tf)
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return dataloaders, ds.get('meta', {}), class_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/plants')
    parser.add_argument('--splits-dir', default=None, help='Path to prepared splits (CSV files)')
    parser.add_argument('--out-dir', default='models')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs-head', type=int, default=5)
    parser.add_argument('--epochs-ft', type=int, default=15)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint-name', default='plant_checkpoint.pth')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ck_path = os.path.join(args.out_dir, args.checkpoint_name)

    dataloaders, meta, class_weights = build_dataloaders(args.data_dir, splits_dir=args.splits_dir, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = load_efficientnet_b0(num_classes=4, pretrained=True)
    
    # Set history path
    history_path = os.path.join(args.out_dir, 'plant_history.json')

    model, history = train_two_step(model, dataloaders, device,
                                    epochs_head=args.epochs_head, epochs_finetune=args.epochs_ft,
                                    checkpoint_path=ck_path, class_weights=class_weights, 
                                    use_focal=False, mixup_alpha=0.2, history_path=history_path)

    # Save metadata alongside checkpoint
    meta_path = ck_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    print(f'Plant model saved to: {ck_path}')
    print(f'Metadata saved to: {meta_path}')
    
    # Generate training plot
    print('\nGenerating training plot...')
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Plant Species Classifier - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Plant Species Classifier - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, 'plant_training.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training plot saved to: {plot_path}')


if __name__ == '__main__':
    main()
