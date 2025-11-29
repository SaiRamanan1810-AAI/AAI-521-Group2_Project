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

from src.data import prepare_disease_dataset, SimpleImageDataset, get_transforms
from src.model import load_efficientnet_b0
from src.train import train_two_step


def build_dataloaders(species_name, data_dir=None, splits_dir=None, batch_size=32, num_workers=4):
    # If splits_dir provided, read CSVs from splits_dir/diseases
    if splits_dir:
        import pandas as _pd
        sp_dir = os.path.join(splits_dir, 'diseases')
        train_df = _pd.read_csv(os.path.join(sp_dir, f'{species_name}_train.csv'))
        val_df = _pd.read_csv(os.path.join(sp_dir, f'{species_name}_val.csv'))
        test_df = _pd.read_csv(os.path.join(sp_dir, f'{species_name}_test.csv'))
        train_pairs = list(zip(train_df['path'].tolist(), train_df['label'].tolist()))
        val_pairs = list(zip(val_df['path'].tolist(), val_df['label'].tolist()))
        test_pairs = list(zip(test_df['path'].tolist(), test_df['label'].tolist()))
        
        # Load actual class names from metadata JSON
        meta_path = os.path.join(sp_dir, f'{species_name}_meta.json')
        if os.path.exists(meta_path):
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        else:
            # Fallback: derive from data_dir if metadata not found
            meta = {'classes': []}
            if data_dir:
                base = os.path.join(data_dir, species_name)
                if os.path.isdir(base):
                    meta['classes'] = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
        
        ds = {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs,
            'meta': meta
        }
    else:
        ds = prepare_disease_dataset(species_name, data_dir)
    tf = get_transforms('disease')
    # create sampler to oversample minority disease classes
    from src.data import make_weighted_sampler
    train_sampler, class_weights = make_weighted_sampler(ds['train'])

    # optionally apply extra augmentation to minority classes
    labels = [lab for _, lab in ds['train']]
    import numpy as _np
    uniques, counts = _np.unique(labels, return_counts=True)
    median = _np.median(counts)
    transform_map = {}
    for cls, cnt in zip(uniques, counts):
        if cnt < median:
            # even heavier disease augment for rare labels
            transform_map[int(cls)] = tf

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
    parser.add_argument('--species', required=True, help='Species name folder under data/diseases')
    parser.add_argument('--data-dir', default='data/diseases')
    parser.add_argument('--splits-dir', default=None, help='Path to prepared splits (CSV files)')
    parser.add_argument('--out-dir', default='models')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs-head', type=int, default=5)
    parser.add_argument('--epochs-ft', type=int, default=15)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint-name', default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ck_name = args.checkpoint_name or f'{args.species}_checkpoint.pth'
    ck_path = os.path.join(args.out_dir, ck_name)

    dataloaders, meta, class_weights = build_dataloaders(args.species, data_dir=args.data_dir, splits_dir=args.splits_dir, batch_size=args.batch_size)
    # Fallback: if meta.classes missing, infer from training labels
    num_classes = len(meta.get('classes', []))
    if num_classes == 0:
        # infer from dataloader dataset labels
        labels = [lab for _, lab in dataloaders['train'].dataset.items]
        num_classes = (max(labels) + 1) if labels else 0

    device = torch.device(args.device)
    model = load_efficientnet_b0(num_classes=num_classes, pretrained=True)

    model, history = train_two_step(model, dataloaders, device,
                                    epochs_head=args.epochs_head, epochs_finetune=args.epochs_ft,
                                    checkpoint_path=ck_path, class_weights=class_weights, use_focal=False, mixup_alpha=0.2)

    meta_path = ck_path + '.meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f)

    print(f'Disease model for {args.species} saved to: {ck_path}')
    print(f'Metadata saved to: {meta_path}')


if __name__ == '__main__':
    main()
