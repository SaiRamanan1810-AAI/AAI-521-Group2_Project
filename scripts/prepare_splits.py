#!/usr/bin/env python3
"""Prepare stratified train/val/test splits for plants and diseases.

Saves CSV files and metadata JSON to `out_dir`:
- plants_train.csv, plants_val.csv, plants_test.csv
- plants_meta.json
- diseases/{Species}_train.csv, ... and {Species}_meta.json

This CLI calls the helpers in `src.data`.
"""
import os
import sys
import argparse
import json
from pathlib import Path

# ensure project root is on path so `src` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data import prepare_plant_dataset, prepare_disease_dataset


def save_split(samples, meta_map, out_csv_path):
    # samples: list of (path, label)
    df = pd.DataFrame(samples, columns=['path', 'label'])
    # attach label name if available
    if meta_map is not None:
        classes = meta_map.get('species') or meta_map.get('classes')
        if classes:
            df['label_name'] = df['label'].apply(lambda x: classes[int(x)])
    df.to_csv(out_csv_path, index=False)


def prepare_plants(data_dir, out_dir, seed=42):
    ds = prepare_plant_dataset(data_dir, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    save_split(ds['train'], ds['meta'], os.path.join(out_dir, 'plants_train.csv'))
    save_split(ds['val'], ds['meta'], os.path.join(out_dir, 'plants_val.csv'))
    save_split(ds['test'], ds['meta'], os.path.join(out_dir, 'plants_test.csv'))
    with open(os.path.join(out_dir, 'plants_meta.json'), 'w') as f:
        json.dump(ds['meta'], f, indent=2)
    print('Saved plant splits and meta to', out_dir)


def prepare_diseases(data_dir, out_dir, seed=42):
    species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    os.makedirs(out_dir, exist_ok=True)
    for sp in species_dirs:
        try:
            ds = prepare_disease_dataset(sp, data_dir, seed=seed)
        except Exception as e:
            print(f'Skipping {sp}: {e}')
            continue
        sp_out = os.path.join(out_dir, 'diseases')
        os.makedirs(sp_out, exist_ok=True)
        save_split(ds['train'], ds['meta'], os.path.join(sp_out, f'{sp}_train.csv'))
        save_split(ds['val'], ds['meta'], os.path.join(sp_out, f'{sp}_val.csv'))
        save_split(ds['test'], ds['meta'], os.path.join(sp_out, f'{sp}_test.csv'))
        with open(os.path.join(sp_out, f'{sp}_meta.json'), 'w') as f:
            json.dump(ds['meta'], f, indent=2)
        print(f'Saved disease splits for {sp} to', sp_out)


def main():
    parser = argparse.ArgumentParser(description='Prepare stratified splits for plants and diseases')
    parser.add_argument('--plants-data', default='data/plants', help='Path to plants folder')
    parser.add_argument('--diseases-data', default='data/diseases', help='Path to diseases folder')
    parser.add_argument('--out-dir', default='data_splits', help='Output directory for CSVs and metadata')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--only', choices=['all', 'plants', 'diseases'], default='all')
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.only in ('all', 'plants'):
        prepare_plants(args.plants_data, str(out), seed=args.seed)
    if args.only in ('all', 'diseases'):
        prepare_diseases(args.diseases_data, str(out), seed=args.seed)

    print('Done preparing splits. Files are in', str(out))


if __name__ == '__main__':
    main()
