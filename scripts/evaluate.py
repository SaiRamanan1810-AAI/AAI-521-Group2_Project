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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import prepare_plant_dataset, prepare_disease_dataset, SimpleImageDataset, get_transforms
from src.model import load_efficientnet_b0
from src.visualize import plot_confusion_matrix, plot_confidence_histogram


def eval_model_on_loader(model, loader, device):
    ys = []
    yps = []
    confs = []
    model.to(device).eval()
    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            ys.extend(yb.numpy())
            yps.extend(preds.tolist())
            confs.extend(probs.max(axis=1).tolist())
    return ys, yps, confs


def build_loader_from_samples(samples, transform, batch_size=32, num_workers=4):
    ds = SimpleImageDataset(samples, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def evaluate_stage1(plant_ck, data_dir, out_dir='reports', device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    with open(plant_ck + '.meta.json', 'r') as f:
        meta = json.load(f)
    species = meta.get('species', [])

    ds = prepare_plant_dataset(data_dir)
    tf = get_transforms('plant')
    test_loader = build_loader_from_samples(ds['test'], tf)

    model = load_efficientnet_b0(num_classes=4, pretrained=False)
    ck = torch.load(plant_ck, map_location=device)
    model.load_state_dict(ck['model_state_dict'])

    ys, yps, confs = eval_model_on_loader(model, test_loader, device)

    cm_path = os.path.join(out_dir, 'stage1_confusion_matrix.png')
    plot_confusion_matrix(ys, yps, species, cm_path)
    ch_path = os.path.join(out_dir, 'stage1_confidence_hist.png')
    plot_confidence_histogram(confs, ch_path)
    print(f'Stage-1 evaluation saved to {out_dir}')


def evaluate_stage2(models_dir='models', data_dir='data/diseases', out_dir='reports', device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    # list species dirs
    species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for sp in species_dirs:
        ck_path = os.path.join(models_dir, f'{sp}_checkpoint.pth')
        meta_path = ck_path + '.meta.json'
        if not os.path.exists(ck_path) or not os.path.exists(meta_path):
            print(f'Skipping {sp}: checkpoint or metadata missing')
            continue
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        classes = meta.get('classes', [])

        ds = prepare_disease_dataset(sp, data_dir)
        tf = get_transforms('disease')
        test_loader = build_loader_from_samples(ds['test'], tf)

        model = load_efficientnet_b0(num_classes=len(classes), pretrained=False)
        ck = torch.load(ck_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])

        ys, yps, confs = eval_model_on_loader(model, test_loader, device)
        cm_path = os.path.join(out_dir, f'stage2_{sp}_confmat.png')
        plot_confusion_matrix(ys, yps, classes, cm_path)
        ch_path = os.path.join(out_dir, f'stage2_{sp}_conf_hist.png')
        plot_confidence_histogram(confs, ch_path)
        print(f'Evaluated {sp}: saved confmat and confidence hist to {out_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plant-checkpoint', default='models/plant_checkpoint.pth')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--plants-data', default='data/plants')
    parser.add_argument('--diseases-data', default='data/diseases')
    parser.add_argument('--out-dir', default='reports')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if os.path.exists(args.plant_checkpoint) and os.path.exists(args.plant_checkpoint + '.meta.json'):
        evaluate_stage1(args.plant_checkpoint, args.plants_data, args.out_dir, device=args.device)
    else:
        print('Plant checkpoint or metadata not found — skipping Stage-1 evaluation')

    evaluate_stage2(models_dir=args.models_dir, data_dir=args.diseases_data, out_dir=args.out_dir, device=args.device)


if __name__ == '__main__':
    main()
