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
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate and return accuracy, precision, recall, and F1 scores."""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate metrics with macro averaging (treats all classes equally)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate metrics with weighted averaging (accounts for class imbalance)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
            'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
            'f1': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
        }
    
    return metrics


def print_metrics(metrics, model_name):
    """Print metrics in a formatted way."""
    print(f"\n{'='*70}")
    print(f"Evaluation Metrics for {model_name}")
    print(f"{'='*70}")
    print(f"\n{'Overall Metrics:':<30}")
    print(f"  {'Accuracy:':<28} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\n{'Macro-Averaged Metrics:':<30} (treats all classes equally)")
    print(f"  {'Precision:':<28} {metrics['precision_macro']:.4f}")
    print(f"  {'Recall:':<28} {metrics['recall_macro']:.4f}")
    print(f"  {'F1-Score:':<28} {metrics['f1_macro']:.4f}")
    print(f"\n{'Weighted-Averaged Metrics:':<30} (accounts for class imbalance)")
    print(f"  {'Precision:':<28} {metrics['precision_weighted']:.4f}")
    print(f"  {'Recall:':<28} {metrics['recall_weighted']:.4f}")
    print(f"  {'F1-Score:':<28} {metrics['f1_weighted']:.4f}")
    print(f"\n{'Per-Class Metrics:':<30}")
    print(f"  {'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"  {'-'*56}")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name:<20} {class_metrics['precision']:<12.4f} "
              f"{class_metrics['recall']:<12.4f} {class_metrics['f1']:<12.4f}")
    print(f"{'='*70}\n")


def save_metrics(metrics, output_path):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")


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

    # Calculate metrics
    metrics = calculate_metrics(ys, yps, species)
    print_metrics(metrics, "Plant Species Classifier (Stage 1)")
    
    # Save metrics to JSON
    metrics_path = os.path.join(out_dir, 'stage1_metrics.json')
    save_metrics(metrics, metrics_path)

    # Generate visualizations
    cm_path = os.path.join(out_dir, 'stage1_confusion_matrix.png')
    plot_confusion_matrix(ys, yps, species, cm_path)
    ch_path = os.path.join(out_dir, 'stage1_confidence_hist.png')
    plot_confidence_histogram(confs, ch_path)
    print(f'Stage-1 evaluation complete. Results saved to {out_dir}\n')


def evaluate_stage2(models_dir='models', data_dir='data/diseases', out_dir='reports', device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    # list species dirs
    species_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    all_metrics = {}
    
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
        
        # Calculate metrics
        metrics = calculate_metrics(ys, yps, classes)
        print_metrics(metrics, f"{sp} Disease Classifier (Stage 2)")
        all_metrics[sp] = metrics
        
        # Save metrics to JSON
        metrics_path = os.path.join(out_dir, f'stage2_{sp}_metrics.json')
        save_metrics(metrics, metrics_path)
        
        # Generate visualizations
        cm_path = os.path.join(out_dir, f'stage2_{sp}_confmat.png')
        plot_confusion_matrix(ys, yps, classes, cm_path)
        ch_path = os.path.join(out_dir, f'stage2_{sp}_conf_hist.png')
        plot_confidence_histogram(confs, ch_path)
        print(f'{sp} evaluation complete. Results saved to {out_dir}\n')
    
    # Save combined metrics for all species
    if all_metrics:
        combined_path = os.path.join(out_dir, 'stage2_all_species_metrics.json')
        with open(combined_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Combined Stage-2 metrics saved to: {combined_path}")


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
