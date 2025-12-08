#!/usr/bin/env python3
"""
Generate training plots from history files saved during training.
"""
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt

# ensure project root is on path so `src` imports work when running scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_training_curves(history_path, title, out_path):
    """
    Plot training and validation loss/accuracy curves from a history JSON file.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved training plot: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate training plots from history files')
    parser.add_argument('--models-dir', default='models', help='Directory containing checkpoints and history files')
    parser.add_argument('--out-dir', default='reports/training_plots', help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check for plant classifier history
    plant_history = os.path.join(args.models_dir, 'plant_history.json')
    if os.path.exists(plant_history):
        plot_training_curves(
            plant_history,
            'Plant Species Classifier (Stage 1)',
            os.path.join(args.out_dir, 'plant_training.png')
        )
    else:
        print(f'Warning: Plant history not found at {plant_history}')
    
    # Check for disease classifier histories
    species = ['Cashew', 'Cassava', 'Maize', 'Tomato']
    for sp in species:
        disease_history = os.path.join(args.models_dir, f'{sp}_history.json')
        if os.path.exists(disease_history):
            plot_training_curves(
                disease_history,
                f'{sp} Disease Classifier (Stage 2)',
                os.path.join(args.out_dir, f'{sp}_disease_training.png')
            )
        else:
            print(f'Warning: {sp} disease history not found at {disease_history}')
    
    print(f'\nAll training plots saved to {args.out_dir}')


if __name__ == '__main__':
    main()
