import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

def plot_training_curves(history_path: str, out_dir: str = 'reports'):
    os.makedirs(out_dir, exist_ok=True)
    with open(history_path, 'r') as f:
        h = json.load(f)
    epochs = list(range(1, len(h.get('train_loss', [])) + 1))
    plt.figure()
    plt.plot(epochs, h.get('train_loss', []), label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.savefig(os.path.join(out_dir, 'loss_vs_epoch.png'))

    plt.figure()
    plt.plot(epochs, h.get('val_f1', []), label='val_f1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 (macro)')
    plt.legend()
    plt.title('F1 vs Epoch')
    plt.savefig(os.path.join(out_dir, 'f1_vs_epoch.png'))


def plot_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(out_path)


def plot_confidence_histogram(confidences, out_path, bins=30):
    plt.figure()
    plt.hist(confidences, bins=bins)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence distribution')
    plt.tight_layout()
    plt.savefig(out_path)


if __name__ == '__main__':
    print('visualize module: use plot_training_curves and plotting helpers')
