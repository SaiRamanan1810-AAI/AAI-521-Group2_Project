"""Visualization helpers: training curves and confusion matrices."""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_training(history: dict, out_path: str):
    """Plot loss and accuracy curves.

    `history` expected keys: train_loss, val_loss, train_acc, val_acc
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("train_loss", []), label="train")
    plt.plot(epochs, history.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("train_acc", []), label="train")
    plt.plot(epochs, history.get("val_acc", []), label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true, y_pred, labels, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
