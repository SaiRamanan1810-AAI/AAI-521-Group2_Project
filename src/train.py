import os
import json
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

from src.model import load_efficientnet_b0, set_parameter_requires_grad


def compute_class_weights(labels):
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def evaluate(model, dataloader, device, criterion=None):
    """Return (accuracy, loss). If criterion is None, loss will be None."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb, _ in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            if criterion is not None:
                loss = criterion(out, yb)
                running_loss += float(loss) * xb.size(0)
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = running_loss / total if (criterion is not None and total > 0) else None
    return accuracy, avg_loss


def train_two_step(model, dataloaders: Dict[str, DataLoader], device: torch.device,
                   epochs_head=5, epochs_finetune=15, lr_head=1e-3, lr_ft=1e-4,
                   checkpoint_path='models/checkpoint.pth', class_weights=None,
                   use_focal=False, focal_gamma=2.0, mixup_alpha=0.0, history_path=None):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Default history path based on checkpoint name if not provided
    if history_path is None:
        history_path = os.path.join(os.path.dirname(checkpoint_path), 'history.json')

    # Step A: freeze encoder, train head
    set_parameter_requires_grad(model, True)
    # unfreeze classifier parameters
    for p in model.classifier.parameters():
        p.requires_grad = True

    # choose criterion: weighted CE or focal loss
    if class_weights is not None:
        cw = class_weights.to(device)
    else:
        cw = None
    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, weight=cw)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_head)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_head)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience = 5
    wait = 0

    for epoch in range(epochs_head):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb, _ in tqdm(dataloaders['train'], desc=f'StepA Epoch {epoch+1}/{epochs_head}'):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            # optionally apply MixUp augmentation on the batch
            if mixup_alpha > 0.0 and np.random.rand() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[idx]
                yb_a, yb_b = yb, yb[idx]
                out = model(xb)
                loss = lam * criterion(out, yb_a) + (1 - lam) * criterion(out, yb_b)
                # For mixup, we can't compute exact accuracy, so skip it
            else:
                out = model(xb)
                loss = criterion(out, yb)
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        train_loss = running_loss / len(dataloaders['train'].dataset)
        train_acc = correct / total if total > 0 else 0.0

        # validation
        val_acc, val_loss = evaluate(model, dataloaders['val'], device, criterion=criterion)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[StepA] Epoch {epoch+1}/{epochs_head} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss if val_loss is None else f'{val_loss:.4f}'} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping during head training')
                break

    # Step B: unfreeze all and fine-tune
    set_parameter_requires_grad(model, False)  # clear previous freeze
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=lr_ft)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_finetune)
    wait = 0

    for epoch in range(epochs_finetune):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb, _ in tqdm(dataloaders['train'], desc=f'FineTune Epoch {epoch+1}/{epochs_finetune}'):
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            if mixup_alpha > 0.0 and np.random.rand() < 0.5:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[idx]
                yb_a, yb_b = yb, yb[idx]
                out = model(xb)
                loss = lam * criterion(out, yb_a) + (1 - lam) * criterion(out, yb_b)
                # For mixup, we can't compute exact accuracy, so skip it
            else:
                out = model(xb)
                loss = criterion(out, yb)
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()
        train_loss = running_loss / len(dataloaders['train'].dataset)
        train_acc = correct / total if total > 0 else 0.0

        val_acc, val_loss = evaluate(model, dataloaders['val'], device, criterion=criterion)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"[FineTune] Epoch {epoch+1}/{epochs_finetune} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss if val_loss is None else f'{val_loss:.4f}'} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping during fine-tune')
                break

    # Save history
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f'Training history saved to: {history_path}')

    # load best
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    return model, history


if __name__ == '__main__':
    print('train.py module. Use train_two_step from other scripts.')
