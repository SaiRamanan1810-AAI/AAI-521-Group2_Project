"""Training script for baseline and transfer-learning models.

CLI usage examples:

    # baseline
    python src/train.py --mode baseline --data_dir data/processed --epochs 10

    # transfer
    python src/train.py --mode transfer --model_name resnet18 --data_dir data/processed --epochs 8

"""
import argparse
import os
# Workaround for macOS OpenMP duplicate-lib abort (exit code 134).
# Setting this allows the process to continue when multiple OpenMP runtimes
# are present (causes 'Initializing libomp.dylib, but found libomp.dylib already initialized').
# This is a pragmatic workaround for development; for production prefer
# ensuring a single OpenMP runtime (installing libomp via brew or using
# consistent conda-forge builds).
if os.environ.get("KMP_DUPLICATE_LIB_OK", "") != "TRUE":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Ensure repository root is on sys.path so `import src.*` works when running
# the script as `python src/train.py`.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils import set_seed, get_device, save_checkpoint, save_json
from src.models.baseline import SimpleCNN
from src.models.transfer import load_pretrained_model
from src.visualize import plot_training, plot_confusion


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(y.cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y.cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    prec, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    return avg_loss, acc, prec, recall, f1, all_targets, all_preds


def get_loaders(data_dir, img_size, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(data_dir) / "val", transform=val_tf)
    test_ds = datasets.ImageFolder(Path(data_dir) / "test", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, train_ds.classes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "transfer"], default="baseline")
    p.add_argument("--model_name", default="resnet18", help="Used when --mode transfer")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", default=".")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, test_loader, classes = get_loaders(args.data_dir, args.img_size, args.batch_size)
    num_classes = len(classes)

    if args.mode == "baseline":
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = load_pretrained_model(args.model_name, num_classes=num_classes, pretrained=True, freeze_backbone=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    out_models = Path(args.out_dir) / "models"
    out_reports = Path(args.out_dir) / "reports"
    out_models.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, prec, recall, f1, _, _ = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={time()-t0:.1f}s")

        # checkpoint
        ckpt_path = out_models / f"{args.mode}_{args.model_name}_epoch{epoch}.pth"
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "classes": classes
        }, str(ckpt_path))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": classes
            }, str(out_models / f"best_{args.mode}_{args.model_name}.pth"))

    # final evaluation on test set
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    metrics = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "f1": test_f1,
        "classes": classes
    }
    save_json(metrics, str(out_reports / f"metrics_{args.mode}_{args.model_name}.json"))

    plot_training(history, str(out_reports / f"training_{args.mode}_{args.model_name}.png"))
    plot_confusion(y_true, y_pred, classes, str(out_reports / f"cm_{args.mode}_{args.model_name}.png"))

    print("Done. Reports saved to:", out_reports)


if __name__ == "__main__":
    main()
