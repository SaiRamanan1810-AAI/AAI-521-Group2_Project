"""Data preprocessing utilities.

This script creates `train/`, `val/`, and `test/` splits from a raw directory
that follows ImageFolder class structure: `raw/<class_name>/*.jpg`.

Usage example:

    python src/data.py --raw_dir data/raw --out_dir data/processed --img_size 224

"""
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image


def set_seed(seed: int):
    random.seed(seed)


def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_and_resize(src: Path, dst: Path, size: int):
    create_dir(dst.parent)
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((size, size))
        img.save(dst)
    except Exception as e:
        print(f"Skipping {src}: {e}")


def split_dataset(raw_dir: str, out_dir: str, img_size: int = 224, val_split: float = 0.1, test_split: float = 0.1, seed: int = 42):
    raw = Path(raw_dir)
    out = Path(out_dir)
    set_seed(seed)

    classes = [p.name for p in raw.iterdir() if p.is_dir()]
    if not classes:
        raise ValueError(f"No class subfolders found in {raw}")

    mapping = defaultdict(list)
    for cls in classes:
        for img_path in (raw / cls).glob("**/*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                mapping[cls].append(img_path)

    for cls, items in mapping.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        test_items = items[:n_test]
        val_items = items[n_test:n_test + n_val]
        train_items = items[n_test + n_val:]

        for subset, list_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
            for src_path in list_items:
                rel = src_path.name
                dst = out / subset / cls / rel
                copy_and_resize(src_path, dst, img_size)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True, help="Raw dataset directory (class subfolders)")
    p.add_argument("--out_dir", required=True, help="Output processed directory")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(args.raw_dir, args.out_dir, args.img_size, args.val_split, args.test_split, args.seed)
