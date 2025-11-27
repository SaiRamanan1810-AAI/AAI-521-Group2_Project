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
from PIL import Image, ImageFile
import shutil


def set_seed(seed: int):
    random.seed(seed)


def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_and_resize(src: Path, dst: Path, size: int, bad_dir: Path = None) -> bool:
    """Copy an image from `src` to `dst`, resizing so shortest side = 256, then center-crop to (size, size).

    For non-square images:
    1. Resize so shortest side = 256 (maintain aspect ratio)
    2. Center-crop to (size, size)

    Returns True on success, False on failure. On failure, if `bad_dir` is
    provided the original file is copied there for inspection.
    """
    create_dir(dst.parent)
    try:
        img = Image.open(src).convert("RGB")
        
        # Resize so shortest side = 256
        w, h = img.size
        if w < h:
            new_w = 256
            new_h = int(256 * h / w)
        else:
            new_h = 256
            new_w = int(256 * w / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Center-crop to (size, size)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        right = left + size
        bottom = top + size
        img = img.crop((left, top, right, bottom))
        
        img.save(dst)
        return True
    except Exception as e:
        # Try a permissive re-open in case of truncated JPEGs
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(src) as img:
                img = img.convert("RGB")
                
                # Resize so shortest side = 256
                w, h = img.size
                if w < h:
                    new_w = 256
                    new_h = int(256 * h / w)
                else:
                    new_h = 256
                    new_w = int(256 * w / h)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                
                # Center-crop to (size, size)
                left = (new_w - size) // 2
                top = (new_h - size) // 2
                right = left + size
                bottom = top + size
                img = img.crop((left, top, right, bottom))
                
                img.save(dst)
                return True
        except Exception:
            print(f"Skipping {src}: {e}")
            if bad_dir is not None:
                try:
                    bad_dst = Path(bad_dir) / src.parent.name / src.name
                    create_dir(bad_dst.parent)
                    shutil.copy2(src, bad_dst)
                except Exception as e2:
                    print(f"Failed to copy bad file {src} to {bad_dir}: {e2}")
            return False


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

    bad_files = []
    bad_dir_path = Path(out) / "bad_files"

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
                ok = copy_and_resize(src_path, dst, img_size, bad_dir=bad_dir_path)
                if not ok:
                    bad_files.append(str(src_path))

    # Save list of bad files for inspection
    if bad_files:
        create_dir(out)
        bad_list_file = out / "bad_files.txt"
        with open(bad_list_file, "w") as f:
            for p in bad_files:
                f.write(p + "\n")
        print(f"Found {len(bad_files)} unreadable files. See {bad_list_file} and {bad_dir_path}")


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
