"""Dataset exploratory script.

Saves reports to `out_dir/reports/eda`:
 - `class_counts.csv` and `class_counts.png`
 - `image_sizes.csv` and `image_sizes_hist.png`
 - `sample_grid.png` (samples per class)
 - `bad_files.txt` and copies of bad files to `bad_files/`

Usage:
    conda activate plant-pest
    python src/eda.py --data_dir data/processed --out_dir . --num_samples 5

"""
from pathlib import Path
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np
import os

# allow opening truncated images for analysis (we'll still mark them bad if they fail)
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from src.utils import save_json
except Exception:
    def save_json(obj, path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)


def gather_image_files(root: Path):
    """Return dict mapping split->list of image paths and mapping split->class->paths."""
    splits = {}
    for split in ["train", "val", "test"]:
        p = root / split
        if not p.exists():
            splits[split] = []
            continue
        paths = [x for x in p.rglob("*") if x.is_file()]
        splits[split] = paths
    return splits


def class_distribution(root: Path, out_dir: Path):
    """Compute counts per class (for train/val/test) and save CSV + plot."""
    counts = {}
    per_split_classes = {}
    for split in ["train", "val", "test"]:
        p = root / split
        if not p.exists():
            counts[split] = {}
            per_split_classes[split] = []
            continue
        class_dirs = [d for d in p.iterdir() if d.is_dir()]
        c = {}
        for cd in class_dirs:
            num = sum(1 for _ in cd.rglob("*") if _.is_file())
            c[cd.name] = num
        counts[split] = c
        per_split_classes[split] = list(c.keys())

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "class_counts.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["split", "class", "count"]
        w.writerow(header)
        for split, cc in counts.items():
            for cls, cnt in sorted(cc.items(), key=lambda x: -x[1]):
                w.writerow([split, cls, cnt])

    # Plot train class distribution if available
    train_counts = counts.get("train", {})
    if train_counts:
        labels = list(train_counts.keys())
        vals = [train_counts[l] for l in labels]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.5), 4))
        ax.bar(labels, vals)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel("# images")
        ax.set_title("Train class distribution")
        plt.tight_layout()
        fig.savefig(out_dir / "class_counts_train.png")
        plt.close(fig)

    return counts


def image_size_distribution(root: Path, out_dir: Path, max_images=10000):
    """Scan images and report (width,height) distribution and a histogram of areas."""
    sizes = []
    scanned = 0
    bad_files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if scanned >= max_images:
            break
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            bad_files.append(str(p))
        scanned += 1

    # Save CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "image_sizes.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["width", "height"])
        for w_, h_ in sizes:
            w.writerow([w_, h_])

    # Histogram of areas
    areas = [w_ * h_ for w_, h_ in sizes]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(areas, bins=50)
    ax.set_title("Image area distribution (sampled)")
    ax.set_xlabel("pixels")
    ax.set_ylabel("count")
    plt.tight_layout()
    fig.savefig(out_dir / "image_sizes_hist.png")
    plt.close(fig)

    return sizes, bad_files


def sample_grid(root: Path, out_dir: Path, num_samples=5, max_cols=8):
    """Create a grid of sample images per class from `train` split."""
    train_dir = root / "train"
    if not train_dir.exists():
        return None
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda d: d.name)
    samples = []
    labels = []
    for cd in class_dirs:
        imgs = [p for p in cd.rglob("*") if p.is_file()]
        if not imgs:
            continue
        chosen = imgs[:num_samples]
        for p in chosen:
            try:
                with Image.open(p) as im:
                    im_rgb = im.convert("RGB")
                    samples.append(im_rgb.copy())
                    labels.append(cd.name)
            except Exception:
                # skip bad images
                continue

    if not samples:
        return None

    cols = min(max_cols, num_samples)
    rows = int(np.ceil(len(samples) / cols))
    fig_w = cols * 2
    fig_h = rows * 2
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.flatten()):
        if i < len(samples):
            ax.imshow(samples[i])
            ax.set_title(labels[i], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out_path = out_dir / "sample_grid.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def check_and_copy_bad_files(root: Path, out_dir: Path):
    """Try opening every image; copy unreadable ones to out_dir/bad_files and list them."""
    bad = []
    bad_dir = out_dir / "bad_files"
    bad_dir.mkdir(parents=True, exist_ok=True)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad.append(str(p))
            try:
                shutil.copy2(p, bad_dir / p.name)
            except Exception:
                pass
    if bad:
        with open(out_dir / "bad_files.txt", "w") as f:
            f.write("\n".join(bad))
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Processed data root containing train/val/test folders")
    p.add_argument("--out_dir", default=".", help="Directory to write reports (will create reports/eda)")
    p.add_argument("--num_samples", type=int, default=5, help="Samples per class for preview grid")
    p.add_argument("--max_images", type=int, default=5000, help="Max images to scan for size distribution")
    args = p.parse_args()

    data_root = Path(args.data_dir)
    out_reports = Path(args.out_dir) / "reports" / "eda"
    out_reports.mkdir(parents=True, exist_ok=True)

    print("Gathering class distribution...")
    counts = class_distribution(data_root, out_reports)
    save_json(counts, str(out_reports / "class_counts.json"))

    print("Scanning image sizes (may take a while)...")
    sizes, bad_files_sample = image_size_distribution(data_root, out_reports, max_images=args.max_images)
    save_json({"sampled_sizes_count": len(sizes)}, str(out_reports / "image_sizes_meta.json"))

    print("Creating sample grid...")
    grid_path = sample_grid(data_root, out_reports, num_samples=args.num_samples)
    if grid_path:
        print("Saved sample grid:", grid_path)

    print("Checking and copying bad files (full scan)...")
    bad_files = check_and_copy_bad_files(data_root, out_reports)
    print(f"Found {len(bad_files)} bad files (copied to {out_reports / 'bad_files'})")

    print("EDA complete. Reports saved to:", out_reports)


if __name__ == "__main__":
    main()
