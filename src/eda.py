import os
import random
import shutil
import hashlib
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime

from src.data import prepare_plant_dataset, prepare_disease_dataset, get_transforms, SimpleImageDataset
from torchvision.utils import make_grid
import torch


def plot_class_balance_plant(data_dir='data/plants', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    ds = prepare_plant_dataset(data_dir)
    species = ds['meta']['species']
    counts = [0]*len(species)
    for _, lab in ds['train']:
        counts[lab] += 1
    plt.figure(figsize=(8,5))
    sns.barplot(x=species, y=counts)
    plt.title('Plant class balance (train)')
    plt.ylabel('Count')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plant_class_balance.png'))


def plot_class_balance_diseases(data_dir='data/diseases', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for sp in species_dirs:
        ds = prepare_disease_dataset(sp, data_dir)
        classes = ds['meta']['classes']
        counts = [0]*len(classes)
        for _, lab in ds['train']:
            counts[lab] += 1
        plt.figure(figsize=(8,5))
        sns.barplot(x=classes, y=counts)
        plt.title(f'Disease class balance for {sp} (train)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{sp}_disease_class_balance.png'))


def save_sanity_grid(data_dir='data/plants', out_path='reports/data_samples.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = prepare_plant_dataset(data_dir)
    samples = ds['train']
    species = ds['meta']['species']
    random.shuffle(samples)
    subset = samples[:25]
    imgs = []
    labels = []
    tf = get_transforms('plant')
    for p, l in subset:
        img = Image.open(p).convert('RGB')
        imgs.append(tf(img))
        labels.append((species[l], os.path.basename(p)))
    grid = make_grid(imgs, nrow=5, normalize=True)
    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(grid.permute(1,2,0).numpy())
    ax.axis('off')
    ax.set_title('Plant Sample Grid (train)', fontsize=16, pad=20)
    
    # Add labels below each image
    for i, (sp, fname) in enumerate(labels):
        row = i // 5
        col = i % 5
        x = (col + 0.5) / 5
        y = (row + 1) / 5 - 0.02
        ax.text(x, y, f'{sp}\n{fname[:15]}...', 
                transform=ax.transAxes, 
                fontsize=7, 
                ha='center', 
                va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_disease_grids(data_dir='data/diseases', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    species_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for sp in species_dirs:
        ds = prepare_disease_dataset(sp, data_dir)
        classes = ds['meta']['classes']
        samples = ds['train']
        
        # Get samples per class
        samples_per_class = {i: [] for i in range(len(classes))}
        for p, l in samples:
            samples_per_class[l].append(p)
        
        # Sample up to 4 images per class
        imgs = []
        labels = []
        tf = get_transforms('disease')
        
        for class_idx, class_name in enumerate(classes):
            class_samples = samples_per_class[class_idx]
            random.shuffle(class_samples)
            for p in class_samples[:4]:
                img = Image.open(p).convert('RGB')
                imgs.append(tf(img))
                labels.append((class_name, os.path.basename(p)))
        
        if len(imgs) == 0:
            continue
        
        # Create grid with 4 columns (4 samples per disease)
        grid = make_grid(imgs, nrow=4, normalize=True)
        fig, ax = plt.subplots(figsize=(12, 3 * len(classes)))
        ax.imshow(grid.permute(1,2,0).numpy())
        ax.axis('off')
        ax.set_title(f'{sp} Disease Sample Grid (train)', fontsize=16, pad=20)
        
        # Add labels below each image
        for i, (disease, fname) in enumerate(labels):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4
            y = (row + 1) / len(classes) - 0.01
            ax.text(x, y, f'{disease}\n{fname[:12]}...', 
                    transform=ax.transAxes, 
                    fontsize=7, 
                    ha='center', 
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{sp}_disease_samples.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved disease grid for {sp} to {out_path}')


def check_unreadable_images(data_dir='data', bad_images_dir='data/bad_images'):
    """
    Check all images in data/plants and data/diseases for readability.
    Move unreadable images to a separate folder and log them.
    
    Returns:
        dict: Statistics about checked and unreadable images
    """
    plants_dir = os.path.join(data_dir, 'plants')
    diseases_dir = os.path.join(data_dir, 'diseases')
    
    unreadable = []
    total_checked = 0
    
    # Check both plant and disease directories
    for root_dir in [plants_dir, diseases_dir]:
        if not os.path.exists(root_dir):
            print(f'Skipping {root_dir} (does not exist)')
            continue
            
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                # Skip non-image files
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    continue
                
                img_path = os.path.join(root, fname)
                total_checked += 1
                
                try:
                    # Try to open and verify the image
                    with Image.open(img_path) as img:
                        img.verify()  # Verify it's not corrupted
                    
                    # Re-open to test actual loading (verify() closes the file)
                    with Image.open(img_path) as img:
                        img.load()  # Force load pixel data
                        
                except Exception as e:
                    print(f'Unreadable image: {img_path} - Error: {str(e)}')
                    unreadable.append((img_path, str(e)))
    
    # Move unreadable images
    if unreadable:
        os.makedirs(bad_images_dir, exist_ok=True)
        log_path = os.path.join(bad_images_dir, 'unreadable_log.txt')
        
        with open(log_path, 'w') as log:
            log.write(f'Found {len(unreadable)} unreadable images out of {total_checked} checked\n\n')
            
            for img_path, error in unreadable:
                # Create destination preserving directory structure
                rel_path = os.path.relpath(img_path, data_dir)
                dest_path = os.path.join(bad_images_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Move the file
                shutil.move(img_path, dest_path)
                log.write(f'{img_path} -> {dest_path}\n')
                log.write(f'  Error: {error}\n\n')
        
        print(f'\nMoved {len(unreadable)} unreadable images to {bad_images_dir}')
        print(f'See log at {log_path}')
    else:
        print(f'\nAll {total_checked} images are readable!')
    
    return {
        'total_checked': total_checked,
        'unreadable': len(unreadable),
        'unreadable_list': unreadable
    }


def get_image_hash(img_path):
    """Compute MD5 hash of image file for duplicate detection."""
    hasher = hashlib.md5()
    with open(img_path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def detect_duplicates(data_dir='data', dup_dir='data/dup_images', remove_duplicates=False):
    """
    Detect duplicate images based on file hash.
    Optionally remove duplicates, keeping the first occurrence.
    
    Args:
        data_dir: Root data directory
        dup_dir: Directory to move duplicate images
        remove_duplicates: If True, move duplicate images to dup_dir
    
    Returns:
        dict: Duplicate groups and statistics
    """
    plants_dir = os.path.join(data_dir, 'plants')
    diseases_dir = os.path.join(data_dir, 'diseases')
    
    hash_map = defaultdict(list)
    total_images = 0
    
    for root_dir in [plants_dir, diseases_dir]:
        if not os.path.exists(root_dir):
            continue
            
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    continue
                
                img_path = os.path.join(root, fname)
                total_images += 1
                
                try:
                    img_hash = get_image_hash(img_path)
                    hash_map[img_hash].append(img_path)
                except Exception as e:
                    print(f'Error hashing {img_path}: {e}')
    
    # Find duplicates (hashes with more than one file)
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    
    duplicate_count = sum(len(paths) - 1 for paths in duplicates.values())
    
    # Move duplicates if requested
    moved_count = 0
    if remove_duplicates and duplicates:
        os.makedirs(dup_dir, exist_ok=True)
        log_path = os.path.join(dup_dir, 'duplicates_log.txt')
        
        with open(log_path, 'w') as log:
            log.write(f'Duplicate Images Removal Log\n')
            log.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            log.write(f'Found {len(duplicates)} groups of duplicates\n\n')
            
            for hash_val, paths in duplicates.items():
                # Keep the first file, move the rest
                keeper = paths[0]
                dups = paths[1:]
                
                log.write(f'\nHash: {hash_val}\n')
                log.write(f'  Keeper: {keeper}\n')
                log.write(f'  Duplicates:\n')
                
                for dup_path in dups:
                    # Create destination preserving directory structure
                    rel_path = os.path.relpath(dup_path, data_dir)
                    dest_path = os.path.join(dup_dir, rel_path)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Move the duplicate
                    shutil.move(dup_path, dest_path)
                    log.write(f'    {dup_path} -> {dest_path}\n')
                    moved_count += 1
        
        print(f'\nMoved {moved_count} duplicate images to {dup_dir}')
        print(f'See log at {log_path}')
    
    return {
        'total_images': total_images,
        'duplicate_groups': len(duplicates),
        'duplicate_images': duplicate_count,
        'duplicates': duplicates,
        'moved_count': moved_count
    }


def analyze_image_dimensions(data_dir='data'):
    """
    Analyze image dimensions across the dataset.
    
    Returns:
        dict: Dimension statistics
    """
    plants_dir = os.path.join(data_dir, 'plants')
    diseases_dir = os.path.join(data_dir, 'diseases')
    
    dimensions = []
    aspect_ratios = []
    total_images = 0
    
    for root_dir in [plants_dir, diseases_dir]:
        if not os.path.exists(root_dir):
            continue
            
        for root, dirs, files in os.walk(root_dir):
            for fname in files:
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    continue
                
                img_path = os.path.join(root, fname)
                total_images += 1
                
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        dimensions.append((w, h))
                        aspect_ratios.append(w / h if h > 0 else 0)
                except Exception as e:
                    print(f'Error reading dimensions for {img_path}: {e}')
    
    dimensions = np.array(dimensions)
    aspect_ratios = np.array(aspect_ratios)
    
    return {
        'total_images': total_images,
        'dimensions': dimensions,
        'aspect_ratios': aspect_ratios,
        'width_stats': {
            'min': int(dimensions[:, 0].min()) if len(dimensions) > 0 else 0,
            'max': int(dimensions[:, 0].max()) if len(dimensions) > 0 else 0,
            'mean': float(dimensions[:, 0].mean()) if len(dimensions) > 0 else 0,
            'std': float(dimensions[:, 0].std()) if len(dimensions) > 0 else 0
        },
        'height_stats': {
            'min': int(dimensions[:, 1].min()) if len(dimensions) > 0 else 0,
            'max': int(dimensions[:, 1].max()) if len(dimensions) > 0 else 0,
            'mean': float(dimensions[:, 1].mean()) if len(dimensions) > 0 else 0,
            'std': float(dimensions[:, 1].std()) if len(dimensions) > 0 else 0
        },
        'aspect_ratio_stats': {
            'min': float(aspect_ratios.min()) if len(aspect_ratios) > 0 else 0,
            'max': float(aspect_ratios.max()) if len(aspect_ratios) > 0 else 0,
            'mean': float(aspect_ratios.mean()) if len(aspect_ratios) > 0 else 0,
            'std': float(aspect_ratios.std()) if len(aspect_ratios) > 0 else 0
        }
    }


def get_class_counts(data_dir='data'):
    """
    Get class counts for both plants and diseases.
    
    Returns:
        dict: Class count summaries
    """
    results = {
        'plants': {},
        'diseases': {}
    }
    
    # Plant counts
    plants_dir = os.path.join(data_dir, 'plants')
    if os.path.exists(plants_dir):
        for species in os.listdir(plants_dir):
            species_path = os.path.join(plants_dir, species)
            if os.path.isdir(species_path):
                count = sum(1 for f in os.listdir(species_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')))
                results['plants'][species] = count
    
    # Disease counts
    diseases_dir = os.path.join(data_dir, 'diseases')
    if os.path.exists(diseases_dir):
        for species in os.listdir(diseases_dir):
            species_path = os.path.join(diseases_dir, species)
            if not os.path.isdir(species_path):
                continue
            
            results['diseases'][species] = {}
            for disease in os.listdir(species_path):
                disease_path = os.path.join(species_path, disease)
                if os.path.isdir(disease_path):
                    count = sum(1 for f in os.listdir(disease_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')))
                    results['diseases'][species][disease] = count
    
    return results


def calculate_imbalance_metrics(class_counts):
    """
    Calculate imbalance metrics for class distribution.
    
    Returns:
        dict: Imbalance metrics
    """
    if not class_counts:
        return {}
    
    counts = np.array(list(class_counts.values()))
    if len(counts) == 0:
        return {}
    
    max_count = counts.max()
    min_count = counts.min()
    mean_count = counts.mean()
    
    return {
        'imbalance_ratio': float(max_count / min_count) if min_count > 0 else float('inf'),
        'max_count': int(max_count),
        'min_count': int(min_count),
        'mean_count': float(mean_count),
        'std_count': float(counts.std()),
        'cv': float(counts.std() / mean_count) if mean_count > 0 else 0  # Coefficient of variation
    }


def plot_dimension_analysis(dim_stats, out_dir='reports'):
    """Plot dimension analysis visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    
    if len(dim_stats['dimensions']) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Width distribution
    axes[0, 0].hist(dim_stats['dimensions'][:, 0], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].axvline(dim_stats['width_stats']['mean'], color='r', linestyle='--', label='Mean')
    axes[0, 0].legend()
    
    # Height distribution
    axes[0, 1].hist(dim_stats['dimensions'][:, 1], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].axvline(dim_stats['height_stats']['mean'], color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()
    
    # Aspect ratio distribution
    axes[1, 0].hist(dim_stats['aspect_ratios'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Aspect Ratio (W/H)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].axvline(dim_stats['aspect_ratio_stats']['mean'], color='r', linestyle='--', label='Mean')
    axes[1, 0].legend()
    
    # Scatter plot of dimensions
    axes[1, 1].scatter(dim_stats['dimensions'][:, 0], dim_stats['dimensions'][:, 1], alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Width (pixels)')
    axes[1, 1].set_ylabel('Height (pixels)')
    axes[1, 1].set_title('Image Dimensions Scatter')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dimension_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved dimension analysis to {out_dir}/dimension_analysis.png')


def generate_html_report(eda_results, out_path='reports/eda_report.html'):
    """
    Generate a comprehensive HTML report of EDA results.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - Multi-Crop Disease Classifier</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-size: 1.5em;
            color: #2c3e50;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 10px 0;
        }}
        .error {{
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin: 10px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>📊 Exploratory Data Analysis Report</h1>
    <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>📈 Dataset Overview</h2>
        <div class="metric">
            <div class="metric-label">Total Images</div>
            <div class="metric-value">{eda_results.get('total_images', 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Plant Species</div>
            <div class="metric-value">{len(eda_results.get('class_counts', {}).get('plants', {}))}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Disease Classes</div>
            <div class="metric-value">{sum(len(diseases) for diseases in eda_results.get('class_counts', {}).get('diseases', {}).values())}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>🌿 Plant Species Distribution</h2>
        <table>
            <tr>
                <th>Species</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""
    
    plant_counts = eda_results.get('class_counts', {}).get('plants', {})
    total_plants = sum(plant_counts.values()) if plant_counts else 0
    
    for species, count in sorted(plant_counts.items()):
        percentage = (count / total_plants * 100) if total_plants > 0 else 0
        html_content += f"""
            <tr>
                <td>{species}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""
    
    html_content += """
        </table>
"""
    
    plant_imbalance = eda_results.get('plant_imbalance', {})
    if plant_imbalance:
        ir = plant_imbalance.get('imbalance_ratio', 0)
        if ir > 2.0:
            html_content += f'<div class="warning">⚠️ High class imbalance detected (ratio: {ir:.2f}). Consider using class weights or oversampling.</div>'
        else:
            html_content += f'<div class="success">✓ Class distribution is relatively balanced (ratio: {ir:.2f}).</div>'
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>🦠 Disease Distribution by Species</h2>
"""
    
    disease_counts = eda_results.get('class_counts', {}).get('diseases', {})
    for species, diseases in sorted(disease_counts.items()):
        html_content += f"""
        <h3>{species}</h3>
        <table>
            <tr>
                <th>Disease</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""
        total_species = sum(diseases.values()) if diseases else 0
        for disease, count in sorted(diseases.items()):
            percentage = (count / total_species * 100) if total_species > 0 else 0
            html_content += f"""
            <tr>
                <td>{disease}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""
        html_content += """
        </table>
"""
        
        species_imbalance = eda_results.get('disease_imbalance', {}).get(species, {})
        if species_imbalance:
            ir = species_imbalance.get('imbalance_ratio', 0)
            if ir > 2.0:
                html_content += f'<div class="warning">⚠️ High imbalance in {species} diseases (ratio: {ir:.2f}).</div>'
            else:
                html_content += f'<div class="success">✓ {species} disease distribution is balanced (ratio: {ir:.2f}).</div>'
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>🔍 Duplicate Detection</h2>
"""
    
    dup_results = eda_results.get('duplicates', {})
    if dup_results.get('duplicate_images', 0) > 0:
        html_content += f"""
        <div class="warning">
            ⚠️ Found {dup_results['duplicate_groups']} groups of duplicate images 
            ({dup_results['duplicate_images']} duplicate files total).
        </div>
        <p>Duplicate groups:</p>
        <ul>
"""
        for hash_val, paths in list(dup_results.get('duplicates', {}).items())[:10]:  # Show first 10
            html_content += f"<li>{len(paths)} duplicates:<ul>"
            for path in paths[:3]:  # Show first 3 paths per group
                html_content += f"<li><code>{path}</code></li>"
            if len(paths) > 3:
                html_content += f"<li>... and {len(paths) - 3} more</li>"
            html_content += "</ul></li>"
        
        if len(dup_results.get('duplicates', {})) > 10:
            html_content += f"<li>... and {len(dup_results.get('duplicates', {})) - 10} more groups</li>"
        
        html_content += """
        </ul>
"""
    else:
        html_content += '<div class="success">✓ No duplicate images detected.</div>'
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>🖼️ Image Dimension Analysis</h2>
"""
    
    dim_stats = eda_results.get('dimensions', {})
    if dim_stats:
        html_content += f"""
        <h3>Width Statistics</h3>
        <div class="metric">
            <div class="metric-label">Min</div>
            <div class="metric-value">{dim_stats['width_stats']['min']}px</div>
        </div>
        <div class="metric">
            <div class="metric-label">Max</div>
            <div class="metric-value">{dim_stats['width_stats']['max']}px</div>
        </div>
        <div class="metric">
            <div class="metric-label">Mean</div>
            <div class="metric-value">{dim_stats['width_stats']['mean']:.0f}px</div>
        </div>
        <div class="metric">
            <div class="metric-label">Std Dev</div>
            <div class="metric-value">{dim_stats['width_stats']['std']:.0f}px</div>
        </div>
        
        <h3>Height Statistics</h3>
        <div class="metric">
            <div class="metric-label">Min</div>
            <div class="metric-value">{dim_stats['height_stats']['min']}px</div>
        </div>
        <div class="metric">
            <div class="metric-label">Max</div>
            <div class="metric-value">{dim_stats['height_stats']['max']}px</div>
        </div>
        <div class="metric">
            <div class="metric-label">Mean</div>
            <div class="metric-value">{dim_stats['height_stats']['mean']:.0f}px</div>
        </div>
        <div class="metric">
            <div class="metric-label">Std Dev</div>
            <div class="metric-value">{dim_stats['height_stats']['std']:.0f}px</div>
        </div>
        
        <h3>Aspect Ratio Statistics</h3>
        <div class="metric">
            <div class="metric-label">Min</div>
            <div class="metric-value">{dim_stats['aspect_ratio_stats']['min']:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Max</div>
            <div class="metric-value">{dim_stats['aspect_ratio_stats']['max']:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Mean</div>
            <div class="metric-value">{dim_stats['aspect_ratio_stats']['mean']:.2f}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Std Dev</div>
            <div class="metric-value">{dim_stats['aspect_ratio_stats']['std']:.2f}</div>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>🚨 Corrupted/Unreadable Images</h2>
"""
    
    corrupted = eda_results.get('corrupted', {})
    if corrupted.get('unreadable', 0) > 0:
        html_content += f"""
        <div class="error">
            ❌ Found {corrupted['unreadable']} unreadable/corrupted images 
            out of {corrupted['total_checked']} checked.
        </div>
        <p>These images have been moved to the bad_images directory.</p>
"""
    else:
        html_content += f'<div class="success">✓ All {corrupted.get("total_checked", 0)} images are readable and valid.</div>'
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>📊 Visualizations</h2>
        
        <h3>Plant Species Class Balance</h3>
        <img src="plant_class_balance.png" alt="Plant Class Balance">
        
        <h3>Plant Sample Grid</h3>
        <p>Random samples from the plant dataset with species labels:</p>
        <img src="data_samples.png" alt="Plant Sample Grid">
        
        <h3>Disease Class Balance by Species</h3>
"""
    
    # Add disease balance charts and sample grids for each species
    disease_counts = eda_results.get('class_counts', {}).get('diseases', {})
    for species in sorted(disease_counts.keys()):
        html_content += f"""
        <h4>{species} Diseases</h4>
        <img src="{species}_disease_class_balance.png" alt="{species} Disease Balance">
        <p>Sample images from {species} disease classes:</p>
        <img src="{species}_disease_samples.png" alt="{species} Disease Samples">
"""
    
    html_content += """
        <h3>Image Dimension Analysis</h3>
        <p>Distribution of image widths, heights, aspect ratios, and dimension scatter plot:</p>
        <img src="dimension_analysis.png" alt="Dimension Analysis">
    </div>
    
    <div class="section">
        <h2>📝 Summary and Recommendations</h2>
"""
    
    # Add recommendations based on findings
    recommendations = []
    
    # Check for imbalance
    plant_imbalance = eda_results.get('plant_imbalance', {})
    if plant_imbalance.get('imbalance_ratio', 0) > 2.0:
        recommendations.append("⚠️ <strong>Class Imbalance:</strong> Consider using class weights, oversampling (SMOTE), or data augmentation for minority classes in plant species classification.")
    
    disease_imbalance = eda_results.get('disease_imbalance', {})
    for species, imb in disease_imbalance.items():
        if imb.get('imbalance_ratio', 0) > 2.0:
            recommendations.append(f"⚠️ <strong>{species} Disease Imbalance:</strong> Apply weighted sampling or augmentation for {species} disease classes.")
    
    # Check for duplicates
    dup_results = eda_results.get('duplicates', {})
    if dup_results.get('duplicate_images', 0) > 0:
        recommendations.append(f"⚠️ <strong>Duplicates Found:</strong> {dup_results['duplicate_images']} duplicate images detected. Consider removing duplicates to prevent data leakage between train/val/test sets.")
    
    # Check for corrupted images
    corrupted = eda_results.get('corrupted', {})
    if corrupted.get('unreadable', 0) > 0:
        recommendations.append(f"⚠️ <strong>Corrupted Images:</strong> {corrupted['unreadable']} unreadable images found and moved to bad_images directory. Review and replace if possible.")
    
    # Dimension recommendations
    dim_stats = eda_results.get('dimensions', {})
    if dim_stats:
        width_std = dim_stats['width_stats']['std']
        height_std = dim_stats['height_stats']['std']
        if width_std > 500 or height_std > 500:
            recommendations.append("ℹ️ <strong>High Dimension Variance:</strong> Images have varying dimensions. Preprocessing with consistent resizing is already applied (256x256 shortest side, 224x224 center crop).")
    
    if not recommendations:
        recommendations.append("✅ <strong>Dataset Quality:</strong> No major issues detected. Dataset is ready for training!")
    
    html_content += "<ul>"
    for rec in recommendations:
        html_content += f"<li>{rec}</li>"
    html_content += "</ul>"
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>🔗 Additional Resources</h2>
        <p>All visualizations and analysis outputs are available in the reports directory:</p>
        <ul>
            <li><code>plant_class_balance.png</code> — Plant species distribution chart</li>
            <li><code>data_samples.png</code> — Sample grid of plant images</li>
            <li><code>{Species}_disease_class_balance.png</code> — Disease balance charts (per species)</li>
            <li><code>{Species}_disease_samples.png</code> — Disease sample grids (per species)</li>
            <li><code>dimension_analysis.png</code> — Image dimension analysis plots</li>
        </ul>
    </div>
    
</body>
</html>
"""
    
    with open(out_path, 'w') as f:
        f.write(html_content)
    
    print(f'HTML report saved to: {out_path}')


def comprehensive_eda(data_dir='data', out_dir='reports', bad_images_dir='data/bad_images', 
                      dup_dir='data/dup_images', remove_duplicates=False):
    """
    Run comprehensive EDA including all analysis components.
    
    Args:
        data_dir: Root data directory
        out_dir: Output directory for reports
        bad_images_dir: Directory to move corrupted images
        dup_dir: Directory to move duplicate images
        remove_duplicates: If True, remove duplicate images
    
    Returns:
        dict: Complete EDA results
    """
    print('='*60)
    print('Running Comprehensive EDA')
    print('='*60)
    
    results = {}
    
    # 1. Check for corrupted images
    print('\n1. Checking for corrupted/unreadable images...')
    corrupted_stats = check_unreadable_images(data_dir=data_dir, bad_images_dir=bad_images_dir)
    results['corrupted'] = corrupted_stats
    
    # 2. Detect duplicates
    print('\n2. Detecting duplicate images...')
    dup_stats = detect_duplicates(data_dir=data_dir, dup_dir=dup_dir, remove_duplicates=remove_duplicates)
    results['duplicates'] = dup_stats
    print(f'   Found {dup_stats["duplicate_groups"]} groups of duplicates ({dup_stats["duplicate_images"]} duplicate images)')
    if remove_duplicates:
        print(f'   Moved {dup_stats.get("moved_count", 0)} duplicate images to {dup_dir}')
    
    # 3. Get class counts
    print('\n3. Analyzing class distributions...')
    class_counts = get_class_counts(data_dir=data_dir)
    results['class_counts'] = class_counts
    
    # Calculate total images
    total_plants = sum(class_counts['plants'].values())
    total_diseases = sum(sum(diseases.values()) for diseases in class_counts['diseases'].values())
    results['total_images'] = total_plants + total_diseases
    
    # 4. Calculate imbalance metrics
    print('\n4. Calculating imbalance metrics...')
    results['plant_imbalance'] = calculate_imbalance_metrics(class_counts['plants'])
    results['disease_imbalance'] = {}
    for species, diseases in class_counts['diseases'].items():
        results['disease_imbalance'][species] = calculate_imbalance_metrics(diseases)
    
    # 5. Analyze dimensions
    print('\n5. Analyzing image dimensions...')
    dim_stats = analyze_image_dimensions(data_dir=data_dir)
    results['dimensions'] = dim_stats
    
    # 6. Generate visualizations
    print('\n6. Generating visualizations...')
    plot_class_balance_plant(data_dir=os.path.join(data_dir, 'plants'), out_dir=out_dir)
    plot_class_balance_diseases(data_dir=os.path.join(data_dir, 'diseases'), out_dir=out_dir)
    save_sanity_grid(data_dir=os.path.join(data_dir, 'plants'), out_path=os.path.join(out_dir, 'data_samples.png'))
    save_disease_grids(data_dir=os.path.join(data_dir, 'diseases'), out_dir=out_dir)
    plot_dimension_analysis(dim_stats, out_dir=out_dir)
    
    # 7. Generate HTML report
    print('\n7. Generating HTML report...')
    generate_html_report(results, out_path=os.path.join(out_dir, 'eda_report.html'))
    
    print('\n' + '='*60)
    print('EDA Complete!')
    print(f'Reports saved to: {out_dir}')
    print('='*60)
    
    return results


if __name__ == '__main__':
    print('Run functions: plot_class_balance_plant, plot_class_balance_diseases, save_sanity_grid')

