#!/usr/bin/env python3
"""
Extract and organize data from data.zip into the expected structure:
  data/
    plants/
      Cashew/
      Cassava/
      Maize/
      Tomato/
    diseases/
      Cashew/
        <disease_class>/
      Cassava/
        <disease_class>/
      Maize/
        <disease_class>/
      Tomato/
        <disease_class>/

Usage:
    python scripts/setup_data.py
    python scripts/setup_data.py --zip-path data.zip --output-dir data
"""
import os
import sys
import argparse
import zipfile
from pathlib import Path


def unzip_and_organize(zip_path: str, output_dir: str, force: bool = False):
    """Extract data.zip and organize into expected structure."""
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    
    if not zip_path.exists():
        print(f"Error: {zip_path} not found")
        return False
    
    # Check if data already exists
    plants_dir = output_dir / 'plants'
    diseases_dir = output_dir / 'diseases'
    
    if plants_dir.exists() and diseases_dir.exists() and not force:
        print(f"Data directories already exist at {output_dir}")
        print("Use --force to re-extract and overwrite")
        return True
    
    print(f"Extracting {zip_path}...")
    
    # Create temp extraction directory
    temp_dir = output_dir / '_temp_extract'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        print(f"Extracted to {temp_dir}")
        
        # Find the actual data structure within the extracted files
        # The zip might have a root folder or be flat
        extracted_items = list(temp_dir.iterdir())
        
        # If there's a single directory, assume that's the root
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            root = extracted_items[0]
        else:
            root = temp_dir
        
        print(f"Organizing data from {root}...")
        
        # Create output structure
        plants_dir.mkdir(parents=True, exist_ok=True)
        diseases_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected species
        species = ['Cashew', 'Cassava', 'Maize', 'Tomato']
        
        # Look for data patterns and organize
        # Strategy: scan for folders that look like species names
        for item in root.rglob('*'):
            if not item.is_dir():
                continue
            
            # Check if this folder matches a species name (case-insensitive)
            item_name = item.name
            matching_species = None
            for sp in species:
                if sp.lower() in item_name.lower():
                    matching_species = sp
                    break
            
            if matching_species:
                # Check if this contains disease subfolders or direct images
                subfolders = [d for d in item.iterdir() if d.is_dir()]
                image_files = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                
                if subfolders and not image_files:
                    # This is a disease folder structure
                    disease_out = diseases_dir / matching_species
                    disease_out.mkdir(parents=True, exist_ok=True)
                    
                    for subfolder in subfolders:
                        target = disease_out / subfolder.name
                        if not target.exists():
                            print(f"  Moving {subfolder.relative_to(temp_dir)} -> {target.relative_to(output_dir)}")
                            subfolder.rename(target)
                        else:
                            print(f"  Skipping {subfolder.name} (already exists)")
                
                elif image_files:
                    # This is a plant folder with direct images
                    plant_out = plants_dir / matching_species
                    plant_out.mkdir(parents=True, exist_ok=True)
                    
                    # Move all images
                    for img in image_files:
                        target = plant_out / img.name
                        if not target.exists():
                            img.rename(target)
                    
                    print(f"  Moved {len(image_files)} images to {plant_out.relative_to(output_dir)}")
        
        # Fallback: if structure isn't organized as expected, try direct copy
        # Check if extracted root has 'plants' and 'diseases' folders
        if (root / 'plants').exists():
            print("Found 'plants' folder, copying...")
            import shutil
            if plants_dir.exists():
                shutil.rmtree(plants_dir)
            shutil.copytree(root / 'plants', plants_dir)
        
        if (root / 'diseases').exists():
            print("Found 'diseases' folder, copying...")
            import shutil
            if diseases_dir.exists():
                shutil.rmtree(diseases_dir)
            shutil.copytree(root / 'diseases', diseases_dir)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"\n✓ Data extraction complete!")
        print(f"  Plants data: {plants_dir}")
        print(f"  Diseases data: {diseases_dir}")
        
        # Verify structure
        verify_structure(output_dir)
        
        return True
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_structure(data_dir: Path):
    """Verify the data directory structure."""
    print("\nVerifying data structure...")
    
    plants_dir = data_dir / 'plants'
    diseases_dir = data_dir / 'diseases'
    
    if not plants_dir.exists():
        print(f"  ⚠ Missing: {plants_dir}")
        return
    
    if not diseases_dir.exists():
        print(f"  ⚠ Missing: {diseases_dir}")
        return
    
    # Count species folders
    plant_species = [d.name for d in plants_dir.iterdir() if d.is_dir()]
    disease_species = [d.name for d in diseases_dir.iterdir() if d.is_dir()]
    
    print(f"  ✓ Plant species found: {len(plant_species)} - {plant_species}")
    print(f"  ✓ Disease species found: {len(disease_species)} - {disease_species}")
    
    # Count images per species
    for sp in plant_species:
        imgs = list((plants_dir / sp).rglob('*.jpg')) + list((plants_dir / sp).rglob('*.jpeg')) + list((plants_dir / sp).rglob('*.png'))
        print(f"    {sp}: {len(imgs)} images")
    
    print()
    for sp in disease_species:
        disease_classes = [d.name for d in (diseases_dir / sp).iterdir() if d.is_dir()]
        total_imgs = sum(len(list((diseases_dir / sp / dc).rglob('*.jpg')) + list((diseases_dir / sp / dc).rglob('*.jpeg')) + list((diseases_dir / sp / dc).rglob('*.png'))) for dc in disease_classes)
        print(f"    {sp}: {len(disease_classes)} disease classes, {total_imgs} images")


def main():
    parser = argparse.ArgumentParser(description='Extract and organize data.zip into expected structure')
    parser.add_argument('--zip-path', default='data.zip', help='Path to data.zip file')
    parser.add_argument('--output-dir', default='data', help='Output directory for extracted data')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if data exists')
    args = parser.parse_args()
    
    success = unzip_and_organize(args.zip_path, args.output_dir, args.force)
    
    if success:
        print("\n✓ Setup complete! You can now run:")
        print("  python scripts/run_eda.py --plants-data data/plants --diseases-data data/diseases --out-dir reports")
        print("  python scripts/prepare_splits.py --plants-data data/plants --diseases-data data/diseases --out-dir data_splits")
        sys.exit(0)
    else:
        print("\n✗ Setup failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
