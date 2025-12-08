#!/usr/bin/env python3
"""
Run inference on plant disease images using the two-stage classification pipeline.
"""
import os
import sys
import argparse
import json
from pathlib import Path

# Fix macOS OpenMP conflict (conda environment issue)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ensure project root is on path so `src` imports work when running scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import InferencePipeline


def load_species_mapping(plant_meta_path):
    """Load species names from plant checkpoint metadata."""
    if os.path.exists(plant_meta_path):
        with open(plant_meta_path, 'r') as f:
            meta = json.load(f)
            return meta.get('species', ['Cashew', 'Cassava', 'Maize', 'Tomato'])
    return ['Cashew', 'Cassava', 'Maize', 'Tomato']


def load_disease_names(species, models_dir):
    """Load disease class names for a species."""
    meta_path = os.path.join(models_dir, f'{species}_checkpoint.pth.meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            return meta.get('classes', [])
    return []


def predict_single_image(pipeline, image_path, species_names, disease_names_map):
    """Run prediction on a single image and return formatted results."""
    results = pipeline.predict(image_path)
    
    # Format plant prediction
    plant_idx = results.get('plant_prediction', 0)
    plant_conf = results.get('plant_confidence', 0.0)
    plant_name = species_names[plant_idx] if plant_idx < len(species_names) else f'Unknown ({plant_idx})'
    
    # Format disease prediction
    disease_idx = results.get('disease_prediction')
    disease_conf = results.get('disease_confidence', 0.0)
    disease_name = 'N/A'
    
    if disease_idx is not None and plant_name in disease_names_map:
        disease_classes = disease_names_map[plant_name]
        disease_name = disease_classes[disease_idx] if disease_idx < len(disease_classes) else f'Unknown ({disease_idx})'
    
    formatted = {
        'image': os.path.basename(image_path),
        'plant_species': plant_name,
        'plant_confidence': f'{plant_conf:.4f}',
        'disease_class': disease_name,
        'disease_confidence': f'{disease_conf:.4f}' if disease_conf else 'N/A',
        'note': results.get('note', '')
    }
    
    return formatted


def predict_batch(pipeline, image_paths, species_names, disease_names_map, verbose=True):
    """Run predictions on multiple images."""
    results = []
    
    for i, img_path in enumerate(image_paths):
        if verbose:
            print(f'Processing ({i+1}/{len(image_paths)}): {os.path.basename(img_path)}')
        
        try:
            result = predict_single_image(pipeline, img_path, species_names, disease_names_map)
            results.append(result)
            
            if verbose:
                print(f'  → Plant: {result["plant_species"]} (conf: {result["plant_confidence"]})')
                print(f'  → Disease: {result["disease_class"]} (conf: {result["disease_confidence"]})')
                if result['note']:
                    print(f'  → Note: {result["note"]}')
        except Exception as e:
            print(f'  ❌ Error: {str(e)}')
            results.append({
                'image': os.path.basename(img_path),
                'error': str(e)
            })
    
    return results


def save_results(results, output_path):
    """Save prediction results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ Results saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on plant disease images using two-stage classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python scripts/run_inference.py --image data/plants/Tomato/img001.jpg
  
  # Directory of images
  python scripts/run_inference.py --image-dir data/plants/Tomato
  
  # Multiple images with output to file
  python scripts/run_inference.py --image img1.jpg img2.jpg img3.jpg --output predictions.json
  
  # Batch inference with custom threshold
  python scripts/run_inference.py --image-dir test_images --threshold 0.7 --output results.json
"""
    )
    
    # Model arguments
    parser.add_argument('--plant-checkpoint', default='models/plant_checkpoint.pth',
                       help='Path to plant (Stage-1) model checkpoint')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing disease model checkpoints')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'],
                       help='Device to run inference on')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for routing to disease classifier')
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', nargs='+', help='Path(s) to image file(s)')
    input_group.add_argument('--image-dir', help='Directory containing images')
    
    # Output arguments
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check if plant checkpoint exists
    if not os.path.exists(args.plant_checkpoint):
        print(f'❌ Error: Plant checkpoint not found at {args.plant_checkpoint}')
        sys.exit(1)
    
    # Load species names
    plant_meta_path = args.plant_checkpoint + '.meta.json'
    species_names = load_species_mapping(plant_meta_path)
    
    # Build disease checkpoint paths
    disease_checkpoints = {}
    disease_names_map = {}
    
    for species in species_names:
        ck_path = os.path.join(args.models_dir, f'{species}_checkpoint.pth')
        if os.path.exists(ck_path):
            disease_checkpoints[species] = ck_path
            disease_names_map[species] = load_disease_names(species, args.models_dir)
        else:
            print(f'⚠️  Warning: Disease checkpoint not found for {species}')
    
    if not disease_checkpoints:
        print('❌ Error: No disease model checkpoints found')
        sys.exit(1)
    
    # Initialize pipeline
    if not args.quiet:
        print('=' * 60)
        print('Initializing Inference Pipeline')
        print('=' * 60)
        print(f'Plant Model: {args.plant_checkpoint}')
        print(f'Disease Models: {len(disease_checkpoints)} loaded')
        print(f'Device: {args.device}')
        print(f'Threshold: {args.threshold}')
        print('=' * 60)
    
    pipeline = InferencePipeline(
        plant_checkpoint=args.plant_checkpoint,
        disease_checkpoints=disease_checkpoints,
        device=args.device,
        threshold=args.threshold
    )
    
    # Collect image paths
    image_paths = []
    
    if args.image:
        for img_path in args.image:
            if os.path.exists(img_path) and os.path.isfile(img_path):
                image_paths.append(img_path)
            else:
                print(f'⚠️  Warning: Image not found: {img_path}')
    
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f'❌ Error: Directory not found: {args.image_dir}')
            sys.exit(1)
        
        for fname in os.listdir(args.image_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                image_paths.append(os.path.join(args.image_dir, fname))
    
    if not image_paths:
        print('❌ Error: No valid images found')
        sys.exit(1)
    
    if not args.quiet:
        print(f'\n📷 Found {len(image_paths)} image(s) to process\n')
    
    # Run predictions
    results = predict_batch(
        pipeline, 
        image_paths, 
        species_names, 
        disease_names_map,
        verbose=not args.quiet
    )
    
    # Save results if output file specified
    if args.output:
        save_results(results, args.output)
    
    # Print summary
    if not args.quiet:
        print('\n' + '=' * 60)
        print('Inference Complete')
        print('=' * 60)
        successful = sum(1 for r in results if 'error' not in r)
        print(f'Successfully processed: {successful}/{len(results)} images')
        
        # Print quick statistics
        if successful > 0:
            plant_counts = {}
            for r in results:
                if 'error' not in r:
                    plant = r.get('plant_species', 'Unknown')
                    plant_counts[plant] = plant_counts.get(plant, 0) + 1
            
            print('\nPlant Species Distribution:')
            for plant, count in sorted(plant_counts.items()):
                print(f'  {plant}: {count}')


if __name__ == '__main__':
    main()
