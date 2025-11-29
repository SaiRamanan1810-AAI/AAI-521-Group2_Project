#!/usr/bin/env python3
import os
import sys
import argparse

# ensure project root is on path so `src` imports work when running scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda import plot_class_balance_plant, plot_class_balance_diseases, save_sanity_grid, save_disease_grids, check_unreadable_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plants-data', default='data/plants', help='Path to plant folders')
    parser.add_argument('--diseases-data', default='data/diseases', help='Path to disease folders')
    parser.add_argument('--out-dir', default='reports', help='Output directory for EDA figures')
    parser.add_argument('--data-root', default='data', help='Root data directory for image checking')
    parser.add_argument('--bad-images-dir', default='data/bad_images', help='Directory to move unreadable images')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('=' * 60)
    print('Checking for unreadable images...')
    print('=' * 60)
    check_unreadable_images(data_dir=args.data_root, bad_images_dir=args.bad_images_dir)
    
    print('\n' + '=' * 60)
    print('Generating plant class balance...')
    print('=' * 60)
    plot_class_balance_plant(data_dir=args.plants_data, out_dir=args.out_dir)
    
    print('\n' + '=' * 60)
    print('Generating disease class balances...')
    print('=' * 60)
    plot_class_balance_diseases(data_dir=args.diseases_data, out_dir=args.out_dir)
    
    print('\n' + '=' * 60)
    print('Saving plant sanity sample grid...')
    print('=' * 60)
    save_sanity_grid(data_dir=args.plants_data, out_path=os.path.join(args.out_dir, 'data_samples.png'))
    
    print('\n' + '=' * 60)
    print('Saving disease sample grids...')
    print('=' * 60)
    save_disease_grids(data_dir=args.diseases_data, out_dir=args.out_dir)
    
    print('\n' + '=' * 60)
    print(f'EDA reports saved to {args.out_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
