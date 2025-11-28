#!/usr/bin/env python3
import os
import sys
import argparse

# ensure project root is on path so `src` imports work when running scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda import plot_class_balance_plant, plot_class_balance_diseases, save_sanity_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plants-data', default='data/plants', help='Path to plant folders')
    parser.add_argument('--diseases-data', default='data/diseases', help='Path to disease folders')
    parser.add_argument('--out-dir', default='reports', help='Output directory for EDA figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('Generating plant class balance...')
    plot_class_balance_plant(data_dir=args.plants_data, out_dir=args.out_dir)
    print('Generating disease class balances...')
    plot_class_balance_diseases(data_dir=args.diseases_data, out_dir=args.out_dir)
    print('Saving sanity sample grid...')
    save_sanity_grid(data_dir=args.plants_data, out_path=os.path.join(args.out_dir, 'data_samples.png'))
    print(f'EDA reports saved to {args.out_dir}')


if __name__ == '__main__':
    main()
