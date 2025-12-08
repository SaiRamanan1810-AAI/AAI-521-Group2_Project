#!/usr/bin/env python3
import os
import sys
import argparse

# ensure project root is on path so `src` imports work when running scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.eda import comprehensive_eda


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive exploratory data analysis')
    parser.add_argument('--data-root', default='data', help='Root data directory')
    parser.add_argument('--out-dir', default='reports', help='Output directory for EDA reports and figures')
    parser.add_argument('--bad-images-dir', default='data/bad_images', help='Directory to move unreadable images')
    parser.add_argument('--dup-dir', default='data/dup_images', help='Directory to move duplicate images')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate images and move to dup-dir')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Run comprehensive EDA
    results = comprehensive_eda(
        data_dir=args.data_root,
        out_dir=args.out_dir,
        bad_images_dir=args.bad_images_dir,
        dup_dir=args.dup_dir,
        remove_duplicates=args.remove_duplicates
    )
    
    print('\n📊 EDA Summary:')
    print(f'   Total Images: {results.get("total_images", 0)}')
    print(f'   Duplicate Groups: {results.get("duplicates", {}).get("duplicate_groups", 0)}')
    print(f'   Corrupted Images: {results.get("corrupted", {}).get("unreadable", 0)}')
    if args.remove_duplicates:
        print(f'   Duplicates Removed: {results.get("duplicates", {}).get("moved_count", 0)}')
    print(f'\n✅ HTML Report: {os.path.join(args.out_dir, "eda_report.html")}')


if __name__ == '__main__':
    main()
