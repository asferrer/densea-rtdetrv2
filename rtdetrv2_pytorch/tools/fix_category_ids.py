#!/usr/bin/env python3
"""
Fix category IDs in COCO JSON to be 1-indexed instead of 0-indexed.

COCO format requires category IDs to start from 1, not 0.
0-indexed categories cause mAP to be 0 because predictions don't match ground truth.
"""

import json
import argparse
from pathlib import Path
import shutil


def fix_category_ids(json_path, output_path=None, backup=True):
    """
    Fix category IDs to be 1-indexed (add 1 to all IDs).

    Args:
        json_path: Path to input JSON
        output_path: Path to output JSON (if None, overwrites input)
        backup: Whether to create backup of original file
    """
    json_path = Path(json_path)

    if output_path is None:
        output_path = json_path
    else:
        output_path = Path(output_path)

    # Load JSON
    print(f"Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Check current category IDs
    cat_ids = [cat['id'] for cat in data['categories']]
    print(f"\nOriginal category IDs: {cat_ids}")

    if min(cat_ids) == 0:
        print("⚠️  Categories are 0-indexed - FIXING")

        # Backup original
        if backup and output_path == json_path:
            backup_path = json_path.parent / f"{json_path.stem}_backup.json"
            print(f"Creating backup: {backup_path}")
            shutil.copy(json_path, backup_path)

        # Fix categories: add 1 to all IDs
        for cat in data['categories']:
            cat['id'] += 1

        # Fix annotations: add 1 to all category_ids
        for ann in data['annotations']:
            ann['category_id'] += 1

        # Verify fix
        new_cat_ids = [cat['id'] for cat in data['categories']]
        print(f"Fixed category IDs: {new_cat_ids}")

        # Save fixed JSON
        print(f"\nSaving fixed JSON: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ FIXED: Categories are now 1-indexed")
        print(f"\nCategory mapping:")
        for cat in data['categories']:
            print(f"  {cat['id']}: {cat['name']}")

        return True

    elif min(cat_ids) == 1:
        print("✅ Categories are already 1-indexed - NO CHANGES NEEDED")
        return False

    else:
        print(f"⚠️  WARNING: Unexpected category IDs starting from {min(cat_ids)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Fix category IDs in COCO JSON (0-indexed → 1-indexed)"
    )
    parser.add_argument(
        'json_files',
        nargs='+',
        help='JSON file(s) to fix'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    args = parser.parse_args()

    print("="*70)
    print("FIXING CATEGORY IDs IN COCO JSON FILES")
    print("="*70)

    for json_file in args.json_files:
        print(f"\n{'='*70}")
        fixed = fix_category_ids(json_file, backup=not args.no_backup)
        print(f"{'='*70}")

    print("\n✅ ALL FILES PROCESSED")


if __name__ == '__main__':
    main()
