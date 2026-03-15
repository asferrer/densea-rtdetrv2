#!/usr/bin/env python3
"""
Convert category IDs from 1-indexed (COCO standard) to 0-indexed (PyTorch standard).

This is needed because the postprocessor has a bug where it uses a global hardcoded
mapping instead of the dataset-specific mapping. By using 0-indexed categories
and remap_mscoco_category=False everywhere, we avoid this bug.

Usage:
    python tools/convert_to_0indexed.py split_sintetico/train.json split_sintetico/val.json
"""

import json
import sys
from pathlib import Path
import shutil


def convert_to_0indexed(json_path, backup=True):
    """
    Convert category IDs from 1-indexed to 0-indexed (subtract 1 from all IDs).

    Args:
        json_path: Path to input JSON
        backup: Whether to create backup of original file
    """
    json_path = Path(json_path)

    # Load JSON
    print(f"Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Check current category IDs
    cat_ids = [cat['id'] for cat in data['categories']]
    print(f"\nOriginal category IDs: {cat_ids}")

    if min(cat_ids) == 1:
        print("⚠️  Categories are 1-indexed - CONVERTING TO 0-INDEXED")

        # Backup original
        if backup:
            backup_path = json_path.parent / f"{json_path.stem}_1indexed_backup.json"
            print(f"Creating backup: {backup_path}")
            shutil.copy(json_path, backup_path)

        # Convert categories: subtract 1 from all IDs
        for cat in data['categories']:
            cat['id'] -= 1

        # Convert annotations: subtract 1 from all category_ids
        for ann in data['annotations']:
            ann['category_id'] -= 1

        # Verify conversion
        new_cat_ids = [cat['id'] for cat in data['categories']]
        print(f"Converted category IDs: {new_cat_ids}")

        # Save converted JSON
        print(f"\nSaving converted JSON: {json_path}")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ CONVERTED: Categories are now 0-indexed")
        print(f"\nCategory mapping:")
        for cat in data['categories']:
            print(f"  {cat['id']}: {cat['name']}")

        return True

    elif min(cat_ids) == 0:
        print("✅ Categories are already 0-indexed - NO CHANGES NEEDED")
        return False

    else:
        print(f"⚠️  WARNING: Unexpected category IDs starting from {min(cat_ids)}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert category IDs from 1-indexed to 0-indexed"
    )
    parser.add_argument(
        'json_files',
        nargs='+',
        help='JSON file(s) to convert'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    args = parser.parse_args()

    print("="*70)
    print("CONVERTING CATEGORY IDs TO 0-INDEXED")
    print("="*70)

    for json_file in args.json_files:
        print(f"\n{'='*70}")
        converted = convert_to_0indexed(json_file, backup=not args.no_backup)
        print(f"{'='*70}")

    print("\n✅ ALL FILES PROCESSED")
    print("\n⚠️  IMPORTANT: Now set `remap_mscoco_category: False` in your config!")


if __name__ == '__main__':
    main()
