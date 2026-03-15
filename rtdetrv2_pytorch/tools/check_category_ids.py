#!/usr/bin/env python3
"""
Simple script to check category IDs in COCO JSON files.
Verifies that categories are 0-indexed as expected.
"""

import json
import sys
from pathlib import Path


def check_json(json_path):
    """Check category IDs in a COCO JSON file."""
    print(f"\n{'='*70}")
    print(f"Checking: {json_path}")
    print(f"{'='*70}")

    with open(json_path) as f:
        data = json.load(f)

    # Get category IDs
    categories = data['categories']
    cat_ids = [cat['id'] for cat in categories]

    print(f"\nCategories ({len(categories)}):")
    for cat in categories:
        print(f"  ID {cat['id']}: {cat['name']}")

    print(f"\nCategory ID range: [{min(cat_ids)}, {max(cat_ids)}]")

    # Check if 0-indexed or 1-indexed
    if min(cat_ids) == 0:
        print(f"✅ Categories are 0-indexed (correct for remap=False)")
    elif min(cat_ids) == 1:
        print(f"❌ Categories are 1-indexed (need to convert to 0-indexed)")
        print(f"   Run: python tools/convert_to_0indexed.py {json_path}")
    else:
        print(f"⚠️  WARNING: Unexpected category IDs starting from {min(cat_ids)}")

    # Sample some annotations
    print(f"\nSample annotations (first 5):")
    for i, ann in enumerate(data['annotations'][:5]):
        print(f"  {ann['id']}: image_id={ann['image_id']}, category_id={ann['category_id']}, bbox={ann['bbox']}")

    # Check annotation category_id range
    ann_cat_ids = [ann['category_id'] for ann in data['annotations']]
    print(f"\nAnnotation category_id range: [{min(ann_cat_ids)}, {max(ann_cat_ids)}]")

    if min(ann_cat_ids) == 0:
        print(f"✅ Annotations use 0-indexed category_ids")
    elif min(ann_cat_ids) == 1:
        print(f"❌ Annotations use 1-indexed category_ids")

    # Check for mismatches
    unique_ann_cats = set(ann_cat_ids)
    unique_def_cats = set(cat_ids)

    if unique_ann_cats != unique_def_cats:
        print(f"\n⚠️  WARNING: Mismatch between defined categories and annotation categories")
        print(f"   Defined: {sorted(unique_def_cats)}")
        print(f"   Used in annotations: {sorted(unique_ann_cats)}")
        missing = unique_ann_cats - unique_def_cats
        if missing:
            print(f"   Missing category definitions for: {sorted(missing)}")
    else:
        print(f"\n✅ All annotation categories have definitions")

    return min(cat_ids) == 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'json_files',
        nargs='+',
        help='JSON file(s) to check'
    )
    args = parser.parse_args()

    all_correct = True
    for json_file in args.json_files:
        is_correct = check_json(json_file)
        all_correct = all_correct and is_correct

    print(f"\n{'='*70}")
    if all_correct:
        print("✅ ALL FILES ARE 0-INDEXED (CORRECT)")
        print("\nYour config should have:")
        print("  remap_mscoco_category: False")
    else:
        print("❌ SOME FILES ARE NOT 0-INDEXED")
        print("\nYou need to:")
        print("  1. Run: python tools/convert_to_0indexed.py split_sintetico/*.json")
        print("  2. Set in config: remap_mscoco_category: False")
    print(f"{'='*70}")
