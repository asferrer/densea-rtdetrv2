#!/usr/bin/env python3
"""
Merge synthetic and real COCO datasets for mixed training.

Creates a combined dataset with configurable sampling ratios from each source.
Handles ID remapping to avoid conflicts.

Usage:
    python tools/merge_synthetic_real.py \
        --synthetic-json ./dataset/dataset_sintetico/synthetic_train.json \
        --real-json ./split_v8/train.json \
        --output-dir ./split_mixed_70synth_30real \
        --synthetic-ratio 0.7 \
        --real-ratio 1.0 \
        --seed 42
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_coco_dataset(json_path: str) -> Dict:
    """Load COCO format JSON dataset."""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    print(f"  Images: {len(dataset['images'])}")
    print(f"  Annotations: {len(dataset['annotations'])}")
    print(f"  Categories: {len(dataset['categories'])}")

    return dataset


def sample_dataset(dataset: Dict, sample_ratio: float, seed: int) -> Dict:
    """
    Sample a fraction of the dataset.

    Args:
        dataset: COCO dataset dict
        sample_ratio: Fraction to sample (0.0-1.0)
        seed: Random seed

    Returns:
        Sampled dataset
    """
    if sample_ratio >= 1.0:
        print(f"  Using 100% of dataset ({len(dataset['images'])} images)")
        return dataset

    random.seed(seed)

    # Sample images
    all_image_ids = [img['id'] for img in dataset['images']]
    num_samples = int(len(all_image_ids) * sample_ratio)
    sampled_image_ids = set(random.sample(all_image_ids, num_samples))

    print(f"  Sampling {sample_ratio*100:.1f}%: "
          f"{num_samples}/{len(all_image_ids)} images")

    # Filter images and annotations
    sampled_images = [img for img in dataset['images']
                     if img['id'] in sampled_image_ids]
    sampled_annotations = [ann for ann in dataset['annotations']
                          if ann['image_id'] in sampled_image_ids]

    sampled_dataset = {
        'info': dataset.get('info', {}),
        'licenses': dataset.get('licenses', []),
        'categories': dataset['categories'],
        'images': sampled_images,
        'annotations': sampled_annotations
    }

    return sampled_dataset


def remap_ids(dataset: Dict, image_id_offset: int, ann_id_offset: int) -> Dict:
    """
    Remap image and annotation IDs to avoid conflicts.

    Args:
        dataset: COCO dataset
        image_id_offset: Offset to add to image IDs
        ann_id_offset: Offset to add to annotation IDs

    Returns:
        Dataset with remapped IDs
    """
    # Create mapping of old -> new image IDs
    old_to_new_image_id = {}
    for img in dataset['images']:
        old_id = img['id']
        new_id = old_id + image_id_offset
        old_to_new_image_id[old_id] = new_id
        img['id'] = new_id

    # Remap annotation IDs and image_id references
    for ann in dataset['annotations']:
        ann['id'] = ann['id'] + ann_id_offset
        ann['image_id'] = old_to_new_image_id[ann['image_id']]

    return dataset


def merge_datasets(
    synthetic_dataset: Dict,
    real_dataset: Dict,
    output_name: str = "mixed"
) -> Dict:
    """
    Merge synthetic and real datasets.

    Args:
        synthetic_dataset: Synthetic COCO dataset
        real_dataset: Real COCO dataset
        output_name: Name for merged dataset

    Returns:
        Merged COCO dataset
    """
    print("\nMerging datasets...")

    # Remap synthetic IDs (add offset to avoid conflicts)
    print("Remapping synthetic IDs...")
    max_real_image_id = max(img['id'] for img in real_dataset['images'])
    max_real_ann_id = max(ann['id'] for ann in real_dataset['annotations'])

    synthetic_remapped = remap_ids(
        synthetic_dataset,
        image_id_offset=max_real_image_id + 1,
        ann_id_offset=max_real_ann_id + 1
    )

    # Merge images and annotations
    merged_images = real_dataset['images'] + synthetic_remapped['images']
    merged_annotations = (real_dataset['annotations'] +
                         synthetic_remapped['annotations'])

    # Use real dataset's categories (should be the same)
    # Verify categories match
    real_cats = {cat['id']: cat['name'] for cat in real_dataset['categories']}
    synth_cats = {cat['id']: cat['name']
                  for cat in synthetic_dataset['categories']}

    if real_cats != synth_cats:
        print("WARNING: Category mismatch between datasets!")
        print(f"  Real categories: {real_cats}")
        print(f"  Synthetic categories: {synth_cats}")

    # Create merged dataset
    merged_dataset = {
        'info': {
            'description': f'{output_name} dataset (synthetic + real)',
            'year': 2025,
            'contributor': 'RT-DETRv2-DenSea',
            'date_created': '2025-01-02'
        },
        'licenses': real_dataset.get('licenses', []),
        'categories': real_dataset['categories'],
        'images': merged_images,
        'annotations': merged_annotations
    }

    print(f"  Total images: {len(merged_images)}")
    print(f"    Real: {len(real_dataset['images'])}")
    print(f"    Synthetic: {len(synthetic_remapped['images'])}")
    print(f"  Total annotations: {len(merged_annotations)}")
    print(f"    Real: {len(real_dataset['annotations'])}")
    print(f"    Synthetic: {len(synthetic_remapped['annotations'])}")

    # Print class distribution
    print("\nClass distribution in merged dataset:")
    category_counts = defaultdict(int)
    for ann in merged_annotations:
        category_counts[ann['category_id']] += 1

    for cat in merged_dataset['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        count = category_counts[cat_id]
        print(f"  {cat_name} (id={cat_id}): {count} annotations")

    return merged_dataset


def save_dataset(dataset: Dict, output_path: str):
    """Save dataset to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged dataset to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Merge synthetic and real COCO datasets"
    )
    parser.add_argument(
        '--synthetic-json',
        required=True,
        help='Synthetic dataset JSON file (e.g., synthetic_train.json)'
    )
    parser.add_argument(
        '--real-json',
        required=True,
        help='Real dataset JSON file (e.g., split_v8/train.json)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--synthetic-ratio',
        type=float,
        default=0.7,
        help='Fraction of synthetic data to use (default: 0.7)'
    )
    parser.add_argument(
        '--real-ratio',
        type=float,
        default=1.0,
        help='Fraction of real data to use (default: 1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    args = parser.parse_args()

    # Load datasets
    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    synthetic_dataset = load_coco_dataset(args.synthetic_json)
    real_dataset = load_coco_dataset(args.real_json)

    # Sample datasets
    print("\n" + "="*60)
    print("SAMPLING DATASETS")
    print("="*60)
    print("Synthetic dataset:")
    synthetic_sampled = sample_dataset(
        synthetic_dataset,
        args.synthetic_ratio,
        args.seed
    )

    print("\nReal dataset:")
    real_sampled = sample_dataset(
        real_dataset,
        args.real_ratio,
        args.seed + 1  # Different seed for real sampling
    )

    # Merge datasets
    print("\n" + "="*60)
    print("MERGING DATASETS")
    print("="*60)
    merged_dataset = merge_datasets(
        synthetic_sampled,
        real_sampled,
        output_name=f"mixed_{int(args.synthetic_ratio*100)}synth_"
                    f"{int(args.real_ratio*100)}real"
    )

    # Save merged dataset
    output_dir = Path(args.output_dir)
    train_path = output_dir / 'train.json'
    save_dataset(merged_dataset, train_path)

    # Summary
    print("\n" + "="*60)
    print("MERGE COMPLETE")
    print("="*60)
    print(f"Merged dataset: {train_path}")
    print(f"  {len(merged_dataset['images'])} images")
    print(f"  {len(merged_dataset['annotations'])} annotations")
    print(f"  Ratio: {args.synthetic_ratio*100:.0f}% synthetic, "
          f"{args.real_ratio*100:.0f}% real")


if __name__ == '__main__':
    main()
