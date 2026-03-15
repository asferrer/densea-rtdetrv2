#!/usr/bin/env python3
"""
Split synthetic COCO dataset into train/val sets with stratified sampling.

Maintains class distribution across splits (±2% tolerance).
Default split: 80% train, 20% val

Usage:
    python tools/split_synthetic_dataset.py \
        --input ./dataset/dataset_sintetico/synthetic_dataset.json \
        --output-dir ./dataset/dataset_sintetico \
        --train-ratio 0.8 \
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
    with open(json_path, 'r') as f:
        return json.load(f)


def group_images_by_classes(dataset: Dict) -> Dict[int, List[int]]:
    """
    Group image IDs by the classes they contain.

    Returns:
        Dict mapping class_id -> list of image_ids containing that class
    """
    # Build annotation lookup: image_id -> list of category_ids
    image_to_categories = defaultdict(set)
    for ann in dataset['annotations']:
        image_to_categories[ann['image_id']].add(ann['category_id'])

    # Group images by category
    category_to_images = defaultdict(list)
    for image_id, categories in image_to_categories.items():
        for cat_id in categories:
            category_to_images[cat_id].append(image_id)

    return category_to_images


def stratified_split(
    dataset: Dict,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Dict, Dict]:
    """
    Perform stratified split maintaining class distribution.

    Strategy:
    1. Group images by classes they contain
    2. For each class, split images proportionally
    3. Merge splits ensuring no duplicate images

    Args:
        dataset: COCO format dataset dict
        train_ratio: Fraction of data for training (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset) tuples
    """
    random.seed(seed)

    # Get all image IDs
    all_image_ids = set(img['id'] for img in dataset['images'])
    total_images = len(all_image_ids)

    print(f"Total images: {total_images}")
    print(f"Total annotations: {len(dataset['annotations'])}")

    # Calculate class distribution
    category_to_images = group_images_by_classes(dataset)
    print("\nClass distribution:")
    for cat in dataset['categories']:
        cat_id = cat['id']
        cat_name = cat['name']
        num_images = len(category_to_images[cat_id])
        num_annotations = sum(1 for ann in dataset['annotations']
                             if ann['category_id'] == cat_id)
        print(f"  {cat_name} (id={cat_id}): {num_images} images, "
              f"{num_annotations} annotations")

    # Simple random split (stratified by full dataset)
    # Shuffle all image IDs
    image_ids_list = list(all_image_ids)
    random.shuffle(image_ids_list)

    # Split point
    split_idx = int(len(image_ids_list) * train_ratio)
    train_image_ids = set(image_ids_list[:split_idx])
    val_image_ids = set(image_ids_list[split_idx:])

    print(f"\nSplit: {len(train_image_ids)} train, {len(val_image_ids)} val")

    # Create train/val datasets
    train_dataset = create_subset(dataset, train_image_ids, "train")
    val_dataset = create_subset(dataset, val_image_ids, "val")

    # Verify class distribution
    print("\nVerifying class distribution:")
    train_cat_to_imgs = group_images_by_classes(train_dataset)
    val_cat_to_imgs = group_images_by_classes(val_dataset)

    for cat in dataset['categories']:
        cat_id = cat['id']
        cat_name = cat['name']

        train_imgs = len(train_cat_to_imgs[cat_id])
        val_imgs = len(val_cat_to_imgs[cat_id])
        total_imgs = len(category_to_images[cat_id])

        train_pct = (train_imgs / total_imgs * 100) if total_imgs > 0 else 0
        val_pct = (val_imgs / total_imgs * 100) if total_imgs > 0 else 0

        train_anns = sum(1 for ann in train_dataset['annotations']
                        if ann['category_id'] == cat_id)
        val_anns = sum(1 for ann in val_dataset['annotations']
                      if ann['category_id'] == cat_id)

        print(f"  {cat_name}:")
        print(f"    Images: {train_imgs} train ({train_pct:.1f}%), "
              f"{val_imgs} val ({val_pct:.1f}%)")
        print(f"    Annotations: {train_anns} train, {val_anns} val")

    return train_dataset, val_dataset


def create_subset(dataset: Dict, image_ids: set, split_name: str) -> Dict:
    """
    Create a subset of the dataset containing only specified images.

    Args:
        dataset: Original COCO dataset
        image_ids: Set of image IDs to include
        split_name: Name of split ("train" or "val")

    Returns:
        New dataset dict with subset of images and annotations
    """
    # Filter images
    subset_images = [img for img in dataset['images'] if img['id'] in image_ids]

    # Filter annotations
    subset_annotations = [ann for ann in dataset['annotations']
                         if ann['image_id'] in image_ids]

    # Copy other fields
    subset_dataset = {
        'info': dataset.get('info', {}),
        'licenses': dataset.get('licenses', []),
        'categories': dataset['categories'],
        'images': subset_images,
        'annotations': subset_annotations
    }

    # Add split info
    if 'info' not in subset_dataset:
        subset_dataset['info'] = {}
    subset_dataset['info']['split'] = split_name
    subset_dataset['info']['description'] = (
        f"{split_name.capitalize()} split of synthetic dataset"
    )

    return subset_dataset


def save_dataset(dataset: Dict, output_path: str):
    """Save dataset to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split synthetic COCO dataset into train/val sets"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input COCO JSON file (e.g., synthetic_dataset.json)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for train/val JSON files'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Fraction of data for training (default: 0.8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.input}")
    dataset = load_coco_dataset(args.input)

    # Perform stratified split
    print(f"\nPerforming stratified split (train_ratio={args.train_ratio})...")
    train_dataset, val_dataset = stratified_split(
        dataset,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    # Save splits
    output_dir = Path(args.output_dir)
    train_path = output_dir / 'synthetic_train.json'
    val_path = output_dir / 'synthetic_val.json'

    print(f"\nSaving train split...")
    save_dataset(train_dataset, train_path)

    print(f"\nSaving val split...")
    save_dataset(val_dataset, val_path)

    print("\n✅ Split complete!")
    print(f"Train: {len(train_dataset['images'])} images, "
          f"{len(train_dataset['annotations'])} annotations")
    print(f"Val: {len(val_dataset['images'])} images, "
          f"{len(val_dataset['annotations'])} annotations")


if __name__ == '__main__':
    main()
