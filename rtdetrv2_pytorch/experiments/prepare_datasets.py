#!/usr/bin/env python3
"""
Dataset preparation for synthetic vs real data experiments.

This script creates the training datasets for all 9 experiments:
1. exp01_synth100: 100% synthetic
2. exp02_synth100_real10: 100% synthetic + 10% real (stratified)
3. exp03_synth100_real25: 100% synthetic + 25% real (stratified)
4. exp04_synth100_real50: 100% synthetic + 50% real (stratified)
5. exp05_synth100_real75: 100% synthetic + 75% real (stratified)
6. exp06_synth100_real100: 100% synthetic + 100% real
7. exp07_real100: 100% real only
8. exp08_real_synth_minor: 100% real + synthetic for Can/Metal_Debris only
9. exp09_real_synth_prop: 100% real + synthetic inversely proportional

Usage:
    python experiments/prepare_datasets.py [--experiment N] [--dry-run]
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
from copy import deepcopy

from config import (
    EXPERIMENTS,
    REAL_ONLY_EXPERIMENTS,
    ALL_EXPERIMENTS,
    REAL_TRAIN_JSON,
    REAL_VAL_JSON,
    REAL_TEST_JSON,
    CLASSES,
    CLASS_NAMES,
    MINORITY_CLASSES,
    MINORITY_CLASS_IDS,
    REAL_CLASS_COUNTS,
    SEED,
    # Versioned dataset support
    SYNTHETIC_DATASETS,
    DEFAULT_SYNTH_VERSION,
    get_synth_config,
    get_available_synth_versions,
    get_versioned_output_paths,
    experiment_uses_synthetic,
    get_real_only_output_paths,
    # Class filtering support
    needs_class_filtering,
    create_filtered_val_json_for_synth_version,
    create_filtered_test_json_for_synth_version,
)


# =============================================================================
# COCO DATASET I/O
# =============================================================================

def load_coco_dataset(json_path: str) -> Dict:
    """Load a COCO format dataset from JSON file."""
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"  Images: {len(dataset['images'])}")
    print(f"  Annotations: {len(dataset['annotations'])}")
    print(f"  Categories: {len(dataset['categories'])}")

    return dataset


def save_coco_dataset(dataset: Dict, output_path: str) -> None:
    """Save a COCO format dataset to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved dataset to: {output_path}")
    print(f"  Images: {len(dataset['images'])}")
    print(f"  Annotations: {len(dataset['annotations'])}")


def get_class_distribution(dataset: Dict) -> Dict[str, int]:
    """Get annotation count per class."""
    # Build category ID to name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in dataset['categories']}

    # Count annotations per category
    counts = defaultdict(int)
    for ann in dataset['annotations']:
        cat_name = cat_id_to_name.get(ann['category_id'], f"unknown_{ann['category_id']}")
        counts[cat_name] += 1

    return dict(counts)


def print_class_distribution(dataset: Dict, title: str = "") -> None:
    """Print class distribution summary."""
    dist = get_class_distribution(dataset)
    total = sum(dist.values())

    if title:
        print(f"\n{title}")
        print("-" * 50)

    for name, count in sorted(dist.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    print(f"  TOTAL: {total}")


# =============================================================================
# DATASET MANIPULATION
# =============================================================================

def get_images_by_category(dataset: Dict) -> Dict[int, Set[int]]:
    """
    Map category_id -> set of image_ids containing that category.

    Returns:
        Dict mapping category ID to set of image IDs
    """
    cat_to_images = defaultdict(set)

    for ann in dataset['annotations']:
        cat_to_images[ann['category_id']].add(ann['image_id'])

    return dict(cat_to_images)


def get_annotations_by_image(dataset: Dict) -> Dict[int, List[Dict]]:
    """
    Map image_id -> list of annotations for that image.

    Returns:
        Dict mapping image ID to list of annotation dicts
    """
    img_to_anns = defaultdict(list)

    for ann in dataset['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    return dict(img_to_anns)


def filter_dataset_by_images(dataset: Dict, image_ids: Set[int]) -> Dict:
    """
    Filter dataset to only include specified images and their annotations.

    Args:
        dataset: COCO format dataset
        image_ids: Set of image IDs to keep

    Returns:
        Filtered dataset
    """
    filtered_images = [img for img in dataset['images'] if img['id'] in image_ids]
    filtered_annotations = [ann for ann in dataset['annotations']
                           if ann['image_id'] in image_ids]

    return {
        'info': dataset.get('info', {}),
        'licenses': dataset.get('licenses', []),
        'categories': dataset['categories'],
        'images': filtered_images,
        'annotations': filtered_annotations,
    }


def create_stratified_subset(
    dataset: Dict,
    ratio: float,
    seed: int = SEED
) -> Dict:
    """
    Create a stratified subset of the dataset maintaining class proportions.

    Args:
        dataset: COCO format dataset
        ratio: Fraction of data to keep (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Subset of the dataset
    """
    if ratio >= 1.0:
        return deepcopy(dataset)

    if ratio <= 0.0:
        return {
            'info': dataset.get('info', {}),
            'licenses': dataset.get('licenses', []),
            'categories': dataset['categories'],
            'images': [],
            'annotations': [],
        }

    random.seed(seed)

    # Get images per category
    cat_to_images = get_images_by_category(dataset)
    all_image_ids = {img['id'] for img in dataset['images']}

    # For stratified sampling, we need to handle multi-label images
    # Strategy: Sample images proportionally from each category, avoiding duplicates

    # Calculate how many images to sample total
    total_images = len(all_image_ids)
    target_count = int(total_images * ratio)

    # Track selected images
    selected_images = set()

    # Sample proportionally from each category
    for cat_id, cat_images in cat_to_images.items():
        cat_images_list = list(cat_images)
        # Number to sample from this category (proportional to its size)
        cat_target = max(1, int(len(cat_images_list) * ratio))

        # Sample from images not yet selected
        available = [img_id for img_id in cat_images_list if img_id not in selected_images]
        if available:
            sample_size = min(cat_target, len(available))
            sampled = random.sample(available, sample_size)
            selected_images.update(sampled)

    # If we haven't reached target, add more random images
    if len(selected_images) < target_count:
        remaining = list(all_image_ids - selected_images)
        additional = min(target_count - len(selected_images), len(remaining))
        if additional > 0:
            selected_images.update(random.sample(remaining, additional))

    # If we have too many, randomly remove some (shouldn't happen often)
    if len(selected_images) > target_count:
        selected_images = set(random.sample(list(selected_images), target_count))

    print(f"  Stratified sampling: {len(selected_images)}/{total_images} images "
          f"({100*len(selected_images)/total_images:.1f}%)")

    return filter_dataset_by_images(dataset, selected_images)


def remap_dataset_ids(
    dataset: Dict,
    image_id_offset: int,
    ann_id_offset: int
) -> Dict:
    """
    Remap image and annotation IDs to avoid conflicts when merging.

    Args:
        dataset: COCO format dataset
        image_id_offset: Offset to add to image IDs
        ann_id_offset: Offset to add to annotation IDs

    Returns:
        Dataset with remapped IDs
    """
    dataset = deepcopy(dataset)

    # Create mapping of old -> new image IDs
    old_to_new_image = {}
    for img in dataset['images']:
        old_id = img['id']
        new_id = old_id + image_id_offset
        old_to_new_image[old_id] = new_id
        img['id'] = new_id

    # Remap annotations
    for ann in dataset['annotations']:
        ann['id'] = ann['id'] + ann_id_offset
        ann['image_id'] = old_to_new_image[ann['image_id']]

    return dataset


def align_categories(dataset: Dict, reference_categories: Dict[int, str]) -> Dict:
    """
    Remap dataset annotations to match reference categories by class name.

    Matches categories by NAME (not ID), remaps annotation IDs accordingly,
    and drops annotations for classes not present in the reference.

    Example: If reference has {4: 'Metal_Debris'} and dataset has
    {5: 'Metal_Debris'}, annotations with category_id=5 are remapped to 4.

    Args:
        dataset: COCO format dataset to filter and remap
        reference_categories: Dict of category_id -> name (target mapping)

    Returns:
        Dataset with remapped category IDs matching the reference
    """
    dataset = deepcopy(dataset)

    # Build name-based mappings
    ref_name_to_id = {name: cat_id for cat_id, name in reference_categories.items()}
    src_id_to_name = {cat['id']: cat['name'] for cat in dataset['categories']}

    # Build source_id -> reference_id remap (matched by class name)
    id_remap = {}
    matched_classes = []
    dropped_classes = []

    for src_id, src_name in src_id_to_name.items():
        if src_name in ref_name_to_id:
            id_remap[src_id] = ref_name_to_id[src_name]
            matched_classes.append(f"{src_name}: {src_id} -> {ref_name_to_id[src_name]}")
        else:
            dropped_classes.append(f"{src_name} (ID {src_id})")

    if matched_classes:
        print(f"    Category remapping:")
        for m in matched_classes:
            print(f"      {m}")
    if dropped_classes:
        print(f"    Dropping classes not in reference: {', '.join(dropped_classes)}")

    # Filter and remap annotations
    valid_src_ids = set(id_remap.keys())
    original_ann_count = len(dataset['annotations'])
    new_annotations = []

    for ann in dataset['annotations']:
        if ann['category_id'] in valid_src_ids:
            ann['category_id'] = id_remap[ann['category_id']]
            new_annotations.append(ann)

    dataset['annotations'] = new_annotations
    filtered_ann_count = len(new_annotations)

    if filtered_ann_count < original_ann_count:
        removed = original_ann_count - filtered_ann_count
        print(f"    Filtered out {removed} annotations with incompatible categories")

    # Remove images that no longer have annotations
    image_ids_with_anns = {ann['image_id'] for ann in dataset['annotations']}
    original_img_count = len(dataset['images'])
    dataset['images'] = [
        img for img in dataset['images']
        if img['id'] in image_ids_with_anns
    ]
    filtered_img_count = len(dataset['images'])

    if filtered_img_count < original_img_count:
        removed = original_img_count - filtered_img_count
        print(f"    Removed {removed} images with no remaining annotations")

    # Update categories to match reference
    dataset['categories'] = [
        {'id': cat_id, 'name': name}
        for cat_id, name in sorted(reference_categories.items())
    ]

    return dataset


def merge_datasets(
    dataset_a: Dict,
    dataset_b: Dict,
    name: str = "merged",
    source_a: str = "synthetic",
    source_b: str = "real"
) -> Dict:
    """
    Merge two COCO datasets, handling ID conflicts and category alignment.

    Args:
        dataset_a: First dataset (IDs kept as-is, categories used as reference)
        dataset_b: Second dataset (IDs will be offset, filtered to match dataset_a categories)
        name: Name for the merged dataset
        source_a: Source domain label for dataset_a images (default: "synthetic")
        source_b: Source domain label for dataset_b images (default: "real")

    Returns:
        Merged dataset with "source" field on each image for balanced sampling
    """
    # Make copies to avoid modifying originals
    dataset_a = deepcopy(dataset_a)
    dataset_b = deepcopy(dataset_b)

    # Tag images with their source domain for balanced batch sampling
    for img in dataset_a['images']:
        img['source'] = source_a
    for img in dataset_b['images']:
        img['source'] = source_b

    # Verify categories match
    cats_a = {cat['id']: cat['name'] for cat in dataset_a['categories']}
    cats_b = {cat['id']: cat['name'] for cat in dataset_b['categories']}

    if cats_a != cats_b:
        print(f"  Category alignment needed:")
        print(f"    Dataset A ({len(cats_a)} classes): {list(cats_a.values())}")
        print(f"    Dataset B ({len(cats_b)} classes): {list(cats_b.values())}")
        print(f"    Using Dataset A categories as reference")

        # Align dataset_b to match dataset_a categories
        dataset_b = align_categories(dataset_b, cats_a)

    # Calculate offsets for dataset_b
    max_image_id_a = max((img['id'] for img in dataset_a['images']), default=0)
    max_ann_id_a = max((ann['id'] for ann in dataset_a['annotations']), default=0)

    # Remap dataset_b IDs
    dataset_b = remap_dataset_ids(
        dataset_b,
        image_id_offset=max_image_id_a + 1,
        ann_id_offset=max_ann_id_a + 1
    )

    # Merge
    merged = {
        'info': {
            'description': f'{name} (merged dataset)',
            'contributor': 'RT-DETRv2-DenSea Experiments',
        },
        'licenses': dataset_a.get('licenses', []),
        'categories': dataset_a['categories'],
        'images': dataset_a['images'] + dataset_b['images'],
        'annotations': dataset_a['annotations'] + dataset_b['annotations'],
    }

    print(f"  Merged: {len(dataset_a['images'])} + {len(dataset_b['images'])} = "
          f"{len(merged['images'])} images")

    return merged


# =============================================================================
# SPECIALIZED DATASET CREATION FOR EXPERIMENTS 8 AND 9
# =============================================================================

def create_minority_balanced_dataset(
    real_dataset: Dict,
    synth_dataset: Dict,
    minority_classes: List[str] = MINORITY_CLASSES,
    target_per_class: int = 3500,
    seed: int = SEED
) -> Dict:
    """
    Create dataset with 100% real + synthetic only for minority classes.

    This is for Experiment 8: Add synthetic data only for Can and Metal_Debris
    to balance these severely underrepresented classes.

    Args:
        real_dataset: Full real dataset
        synth_dataset: Full synthetic dataset
        minority_classes: List of class names to augment (default: Can, Metal_Debris)
        target_per_class: Target annotation count per minority class
        seed: Random seed

    Returns:
        Combined dataset
    """
    random.seed(seed)

    # Get category IDs for minority classes
    cat_name_to_id_synth = {cat['name']: cat['id'] for cat in synth_dataset['categories']}

    # Get synthetic images containing minority classes
    synth_cat_to_images = get_images_by_category(synth_dataset)
    synth_img_to_anns = get_annotations_by_image(synth_dataset)

    selected_synth_images = set()

    for class_name in minority_classes:
        if class_name not in cat_name_to_id_synth:
            print(f"  WARNING: Class '{class_name}' not found in synthetic dataset")
            continue

        cat_id = cat_name_to_id_synth[class_name]
        real_count = REAL_CLASS_COUNTS.get(class_name, 0)
        needed = max(0, target_per_class - real_count)

        print(f"  {class_name}: real={real_count}, target={target_per_class}, "
              f"need={needed} synthetic annotations")

        if needed <= 0:
            continue

        # Get images with this class from synthetic data
        synth_images_with_class = list(synth_cat_to_images.get(cat_id, set()))
        random.shuffle(synth_images_with_class)

        # Add images until we have enough annotations
        added_annotations = 0
        for img_id in synth_images_with_class:
            if added_annotations >= needed:
                break

            # Count annotations of this class in this image
            img_anns = synth_img_to_anns.get(img_id, [])
            class_anns = [ann for ann in img_anns if ann['category_id'] == cat_id]

            selected_synth_images.add(img_id)
            added_annotations += len(class_anns)

        print(f"    Added {added_annotations} annotations from "
              f"{len([i for i in selected_synth_images if i in synth_cat_to_images.get(cat_id, set())])} images")

    # Filter synthetic dataset to selected images
    synth_subset = filter_dataset_by_images(synth_dataset, selected_synth_images)

    # Merge: synth as dataset_a (reference categories) so num_classes matches config
    merged = merge_datasets(synth_subset, real_dataset, name="real_synth_minority")

    return merged


def create_inverse_proportional_dataset(
    real_dataset: Dict,
    synth_dataset: Dict,
    target_total: int = 4500,
    seed: int = SEED
) -> Dict:
    """
    Create dataset with 100% real + synthetic inversely proportional to real count.

    This is for Experiment 9: Add more synthetic data for classes with fewer
    real annotations, proportional to the deficit.

    Args:
        real_dataset: Full real dataset
        synth_dataset: Full synthetic dataset
        target_total: Target annotation count per class
        seed: Random seed

    Returns:
        Combined dataset
    """
    random.seed(seed)

    # Get current real class distribution
    real_dist = get_class_distribution(real_dataset)

    # Calculate how much synthetic to add per class
    synth_to_add = {}
    for class_name in CLASS_NAMES:
        real_count = real_dist.get(class_name, 0)
        deficit = max(0, target_total - real_count)
        synth_to_add[class_name] = deficit

    print("  Synthetic annotations to add per class:")
    for class_name, count in sorted(synth_to_add.items(), key=lambda x: -x[1]):
        print(f"    {class_name}: +{count}")

    # Get synthetic category mapping
    cat_name_to_id_synth = {cat['name']: cat['id'] for cat in synth_dataset['categories']}
    synth_cat_to_images = get_images_by_category(synth_dataset)
    synth_img_to_anns = get_annotations_by_image(synth_dataset)

    selected_synth_images = set()

    # Process classes from most needed to least
    for class_name, needed in sorted(synth_to_add.items(), key=lambda x: -x[1]):
        if needed <= 0:
            continue

        if class_name not in cat_name_to_id_synth:
            print(f"  WARNING: Class '{class_name}' not found in synthetic dataset")
            continue

        cat_id = cat_name_to_id_synth[class_name]

        # Get images with this class from synthetic data
        synth_images_with_class = list(synth_cat_to_images.get(cat_id, set()))
        # Prioritize images not already selected
        unselected = [i for i in synth_images_with_class if i not in selected_synth_images]
        already_selected = [i for i in synth_images_with_class if i in selected_synth_images]

        random.shuffle(unselected)

        # Add images until we have enough annotations
        added_annotations = 0
        for img_id in unselected:
            if added_annotations >= needed:
                break

            img_anns = synth_img_to_anns.get(img_id, [])
            class_anns = [ann for ann in img_anns if ann['category_id'] == cat_id]

            selected_synth_images.add(img_id)
            added_annotations += len(class_anns)

    print(f"  Total synthetic images selected: {len(selected_synth_images)}")

    # Filter synthetic dataset to selected images
    synth_subset = filter_dataset_by_images(synth_dataset, selected_synth_images)

    # Merge: synth as dataset_a (reference categories) so num_classes matches config
    merged = merge_datasets(synth_subset, real_dataset, name="real_synth_proportional")

    return merged


# =============================================================================
# MAIN DATASET GENERATION
# =============================================================================

def generate_experiment_dataset(
    exp: Dict,
    real_dataset: Dict,
    synth_dataset: Dict
) -> Dict:
    """
    Generate the training dataset for a single experiment.

    Args:
        exp: Experiment configuration dict
        real_dataset: Full real dataset
        synth_dataset: Full synthetic dataset

    Returns:
        Generated dataset for this experiment
    """
    exp_name = exp['name']
    strategy = exp['strategy']
    synth_ratio = exp['synth_ratio']
    real_ratio = exp['real_ratio']

    print(f"\n{'='*60}")
    print(f"Generating dataset for: {exp_name}")
    print(f"Strategy: {strategy}, Synth: {synth_ratio}, Real: {real_ratio}")
    print(f"{'='*60}")

    # Experiment 1: 100% synthetic only
    if exp['id'] == 1:
        result = deepcopy(synth_dataset)
        for img in result['images']:
            img['source'] = 'synthetic'
        return result

    # Experiment 7: 100% real only
    if exp['id'] == 7:
        result = deepcopy(real_dataset)
        for img in result['images']:
            img['source'] = 'real'
        return result

    # Real-only experiments with stratified sampling (exp11-exp14)
    # These have synth_ratio=0.0 and real_ratio < 1.0
    if synth_ratio == 0.0 and real_ratio < 1.0:
        print(f"Creating stratified real-only subset with {real_ratio*100:.0f}% of real data...")
        result = create_stratified_subset(real_dataset, real_ratio)
        for img in result['images']:
            img['source'] = 'real'
        return result

    # Experiment 8: Real + synthetic for minority classes only
    if exp['id'] == 8:
        return create_minority_balanced_dataset(real_dataset, synth_dataset)

    # Experiment 9: Real + synthetic inversely proportional
    if exp['id'] == 9:
        return create_inverse_proportional_dataset(real_dataset, synth_dataset)

    # Experiments 2-6: Synthetic + stratified real
    if strategy == "stratified":
        print(f"Creating stratified subset with {real_ratio*100:.0f}% of real data...")
        real_subset = create_stratified_subset(real_dataset, real_ratio)
        print(f"Merging with 100% synthetic data...")
        return merge_datasets(synth_dataset, real_subset, name=exp_name)

    # Experiment 6: All synthetic + all real
    if strategy == "none" and synth_ratio == 1.0 and real_ratio == 1.0:
        return merge_datasets(synth_dataset, real_dataset, name=exp_name)

    raise ValueError(f"Unknown experiment configuration: {exp}")


def generate_all_datasets(
    experiments: Optional[List[int]] = None,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    dry_run: bool = False,
    real_only: bool = False
) -> Dict[str, str]:
    """
    Generate datasets for all (or specified) experiments.

    Args:
        experiments: List of experiment IDs to generate (None = all)
        synth_version: Synthetic dataset version to use ('cleansea', 'densea', 'taco')
        dry_run: If True, don't save files, just print what would be done
        real_only: If True, generate only real-only baseline experiments

    Returns:
        Dict mapping experiment name to output path
    """
    # Select experiment list based on mode
    if real_only:
        exp_list = REAL_ONLY_EXPERIMENTS
        print("\n" + "="*60)
        print("GENERATING REAL-ONLY BASELINE DATASETS")
        print("="*60)
    else:
        exp_list = ALL_EXPERIMENTS
        print("\n" + "="*60)
        print(f"GENERATING DATASETS [version: {synth_version}]")
        print("="*60)

    # Load source datasets
    print("\n" + "="*60)
    print("LOADING SOURCE DATASETS")
    print("="*60)

    try:
        real_dataset = load_coco_dataset(REAL_TRAIN_JSON)
        print_class_distribution(real_dataset, "Real data class distribution:")
    except FileNotFoundError:
        print(f"ERROR: Real dataset not found at {REAL_TRAIN_JSON}")
        return {}

    # Only load synthetic dataset if needed (not in real_only mode)
    synth_dataset = None
    if not real_only:
        # Get version-specific synthetic dataset path
        synth_config = get_synth_config(synth_version)
        synth_json_path = synth_config['train_json']

        print(f"\nSynthetic dataset version: {synth_version} ({synth_config['name']})")

        try:
            synth_dataset = load_coco_dataset(synth_json_path)
            print_class_distribution(synth_dataset, f"Synthetic data distribution [{synth_version}]:")
        except FileNotFoundError:
            print(f"ERROR: Synthetic dataset not found at {synth_json_path}")
            print(f"Available versions: {get_available_synth_versions()}")
            return {}

        # Create filtered validation and test files if needed (when synthetic has fewer classes)
        if needs_class_filtering(synth_version) and not dry_run:
            print("\n" + "-"*60)
            print("CREATING FILTERED VALIDATION FILE")
            print("-"*60)
            create_filtered_val_json_for_synth_version(REAL_VAL_JSON, synth_version)

            print("\n" + "-"*60)
            print("CREATING FILTERED TEST FILE")
            print("-"*60)
            create_filtered_test_json_for_synth_version(REAL_TEST_JSON, synth_version)

    # Get versioned output paths
    versioned_paths = get_versioned_output_paths(synth_version)

    # Generate datasets
    output_paths = {}

    for exp in exp_list:
        if experiments is not None and exp['id'] not in experiments:
            continue

        try:
            dataset = generate_experiment_dataset(exp, real_dataset, synth_dataset)

            # Determine output path based on experiment type
            if experiment_uses_synthetic(exp):
                # Versioned path for experiments using synthetic data
                output_path = Path(versioned_paths['datasets']) / exp['name'] / "train.json"
            else:
                # Shared path for real-only experiments
                real_only_paths = get_real_only_output_paths(exp['name'])
                output_path = real_only_paths['dataset']

            output_paths[exp['name']] = str(output_path)

            if not dry_run:
                save_coco_dataset(dataset, output_path)
            else:
                print(f"[DRY RUN] Would save to: {output_path}")

            print_class_distribution(dataset, f"Final distribution for {exp['name']}:")

        except Exception as e:
            print(f"ERROR generating dataset for {exp['name']}: {e}")
            import traceback
            traceback.print_exc()

    return output_paths


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training datasets for synthetic vs real experiments"
    )
    parser.add_argument(
        '--experiment', '-e',
        type=int,
        nargs='+',
        help='Experiment ID(s) to generate. Default: all'
    )
    parser.add_argument(
        '--synth-version', '-v',
        choices=get_available_synth_versions(),
        default=DEFAULT_SYNTH_VERSION,
        help=f'Synthetic dataset version to use (default: {DEFAULT_SYNTH_VERSION})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't save files, just print what would be done"
    )
    parser.add_argument(
        '--real-only',
        action='store_true',
        help="Generate only real-only baseline experiments (exp11-exp14)"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List all experiments and exit"
    )
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help="List available synthetic dataset versions and exit"
    )

    args = parser.parse_args()

    if args.list_versions:
        print("\nAvailable synthetic dataset versions:")
        print("-" * 70)
        for version, config in SYNTHETIC_DATASETS.items():
            print(f"  {version}: {config['name']}")
            print(f"    Path: {config['train_json']}")
            print(f"    {config['description']}")
        return

    if args.list:
        print("\nSynthetic + Real experiments:")
        print("-" * 70)
        for exp in EXPERIMENTS:
            print(f"  {exp['id']:2d}. {exp['name']}: {exp['description']}")
        print("\nReal-only baseline experiments:")
        print("-" * 70)
        for exp in REAL_ONLY_EXPERIMENTS:
            print(f"  {exp['id']:2d}. {exp['name']}: {exp['description']}")
        return

    print("\n" + "="*70)
    print("DATASET PREPARATION FOR SYNTHETIC VS REAL EXPERIMENTS")
    print(f"Synthetic version: {args.synth_version}")
    print("="*70)

    output_paths = generate_all_datasets(
        experiments=args.experiment,
        synth_version=args.synth_version,
        dry_run=args.dry_run,
        real_only=args.real_only
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if output_paths:
        print("\nGenerated datasets:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")
    else:
        print("\nNo datasets generated.")

    print("\nNext step: Run 'python experiments/generate_configs.py' to create YAML configs")


if __name__ == '__main__':
    main()
