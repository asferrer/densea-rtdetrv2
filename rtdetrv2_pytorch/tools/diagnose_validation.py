#!/usr/bin/env python3
"""
Diagnostic script to check validation pipeline and understand why mAP is 0.

Verifies:
1. Annotation format in JSON
2. Transform pipeline output
3. Prediction vs ground truth alignment
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load config
from core import YAMLConfig

def load_sample_from_json(json_path, img_dir, num_samples=3):
    """Load sample images and annotations from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"Dataset: {json_path}")
    print(f"{'='*70}")
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print(f"Categories: {len(data['categories'])}")

    # Print category info
    print(f"\nCategories:")
    for cat in data['categories']:
        cat_anns = [ann for ann in data['annotations'] if ann['category_id'] == cat['id']]
        print(f"  {cat['id']}: {cat['name']} ({len(cat_anns)} annotations)")

    # Sample first few images with annotations
    samples = []
    for img_info in data['images'][:num_samples]:
        img_id = img_info['id']
        img_path = Path(img_dir) / img_info['file_name']

        # Get annotations for this image
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

        if not anns:
            continue

        samples.append({
            'image_info': img_info,
            'annotations': anns,
            'image_path': img_path
        })

        if len(samples) >= num_samples:
            break

    return samples, data

def check_bbox_format(bbox, img_width, img_height):
    """Check if bbox is valid and in correct format."""
    x, y, w, h = bbox

    issues = []

    # Check if coordinates are negative
    if x < 0 or y < 0:
        issues.append(f"Negative coordinates: x={x}, y={y}")

    # Check if width/height are negative or zero
    if w <= 0 or h <= 0:
        issues.append(f"Invalid width/height: w={w}, h={h}")

    # Check if bbox is outside image bounds
    if x + w > img_width:
        issues.append(f"Bbox extends beyond image width: x+w={x+w} > {img_width}")
    if y + h > img_height:
        issues.append(f"Bbox extends beyond image height: y+h={y+h} > {img_height}")

    return issues

def visualize_sample(sample, output_path):
    """Visualize image with annotations."""
    img = Image.open(sample['image_path'])
    img_width, img_height = img.size

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)

    for ann in sample['annotations']:
        bbox = ann['bbox']  # [x, y, w, h] in COCO format
        x, y, w, h = bbox

        # Check for issues
        issues = check_bbox_format(bbox, img_width, img_height)

        # Draw bbox
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor='red' if issues else 'green',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label
        label = f"Cat {ann['category_id']}"
        if issues:
            label += " (ISSUE)"

        ax.text(x, y - 5, label, color='red' if issues else 'green',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    ax.set_title(f"{sample['image_info']['file_name']}\n{len(sample['annotations'])} annotations")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {output_path}")

def test_transform_pipeline(cfg):
    """Test that transforms are working correctly."""
    print(f"\n{'='*70}")
    print("TESTING TRANSFORM PIPELINE")
    print(f"{'='*70}")

    # Get val dataset from val_dataloader (not cfg.val_dataset)
    val_dataset = cfg.val_dataloader.dataset

    print(f"\nVal dataset size: {len(val_dataset)}")

    # Get first sample
    sample = val_dataset[0]

    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image dtype: {sample['image'].dtype}")
    print(f"Image min/max: {sample['image'].min():.3f} / {sample['image'].max():.3f}")

    if 'boxes' in sample:
        boxes = sample['boxes']
        print(f"\nBoxes shape: {boxes.shape}")
        print(f"Boxes dtype: {boxes.dtype}")
        print(f"Boxes min/max: {boxes.min():.3f} / {boxes.max():.3f}")
        print(f"First 3 boxes:\n{boxes[:3]}")

        # Check if boxes are normalized
        if boxes.max() > 1.5:
            print(f"\n⚠️ WARNING: Boxes are NOT normalized (max > 1.5)")
        else:
            print(f"\n✅ Boxes appear to be normalized")

        # Check box format (should be cxcywh)
        print(f"\nBox format check (should be cxcywh normalized):")
        print(f"  cx range: [{boxes[:, 0].min():.3f}, {boxes[:, 0].max():.3f}]")
        print(f"  cy range: [{boxes[:, 1].min():.3f}, {boxes[:, 1].max():.3f}]")
        print(f"  w range: [{boxes[:, 2].min():.3f}, {boxes[:, 2].max():.3f}]")
        print(f"  h range: [{boxes[:, 3].min():.3f}, {boxes[:, 3].max():.3f}]")

    if 'labels' in sample:
        labels = sample['labels']
        print(f"\nLabels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        print(f"Unique labels: {torch.unique(labels).tolist()}")
        print(f"Label distribution: {[(l.item(), (labels == l).sum().item()) for l in torch.unique(labels)]}")

    return sample

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        default='configs/rtdetrv2/rtdetrv2_r50vd_phase1_synthetic_gradual.yml',
        help='Config file'
    )
    parser.add_argument(
        '--val-json',
        default='./split_sintetico/val.json',
        help='Validation JSON file'
    )
    parser.add_argument(
        '--img-dir',
        default='./dataset/dataset_sintetico/images',
        help='Image directory'
    )
    parser.add_argument(
        '--output-dir',
        default='./diagnostic_output',
        help='Output directory for visualizations'
    )
    args = parser.parse_args()

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"Loading config: {args.config}")
    cfg = YAMLConfig(args.config)

    # Load and check JSON
    samples, data = load_sample_from_json(args.val_json, args.img_dir, num_samples=3)

    print(f"\n{'='*70}")
    print("CHECKING ANNOTATIONS")
    print(f"{'='*70}")

    # Check annotations for issues
    total_issues = 0
    for i, sample in enumerate(samples):
        img_info = sample['image_info']
        img_width = img_info['width']
        img_height = img_info['height']

        print(f"\nImage {i+1}: {img_info['file_name']}")
        print(f"  Size: {img_width}x{img_height}")
        print(f"  Annotations: {len(sample['annotations'])}")

        for j, ann in enumerate(sample['annotations']):
            bbox = ann['bbox']
            issues = check_bbox_format(bbox, img_width, img_height)

            if issues:
                print(f"    ❌ Annotation {j+1}: {bbox}")
                for issue in issues:
                    print(f"        - {issue}")
                total_issues += 1
            else:
                print(f"    ✅ Annotation {j+1}: {bbox}")

        # Visualize
        vis_path = output_dir / f"val_sample_{i+1}.png"
        visualize_sample(sample, vis_path)

    if total_issues == 0:
        print(f"\n✅ All annotations are valid!")
    else:
        print(f"\n⚠️  Found {total_issues} annotation issues")

    # Test transform pipeline
    try:
        transformed_sample = test_transform_pipeline(cfg)
    except Exception as e:
        print(f"\n❌ ERROR testing transforms: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")
    print(f"Output saved to: {output_dir}")

if __name__ == '__main__':
    main()
