#!/usr/bin/env python3
"""
Visualize model predictions on synthetic validation set.

This script helps diagnose what the model is actually learning by:
1. Loading a trained model checkpoint
2. Running inference on validation images
3. Visualizing predictions (bboxes, scores, classes) overlaid on images
4. Saving visualizations for manual inspection

Usage:
    python tools/visualize_predictions.py \
        --config configs/rtdetrv2/rtdetrv2_r50vd_phase1_synthetic_gradual_simple.yml \
        --checkpoint output/phase1_synthetic_gradual_simple/best_model.pth \
        --num-samples 50 \
        --output-dir visualizations/synthetic_predictions \
        --score-threshold 0.01  # Very low threshold to see all predictions
"""

import argparse
import os
import sys
from pathlib import Path
import json
import random

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import YAMLConfig


# Category names (0-indexed)
CATEGORIES = {
    0: 'Bottle',
    1: 'Can',
    2: 'Fishing_Net',
    3: 'Glove',
    4: 'Metal_Debris',
    5: 'Plastic_Debris'
}

# Colors for visualization (one per class)
COLORS = {
    0: (255, 0, 0),      # Bottle - Red
    1: (0, 255, 0),      # Can - Green
    2: (0, 0, 255),      # Fishing_Net - Blue
    3: (255, 255, 0),    # Glove - Yellow
    4: (255, 0, 255),    # Metal_Debris - Magenta
    5: (0, 255, 255),    # Plastic_Debris - Cyan
}


def load_model(config_path, checkpoint_path, device='cuda'):
    """Load trained RT-DETR model from checkpoint."""
    print(f"Loading config from: {config_path}")
    cfg = YAMLConfig(config_path)

    print(f"Building model...")
    model = cfg.model.to(device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        print("Using EMA weights")
        model.load_state_dict(checkpoint['ema']['module'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully")
    return model, cfg


def load_validation_samples(ann_file, img_folder, num_samples=50, seed=42):
    """Load random samples from validation set."""
    random.seed(seed)

    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)

    # Sample random images
    images = coco_data['images']
    if num_samples < len(images):
        sampled_images = random.sample(images, num_samples)
    else:
        sampled_images = images

    # Build annotation lookup
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_anns:
            image_id_to_anns[img_id] = []
        image_id_to_anns[img_id].append(ann)

    # Create samples list
    samples = []
    for img_info in sampled_images:
        img_path = os.path.join(img_folder, img_info['file_name'])
        if os.path.exists(img_path):
            samples.append({
                'image_path': img_path,
                'image_info': img_info,
                'annotations': image_id_to_anns.get(img_info['id'], [])
            })

    print(f"Loaded {len(samples)} validation samples")
    return samples


def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocess image for inference."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size

    # Resize maintaining aspect ratio
    img_resized = img.resize(target_size, Image.BILINEAR)

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    return img_tensor, (orig_w, orig_h), img


def run_inference(model, img_tensor, device='cuda'):
    """Run inference on preprocessed image."""
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
    return outputs


def postprocess_outputs(outputs, orig_size, target_size=(640, 640), score_threshold=0.05):
    """
    Postprocess model outputs to get final predictions.

    Returns:
        List of dicts with keys: bbox, score, label
    """
    # RT-DETR outputs: dict with 'pred_boxes' and 'pred_logits'
    # pred_boxes: [batch, num_queries, 4] in normalized cxcywh format
    # pred_logits: [batch, num_queries, num_classes]

    pred_boxes = outputs['pred_boxes'][0]  # [num_queries, 4]
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]

    # Get class probabilities
    pred_scores = pred_logits.sigmoid()  # [num_queries, num_classes]

    # Get max score and class for each query
    max_scores, pred_labels = pred_scores.max(dim=-1)  # [num_queries]

    # Filter by score threshold
    keep = max_scores > score_threshold

    boxes = pred_boxes[keep]  # [num_keep, 4]
    scores = max_scores[keep]  # [num_keep]
    labels = pred_labels[keep]  # [num_keep]

    # Convert boxes from normalized cxcywh to xyxy in original image coordinates
    orig_w, orig_h = orig_size

    predictions = []
    for box, score, label in zip(boxes, scores, labels):
        cx, cy, w, h = box.cpu().numpy()

        # Denormalize
        cx *= orig_w
        cy *= orig_h
        w *= orig_w
        h *= orig_h

        # Convert to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        predictions.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'score': float(score),
            'label': int(label)
        })

    return predictions


def visualize_predictions(image_path, predictions, ground_truth, output_path,
                         score_threshold=0.05):
    """
    Visualize predictions vs ground truth.

    Creates side-by-side visualization:
    - Left: Ground truth annotations
    - Right: Model predictions
    """
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Left subplot: Ground Truth
    ax1.imshow(img)
    ax1.set_title(f'Ground Truth ({len(ground_truth)} objects)', fontsize=16, fontweight='bold')
    ax1.axis('off')

    for ann in ground_truth:
        bbox = ann['bbox']  # COCO format [x, y, w, h]
        cat_id = ann['category_id']

        x, y, w, h = bbox
        color = [c/255.0 for c in COLORS[cat_id]]

        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                edgecolor=color, facecolor='none')
        ax1.add_patch(rect)

        # Add label
        ax1.text(x, y - 5, CATEGORIES[cat_id],
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                fontsize=10, color='white', fontweight='bold')

    # Right subplot: Predictions
    ax2.imshow(img)

    # Filter predictions by score threshold
    filtered_preds = [p for p in predictions if p['score'] >= score_threshold]

    ax2.set_title(f'Predictions ({len(filtered_preds)} objects, threshold={score_threshold:.2f})',
                 fontsize=16, fontweight='bold')
    ax2.axis('off')

    for pred in filtered_preds:
        x1, y1, x2, y2 = pred['bbox']
        label = pred['label']
        score = pred['score']

        color = [c/255.0 for c in COLORS[label]]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                edgecolor=color, facecolor='none')
        ax2.add_patch(rect)

        # Add label with score
        text = f"{CATEGORIES[label]} {score:.2f}"
        ax2.text(x1, y1 - 5, text,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                fontsize=10, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--val-images', default='./dataset/dataset_sintetico/images',
                       help='Path to validation images directory')
    parser.add_argument('--val-annotations', default='./split_sintetico/val.json',
                       help='Path to validation annotations JSON')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of validation samples to visualize')
    parser.add_argument('--output-dir', default='./visualizations/synthetic_predictions',
                       help='Output directory for visualizations')
    parser.add_argument('--score-threshold', type=float, default=0.01,
                       help='Score threshold for visualization (use low value to see all predictions)')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, cfg = load_model(args.config, args.checkpoint, device)

    # Load validation samples
    samples = load_validation_samples(args.val_annotations, args.val_images,
                                     args.num_samples, args.seed)

    # Run inference and visualize
    print(f"\nProcessing {len(samples)} samples...")

    all_predictions = []
    all_scores = []

    for i, sample in enumerate(samples):
        print(f"Processing {i+1}/{len(samples)}: {sample['image_info']['file_name']}", end='\r')

        # Preprocess
        img_tensor, orig_size, orig_img = preprocess_image(sample['image_path'])

        # Inference
        outputs = run_inference(model, img_tensor, device)

        # Postprocess
        predictions = postprocess_outputs(outputs, orig_size,
                                         score_threshold=args.score_threshold)

        # Collect statistics
        all_predictions.extend(predictions)
        all_scores.extend([p['score'] for p in predictions])

        # Visualize
        output_filename = f"pred_{i:04d}_{sample['image_info']['file_name']}"
        output_path = os.path.join(args.output_dir, output_filename.replace('.jpg', '.png'))

        visualize_predictions(sample['image_path'], predictions, sample['annotations'],
                            output_path, args.score_threshold)

    print(f"\n\nVisualization complete! Saved to: {args.output_dir}")

    # Print statistics
    print(f"\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Total images processed: {len(samples)}")
    print(f"Total predictions (score > {args.score_threshold}): {len(all_predictions)}")
    print(f"Average predictions per image: {len(all_predictions)/len(samples):.2f}")

    if all_scores:
        print(f"\nScore distribution:")
        print(f"  Min score:  {min(all_scores):.4f}")
        print(f"  Max score:  {max(all_scores):.4f}")
        print(f"  Mean score: {np.mean(all_scores):.4f}")
        print(f"  Median score: {np.median(all_scores):.4f}")

        # Count by class
        class_counts = {}
        for pred in all_predictions:
            label = pred['label']
            class_counts[label] = class_counts.get(label, 0) + 1

        print(f"\nPredictions by class:")
        for label, count in sorted(class_counts.items()):
            print(f"  {CATEGORIES[label]:15s}: {count:4d} ({count/len(all_predictions)*100:.1f}%)")
    else:
        print(f"\n⚠️  WARNING: No predictions found with score > {args.score_threshold}")
        print(f"   Try lowering --score-threshold (current: {args.score_threshold})")

    print(f"\n✅ Done! Check visualizations in: {args.output_dir}")


if __name__ == '__main__':
    main()
