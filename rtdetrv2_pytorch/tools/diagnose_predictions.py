#!/usr/bin/env python3
"""
Deep diagnostic script to understand why mAP=0 despite losses decreasing.

Checks:
1. Model prediction format and values
2. Ground truth format after transforms
3. Postprocessor output
4. What COCO evaluator actually receives
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


def inspect_model_predictions(cfg, num_samples=3):
    """Inspect raw model outputs vs postprocessed predictions."""
    print(f"\n{'='*70}")
    print("INSPECTING MODEL PREDICTIONS")
    print(f"{'='*70}")

    # Get model and set to eval mode
    model = cfg.model
    model.eval()
    model.to(cfg.device)

    # Get postprocessor
    postprocessor = cfg.postprocessor

    # Get val dataloader
    val_loader = cfg.val_dataloader

    print(f"\nModel: {type(model).__name__}")
    print(f"Postprocessor: {type(postprocessor).__name__}")
    print(f"Device: {cfg.device}")

    # Get a few batches
    for batch_idx, (images, targets) in enumerate(val_loader):
        if batch_idx >= num_samples:
            break

        images = images.to(cfg.device)

        print(f"\n{'='*70}")
        print(f"Batch {batch_idx + 1}")
        print(f"{'='*70}")

        print(f"\nInput:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")

        print(f"\nTargets ({len(targets)} images):")
        for i, target in enumerate(targets[:3]):  # Show first 3
            print(f"  Image {i}:")
            print(f"    boxes shape: {target['boxes'].shape}")
            print(f"    boxes range: [{target['boxes'].min():.3f}, {target['boxes'].max():.3f}]")
            print(f"    labels: {target['labels'].tolist()}")
            print(f"    labels range: [{target['labels'].min()}, {target['labels'].max()}]")
            if target['boxes'].shape[0] > 0:
                print(f"    First box: {target['boxes'][0].tolist()}")

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        print(f"\nRaw Model Outputs:")
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
                else:
                    print(f"  {key}: {type(value)}")

        # Get predictions from last decoder layer
        if 'pred_logits' in outputs and 'pred_boxes' in outputs:
            pred_logits = outputs['pred_logits']  # [batch, num_queries, num_classes]
            pred_boxes = outputs['pred_boxes']    # [batch, num_queries, 4]

            print(f"\nDecoder Outputs (before postprocessing):")
            print(f"  pred_logits shape: {pred_logits.shape}")
            print(f"  pred_boxes shape: {pred_boxes.shape}")
            print(f"  pred_logits range: [{pred_logits.min():.3f}, {pred_logits.max():.3f}]")
            print(f"  pred_boxes range: [{pred_boxes.min():.3f}, {pred_boxes.max():.3f}]")

            # Get scores
            scores = pred_logits.sigmoid()
            print(f"  scores (after sigmoid) range: [{scores.min():.3f}, {scores.max():.3f}]")
            print(f"  max score per query: {scores.max(dim=-1)[0][0, :10].tolist()}")  # First 10 queries

            # Check if ANY query has high confidence
            max_score_per_class = scores.max(dim=1)[0]  # [batch, num_classes]
            print(f"  max score per class: {max_score_per_class[0].tolist()}")

            high_conf_queries = (scores > 0.3).sum()
            print(f"  queries with score > 0.3: {high_conf_queries.item()}")

        # Postprocess
        orig_target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(cfg.device)
        results = postprocessor(outputs, orig_target_sizes)

        print(f"\nPostprocessed Predictions:")
        for i, result in enumerate(results[:3]):  # Show first 3
            print(f"  Image {i}:")
            print(f"    labels shape: {result['labels'].shape}")
            print(f"    labels: {result['labels'].tolist() if result['labels'].shape[0] > 0 else []}")
            print(f"    scores shape: {result['scores'].shape}")
            print(f"    scores: {result['scores'].tolist() if result['scores'].shape[0] > 0 else []}")
            print(f"    boxes shape: {result['boxes'].shape}")
            if result['boxes'].shape[0] > 0:
                print(f"    boxes range: [{result['boxes'].min():.3f}, {result['boxes'].max():.3f}]")
                print(f"    First box: {result['boxes'][0].tolist()}")
                print(f"    First score: {result['scores'][0].item():.4f}")
            else:
                print(f"    ⚠️  NO DETECTIONS")


def inspect_ground_truth_format(cfg):
    """Check exact format of ground truth that evaluator receives."""
    print(f"\n{'='*70}")
    print("INSPECTING GROUND TRUTH FORMAT FOR EVALUATOR")
    print(f"{'='*70}")

    # Get val dataloader
    val_loader = cfg.val_dataloader

    # Get one batch
    images, targets = next(iter(val_loader))

    print(f"\nGround Truth Format (what evaluator receives):")
    for i, target in enumerate(targets[:3]):
        print(f"\nImage {i}:")
        print(f"  Keys: {target.keys()}")

        if 'boxes' in target:
            boxes = target['boxes']
            print(f"  boxes:")
            print(f"    Type: {type(boxes)}")
            print(f"    Shape: {boxes.shape}")
            print(f"    Dtype: {boxes.dtype}")
            print(f"    Range: [{boxes.min():.3f}, {boxes.max():.3f}]")
            print(f"    Format: {'NORMALIZED cxcywh' if boxes.max() <= 1.5 else 'PIXEL coordinates'}")
            if boxes.shape[0] > 0:
                print(f"    First box: {boxes[0].tolist()}")

        if 'labels' in target:
            labels = target['labels']
            print(f"  labels:")
            print(f"    Values: {labels.tolist()}")
            print(f"    Range: [{labels.min()}, {labels.max()}]")
            print(f"    Dtype: {labels.dtype}")

        if 'image_id' in target:
            print(f"  image_id: {target['image_id']}")

        if 'orig_size' in target:
            print(f"  orig_size: {target['orig_size']}")


def check_evaluator_input(cfg):
    """Check what format the evaluator expects vs what it receives."""
    print(f"\n{'='*70}")
    print("CHECKING EVALUATOR INPUT FORMAT")
    print(f"{'='*70}")

    from data import get_coco_api_from_dataset

    # Get COCO ground truth
    val_dataset = cfg.val_dataloader.dataset
    coco_gt = get_coco_api_from_dataset(val_dataset)

    print(f"\nCOCO GT Categories:")
    for cat in coco_gt.dataset['categories']:
        print(f"  {cat}")

    # Get one annotation
    ann_ids = coco_gt.getAnnIds(imgIds=[coco_gt.imgs[list(coco_gt.imgs.keys())[0]]['id']])
    anns = coco_gt.loadAnns(ann_ids)

    print(f"\nSample GT Annotation (from COCO API):")
    for ann in anns[:3]:
        print(f"  {ann}")

    print(f"\nExpected format:")
    print(f"  bbox: [x, y, width, height] in PIXELS (COCO format)")
    print(f"  category_id: 1-indexed (COCO standard)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        default='configs/rtdetrv2/rtdetrv2_r50vd_phase1_synthetic_gradual.yml',
        help='Config file'
    )
    parser.add_argument(
        '-r', '--resume',
        help='Checkpoint to load'
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.config}")
    cfg = YAMLConfig(args.config)

    # Load checkpoint if provided
    if args.resume:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        # Access model to trigger creation, then load weights
        model = cfg.model
        model.load_state_dict(checkpoint['model'])
    else:
        # Just access model to ensure it's created
        model = cfg.model

    # Set model to eval mode
    cfg.model.eval()

    # Run diagnostics
    inspect_ground_truth_format(cfg)
    check_evaluator_input(cfg)
    inspect_model_predictions(cfg, num_samples=2)

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")

    print(f"\n⚠️  CRITICAL CHECKS:")
    print(f"1. Are GT boxes in NORMALIZED cxcywh [0,1]?")
    print(f"2. Are GT labels in range [0, num_classes)?")
    print(f"3. Are predictions being generated (scores > 0.3)?")
    print(f"4. Do prediction boxes match GT format?")


if __name__ == '__main__':
    main()
