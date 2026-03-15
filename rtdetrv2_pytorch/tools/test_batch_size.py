#!/usr/bin/env python3
"""
Test incrementalmente el batch size máximo soportado sin causar OOM.

Este script prueba diferentes batch sizes de forma segura, liberando memoria
entre pruebas para encontrar el máximo batch size que tu GPU puede manejar.

Usage:
    python tools/test_batch_size.py \
        -c configs/rtdetrv2/rtdetrv2_r50vd_phase1_synthetic_pure.yml \
        --start-batch 8 \
        --max-batch 64 \
        --increment 8
"""

import argparse
import sys
import torch
import gc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core import YAMLConfig


def test_batch_size(cfg, batch_size: int, num_iterations: int = 5) -> bool:
    """
    Test if a specific batch size works without OOM.

    Args:
        cfg: Config object
        batch_size: Batch size to test
        num_iterations: Number of forward+backward passes to test

    Returns:
        True if batch size works, False if OOM
    """
    print(f"\n{'='*60}")
    print(f"Testing batch_size = {batch_size}")
    print(f"{'='*60}")

    try:
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Create model
        print("Creating model...")
        cfg.yaml_cfg['train_dataloader']['total_batch_size'] = batch_size
        model = cfg.model.cuda()
        model.train()

        # Create dummy data
        print(f"Creating dummy batch (size={batch_size})...")
        dummy_images = torch.randn(
            batch_size, 3, 640, 640,
            device='cuda',
            dtype=torch.float16 if cfg.yaml_cfg.get('use_amp', False) else torch.float32
        )

        dummy_targets = []
        for i in range(batch_size):
            # Create random boxes and labels
            num_objects = torch.randint(1, 10, (1,)).item()
            boxes = torch.rand(num_objects, 4, device='cuda')
            boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]  # Convert to xyxy
            boxes = boxes.clamp(0, 1)  # Normalize

            labels = torch.randint(0, cfg.yaml_cfg['num_classes'], (num_objects,), device='cuda')

            dummy_targets.append({
                'boxes': boxes,
                'labels': labels
            })

        # Test forward + backward
        print(f"Running {num_iterations} forward+backward passes...")
        for i in range(num_iterations):
            # Forward
            outputs = model(dummy_images, targets=dummy_targets)

            # Compute loss
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif isinstance(outputs, dict):
                loss = sum(v for k, v in outputs.items() if 'loss' in k.lower())
            else:
                loss = outputs

            # Backward
            loss.backward()

            # Clear gradients
            model.zero_grad()

            # Monitor VRAM
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            vram_reserved = torch.cuda.memory_reserved() / 1024**3

            print(f"  Iteration {i+1}/{num_iterations}: "
                  f"VRAM allocated={vram_allocated:.2f}GB, "
                  f"reserved={vram_reserved:.2f}GB")

        # Success
        max_vram = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\n✅ SUCCESS: batch_size={batch_size} works!")
        print(f"   Peak VRAM usage: {max_vram:.2f}GB")

        # Cleanup
        del model, dummy_images, dummy_targets, outputs, loss
        torch.cuda.empty_cache()
        gc.collect()

        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ OOM: batch_size={batch_size} is too large")

            # Cleanup
            torch.cuda.empty_cache()
            gc.collect()

            return False
        else:
            # Other error, re-raise
            raise

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test maximum batch size without OOM"
    )
    parser.add_argument(
        '-c', '--config',
        required=True,
        help='Config file path'
    )
    parser.add_argument(
        '--start-batch',
        type=int,
        default=8,
        help='Starting batch size (default: 8)'
    )
    parser.add_argument(
        '--max-batch',
        type=int,
        default=64,
        help='Maximum batch size to test (default: 64)'
    )
    parser.add_argument(
        '--increment',
        type=int,
        default=8,
        help='Batch size increment (default: 8)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of iterations per test (default: 5)'
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    cfg = YAMLConfig(args.config)

    # Print GPU info
    print(f"\n{'='*60}")
    print("GPU Information")
    print(f"{'='*60}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")

    # Test batch sizes
    batch_sizes = range(args.start_batch, args.max_batch + 1, args.increment)
    results = {}

    for batch_size in batch_sizes:
        success = test_batch_size(cfg, batch_size, args.iterations)
        results[batch_size] = success

        if not success:
            print(f"\n⚠️  Stopping at batch_size={batch_size} (OOM)")
            break

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful_batches = [bs for bs, success in results.items() if success]
    if successful_batches:
        max_batch = max(successful_batches)
        print(f"\n✅ Maximum safe batch size: {max_batch}")
        print(f"\nRecommendations:")
        print(f"  - Safe batch size: {max_batch}")
        print(f"  - Conservative (80%): {int(max_batch * 0.8)}")
        print(f"  - Aggressive (90%): {int(max_batch * 0.9)}")

        print(f"\nUpdate your config:")
        print(f"  train_dataloader:")
        print(f"    total_batch_size: {max_batch}")
    else:
        print(f"\n❌ No batch size worked - check your config!")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
