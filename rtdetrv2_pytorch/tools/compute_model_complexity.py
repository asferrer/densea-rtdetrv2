"""
Compute model complexity (parameters, FLOPs, model size) for RT-DETRv2.

Usage (inside Docker container):
    cd /app/RT-DETR
    python tools/compute_model_complexity.py \
        -c configs/my_experiment.yml \
        -w output/best.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from src.core import YAMLConfig


def compute_complexity(
    config_path: str,
    weights_path: str = None,
    device: str = "cuda:0",
    input_shape: tuple = (1, 3, 640, 640),
):
    cfg = YAMLConfig(config_path, device=device)
    model = cfg.model.to(device)
    model.eval()

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs via torch profiler
    dummy = torch.rand(*input_shape, dtype=torch.float32, device=device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            model(dummy)

    total_flops = sum(
        evt.flops for evt in prof.key_averages() if evt.flops is not None
    )

    # Model size from weights file
    model_size_mb = None
    if weights_path and os.path.isfile(weights_path):
        model_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    # Print results
    print("=" * 60)
    print("RT-DETRv2 Model Complexity")
    print("=" * 60)
    print(f"Config:              {config_path}")
    print(f"Input shape:         {input_shape}")
    print(f"Total params:        {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"Trainable params:    {trainable_params:,} ({trainable_params / 1e6:.1f}M)")
    print(f"GFLOPs:              {total_flops / 1e9:.1f}")
    if model_size_mb is not None:
        print(f"Weights file size:   {model_size_mb:.1f} MB ({weights_path})")
    print("=" * 60)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gflops": total_flops / 1e9,
        "model_size_mb": model_size_mb,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RT-DETRv2 model complexity")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML config")
    parser.add_argument("-w", "--weights", type=str, default=None, help="Weights .pth")
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    args = parser.parse_args()

    compute_complexity(args.config, args.weights, args.device)
