"""
Traceable inference-latency benchmark for RT-DETRv2 (PyTorch, batch 1).

Protocol: >=50 warmup iterations, >=300 timed iterations, torch.cuda.synchronize()
around every timed forward pass, reports mean / std / median / p95 and FPS.

Usage (in the training container):
    cd /app/RT-DETR
    python tools/benchmark_latency.py \
        -c configs/my_experiment.yml \
        -w output/best.pth \
        --size 640 640
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from src.core import YAMLConfig


def driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unavailable (nvidia-smi not reachable from container)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-w", "--weights", default=None)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--size", type=int, nargs=2, default=[640, 640], metavar=("H", "W"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = YAMLConfig(args.config, device=args.device)
    model = cfg.model.to(device)

    if args.weights:
        state = torch.load(args.weights, map_location="cpu", weights_only=False)
        sd = state.get("ema", {}).get("module") if "ema" in state else None
        sd = sd or state.get("model", state)
        model.load_state_dict(sd)

    model.eval()

    h, w = args.size
    dummy = torch.rand(1, 3, h, w, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(dummy)
        torch.cuda.synchronize()

        timings = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.mean(timings)
    std = statistics.pstdev(timings)
    median = statistics.median(timings)
    p95 = sorted(timings)[int(0.95 * len(timings)) - 1]

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "driver_version": driver_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "amp": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }

    print("=" * 64)
    print("RT-DETRv2 inference latency (PyTorch FP32, batch 1)")
    print("=" * 64)
    print(f"Config:        {args.config}")
    print(f"Weights:       {args.weights}")
    print(f"Input:         1x3x{h}x{w}")
    print(f"Warmup iters:  {args.warmup}")
    print(f"Timed iters:   {args.iters}")
    print("-" * 64)
    for k, v in env.items():
        print(f"{k:16s} {v}")
    print("-" * 64)
    print(f"Mean latency:   {mean:8.3f} ms")
    print(f"Std dev:        {std:8.3f} ms")
    print(f"Median:         {median:8.3f} ms")
    print(f"p95:            {p95:8.3f} ms")
    print(f"Throughput:     {1000.0 / mean:8.2f} FPS")
    print("=" * 64)

    out = {
        "model": "RT-DETRv2-R50",
        "config": args.config,
        "weights": args.weights,
        "input_shape": [1, 3, h, w],
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "mean_ms": mean,
        "std_ms": std,
        "median_ms": median,
        "p95_ms": p95,
        "fps": 1000.0 / mean,
        "environment": env,
    }
    json_path = os.path.splitext(os.environ.get("BENCH_JSON", "results/latency.json"))[0] + ".json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON saved to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
