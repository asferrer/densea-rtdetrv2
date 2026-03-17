#!/usr/bin/env python
"""
Build a TensorRT INT8+FP16 engine using nvidia-modelopt for optimal Q/DQ placement.

Designed for memory-constrained devices (Jetson Orin Nano, 8 GB unified).
Uses nvidia-modelopt which places Q/DQ nodes optimized for TensorRT fusion patterns,
producing better INT8 engines than generic ONNX Runtime quantization.

Runs quantization and engine building as SEPARATE PROCESSES to avoid OOM:
  Phase 1 (quantize): Spawns a subprocess that runs modelopt on CPU.
                       When the subprocess exits, the OS reclaims ALL its
                       memory — guaranteed clean slate for Phase 2.
  Phase 2 (build):    Loads quantized ONNX into TensorRT builder.

Usage:
    # Full pipeline (quantize + build):
    python references/deploy/build_int8_engine.py \
        --onnx /path/to/model_simplified_bs1.onnx \
        --calib_dir calibration_set \
        --output /path/to/model_int8.engine

    # Quantize only (no GPU needed):
    python references/deploy/build_int8_engine.py \
        --onnx /path/to/model_simplified_bs1.onnx \
        --calib_dir calibration_set \
        --quantize_only

    # Build only (from previously quantized ONNX):
    python references/deploy/build_int8_engine.py \
        --onnx /path/to/model_int8.onnx \
        --output /path/to/model_int8.engine \
        --build_only

Requirements:
    pip install nvidia-modelopt==0.19.0 --no-deps
    pip install onnx-graphsurgeon pydantic pulp ml_dtypes cppimport ninja regex rich safetensors

References:
    - https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_onnx_quantization.html
    - https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
"""

import os
import sys
import argparse
import subprocess
import random
import time
import gc
import cv2
import numpy as np

INPUT_H = 640
INPUT_W = 640
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


# ===========================================================================
# Preprocessing (matches inference pipeline exactly)
# ===========================================================================

def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Resize, normalize, transpose to NCHW float32."""
    h, w = image_bgr.shape[:2]
    interp = cv2.INTER_AREA if h > INPUT_H else cv2.INTER_LINEAR
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=interp)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(img[np.newaxis])  # 1x3x640x640


def collect_image_paths(calib_dir: str, max_images: int = 0) -> list:
    """Collect, shuffle, and optionally limit image paths."""
    paths = []
    if not calib_dir or not os.path.isdir(calib_dir):
        return paths
    for fname in os.listdir(calib_dir):
        if fname.lower().endswith(SUPPORTED_EXTS):
            paths.append(os.path.join(calib_dir, fname))
    random.shuffle(paths)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def prepare_calibration_data(calib_dir: str, max_images: int) -> dict:
    """
    Load and preprocess calibration images into numpy arrays.
    Returns dict with input names as keys for multi-input models.
    """
    image_paths = collect_image_paths(calib_dir, max_images)
    if not image_paths:
        print(f"[ERROR] No images found in {calib_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[Calibration] Loading {len(image_paths)} images...")

    images_list = []
    sizes_list = []
    loaded = 0

    for i, path in enumerate(image_paths):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"  WARNING: Could not read {path}, skipping")
            continue

        batch = preprocess(img_bgr)  # (1, 3, 640, 640)
        orig_sizes = np.array(
            [[img_bgr.shape[1], img_bgr.shape[0]]], dtype=np.int32
        )  # (1, 2)

        images_list.append(batch)
        sizes_list.append(orig_sizes)
        loaded += 1

        if (loaded % 10 == 0) or (i == len(image_paths) - 1):
            print(f"  [{loaded}/{len(image_paths)}] preprocessed")

    # Stack into (N, 3, 640, 640) and (N, 2)
    images_array = np.concatenate(images_list, axis=0)
    sizes_array = np.concatenate(sizes_list, axis=0)

    print(f"[Calibration] images shape: {images_array.shape} "
          f"({images_array.nbytes / 1024 / 1024:.1f} MiB)")
    print(f"[Calibration] orig_target_sizes shape: {sizes_array.shape}")

    return {"images": images_array, "orig_target_sizes": sizes_array}


# ===========================================================================
# PHASE 1: Quantize ONNX with nvidia-modelopt — runs in a SUBPROCESS
# ===========================================================================

def _run_quantize_subprocess(onnx_path: str, calib_dir: str,
                             output_onnx: str, calibrate_method: str,
                             max_images: int):
    """
    Spawn a child process that does the modelopt quantization.

    This is the key to avoiding OOM on Jetson: the subprocess allocates
    memory for modelopt + model + calibration, and when it exits ALL that
    memory is returned to the OS. The parent process (TRT builder) then
    starts with a clean memory slate.
    """
    script = os.path.abspath(__file__)
    cmd = [
        sys.executable, script,
        "--_quantize_worker",
        "--onnx", os.path.abspath(onnx_path),
        "--calib_dir", os.path.abspath(calib_dir),
        "--output_onnx", os.path.abspath(output_onnx),
        "--calibrate_method", calibrate_method,
        "--max_images", str(max_images),
    ]

    # Keep CUDA visible (modelopt/TRT needs to see GPU for initialization)
    # but force calibration inference on CPU via calibration_eps=["cpu"]
    env = os.environ.copy()
    # Fix libstdc++ for torch import on Jetson
    env["LD_PRELOAD"] = "/usr/lib/aarch64-linux-gnu/libstdc++.so.6"

    print(f"[Quantize] Spawning subprocess (calibration on CPU)...")
    print(f"[Quantize] Backend: nvidia-modelopt (TRT-optimized Q/DQ placement)")
    print(f"[Quantize] Max images: {max_images} | Method: {calibrate_method}")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print("[Quantize] ERROR: Subprocess failed!", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(output_onnx):
        print(f"[Quantize] ERROR: Output not found: {output_onnx}", file=sys.stderr)
        sys.exit(1)

    onnx_size = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"[Quantize] Subprocess done. Output: {output_onnx} ({onnx_size:.1f} MiB)")
    print("[Quantize] Subprocess memory fully released by OS.\n")


def _quantize_worker(args):
    """
    Worker function that runs inside the subprocess.
    Uses nvidia-modelopt for TRT-optimized Q/DQ node placement.
    """
    try:
        from modelopt.onnx.quantization import quantize
    except ImportError as e:
        print(f"[ERROR] nvidia-modelopt not available: {e}", file=sys.stderr)
        print("[ERROR] Install with: pip install nvidia-modelopt==0.19.0 --no-deps",
              file=sys.stderr)
        sys.exit(1)

    # Prepare calibration data as numpy arrays
    calib_data = prepare_calibration_data(args.calib_dir, args.max_images)

    # Map method names to modelopt format
    method_map = {
        "entropy": "entropy",
        "minmax": "max",        # modelopt uses "max" for MinMax
        "percentile": "percentile",
    }
    cal_method = method_map.get(args.calibrate_method, "max")

    print(f"\n[Quantize:worker] Backend: nvidia-modelopt (TRT-optimized)")
    print(f"[Quantize:worker] Method: {cal_method} | Mode: int8")
    print(f"[Quantize:worker] Input:  {args.onnx}")
    print(f"[Quantize:worker] Output: {args.output_onnx}")

    # Exclude decoder, postprocessor, and score/bbox heads from INT8.
    # Deformable attention + sigmoid are extremely sensitive to quantization
    # and collapse output scores to ~0 when quantized to INT8.
    # Only the backbone and encoder (Conv/BN layers) are quantized.
    nodes_to_exclude = [
        ".*decoder.*",
        ".*postprocessor.*",
        ".*enc_score_head.*",
        ".*enc_bbox_head.*",
        ".*dec_score_head.*",
        ".*dec_bbox_head.*",
        ".*query_pos_head.*",
    ]
    print(f"[Quantize:worker] Excluding from INT8: {nodes_to_exclude}")

    t0 = time.time()
    quantize(
        onnx_path=args.onnx,
        quantize_mode="int8",
        calibration_data=calib_data,
        calibration_method=cal_method,
        calibration_eps=["cpu"],        # CPU-only for Jetson memory safety
        nodes_to_exclude=nodes_to_exclude,
        output_path=args.output_onnx,
        use_external_data_format=False,
        verbose=True,
    )
    elapsed = time.time() - t0

    # Free memory before exit
    del calib_data
    gc.collect()

    output_size = os.path.getsize(args.output_onnx) / (1024 * 1024)
    print(f"\n[Quantize:worker] Done in {elapsed:.1f}s — {output_size:.1f} MiB")


# ===========================================================================
# PHASE 2: Build TensorRT engine (imports TRT/pycuda lazily)
# ===========================================================================

def build_engine(onnx_path: str, args):
    """Build TensorRT engine from quantized ONNX with Q/DQ nodes."""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    t_start = time.time()

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[Builder] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  Parse error: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    print(f"[Builder] Network: {network.num_inputs} inputs, {network.num_outputs} outputs")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}: {inp.name} shape={inp.shape} dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name} shape={out.shape} dtype={out.dtype}")

    # --- Config ---
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace << 20)
    config.builder_optimization_level = args.opt_level

    # Use STRONGLY_TYPED for explicit Q/DQ models (nvidia-modelopt).
    # This forces TRT to strictly follow the Q/DQ precision decisions
    # instead of freely choosing INT8/FP16/FP32 per layer.
    # See: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html
    if hasattr(trt.BuilderFlag, 'STRONGLY_TYPED'):
        config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)
        print(f"[Builder] Mode: STRONGLY_TYPED (explicit Q/DQ from modelopt)")
    else:
        # Fallback for older TensorRT versions
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        print(f"[Builder] Mode: EXPLICIT INT8+FP16 (modelopt Q/DQ nodes)")
    print(f"[Builder] Workspace: {args.workspace} MiB | OptLevel: {args.opt_level}")

    # --- Optimization profile ---
    profile = builder.create_optimization_profile()
    profile.set_shape("images",
                      (1, 3, INPUT_H, INPUT_W),
                      (1, 3, INPUT_H, INPUT_W),
                      (1, 3, INPUT_H, INPUT_W))
    profile.set_shape("orig_target_sizes", (1, 2), (1, 2), (1, 2))
    config.add_optimization_profile(profile)

    # --- Timing cache ---
    if args.timing_cache and os.path.exists(args.timing_cache):
        print(f"[Builder] Loading timing cache: {args.timing_cache}")
        with open(args.timing_cache, "rb") as f:
            tc = config.create_timing_cache(f.read())
            config.set_timing_cache(tc, ignore_mismatch=False)
    else:
        tc = config.create_timing_cache(b"")
        config.set_timing_cache(tc, ignore_mismatch=False)

    # --- Build ---
    print("[Builder] Building engine... (30-60 min on Jetson)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[Builder] ERROR: Engine build failed!", file=sys.stderr)
        print("[Builder] Try --opt_level 2 or --workspace 1024", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - t_start
    engine_mb = serialized.nbytes / (1024 * 1024)
    print(f"[Builder] Built in {elapsed:.0f}s ({elapsed / 60:.1f} min) — {engine_mb:.1f} MiB")

    # --- Save ---
    with open(args.output, "wb") as f:
        f.write(serialized)
    print(f"[Builder] Saved: {args.output}")

    # --- Save timing cache ---
    if args.timing_cache:
        timing_cache = config.get_timing_cache()
        with open(args.timing_cache, "wb") as f:
            f.write(bytearray(timing_cache.serialize()))
        print(f"[Builder] Timing cache saved: {args.timing_cache}")

    # --- Verify ---
    print("\n[Verify] Loading engine...")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        print("[Verify] FAILED!", file=sys.stderr)
        sys.exit(1)
    context = engine.create_execution_context()
    print(f"[Verify] OK — {engine.num_io_tensors} I/O tensors:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"  {mode.name:6s} {name}: shape={tuple(shape)} dtype={dtype}")
    del context, engine

    print(f"\n[Done] Engine: {args.output}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Build TensorRT INT8 engine with nvidia-modelopt quantization",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--onnx", type=str, required=True,
                   help="Path to simplified ONNX model (batch size must be fixed)")
    p.add_argument("--calib_dir", type=str, default="",
                   help="Directory with calibration images")
    p.add_argument("--output", type=str, default="model_int8.engine",
                   help="Output engine path")
    p.add_argument("--workspace", type=int, default=2048,
                   help="Workspace MiB (default: 2048, reduce if OOM)")
    p.add_argument("--opt_level", type=int, default=3,
                   help="Builder optimization level 1-5 (default: 3)")
    p.add_argument("--timing_cache", type=str, default="timing_cache.bin",
                   help="Timing cache path")
    p.add_argument("--calibrate_method", type=str, default="minmax",
                   choices=["entropy", "minmax"],
                   help="Calibration method (default: minmax, uses less RAM)")
    p.add_argument("--max_images", type=int, default=50,
                   help="Max calibration images (default: 50, reduce if OOM)")
    p.add_argument("--quantize_only", action="store_true",
                   help="Only quantize ONNX (no engine build, no GPU needed)")
    p.add_argument("--build_only", action="store_true",
                   help="Only build engine (ONNX must already have Q/DQ nodes)")

    # Hidden arg: the subprocess worker mode
    p.add_argument("--_quantize_worker", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--output_onnx", type=str, default="",
                   help=argparse.SUPPRESS)

    args = p.parse_args()

    # --- Subprocess worker mode (called internally) ---
    if args._quantize_worker:
        _quantize_worker(args)
        return

    # --- Derive output ONNX name ---
    base, ext = os.path.splitext(args.onnx)
    quantized_onnx = args.output_onnx if args.output_onnx else f"{base}_int8{ext}"

    # --- Normal user-facing modes ---
    if args.build_only:
        build_engine(args.onnx, args)

    elif args.quantize_only:
        if not args.calib_dir:
            p.error("--calib_dir required for quantization")
        _run_quantize_subprocess(
            args.onnx, args.calib_dir, quantized_onnx,
            args.calibrate_method, args.max_images,
        )
        print(f"\nQuantized ONNX ready: {quantized_onnx}")
        print(f"To build engine:\n  python {sys.argv[0]} "
              f"--onnx {quantized_onnx} --output {args.output} --build_only")

    else:
        # Full pipeline: quantize (subprocess) → build
        if not args.calib_dir:
            p.error("--calib_dir required for quantization")

        # Phase 1: subprocess — all memory freed when it exits
        _run_quantize_subprocess(
            args.onnx, args.calib_dir, quantized_onnx,
            args.calibrate_method, args.max_images,
        )

        # Phase 2: TRT build — starts with clean memory
        build_engine(quantized_onnx, args)


if __name__ == "__main__":
    main()
