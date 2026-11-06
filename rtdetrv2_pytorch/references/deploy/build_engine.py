#!/usr/bin/env python
"""
Build TensorRT engine from ONNX with FP16 or INT8 calibration.

For INT8, uses a calibration dataset folder to compute activation ranges.
This is required because trtexec --calib does not support custom models
with arbitrary preprocessing.

Usage:
  # FP16 (recommended, zero mAP loss)
  python build_engine.py \
    --onnx model.onnx \
    --output model_fp16.engine \
    --fp16

  # INT8 mixed precision with calibration
  python build_engine.py \
    --onnx model.onnx \
    --output model_int8.engine \
    --int8 --fp16 \
    --calib_dir /path/to/calibration_images/ \
    --calib_count 500

  # INT8 with cache (reuse calibration across builds)
  python build_engine.py \
    --onnx model.onnx \
    --output model_int8.engine \
    --int8 --fp16 \
    --calib_dir /path/to/calibration_images/ \
    --calib_cache calibration.cache
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2
import tensorrt as trt

INPUT_H = 640
INPUT_W = 640
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def preprocess_image(image_path: str) -> np.ndarray:
    """Match the same preprocessing as inference (no ImageNet normalization)."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    interp = cv2.INTER_AREA if h > INPUT_H else cv2.INTER_LINEAR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=interp)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(img[np.newaxis])  # 1x3x640x640


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator using representative images from a directory."""

    def __init__(
        self,
        calib_dir: str,
        max_count: int = 500,
        cache_file: str = None,
    ):
        super().__init__()
        self.cache_file = cache_file

        # Collect image paths
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(
                glob.glob(os.path.join(calib_dir, '**', ext), recursive=True)
            )

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {calib_dir}"
            )

        # Limit and shuffle for representative distribution
        np.random.seed(42)
        np.random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:max_count]
        self.batch_idx = 0

        print(f"[INT8] Calibrator initialized with {len(self.image_paths)} images")

        # Pre-allocate device buffer for single image
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.device_input = cuda.mem_alloc(
            1 * 3 * INPUT_H * INPUT_W * np.dtype(np.float32).itemsize
        )

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        """Feed one calibration image at a time."""
        if self.batch_idx >= len(self.image_paths):
            return None

        img = preprocess_image(self.image_paths[self.batch_idx])
        self.batch_idx += 1

        if img is None:
            # Skip bad images, try next
            return self.get_batch(names)

        import pycuda.driver as cuda
        cuda.memcpy_htod(self.device_input, img)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """Load calibration cache if it exists (speeds up repeated builds)."""
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"[INT8] Loading calibration cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Save calibration cache for reuse."""
        if self.cache_file:
            print(f"[INT8] Saving calibration cache: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                f.write(cache)


def build_engine(args):
    """Build TensorRT engine from ONNX with specified precision."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Parsing ONNX: {args.onnx}")
    with open(args.onnx, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            sys.exit(1)

    # Print network info
    print(f"  Inputs:  {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"  Outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")

    # Builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_mb * (1 << 20)
    )
    config.builder_optimization_level = args.opt_level

    # Optimization profiles for static shapes
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        (1, 3, INPUT_H, INPUT_W),
        (1, 3, INPUT_H, INPUT_W),
        (1, 3, INPUT_H, INPUT_W),
    )
    profile.set_shape("orig_target_sizes", (1, 2), (1, 2), (1, 2))
    config.add_optimization_profile(profile)

    # Timing cache
    if args.timing_cache:
        if os.path.exists(args.timing_cache):
            with open(args.timing_cache, 'rb') as f:
                cache = config.create_timing_cache(f.read())
        else:
            cache = config.create_timing_cache(b"")
        config.set_timing_cache(cache, ignore_mismatch=False)

    # Precision flags
    precision_str = "FP32"
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        precision_str = "FP16"

    if args.int8:
        config.set_flag(trt.BuilderFlag.INT8)
        precision_str = "INT8+FP16" if args.fp16 else "INT8"

        if not args.calib_dir:
            print("ERROR: --calib_dir required for INT8 calibration")
            sys.exit(1)

        calibrator = ImageCalibrator(
            calib_dir=args.calib_dir,
            max_count=args.calib_count,
            cache_file=args.calib_cache,
        )
        config.int8_calibrator = calibrator

    print(f"\nBuilding {precision_str} engine (opt_level={args.opt_level}, "
          f"workspace={args.workspace_mb}MB)...")
    print("This may take several minutes on Jetson...")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: Engine build failed!")
        sys.exit(1)

    # Save engine
    with open(args.output, 'wb') as f:
        f.write(serialized)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nEngine saved: {args.output} ({size_mb:.1f} MB)")
    print(f"Precision: {precision_str}")

    # Save timing cache
    if args.timing_cache:
        cache = config.get_timing_cache()
        with open(args.timing_cache, 'wb') as f:
            f.write(cache.serialize())
        print(f"Timing cache saved: {args.timing_cache}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine with FP16 or INT8 calibration",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--onnx', type=str, required=True,
        help="Input ONNX model path",
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help="Output engine file path",
    )

    prec = parser.add_argument_group('Precision')
    prec.add_argument('--fp16', action='store_true', help="Enable FP16")
    prec.add_argument('--int8', action='store_true', help="Enable INT8")

    calib = parser.add_argument_group('INT8 Calibration')
    calib.add_argument(
        '--calib_dir', type=str,
        help="Directory with calibration images",
    )
    calib.add_argument(
        '--calib_count', type=int, default=500,
        help="Number of calibration images (default: 500)",
    )
    calib.add_argument(
        '--calib_cache', type=str, default='calibration.cache',
        help="Calibration cache file (reuse across builds)",
    )

    build = parser.add_argument_group('Builder Options')
    build.add_argument(
        '--workspace_mb', type=int, default=4096,
        help="Workspace size in MB (default: 4096)",
    )
    build.add_argument(
        '--opt_level', type=int, default=5, choices=range(6),
        help="Builder optimization level 0-5 (default: 5, max)",
    )
    build.add_argument(
        '--timing_cache', type=str, default='timing_cache.bin',
        help="Timing cache file path",
    )

    args = parser.parse_args()

    if not args.fp16 and not args.int8:
        print("WARNING: No precision flag set, building FP32 (slowest).")
        print("Consider adding --fp16 for ~2x speedup with zero mAP loss.")

    build_engine(args)
