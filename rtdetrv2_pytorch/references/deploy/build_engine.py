#!/usr/bin/env python
"""
Build TensorRT engine from ONNX with FP16 or INT8 calibration.
Mixed precision: decoder/postprocessor layers forced to FP32 to preserve
accuracy with deformable attention (critical for RT-DETRv2).

For INT8, uses native TRT calibration with representative images.

Usage:
  # FP32 (baseline)
  python build_engine.py --onnx model.onnx --output model_fp32.engine

  # FP16 mixed precision (backbone FP16, decoder FP32)
  python build_engine.py --onnx model.onnx --output model_fp16.engine --fp16

  # INT8 mixed precision with calibration (backbone INT8, decoder FP32)
  python build_engine.py \
    --onnx model.onnx \
    --output model_int8.engine \
    --int8 --fp16 \
    --calib_dir /path/to/calibration_images/ \
    --calib_count 500
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

# Patterns for layers that MUST stay in FP32 (deformable attention is FP16-sensitive)
FP32_LAYER_PATTERNS = [
    '/decoder/',
    '/postprocessor/',
    'Sigmoid',
    'Softmax',
    'enc_score_head',
    'enc_bbox_head',
    'dec_score_head',
    'dec_bbox_head',
    'query_pos_head',
]


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
    """INT8 calibrator using representative images from a directory.
    Uses cudart (via tensorrt) instead of pycuda for CUDA memory management.
    """

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
            raise FileNotFoundError(f"No images found in {calib_dir}")

        # Limit and shuffle for representative distribution
        np.random.seed(42)
        np.random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:max_count]
        self.batch_idx = 0

        print(f"[INT8] Calibrator initialized with {len(self.image_paths)} images")

        # Allocate device buffer using cudart
        from cuda import cudart
        self.cudart = cudart
        nbytes = 1 * 3 * INPUT_H * INPUT_W * np.dtype(np.float32).itemsize
        err, self.device_input = cudart.cudaMalloc(nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed: {err}")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        """Feed one calibration image at a time."""
        if self.batch_idx >= len(self.image_paths):
            return None

        img = preprocess_image(self.image_paths[self.batch_idx])
        self.batch_idx += 1

        if img is None:
            return self.get_batch(names)

        err, = self.cudart.cudaMemcpy(
            self.device_input, img.ctypes.data, img.nbytes,
            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"[INT8] Loading calibration cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            print(f"[INT8] Saving calibration cache: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

    def __del__(self):
        if hasattr(self, 'device_input') and self.device_input:
            self.cudart.cudaFree(self.device_input)


class ImageCalibratorMinMax(trt.IInt8MinMaxCalibrator):
    """INT8 calibrator using MinMax algorithm (alternative to entropy)."""

    def __init__(self, calib_dir, max_count=500, cache_file=None):
        super().__init__()
        self._inner = ImageCalibrator(calib_dir, max_count, cache_file)

    def get_batch_size(self):
        return self._inner.get_batch_size()

    def get_batch(self, names):
        return self._inner.get_batch(names)

    def read_calibration_cache(self):
        return self._inner.read_calibration_cache()

    def write_calibration_cache(self, cache):
        self._inner.write_calibration_cache(cache)


def force_fp32_on_sensitive_layers(network):
    """Force FP32 precision on decoder/postprocessor layers.
    Skips Constant layers and layers with integer output types (Int64/Int32)
    which cannot be assigned floating-point precision in TRT.
    Returns count of layers forced to FP32.
    """
    # TRT types that are integer (cannot be forced to FP32)
    int_types = {trt.int64, trt.int32, trt.int8, trt.bool}

    fp32_count = 0
    skipped = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        name = layer.name

        if not any(pattern in name for pattern in FP32_LAYER_PATTERNS):
            continue

        # Skip Constant layers — they have fixed dtype from ONNX weights
        if layer.type == trt.LayerType.CONSTANT:
            skipped += 1
            continue

        # Skip layers whose outputs are integer types
        has_int_output = False
        for j in range(layer.num_outputs):
            if layer.get_output(j).dtype in int_types:
                has_int_output = True
                break
        if has_int_output:
            skipped += 1
            continue

        layer.precision = trt.float32
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float32)
        fp32_count += 1

    if skipped > 0:
        print(f"  Skipped {skipped} integer/constant layers (kept native dtype)")
    return fp32_count


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

    print(f"  Inputs:  {[network.get_input(i).name for i in range(network.num_inputs)]}")
    print(f"  Outputs: {[network.get_output(i).name for i in range(network.num_outputs)]}")
    print(f"  Total layers: {network.num_layers}")

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
    use_mixed_precision = args.fp16 or args.int8

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        precision_str = "FP16"

    if args.int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if args.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        precision_str = "INT8+FP16" if args.fp16 else "INT8"

        if not args.calib_dir:
            print("ERROR: --calib_dir required for INT8 calibration")
            sys.exit(1)

        CalibratorClass = ImageCalibratorMinMax if args.calib_method == 'minmax' else ImageCalibrator
        calibrator = CalibratorClass(
            calib_dir=args.calib_dir,
            max_count=args.calib_count,
            cache_file=args.calib_cache,
        )
        config.int8_calibrator = calibrator
        print(f"  Calibration method: {args.calib_method}")

    # Mixed precision: force FP32 on decoder/postprocessor to prevent
    # accuracy degradation from deformable attention + sigmoid in reduced precision.
    # For INT8: use PREFER (not OBEY) to avoid TRT 10.x internal graph assertion errors
    # when combining legacy calibrator with strict precision constraints.
    if use_mixed_precision:
        if args.int8:
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        else:
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        fp32_count = force_fp32_on_sensitive_layers(network)
        print(f"  Mixed precision: {fp32_count} decoder/postprocessor layers forced to FP32")
        constraint_mode = "PREFER" if args.int8 else "OBEY"
        precision_str += f" (mixed, decoder=FP32, {constraint_mode})"

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
        description="Build TensorRT engine with FP16/INT8 mixed precision",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--onnx', type=str, required=True, help="Input ONNX model path")
    parser.add_argument('--output', type=str, required=True, help="Output engine file path")

    prec = parser.add_argument_group('Precision')
    prec.add_argument('--fp16', action='store_true', help="Enable FP16 (backbone)")
    prec.add_argument('--int8', action='store_true', help="Enable INT8 (backbone)")

    calib = parser.add_argument_group('INT8 Calibration')
    calib.add_argument('--calib_dir', type=str, help="Directory with calibration images")
    calib.add_argument('--calib_count', type=int, default=500,
                       help="Number of calibration images (default: 500)")
    calib.add_argument('--calib_cache', type=str, default='calibration.cache',
                       help="Calibration cache file (reuse across builds)")
    calib.add_argument('--calib_method', type=str, default='entropy',
                       choices=['entropy', 'minmax'],
                       help="INT8 calibration algorithm (default: entropy)")

    build = parser.add_argument_group('Builder Options')
    build.add_argument('--workspace_mb', type=int, default=4096,
                       help="Workspace size in MB (default: 4096)")
    build.add_argument('--opt_level', type=int, default=5, choices=range(6),
                       help="Builder optimization level 0-5 (default: 5, max)")
    build.add_argument('--timing_cache', type=str, default='timing_cache.bin',
                       help="Timing cache file path")

    args = parser.parse_args()

    if not args.fp16 and not args.int8:
        print("WARNING: No precision flag set, building FP32 (slowest).")
        print("Consider adding --fp16 for ~2x speedup.")

    build_engine(args)
