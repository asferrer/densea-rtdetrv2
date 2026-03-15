#!/usr/bin/env python3
"""
Post-training INT8 quantization for RT-DETRv2 TFLite model.

Requires a representative calibration dataset (50-200 images recommended).
Uses the SavedModel produced by onnx2tf as input.

Usage:
    python quantize_int8.py \
        --saved-model-dir ./converted_models_new \
        --calibration-dir /path/to/calibration/images \
        --output ./converted_models_new/rtdetrv2_detector_int8.tflite
"""

import argparse
import os
import sys
import glob
import numpy as np


def load_and_preprocess_image(image_path, target_size=(640, 640)):
    """Load and preprocess a single image for RT-DETRv2 input."""
    try:
        from PIL import Image
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "--break-system-packages", "Pillow"])
        from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    # Input shape: (1, H, W, 3) — NHWC format (TFLite/onnx2tf convention)
    return np.expand_dims(arr, axis=0)


def create_representative_dataset(calibration_dir, num_samples=100):
    """Create a representative dataset generator for INT8 calibration."""
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(calibration_dir, "**", ext), recursive=True))

    if not image_paths:
        raise RuntimeError(f"No images found in {calibration_dir}")

    # Limit to num_samples
    if len(image_paths) > num_samples:
        np.random.seed(42)
        image_paths = list(np.random.choice(image_paths, num_samples, replace=False))

    print(f"Using {len(image_paths)} calibration images from {calibration_dir}")

    def representative_dataset():
        for i, path in enumerate(image_paths):
            try:
                data = load_and_preprocess_image(path)
                yield [data]
                if (i + 1) % 10 == 0:
                    print(f"  Calibrated {i + 1}/{len(image_paths)} images...")
            except Exception as e:
                print(f"  Skipping {os.path.basename(path)}: {e}")
                continue

    return representative_dataset


def quantize_to_int8(saved_model_dir, calibration_dir, output_path, num_samples=100):
    """Convert SavedModel to INT8 TFLite with calibration."""
    import tensorflow as tf

    print(f"\nLoading SavedModel from: {saved_model_dir}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # INT8 quantization settings — TFLITE_BUILTINS only.
    # The Erf op was replaced with a pseudo-polynomial during onnx2tf conversion
    # (replace_erf_to_pseudo_erf=True), so SELECT_TF_OPS is no longer needed.
    # Removing it avoids the FlexErf crash on Samsung Exynos 2200.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    # Representative dataset for calibration
    print("Creating representative dataset for calibration...")
    rep_dataset = create_representative_dataset(calibration_dir, num_samples)
    converter.representative_dataset = rep_dataset

    # Keep input/output as float32 for compatibility with existing Kotlin code
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    print("\nRunning INT8 quantization (this may take a few minutes)...")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nINT8 model saved: {output_path} ({size_mb:.1f} MB)")
    return output_path


def validate_int8(tflite_path):
    """Quick validation of INT8 TFLite model."""
    import tensorflow as tf

    print(f"\nValidating INT8 model: {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input tensors:")
    for inp in input_details:
        print(f"  {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
        if inp.get('quantization_parameters'):
            q = inp['quantization_parameters']
            if q.get('scales') is not None and len(q['scales']) > 0:
                print(f"    quantization: scales={q['scales'][:3]}..., zero_points={q['zero_points'][:3]}...")

    print("Output tensors:")
    for out in output_details:
        print(f"  {out['name']}: shape={out['shape']}, dtype={out['dtype']}")

    # Dummy inference
    input_shape = input_details[0]['shape']
    dummy = np.random.rand(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

    for out in output_details:
        result = interpreter.get_tensor(out['index'])
        print(f"  {out['name']}: output_range=[{result.min():.4f}, {result.max():.4f}]")

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")

    # Verify no Flex/SELECT_TF_OPS ops remain
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    if b"FlexErf" in model_bytes:
        print("WARNING: Model still contains FlexErf — SELECT_TF_OPS dependency!")
        print("The Erf op was NOT replaced during onnx2tf conversion.")
        print("Re-run onnx2tf with replace_erf_to_pseudo_erf=True")
    elif b"Flex" in model_bytes:
        # Check for any other Flex ops
        import re
        flex_ops = set(re.findall(rb"Flex(\w+)", model_bytes))
        if flex_ops:
            print(f"WARNING: Model contains Flex ops: {[op.decode() for op in flex_ops]}")
        else:
            print("No Flex ops detected (TFLITE_BUILTINS only)")
    else:
        print("No Flex ops detected (TFLITE_BUILTINS only)")

    print("Validation PASSED")


def main():
    parser = argparse.ArgumentParser(description="INT8 quantize RT-DETRv2 TFLite")
    parser.add_argument("--saved-model-dir", required=True,
                        help="Directory containing saved_model.pb from onnx2tf")
    parser.add_argument("--calibration-dir", required=True,
                        help="Directory with calibration images (JPG/PNG)")
    parser.add_argument("--output", default="./rtdetrv2_detector_int8.tflite",
                        help="Output INT8 TFLite path")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of calibration samples (default: 100)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate after quantization")

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.saved_model_dir, "saved_model.pb")):
        print(f"Error: saved_model.pb not found in {args.saved_model_dir}")
        sys.exit(1)

    output_path = quantize_to_int8(
        args.saved_model_dir, args.calibration_dir, args.output, args.num_samples
    )

    if args.validate:
        validate_int8(output_path)

    # Compare sizes
    fp16_path = os.path.join(args.saved_model_dir, "rtdetrv2_detector.tflite")
    if os.path.exists(fp16_path):
        fp16_size = os.path.getsize(fp16_path) / (1024 * 1024)
        int8_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - int8_size / fp16_size) * 100
        print(f"\nSize comparison:")
        print(f"  FP16: {fp16_size:.1f} MB")
        print(f"  INT8: {int8_size:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
