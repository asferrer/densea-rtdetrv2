#!/usr/bin/env python3
"""
Convert RT-DETRv2 model from ONNX to TFLite for Android deployment.

Usage:
    python convert_rtdetr.py --onnx-path /path/to/rtdetrv2.onnx --output-dir ./converted_models

Requirements:
    pip install onnx onnx2tf tensorflow tf-keras ai-edge-litert numpy onnx-simplifier

Recommended Docker image (all deps pre-installed):
    docker.io/pinto0309/onnx2tf:latest
    See Dockerfile.convert in this repo for usage.
"""

import argparse
import os
import sys
import traceback
import numpy as np


def check_environment():
    """Verify conversion dependencies are installed and compatible."""
    print("Checking conversion environment...")
    errors = []
    warnings = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python: {py_ver}")
    if sys.version_info < (3, 10):
        errors.append(f"Python {py_ver} detected. onnx2tf requires Python >= 3.10.")

    # onnx
    try:
        import onnx
        print(f"  onnx: {onnx.__version__}")
    except ImportError:
        errors.append("onnx not installed. Run: pip install onnx>=1.15")

    # onnx2tf
    try:
        import onnx2tf
        print(f"  onnx2tf: {onnx2tf.__version__}")
    except ImportError:
        errors.append("onnx2tf not installed. Run: pip install onnx2tf")

    # tensorflow
    try:
        import tensorflow as tf
        print(f"  tensorflow: {tf.__version__}")
    except ImportError:
        errors.append("TensorFlow not installed. Run: pip install tensorflow>=2.18 tf-keras>=2.18")

    # ai-edge-litert (needed by onnx2tf for TFLite export)
    try:
        import ai_edge_litert
        print(f"  ai-edge-litert: {ai_edge_litert.__version__}")
    except ImportError:
        warnings.append("ai-edge-litert not installed (may be needed). Run: pip install ai-edge-litert")

    # numpy
    print(f"  numpy: {np.__version__}")

    # Warn about dead onnx-tf
    try:
        import onnx_tf  # noqa: F401
        warnings.append(
            "onnx-tf is installed but ABANDONED (last release: 2022). "
            "It is incompatible with onnx >= 1.14. Consider uninstalling: pip uninstall onnx-tf"
        )
    except ImportError:
        pass

    for w in warnings:
        print(f"  [WARN] {w}")

    if errors:
        print("\nEnvironment check FAILED:")
        for e in errors:
            print(f"  [ERROR] {e}")
        print("\nRecommended: use the official onnx2tf Docker image:")
        print("  docker build -f Dockerfile.convert -t rtdetr-convert .")
        raise SystemExit(1)

    print("  Environment OK.\n")

def convert_onnx_to_tflite(onnx_path: str, output_dir: str, quantize: str = "fp16",
                           verbose: bool = False):
    """
    Convert ONNX model to TFLite using onnx2tf.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory
        quantize: Quantization mode: "none", "fp16", "int8"
        verbose: Enable verbose onnx2tf output for debugging
    """
    import onnx

    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"Model verified: {len(model.graph.node)} nodes")

    for inp in model.graph.input:
        print(f"  Input: {inp.name}, shape: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    for out in model.graph.output:
        print(f"  Output: {out.name}, shape: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")

    os.makedirs(output_dir, exist_ok=True)

    print("\nConverting with onnx2tf...")
    import onnx2tf

    # Fix dynamic batch dimension to static batch=1
    # onnx2tf requires static shapes; the ONNX export uses dynamic 'N' for batch
    overwrite_input_shape = ["images:1,3,640,640"]

    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=output_dir,
            output_signaturedefs=True,
            non_verbose=not verbose,
            overwrite_input_shape=overwrite_input_shape,
            # Replace Erf with polynomial approximation using only TFLite builtins.
            # This avoids the SELECT_TF_OPS / FlexErf dependency that crashes on
            # Samsung Exynos 2200 (TF Select Ops 2.16.1 Erf attribute mismatch).
            replace_to_pseudo_operators=["Erf"],
            copy_onnx_input_output_names_to_tflite=True,
        )
    except Exception as e:
        print(f"\n[ERROR] onnx2tf conversion failed: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        if e.__cause__:
            print(f"\nCaused by: {type(e.__cause__).__name__}: {e.__cause__}")
        print("\nTroubleshooting:")
        print("  1. Run with --verbose to see which operator fails")
        print("  2. Try simplifying the ONNX first: python -m onnxsim model.onnx model_sim.onnx")
        print("  3. Use the official Docker image: docker build -f Dockerfile.convert -t rtdetr-convert .")
        raise

    # Find the generated tflite file
    # onnx2tf v2.x generates both *_float32.tflite and *_float16.tflite
    tflite_files = [f for f in os.listdir(output_dir) if f.endswith('.tflite')]
    if not tflite_files:
        raise RuntimeError(f"onnx2tf completed but no .tflite file found in {output_dir}")

    # Select the right variant based on requested quantization
    if quantize == "fp16":
        fp16_files = [f for f in tflite_files if 'float16' in f]
        if fp16_files:
            src = os.path.join(output_dir, fp16_files[0])
            print(f"Using onnx2tf-generated FP16 model: {fp16_files[0]}")
        else:
            # Fallback: pick smallest file (likely FP16) or apply manual quantization
            src = os.path.join(output_dir, min(tflite_files, key=lambda f: os.path.getsize(os.path.join(output_dir, f))))
    else:
        fp32_files = [f for f in tflite_files if 'float32' in f]
        if fp32_files:
            src = os.path.join(output_dir, fp32_files[0])
        else:
            src = os.path.join(output_dir, tflite_files[0])

    dst = os.path.join(output_dir, "rtdetrv2_detector.tflite")
    if src != dst:
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)

    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"Saved: {dst} ({size_mb:.1f} MB)")

    return dst

    # NOTE: onnx-tf fallback was removed.
    # onnx-tf (onnx_tf.backend) was abandoned in 2022 (last release v1.10.0).
    # It uses `from onnx import mapping` which was removed in onnx >= 1.14.
    # There is no supported workaround. Use onnx2tf instead.


def apply_quantization(tflite_path: str, output_dir: str, mode: str):
    """Apply post-training quantization to existing TFLite model."""
    import tensorflow as tf
    import glob

    print(f"\nApplying {mode} quantization...")

    # onnx2tf saves SavedModel in various locations — search for it
    saved_model_path = None
    candidates = [
        os.path.join(output_dir, "saved_model"),
        output_dir,  # onnx2tf sometimes puts saved_model.pb directly here
    ]
    # Also search for any saved_model.pb recursively under output_dir
    for pb in glob.glob(os.path.join(output_dir, "**", "saved_model.pb"), recursive=True):
        candidates.insert(0, os.path.dirname(pb))

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "saved_model.pb")):
            saved_model_path = candidate
            break

    if saved_model_path is None:
        print(f"WARNING: SavedModel not found under {output_dir}. Skipping {mode} quantization.")
        print(f"The FP32 TFLite model at {tflite_path} is still usable.")
        return

    print(f"Found SavedModel at: {saved_model_path}")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    if mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    quantized_model = converter.convert()

    output_path = os.path.join(output_dir, f"rtdetrv2_detector_{mode}.tflite")
    with open(output_path, 'wb') as f:
        f.write(quantized_model)

    original_size = os.path.getsize(tflite_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original: {original_size:.1f} MB -> Quantized: {quantized_size:.1f} MB ({mode})")

    # Replace original with quantized version
    os.replace(output_path, tflite_path)
    print(f"Replaced {tflite_path} with {mode} quantized version")


def validate_tflite(tflite_path: str):
    """Validate converted TFLite model."""
    import tensorflow as tf

    print(f"\nValidating: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input tensors:")
    for inp in input_details:
        print(f"  {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")

    print("Output tensors:")
    for out in output_details:
        print(f"  {out['name']}: shape={out['shape']}, dtype={out['dtype']}")

    # Run dummy inference
    input_shape = input_details[0]['shape']
    dummy_input = np.random.rand(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"\nDummy inference output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

    size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    print("Validation PASSED")


def main():
    parser = argparse.ArgumentParser(description="Convert RT-DETRv2 ONNX to TFLite")
    parser.add_argument("--onnx-path", required=True, help="Path to ONNX model")
    parser.add_argument("--output-dir", default="./converted_models", help="Output directory")
    parser.add_argument("--quantize", choices=["none", "fp16", "int8"], default="fp16",
                       help="Quantization mode (default: fp16)")
    parser.add_argument("--validate", action="store_true", help="Validate after conversion")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose onnx2tf output for debugging")

    args = parser.parse_args()

    check_environment()

    if not os.path.exists(args.onnx_path):
        print(f"Error: ONNX model not found: {args.onnx_path}")
        sys.exit(1)

    tflite_path = convert_onnx_to_tflite(
        args.onnx_path, args.output_dir, args.quantize, verbose=args.verbose
    )

    if args.validate and tflite_path:
        validate_tflite(tflite_path)

    print("\nDone! Copy the .tflite file to:")
    print("  mobile/android/app/src/main/assets/models/rtdetrv2_detector.tflite")


if __name__ == "__main__":
    main()
