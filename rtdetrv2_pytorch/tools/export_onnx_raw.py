# Copyright(c) 2023 lyuwenyu. All Rights Reserved.
# Modified: Export raw model outputs WITHOUT postprocessor for TFLite conversion.
#
# The standard export_onnx.py bakes RTDETRPostProcessor into the ONNX graph,
# which uses torchvision.ops.box_convert, torch.topk, and torch.gather on int64
# tensors - all problematic for TFLite. This script exports only the core model
# so postprocessing is done in Kotlin on Android.
#
# Outputs:
#   - pred_logits: (1, num_queries, num_classes) float32  [raw logits, apply sigmoid]
#   - pred_boxes:  (1, num_queries, 4)           float32  [cxcywh normalized 0-1]

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import torch.nn as nn
from torch.ao.quantization import fuse_modules
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from src.core import YAMLConfig
    from src.nn.backbone.common import FrozenBatchNorm2d
except ImportError:
    logging.error("Could not import YAMLConfig from src.core. Ensure the project structure and PYTHONPATH are correct.")
    sys.exit(1)


def replace_frozen_bn_with_bn(model: nn.Module) -> nn.Module:
    """Recursively replaces FrozenBatchNorm2d with standard nn.BatchNorm2d in eval mode."""
    for name, module in model.named_children():
        if isinstance(module, FrozenBatchNorm2d):
            new_bn = nn.BatchNorm2d(module.num_features)
            new_bn.load_state_dict(module.state_dict())
            new_bn.eval()
            setattr(model, name, new_bn)
            logging.info(f"Replaced FrozenBatchNorm2d at: {name}")
        else:
            replace_frozen_bn_with_bn(module)
    return model


# Fusion candidates (same as export_onnx.py)
POTENTIAL_FUSION_CANDIDATES = [
    ['backbone.conv1.conv1_1.conv', 'backbone.conv1.conv1_1.norm'],
    ['backbone.conv1.conv1_2.conv', 'backbone.conv1.conv1_2.norm'],
    ['backbone.conv1.conv1_3.conv', 'backbone.conv1.conv1_3.norm'],
    ['backbone.res_layers.0.blocks.0.short.conv', 'backbone.res_layers.0.blocks.0.short.norm'],
    ['backbone.res_layers.0.blocks.0.branch2a.conv', 'backbone.res_layers.0.blocks.0.branch2a.norm'],
    ['backbone.res_layers.0.blocks.0.branch2b.conv', 'backbone.res_layers.0.blocks.0.branch2b.norm'],
    ['backbone.res_layers.0.blocks.1.branch2a.conv', 'backbone.res_layers.0.blocks.1.branch2a.norm'],
    ['backbone.res_layers.0.blocks.1.branch2b.conv', 'backbone.res_layers.0.blocks.1.branch2b.norm'],
    ['backbone.res_layers.1.blocks.0.short.conv.conv', 'backbone.res_layers.1.blocks.0.short.conv.norm'],
    ['backbone.res_layers.1.blocks.0.branch2a.conv', 'backbone.res_layers.1.blocks.0.branch2a.norm'],
    ['backbone.res_layers.1.blocks.0.branch2b.conv', 'backbone.res_layers.1.blocks.0.branch2b.norm'],
    ['backbone.res_layers.1.blocks.1.branch2a.conv', 'backbone.res_layers.1.blocks.1.branch2a.norm'],
    ['backbone.res_layers.1.blocks.1.branch2b.conv', 'backbone.res_layers.1.blocks.1.branch2b.norm'],
    ['backbone.res_layers.2.blocks.0.short.conv.conv', 'backbone.res_layers.2.blocks.0.short.conv.norm'],
    ['backbone.res_layers.2.blocks.0.branch2a.conv', 'backbone.res_layers.2.blocks.0.branch2a.norm'],
    ['backbone.res_layers.2.blocks.0.branch2b.conv', 'backbone.res_layers.2.blocks.0.branch2b.norm'],
    ['backbone.res_layers.2.blocks.1.branch2a.conv', 'backbone.res_layers.2.blocks.1.branch2a.norm'],
    ['backbone.res_layers.2.blocks.1.branch2b.conv', 'backbone.res_layers.2.blocks.1.branch2b.norm'],
    ['backbone.res_layers.3.blocks.0.short.conv.conv', 'backbone.res_layers.3.blocks.0.short.conv.norm'],
    ['backbone.res_layers.3.blocks.0.branch2a.conv', 'backbone.res_layers.3.blocks.0.branch2a.norm'],
    ['backbone.res_layers.3.blocks.0.branch2b.conv', 'backbone.res_layers.3.blocks.0.branch2b.norm'],
    ['backbone.res_layers.3.blocks.1.branch2a.conv', 'backbone.res_layers.3.blocks.1.branch2a.norm'],
    ['backbone.res_layers.3.blocks.1.branch2b.conv', 'backbone.res_layers.3.blocks.1.branch2b.norm'],
    ['decoder.input_proj.0.conv', 'decoder.input_proj.0.norm'],
    ['decoder.input_proj.1.conv', 'decoder.input_proj.1.norm'],
    ['decoder.input_proj.2.conv', 'decoder.input_proj.2.norm'],
    ['encoder.input_proj.0.conv', 'encoder.input_proj.0.norm'],
    ['encoder.input_proj.1.conv', 'encoder.input_proj.1.norm'],
    ['encoder.input_proj.2.conv', 'encoder.input_proj.2.norm'],
    ['encoder.lateral_convs.0.conv', 'encoder.lateral_convs.0.norm'],
    ['encoder.lateral_convs.1.conv', 'encoder.lateral_convs.1.norm'],
    ['encoder.fpn_blocks.0.conv1.conv', 'encoder.fpn_blocks.0.conv1.norm'],
    ['encoder.fpn_blocks.0.conv2.conv', 'encoder.fpn_blocks.0.conv2.norm'],
    ['encoder.fpn_blocks.0.bottlenecks.0.conv1.conv', 'encoder.fpn_blocks.0.bottlenecks.0.conv1.norm'],
    ['encoder.fpn_blocks.0.bottlenecks.0.conv2.conv', 'encoder.fpn_blocks.0.bottlenecks.0.conv2.norm'],
    ['encoder.fpn_blocks.0.bottlenecks.1.conv1.conv', 'encoder.fpn_blocks.0.bottlenecks.1.conv1.norm'],
    ['encoder.fpn_blocks.0.bottlenecks.1.conv2.conv', 'encoder.fpn_blocks.0.bottlenecks.1.conv2.norm'],
    ['encoder.fpn_blocks.0.bottlenecks.2.conv1.conv', 'encoder.fpn_blocks.0.bottlenecks.2.conv1.norm'],
    ['encoder.fpn_blocks.0.bottlenecks.2.conv2.conv', 'encoder.fpn_blocks.0.bottlenecks.2.conv2.norm'],
    ['encoder.fpn_blocks.0.conv3.conv', 'encoder.fpn_blocks.0.conv3.norm'],
    ['encoder.fpn_blocks.1.conv1.conv', 'encoder.fpn_blocks.1.conv1.norm'],
    ['encoder.fpn_blocks.1.conv2.conv', 'encoder.fpn_blocks.1.conv2.norm'],
    ['encoder.fpn_blocks.1.bottlenecks.0.conv1.conv', 'encoder.fpn_blocks.1.bottlenecks.0.conv1.norm'],
    ['encoder.fpn_blocks.1.bottlenecks.0.conv2.conv', 'encoder.fpn_blocks.1.bottlenecks.0.conv2.norm'],
    ['encoder.fpn_blocks.1.bottlenecks.1.conv1.conv', 'encoder.fpn_blocks.1.bottlenecks.1.conv1.norm'],
    ['encoder.fpn_blocks.1.bottlenecks.1.conv2.conv', 'encoder.fpn_blocks.1.bottlenecks.1.conv2.norm'],
    ['encoder.fpn_blocks.1.bottlenecks.2.conv1.conv', 'encoder.fpn_blocks.1.bottlenecks.2.conv1.norm'],
    ['encoder.fpn_blocks.1.bottlenecks.2.conv2.conv', 'encoder.fpn_blocks.1.bottlenecks.2.conv2.norm'],
    ['encoder.fpn_blocks.1.conv3.conv', 'encoder.fpn_blocks.1.conv3.norm'],
    ['encoder.downsample_convs.0.conv', 'encoder.downsample_convs.0.norm'],
    ['encoder.downsample_convs.1.conv', 'encoder.downsample_convs.1.norm'],
    ['encoder.pan_blocks.0.conv1.conv', 'encoder.pan_blocks.0.conv1.norm'],
    ['encoder.pan_blocks.0.conv2.conv', 'encoder.pan_blocks.0.conv2.norm'],
    ['encoder.pan_blocks.0.bottlenecks.0.conv1.conv', 'encoder.pan_blocks.0.bottlenecks.0.conv1.norm'],
    ['encoder.pan_blocks.0.bottlenecks.0.conv2.conv', 'encoder.pan_blocks.0.bottlenecks.0.conv2.norm'],
    ['encoder.pan_blocks.0.bottlenecks.1.conv1.conv', 'encoder.pan_blocks.0.bottlenecks.1.conv1.norm'],
    ['encoder.pan_blocks.0.bottlenecks.1.conv2.conv', 'encoder.pan_blocks.0.bottlenecks.1.conv2.norm'],
    ['encoder.pan_blocks.0.bottlenecks.2.conv1.conv', 'encoder.pan_blocks.0.bottlenecks.2.conv1.norm'],
    ['encoder.pan_blocks.0.bottlenecks.2.conv2.conv', 'encoder.pan_blocks.0.bottlenecks.2.conv2.norm'],
    ['encoder.pan_blocks.0.conv3.conv', 'encoder.pan_blocks.0.conv3.norm'],
    ['encoder.pan_blocks.1.conv1.conv', 'encoder.pan_blocks.1.conv1.norm'],
    ['encoder.pan_blocks.1.conv2.conv', 'encoder.pan_blocks.1.conv2.norm'],
    ['encoder.pan_blocks.1.bottlenecks.0.conv1.conv', 'encoder.pan_blocks.1.bottlenecks.0.conv1.norm'],
    ['encoder.pan_blocks.1.bottlenecks.0.conv2.conv', 'encoder.pan_blocks.1.bottlenecks.0.conv2.norm'],
    ['encoder.pan_blocks.1.bottlenecks.1.conv1.conv', 'encoder.pan_blocks.1.bottlenecks.1.conv1.norm'],
    ['encoder.pan_blocks.1.bottlenecks.1.conv2.conv', 'encoder.pan_blocks.1.bottlenecks.1.conv2.norm'],
    ['encoder.pan_blocks.1.bottlenecks.2.conv1.conv', 'encoder.pan_blocks.1.bottlenecks.2.conv1.norm'],
    ['encoder.pan_blocks.1.bottlenecks.2.conv2.conv', 'encoder.pan_blocks.1.bottlenecks.2.conv2.norm'],
    ['encoder.pan_blocks.1.conv3.conv', 'encoder.pan_blocks.1.conv3.norm'],
]


def main(args):
    logging.info(f"Loading configuration from: {args.config}")
    cfg = YAMLConfig(args.config, resume=args.resume if hasattr(args, 'resume') else None)

    # --- Model Loading ---
    if args.resume:
        logging.info(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

        if 'ema' in checkpoint and checkpoint['ema'] is not None:
            logging.info("Using EMA model weights.")
            state = checkpoint['ema']['module'] if 'module' in checkpoint['ema'] else checkpoint['ema']
        elif 'model' in checkpoint:
            logging.info("Using base model weights.")
            state = checkpoint['model']
        else:
            logging.error("Could not find 'ema' or 'model' in the checkpoint.")
            sys.exit(1)

        try:
            cfg.model.load_state_dict(state)
            logging.info("Checkpoint weights loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading state_dict: {e}. Check compatibility.")
            traceback.print_exc()
            sys.exit(1)
    else:
        logging.warning("No checkpoint provided (`--resume`). Using default initialized weights.")

    # --- Raw Model Wrapper (NO postprocessor) ---
    class RawDeployModel(nn.Module):
        """Wraps the core model WITHOUT the postprocessor.

        Outputs raw pred_logits and pred_boxes tensors that are simple
        float32 arrays, fully compatible with TFLite conversion.
        """
        def __init__(self, model_config, perform_fusion=False):
            super().__init__()
            logging.info("Preparing RAW model (no postprocessor) for deployment...")

            self.model = model_config.model
            self.model.eval()

            # Replace FrozenBatchNorm2d
            logging.info("Replacing FrozenBatchNorm2d layers...")
            self.model = replace_frozen_bn_with_bn(self.model)

            # Perform Conv+BN fusion if requested
            if perform_fusion:
                logging.info("Attempting dynamic layer fusion (Conv+BN)...")
                validated = []
                for conv_path, bn_path in POTENTIAL_FUSION_CANDIDATES:
                    try:
                        conv_mod = self.model.get_submodule(conv_path)
                        bn_mod = self.model.get_submodule(bn_path)
                        if isinstance(conv_mod, nn.Conv2d) and isinstance(bn_mod, nn.BatchNorm2d):
                            validated.append([conv_path, bn_path])
                    except AttributeError:
                        pass

                if validated:
                    logging.info(f"Found {len(validated)} valid module pairs to fuse.")
                    try:
                        self.model = fuse_modules(self.model, validated, inplace=True)
                        self.model.eval()
                        logging.info("Fusion completed successfully.")
                    except Exception as e:
                        logging.error(f"Fusion error: {e}")
                        traceback.print_exc()
                        logging.warning("Continuing without fusion.")
                else:
                    logging.warning("No valid modules for fusion.")
            else:
                logging.info("Layer fusion skipped.")

            logging.info("Raw model prepared for export.")

        def forward(self, images):
            """Forward pass returning raw logits and boxes.

            Returns:
                pred_logits: (batch, num_queries, num_classes) - raw logits (apply sigmoid for scores)
                pred_boxes:  (batch, num_queries, 4) - cxcywh normalized [0,1]
            """
            outputs = self.model(images)
            return outputs['pred_logits'], outputs['pred_boxes']

    # Create raw deployment model
    try:
        deploy_model = RawDeployModel(cfg, perform_fusion=args.fuse_layers)
        deploy_model.eval()
    except Exception as e:
        logging.error(f"Failed to initialize RawDeployModel: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Dummy Input (single input: images only) ---
    batch_size = 1
    input_shape = (batch_size, 3, args.input_height, args.input_width)
    dummy_images = torch.randn(input_shape, dtype=torch.float32)
    logging.info(f"Dummy input shape: {dummy_images.shape}")

    # --- Verify forward pass ---
    logging.info("Running test forward pass...")
    with torch.no_grad():
        pred_logits, pred_boxes = deploy_model(dummy_images)
    logging.info(f"  pred_logits shape: {pred_logits.shape}")  # Expected: (1, 150, 8)
    logging.info(f"  pred_boxes shape:  {pred_boxes.shape}")   # Expected: (1, 150, 4)

    # --- ONNX Export ---
    output_onnx_file = args.output_file
    input_names = ['images']
    output_names = ['pred_logits', 'pred_boxes']
    dynamic_axes = {
        'images': {0: 'N'},
        'pred_logits': {0: 'N'},
        'pred_boxes': {0: 'N'},
    }

    logging.info(f"Exporting RAW model to ONNX: {output_onnx_file}")
    logging.info(f"  Input:  {input_names} -> {list(dummy_images.shape)}")
    logging.info(f"  Output: {output_names} -> logits {list(pred_logits.shape)}, boxes {list(pred_boxes.shape)}")

    try:
        torch.onnx.export(
            deploy_model,
            dummy_images,  # Single input (no orig_target_sizes)
            output_onnx_file,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            verbose=False,
            do_constant_folding=True,
            export_params=True,
        )
        logging.info("ONNX export completed successfully.")
    except Exception as e:
        logging.error(f"Error during ONNX export: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Verification ---
    if args.check:
        logging.info(f"Verifying ONNX model: {output_onnx_file}")
        try:
            import onnx
            onnx_model = onnx.load(output_onnx_file)
            onnx.checker.check_model(onnx_model)
            logging.info("ONNX model verification PASSED.")

            # Print I/O info
            for inp in onnx_model.graph.input:
                shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
                logging.info(f"  Input:  {inp.name} -> {shape}")
            for out in onnx_model.graph.output:
                shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
                logging.info(f"  Output: {out.name} -> {shape}")

        except Exception as e:
            logging.error(f"ONNX verification failed: {e}")
            traceback.print_exc()

    # --- Simplification ---
    if args.simplify:
        try:
            import onnx
            import onnxsim
        except ImportError:
            logging.error("Install onnx and onnxsim: pip install onnx onnxsim")
            sys.exit(1)

        simplified_file = output_onnx_file.replace('.onnx', '_simplified.onnx')
        if simplified_file == output_onnx_file:
            simplified_file = output_onnx_file + '.simplified'
        logging.info(f"Simplifying ONNX model -> {simplified_file}")

        try:
            onnx_model = onnx.load(output_onnx_file)
            test_input_shapes = {'images': list(dummy_images.shape)}
            model_simplified, check = onnxsim.simplify(
                onnx_model,
                test_input_shapes=test_input_shapes,
            )
            if check:
                onnx.save(model_simplified, simplified_file)
                logging.info(f"Simplified model saved: {simplified_file}")
                # Also overwrite the original with the simplified version
                onnx.save(model_simplified, output_onnx_file)
                logging.info(f"Original file overwritten with simplified model: {output_onnx_file}")
            else:
                logging.warning("Simplification did not improve the model.")
        except Exception as e:
            logging.error(f"Simplification error: {e}")
            traceback.print_exc()

    # --- Summary ---
    file_size_mb = os.path.getsize(output_onnx_file) / (1024 * 1024)
    logging.info(f"\n{'='*60}")
    logging.info(f"Export complete: {output_onnx_file} ({file_size_mb:.1f} MB)")
    logging.info(f"  Input:  images {list(dummy_images.shape)} float32")
    logging.info(f"  Output: pred_logits {list(pred_logits.shape)} float32 (raw, apply sigmoid)")
    logging.info(f"  Output: pred_boxes  {list(pred_boxes.shape)} float32 (cxcywh normalized)")
    logging.info(f"{'='*60}")
    logging.info(f"\nNext step: Convert to TFLite:")
    logging.info(f"  python convert_rtdetr.py --onnx-path {output_onnx_file} --quantize fp16 --validate")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Export RT-DETRv2 to ONNX WITHOUT postprocessor (raw outputs for TFLite)"
    )
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to model YAML config.')
    parser.add_argument('--resume', '-r', type=str,
                        help='Path to checkpoint (.pth).')
    parser.add_argument('--output_file', '-o', type=str, default='rtdetrv2_raw.onnx',
                        help='Output ONNX filename.')
    parser.add_argument('--check', action='store_true',
                        help='Verify ONNX model after export.')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify with onnx-simplifier.')
    parser.add_argument('--input-height', type=int, default=640,
                        help='Input height.')
    parser.add_argument('--input-width', type=int, default=640,
                        help='Input width.')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version.')
    parser.add_argument('--fuse-layers', action='store_true',
                        help='Fuse Conv+BN layers before export.')

    args = parser.parse_args()
    main(args)
