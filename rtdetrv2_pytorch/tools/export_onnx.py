# Copyright(c) 2023 lyuwenyu. All Rights Reserved.

import os
import sys
# Add the parent directory to the path to find local modules like 'src'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import torch.nn as nn
# Import the specific fusion function
from torch.ao.quantization import fuse_modules
import onnx
import onnxsim
import logging # Use logging for more informative messages
import traceback # For detailed error printing

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import configuration (assuming it's in src.core)
try:
    from src.core import YAMLConfig
    from src.nn.backbone.common import FrozenBatchNorm2d
except ImportError:
    logging.error("Could not import YAMLConfig from src.core. Ensure the project structure and PYTHONPATH are correct.")
    sys.exit(1)

def replace_frozen_bn_with_bn(model: nn.Module) -> nn.Module:
    """
    Recursively replaces all instances of FrozenBatchNorm2d with a standard
    nn.BatchNorm2d layer in evaluation mode.
    """
    for name, module in model.named_children():
        if isinstance(module, FrozenBatchNorm2d):
            # Create a new standard BatchNorm2d layer
            new_bn = nn.BatchNorm2d(module.num_features)
            
            # Copy the state from the FrozenBatchNorm2d layer
            new_bn.load_state_dict(module.state_dict())
            
            # Set the new layer to evaluation mode to freeze its statistics
            new_bn.eval()
            
            # Replace the old layer with the new one
            setattr(model, name, new_bn)
            logging.info(f"Replaced FrozenBatchNorm2d with nn.BatchNorm2d at: {name}")
        else:
            # Recurse into submodules
            replace_frozen_bn_with_bn(module)
    return model

# --- Main Export Function ---
def main(args):
    """
    Main function to load the model, prepare it (with fusion), and export it to ONNX.
    """
    logging.info(f"Loading configuration from: {args.config}")
    cfg = YAMLConfig(args.config, resume=args.resume if hasattr(args, 'resume') else None)

    # --- Model Loading ---
    if args.resume:
        logging.info(f"Loading checkpoint from: {args.resume}")
        # Set weights_only=True for security unless you trust the source completely
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False) # Consider weights_only=True

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
            # Load state dict *before* fusion
            cfg.model.load_state_dict(state)
            logging.info("Checkpoint weights loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading state_dict: {e}. Check compatibility.")
            traceback.print_exc()
            sys.exit(1)
    else:
        logging.warning("No checkpoint provided (`--resume`). Using default initialized weights.")

    # --- Model Preparation for Deployment (Fusion included) ---
    class DeployModel(nn.Module):
        def __init__(self, model_config, postprocessor_config, perform_fusion=False):
            super().__init__()
            logging.info("Preparing model and postprocessor for deployment...")

            self.model = model_config.model
            self.postprocessor = getattr(postprocessor_config, 'postprocessor', None)

            # 1. Set to evaluation mode *before* fusion
            self.model.eval()
            if self.postprocessor:
                 self.postprocessor.eval()

            # 2. Replace FrozenBatchNorm2d before attempting fusion
            logging.info("Replacing FrozenBatchNorm2d layers with standard nn.BatchNorm2d in eval mode...")
            self.model = replace_frozen_bn_with_bn(self.model)

            # 3. Perform Layer Fusion if requested
            if perform_fusion:
                logging.info("Attempting dynamic layer fusion (Conv+BN)...")
                
                # List of potential candidates for fusion
                potential_fusion_candidates = [
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
                
                validated_modules_to_fuse = []
                for conv_path, bn_path in potential_fusion_candidates:
                    try:
                        # Use model.get_submodule() which is the correct way to get nested modules
                        conv_mod = self.model.get_submodule(conv_path)
                        bn_mod = self.model.get_submodule(bn_path)
                        
                        # Check if they are the correct types for fusion
                        if isinstance(conv_mod, nn.Conv2d) and isinstance(bn_mod, nn.BatchNorm2d):
                            validated_modules_to_fuse.append([conv_path, bn_path])
                        # else:
                        #     logging.debug(f"Skipping fusion for '{conv_path}': not a valid Conv/BN pair.")
                    except AttributeError:
                        # This will catch cases where a path like 'short.conv.conv' doesn't exist
                        # because 'short' might be an nn.Identity() module. This is expected.
                        # logging.warning(f"Could not find module at path '{conv_path}' or '{bn_path}'. Skipping.")
                        pass # It's normal for some paths not to exist, so we just skip them silently

                if validated_modules_to_fuse:
                    logging.info(f"Found {len(validated_modules_to_fuse)} valid module pairs to fuse.")
                    try:
                        self.model = fuse_modules(self.model, validated_modules_to_fuse, inplace=True)
                        self.model.eval()
                        logging.info("Fusion completed successfully.")
                    except Exception as e:
                        logging.error(f"An unexpected error occurred during fusion: {e}")
                        traceback.print_exc()
                        logging.warning("Continuing export without fusion.")
                else:
                    logging.warning("No valid modules found for fusion. Continuing without fusion.")
            else:
                logging.info("Layer fusion skipped as per arguments.")

            logging.info("Model and postprocessor prepared for export.")

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    # Create instance of the deployment-ready model
    try:
        # Pass perform_fusion flag based on args
        deploy_model = DeployModel(cfg, cfg, perform_fusion=args.fuse_layers)
        deploy_model.eval()
    except Exception as e:
        logging.error(f"Failed to initialize DeployModel: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Dummy Input Data ---
    batch_size = 1
    input_shape = (batch_size, 3, args.input_height, args.input_width)
    dummy_images = torch.randn(input_shape, dtype=torch.float32)
    # IMPORTANTE: orig_target_sizes debe ser [W, H] según la convención del postprocessor
    dummy_orig_sizes = torch.tensor([[args.input_width, args.input_height]] * batch_size, dtype=torch.int64)
    logging.info(f"Dummy input shape (images): {dummy_images.shape}")
    logging.info(f"Dummy input (orig_target_sizes): {dummy_orig_sizes.shape}, dtype={dummy_orig_sizes.dtype}")

    # --- ONNX Export Configuration ---
    output_onnx_file = args.output_file
    input_names = ['images', 'orig_target_sizes']
    output_names = ['labels', 'boxes', 'scores'] # Adjust if your model outputs differ
    dynamic_axes = {
        input_names[0]: {0: 'N'},
        input_names[1]: {0: 'N'},
        # Add dynamic axes for outputs if needed
    }

    # Opset version: Defaulting to 17 for better compatibility with PyTorch 2.5 & TRT 10.x
    opset_version = args.opset
    logging.info(f"Targeting TensorRT 10.3.0. Using ONNX Opset Version: {opset_version}")
    logging.info(f"Exporting model to ONNX: {output_onnx_file}")
    logging.info(f"  Input Names: {input_names}")
    logging.info(f"  Output Names: {output_names}")
    logging.info(f"  Dynamic Axes: {dynamic_axes}")

    # --- Export ---
    try:
        torch.onnx.export(
            deploy_model,
            (dummy_images, dummy_orig_sizes),
            output_onnx_file,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False,
            do_constant_folding=True,
            export_params=True,
        )
        logging.info("ONNX export completed successfully.")
    except Exception as e:
        logging.error(f"Error during ONNX export: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Optional Verification ---
    if args.check:
        logging.info(f"Verifying ONNX model: {output_onnx_file}")
        # Verification code remains the same...
        try:
            onnx_model = onnx.load(output_onnx_file)
            onnx.checker.check_model(onnx_model)
            logging.info("ONNX model verification passed.")
        except onnx.checker.ValidationError as e:
             logging.error(f"ONNX model verification failed: {e}")
        except Exception as e:
            logging.error(f"Error during ONNX model verification: {e}")
            traceback.print_exc()


    # --- Optional Simplification ---
    if args.simplify:
        # Simplification code remains the same...
        simplified_onnx_file = output_onnx_file.replace('.onnx', '_simplified.onnx')
        if simplified_onnx_file == output_onnx_file:
             simplified_onnx_file = output_onnx_file.replace('.onnx', '.simpl.onnx')
        logging.info(f"Simplifying ONNX model with onnx-simplifier...")
        logging.info(f"  Input file: {output_onnx_file}")
        logging.info(f"  Output file: {simplified_onnx_file}")
        try:
            onnx_model = onnx.load(output_onnx_file)
            test_input_shapes = {
                input_names[0]: list(dummy_images.shape),
                input_names[1]: list(dummy_orig_sizes.shape)
            }
            logging.info(f"  Using test_input_shapes for simplification: {test_input_shapes}")
            model_simplified, check = onnxsim.simplify(
                onnx_model,
                test_input_shapes=test_input_shapes
            )
            if check:
                onnx.save(model_simplified, simplified_onnx_file)
                logging.info(f"Simplified ONNX model saved to: {simplified_onnx_file}")
            else:
                logging.warning("ONNX model simplification failed or did not produce changes.")
        except Exception as e:
            logging.error(f"Error during ONNX model simplification: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX with optional fusion for TensorRT")

    # Existing arguments
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the model YAML configuration file.')
    parser.add_argument('--resume', '-r', type=str, help='Path to the checkpoint file (.pth) to load weights.')
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx', help='Name for the output ONNX file.')
    parser.add_argument('--check', action='store_true', help='Verify the ONNX model after export.')
    parser.add_argument('--simplify', action='store_true', help='Simplify the ONNX model using onnx-simplifier.')

    # Added/Improved arguments
    parser.add_argument('--input-height', type=int, default=640, help='Input image height for export.')
    parser.add_argument('--input-width', type=int, default=640, help='Input image width for export.')
    # Defaulting opset to 17 for TRT 10.x compatibility
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version (e.g., 13, 16, 17). Check TensorRT compatibility.')
    # Argument to enable/disable fusion
    parser.add_argument('--fuse-layers', action='store_true', help='Attempt to fuse Conv+BN layers before exporting.')

    args = parser.parse_args()

    # Dependency checks remain the same...
    if args.fuse_layers:
         try:
             from torch.ao.quantization import fuse_modules
         except ImportError:
             logging.error("`torch.ao.quantization` needed for --fuse-layers. Update PyTorch if necessary.")
             sys.exit(1)

    if args.simplify:
        try:
            import onnxsim
        except ImportError:
            logging.error("Install 'onnx-simplifier' for --simplify: pip install onnxsim")
            sys.exit(1)
    if args.check or args.simplify:
         try:
            import onnx
         except ImportError:
            logging.error("Install 'onnx' for --check or --simplify: pip install onnx")
            sys.exit(1)

    main(args)
