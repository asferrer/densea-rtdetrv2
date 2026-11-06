"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS

########################################
# Main training function
########################################

def main(args) -> None:
    """Train RT-DETRv2 from a YAML configuration."""
    # Distributed setup
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # Only one of from_scratch, resume or tuning is allowed
    assert not all([args.tuning, args.resume]), \
        'Only one of from_scratch, resume or tuning is supported at a time'

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ['update'] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg:', cfg.__dict__)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    dataset_train = cfg.train_dataloader.dataset

    print("Analyzing the training dataset...")
    # Optional: save a visual analysis of the training dataset
    dataset_train.analyze_dataset(
        save_path=f"{cfg.output_dir}/train_dataset_analysis.png",
        split_name="Training"
    )

    print("Analyzing the validation dataset...")
    dataset_val = cfg.val_dataloader.dataset
    dataset_val.analyze_dataset(
        save_path=f"{cfg.output_dir}/val_dataset_analysis.png",
        split_name="Validation"
    )

    # Create the solver from the task defined in the YAML
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='Fine-tune from checkpoint')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device (GPU, CPU, etc.)')
    parser.add_argument('--seed', type=int, default=21, help='Random seed for reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='Automatic mixed precision training')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--summary-dir', type=str, help='Directory for tensorboard summaries')
    parser.add_argument('--test-only', action='store_true', default=False, help='Only run validation')

    # YAML configuration overrides
    parser.add_argument('-u', '--update', nargs='+', help='Update YAML configuration')

    # Environment arguments
    parser.add_argument('--print-method', type=str, default='builtin', help='Print method')
    parser.add_argument('--print-rank', type=int, default=0, help='Rank ID to print from')
    parser.add_argument('--local-rank', type=int, help='Local rank ID')

    args = parser.parse_args()
    main(args)
