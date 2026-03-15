#!/usr/bin/env python3
"""
Generate YAML configuration files for all experiments.

This script creates one YAML config file per experiment, based on the
template (config_template.yml) and the successful baseline configuration.

Usage:
    python experiments/generate_configs.py [--experiment N] [--dry-run]
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    EXPERIMENTS,
    REAL_ONLY_EXPERIMENTS,
    ALL_EXPERIMENTS,
    REAL_IMAGE_DIRS,
    REAL_VAL_JSON,
    MAX_EPOCHS,
    OUTPUT_BASE_DIR,
    NUM_CLASSES,
    BATCH_SIZE,
    SEED,
    # Versioned dataset support
    SYNTHETIC_DATASETS,
    DEFAULT_SYNTH_VERSION,
    get_synth_config,
    get_available_synth_versions,
    get_versioned_output_paths,
    get_experiment_output_paths,
    experiment_uses_synthetic,
    get_real_only_output_paths,
    # Class filtering support
    get_num_classes_for_experiment,
    get_val_json_path_for_experiment,
    get_test_json_path_for_experiment,
    needs_class_filtering,
    # Experiment type classification
    get_experiment_type,
    # Batch mode support
    BATCH_MODES,
    DEFAULT_BATCH_MODE,
)


# =============================================================================
# YAML GENERATION
# =============================================================================

def format_image_dirs_yaml(dirs: List[str], indent: int = 6) -> str:
    """
    Format a list of image directories as YAML list items.

    Args:
        dirs: List of directory paths
        indent: Number of spaces for indentation

    Returns:
        YAML-formatted string
    """
    prefix = " " * indent + "- "
    return "\n".join(prefix + d for d in dirs)


def get_image_dirs_for_experiment(
    exp: Dict,
    synth_version: str = DEFAULT_SYNTH_VERSION
) -> List[str]:
    """
    Get the list of image directories for an experiment.

    Args:
        exp: Experiment configuration dict
        synth_version: Synthetic dataset version to use

    Returns:
        List of image directory paths
    """
    synth_ratio = exp['synth_ratio']
    real_ratio = exp['real_ratio']

    dirs = []

    # Add real image directories if using any real data
    if real_ratio > 0:
        dirs.extend(REAL_IMAGE_DIRS)

    # Add synthetic image directory ONLY if synth_ratio > 0
    # This excludes synthetic directory for real-only experiments (exp11-exp14)
    if synth_ratio not in [0, 0.0, None]:
        # Get version-specific synthetic image directory
        synth_config = get_synth_config(synth_version)
        dirs.append(synth_config['images_dir'])

    return dirs


def generate_config_content(
    exp: Dict,
    template: str,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE,
    eval_mode: bool = False
) -> str:
    """
    Generate the YAML config content for an experiment.

    Args:
        exp: Experiment configuration dict
        template: Template content
        synth_version: Synthetic dataset version to use
        batch_mode: Batch sampling mode ('standard' or 'balanced')
        eval_mode: If True, use test JSON instead of val JSON for evaluation

    Returns:
        Generated YAML content
    """
    exp_name = exp['name']

    # Determine paths based on experiment type and batch mode
    if experiment_uses_synthetic(exp):
        # Versioned paths for experiments using synthetic data
        versioned_paths = get_versioned_output_paths(synth_version, batch_mode)
        # Dataset path is always from the shared datasets dir (not batch_mode dependent)
        train_anno_path = f"/app/RT-DETR/experiment_output/{synth_version}/datasets/{exp_name}/train.json"
        # Training output dir uses batch_mode
        output_dir = f"./{versioned_paths['training']}/{exp_name}"
        version_label = f" [version: {synth_version}]"
    else:
        # Shared paths for real-only experiments
        real_only_paths = get_real_only_output_paths(exp_name, batch_mode)
        train_anno_path = f"/app/RT-DETR/experiment_output/real_only/datasets/{exp_name}/train.json"
        output_dir = f"./{real_only_paths['training']}"
        version_label = " [real-only]"

    # Get image directories (version-aware)
    image_dirs = get_image_dirs_for_experiment(exp, synth_version)
    image_dirs_yaml = format_image_dirs_yaml(image_dirs)

    # For validation, always use real image directories
    val_image_dirs_yaml = format_image_dirs_yaml(REAL_IMAGE_DIRS)

    # Get num_classes based on experiment and synthetic version
    num_classes = get_num_classes_for_experiment(exp, synth_version)

    # Get annotation path for val_dataloader (val for training, test for evaluation)
    if eval_mode:
        val_anno_path = get_test_json_path_for_experiment(exp, synth_version)
    else:
        val_anno_path = get_val_json_path_for_experiment(exp, synth_version)

    # Check if class filtering is applied
    uses_filtered_val = experiment_uses_synthetic(exp) and needs_class_filtering(synth_version)

    # Determine if balanced domain sampling is needed
    # Only inject DomainBalancedSampler in 'balanced' mode for mixed experiments
    exp_type = get_experiment_type(exp)
    if batch_mode == 'balanced' and exp_type == 'mixed':
        shuffle_value = "False"
        sampler_config = (
            f"  sampler:\n"
            f"    type: DomainBalancedSampler\n"
            f"    batch_size: {BATCH_SIZE}\n"
            f"    seed: {SEED}"
        )
        balanced_label = "DomainBalancedSampler (50/50 real/synthetic per batch)"
    else:
        shuffle_value = "True"
        sampler_config = ""
        balanced_label = "Standard random sampling"

    # Calculate relative include prefix from config dir back to project root
    # standard:  experiment_output/<version>/configs/          → 3 levels → ../../..
    # balanced:  experiment_output/<version>/balanced/configs/  → 4 levels → ../../../..
    if batch_mode == 'standard':
        include_prefix = '../../..'
    else:
        include_prefix = '../../../..'

    # Replace placeholders
    content = template
    content = content.replace("${INCLUDE_PREFIX}", include_prefix)
    content = content.replace("${OUTPUT_DIR}", output_dir)
    content = content.replace("${EPOCHS}", str(MAX_EPOCHS))
    content = content.replace("${IMAGE_DIRS}", image_dirs_yaml)
    content = content.replace("${VAL_IMAGE_DIRS}", val_image_dirs_yaml)
    content = content.replace("${TRAIN_ANNO_PATH}", train_anno_path)
    content = content.replace("${VAL_ANNO_PATH}", val_anno_path)
    content = content.replace("${NUM_CLASSES}", str(num_classes))
    content = content.replace("${SHUFFLE}", shuffle_value)
    content = content.replace("${SAMPLER_CONFIG}", sampler_config)

    # Add experiment header with version info
    header = f"""# =============================================================================
# Experiment: {exp_name}{version_label}
# Description: {exp['description']}
# Generated by: experiments/generate_configs.py
# =============================================================================
#
# Synthetic version: {synth_version if experiment_uses_synthetic(exp) else 'N/A (real-only)'}
# Synthetic ratio: {exp['synth_ratio']}
# Real ratio: {exp['real_ratio']}
# Strategy: {exp['strategy']}
# Batch mode: {batch_mode}
# Batch sampling: {balanced_label}
#
# Number of classes: {num_classes}{' (filtered)' if uses_filtered_val else ''}
# Train annotation: {train_anno_path}
# Val annotation: {val_anno_path}{' (filtered)' if uses_filtered_val else ''}
# Output directory: {output_dir}
# =============================================================================

"""
    return header + content


def load_template() -> str:
    """Load the config template file."""
    template_path = Path(__file__).parent / "config_template.yml"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_config(content: str, output_path: Path) -> None:
    """Save config content to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  Saved: {output_path}")


def generate_configs(
    experiments: Optional[List[int]] = None,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    dry_run: bool = False,
    real_only: bool = False,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, str]:
    """
    Generate YAML configs for all (or specified) experiments.

    Args:
        experiments: List of experiment IDs to generate (None = all)
        synth_version: Synthetic dataset version to use
        dry_run: If True, don't save files
        real_only: If True, generate only real-only baseline experiments
        batch_mode: Batch sampling mode ('standard' or 'balanced')

    Returns:
        Dict mapping experiment name to output path
    """
    # Select experiment list based on mode
    if real_only:
        exp_list = REAL_ONLY_EXPERIMENTS
    else:
        exp_list = ALL_EXPERIMENTS

    # Load template
    print("Loading config template...")
    template = load_template()

    # Get versioned output paths (with batch_mode)
    versioned_paths = get_versioned_output_paths(synth_version, batch_mode)

    output_paths = {}

    for exp in exp_list:
        if experiments is not None and exp['id'] not in experiments:
            continue

        exp_name = exp['name']
        print(f"\nGenerating config for: {exp_name}")

        try:
            content = generate_config_content(exp, template, synth_version, batch_mode)

            # Determine output path based on experiment type and batch mode
            if experiment_uses_synthetic(exp):
                # Versioned path for experiments using synthetic data
                output_path = Path(versioned_paths['configs']) / f"{exp_name}.yml"
            else:
                # Real-only path (batch_mode aware)
                real_only_paths = get_real_only_output_paths(exp_name, batch_mode)
                output_path = real_only_paths['config']

            output_paths[exp_name] = str(output_path)

            if not dry_run:
                save_config(content, output_path)
            else:
                print(f"  [DRY RUN] Would save to: {output_path}")

            # Generate eval config (uses test.json instead of val.json)
            eval_content = generate_config_content(
                exp, template, synth_version, batch_mode, eval_mode=True
            )
            eval_path = output_path.with_stem(f"{exp_name}_eval")

            if not dry_run:
                save_config(eval_content, eval_path)
            else:
                print(f"  [DRY RUN] Would save eval config to: {eval_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return output_paths


# =============================================================================
# VALIDATION
# =============================================================================

def validate_configs(configs_dir: Path) -> bool:
    """
    Validate generated configs by attempting to load them.

    Args:
        configs_dir: Directory containing config files

    Returns:
        True if all configs are valid
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from src.core.yaml_config import YAMLConfig
    except ImportError:
        print("WARNING: Could not import YAMLConfig for validation")
        return True

    all_valid = True
    config_files = list(configs_dir.glob("*.yml"))

    print(f"\nValidating {len(config_files)} config files...")

    for config_path in config_files:
        try:
            # Just try to load - don't instantiate components
            YAMLConfig(str(config_path))
            print(f"  OK: {config_path.name}")
        except Exception as e:
            print(f"  FAIL: {config_path.name} - {e}")
            all_valid = False

    return all_valid


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML configs for synthetic vs real experiments"
    )
    parser.add_argument(
        '--experiment', '-e',
        type=int,
        nargs='+',
        help='Experiment ID(s) to generate. Default: all'
    )
    parser.add_argument(
        '--synth-version', '-v',
        choices=get_available_synth_versions(),
        default=DEFAULT_SYNTH_VERSION,
        help=f'Synthetic dataset version to use (default: {DEFAULT_SYNTH_VERSION})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't save files, just print what would be done"
    )
    parser.add_argument(
        '--real-only',
        action='store_true',
        help="Generate only real-only baseline experiments (exp11-exp14)"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help="Validate generated configs after creation"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List all experiments and exit"
    )
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help="List available synthetic dataset versions and exit"
    )
    parser.add_argument(
        '--batch-mode',
        choices=BATCH_MODES,
        default=DEFAULT_BATCH_MODE,
        help=f'Batch sampling mode (default: {DEFAULT_BATCH_MODE})'
    )

    args = parser.parse_args()

    if args.list_versions:
        print("\nAvailable synthetic dataset versions:")
        print("-" * 70)
        for version, config in SYNTHETIC_DATASETS.items():
            print(f"  {version}: {config['name']}")
            print(f"    Images: {config['images_dir']}")
        return

    if args.list:
        print("\nSynthetic + Real experiments:")
        print("-" * 70)
        for exp in EXPERIMENTS:
            print(f"  {exp['id']:2d}. {exp['name']}: {exp['description']}")
        print("\nReal-only baseline experiments:")
        print("-" * 70)
        for exp in REAL_ONLY_EXPERIMENTS:
            print(f"  {exp['id']:2d}. {exp['name']}: {exp['description']}")
        return

    print("\n" + "="*70)
    print("GENERATING YAML CONFIGURATIONS")
    print(f"Synthetic version: {args.synth_version}")
    print(f"Batch mode: {args.batch_mode}")
    print("="*70)

    output_paths = generate_configs(
        experiments=args.experiment,
        synth_version=args.synth_version,
        dry_run=args.dry_run,
        real_only=args.real_only,
        batch_mode=args.batch_mode
    )

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if output_paths:
        print(f"\nGenerated {len(output_paths)} config files:")
        for name, path in output_paths.items():
            print(f"  {name}: {path}")

        if args.validate and not args.dry_run:
            versioned_paths = get_versioned_output_paths(args.synth_version, args.batch_mode)
            configs_dir = Path(versioned_paths['configs'])
            if validate_configs(configs_dir):
                print("\nAll configs validated successfully!")
            else:
                print("\nSome configs failed validation!")

    else:
        print("\nNo configs generated.")

    print("\nNext step: Run 'python experiments/run_experiments.py' to start training")


if __name__ == '__main__':
    main()
