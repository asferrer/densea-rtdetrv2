"""
Experiments module for RT-DETRv2 synthetic vs real data analysis.

This module provides tools to:
1. Prepare datasets with different synthetic/real ratios
2. Generate training configurations
3. Run experiments automatically
4. Collect and visualize results

Main entry point:
    python experiments/run_full_pipeline.py

Individual scripts:
    python experiments/prepare_datasets.py    # Generate merged datasets
    python experiments/generate_configs.py    # Generate YAML configs
    python experiments/run_experiments.py     # Run training + evaluation
    python experiments/collect_results.py     # Aggregate results
"""

from .config import EXPERIMENTS, REAL_IMAGE_DIRS, SYNTHETIC_IMAGES_DIR

__all__ = ['EXPERIMENTS', 'REAL_IMAGE_DIRS', 'SYNTHETIC_IMAGES_DIR']
