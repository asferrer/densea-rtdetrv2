"""
Centralized configuration for synthetic vs real data experiments.

This module defines:
- Dataset paths (real and synthetic)
- Experiment definitions
- Training hyperparameters
- Class information

Update SYNTHETIC_TRAIN_JSON and SYNTHETIC_IMAGES_DIR when the new
synthetic dataset with 8 classes (including Mask) is ready.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

# =============================================================================
# DATASET PATHS (relative to /app/RT-DETR inside Docker)
# =============================================================================

# Real dataset paths
REAL_TRAIN_JSON = "/app/RT-DETR/split_v8/train.json"
REAL_VAL_JSON = "/app/RT-DETR/split_v8/val.json"
REAL_TEST_JSON = "/app/RT-DETR/split_v8/test.json"

# =============================================================================
# SYNTHETIC DATASET VERSIONS
# =============================================================================

# Registry of available synthetic dataset versions
# Each entry includes:
#   - name: Human-readable name
#   - train_json: Path to annotation file (Docker path)
#   - images_dir: Path to images directory
#   - description: Brief description
#   - classes: Dict mapping synth class ID -> class name
#   - class_mapping: Dict mapping synth class ID -> real dataset class ID
#   - num_classes: Number of classes in this synthetic dataset

SYNTHETIC_DATASETS: Dict[str, Dict[str, Any]] = {
    'cleansea': {
        'name': 'CleanSea Synthetic',
        'train_json': '/app/RT-DETR/dataset/dataset_sintetico_cleansea/synthetic_dataset.json',
        'images_dir': './dataset/dataset_sintetico_cleansea/images',
        'description': 'Synthetic data from CleanSea domain (7 classes, no Mask)',
        'num_classes': 7,
        'classes': {
            0: 'Bottle',
            1: 'Can',
            2: 'Fishing_Net',
            3: 'Glove',
            4: 'Metal_Debris',
            5: 'Plastic_Debris',
            6: 'Tire',
        },
        # Maps synthetic class ID -> real dataset class ID
        'class_mapping': {
            0: 0,  # Bottle -> Bottle
            1: 1,  # Can -> Can
            2: 2,  # Fishing_Net -> Fishing_Net
            3: 3,  # Glove -> Glove
            4: 5,  # Metal_Debris -> Metal_Debris (real ID 5)
            5: 6,  # Plastic_Debris -> Plastic_Debris (real ID 6)
            6: 7,  # Tire -> Tire (real ID 7)
        },
        # Real class IDs that are NOT in this synthetic dataset
        'missing_real_classes': [4],  # Mask
    },
    'densea': {
        'name': 'DenSea Synthetic',
        'train_json': '/app/RT-DETR/dataset/dataset_sintetico_densea/synthetic_dataset.json',
        'images_dir': './dataset/dataset_sintetico_densea/images',
        'description': 'Enhanced DenSea synthetic dataset (8 classes, full alignment)',
        'num_classes': 8,
        'classes': {
            0: 'Bottle',
            1: 'Can',
            2: 'Fishing_Net',
            3: 'Glove',
            4: 'Mask',
            5: 'Metal_Debris',
            6: 'Plastic_Debris',
            7: 'Tire',
        },
        # Maps synthetic class ID -> real dataset class ID (1:1 alignment)
        'class_mapping': {
            0: 0,  # Bottle -> Bottle
            1: 1,  # Can -> Can
            2: 2,  # Fishing_Net -> Fishing_Net
            3: 3,  # Glove -> Glove
            4: 4,  # Mask -> Mask
            5: 5,  # Metal_Debris -> Metal_Debris
            6: 6,  # Plastic_Debris -> Plastic_Debris
            7: 7,  # Tire -> Tire
        },
        'missing_real_classes': [],  # Full alignment
    },
    'taco': {
        'name': 'TACO Synthetic',
        'train_json': '/app/RT-DETR/dataset/dataset_sintetico_taco/synthetic_dataset.json',
        'images_dir': './dataset/dataset_sintetico_taco/images',
        'description': 'Synthetic data from TACO debris dataset (6 classes, no Mask/Tire)',
        'num_classes': 6,
        'classes': {
            0: 'Bottle',
            1: 'Can',
            2: 'Fishing_Net',
            3: 'Glove',
            4: 'Metal_Debris',
            5: 'Plastic_Debris',
        },
        # Maps synthetic class ID -> real dataset class ID
        'class_mapping': {
            0: 0,  # Bottle -> Bottle
            1: 1,  # Can -> Can
            2: 2,  # Fishing_Net -> Fishing_Net
            3: 3,  # Glove -> Glove
            4: 5,  # Metal_Debris -> Metal_Debris (real ID 5)
            5: 6,  # Plastic_Debris -> Plastic_Debris (real ID 6)
        },
        # Real class IDs that are NOT in this synthetic dataset
        'missing_real_classes': [4, 7],  # Mask, Tire
    },
}

# Default synthetic version (for backward compatibility)
DEFAULT_SYNTH_VERSION = 'densea'

# Legacy variables (for backward compatibility - use SYNTHETIC_DATASETS instead)
SYNTHETIC_TRAIN_JSON = SYNTHETIC_DATASETS[DEFAULT_SYNTH_VERSION]['train_json']
SYNTHETIC_IMAGES_DIR = SYNTHETIC_DATASETS[DEFAULT_SYNTH_VERSION]['images_dir']

# Real image directories (for MultiDirCocoDetection)
REAL_IMAGE_DIRS = [
    "./dataset/cleansea_dataset/CocoFormatDataset/train_coco/",
    "./dataset/Ocean_garbage/train/",
    "./dataset/Ocean_garbage/valid/",
    "./dataset/Ocean_garbage/test/",
    "./dataset/Neural_Ocean/train/",
    "./dataset/Neural_Ocean/valid/",
    "./dataset/Neural_Ocean/test/",
    "./dataset/cleansea_dataset/CocoFormatDataset/test_coco/",
]

# Output directories (base)
OUTPUT_BASE_DIR = "./experiment_output"

# Legacy output directories (for backward compatibility - use versioned functions instead)
DATASETS_OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/datasets"
CONFIGS_OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/configs"
TRAINING_OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/training"
EVALUATION_OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/evaluation"
RESULTS_OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/results"

# Cross-version comparison output directory
COMPARISON_OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/comparison"

# =============================================================================
# CLASS INFORMATION
# =============================================================================

# Class definitions (0-indexed, matching real dataset)
CLASSES = {
    0: "Bottle",
    1: "Can",
    2: "Fishing_Net",
    3: "Glove",
    4: "Mask",
    5: "Metal_Debris",
    6: "Plastic_Debris",
    7: "Tire",
}

CLASS_NAMES = list(CLASSES.values())
NUM_CLASSES = len(CLASSES)

# Minority classes (< 2% of annotations in real data)
MINORITY_CLASSES = ["Can", "Metal_Debris"]
MINORITY_CLASS_IDS = [1, 5]

# Real data class distribution (from split_v8/train.json)
REAL_CLASS_COUNTS = {
    "Bottle": 4597,
    "Can": 396,
    "Fishing_Net": 2824,
    "Glove": 2609,
    "Mask": 3118,
    "Metal_Debris": 99,
    "Plastic_Debris": 4025,
    "Tire": 3712,
}

# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "exp01_synth100",
        "synth_ratio": 1.0,
        "real_ratio": 0.0,
        "strategy": "none",
        "description": "100% synthetic only - baseline for synthetic data",
    },
    {
        "id": 2,
        "name": "exp02_synth100_real10",
        "synth_ratio": 1.0,
        "real_ratio": 0.1,
        "strategy": "stratified",
        "description": "100% synthetic + 10% real (stratified sampling)",
    },
    {
        "id": 3,
        "name": "exp03_synth100_real25",
        "synth_ratio": 1.0,
        "real_ratio": 0.25,
        "strategy": "stratified",
        "description": "100% synthetic + 25% real (stratified sampling)",
    },
    {
        "id": 4,
        "name": "exp04_synth100_real50",
        "synth_ratio": 1.0,
        "real_ratio": 0.5,
        "strategy": "stratified",
        "description": "100% synthetic + 50% real (stratified sampling)",
    },
    {
        "id": 5,
        "name": "exp05_synth100_real75",
        "synth_ratio": 1.0,
        "real_ratio": 0.75,
        "strategy": "stratified",
        "description": "100% synthetic + 75% real (stratified sampling)",
    },
    {
        "id": 6,
        "name": "exp06_synth100_real100",
        "synth_ratio": 1.0,
        "real_ratio": 1.0,
        "strategy": "none",
        "description": "100% synthetic + 100% real - all data combined",
    },
    {
        "id": 7,
        "name": "exp07_real100",
        "synth_ratio": 0.0,
        "real_ratio": 1.0,
        "strategy": "none",
        "description": "100% real only - baseline for comparison",
    },
    {
        "id": 8,
        "name": "exp08_real_synth_minor",
        "synth_ratio": "minority",
        "real_ratio": 1.0,
        "strategy": "minority_only",
        "description": "100% real + synthetic only for Can and Metal_Debris",
    },
    {
        "id": 9,
        "name": "exp09_real_synth_prop",
        "synth_ratio": "proportional",
        "real_ratio": 1.0,
        "strategy": "inverse_proportional",
        "description": "100% real + synthetic inversely proportional to class count",
    },
]

# =============================================================================
# REAL-ONLY BASELINE EXPERIMENTS (for paired comparison)
# =============================================================================

REAL_ONLY_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "id": 11,
        "name": "exp11_real10",
        "synth_ratio": 0.0,
        "real_ratio": 0.1,
        "strategy": "stratified",
        "description": "10% real only - baseline for comparison with exp02",
        "compare_to": 2,
    },
    {
        "id": 12,
        "name": "exp12_real25",
        "synth_ratio": 0.0,
        "real_ratio": 0.25,
        "strategy": "stratified",
        "description": "25% real only - baseline for comparison with exp03",
        "compare_to": 3,
    },
    {
        "id": 13,
        "name": "exp13_real50",
        "synth_ratio": 0.0,
        "real_ratio": 0.5,
        "strategy": "stratified",
        "description": "50% real only - baseline for comparison with exp04",
        "compare_to": 4,
    },
    {
        "id": 14,
        "name": "exp14_real75",
        "synth_ratio": 0.0,
        "real_ratio": 0.75,
        "strategy": "stratified",
        "description": "75% real only - baseline for comparison with exp05",
        "compare_to": 5,
    },
]

# Combined list for unified iteration
ALL_EXPERIMENTS = EXPERIMENTS + REAL_ONLY_EXPERIMENTS

# Comparison pairs: real-only ID -> synth+real ID
COMPARISON_PAIRS = {
    11: 2,   # exp11_real10 vs exp02_synth100_real10
    12: 3,   # exp12_real25 vs exp03_synth100_real25
    13: 4,   # exp13_real50 vs exp04_synth100_real50
    14: 5,   # exp14_real75 vs exp05_synth100_real75
    7: 6,    # exp07_real100 vs exp06_synth100_real100
}

# Reverse mapping: synth+real ID -> real-only ID
SYNTH_TO_REAL_PAIRS = {v: k for k, v in COMPARISON_PAIRS.items() if v}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training configuration (from successful baseline rtdetrv2_r50vd_densea_v9_focal.yml)
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_METRIC = "coco_eval_bbox"

BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 21

# Optimizer settings
LEARNING_RATE = 0.00005
BACKBONE_LR = 0.000005  # 10x lower than main LR
WEIGHT_DECAY = 0.05

# Model settings
BACKBONE_DEPTH = 50
FREEZE_AT = 2
FREEZE_NORM = True

# =============================================================================
# BATCH SAMPLING MODES
# =============================================================================

# Available batch sampling modes:
#   - 'standard': Random sampling (original behavior, fully backward compatible)
#   - 'balanced': 50/50 real/synthetic per batch using DomainBalancedSampler
BATCH_MODES = ['standard', 'balanced']
DEFAULT_BATCH_MODE = 'standard'

# =============================================================================
# HELPER FUNCTIONS - VERSIONED
# =============================================================================

def get_synth_config(version: str) -> Dict[str, str]:
    """Get configuration for a specific synthetic dataset version.

    Args:
        version: Synthetic dataset version key ('cleansea', 'densea', 'taco')

    Returns:
        Dictionary with 'name', 'train_json', 'images_dir', 'description'

    Raises:
        ValueError: If version is not found in SYNTHETIC_DATASETS
    """
    if version not in SYNTHETIC_DATASETS:
        available = list(SYNTHETIC_DATASETS.keys())
        raise ValueError(f"Unknown synthetic version: '{version}'. Available: {available}")
    return SYNTHETIC_DATASETS[version].copy()


def get_available_synth_versions() -> List[str]:
    """Get list of available synthetic dataset version keys."""
    return list(SYNTHETIC_DATASETS.keys())


def get_version_output_dir(version: str) -> str:
    """Get versioned output base directory.

    Args:
        version: Synthetic dataset version key

    Returns:
        Path string like './experiment_output/densea'
    """
    return f"{OUTPUT_BASE_DIR}/{version}"


def get_versioned_output_paths(
    version: str,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, str]:
    """Get all output directory paths for a specific version and batch mode.

    Args:
        version: Synthetic dataset version key
        batch_mode: Batch sampling mode ('standard' or 'balanced')

    Returns:
        Dictionary with 'datasets', 'configs', 'training', 'evaluation', 'results' paths
    """
    base = get_version_output_dir(version)
    if batch_mode == 'standard':
        # Backward compatible: same paths as before
        return {
            'datasets': f"{base}/datasets",
            'configs': f"{base}/configs",
            'training': f"{base}/training",
            'evaluation': f"{base}/evaluation",
            'results': f"{base}/results",
        }
    else:
        # New mode: datasets are shared, rest goes under batch_mode subdirectory
        return {
            'datasets': f"{base}/datasets",
            'configs': f"{base}/{batch_mode}/configs",
            'training': f"{base}/{batch_mode}/training",
            'evaluation': f"{base}/{batch_mode}/evaluation",
            'results': f"{base}/{batch_mode}/results",
        }


# =============================================================================
# HELPER FUNCTIONS - EXPERIMENTS
# =============================================================================

def get_experiment_by_id(exp_id: int) -> Dict[str, Any]:
    """Get experiment configuration by ID."""
    for exp in ALL_EXPERIMENTS:
        if exp["id"] == exp_id:
            return exp
    raise ValueError(f"Experiment with ID {exp_id} not found")


def get_experiment_by_name(name: str) -> Dict[str, Any]:
    """Get experiment configuration by name."""
    for exp in ALL_EXPERIMENTS:
        if exp["name"] == name:
            return exp
    raise ValueError(f"Experiment with name '{name}' not found")


def get_all_image_dirs(
    include_synthetic: bool = True,
    synth_version: str = DEFAULT_SYNTH_VERSION
) -> List[str]:
    """Get list of all image directories for MultiDirCocoDetection.

    Args:
        include_synthetic: Whether to include synthetic image directory
        synth_version: Which synthetic dataset version to use

    Returns:
        List of image directory paths
    """
    dirs = REAL_IMAGE_DIRS.copy()
    if include_synthetic:
        synth_config = get_synth_config(synth_version)
        dirs.append(synth_config['images_dir'])
    return dirs


def get_experiment_output_paths(
    exp_name: str,
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, Path]:
    """Get output paths for an experiment with version and batch mode support.

    Args:
        exp_name: Experiment name (e.g., 'exp01_synth100')
        synth_version: Synthetic dataset version for output organization
        batch_mode: Batch sampling mode ('standard' or 'balanced')

    Returns:
        Dictionary with paths for dataset, config, training, evaluation, checkpoints
    """
    paths = get_versioned_output_paths(synth_version, batch_mode)
    return {
        "dataset": Path(paths['datasets']) / exp_name / "train.json",
        "config": Path(paths['configs']) / f"{exp_name}.yml",
        "eval_config": Path(paths['configs']) / f"{exp_name}_eval.yml",
        "training": Path(paths['training']) / exp_name,
        "evaluation": Path(paths['evaluation']) / exp_name,
        "checkpoint_best": Path(paths['training']) / exp_name / "best.pth",
        "checkpoint_last": Path(paths['training']) / exp_name / "last.pth",
    }


def get_comparison_pair(exp_id: int) -> Optional[int]:
    """Get the paired experiment ID for comparison.

    For real-only experiments, returns the corresponding synth+real experiment.
    For synth+real experiments, returns the corresponding real-only experiment.
    """
    # Check if it's a real-only experiment
    if exp_id in COMPARISON_PAIRS:
        return COMPARISON_PAIRS[exp_id]
    # Check if it's a synth+real experiment
    if exp_id in SYNTH_TO_REAL_PAIRS:
        return SYNTH_TO_REAL_PAIRS[exp_id]
    return None


def get_experiment_type(exp: Dict[str, Any]) -> str:
    """Classify experiment as 'real_only', 'synth_only', or 'mixed'.

    Args:
        exp: Experiment configuration dict

    Returns:
        One of: 'real_only', 'synth_only', 'mixed'
    """
    synth_ratio = exp.get('synth_ratio', 0)
    real_ratio = exp.get('real_ratio', 0)

    # Handle special string values
    if isinstance(synth_ratio, str):
        synth_ratio = 1.0  # "minority" or "proportional" means some synthetic

    if synth_ratio == 0 or synth_ratio == 0.0:
        return 'real_only'
    elif real_ratio == 0 or real_ratio == 0.0:
        return 'synth_only'
    else:
        return 'mixed'


def get_all_experiments(include_real_only: bool = True) -> List[Dict[str, Any]]:
    """Get all experiments, optionally including real-only baselines.

    Args:
        include_real_only: If True, include real-only baseline experiments

    Returns:
        List of experiment configuration dicts
    """
    if include_real_only:
        return ALL_EXPERIMENTS
    return EXPERIMENTS


def get_real_only_experiments() -> List[Dict[str, Any]]:
    """Get only the real-only baseline experiments."""
    return REAL_ONLY_EXPERIMENTS


def is_real_only_experiment(exp_id: int) -> bool:
    """Check if an experiment ID is a real-only baseline experiment."""
    return any(exp['id'] == exp_id for exp in REAL_ONLY_EXPERIMENTS)


def experiment_uses_synthetic(exp: Dict[str, Any]) -> bool:
    """Check if an experiment uses synthetic data.

    Args:
        exp: Experiment configuration dict

    Returns:
        True if experiment uses any synthetic data
    """
    synth_ratio = exp.get('synth_ratio', 0)
    if isinstance(synth_ratio, str):
        return True  # "minority" or "proportional" uses synthetic
    return synth_ratio > 0


def get_real_only_output_paths(
    exp_name: str,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, Path]:
    """Get output paths for real-only experiments (not versioned).

    Real-only experiments are stored in a shared 'real_only' directory
    since they don't depend on which synthetic dataset version is selected.

    Args:
        exp_name: Experiment name (e.g., 'exp11_real10')
        batch_mode: Batch sampling mode ('standard' or 'balanced')

    Returns:
        Dictionary with paths for dataset, config, training, evaluation, checkpoints
    """
    if batch_mode == 'standard':
        base = f"{OUTPUT_BASE_DIR}/real_only"
    else:
        base = f"{OUTPUT_BASE_DIR}/real_only/{batch_mode}"
    return {
        "dataset": Path(f"{OUTPUT_BASE_DIR}/real_only") / "datasets" / exp_name / "train.json",
        "config": Path(base) / "configs" / f"{exp_name}.yml",
        "eval_config": Path(base) / "configs" / f"{exp_name}_eval.yml",
        "training": Path(base) / "training" / exp_name,
        "evaluation": Path(base) / "evaluation" / exp_name,
        "checkpoint_best": Path(base) / "training" / exp_name / "best.pth",
        "checkpoint_last": Path(base) / "training" / exp_name / "last.pth",
    }


def get_experiment_paths_smart(
    exp: Dict[str, Any],
    synth_version: str = DEFAULT_SYNTH_VERSION,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, Path]:
    """Get output paths for an experiment, automatically choosing versioned or not.

    For experiments that use synthetic data, returns versioned paths.
    For real-only experiments, returns shared real_only paths.

    Args:
        exp: Experiment configuration dict
        synth_version: Synthetic dataset version (only used if experiment uses synthetic)
        batch_mode: Batch sampling mode ('standard' or 'balanced')

    Returns:
        Dictionary with paths for dataset, config, training, evaluation, checkpoints
    """
    if experiment_uses_synthetic(exp):
        return get_experiment_output_paths(exp['name'], synth_version, batch_mode)
    else:
        return get_real_only_output_paths(exp['name'], batch_mode)


# =============================================================================
# HELPER FUNCTIONS - CLASS MAPPING
# =============================================================================

def get_synth_num_classes(version: str) -> int:
    """Get the number of classes in a synthetic dataset version.

    Args:
        version: Synthetic dataset version key

    Returns:
        Number of classes (e.g., 7 for CleanSea, 8 for DenSea, 6 for TACO)
    """
    synth_config = get_synth_config(version)
    return synth_config.get('num_classes', NUM_CLASSES)


def get_synth_class_names(version: str) -> List[str]:
    """Get class names for a synthetic dataset version.

    Args:
        version: Synthetic dataset version key

    Returns:
        List of class names in order of class IDs
    """
    synth_config = get_synth_config(version)
    classes = synth_config.get('classes', CLASSES)
    return [classes[i] for i in sorted(classes.keys())]


def get_synth_to_real_mapping(version: str) -> Dict[int, int]:
    """Get mapping from synthetic class IDs to real dataset class IDs.

    Args:
        version: Synthetic dataset version key

    Returns:
        Dict mapping synth_class_id -> real_class_id
    """
    synth_config = get_synth_config(version)
    return synth_config.get('class_mapping', {i: i for i in range(NUM_CLASSES)})


def get_real_to_synth_mapping(version: str) -> Dict[int, int]:
    """Get mapping from real dataset class IDs to synthetic class IDs.

    Args:
        version: Synthetic dataset version key

    Returns:
        Dict mapping real_class_id -> synth_class_id
        Only includes classes that exist in the synthetic dataset.
    """
    synth_to_real = get_synth_to_real_mapping(version)
    return {v: k for k, v in synth_to_real.items()}


def get_missing_real_classes(version: str) -> List[int]:
    """Get list of real dataset class IDs missing from a synthetic dataset.

    Args:
        version: Synthetic dataset version key

    Returns:
        List of real class IDs not present in the synthetic dataset
    """
    synth_config = get_synth_config(version)
    return synth_config.get('missing_real_classes', [])


def get_valid_real_classes_for_synth(version: str) -> List[int]:
    """Get list of real dataset class IDs that are present in a synthetic dataset.

    These are the classes that should be considered for evaluation when
    training with this synthetic dataset version.

    Args:
        version: Synthetic dataset version key

    Returns:
        List of real class IDs that have corresponding synthetic classes
    """
    synth_to_real = get_synth_to_real_mapping(version)
    return sorted(synth_to_real.values())


def get_valid_real_class_names_for_synth(version: str) -> List[str]:
    """Get list of class names that are valid for evaluation with a synthetic version.

    Args:
        version: Synthetic dataset version key

    Returns:
        List of class names that exist in both real and synthetic datasets
    """
    valid_ids = get_valid_real_classes_for_synth(version)
    return [CLASSES[i] for i in valid_ids]


def needs_class_filtering(version: str) -> bool:
    """Check if a synthetic dataset version needs class filtering.

    Returns True if the synthetic dataset has fewer classes than the real dataset,
    meaning validation annotations need to be filtered.

    Args:
        version: Synthetic dataset version key

    Returns:
        True if class filtering is needed
    """
    return len(get_missing_real_classes(version)) > 0


def get_filtered_val_json_path(version: str) -> Path:
    """Get path for the filtered validation JSON for a synthetic version.

    When a synthetic dataset has fewer classes than the real dataset,
    we need to create a filtered validation file that only includes
    annotations for classes present in the synthetic dataset.

    Args:
        version: Synthetic dataset version key

    Returns:
        Path to the filtered validation JSON
    """
    paths = get_versioned_output_paths(version)
    return Path(paths['datasets']) / "val_filtered.json"


def get_filtered_test_json_path(version: str) -> Path:
    """Get path for the filtered test JSON for a synthetic version.

    When a synthetic dataset has fewer classes than the real dataset,
    we need to create a filtered test file that only includes
    annotations for classes present in the synthetic dataset.

    Args:
        version: Synthetic dataset version key

    Returns:
        Path to the filtered test JSON
    """
    paths = get_versioned_output_paths(version)
    return Path(paths['datasets']) / "test_filtered.json"


def print_class_mapping_info(version: str) -> None:
    """Print detailed class mapping information for debugging.

    Args:
        version: Synthetic dataset version key
    """
    synth_config = get_synth_config(version)

    print(f"\n{'='*60}")
    print(f"Class Mapping Info: {version}")
    print(f"{'='*60}")
    print(f"Dataset: {synth_config['name']}")
    print(f"Number of classes: {synth_config['num_classes']}")
    print(f"Description: {synth_config['description']}")

    print(f"\nSynthetic Classes:")
    classes = synth_config.get('classes', {})
    for idx in sorted(classes.keys()):
        print(f"  {idx}: {classes[idx]}")

    print(f"\nMapping to Real Dataset:")
    mapping = synth_config.get('class_mapping', {})
    for synth_id in sorted(mapping.keys()):
        real_id = mapping[synth_id]
        print(f"  Synth {synth_id} ({classes[synth_id]}) -> Real {real_id} ({CLASSES[real_id]})")

    missing = synth_config.get('missing_real_classes', [])
    if missing:
        print(f"\nMissing Real Classes (not in synthetic):")
        for real_id in missing:
            print(f"  Real {real_id}: {CLASSES[real_id]}")
    else:
        print(f"\nNo missing classes - full alignment with real dataset")

    print(f"{'='*60}\n")


# =============================================================================
# HELPER FUNCTIONS - ANNOTATION FILTERING
# =============================================================================

def filter_coco_annotations_by_classes(
    coco_data: Dict[str, Any],
    valid_class_ids: List[int],
    remap_category_ids: bool = True
) -> Dict[str, Any]:
    """Filter COCO annotations to only include specified classes.

    This function filters out annotations for classes not in the valid list
    and optionally remaps category IDs to be consecutive starting from 0.

    Args:
        coco_data: COCO format dict with 'images', 'annotations', 'categories'
        valid_class_ids: List of category IDs to keep (in real dataset numbering)
        remap_category_ids: If True, remap category IDs to consecutive 0-indexed

    Returns:
        Filtered COCO format dict
    """
    import copy

    filtered = copy.deepcopy(coco_data)

    # Filter categories
    filtered['categories'] = [
        cat for cat in coco_data.get('categories', [])
        if cat['id'] in valid_class_ids
    ]

    # Create ID remapping if needed
    if remap_category_ids:
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(valid_class_ids))}
        # Update category IDs
        for cat in filtered['categories']:
            cat['id'] = old_to_new[cat['id']]
    else:
        old_to_new = {cat_id: cat_id for cat_id in valid_class_ids}

    # Filter annotations
    filtered['annotations'] = [
        ann for ann in coco_data.get('annotations', [])
        if ann['category_id'] in valid_class_ids
    ]

    # Remap category IDs in annotations
    for ann in filtered['annotations']:
        ann['category_id'] = old_to_new[ann['category_id']]

    # Remove images without annotations
    image_ids_with_annotations = set(ann['image_id'] for ann in filtered['annotations'])
    filtered['images'] = [
        img for img in coco_data.get('images', [])
        if img['id'] in image_ids_with_annotations
    ]

    return filtered


def create_filtered_val_json_for_synth_version(
    real_val_json_path: str,
    synth_version: str,
    output_path: Optional[Path] = None
) -> Path:
    """Create a filtered validation JSON for a synthetic dataset version.

    This filters the real validation dataset to only include classes that
    exist in the specified synthetic dataset version.

    Args:
        real_val_json_path: Path to real validation JSON (can be Docker path)
        synth_version: Synthetic dataset version key
        output_path: Optional output path (defaults to versioned datasets dir)

    Returns:
        Path to the filtered validation JSON
    """
    import json

    # Resolve path (handle Docker paths)
    if real_val_json_path.startswith('/app/RT-DETR/'):
        local_path = Path(real_val_json_path.replace('/app/RT-DETR/', ''))
    else:
        local_path = Path(real_val_json_path)

    # Check if filtering is needed
    if not needs_class_filtering(synth_version):
        print(f"  {synth_version}: No filtering needed (all classes present)")
        return local_path

    # Load validation data
    with open(local_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Get valid class IDs for this synthetic version
    valid_class_ids = get_valid_real_classes_for_synth(synth_version)

    print(f"  Filtering validation for {synth_version}:")
    print(f"    Original categories: {len(val_data.get('categories', []))}")
    print(f"    Valid class IDs: {valid_class_ids}")
    print(f"    Original annotations: {len(val_data.get('annotations', []))}")

    # Filter annotations
    filtered_data = filter_coco_annotations_by_classes(
        val_data,
        valid_class_ids,
        remap_category_ids=True  # Remap to match synthetic dataset IDs
    )

    print(f"    Filtered categories: {len(filtered_data.get('categories', []))}")
    print(f"    Filtered annotations: {len(filtered_data.get('annotations', []))}")
    print(f"    Filtered images: {len(filtered_data.get('images', []))}")

    # Determine output path
    if output_path is None:
        output_path = get_filtered_val_json_path(synth_version)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save filtered data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f)

    print(f"    Saved to: {output_path}")
    return output_path


def create_filtered_test_json_for_synth_version(
    real_test_json_path: str,
    synth_version: str,
    output_path: Optional[Path] = None
) -> Path:
    """Create a filtered test JSON for a synthetic dataset version.

    This filters the real test dataset to only include classes that
    exist in the specified synthetic dataset version.

    Args:
        real_test_json_path: Path to real test JSON (can be Docker path)
        synth_version: Synthetic dataset version key
        output_path: Optional output path (defaults to versioned datasets dir)

    Returns:
        Path to the filtered test JSON
    """
    import json

    # Resolve path (handle Docker paths)
    if real_test_json_path.startswith('/app/RT-DETR/'):
        local_path = Path(real_test_json_path.replace('/app/RT-DETR/', ''))
    else:
        local_path = Path(real_test_json_path)

    # Check if filtering is needed
    if not needs_class_filtering(synth_version):
        print(f"  {synth_version}: No test filtering needed (all classes present)")
        return local_path

    # Load test data
    with open(local_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Get valid class IDs for this synthetic version
    valid_class_ids = get_valid_real_classes_for_synth(synth_version)

    print(f"  Filtering test set for {synth_version}:")
    print(f"    Original categories: {len(test_data.get('categories', []))}")
    print(f"    Valid class IDs: {valid_class_ids}")
    print(f"    Original annotations: {len(test_data.get('annotations', []))}")

    # Filter annotations
    filtered_data = filter_coco_annotations_by_classes(
        test_data,
        valid_class_ids,
        remap_category_ids=True  # Remap to match synthetic dataset IDs
    )

    print(f"    Filtered categories: {len(filtered_data.get('categories', []))}")
    print(f"    Filtered annotations: {len(filtered_data.get('annotations', []))}")
    print(f"    Filtered images: {len(filtered_data.get('images', []))}")

    # Determine output path
    if output_path is None:
        output_path = get_filtered_test_json_path(synth_version)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save filtered data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f)

    print(f"    Saved to: {output_path}")
    return output_path


def get_num_classes_for_experiment(
    exp: Dict[str, Any],
    synth_version: str = DEFAULT_SYNTH_VERSION
) -> int:
    """Get the correct number of classes for an experiment.

    For experiments using synthetic data, this returns the number of classes
    in the synthetic dataset. For real-only experiments, returns full class count.

    Args:
        exp: Experiment configuration dict
        synth_version: Synthetic dataset version

    Returns:
        Number of classes for the model
    """
    if experiment_uses_synthetic(exp):
        return get_synth_num_classes(synth_version)
    return NUM_CLASSES


def get_val_json_path_for_experiment(
    exp: Dict[str, Any],
    synth_version: str = DEFAULT_SYNTH_VERSION
) -> str:
    """Get the appropriate validation JSON path for an experiment.

    For experiments using synthetic data with fewer classes, returns the
    filtered validation JSON path. Otherwise returns the full validation JSON.

    Args:
        exp: Experiment configuration dict
        synth_version: Synthetic dataset version

    Returns:
        Path to validation JSON (Docker path format)
    """
    if experiment_uses_synthetic(exp) and needs_class_filtering(synth_version):
        # Return filtered validation path (Docker format with forward slashes)
        filtered_path = get_filtered_val_json_path(synth_version)
        # Convert to Docker-style path with forward slashes
        filtered_str = str(filtered_path).replace('\\', '/')
        return f"/app/RT-DETR/{filtered_str}"
    return REAL_VAL_JSON


def get_test_json_path_for_experiment(
    exp: Dict[str, Any],
    synth_version: str = DEFAULT_SYNTH_VERSION
) -> str:
    """Get the appropriate test JSON path for final evaluation of an experiment.

    For experiments using synthetic data with fewer classes, returns the
    filtered test JSON path. Otherwise returns the full test JSON.

    Args:
        exp: Experiment configuration dict
        synth_version: Synthetic dataset version

    Returns:
        Path to test JSON (Docker path format)
    """
    if experiment_uses_synthetic(exp) and needs_class_filtering(synth_version):
        # Return filtered test path (Docker format with forward slashes)
        filtered_path = get_filtered_test_json_path(synth_version)
        # Convert to Docker-style path with forward slashes
        filtered_str = str(filtered_path).replace('\\', '/')
        return f"/app/RT-DETR/{filtered_str}"
    return REAL_TEST_JSON
