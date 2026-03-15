#!/usr/bin/env python3
"""
Main orchestrator for running all synthetic vs real experiments.

This script:
1. Runs training for each experiment sequentially
2. Evaluates each model on the real test set
3. Saves state for resume capability
4. Collects results after each experiment

Usage:
    # Run all experiments from the beginning
    python experiments/run_experiments.py

    # Run from a specific experiment
    python experiments/run_experiments.py --start-from 5

    # Resume from last incomplete experiment (auto-detects checkpoints)
    python experiments/run_experiments.py --resume

    # Resume and select which experiment to continue from
    python experiments/run_experiments.py --resume --select

    # Run a single experiment
    python experiments/run_experiments.py --only 3
"""

import json
import shutil
import subprocess
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from config import (
    EXPERIMENTS,
    REAL_ONLY_EXPERIMENTS,
    ALL_EXPERIMENTS,
    OUTPUT_BASE_DIR,
    SEED,
    # Versioned dataset support
    SYNTHETIC_DATASETS,
    DEFAULT_SYNTH_VERSION,
    get_available_synth_versions,
    get_versioned_output_paths,
    get_experiment_output_paths,
    experiment_uses_synthetic,
    get_real_only_output_paths,
    get_experiment_type,
    # Batch mode support
    BATCH_MODES,
    DEFAULT_BATCH_MODE,
)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class ExperimentState:
    """Manages experiment progress state for resume capability."""

    def __init__(
        self,
        output_dir: str,
        synth_version: str = DEFAULT_SYNTH_VERSION,
        batch_mode: str = DEFAULT_BATCH_MODE
    ):
        self.output_dir = Path(output_dir)
        self.synth_version = synth_version
        self.batch_mode = batch_mode
        # Version and batch-mode specific state file
        if batch_mode == 'standard':
            self.state_file = self.output_dir / f"experiment_state_{synth_version}.json"
        else:
            self.state_file = self.output_dir / f"experiment_state_{synth_version}_{batch_mode}.json"
        self.results: Dict[str, Dict] = {}
        self.load()

    def load(self) -> None:
        """Load state from file if it exists."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.results = data.get('results', {})
            print(f"Loaded state from {self.state_file}")
            print(f"  Version: {self.synth_version}")
            print(f"  Batch mode: {self.batch_mode}")
            print(f"  Completed experiments: {list(self.results.keys())}")

    def save(self) -> None:
        """Save current state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'last_updated': datetime.now().isoformat(),
            'synth_version': self.synth_version,
            'batch_mode': self.batch_mode,
            'results': self.results,
        }

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def mark_completed(self, exp_name: str, result: Dict) -> None:
        """Mark an experiment as completed."""
        self.results[exp_name] = {
            'completed_at': datetime.now().isoformat(),
            **result
        }
        self.save()

    def mark_failed(self, exp_name: str, error: str) -> None:
        """Mark an experiment as failed."""
        self.results[exp_name] = {
            'failed_at': datetime.now().isoformat(),
            'error': error,
            'status': 'failed'
        }
        self.save()

    def is_completed(self, exp_name: str) -> bool:
        """Check if an experiment is completed."""
        result = self.results.get(exp_name, {})
        return result.get('status') != 'failed' and 'completed_at' in result

    def get_last_completed_id(self, include_real_only: bool = True) -> int:
        """Get the ID of the last successfully completed experiment.

        Args:
            include_real_only: If True, include real-only experiments in the search
        """
        exp_list = ALL_EXPERIMENTS if include_real_only else EXPERIMENTS
        completed_ids = []
        for exp in exp_list:
            if self.is_completed(exp['name']):
                completed_ids.append(exp['id'])
        return max(completed_ids) if completed_ids else 0


# =============================================================================
# CHECKPOINT DETECTION
# =============================================================================

def get_experiment_checkpoint_status(
    exp: Dict,
    synth_version: str,
    batch_mode: str = DEFAULT_BATCH_MODE
) -> Dict[str, Any]:
    """
    Check checkpoint status for an experiment.

    Returns dict with:
        has_last: bool - last.pth exists (partial training)
        has_best: bool - best.pth exists
        last_path: Path or None
        best_path: Path or None
    """
    if experiment_uses_synthetic(exp):
        paths = get_experiment_output_paths(exp['name'], synth_version, batch_mode)
    else:
        paths = get_real_only_output_paths(exp['name'], batch_mode)

    last_path = paths['checkpoint_last']
    best_path = paths['checkpoint_best']

    return {
        'has_last': last_path.exists(),
        'has_best': best_path.exists(),
        'last_path': last_path if last_path.exists() else None,
        'best_path': best_path if best_path.exists() else None,
    }


# =============================================================================
# CROSS-BATCH-MODE REUSE
# =============================================================================

def is_batch_mode_invariant(exp: Dict) -> bool:
    """Check if an experiment produces the same results regardless of batch_mode.

    Experiments that only use real data or only use synthetic data are not
    affected by DomainBalancedSampler, so their results are identical
    across batch modes.
    """
    exp_type = get_experiment_type(exp)
    return exp_type in ('real_only', 'synth_only')


def find_completed_in_standard_mode(
    exp: Dict,
    synth_version: str,
) -> Optional[Dict[str, Path]]:
    """Check if a batch-mode-invariant experiment was already completed in standard mode.

    Returns the standard-mode paths dict if best.pth exists, None otherwise.
    Also checks all other synth versions for real-only experiments, since they
    are version-independent.
    """
    # Project root: resolve OUTPUT_BASE_DIR relative to /app/RT-DETR
    project_root = Path(__file__).resolve().parent.parent

    if experiment_uses_synthetic(exp):
        # Synth-only: check same version, standard batch_mode
        paths = get_experiment_output_paths(exp['name'], synth_version, 'standard')
        best = project_root / paths['checkpoint_best']
        if best.exists():
            return {k: project_root / v for k, v in paths.items()}
    else:
        # Real-only: check standard batch_mode (paths are version-independent)
        paths = get_real_only_output_paths(exp['name'], 'standard')
        best = project_root / paths['checkpoint_best']
        if best.exists():
            return {k: project_root / v for k, v in paths.items()}

    return None


def reuse_from_standard_mode(
    exp: Dict,
    synth_version: str,
    batch_mode: str,
) -> Optional[Dict[str, Any]]:
    """Reuse training/evaluation results from standard mode via symlinks.

    For batch-mode-invariant experiments (real-only, synth-only), creates
    symlinks from the current batch_mode paths to the standard-mode outputs,
    avoiding redundant training.

    Returns a result dict (like run_single_experiment) if reused, None otherwise.
    """
    if batch_mode == 'standard':
        return None

    if not is_batch_mode_invariant(exp):
        return None

    std_paths = find_completed_in_standard_mode(exp, synth_version)
    if std_paths is None:
        return None

    project_root = Path(__file__).resolve().parent.parent

    # Get the current batch_mode paths (resolved to absolute)
    if experiment_uses_synthetic(exp):
        cur_paths_rel = get_experiment_output_paths(exp['name'], synth_version, batch_mode)
    else:
        cur_paths_rel = get_real_only_output_paths(exp['name'], batch_mode)
    cur_paths = {k: project_root / v for k, v in cur_paths_rel.items()}

    # Create symlinks for training and evaluation directories
    for key in ('training', 'evaluation'):
        src = std_paths[key].resolve()
        dst = cur_paths[key]

        if dst.is_symlink():
            # Already a symlink (from a previous reuse) — skip
            continue

        if dst.exists():
            # Real directory exists (from a failed run) — check if it has
            # useful content (best.pth). If not, replace with symlink.
            if not (dst / 'best.pth').exists():
                shutil.rmtree(dst)
            else:
                continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src, dst)
        print(f"  Symlinked: {dst} -> {src}")

    return {
        'exp_id': exp['id'],
        'exp_name': exp['name'],
        'description': exp['description'],
        'training_dir': str(std_paths['training']),
        'evaluation_dir': str(std_paths['evaluation']),
        'checkpoint': str(std_paths['checkpoint_best']),
        'status': 'completed',
        'reused_from': 'standard',
        'duration_seconds': 0,
        'duration_formatted': '0:00:00 (reused)',
    }


# =============================================================================
# TRAINING
# =============================================================================

def run_training(
    config_path: Path,
    output_dir: Path,
    device: str = 'cuda',
    seed: int = SEED,
    use_amp: bool = True,
    resume_from: Optional[Path] = None
) -> bool:
    """
    Run training for a single experiment.

    Args:
        config_path: Path to the YAML config file
        output_dir: Output directory for training
        device: Device to use (cuda/cpu)
        seed: Random seed
        use_amp: Whether to use automatic mixed precision
        resume_from: Path to checkpoint to resume from (e.g., last.pth)

    Returns:
        True if training completed successfully
    """
    # Build command
    cmd = [
        sys.executable, 'tools/train.py',
        '-c', str(config_path),
        '-d', device,
        '--seed', str(seed),
    ]

    if resume_from is not None:
        cmd.extend(['-r', str(resume_from)])

    if use_amp:
        cmd.append('--use-amp')

    # Create log file
    log_dir = output_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train.log'

    # When resuming, append to log instead of overwriting
    log_mode = 'a' if resume_from is not None else 'w'

    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Log file: {log_file}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("-" * 60)

    # Run training
    try:
        with open(log_file, log_mode, encoding='utf-8') as log:
            if resume_from is not None:
                log.write(f"\n{'='*60}\n")
                log.write(f"RESUMED at {datetime.now().isoformat()}\n")
                log.write(f"Checkpoint: {resume_from}\n")
                log.write(f"{'='*60}\n\n")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent.parent)  # rtdetrv2_pytorch dir
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log.write(line)
                log.flush()

            process.wait()

        if process.returncode != 0:
            print(f"\nTraining failed with return code: {process.returncode}")
            return False

        return True

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        process.terminate()
        return False

    except Exception as e:
        print(f"\nTraining error: {e}")
        return False


# =============================================================================
# EVALUATION
# =============================================================================

def run_evaluation(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str = 'cuda'
) -> bool:
    """
    Run evaluation on the test set.

    Args:
        config_path: Path to the YAML config file
        checkpoint_path: Path to the model checkpoint
        output_dir: Output directory for evaluation results
        device: Device to use

    Returns:
        True if evaluation completed successfully
    """
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False

    # Build command
    cmd = [
        sys.executable, 'tools/eval.py',
        '-c', str(config_path),
        '-r', str(checkpoint_path),
        '--output_dir', str(output_dir),
        '-d', device,
        '--coco_eval',
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )

        return result.returncode == 0

    except Exception as e:
        print(f"\nEvaluation error: {e}")
        return False


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Main experiment runner class."""

    def __init__(
        self,
        synth_version: str = DEFAULT_SYNTH_VERSION,
        device: str = 'cuda',
        use_amp: bool = True,
        batch_mode: str = DEFAULT_BATCH_MODE
    ):
        self.synth_version = synth_version
        self.device = device
        self.use_amp = use_amp
        self.batch_mode = batch_mode

        # Get versioned results directory (batch_mode aware)
        versioned_paths = get_versioned_output_paths(synth_version, batch_mode)
        self.state = ExperimentState(versioned_paths['results'], synth_version, batch_mode)

    def _get_exp_paths(self, exp: Dict) -> Dict[str, Path]:
        """Get paths for an experiment based on its type and batch mode."""
        if experiment_uses_synthetic(exp):
            return get_experiment_output_paths(exp['name'], self.synth_version, self.batch_mode)
        else:
            return get_real_only_output_paths(exp['name'], self.batch_mode)

    def get_experiment_status_detail(self, exp: Dict) -> str:
        """Get detailed status string for an experiment."""
        exp_name = exp['name']

        if self.state.is_completed(exp_name):
            return "DONE"

        result = self.state.results.get(exp_name, {})
        if result.get('status') == 'failed':
            error = result.get('error', 'unknown')
            # Check if there's a partial checkpoint despite failure
            ckpt = get_experiment_checkpoint_status(exp, self.synth_version, self.batch_mode)
            if ckpt['has_last']:
                return f"FAILED (checkpoint available)"
            return f"FAILED ({error})"

        # Not in state - check for partial training
        ckpt = get_experiment_checkpoint_status(exp, self.synth_version, self.batch_mode)
        if ckpt['has_last']:
            return "PARTIAL (last.pth found)"

        # Check if reusable from standard mode
        if self.batch_mode != 'standard' and is_batch_mode_invariant(exp):
            std_paths = find_completed_in_standard_mode(exp, self.synth_version)
            if std_paths is not None:
                return "REUSABLE (from standard mode)"

        return "PENDING"

    def find_resumable_experiments(
        self,
        include_real_only: bool = True
    ) -> List[Tuple[Dict, Path]]:
        """
        Find experiments that can be resumed from a checkpoint.

        Returns list of (experiment_dict, checkpoint_path) tuples.
        """
        exp_list = ALL_EXPERIMENTS if include_real_only else EXPERIMENTS
        resumable = []

        for exp in exp_list:
            if self.state.is_completed(exp['name']):
                continue

            ckpt = get_experiment_checkpoint_status(exp, self.synth_version, self.batch_mode)
            if ckpt['has_last']:
                resumable.append((exp, ckpt['last_path']))

        return resumable

    def find_first_incomplete(
        self,
        include_real_only: bool = True
    ) -> Optional[int]:
        """
        Find the first experiment that is not completed.

        Returns experiment ID or None if all are completed.
        """
        exp_list = ALL_EXPERIMENTS if include_real_only else EXPERIMENTS

        for exp in exp_list:
            if not self.state.is_completed(exp['name']):
                return exp['id']

        return None

    def run_single_experiment(
        self,
        exp: Dict,
        resume_training: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single experiment (training + evaluation).

        Args:
            exp: Experiment configuration dict
            resume_training: If True, resume from last.pth if available

        Returns:
            Result dict with status and paths
        """
        exp_name = exp['name']
        paths = self._get_exp_paths(exp)

        result = {
            'exp_id': exp['id'],
            'exp_name': exp_name,
            'description': exp['description'],
        }

        # Check if config exists
        config_path = paths['config']
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Check if dataset exists
        dataset_path = paths['dataset']
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Detect checkpoint for resume
        resume_from = None
        if resume_training:
            ckpt = get_experiment_checkpoint_status(exp, self.synth_version, self.batch_mode)
            if ckpt['has_last']:
                resume_from = ckpt['last_path']
                print(f"\n  Checkpoint detected: {resume_from}")
                print(f"  Resuming training from last checkpoint...")

        # Phase 1: Training
        print(f"\n{'='*60}")
        if resume_from:
            print(f"[1/2] RESUMING TRAINING: {exp_name}")
        else:
            print(f"[1/2] TRAINING: {exp_name}")
        print(f"{'='*60}")

        training_dir = paths['training']
        train_success = run_training(
            config_path=config_path,
            output_dir=training_dir,
            device=self.device,
            seed=SEED,
            use_amp=self.use_amp,
            resume_from=resume_from
        )

        if not train_success:
            result['status'] = 'training_failed'
            return result

        result['training_dir'] = str(training_dir)

        # Phase 2: Evaluation (uses eval config with test split if available)
        print(f"\n{'='*60}")
        print(f"[2/2] EVALUATION: {exp_name}")
        print(f"{'='*60}")

        checkpoint_path = paths['checkpoint_best']
        eval_dir = paths['evaluation']

        # Use eval config (test.json) if available, otherwise fall back to training config
        eval_config_path = paths.get('eval_config')
        if eval_config_path is not None and eval_config_path.exists():
            print(f"  Using eval config (test split): {eval_config_path}")
        else:
            eval_config_path = config_path
            print(f"  Eval config not found, using training config: {config_path}")

        eval_success = run_evaluation(
            config_path=eval_config_path,
            checkpoint_path=checkpoint_path,
            output_dir=eval_dir,
            device=self.device
        )

        if not eval_success:
            result['status'] = 'evaluation_failed'
            return result

        result['evaluation_dir'] = str(eval_dir)
        result['checkpoint'] = str(checkpoint_path)
        result['status'] = 'completed'

        return result

    def eval_single_experiment(self, exp: Dict) -> Dict[str, Any]:
        """
        Run evaluation only (no training) for an experiment with existing checkpoint.

        Uses the eval config (with test split) if available, otherwise falls back
        to the training config.

        Args:
            exp: Experiment configuration dict

        Returns:
            Result dict with status and paths
        """
        exp_name = exp['name']
        paths = self._get_exp_paths(exp)

        checkpoint_path = paths['checkpoint_best']
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found: {checkpoint_path}")

        # Use eval config (test.json) if available, otherwise fall back to training config
        eval_config_path = paths.get('eval_config')
        if eval_config_path is not None and eval_config_path.exists():
            config_to_use = eval_config_path
        else:
            config_to_use = paths['config']
            print(f"  Warning: Eval config not found, using training config")

        eval_dir = paths['evaluation']

        print(f"\n{'='*60}")
        print(f"EVALUATION ONLY: {exp_name}")
        print(f"  Config: {config_to_use}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        eval_success = run_evaluation(
            config_path=config_to_use,
            checkpoint_path=checkpoint_path,
            output_dir=eval_dir,
            device=self.device
        )

        result = {
            'exp_id': exp['id'],
            'exp_name': exp_name,
            'description': exp['description'],
            'evaluation_dir': str(eval_dir),
            'checkpoint': str(checkpoint_path),
            'status': 'completed' if eval_success else 'evaluation_failed',
        }
        return result

    def eval_all(
        self,
        start_from: int = 1,
        only: Optional[int] = None,
        real_only: bool = False,
        exclude_real_only: bool = False,
    ) -> None:
        """
        Run evaluation only for all experiments that have a trained checkpoint.

        This does NOT re-train any models. It only evaluates existing best.pth
        checkpoints using the eval config (test split).

        Args:
            start_from: Experiment ID to start from
            only: If set, run only this experiment ID
            real_only: If True, run only real-only baseline experiments
            exclude_real_only: If True, exclude real-only experiments
        """
        # Select experiment list based on filters
        if real_only:
            exp_list = REAL_ONLY_EXPERIMENTS
        elif exclude_real_only:
            exp_list = EXPERIMENTS
        else:
            exp_list = ALL_EXPERIMENTS

        experiments_to_eval = []

        for exp in exp_list:
            if only is not None:
                if exp['id'] == only:
                    experiments_to_eval.append(exp)
            elif exp['id'] >= start_from:
                experiments_to_eval.append(exp)

        if not experiments_to_eval:
            print("No experiments to evaluate!")
            return

        # Filter to only experiments with checkpoints
        experiments_with_checkpoints = []
        for exp in experiments_to_eval:
            paths = self._get_exp_paths(exp)
            if paths['checkpoint_best'].exists():
                experiments_with_checkpoints.append(exp)
            else:
                print(f"  Skipping {exp['name']} (no best.pth found)")

        if not experiments_with_checkpoints:
            print("No experiments have trained checkpoints!")
            return

        print("\n" + "="*70)
        print("EVALUATION-ONLY MODE (using test split)")
        print(f"Synthetic version: {self.synth_version}")
        print(f"Batch mode: {self.batch_mode}")
        print("="*70)
        print(f"\nExperiments to evaluate: {len(experiments_with_checkpoints)}")
        for exp in experiments_with_checkpoints:
            print(f"  {exp['id']:2d}. {exp['name']}")

        print("\n" + "="*70)
        print("STARTING EVALUATIONS")
        print("="*70)

        for exp in experiments_with_checkpoints:
            exp_name = exp['name']

            print(f"\n{'#'*70}")
            print(f"# EXPERIMENT {exp['id']}: {exp_name}")
            print(f"# {exp['description']}")
            print(f"{'#'*70}")

            start_time = datetime.now()

            try:
                result = self.eval_single_experiment(exp)

                end_time = datetime.now()
                duration = end_time - start_time
                result['duration_seconds'] = duration.total_seconds()
                result['duration_formatted'] = str(duration).split('.')[0]

                if result['status'] == 'completed':
                    print(f"\n{'='*60}")
                    print(f"EVAL COMPLETED: {exp_name}")
                    print(f"Duration: {result['duration_formatted']}")
                    print(f"{'='*60}")
                else:
                    print(f"\nEVAL FAILED: {exp_name} - {result['status']}")

            except KeyboardInterrupt:
                print(f"\n\nInterrupted during {exp_name}")
                return

            except Exception as e:
                print(f"\nERROR evaluating {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "="*70)
        print("ALL EVALUATIONS FINISHED")
        print(f"Synthetic version: {self.synth_version}")
        print(f"Batch mode: {self.batch_mode}")
        print("="*70)

    def run_all(
        self,
        start_from: int = 1,
        only: Optional[int] = None,
        real_only: bool = False,
        exclude_real_only: bool = False,
        resume_training: bool = False
    ) -> None:
        """
        Run all experiments sequentially.

        Args:
            start_from: Experiment ID to start from
            only: If set, run only this experiment ID
            real_only: If True, run only real-only baseline experiments (exp11-exp14)
            exclude_real_only: If True, exclude real-only experiments
            resume_training: If True, resume interrupted training from last checkpoint
        """
        # Select experiment list based on filters
        if real_only:
            exp_list = REAL_ONLY_EXPERIMENTS
        elif exclude_real_only:
            exp_list = EXPERIMENTS
        else:
            exp_list = ALL_EXPERIMENTS

        experiments_to_run = []

        for exp in exp_list:
            if only is not None:
                if exp['id'] == only:
                    experiments_to_run.append(exp)
            elif exp['id'] >= start_from:
                experiments_to_run.append(exp)

        if not experiments_to_run:
            print("No experiments to run!")
            return

        print("\n" + "="*70)
        print("EXPERIMENT EXECUTION PLAN")
        print(f"Synthetic version: {self.synth_version}")
        print(f"Batch mode: {self.batch_mode}")
        if resume_training:
            print("Mode: RESUME (will use existing checkpoints)")
        print("="*70)
        print(f"\nExperiments to run: {len(experiments_to_run)}")
        for exp in experiments_to_run:
            status = self.get_experiment_status_detail(exp)
            print(f"  {exp['id']:2d}. {exp['name']} [{status}]")

        print("\n" + "="*70)
        print(f"STARTING EXPERIMENTS [version: {self.synth_version}, batch: {self.batch_mode}]")
        print("="*70)

        for exp in experiments_to_run:
            exp_name = exp['name']

            # Skip if already completed (unless running single experiment)
            if only is None and self.state.is_completed(exp_name):
                print(f"\nSkipping {exp_name} (already completed)")
                continue

            # Reuse from standard mode for batch-mode-invariant experiments
            if only is None:
                reuse_result = reuse_from_standard_mode(
                    exp, self.synth_version, self.batch_mode
                )
                if reuse_result is not None:
                    print(f"\n{'='*70}")
                    print(f"REUSING {exp_name} from standard mode (batch-mode invariant)")
                    print(f"{'='*70}")
                    self.state.mark_completed(exp_name, reuse_result)
                    continue

            print(f"\n{'#'*70}")
            print(f"# EXPERIMENT {exp['id']}: {exp_name}")
            print(f"# {exp['description']}")
            print(f"{'#'*70}")

            start_time = datetime.now()

            try:
                result = self.run_single_experiment(
                    exp,
                    resume_training=resume_training
                )

                end_time = datetime.now()
                duration = end_time - start_time
                result['duration_seconds'] = duration.total_seconds()
                result['duration_formatted'] = str(duration).split('.')[0]

                if result['status'] == 'completed':
                    print(f"\n{'='*60}")
                    print(f"COMPLETED: {exp_name}")
                    print(f"Duration: {result['duration_formatted']}")
                    print(f"{'='*60}")
                    self.state.mark_completed(exp_name, result)
                else:
                    print(f"\nFAILED: {exp_name} - {result['status']}")
                    self.state.mark_failed(exp_name, result['status'])

            except KeyboardInterrupt:
                print(f"\n\nInterrupted during {exp_name}")
                print("Progress has been saved. Use --resume to continue.")
                return

            except Exception as e:
                print(f"\nERROR in {exp_name}: {e}")
                import traceback
                traceback.print_exc()
                self.state.mark_failed(exp_name, str(e))
                continue

        print("\n" + "="*70)
        print("ALL EXPERIMENTS FINISHED")
        print(f"Synthetic version: {self.synth_version}")
        print(f"Batch mode: {self.batch_mode}")
        print("="*70)

        # Summary
        completed = sum(1 for exp in exp_list if self.state.is_completed(exp['name']))
        versioned_paths = get_versioned_output_paths(self.synth_version, self.batch_mode)
        print(f"\nCompleted: {completed}/{len(exp_list)}")
        print(f"\nResults saved to: {versioned_paths['results']}")
        print(f"\nNext step: Run 'python experiments/collect_results.py -v {self.synth_version} --batch-mode {self.batch_mode}' to aggregate results")
        print(f"Then run: 'python experiments/collect_results.py -v {self.synth_version} --compare-batch-modes' to compare batch modes")


# =============================================================================
# CLI
# =============================================================================

def show_status(runner: ExperimentRunner, synth_version: str, batch_mode: str = DEFAULT_BATCH_MODE) -> None:
    """Display detailed experiment status."""
    print(f"\nExperiment Status [version: {synth_version}, batch: {batch_mode}]")
    print("="*70)
    print("\nSynthetic + Real Experiments:")
    print("-" * 70)
    for exp in EXPERIMENTS:
        status = runner.get_experiment_status_detail(exp)
        result = runner.state.results.get(exp['name'], {})
        duration = result.get('duration_formatted', '-')
        print(f"  {exp['id']:2d}. {exp['name']:30s} [{status:>30s}] {duration}")
    print("\nReal-Only Baseline Experiments:")
    print("-" * 70)
    for exp in REAL_ONLY_EXPERIMENTS:
        status = runner.get_experiment_status_detail(exp)
        result = runner.state.results.get(exp['name'], {})
        duration = result.get('duration_formatted', '-')
        print(f"  {exp['id']:2d}. {exp['name']:30s} [{status:>30s}] {duration}")


def interactive_select_experiment(
    runner: ExperimentRunner,
    include_real_only: bool = True
) -> Optional[int]:
    """
    Let the user select which experiment to resume from.

    Shows all incomplete experiments and lets user pick one.
    Returns the selected experiment ID.
    """
    exp_list = ALL_EXPERIMENTS if include_real_only else EXPERIMENTS

    # Collect incomplete experiments with status info
    incomplete = []
    for exp in exp_list:
        if runner.state.is_completed(exp['name']):
            continue

        ckpt = get_experiment_checkpoint_status(exp, runner.synth_version)
        status = runner.get_experiment_status_detail(exp)
        incomplete.append((exp, ckpt, status))

    if not incomplete:
        print("\nAll experiments are already completed!")
        return None

    print(f"\nIncomplete experiments [version: {runner.synth_version}]:")
    print("-" * 70)

    for i, (exp, ckpt, status) in enumerate(incomplete):
        checkpoint_info = ""
        if ckpt['has_last']:
            checkpoint_info = " << has checkpoint, will resume training"
        print(f"  {i+1:2d}) exp {exp['id']:2d}: {exp['name']:30s} [{status}]{checkpoint_info}")

    print(f"\n  0) Start from the first incomplete experiment (exp {incomplete[0][0]['id']})")
    print()

    while True:
        try:
            choice = input("Select experiment number (0 for auto): ").strip()
            if choice == '' or choice == '0':
                return incomplete[0][0]['id']

            idx = int(choice) - 1
            if 0 <= idx < len(incomplete):
                selected = incomplete[idx][0]
                return selected['id']

            print(f"  Invalid choice. Enter 1-{len(incomplete)} or 0 for auto.")
        except (ValueError, EOFError):
            print(f"  Invalid input. Enter 1-{len(incomplete)} or 0 for auto.")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic vs real data experiments"
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=None,
        help='Experiment ID to start from'
    )
    parser.add_argument(
        '--synth-version', '-v',
        choices=get_available_synth_versions(),
        default=DEFAULT_SYNTH_VERSION,
        help=f'Synthetic dataset version to use (default: {DEFAULT_SYNTH_VERSION})'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last incomplete experiment, auto-resuming training from checkpoint if available'
    )
    parser.add_argument(
        '--select',
        action='store_true',
        help='Used with --resume: interactively select which experiment to resume from'
    )
    parser.add_argument(
        '--only',
        type=int,
        help='Run only this experiment ID'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision'
    )
    parser.add_argument(
        '--real-only',
        action='store_true',
        help='Run only real-only baseline experiments (exp11-exp14)'
    )
    parser.add_argument(
        '--exclude-real-only',
        action='store_true',
        help='Exclude real-only experiments (run only synth+real experiments)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List experiments and their status'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current experiment status'
    )
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help='List available synthetic dataset versions'
    )
    parser.add_argument(
        '--batch-mode',
        choices=BATCH_MODES,
        default=DEFAULT_BATCH_MODE,
        help=f'Batch sampling mode (default: {DEFAULT_BATCH_MODE})'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Run evaluation only (no training) on existing checkpoints using test split'
    )

    args = parser.parse_args()

    if args.list_versions:
        print("\nAvailable synthetic dataset versions:")
        print("-" * 70)
        for version, config in SYNTHETIC_DATASETS.items():
            print(f"  {version}: {config['name']}")
            print(f"    {config['description']}")
        return

    runner = ExperimentRunner(
        synth_version=args.synth_version,
        device=args.device,
        use_amp=not args.no_amp,
        batch_mode=args.batch_mode
    )

    if args.list or args.status:
        show_status(runner, args.synth_version, args.batch_mode)
        return

    # Eval-only mode: run evaluation on existing checkpoints using test split
    if args.eval_only:
        start_from = args.start_from if args.start_from is not None else 1
        runner.eval_all(
            start_from=start_from,
            only=args.only,
            real_only=args.real_only,
            exclude_real_only=args.exclude_real_only,
        )
        return

    # Determine start_from and resume_training flag
    include_real_only = not args.exclude_real_only
    resume_training = False
    start_from = args.start_from if args.start_from is not None else 1

    if args.resume:
        resume_training = True

        if args.select:
            # Interactive selection
            selected_id = interactive_select_experiment(runner, include_real_only)
            if selected_id is None:
                return
            start_from = selected_id
            print(f"\nResuming from experiment {start_from}")
        else:
            # Auto-detect: find first incomplete experiment
            first_incomplete = runner.find_first_incomplete(include_real_only)
            if first_incomplete is None:
                print("\nAll experiments are already completed!")
                return
            start_from = first_incomplete

            # Show what we found
            resumable = runner.find_resumable_experiments(include_real_only)
            if resumable:
                print(f"\nFound {len(resumable)} experiment(s) with checkpoints:")
                for exp, ckpt_path in resumable:
                    print(f"  exp {exp['id']:2d}: {exp['name']} -> {ckpt_path}")

            print(f"\nResuming from experiment {start_from}")
            print("(use --select for interactive experiment selection)")

    runner.run_all(
        start_from=start_from,
        only=args.only,
        real_only=args.real_only,
        exclude_real_only=args.exclude_real_only,
        resume_training=resume_training
    )


if __name__ == '__main__':
    main()
