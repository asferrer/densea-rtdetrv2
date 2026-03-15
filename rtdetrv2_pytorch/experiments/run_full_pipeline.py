#!/usr/bin/env python3
"""
Full Pipeline Orchestrator for Synthetic vs Real Data Experiments.

This script runs the complete experiment pipeline automatically:
1. Prepare datasets (merge synthetic + real with different ratios)
2. Generate YAML configurations
3. Run all 9 training experiments with evaluation
4. Collect and visualize results

Usage:
    # Run complete pipeline
    python experiments/run_full_pipeline.py

    # Run from a specific phase
    python experiments/run_full_pipeline.py --start-phase 3

    # Skip dataset preparation (if already done)
    python experiments/run_full_pipeline.py --skip-prepare

    # Run specific experiments only
    python experiments/run_full_pipeline.py --experiments 1 7 8 9

    # Dry run (show what would be done)
    python experiments/run_full_pipeline.py --dry-run
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from config import (
    EXPERIMENTS,
    REAL_ONLY_EXPERIMENTS,
    ALL_EXPERIMENTS,
    REAL_TRAIN_JSON,
    REAL_TEST_JSON,
    OUTPUT_BASE_DIR,
    RESULTS_OUTPUT_DIR,
    # Versioned dataset support
    SYNTHETIC_DATASETS,
    DEFAULT_SYNTH_VERSION,
    get_synth_config,
    get_available_synth_versions,
    get_versioned_output_paths,
    # Batch mode support
    BATCH_MODES,
    DEFAULT_BATCH_MODE,
    # Class filtering support
    needs_class_filtering,
    create_filtered_test_json_for_synth_version,
    get_filtered_test_json_path,
)


# =============================================================================
# PIPELINE PHASES
# =============================================================================

PHASES = [
    {"id": 0, "name": "prepare_real_only", "description": "Prepare real-only baseline datasets"},
    {"id": 1, "name": "prepare_datasets", "description": "Prepare synthetic+real training datasets"},
    {"id": 2, "name": "generate_configs", "description": "Generate YAML configurations"},
    {"id": 3, "name": "run_experiments", "description": "Run training and evaluation"},
    {"id": 4, "name": "collect_results", "description": "Collect and visualize results"},
    {"id": 5, "name": "compare_batch_modes", "description": "Compare results across batch modes (if multiple modes available)"},
]


class PipelineOrchestrator:
    """Main orchestrator for the full experiment pipeline."""

    def __init__(
        self,
        experiments: Optional[List[int]] = None,
        synth_version: str = DEFAULT_SYNTH_VERSION,
        device: str = 'cuda',
        dry_run: bool = False,
        include_real_only: bool = True,
        batch_mode: str = DEFAULT_BATCH_MODE,
        resume: bool = False,
        eval_only: bool = False
    ):
        self.experiments = experiments
        self.synth_version = synth_version
        self.device = device
        self.dry_run = dry_run
        self.include_real_only = include_real_only
        self.batch_mode = batch_mode
        self.resume = resume
        self.eval_only = eval_only
        self.base_dir = Path(__file__).parent

        # Version and batch-mode specific state file
        versioned_paths = get_versioned_output_paths(synth_version, batch_mode)
        self.state_file = Path(versioned_paths['results']) / "pipeline_state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load pipeline state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"phases_completed": [], "started_at": None}

    def _save_state(self) -> None:
        """Save pipeline state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _run_script(
        self,
        script_name: str,
        args: List[str] = None,
        pass_version: bool = True
    ) -> bool:
        """
        Run a Python script from the experiments directory.

        Args:
            script_name: Name of the script (without .py)
            args: Additional command line arguments
            pass_version: Whether to pass --synth-version to the script

        Returns:
            True if successful
        """
        script_path = self.base_dir / f"{script_name}.py"
        cmd = [sys.executable, str(script_path)]

        # Pass synthetic version to child scripts (except for real-only operations)
        if pass_version:
            cmd.extend(['--synth-version', self.synth_version])

        if args:
            cmd.extend(args)

        print(f"\n{'─'*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'─'*60}\n")

        if self.dry_run:
            print("[DRY RUN] Would execute the above command")
            return True

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.base_dir.parent),  # rtdetrv2_pytorch
                check=False
            )
            return result.returncode == 0

        except KeyboardInterrupt:
            print("\n\nPipeline interrupted by user.")
            return False

        except Exception as e:
            print(f"\nError running {script_name}: {e}")
            return False

    def _check_prerequisites(self) -> bool:
        """Check if prerequisites are met before running pipeline."""
        print("\n" + "="*60)
        print("CHECKING PREREQUISITES")
        print(f"Synthetic version: {self.synth_version}")
        print("="*60)

        errors = []

        # Check if synthetic dataset exists (version-specific)
        synth_config = get_synth_config(self.synth_version)
        synth_train_json = synth_config['train_json']
        synth_path = Path(synth_train_json.replace("/app/RT-DETR/", ""))
        if not synth_path.exists():
            # Try absolute path resolution
            alt_path = self.base_dir.parent / synth_path
            if not alt_path.exists():
                errors.append(
                    f"Synthetic dataset [{self.synth_version}] not found:\n"
                    f"  Expected: {synth_train_json}\n"
                    f"  Available versions: {', '.join(get_available_synth_versions())}"
                )

        # Check if real dataset exists
        real_path = Path(REAL_TRAIN_JSON.replace("/app/RT-DETR/", ""))
        if not real_path.exists():
            alt_path = self.base_dir.parent / real_path
            if not alt_path.exists():
                errors.append(f"Real dataset not found: {REAL_TRAIN_JSON}")

        if errors:
            print("\n⚠️  Prerequisites not met:")
            for error in errors:
                print(f"\n  • {error}")
            return False

        print("\n✓ All prerequisites met")
        return True

    def phase_prepare_real_only(self) -> bool:
        """Phase 0: Prepare real-only baseline datasets."""
        print("\n" + "="*60)
        print("PHASE 0: PREPARE REAL-ONLY BASELINE DATASETS")
        print("="*60)

        args = ['--real-only']
        if self.experiments:
            # Filter to only real-only experiment IDs (11-14)
            real_only_ids = [e for e in self.experiments if 11 <= e <= 14]
            if real_only_ids:
                args.extend(['--experiment'] + [str(e) for e in real_only_ids])
            elif not any(11 <= e <= 14 for e in self.experiments):
                # No real-only experiments requested, skip this phase
                print("No real-only experiments in selection, skipping...")
                return True

        # Real-only doesn't need synthetic version, but pass it anyway for consistency
        return self._run_script("prepare_datasets", args)

    def phase_prepare_datasets(self) -> bool:
        """Phase 1: Prepare synthetic+real training datasets."""
        print("\n" + "="*60)
        print(f"PHASE 1: PREPARE SYNTHETIC+REAL DATASETS [version: {self.synth_version}]")
        print("="*60)

        args = []
        if self.experiments:
            # Filter out real-only experiment IDs (only process synth+real: 1-9)
            synth_real_ids = [e for e in self.experiments if 1 <= e <= 9]
            if synth_real_ids:
                args.extend(['--experiment'] + [str(e) for e in synth_real_ids])
            elif not any(1 <= e <= 9 for e in self.experiments):
                # No synth+real experiments requested, skip this phase
                print("No synthetic+real experiments in selection, skipping...")
                return True

        return self._run_script("prepare_datasets", args)

    def phase_generate_configs(self) -> bool:
        """Phase 2: Generate YAML configurations."""
        print("\n" + "="*60)
        print(f"PHASE 2: GENERATE CONFIGURATIONS [version: {self.synth_version}, batch: {self.batch_mode}]")
        print("="*60)

        args = ['--batch-mode', self.batch_mode]
        if self.experiments:
            args.extend(['--experiment'] + [str(e) for e in self.experiments])

        return self._run_script("generate_configs", args)

    def phase_run_experiments(self) -> bool:
        """Phase 3: Run training and evaluation (or eval-only)."""
        if self.eval_only:
            print("\n" + "="*60)
            print(f"PHASE 3: EVALUATE ONLY (test split) [version: {self.synth_version}, batch: {self.batch_mode}]")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"PHASE 3: RUN EXPERIMENTS [version: {self.synth_version}, batch: {self.batch_mode}]")
            print("="*60)

        args = ['--device', self.device, '--batch-mode', self.batch_mode]

        if self.eval_only:
            args.append('--eval-only')

        if self.resume:
            args.append('--resume')

        if self.experiments:
            # Run each experiment individually
            for exp_id in self.experiments:
                exp_args = args + ['--only', str(exp_id)]
                success = self._run_script("run_experiments", exp_args)
                if not success:
                    print(f"\nExperiment {exp_id} failed. Continuing with next...")
        else:
            # Run all experiments
            return self._run_script("run_experiments", args)

        return True

    def phase_collect_results(self) -> bool:
        """Phase 4: Collect and visualize results."""
        print("\n" + "="*60)
        print(f"PHASE 4: COLLECT RESULTS [version: {self.synth_version}, batch: {self.batch_mode}]")
        print("="*60)

        return self._run_script("collect_results", ['--batch-mode', self.batch_mode])

    def phase_compare_batch_modes(self) -> bool:
        """Phase 5: Compare results across batch modes (if multiple modes have results)."""
        # Quick check: do at least 2 batch modes have state files?
        modes_with_results = 0
        for bm in BATCH_MODES:
            versioned_paths = get_versioned_output_paths(self.synth_version, bm)
            results_dir = Path(versioned_paths['results'])
            state_file = results_dir / f"experiment_state_{self.synth_version}.json"
            # For standard mode, the state file has a different naming pattern
            state_file_alt = results_dir / f"experiment_state_{self.synth_version}_{bm}.json"
            if state_file.exists() or state_file_alt.exists():
                modes_with_results += 1

        if modes_with_results < 2:
            print(f"\nSkipping batch-mode comparison: only {modes_with_results} mode(s) have results.")
            print("Run the pipeline with a different --batch-mode to enable comparison.")
            return True  # Not a failure, just nothing to compare yet

        print("\n" + "="*60)
        print(f"PHASE 5: COMPARE BATCH MODES [version: {self.synth_version}]")
        print(f"Found {modes_with_results} batch modes with results")
        print("="*60)

        # Run batch-mode comparison for this version
        batch_ok = self._run_script("collect_results", ['--compare-batch-modes'])

        # Also run unified comparison (all versions × all batch modes)
        print("\n" + "-"*60)
        print("Running unified comparison (all versions × batch modes)...")
        print("-"*60)
        unified_ok = self._run_script("collect_results", ['--compare-all'])

        if not unified_ok:
            print("⚠️ Unified comparison failed (non-critical)")

        return batch_ok

    def run_full_pipeline(
        self,
        start_phase: int = 0,
        skip_prepare: bool = False,
        skip_configs: bool = False,
        skip_real_only: bool = False
    ) -> bool:
        """
        Run the complete pipeline.

        Args:
            start_phase: Phase to start from (0-4)
            skip_prepare: Skip dataset preparation (both phases 0 and 1)
            skip_configs: Skip config generation
            skip_real_only: Skip real-only baseline experiments (Phase 0)

        Returns:
            True if pipeline completed successfully
        """
        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█   SYNTHETIC VS REAL DATA EXPERIMENT PIPELINE" + " "*12 + "█")
        print("█" + " "*58 + "█")
        print("█"*60)

        # Record start time
        start_time = datetime.now()
        self.state["started_at"] = start_time.isoformat()
        self.state["synth_version"] = self.synth_version
        self._save_state()

        # Get version info
        synth_config = get_synth_config(self.synth_version)

        print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Synthetic version: {self.synth_version} ({synth_config['name']})")
        print(f"Batch mode: {self.batch_mode}")
        print(f"Device: {self.device}")
        print(f"Dry run: {self.dry_run}")
        print(f"Eval-only mode: {self.eval_only}")
        print(f"Include real-only baselines: {self.include_real_only and not skip_real_only}")

        if self.experiments:
            print(f"Experiments: {self.experiments}")
        else:
            print(f"Experiments: All ({len(ALL_EXPERIMENTS)})")

        # Check prerequisites
        if not self.dry_run and not self._check_prerequisites():
            print("\n❌ Pipeline aborted due to missing prerequisites.")
            return False

        # Phase 0: Prepare real-only baseline datasets
        if start_phase <= 0 and not skip_prepare and not skip_real_only and self.include_real_only and not self.eval_only:
            if not self.phase_prepare_real_only():
                print("\n❌ Phase 0 failed: Real-only dataset preparation")
                return False
            self.state["phases_completed"].append(0)
            self._save_state()

        # Phase 1: Prepare synthetic+real datasets
        if start_phase <= 1 and not skip_prepare and not self.eval_only:
            if not self.phase_prepare_datasets():
                print("\n❌ Phase 1 failed: Dataset preparation")
                return False
            self.state["phases_completed"].append(1)
            self._save_state()

        # Eval-only: ensure filtered test JSON exists for versions that need it
        if self.eval_only and needs_class_filtering(self.synth_version):
            filtered_path = get_filtered_test_json_path(self.synth_version)
            if not Path(filtered_path).exists():
                print("\n" + "="*60)
                print(f"CREATING FILTERED TEST JSON for {self.synth_version}")
                print("="*60)
                if not self.dry_run:
                    try:
                        create_filtered_test_json_for_synth_version(
                            REAL_TEST_JSON, self.synth_version
                        )
                        print(f"  Created: {filtered_path}")
                    except Exception as e:
                        print(f"  ERROR creating filtered test JSON: {e}")
                        return False
                else:
                    print(f"  [DRY RUN] Would create: {filtered_path}")

        # Phase 2: Generate configs (always run in eval-only to generate eval configs)
        if start_phase <= 2 and not skip_configs:
            if not self.phase_generate_configs():
                print("\n❌ Phase 2 failed: Config generation")
                return False
            self.state["phases_completed"].append(2)
            self._save_state()

        # Phase 3: Run experiments
        if start_phase <= 3:
            if not self.phase_run_experiments():
                print("\n⚠️ Phase 3: Some experiments may have failed")
                # Continue to collect results anyway
            self.state["phases_completed"].append(3)
            self._save_state()

        # Phase 4: Collect results
        if start_phase <= 4:
            if not self.phase_collect_results():
                print("\n❌ Phase 4 failed: Results collection")
                return False
            self.state["phases_completed"].append(4)
            self._save_state()

        # Phase 5: Compare batch modes (if multiple modes have results)
        if start_phase <= 5:
            if not self.phase_compare_batch_modes():
                print("\n⚠️ Phase 5: Batch mode comparison failed (non-critical)")
            self.state["phases_completed"].append(5)
            self._save_state()

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "█"*60)
        print("█" + " "*58 + "█")
        print("█   PIPELINE COMPLETED" + " "*36 + "█")
        print("█" + " "*58 + "█")
        print("█"*60)

        # Get versioned results path
        versioned_paths = get_versioned_output_paths(self.synth_version, self.batch_mode)

        print(f"\nFinished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {str(duration).split('.')[0]}")
        print(f"Synthetic version: {self.synth_version}")
        print(f"Batch mode: {self.batch_mode}")
        print(f"\nResults available at: {versioned_paths['results']}")
        print(f"\nTo compare across batch modes (if not auto-generated), run:")
        print(f"  python experiments/collect_results.py -v {self.synth_version} --compare-batch-modes")
        print(f"\nTo compare across versions, run:")
        print(f"  python experiments/collect_results.py --compare-versions")
        print(f"\nTo compare all versions × batch modes combined, run:")
        print(f"  python experiments/collect_results.py --compare-all")

        return True


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the full synthetic vs real data experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_full_pipeline.py

  # Run only specific experiments
  python run_full_pipeline.py --experiments 1 7 8 9

  # Skip dataset preparation (if already done)
  python run_full_pipeline.py --skip-prepare

  # Resume from a specific phase
  python run_full_pipeline.py --start-phase 3

  # Dry run to see what would be executed
  python run_full_pipeline.py --dry-run
        """
    )

    parser.add_argument(
        '--experiments', '-e',
        type=int,
        nargs='+',
        help='Run only these experiment IDs (1-9 synth+real, 11-14 real-only)'
    )
    parser.add_argument(
        '--synth-version', '-v',
        choices=get_available_synth_versions(),
        default=DEFAULT_SYNTH_VERSION,
        help=f'Synthetic dataset version to use (default: {DEFAULT_SYNTH_VERSION})'
    )
    parser.add_argument(
        '--start-phase',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help='Start from this phase (0=real-only, 1=synth+real, 2=configs, 3=train, 4=results, 5=compare-batch-modes)'
    )
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip dataset preparation phases (0 and 1)'
    )
    parser.add_argument(
        '--skip-configs',
        action='store_true',
        help='Skip config generation phase'
    )
    parser.add_argument(
        '--skip-real-only',
        action='store_true',
        help='Skip real-only baseline experiments (Phase 0)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Device to use for training (default: cuda)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    parser.add_argument(
        '--list-phases',
        action='store_true',
        help='List all phases and exit'
    )
    parser.add_argument(
        '--list-versions',
        action='store_true',
        help='List available synthetic dataset versions and exit'
    )
    parser.add_argument(
        '--batch-mode',
        choices=BATCH_MODES,
        default=DEFAULT_BATCH_MODE,
        help=f'Batch sampling mode (default: {DEFAULT_BATCH_MODE})'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume incomplete experiments from their last checkpoint instead of restarting them'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Run evaluation only on existing checkpoints using test split (skips dataset prep and training)'
    )

    args = parser.parse_args()

    if args.list_versions:
        print("\nAvailable synthetic dataset versions:")
        print("-" * 70)
        for version, config in SYNTHETIC_DATASETS.items():
            default_marker = " (default)" if version == DEFAULT_SYNTH_VERSION else ""
            print(f"  {version}{default_marker}: {config['name']}")
            print(f"    {config['description']}")
        return

    if args.list_phases:
        print("\nPipeline Phases:")
        print("-" * 50)
        for phase in PHASES:
            print(f"  {phase['id']}. {phase['name']}: {phase['description']}")
        print()
        return

    orchestrator = PipelineOrchestrator(
        experiments=args.experiments,
        synth_version=args.synth_version,
        device=args.device,
        dry_run=args.dry_run,
        include_real_only=not args.skip_real_only,
        batch_mode=args.batch_mode,
        resume=args.resume,
        eval_only=args.eval_only
    )

    success = orchestrator.run_full_pipeline(
        start_phase=args.start_phase,
        skip_prepare=args.skip_prepare,
        skip_configs=args.skip_configs,
        skip_real_only=args.skip_real_only
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
