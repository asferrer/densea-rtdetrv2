# Changes with respect to upstream

Upstream: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR), directory `rtdetrv2_pytorch/` and `LICENSE`, at
commit `70eed1e46c5d705134069a3d55090e58d529f42d`. The first commit of this repository is that tree, unchanged. This
file lists the files of upstream that were modified (Apache-2.0, section 4b), and what was added or removed. All paths
are relative to `rtdetrv2_pytorch/` unless noted.

## Modified files

| File | Change |
|---|---|
| `src/core/_config.py` | New options `early_stopping_patience`, `early_stopping_min_delta`, `early_stopping_metric`, `enable_per_class_metrics`. |
| `src/core/yaml_utils.py` | `__include__` paths are resolved as normalised absolute paths. |
| `src/data/dataset/__init__.py`, `coco_dataset.py` | New `MultiDirCocoDetection` (images in several directories). |
| `src/data/dataset/coco_eval.py` | Optional evaluation with faster-coco-eval (`FAST_COCOEVAL=1`), per-class metrics. |
| `src/data/transforms/__init__.py`, `_transforms.py`, `container.py` | Rotation, Gaussian blur and noise augmentations; policy handling in `Compose`. |
| `src/misc/dist_utils.py` | `save_checkpoint_fast`: atomic checkpoint write (memory buffer, temporary file, fsync, `os.replace`). |
| `src/solver/_solver.py`, `clas_solver.py` | EMA state restored when tuning from a checkpoint; atomic checkpoints. |
| `src/solver/det_engine.py` | `torch.amp.GradScaler`; autocast in the forward pass of the training step. |
| `src/solver/det_solver.py` | Early stopping; `best.pth` = best validation AP of the EMA weights; best value and patience rebuilt from `log.txt` on resume; per-class metrics. |
| `src/zoo/rtdetr/rtdetr_postprocessor.py` | Softmax over the class dimension. |
| `src/zoo/rtdetr/rtdetrv2_criterion.py` | Optional per-class weights in the classification loss. |
| `tools/train.py` | `--seed`, `--test-only` and `--use-amp` options and `--update` overrides; saves an analysis plot of the class distribution of the training and validation splits. |
| `tools/export_onnx.py` | ONNX export with Conv-BN fusion, graph simplification with onnx-simplifier and logging. |
| `configs/runtime.yml`, `configs/rtdetrv2/include/dataloader.yml` | EMA on by default, early-stopping and per-class-metrics defaults; the augmentation pipeline is defined in the experiment configs. |
| `references/deploy/rtdetrv2_onnxruntime.py` | TensorRT and CUDA execution providers when available. |
| `references/deploy/rtdetrv2_tensorrt.py` | Inference script extended with an evaluation mode (pycocotools metrics, per-class analysis). |
| `requirements.txt` | Pinned versions. |

## Added files

- `configs/densea/rtdetrv2_r50vd_densea.template.yml`, `configs/dataset/densea_detection.yml`,
  `configs/rtdetrv2/include/optimizer_v2.yml`, `tools/make_densea_config.py`: configuration of the study.
- `tools/eval.py`, `tools/infer.py`, `tools/replace_erf_onnx.py`, `tools/create_calibration_set.py`,
  `tools/benchmark_latency.py`, `tools/compute_model_complexity.py`,
  `references/deploy/build_engine.py`, `references/deploy/rtdetrv2_onnxruntime_v2.py`,
  `references/deploy/rtdetrv2_tensorrt_optimized.py`: evaluation, export and deployment tools.
- `tests/test_resume_state.py` (CPU only).
- At the repository root: `Dockerfile`, `docker-compose.yml`, `.gitignore`, `.gitattributes`, `.dockerignore`,
  `README.md`, `NOTICE`, `CHANGES.md`, `CITATION.cff`.

## Removed files

`rtdetrv2_pytorch/README.md` (the model zoo and COCO usage notes of upstream; replaced by the `README.md` at the
root). The other tools of the working repository (dataset diagnostics, synthetic-data validation scripts, the Jetson
copy of `src/`) are not part of this release.
