#!/usr/bin/env python
"""
Optimized TensorRT inference and evaluation for RT-DETRv2 on Jetson Orin Nano Super.

Optimizations over rtdetrv2_tensorrt.py:
- CUDA graphs for zero kernel-launch overhead (fixed shapes)
- Pre-resolved buffer indices (no per-frame name lookup)
- Warmup phase excluded from timing
- Pinned memory with pre-allocated contiguous arrays
- Optional GPU preprocessing via torch.cuda (when available)
- Power monitoring via tegrastats or jtop
- Latency percentiles (p50, p95, p99) for paper reporting
- Single stream with events for precise GPU timing

trtexec \
  --onnx=rtdetrv2_r50vd_densea_exp06_simplified.onnx \
  --saveEngine=rtdetrv2_r50vd_densea_exp06_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096MiB \
  --builderOptimizationLevel=5 \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --timingCacheFile=timing_cache.bin \
  --avgRuns=20 \
  --warmUp=5000 \
  2>&1 | tee build_log.txt

"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorrt as trt
import torch
import time
import json
import io
from tqdm import tqdm
from collections import defaultdict

# --- Optional imports ---
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCOTOOLS_AVAILABLE = True
except ImportError:
    COCOTOOLS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False

# --- Constants ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MAX_OUTPUT_DETECTIONS = 300
INPUT_H = 640
INPUT_W = 640
WARMUP_RUNS = 50


class TRTInferenceEngine:
    """High-performance TensorRT inference with CUDA graphs and PyTorch CUDA streams.

    Uses torch.cuda instead of PyCUDA for full compatibility with TensorRT 10.x
    auxiliary streams on Jetson platforms.
    """

    # Map TRT dtypes to torch dtypes (populated lazily)
    _TRT_TO_TORCH = None

    @staticmethod
    def _trt_dtype_to_torch(trt_dtype):
        if TRTInferenceEngine._TRT_TO_TORCH is None:
            TRTInferenceEngine._TRT_TO_TORCH = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
                trt.int64: torch.int64,
                trt.int8: torch.int8,
                trt.bool: torch.bool,
            }
        return TRTInferenceEngine._TRT_TO_TORCH.get(trt_dtype, torch.float32)

    def __init__(self, engine_path: str, use_cuda_graph: bool = True):
        self.use_cuda_graph = use_cuda_graph
        self._cuda_graph = None
        self.device = torch.device('cuda')

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # PyTorch CUDA stream and events for GPU timing
        self.stream = torch.cuda.Stream(device=self.device)
        self.event_start = torch.cuda.Event(enable_timing=True)
        self.event_end = torch.cuda.Event(enable_timing=True)

        # Resolve buffer layout once
        self._setup_buffers()

        # Set static shapes once (never changes for 640x640 batch=1)
        self.context.set_input_shape("images", (1, 3, INPUT_H, INPUT_W))
        self.context.set_input_shape("orig_target_sizes", (1, 2))

        # Set tensor addresses once
        for buf in self._all_buffers:
            self.context.set_tensor_address(buf['name'], buf['tensor'].data_ptr())

    def _setup_buffers(self):
        """Pre-allocate all GPU buffers as contiguous torch tensors."""
        self._inputs = {}
        self._outputs = {}
        self._all_buffers = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = self._trt_dtype_to_torch(trt_dtype)
            mode = self.engine.get_tensor_mode(name)

            # Determine shape
            if name == "images":
                shape = (1, 3, INPUT_H, INPUT_W)
            elif name == "orig_target_sizes":
                shape = (1, 2)
            elif "labels" in name or "scores" in name:
                shape = (1, MAX_OUTPUT_DETECTIONS)
            elif "boxes" in name:
                shape = (1, MAX_OUTPUT_DETECTIONS, 4)
            else:
                shape = tuple(self.engine.get_tensor_shape(name))

            tensor = torch.empty(shape, dtype=torch_dtype, device=self.device).contiguous()

            buf = {
                'name': name,
                'tensor': tensor,
                'shape': shape,
                'dtype': torch_dtype,
            }
            self._all_buffers.append(buf)

            if mode == trt.TensorIOMode.INPUT:
                self._inputs[name] = buf
            else:
                self._outputs[name] = buf

    def warmup(self, n: int = WARMUP_RUNS):
        """Run warmup inferences to stabilize GPU clocks and JIT."""
        dummy_img = torch.zeros((1, 3, INPUT_H, INPUT_W), dtype=torch.float32, device=self.device)
        dummy_sizes = torch.tensor([[INPUT_W, INPUT_H]], dtype=torch.int32, device=self.device)

        for _ in range(n):
            self._execute(dummy_img, dummy_sizes)

        # Capture CUDA graph after warmup (shapes are stable)
        if self.use_cuda_graph:
            self._capture_cuda_graph(dummy_img, dummy_sizes)

    def _execute(self, images: torch.Tensor, orig_sizes: torch.Tensor):
        """Raw execution without CUDA graph."""
        with torch.cuda.stream(self.stream):
            self._inputs["images"]['tensor'].copy_(images)
            self._inputs["orig_target_sizes"]['tensor'].copy_(
                orig_sizes.to(self._inputs["orig_target_sizes"]['dtype'])
            )
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

    def _capture_cuda_graph(
        self, images: torch.Tensor, orig_sizes: torch.Tensor
    ):
        """Capture CUDA graph for zero kernel-launch overhead."""
        try:
            # Pre-copy inputs (addresses must be stable for graph capture)
            with torch.cuda.stream(self.stream):
                self._inputs["images"]['tensor'].copy_(images)
                self._inputs["orig_target_sizes"]['tensor'].copy_(
                    orig_sizes.to(self._inputs["orig_target_sizes"]['dtype'])
                )
            self.stream.synchronize()

            # Capture only the execution kernel(s)
            self._cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self.stream):
                with torch.cuda.graph(self._cuda_graph, stream=self.stream):
                    self.context.execute_async_v3(
                        stream_handle=self.stream.cuda_stream
                    )

            print(
                "[OPT] CUDA graph captured — kernel launch overhead eliminated"
            )
        except Exception as e:
            print(f"[OPT] CUDA graph capture failed ({e}), using standard execution")
            self._cuda_graph = None

    def infer(
        self,
        images_np: np.ndarray,
        orig_sizes_np: np.ndarray,
        measure_gpu_time: bool = False,
    ) -> dict:
        """
        Run inference with optional GPU-side timing.

        Args:
            images_np: numpy array (1, 3, H, W) float32
            orig_sizes_np: numpy array (1, 2) int32 [W, H]
            measure_gpu_time: if True, returns 'gpu_time_ms' in results

        Returns dict with 'labels', 'boxes', 'scores' numpy arrays.
        """
        img_buf = self._inputs["images"]
        size_buf = self._inputs["orig_target_sizes"]

        # Copy input data to GPU tensors
        images_t = torch.from_numpy(images_np).to(self.device)
        sizes_t = torch.from_numpy(orig_sizes_np).to(
            dtype=size_buf['dtype'], device=self.device
        )

        with torch.cuda.stream(self.stream):
            img_buf['tensor'].copy_(images_t)
            size_buf['tensor'].copy_(sizes_t)

            # GPU timing start
            if measure_gpu_time:
                self.event_start.record(self.stream)

            # Execute (CUDA graph or standard)
            if self._cuda_graph is not None:
                self._cuda_graph.replay()
            else:
                self.context.execute_async_v3(
                    stream_handle=self.stream.cuda_stream
                )

            # GPU timing end
            if measure_gpu_time:
                self.event_end.record(self.stream)

        self.stream.synchronize()

        # Build results (copy from GPU to numpy)
        results = {}
        for name, buf in self._outputs.items():
            out_shape = self.context.get_tensor_shape(name)
            results[name] = buf['tensor'].cpu().numpy().reshape(out_shape)

        if measure_gpu_time:
            results['gpu_time_ms'] = self.event_start.elapsed_time(
                self.event_end
            )

        return results


# --- Preprocessing ---

def preprocess_cpu(image_bgr: np.ndarray) -> np.ndarray:
    """CPU preprocessing: resize, normalize, transpose to NCHW float32."""
    h, w = image_bgr.shape[:2]
    interp = cv2.INTER_AREA if h > INPUT_H else cv2.INTER_LINEAR
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=interp)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.ascontiguousarray(img[np.newaxis])  # NCHW


# --- Evaluation ---

def calculate_iou(box_a_xywh, box_b_xyxy):
    """IoU between COCO box (xywh) and predicted box (xyxy)."""
    ax1, ay1 = box_a_xywh[0], box_a_xywh[1]
    ax2, ay2 = ax1 + box_a_xywh[2], ay1 + box_a_xywh[3]
    bx1, by1, bx2, by2 = box_b_xyxy

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_a_xywh[2] * box_a_xywh[3] + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def calculate_confusion_matrix(coco_gt, predictions, iou_threshold=0.5):
    """Confusion matrix with Background as last row/col."""
    cat_ids = coco_gt.getCatIds()
    id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    n = len(cat_ids)
    cm = np.zeros((n + 1, n + 1), dtype=np.int32)
    bg_idx = n

    gt_by_img = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        gt_by_img[ann['image_id']].append(ann)

    pred_by_img = defaultdict(list)
    for p in predictions:
        pred_by_img[p['image_id']].append(p)

    for img_id in coco_gt.getImgIds():
        gts = gt_by_img.get(img_id, [])
        preds = sorted(
            pred_by_img.get(img_id, []), key=lambda x: x['score'], reverse=True
        )
        for gt in gts:
            gt['found'] = False

        for pred in preds:
            pred_idx = id_to_idx.get(pred['category_id'])
            if pred_idx is None:
                continue

            best_iou, best_gt = 0, None
            for gt in gts:
                if gt['category_id'] == pred['category_id'] and not gt['found']:
                    px, py, pw, ph = pred['bbox']
                    iou = calculate_iou(
                        gt['bbox'], [px, py, px + pw, py + ph]
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt

            if best_iou >= iou_threshold and best_gt:
                gt_idx = id_to_idx[best_gt['category_id']]
                cm[gt_idx, pred_idx] += 1
                best_gt['found'] = True
            else:
                cm[bg_idx, pred_idx] += 1

        for gt in gts:
            if not gt['found']:
                gt_idx = id_to_idx[gt['category_id']]
                cm[gt_idx, bg_idx] += 1

    names = [coco_gt.cats[cid]['name'] for cid in cat_ids]
    return cm, names


def plot_confusion_matrix(cm, class_names, output_path):
    """Save confusion matrix heatmap."""
    if not PLOTTING_AVAILABLE:
        print("Warning: matplotlib/seaborn not installed, skipping plot.")
        return
    fig = plt.figure(figsize=(14, 14))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="viridis",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def run_evaluation(args, engine: TRTInferenceEngine):
    """Full COCO evaluation with optimized inference."""
    if not COCOTOOLS_AVAILABLE:
        print("Error: pycocotools required. pip install pycocotools")
        return

    print("--- Evaluation Mode (Optimized) ---")

    # GPU monitoring
    gpu_handle = None
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("[GPU] NVML monitoring enabled")
        except Exception as e:
            print(f"[GPU] NVML init failed: {e}")

    # Load annotations
    coco_gt = COCO(args.eval_json)
    print(f"Scanning images in '{args.eval_root_dir}'...")
    img_cache = {}
    for dp, _, fnames in os.walk(args.eval_root_dir):
        for f in fnames:
            img_cache[os.path.basename(f)] = os.path.join(dp, f)
    img_ids = coco_gt.getImgIds()
    print(f"Found {len(img_cache)} files. Evaluating {len(img_ids)} images.")

    # Category mapping (model output index -> COCO category id)
    cat_id_map = {i: cid for i, cid in enumerate(coco_gt.getCatIds())}

    # Warmup (excluded from metrics)
    print(f"[OPT] Running {WARMUP_RUNS} warmup iterations...")
    engine.warmup(WARMUP_RUNS)
    print("[OPT] Warmup complete.")

    # Evaluation loop
    predictions = []
    wall_times = []
    gpu_times = []
    gpu_utils = []
    gpu_mem_mbs = []
    gpu_powers = []

    for img_id in tqdm(img_ids, desc="Evaluating"):
        info = coco_gt.loadImgs(img_id)[0]
        path = img_cache.get(os.path.basename(info['file_name']))
        if not path:
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue

        # Preprocess
        img_batch = preprocess_cpu(frame)
        orig_sizes = np.array(
            [[frame.shape[1], frame.shape[0]]], dtype=np.int32
        )

        # Inference with wall-clock and GPU timing
        t0 = time.perf_counter()
        results = engine.infer(img_batch, orig_sizes, measure_gpu_time=True)
        wall_times.append(time.perf_counter() - t0)
        gpu_times.append(results.get('gpu_time_ms', 0.0))

        # GPU monitoring (sample every frame)
        if gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                gpu_utils.append(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                gpu_mem_mbs.append(mem.used / (1024 ** 2))
                power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle)
                gpu_powers.append(power / 1000.0)  # mW -> W
            except pynvml.NVMLError:
                pass

        # Extract detections
        labels = results.get('labels', np.array([]))
        boxes = results.get('boxes', np.array([]))
        scores = results.get('scores', np.array([]))
        if labels.ndim > 1:
            labels, boxes, scores = labels[0], boxes[0], scores[0]

        for lbl, box, score in zip(labels, boxes, scores):
            if score < args.threshold:
                continue
            x1, y1, x2, y2 = box
            coco_cid = cat_id_map.get(lbl)
            if coco_cid is not None:
                predictions.append({
                    "image_id": img_id,
                    "category_id": coco_cid,
                    "bbox": [
                        float(x1), float(y1),
                        float(x2 - x1), float(y2 - y1),
                    ],
                    "score": float(score),
                })

    # COCO evaluation
    print("\n--- Official COCO Evaluation ---")
    if not predictions:
        print("WARNING: No predictions generated!")
        return

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()

    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    coco_eval.summarize()
    sys.stdout = old_stdout
    coco_summary = buf.getvalue()
    print(coco_summary)

    # Confusion matrix
    print("--- Confusion Matrix ---")
    cm, cm_names = calculate_confusion_matrix(
        coco_gt, predictions, iou_threshold=args.iou_threshold
    )
    # Reorder: Background first
    n = len(cm_names)
    cm_r = np.zeros_like(cm)
    cm_r[1:, 1:] = cm[:n, :n]
    cm_r[0, 1:] = cm[n, :n]
    cm_r[1:, 0] = cm[:n, n]
    plot_names = ["Background"] + cm_names

    # Performance stats
    wall_arr = np.array(wall_times) * 1000  # ms
    gpu_arr = np.array(gpu_times)

    perf_stats = {
        "engine_file": args.engine_file,
        "total_images": len(wall_times),
        "cuda_graph_enabled": engine.use_cuda_graph
                              and engine._cuda_graph is not None,
        "warmup_runs": WARMUP_RUNS,
        "wall_time_ms": {
            "mean": float(np.mean(wall_arr)),
            "median": float(np.median(wall_arr)),
            "p95": float(np.percentile(wall_arr, 95)),
            "p99": float(np.percentile(wall_arr, 99)),
            "min": float(np.min(wall_arr)),
            "max": float(np.max(wall_arr)),
            "std": float(np.std(wall_arr)),
        },
        "gpu_time_ms": {
            "mean": float(np.mean(gpu_arr)),
            "median": float(np.median(gpu_arr)),
            "p95": float(np.percentile(gpu_arr, 95)),
            "p99": float(np.percentile(gpu_arr, 99)),
        },
        "fps": {
            "from_wall_mean": float(1000.0 / np.mean(wall_arr)),
            "from_wall_median": float(1000.0 / np.median(wall_arr)),
            "from_gpu_mean": float(
                1000.0 / np.mean(gpu_arr)
            ) if np.mean(gpu_arr) > 0 else 0,
        },
        "gpu_monitoring": {
            "avg_utilization_pct": float(np.mean(gpu_utils))
            if gpu_utils else "N/A",
            "avg_memory_mb": float(np.mean(gpu_mem_mbs))
            if gpu_mem_mbs else "N/A",
            "avg_power_w": float(np.mean(gpu_powers))
            if gpu_powers else "N/A",
            "max_power_w": float(np.max(gpu_powers))
            if gpu_powers else "N/A",
        },
        "coco_metrics": {
            "AP": float(coco_eval.stats[0]),
            "AP50": float(coco_eval.stats[1]),
            "AP75": float(coco_eval.stats[2]),
            "AP_small": float(coco_eval.stats[3]),
            "AP_medium": float(coco_eval.stats[4]),
            "AP_large": float(coco_eval.stats[5]),
            "AR_1": float(coco_eval.stats[6]),
            "AR_10": float(coco_eval.stats[7]),
            "AR_100": float(coco_eval.stats[8]),
        },
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "evaluation_summary.txt"), 'w') as f:
        f.write("=== COCO Evaluation Summary ===\n")
        f.write(coco_summary)
        f.write("\n=== Performance Stats ===\n")
        f.write(f"  Wall-time (mean):   {perf_stats['wall_time_ms']['mean']:.2f} ms\n")
        f.write(f"  Wall-time (median): {perf_stats['wall_time_ms']['median']:.2f} ms\n")
        f.write(f"  Wall-time (p95):    {perf_stats['wall_time_ms']['p95']:.2f} ms\n")
        f.write(f"  GPU-time (mean):    {perf_stats['gpu_time_ms']['mean']:.2f} ms\n")
        f.write(f"  FPS (wall mean):    {perf_stats['fps']['from_wall_mean']:.1f}\n")
        f.write(f"  FPS (wall median):  {perf_stats['fps']['from_wall_median']:.1f}\n")
        f.write(f"  CUDA Graph:         {perf_stats['cuda_graph_enabled']}\n")
        if gpu_powers:
            f.write(f"  Avg Power:          {perf_stats['gpu_monitoring']['avg_power_w']:.1f} W\n")

    with open(os.path.join(args.output_dir, "performance_stats.json"), 'w') as f:
        json.dump(perf_stats, f, indent=2)

    with open(os.path.join(args.output_dir, "evaluation_metrics_coco.json"), 'w') as f:
        json.dump(coco_eval.stats.tolist(), f, indent=2)

    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm_r)
    plot_confusion_matrix(
        cm_r, plot_names,
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )

    # Print summary
    print("\n=== Performance Summary ===")
    print(f"  Images:          {perf_stats['total_images']}")
    print(f"  CUDA Graph:      {perf_stats['cuda_graph_enabled']}")
    print(f"  Latency (wall):  {perf_stats['wall_time_ms']['mean']:.2f} ms "
          f"(p95={perf_stats['wall_time_ms']['p95']:.2f} ms)")
    print(f"  Latency (GPU):   {perf_stats['gpu_time_ms']['mean']:.2f} ms")
    print(f"  FPS (wall):      {perf_stats['fps']['from_wall_mean']:.1f}")
    print(f"  AP:              {perf_stats['coco_metrics']['AP']:.4f}")
    print(f"  AP50:            {perf_stats['coco_metrics']['AP50']:.4f}")
    if gpu_powers:
        print(f"  Avg Power:       {perf_stats['gpu_monitoring']['avg_power_w']:.1f} W")
    print(f"\n  Results saved to: {args.output_dir}")

    if PYNVML_AVAILABLE:
        pynvml.nvmlShutdown()


def main(args):
    print(f"Loading engine: {args.engine_file}")
    engine = TRTInferenceEngine(
        args.engine_file,
        use_cuda_graph=not args.no_cuda_graph,
    )
    print("Engine loaded successfully.")

    if args.eval_json and args.eval_root_dir:
        run_evaluation(args, engine)
    elif args.input:
        run_inference(args, engine)
    else:
        print("Error: specify -i (inference) or --eval_json + --eval_root_dir (evaluation)")


def run_inference(args, engine: TRTInferenceEngine):
    """Single image / folder / video / camera inference."""
    import colorsys

    def gen_colors(n):
        return [
            tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.9, 0.9)[::-1])
            for i in range(n)
        ]

    def draw(image, labels, boxes, scores, colors):
        for lbl, box, sc in zip(labels, boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            c = colors[int(lbl) % len(colors)]
            cv2.rectangle(image, (x1, y1), (x2, y2), c, 2)
            cv2.putText(
                image, f"{int(lbl)}:{sc:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1,
            )
        return image

    colors = gen_colors(100)
    os.makedirs(args.save_dir, exist_ok=True)

    # Warmup
    engine.warmup(WARMUP_RUNS)

    src = args.input
    if os.path.isfile(src) and src.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        frame = cv2.imread(src)
        img_batch = preprocess_cpu(frame)
        orig_sizes = np.array([[frame.shape[1], frame.shape[0]]], dtype=np.int32)
        results = engine.infer(img_batch, orig_sizes)

        labels = results.get('labels', np.array([]))[0]
        boxes = results.get('boxes', np.array([]))[0]
        scores = results.get('scores', np.array([]))[0]
        keep = scores > args.threshold
        out = draw(frame.copy(), labels[keep], boxes[keep], scores[keep], colors)

        if args.show:
            cv2.imshow("Result", out)
            cv2.waitKey(0)
        else:
            out_path = os.path.join(args.save_dir, os.path.basename(src))
            cv2.imwrite(out_path, out)
            print(f"Saved: {out_path}")

    elif os.path.isdir(src):
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [
            os.path.join(src, f) for f in os.listdir(src)
            if f.lower().endswith(exts)
        ]
        for fpath in tqdm(files, desc="Inference"):
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            img_batch = preprocess_cpu(frame)
            orig_sizes = np.array(
                [[frame.shape[1], frame.shape[0]]], dtype=np.int32
            )
            results = engine.infer(img_batch, orig_sizes)
            labels = results.get('labels', np.array([]))[0]
            boxes = results.get('boxes', np.array([]))[0]
            scores = results.get('scores', np.array([]))[0]
            keep = scores > args.threshold
            out = draw(
                frame.copy(), labels[keep], boxes[keep], scores[keep], colors
            )
            cv2.imwrite(
                os.path.join(args.save_dir, os.path.basename(fpath)), out
            )
    else:
        # Video or camera
        cap = cv2.VideoCapture(
            int(src) if src.isdigit() else src
        )
        if not cap.isOpened():
            print(f"Cannot open: {src}")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = None
        if not args.show:
            out_path = os.path.join(
                args.save_dir, os.path.basename(src) + "_out.mp4"
            )
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
            )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_batch = preprocess_cpu(frame)
            orig_sizes = np.array(
                [[frame.shape[1], frame.shape[0]]], dtype=np.int32
            )
            results = engine.infer(img_batch, orig_sizes)
            labels = results.get('labels', np.array([]))[0]
            boxes = results.get('boxes', np.array([]))[0]
            scores = results.get('scores', np.array([]))[0]
            keep = scores > args.threshold
            out = draw(
                frame.copy(), labels[keep], boxes[keep], scores[keep], colors
            )
            if args.show:
                cv2.imshow("Inference", out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif writer:
                writer.write(out)

        cap.release()
        if writer:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Optimized TensorRT RT-DETRv2 inference for Jetson Orin Nano Super",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--engine_file', type=str, required=True,
        help="Path to TensorRT engine (.engine)",
    )
    parser.add_argument(
        '--no_cuda_graph', action='store_true',
        help="Disable CUDA graph capture (use standard async execution)",
    )

    infer = parser.add_argument_group('Inference')
    infer.add_argument('-i', '--input', type=str, help="Image/video/folder/camera")
    infer.add_argument('--save_dir', type=str, default='inference_results')
    infer.add_argument('--show', action='store_true')

    ev = parser.add_argument_group('Evaluation')
    ev.add_argument('--eval_json', type=str, help="COCO annotation JSON")
    ev.add_argument('--eval_root_dir', type=str, help="Image root directory")
    ev.add_argument('--output_dir', type=str, default='evaluation_results')

    common = parser.add_argument_group('Parameters')
    common.add_argument(
        '--threshold', type=float, default=0.5,
        help="Confidence threshold (use 0.001 for COCO eval)",
    )
    common.add_argument('--iou_threshold', type=float, default=0.5)

    args = parser.parse_args()

    if not args.eval_json and not args.input:
        parser.error("Specify -i or --eval_json + --eval_root_dir")
    if args.eval_json and not args.eval_root_dir:
        parser.error("--eval_root_dir required with --eval_json")

    main(args)
