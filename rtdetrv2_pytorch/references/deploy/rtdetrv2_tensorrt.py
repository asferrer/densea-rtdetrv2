#!/usr/bin/env python
"""
TensorRT inference and evaluation script for RT-DETRv2 models.

Features:
- Handles image, video, folder, camera/stream, and evaluation modes.
- Optimized for TensorRT engines (.engine files).
- Evaluation mode uses the official `pycocotools` library for standard metrics.
- Uses Hungarian matching algorithm (same as eval.py) for fair comparison.
- Comprehensive per-class metrics: TP/FN/FP/Recall/Precision/F1/AP.
- Score distribution analysis for threshold optimization.
- Saves all metrics to an output directory for direct comparison with PyTorch eval.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from tqdm import tqdm
import colorsys
import json
from collections import defaultdict, Counter
import io

# --- Optional Imports for Metrics ---
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCOTOOLS_AVAILABLE = True
except ImportError:
    COCOTOOLS_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Hungarian matching will not be available.")

# TensorRT Logger Severity
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MAX_OUTPUT_DETECTIONS = 300
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

# --- Utility Functions (same as ONNX) ---

def generate_color_map(num_classes: int) -> list:
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

def draw_detections_cv2(image, labels, boxes, scores, label_map=None, color_map=None):
    if label_map is None: label_map = {}
    if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0:
        return image
    color_dict = {i: color for i, color in enumerate(color_map)} if color_map else {}
    for i, box in enumerate(boxes):
        label_id = int(labels[i])
        score = scores[i]
        x1, y1, x2, y2 = map(int, box)
        color = color_dict.get(label_id, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        class_name = label_map.get(label_id, str(label_id))
        text = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4; thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tag_y = y1 - text_h - baseline - 3
        if tag_y < 0: tag_y = y1 + baseline + 1
        cv2.rectangle(image, (x1, tag_y), (x1 + text_w + 2, tag_y + text_h + baseline), color, -1)
        cv2.putText(image, text, (x1 + 1, tag_y + text_h), font, font_scale, (0,0,0), thickness)
    return image

def auto_detect_mode(input_source: str) -> str:
    if not isinstance(input_source, str):
         try: int(input_source); return "camera"
         except (ValueError, TypeError): raise TypeError(f"Input must be a path (str) or camera index (int), got {type(input_source)}")
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_exts = ['.mp4', '.avi', '.mkv', '.mov']
    if os.path.isdir(input_source): return "folder"
    elif os.path.exists(input_source):
        ext = os.path.splitext(input_source)[1].lower()
        if ext in image_exts: return "image"
        elif ext in video_exts: return "video"
        else: raise ValueError(f"Unsupported file extension: {ext}")
    else:
        try: int(input_source); return "camera"
        except ValueError:
            if input_source.startswith(("rtsp://", "http://", "https://")): return "camera"
            else: raise FileNotFoundError(f"Input source not found: {input_source}")

def preprocess_image(image_bgr, target_height, target_width):
    original_h, original_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if original_h > target_height else cv2.INTER_LINEAR
    img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=interpolation)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)
    batch_images = np.expand_dims(img_chw, axis=0)
    return np.ascontiguousarray(batch_images)

def postprocess(labels, boxes, scores, confidence_thr):
    keep = scores > confidence_thr
    if not np.any(keep): return np.array([]), np.array([]), np.array([])
    return labels[keep].astype(int), boxes[keep].astype(int), scores[keep]

# --- TensorRT Core Functions ---
def load_engine(engine_file_path):
    if not os.path.exists(engine_file_path): raise FileNotFoundError(f"TensorRT engine file not found: {engine_file_path}")
    print(f"Loading engine from: {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None: raise RuntimeError("Failed to load TensorRT engine.")
    print("TensorRT engine loaded successfully."); return engine

def allocate_buffers(engine, batch_size=1):
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding_idx in range(engine.num_io_tensors):
        binding_name = engine.get_tensor_name(binding_idx)
        shape_from_engine = engine.get_tensor_shape(binding_name)
        dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        mode = engine.get_tensor_mode(binding_name)
        if -1 in shape_from_engine:
            profile_shapes = engine.get_tensor_profile_shape(binding_name, 0)
            shape = profile_shapes[2]
            shape = tuple([batch_size if dim == -1 else dim for dim in shape])
        else: shape = shape_from_engine
        if mode == trt.TensorIOMode.OUTPUT:
            if "labels" in binding_name or "scores" in binding_name: shape = (batch_size, MAX_OUTPUT_DETECTIONS)
            elif "boxes" in binding_name: shape = (batch_size, MAX_OUTPUT_DETECTIONS, 4)
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        buffer_info = {'host': host_mem, 'device': device_mem, 'shape': shape, 'name': binding_name}
        if mode == trt.TensorIOMode.INPUT: inputs.append(buffer_info)
        else: outputs.append(buffer_info)
    return inputs, outputs, bindings, stream

def infer_tensorrt(context, engine, inputs, outputs, stream, batch_input_image, batch_orig_size):
    images_buffer = next(inp for inp in inputs if inp['name'] == "images")
    sizes_buffer = next(inp for inp in inputs if inp['name'] == "orig_target_sizes")
    np.copyto(images_buffer['host'], batch_input_image.ravel())
    np.copyto(sizes_buffer['host'], batch_orig_size.ravel())
    cuda.memcpy_htod_async(images_buffer['device'], images_buffer['host'], stream)
    cuda.memcpy_htod_async(sizes_buffer['device'], sizes_buffer['host'], stream)
    context.set_input_shape("images", batch_input_image.shape)
    context.set_input_shape("orig_target_sizes", batch_orig_size.shape)
    for buf in inputs + outputs: context.set_tensor_address(buf['name'], buf['device'])
    context.execute_async_v3(stream_handle=stream.handle)
    for out in outputs: cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()
    results = {}
    for out in outputs:
        output_shape = context.get_tensor_shape(out['name'])
        expected_size = trt.volume(output_shape)
        results[out['name']] = out['host'][:expected_size].reshape(output_shape)
    return results

# --- Evaluation Functions (same as eval.py and ONNX) ---

def iou(boxA, boxB):
    """Calculate IoU between two xyxy boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

def match_detections_hungarian(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """Hungarian matching algorithm (same as eval.py)."""
    if not SCIPY_AVAILABLE:
        print("Warning: scipy not available. Using greedy matching instead.")
        return match_detections_greedy(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh)

    if len(pred_boxes) == 0:
        unmatched_gts = list(gt_labels)
        return [], unmatched_gts, []

    if len(gt_boxes) == 0:
        unmatched_preds = list(pred_labels)
        return [], [], unmatched_preds

    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = iou(pred_box, gt_box)

    cost_matrix = -iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    matched_pred_indices = set()
    matched_gt_indices = set()

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_thresh:
            matched_pairs.append((gt_labels[gt_idx], pred_labels[pred_idx]))
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)

    unmatched_preds = [pred_labels[i] for i in range(num_preds) if i not in matched_pred_indices]
    unmatched_gts = [gt_labels[j] for j in range(num_gts) if j not in matched_gt_indices]

    return matched_pairs, unmatched_gts, unmatched_preds

def match_detections_greedy(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """Greedy matching algorithm (fallback)."""
    if len(pred_boxes) == 0:
        unmatched_gts = list(gt_labels)
        return [], unmatched_gts, []

    if len(gt_boxes) == 0:
        unmatched_preds = list(pred_labels)
        return [], [], unmatched_preds

    preds = sorted(zip(pred_boxes, pred_labels, pred_scores), key=lambda x: x[2], reverse=True)
    matched_pairs = []
    unmatched_preds = []
    assigned_gt = set()

    for p_box, p_label, p_score in preds:
        best_iou = -1
        best_gt_idx = -1

        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in assigned_gt:
                continue
            current_iou = iou(p_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            assigned_gt.add(best_gt_idx)
            matched_pairs.append((gt_labels[best_gt_idx], p_label))
        else:
            unmatched_preds.append(p_label)

    unmatched_gts = [gt_labels[i] for i in range(len(gt_labels)) if i not in assigned_gt]
    return matched_pairs, unmatched_gts, unmatched_preds

def plot_score_distributions(score_distribution, label_map, output_dir, detection_threshold=0.5):
    """Generate score distribution histograms per class (from eval.py)."""
    if not PLOTTING_AVAILABLE:
        return

    num_classes = len(score_distribution)
    cols = 4
    rows = (num_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if num_classes > 1 else [axes]

    sorted_classes = sorted(score_distribution.items())

    for idx, (cat_id, scores) in enumerate(sorted_classes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        cat_name = label_map.get(cat_id, str(cat_id))

        if scores:
            scores_filtered = [s for s in scores if s > 0.01]

            if scores_filtered:
                ax.hist(scores_filtered, bins=50, range=(0, 1), color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.axvline(x=detection_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={detection_threshold}')
                ax.axvline(x=0.25, color='orange', linestyle='--', linewidth=1.5, label='Threshold=0.25')

                above_threshold = sum(1 for s in scores if s > detection_threshold)
                above_025 = sum(1 for s in scores if s > 0.25)
                above_01 = sum(1 for s in scores if s > 0.1)
                total = len(scores)

                ax.set_title(f"{cat_name}\n>{detection_threshold}: {above_threshold} | >0.25: {above_025} | >0.1: {above_01}\nTotal: {total}",
                            fontsize=9)
            else:
                ax.set_title(f"{cat_name}\nNo predictions >0.01", fontsize=9)
        else:
            ax.set_title(f"{cat_name}\nNo predictions", fontsize=9)

        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_xlim(0, 1)

    for idx in range(len(sorted_classes), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Score distribution per class (evaluation threshold: {detection_threshold})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score distribution saved to: {output_dir}/score_distributions.png")

def plot_confusion_matrix(cm, class_names, output_path):
    if not PLOTTING_AVAILABLE:
        print("Warning: Matplotlib/Seaborn not installed. Skipping confusion matrix plot.")
        return
    try:
        num_classes = len(class_names)
        fig_size = max(10, num_classes * 0.8)

        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Ground Truth", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to {output_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def run_evaluation(args, context, engine, inputs, outputs, stream):
    global PYNVML_AVAILABLE
    if not COCOTOOLS_AVAILABLE:
        print("Error: `pycocotools` is required. Install with: pip install pycocotools"); return

    print("\n" + "=" * 70)
    print("TENSORRT EVALUATION MODE")
    print("=" * 70)
    print(f"  Matching method:          hungarian")
    print(f"  IoU threshold:            {args.iou_threshold}")
    print(f"  Detection threshold:      {args.detection_threshold}")
    print(f"  COCO eval threshold:      0.001 (low threshold for mAP)")
    print("=" * 70 + "\n")

    handle = None
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit(); handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("NVML initialized for GPU monitoring.")
        except Exception as e:
            print(f"Warning: NVML initialization failed. GPU monitoring disabled. Error: {e}")
            PYNVML_AVAILABLE = False

    coco_gt = COCO(args.eval_json)
    print(f"Scanning for image paths in '{args.eval_root_dir}'...")
    image_path_cache = {os.path.basename(f): os.path.join(dp, f) for dp, _, fn in os.walk(args.eval_root_dir) for f in fn}
    print(f"Found {len(image_path_cache)} files. Evaluating on {len(coco_gt.getImgIds())} images.")

    label_map = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}

    predictions_for_coco = []
    inference_times, gpu_utilizations, gpu_memory_used = [], [], []

    cat_id_map = {i: cat_id for i, cat_id in enumerate(coco_gt.getCatIds())}

    # Per-class metrics
    all_y_true, all_y_pred = [], []
    fp_counter, fn_counter = Counter(), Counter()
    score_distribution = {cat_id: [] for cat_id in label_map.keys()}

    total_matched = 0
    total_fn = 0
    total_fp = 0
    total_gt = 0
    total_pred = 0

    for image_id in tqdm(coco_gt.getImgIds(), desc="Evaluating Images"):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = image_path_cache.get(os.path.basename(image_info['file_name']))
        if not image_path: continue
        frame = cv2.imread(image_path)
        if frame is None: continue

        input_image_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)

        start_time = time.perf_counter()
        # IMPORTANT: orig_target_sizes must be [W, H] per the postprocessor convention
        trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, np.array([[frame.shape[1], frame.shape[0]]]))
        inference_times.append(time.perf_counter() - start_time)

        if PYNVML_AVAILABLE and handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilizations.append(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_used.append(mem.used / (1024**2))
            except pynvml.NVMLError as e:
                if "Not Supported" in str(e) and handle:
                    print(f"\nWarning: GPU metric collection not supported. Disabling.")
                    handle = None

        if trt_outputs is None: continue
        labels_out, boxes_out, scores_out = [trt_outputs.get(n, np.array([])) for n in ["labels", "boxes", "scores"]]
        if labels_out.ndim > 1: labels_out, boxes_out, scores_out = labels_out[0], boxes_out[0], scores_out[0]

        # Get GT annotations
        ann_ids = coco_gt.getAnnIds(imgIds=image_id)
        annotations = coco_gt.loadAnns(ann_ids)

        gt_boxes = []
        gt_labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])
            gt_labels.append(ann['category_id'])

        total_gt += len(gt_labels)

        # Collect score distribution (BEFORE filtering)
        for j in range(len(labels_out)):
            pred_class = cat_id_map.get(labels_out[j])
            if pred_class is not None and pred_class in score_distribution:
                score_distribution[pred_class].append(scores_out[j])

        # COCO predictions (low threshold for mAP)
        for label_id, box, score in zip(labels_out, boxes_out, scores_out):
            if score < 0.001: continue
            x1, y1, x2, y2 = box
            bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            coco_cat_id = cat_id_map.get(label_id)
            if coco_cat_id is not None:
                predictions_for_coco.append({"image_id": image_id, "category_id": coco_cat_id, "bbox": bbox_coco, "score": float(score)})

        # Filter for confusion matrix and per-class metrics (use detection_threshold)
        filtered_indices = scores_out > args.detection_threshold
        filtered_pred_boxes = []
        filtered_pred_labels = []
        filtered_pred_scores = []

        for j, keep in enumerate(filtered_indices):
            if keep:
                box = boxes_out[j]
                coco_cat_id = cat_id_map.get(labels_out[j])
                if coco_cat_id is not None:
                    filtered_pred_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
                    filtered_pred_labels.append(coco_cat_id)
                    filtered_pred_scores.append(float(scores_out[j]))

        total_pred += len(filtered_pred_labels)

        # Hungarian matching
        matched, unmatched_gts, unmatched_preds = match_detections_hungarian(
            gt_boxes, gt_labels, filtered_pred_boxes, filtered_pred_labels, filtered_pred_scores,
            args.iou_threshold)

        total_matched += len(matched)
        total_fn += len(unmatched_gts)
        total_fp += len(unmatched_preds)

        for gt_lab, pred_lab in matched:
            all_y_true.append(gt_lab)
            all_y_pred.append(pred_lab)
        for gt_lab in unmatched_gts:
            fn_counter[gt_lab] += 1
        for pred_lab in unmatched_preds:
            fp_counter[pred_lab] += 1

    print("\n--- Performance Results ---")
    if len(inference_times) > 1:
        avg_latency_ms = np.mean(inference_times[1:]) * 1000
        avg_fps = 1.0 / np.mean(inference_times[1:])
        print(f"Average latency: {avg_latency_ms:.2f} ms")
        print(f"Throughput (FPS): {avg_fps:.2f} FPS")

    # Score distributions
    print("\n--- Generating per-class score distribution ---")
    plot_score_distributions(score_distribution, label_map, args.output_dir, args.detection_threshold)

    # COCO Evaluation
    print("\n--- Running Official COCO Evaluation ---")
    coco_dt = coco_gt.loadRes(predictions_for_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate()

    original_stdout = sys.stdout; string_stdout = io.StringIO()
    sys.stdout = string_stdout; coco_eval.summarize(); sys.stdout = original_stdout
    coco_summary = string_stdout.getvalue()
    print(coco_summary)

    # Confusion Matrix
    print("\n--- Confusion Matrix (Hungarian Matching) ---")
    labels_to_report = sorted(list(set(all_y_true + all_y_pred)))
    target_names = [label_map.get(l, f"Class_{l}") for l in labels_to_report]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_to_report)

    os.makedirs(args.output_dir, exist_ok=True)
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, target_names, cm_path)

    # Per-class AP
    print("\n--- Per-class AP (COCO) ---")
    precisions = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()
    cat_names_list = [coco_gt.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]

    per_class_ap = {}
    per_class_ap50 = {}
    per_class_ap75 = {}
    per_class_report = "Per-class AP (COCO Metrics)\n" + "=" * 60 + "\n"
    per_class_report += f"{'Class':<20} {'AP@0.5:0.95':>12} {'AP@0.50':>12} {'AP@0.75':>12}\n"
    per_class_report += "-" * 60 + "\n"

    for idx, cat_id in enumerate(cat_ids):
        cat_name = cat_names_list[idx]

        precision_all = precisions[:, :, idx, 0, -1]
        precision_all = precision_all[precision_all > -1]
        ap = np.mean(precision_all) if precision_all.size else 0.0
        per_class_ap[cat_name] = ap

        precision_50 = precisions[0, :, idx, 0, -1]
        precision_50 = precision_50[precision_50 > -1]
        ap50 = np.mean(precision_50) if precision_50.size else 0.0
        per_class_ap50[cat_name] = ap50

        precision_75 = precisions[5, :, idx, 0, -1]
        precision_75 = precision_75[precision_75 > -1]
        ap75 = np.mean(precision_75) if precision_75.size else 0.0
        per_class_ap75[cat_name] = ap75

        per_class_report += f"{cat_name:<20} {ap:>12.4f} {ap50:>12.4f} {ap75:>12.4f}\n"

    print(per_class_report)

    # Save reports
    perf_stats = {
        "total_images": len(inference_times),
        "avg_latency_ms": np.mean(inference_times) * 1000 if inference_times else 0,
        "avg_fps": 1.0 / np.mean(inference_times) if inference_times else 0,
        "avg_gpu_utilization_percent": float(np.mean(gpu_utilizations)) if gpu_utilizations else "N/A",
        "avg_gpu_memory_used_mb": float(np.mean(gpu_memory_used)) if gpu_memory_used else "N/A"
    }

    with open(os.path.join(args.output_dir, "evaluation_summary.txt"), 'w') as f:
        f.write("--- Official COCO Evaluation Summary ---\n"); f.write(coco_summary)
        f.write("\n--- Performance Stats ---\n")
        for key, val in perf_stats.items():
            line = f"{key:30}: {val:.2f}\n" if isinstance(val, (float, np.floating)) else f"{key:30}: {val}\n"
            f.write(line)
        f.write("\n" + per_class_report)

    with open(os.path.join(args.output_dir, "performance_stats.json"), 'w') as f: json.dump(perf_stats, f, indent=4)
    with open(os.path.join(args.output_dir, "evaluation_metrics_coco.json"), 'w') as f: json.dump(coco_eval.stats.tolist(), f, indent=4)
    with open(os.path.join(args.output_dir, "per_class_ap_report.txt"), 'w') as f: f.write(per_class_report)

    # Diagnostic summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"\n{'GLOBAL STATISTICS':^70}")
    print("-" * 70)
    print(f"  Total Ground Truth:       {total_gt:>8}")
    print(f"  Total predictions:       {total_pred:>8}")
    print(f"  Total matched:           {total_matched:>8}")
    print(f"  False negatives (FN):    {total_fn:>8}  ({100*total_fn/max(total_gt,1):.1f}% of GT)")
    print(f"  False positives (FP):    {total_fp:>8}  ({100*total_fp/max(total_pred,1):.1f}% of pred)")

    global_recall = total_matched / max(total_gt, 1)
    global_precision = total_matched / max(total_pred, 1)
    print(f"\n  Global recall:           {global_recall:>8.2%}")
    print(f"  Global precision:        {global_precision:>8.2%}")

    # Per-class metrics
    class_tp_count = Counter()
    for yt, yp in zip(all_y_true, all_y_pred):
        if yt == yp:
            class_tp_count[yt] += 1

    print(f"\n{'PER-CLASS ANALYSIS':^90}")
    print("-" * 90)
    print(f"{'Class':<20} {'GT':>6} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>8} {'Precision':>10} {'F1':>8}")
    print("-" * 90)

    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]
        tp = class_tp_count.get(cat_id, 0)
        fn = fn_counter.get(cat_id, 0)
        fp = fp_counter.get(cat_id, 0)
        total_gt_class = tp + fn
        total_pred_class = tp + fp
        recall = tp / max(total_gt_class, 1)
        precision = tp / max(total_pred_class, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
        print(f"{cat_name:<20} {total_gt_class:>6} {tp:>6} {fn:>6} {fp:>6} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f}")

    print("=" * 70)
    print(f"\nReports saved to: {args.output_dir}")

    if PYNVML_AVAILABLE: pynvml.nvmlShutdown()

# --- Main Execution Logic ---
def main(args):
    engine = load_engine(args.engine_file)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size=1)

    if args.eval_json and args.eval_root_dir:
        run_evaluation(args, context, engine, inputs, outputs, stream); return

    if not args.input:
        print("Error: No input for inference. Use '-i' or evaluation paths."); return

    mode = auto_detect_mode(args.input)
    os.makedirs(args.save_dir, exist_ok=True)
    FALLBACK_CLASS_MAP = {i: f"class_{i}" for i in range(200)}
    color_map = generate_color_map(len(FALLBACK_CLASS_MAP))
    engine_name = os.path.splitext(os.path.basename(args.engine_file))[0]

    if mode == "image":
        frame = cv2.imread(args.input)
        if frame is None: print(f"Error reading image: {args.input}"); return
        input_image_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
        # IMPORTANT: orig_target_sizes must be [W, H] per the postprocessor convention
        trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, np.array([[frame.shape[1], frame.shape[0]]]))
        if trt_outputs:
            labels_out, boxes_out, scores_out = [trt_outputs.get(n, np.array([])) for n in ["labels", "boxes", "scores"]]
            if labels_out.ndim > 1: labels_out, boxes_out, scores_out = labels_out[0], boxes_out[0], scores_out[0]
            labels, boxes, scores = postprocess(labels_out, boxes_out, scores_out, args.threshold)
            annotated_frame = draw_detections_cv2(frame.copy(), labels, boxes, scores, FALLBACK_CLASS_MAP, color_map)
            if args.show: cv2.imshow("Inference", annotated_frame); cv2.waitKey(0)
            else:
                out_path = os.path.join(args.save_dir, f"{os.path.basename(args.input)}_{engine_name}.jpg")
                cv2.imwrite(out_path, annotated_frame); print(f"Saved to {out_path}")

    elif mode == "folder":
        if args.show: print("Warning: --show not supported for folder mode.")
        image_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_path in tqdm(image_files, desc="Processing folder"):
            frame = cv2.imread(image_path)
            if frame is None: continue
            input_image_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
            # IMPORTANT: orig_target_sizes must be [W, H] per the postprocessor convention
            trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, np.array([[frame.shape[1], frame.shape[0]]]))
            if trt_outputs:
                labels_out, boxes_out, scores_out = [trt_outputs.get(n, np.array([])) for n in ["labels", "boxes", "scores"]]
                if labels_out.ndim > 1: labels_out, boxes_out, scores_out = labels_out[0], boxes_out[0], scores_out[0]
                labels, boxes, scores = postprocess(labels_out, boxes_out, scores_out, args.threshold)
                annotated_frame = draw_detections_cv2(frame.copy(), labels, boxes, scores, FALLBACK_CLASS_MAP, color_map)
                out_path = os.path.join(args.save_dir, f"{os.path.basename(image_path)}_{engine_name}.jpg")
                cv2.imwrite(out_path, annotated_frame)

    elif mode in ["video", "camera"]:
        cap = cv2.VideoCapture(int(args.input) if mode == "camera" and args.input.isdigit() else args.input)
        if not cap.isOpened(): print(f"Error opening source: {args.input}"); return
        w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
        out_writer = None
        if not args.show:
            name = f"cam_{args.input}" if mode == "camera" else os.path.splitext(os.path.basename(args.input))[0]
            out_path = os.path.join(args.save_dir, f"{name}_{engine_name}.mp4")
            out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        while True:
            ret, frame = cap.read()
            if not ret: break
            input_image_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
            # IMPORTANT: orig_target_sizes must be [W, H] per the postprocessor convention
            trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, np.array([[frame.shape[1], frame.shape[0]]]))
            annotated_frame = frame.copy()
            if trt_outputs:
                labels_out, boxes_out, scores_out = [trt_outputs.get(n, np.array([])) for n in ["labels", "boxes", "scores"]]
                if labels_out.ndim > 1: labels_out, boxes_out, scores_out = labels_out[0], boxes_out[0], scores_out[0]
                labels, boxes, scores = postprocess(labels_out, boxes_out, scores_out, args.threshold)
                annotated_frame = draw_detections_cv2(annotated_frame, labels, boxes, scores, FALLBACK_CLASS_MAP, color_map)
            if args.show: cv2.imshow("Inference", annotated_frame);
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            elif out_writer: out_writer.write(annotated_frame)
        cap.release();
        if out_writer: out_writer.release()

    cv2.destroyAllWindows()
    print("Script finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TensorRT inference and evaluation script for RT-DETRv2 models (eval.py compatible).",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--engine_file', type=str, required=True, help="Path to the TensorRT engine file.")

    inference_group = parser.add_argument_group('Inference Arguments')
    inference_group.add_argument('-i', '--input', type=str, help="Path for inference (image, video, folder, or camera index).")
    inference_group.add_argument('--save_dir', type=str, default='inference_results_tensorrt', help="Directory to save inference outputs.")
    inference_group.add_argument('--show', action='store_true', help="Display inference results in a window.")

    eval_group = parser.add_argument_group('Evaluation Arguments')
    eval_group.add_argument('--eval_json', type=str, help="Path to the COCO-style JSON annotation file.")
    eval_group.add_argument('--eval_root_dir', type=str, help="Root directory of the dataset images for evaluation.")
    eval_group.add_argument('--output_dir', type=str, default='evaluation_results_tensorrt', help="Directory to save all evaluation metric files.")

    param_group = parser.add_argument_group('Common Parameters')
    param_group.add_argument('--threshold', type=str, default=0.5, help="Confidence threshold for inference display.")
    param_group.add_argument('--detection_threshold', type=float, default=0.5, help="Confidence threshold for per-class metrics and confusion matrix.")
    param_group.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for Hungarian matching.")

    args = parser.parse_args()

    if not args.eval_json and not args.input:
        parser.error("You must specify either inference input with '-i' or evaluation paths.")
    if args.eval_json and not args.eval_root_dir:
        parser.error("'--eval_root_dir' is required when using '--eval_json'.")

    if args.eval_json and not COCOTOOLS_AVAILABLE:
        print("\nERROR: 'pycocotools' is required for evaluation mode.\nInstall with: pip install pycocotools\n")
        sys.exit(1)

    main(args)
