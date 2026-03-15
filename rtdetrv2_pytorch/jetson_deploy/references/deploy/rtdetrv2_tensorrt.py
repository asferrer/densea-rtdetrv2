#!/usr/bin/env python
"""
TensorRT inference and evaluation script for RT-DETRv2 models.

Features:
- Handles image, video, folder, camera/stream, and evaluation modes.
- Optimized for TensorRT engines (.engine files).
- Evaluation mode uses the official `pycocotools` library for standard metrics.
- It also calculates and saves a confusion matrix, now with "Background" as the first class
  for clearer visual analysis of False Positives and False Negatives.
- Saves all metrics to an output directory: COCO summary, performance stats, and confusion matrix.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Important: Initializes CUDA context and handles cleanup
import time
from tqdm import tqdm
import colorsys
import json
from collections import defaultdict
import io

# --- Optional Imports for Metrics ---
try:
    # NOTE: The 'nvidia-ml-py' package provides the 'pynvml' module.
    # Install with: pip install nvidia-ml-py
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
    # Official COCO evaluation library
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCOTOOLS_AVAILABLE = True
except ImportError:
    COCOTOOLS_AVAILABLE = False


# TensorRT Logger Severity
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MAX_OUTPUT_DETECTIONS = 300
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

# --- Utility Functions ---

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

# --- Preprocessing and Postprocessing ---

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

# --- Evaluation Functions ---

def calculate_iou(boxA_xywh, boxB_xyxy):
    """Calculates IoU between a COCO box (xywh) and a predicted box (xyxy)."""
    boxA_xyxy = [boxA_xywh[0], boxA_xywh[1], boxA_xywh[0] + boxA_xywh[2], boxA_xywh[1] + boxA_xywh[3]]
    xA = max(boxA_xyxy[0], boxB_xyxy[0]); yA = max(boxA_xyxy[1], boxB_xyxy[1])
    xB = min(boxA_xyxy[2], boxB_xyxy[2]); yB = min(boxA_xyxy[3], boxB_xyxy[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA_xywh[2] * boxA_xywh[3]
    boxBArea = (boxB_xyxy[2] - boxB_xyxy[0]) * (boxB_xyxy[3] - boxB_xyxy[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

def plot_confusion_matrix(cm, class_names, output_path):
    if not PLOTTING_AVAILABLE:
        print("Warning: Matplotlib/Seaborn not installed. Skipping confusion matrix plot.")
        return
    try:
        figure = plt.figure(figsize=(14, 14))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("True Label"); plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix"); plt.tight_layout()
        plt.savefig(output_path); plt.close()
        print(f"Confusion matrix plot saved to {output_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def calculate_confusion_matrix(coco_gt, predictions_for_coco, iou_threshold=0.5):
    """Calculates a confusion matrix from predictions and ground truth."""
    cat_ids = coco_gt.getCatIds()
    id_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    num_classes = len(cat_ids)
    # Last row/col is for background
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    background_idx = num_classes

    gt_by_image = defaultdict(list)
    for ann in coco_gt.dataset['annotations']:
        gt_by_image[ann['image_id']].append(ann)
    
    preds_by_image = defaultdict(list)
    for pred in predictions_for_coco:
        preds_by_image[pred['image_id']].append(pred)

    for img_id in coco_gt.getImgIds():
        gts = gt_by_image.get(img_id, [])
        preds = sorted(preds_by_image.get(img_id, []), key=lambda p: p['score'], reverse=True)
        
        for gt in gts: gt['found'] = False

        for pred in preds:
            pred_cat_idx = id_to_idx.get(pred['category_id'])
            if pred_cat_idx is None: continue

            best_iou = 0; best_gt = None
            for gt in gts:
                # Check for same category and if GT has already been matched
                if gt['category_id'] == pred['category_id'] and not gt['found']:
                    iou = calculate_iou(gt['bbox'], [pred['bbox'][0], pred['bbox'][1], pred['bbox'][0]+pred['bbox'][2], pred['bbox'][1]+pred['bbox'][3]])
                    if iou > best_iou:
                        best_iou = iou; best_gt = gt
            
            if best_iou >= iou_threshold:
                if best_gt:
                    gt_cat_idx = id_to_idx[best_gt['category_id']]
                    confusion_matrix[gt_cat_idx, pred_cat_idx] += 1 # True Positive
                    best_gt['found'] = True
            else:
                # This prediction did not match any GT of the same class well enough.
                # It's a False Positive. The true class is 'background'.
                confusion_matrix[background_idx, pred_cat_idx] += 1

        # Any GTs not found are False Negatives. The prediction is 'background'.
        for gt in gts:
            if not gt['found']:
                gt_cat_idx = id_to_idx[gt['category_id']]
                confusion_matrix[gt_cat_idx, background_idx] += 1
    
    return confusion_matrix, [coco_gt.cats[cat_id]['name'] for cat_id in cat_ids]


def run_evaluation(args, context, engine, inputs, outputs, stream):
    global PYNVML_AVAILABLE
    if not COCOTOOLS_AVAILABLE:
        print("Error: `pycocotools` is required. Install with: pip install pycocotools"); return

    print("--- Starting Evaluation Mode ---")
    
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

    predictions_for_coco = []
    inference_times, gpu_utilizations, gpu_memory_used = [], [], []

    cat_id_map = {i: cat_id for i, cat_id in enumerate(coco_gt.getCatIds())}

    for image_id in tqdm(coco_gt.getImgIds(), desc="Evaluating Images"):
        image_info = coco_gt.loadImgs(image_id)[0]
        image_path = image_path_cache.get(os.path.basename(image_info['file_name']))
        if not image_path: continue
        frame = cv2.imread(image_path)
        if frame is None: continue
        
        input_image_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)

        start_time = time.perf_counter()
        # IMPORTANTE: orig_target_sizes debe ser [W, H] según la convención del postprocessor
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
        
        for label_id, box, score in zip(labels_out, boxes_out, scores_out):
            if score < args.threshold: continue
            x1, y1, x2, y2 = box
            bbox_coco = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            coco_cat_id = cat_id_map.get(label_id)
            if coco_cat_id is not None:
                predictions_for_coco.append({"image_id": image_id, "category_id": coco_cat_id, "bbox": bbox_coco, "score": float(score)})

    print("\n--- Running Official COCO Evaluation ---")
    coco_dt = coco_gt.loadRes(predictions_for_coco)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate(); coco_eval.accumulate()
    
    original_stdout = sys.stdout; string_stdout = io.StringIO()
    sys.stdout = string_stdout; coco_eval.summarize(); sys.stdout = original_stdout
    coco_summary = string_stdout.getvalue()
    print(coco_summary)

    print("\n--- Calculating Confusion Matrix ---")
    cm, cm_class_names = calculate_confusion_matrix(coco_gt, predictions_for_coco, iou_threshold=args.iou_threshold)
    
    # --- NEW: Reorder confusion matrix to have Background first ---
    num_classes = len(cm_class_names)
    cm_reordered = np.zeros_like(cm)
    # Main TP/confusion block (indices 1 to N)
    cm_reordered[1:, 1:] = cm[:num_classes, :num_classes]
    # Background row (False Positives)
    cm_reordered[0, 1:] = cm[num_classes, :num_classes]
    # Background column (False Negatives)
    cm_reordered[1:, 0] = cm[:num_classes, num_classes]
    # Reordered labels for plotting
    class_names_for_plot = ["Background"] + cm_class_names
    # --- END NEW SECTION ---
    
    os.makedirs(args.output_dir, exist_ok=True)
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
    print(f"Summary report saved to {os.path.join(args.output_dir, 'evaluation_summary.txt')}")
    
    with open(os.path.join(args.output_dir, "performance_stats.json"), 'w') as f: json.dump(perf_stats, f, indent=4)
    with open(os.path.join(args.output_dir, "evaluation_metrics_coco.json"), 'w') as f: json.dump(coco_eval.stats.tolist(), f, indent=4)

    cm_path_npy = os.path.join(args.output_dir, "confusion_matrix.npy")
    np.save(cm_path_npy, cm_reordered)
    print(f"Reordered confusion matrix data saved to {cm_path_npy}")
    plot_confusion_matrix(cm_reordered, class_names_for_plot, os.path.join(args.output_dir, "confusion_matrix.png"))

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
        # IMPORTANTE: orig_target_sizes debe ser [W, H] según la convención del postprocessor
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
            # IMPORTANTE: orig_target_sizes debe ser [W, H] según la convención del postprocessor
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
            # IMPORTANTE: orig_target_sizes debe ser [W, H] según la convención del postprocessor
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
        description="TensorRT inference and evaluation script for RT-DETRv2 models.",
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--engine_file', type=str, required=True, help="Path to the TensorRT engine file.")
    
    inference_group = parser.add_argument_group('Inference Arguments')
    inference_group.add_argument('-i', '--input', type=str, help="Path for inference (image, video, folder, or camera index).")
    inference_group.add_argument('--save_dir', type=str, default='inference_results', help="Directory to save inference outputs.")
    inference_group.add_argument('--show', action='store_true', help="Display inference results in a window.")
    
    eval_group = parser.add_argument_group('Evaluation Arguments')
    eval_group.add_argument('--eval_json', type=str, help="Path to the COCO-style JSON annotation file.")
    eval_group.add_argument('--eval_root_dir', type=str, help="Root directory of the dataset images for evaluation.")
    eval_group.add_argument('--output_dir', type=str, default='evaluation_results', help="Directory to save all evaluation metric files.")

    param_group = parser.add_argument_group('Common Parameters')
    param_group.add_argument('--threshold', type=float, default=0.5, help="Confidence threshold for detections.\n(For fair COCO evaluation, use a low value like 0.001)")
    param_group.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for confusion matrix calculation.")
    
    args = parser.parse_args()
    
    if not args.eval_json and not args.input:
        parser.error("You must specify either inference input with '-i' or evaluation paths.")
    if args.eval_json and not args.eval_root_dir:
        parser.error("'--eval_root_dir' is required when using '--eval_json'.")

    if args.eval_json and not COCOTOOLS_AVAILABLE:
        print("\nERROR: 'pycocotools' is required for evaluation mode.\nInstall with: pip install pycocotools\n")
        sys.exit(1)

    main(args)