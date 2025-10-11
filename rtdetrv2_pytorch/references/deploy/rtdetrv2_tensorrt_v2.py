#!/usr/bin/env python
"""
TensorRT inference and evaluation script for RT-DETRv2 models.

Features:
- Handles image, video, folder, camera/stream, and evaluation modes.
- Optimized for TensorRT engines (.engine files), especially with FP16.
- Supports engines built with dynamic shapes (dynamic batch size).
- Allocates necessary host (pinned) and device (GPU) memory using PyCUDA.
- Performs asynchronous inference using CUDA streams.
- Includes preprocessing and postprocessing steps.
- Evaluation mode to compute mAP using a COCO-style JSON annotation file.
  - Recursively finds images in any subdirectory.
  - Caches image paths for high performance.
- Saves all evaluation metrics to an output directory:
  - Performance stats (FPS, latency, GPU usage) in JSON and TXT.
  - mAP results per class in JSON.
  - Confusion matrix (raw .npy and optional .png visualization).
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


# TensorRT Logger Severity
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Default maximum number of detections expected from the model output.
MAX_OUTPUT_DETECTIONS = 300

# Default input dimensions (adjust if your model differs)
INPUT_HEIGHT = 640
INPUT_WIDTH = 640

# --- Utility Functions ---

def generate_color_map(num_classes: int) -> list:
    """Generates a list of distinct BGR colors for drawing."""
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        # Convert to BGR for OpenCV
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors

def draw_detections_cv2(image, labels, boxes, scores, label_map=None, color_map=None):
    """
    Draws bounding boxes and labels on an image using OpenCV with a minimalist style.
    """
    if label_map is None: label_map = {}
    if not isinstance(labels, (list, np.ndarray)) or len(labels) == 0:
        return image # Return original image if no detections
    img_h, img_w = image.shape[:2]
    # Create a dictionary from the color_map list for easier lookup
    color_dict = {i: color for i, color in enumerate(color_map)} if color_map else {}

    for i, box in enumerate(boxes):
        label_id = int(labels[i])
        score = scores[i]
        x1, y1, x2, y2 = map(int, box) # Ensure coordinates are integers
        
        color = color_dict.get(label_id, (0, 255, 0)) # Default to green
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        class_name = label_map.get(label_id, str(label_id))
        text = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        tag_x = x1
        tag_y = y1 - text_h - baseline - 3 # Default position above
        if tag_y < 0: # If tag goes above image, place it inside the box
            tag_y = y1 + baseline + 1
        cv2.rectangle(image, (tag_x, tag_y), (tag_x + text_w + 2, tag_y + text_h + baseline), color, -1)
        cv2.putText(image, text, (tag_x + 1, tag_y + text_h), font, font_scale, (0,0,0), thickness)
    return image

def auto_detect_mode(input_source: str) -> str:
    """
    Automatically detects the input mode (image, video, folder, camera/stream).
    """
    if not isinstance(input_source, str):
         try: int(input_source); return "camera"
         except (ValueError, TypeError): raise TypeError(f"Input must be path (str) or camera index (int), got {type(input_source)}")

    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_exts = ['.mp4', '.avi', '.mkv', '.mov']

    if os.path.isdir(input_source):
        return "folder"
    elif os.path.exists(input_source):
        ext = os.path.splitext(input_source)[1].lower()
        if ext in image_exts: return "image"
        elif ext in video_exts: return "video"
        else:
            # Fallback check for video files without common extensions
            cap_test = cv2.VideoCapture(input_source)
            if cap_test.isOpened(): cap_test.release(); return "video"
            else: print(f"Warning: Unknown extension '{ext}', treating as image."); return "image"
    else:
        try: int(input_source); return "camera"
        except ValueError:
            if input_source.startswith(("rtsp://", "http://", "https://")): return "camera"
            else: raise ValueError(f"Input source not found or not recognized: {input_source}")

# --- Preprocessing and Postprocessing ---

def preprocess_image(image_bgr, target_height, target_width):
    """
    Preprocesses an image (BGR NumPy array) for RT-DETR inference.
    """
    original_h, original_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if original_h > target_height else cv2.INTER_LINEAR
    img_resized = cv2.resize(img_rgb, (target_width, target_height), interpolation=interpolation)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)
    batch_images = np.expand_dims(img_chw, axis=0)
    batch_sizes = np.array([[original_w, original_h]], dtype=np.int64)
    return np.ascontiguousarray(batch_images), np.ascontiguousarray(batch_sizes)

def postprocess(labels, boxes, scores, confidence_thr):
    """
    Postprocesses the raw outputs from the RT-DETR model.
    """
    keep = scores > confidence_thr
    if not np.any(keep): return np.array([]), np.array([]), np.array([])
    return labels[keep].astype(int), boxes[keep].astype(int), scores[keep]


# --- TensorRT Core Functions ---
def load_engine(engine_file_path):
    """Loads a TensorRT engine from file."""
    if not os.path.exists(engine_file_path): raise FileNotFoundError(f"TensorRT engine file not found: {engine_file_path}")
    print(f"Loading engine from: {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None: raise RuntimeError("Failed to load TensorRT engine.")
    print("TensorRT engine loaded successfully."); return engine

def allocate_buffers(engine, batch_size=1):
    """
    Allocates host (pinned) and device (GPU) buffers for TRT engine bindings.
    """
    inputs = []; outputs = []; bindings = []; stream = cuda.Stream()
    print(f"Allocating buffers for max batch size {batch_size} (based on profile or assumption)...")
    for binding_idx in range(engine.num_io_tensors):
        binding_name = engine.get_tensor_name(binding_idx)
        shape_from_engine = engine.get_tensor_shape(binding_name)
        dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        mode = engine.get_tensor_mode(binding_name)
        
        if -1 in shape_from_engine:
            profile_shapes = engine.get_tensor_profile_shape(binding_name, 0)
            shape = profile_shapes[2] # max shape
            shape = tuple([batch_size if dim == -1 else dim for dim in shape])
        else:
            shape = shape_from_engine
            
        if mode == trt.TensorIOMode.OUTPUT:
            if "labels" in binding_name or "scores" in binding_name: shape = (batch_size, MAX_OUTPUT_DETECTIONS)
            elif "boxes" in binding_name: shape = (batch_size, MAX_OUTPUT_DETECTIONS, 4)
            
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        buffer_info = {'host': host_mem, 'device': device_mem, 'shape': shape, 'name': binding_name, 'allocated_size': size}
        if mode == trt.TensorIOMode.INPUT: inputs.append(buffer_info)
        else: outputs.append(buffer_info)
    return inputs, outputs, bindings, stream

def infer_tensorrt(context, engine, inputs, outputs, stream, batch_input_image, batch_orig_size):
    """
    Performs asynchronous inference using TensorRT context and pre-allocated buffers.
    """
    images_buffer = next(inp for inp in inputs if inp['name'] == "images")
    sizes_buffer = next(inp for inp in inputs if inp['name'] == "orig_target_sizes")
    
    np.copyto(images_buffer['host'], batch_input_image.ravel())
    np.copyto(sizes_buffer['host'], batch_orig_size.ravel())
    
    cuda.memcpy_htod_async(images_buffer['device'], images_buffer['host'], stream)
    cuda.memcpy_htod_async(sizes_buffer['device'], sizes_buffer['host'], stream)
    
    context.set_input_shape("images", batch_input_image.shape)
    context.set_input_shape("orig_target_sizes", batch_orig_size.shape)
    
    for buf in inputs + outputs:
        context.set_tensor_address(buf['name'], buf['device'])
        
    context.execute_async_v3(stream_handle=stream.handle)
    
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        
    stream.synchronize()
    
    results = {}
    for out in outputs:
        output_shape = context.get_tensor_shape(out['name'])
        expected_size = trt.volume(output_shape)
        results[out['name']] = out['host'][:expected_size].reshape(output_shape)
    return results

# --- Evaluation Functions ---

def calculate_iou(boxA_xywh, boxB_xyxy):
    """Calculates Intersection over Union (IoU) between a COCO box (xywh) and a predicted box (xyxy)."""
    boxA_xyxy = [boxA_xywh[0], boxA_xywh[1], boxA_xywh[0] + boxA_xywh[2], boxA_xywh[1] + boxA_xywh[3]]
    xA = max(boxA_xyxy[0], boxB_xyxy[0]); yA = max(boxA_xyxy[1], boxB_xyxy[1])
    xB = min(boxA_xyxy[2], boxB_xyxy[2]); yB = min(boxA_xyxy[3], boxB_xyxy[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA_xywh[2] * boxA_xywh[3]
    boxBArea = (boxB_xyxy[2] - boxB_xyxy[0]) * (boxB_xyxy[3] - boxB_xyxy[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

def compute_ap(recall, precision):
    """Computes Average Precision from recall and precision arrays."""
    mrec = np.concatenate(([0.], recall, [1.])); mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1): mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

def plot_confusion_matrix(cm, class_names, output_path):
    """Saves a graphical plot of the confusion matrix."""
    if not PLOTTING_AVAILABLE:
        print("Warning: Matplotlib/Seaborn not installed. Skipping confusion matrix plot.")
        print("Install with: pip install matplotlib seaborn")
        return
    try:
        figure = plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=class_names, yticklabels=class_names)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Confusion matrix plot saved to {output_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def run_evaluation(args, context, engine, inputs, outputs, stream):
    """Runs the complete evaluation process and saves all metrics."""
    global PYNVML_AVAILABLE
    
    print("--- Starting Evaluation Mode ---")
    
    # --- Initialize NVML for GPU monitoring ---
    handle = None
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            print("NVML initialized for GPU monitoring.")
        except Exception as e:
            print(f"Warning: Could not initialize NVML. GPU monitoring disabled. Error: {e}")
            print("To enable, please install the official package: pip install nvidia-ml-py")
            PYNVML_AVAILABLE = False

    # --- Load and Parse Annotations ---
    with open(args.eval_json) as f: annotations_data = json.load(f)
    CLASS_MAP = {cat['id']: cat['name'] for cat in annotations_data['categories']}
    images_info = {img['id']: img for img in annotations_data['images']}
    annotations_info = defaultdict(list)
    for ann in annotations_data['annotations']: annotations_info[ann['image_id']].append(ann)

    # --- Cache Image Paths ---
    print(f"Scanning for image paths in '{args.eval_root_dir}'...")
    image_path_cache = {os.path.basename(f): os.path.join(dp, f) for dp, _, fn in os.walk(args.eval_root_dir) for f in fn}
    print(f"Found {len(image_path_cache)} files. Starting evaluation on {len(images_info)} annotated images.")

    # --- Evaluation Loop ---
    all_predictions = defaultdict(list)
    ground_truths = defaultdict(list)
    inference_times, gpu_utilizations, gpu_memory_used = [], [], []

    for image_id, image_details in tqdm(images_info.items(), desc="Evaluating Images"):
        image_basename = os.path.basename(image_details['file_name'])
        image_path = image_path_cache.get(image_basename)
        if not image_path: continue
        frame = cv2.imread(image_path)
        if frame is None: continue
        
        input_image_batch, input_size_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
        
        start_time = time.perf_counter()
        trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
        inference_times.append(time.perf_counter() - start_time)

        if PYNVML_AVAILABLE and handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilizations.append(util.gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_used.append(mem.used / (1024**2)) # In MB
            except pynvml.NVMLError as e:
                if "Not Supported" in str(e):
                    if handle: # Print only once
                        print(f"\nWarning: GPU metric collection is not supported by the driver/GPU. Disabling.")
                    handle = None # Stop trying to prevent repeated warnings
                else:
                    print(f"\nWarning: An NVML error occurred: {e}")

        if trt_outputs is None: continue
        labels_out, boxes_out, scores_out = [trt_outputs.get(n, np.array([])) for n in ["labels", "boxes", "scores"]]
        if labels_out.ndim > 1: labels_out, boxes_out, scores_out = labels_out[0], boxes_out[0], scores_out[0]
        
        for label_id, box, score in zip(labels_out, boxes_out, scores_out):
            if score >= args.threshold and label_id in CLASS_MAP:
                all_predictions[label_id].append({'image_id': image_id, 'bbox': box, 'score': score})
        for ann in annotations_info.get(image_id, []):
            if ann['category_id'] in CLASS_MAP:
                ground_truths[ann['category_id']].append({'image_id': image_id, 'bbox': ann['bbox'], 'used': False})
    
    # --- Calculate Metrics ---
    print("\n--- Calculating Final Metrics ---")
    average_precisions = {}
    class_ids = sorted(CLASS_MAP.keys())
    id_to_idx = {id: i for i, id in enumerate(class_ids)}
    num_classes = len(class_ids)
    
    confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    background_idx = num_classes

    for class_id in class_ids:
        class_idx = id_to_idx[class_id]
        class_name = CLASS_MAP[class_id]
        preds = sorted(all_predictions.get(class_id, []), key=lambda x: x['score'], reverse=True)
        gts_for_class = [gt for gt in ground_truths.get(class_id, [])]
        num_gts = len(gts_for_class)
        
        for gt in gts_for_class: gt['used'] = False

        if num_gts == 0:
            average_precisions[class_name] = 1.0 if not preds else 0.0
            for _ in preds: confusion_matrix[background_idx, class_idx] += 1
            continue

        tp = np.zeros(len(preds)); fp = np.zeros(len(preds))
        
        for i, pred in enumerate(preds):
            gt_matches = [gt for gt in gts_for_class if gt['image_id'] == pred['image_id'] and not gt['used']]
            best_iou, best_gt_idx = 0, -1
            if gt_matches:
                ious = [calculate_iou(gt['bbox'], pred['bbox']) for gt in gt_matches]
                best_match_local_idx = np.argmax(ious)
                if ious[best_match_local_idx] >= args.iou_threshold:
                    original_gt_obj = gt_matches[best_match_local_idx]
                    best_gt_idx = next((idx for idx, gt in enumerate(gts_for_class) if gt is original_gt_obj), -1)

            if best_gt_idx != -1 and not gts_for_class[best_gt_idx]['used']:
                tp[i] = 1; gts_for_class[best_gt_idx]['used'] = True
                confusion_matrix[class_idx, class_idx] += 1 # Correct detection
            else:
                fp[i] = 1; confusion_matrix[background_idx, class_idx] += 1 # FP

        fn_count = len([gt for gt in gts_for_class if not gt['used']])
        confusion_matrix[class_idx, background_idx] += fn_count

        tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
        recall = tp_cumsum / num_gts if num_gts > 0 else np.zeros_like(tp_cumsum)
        precision = np.divide(tp_cumsum, (tp_cumsum + fp_cumsum), out=np.zeros_like(tp_cumsum, dtype=float), where=(tp_cumsum + fp_cumsum) != 0)
        average_precisions[class_name] = compute_ap(recall, precision)

    # --- Save Metrics ---
    os.makedirs(args.output_dir, exist_ok=True)

    perf_stats = {
        "total_images": len(inference_times),
        "avg_latency_ms": np.mean(inference_times) * 1000 if inference_times else 0,
        "std_latency_ms": np.std(inference_times) * 1000 if inference_times else 0,
        "min_latency_ms": np.min(inference_times) * 1000 if inference_times else 0,
        "max_latency_ms": np.max(inference_times) * 1000 if inference_times else 0,
        "avg_fps": 1.0 / np.mean(inference_times) if inference_times else 0,
        "avg_gpu_utilization_percent": float(np.mean(gpu_utilizations)) if gpu_utilizations else "N/A",
        "avg_gpu_memory_used_mb": float(np.mean(gpu_memory_used)) if gpu_memory_used else "N/A"
    }
    
    mean_ap = np.mean(list(average_precisions.values())) if average_precisions else 0
    eval_metrics = {"mAP": mean_ap, "per_class_AP": average_precisions}

    with open(os.path.join(args.output_dir, "evaluation_summary.txt"), 'w') as f:
        f.write("--- Evaluation Summary ---\n")
        f.write(f"mAP@{args.iou_threshold:.2f}: {mean_ap:.4f}\n\n")
        f.write("--- Per-Class AP ---\n")
        for name, ap in average_precisions.items(): f.write(f"{name:20}: {ap:.4f}\n")
        f.write("\n--- Performance Stats ---\n")
        # --- CORRECTED LINE ---
        for key, val in perf_stats.items():
            line = f"{key:30}: {val:.2f}\n" if isinstance(val, (float, np.floating)) else f"{key:30}: {val}\n"
            f.write(line)
        # --- END CORRECTION ---
    print(f"Summary report saved to {os.path.join(args.output_dir, 'evaluation_summary.txt')}")
    
    with open(os.path.join(args.output_dir, "performance_stats.json"), 'w') as f: json.dump(perf_stats, f, indent=4)
    with open(os.path.join(args.output_dir, "evaluation_metrics.json"), 'w') as f: json.dump(eval_metrics, f, indent=4)

    cm_path_npy = os.path.join(args.output_dir, "confusion_matrix.npy")
    np.save(cm_path_npy, confusion_matrix)
    print(f"Confusion matrix data saved to {cm_path_npy}")
    
    class_names_for_plot = [CLASS_MAP[cid] for cid in class_ids] + ["Background (FN)"]
    plot_confusion_matrix(confusion_matrix, class_names_for_plot, os.path.join(args.output_dir, "confusion_matrix.png"))

    if PYNVML_AVAILABLE: pynvml.nvmlShutdown()

# --- Main Execution Logic ---
def main(args):
    # --- Setup ---
    engine = load_engine(args.engine_file)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine, batch_size=1)
    
    # --- Mode Selection ---
    if args.eval_json and args.eval_root_dir:
        run_evaluation(args, context, engine, inputs, outputs, stream)
        return 
    
    if not args.input:
        print("Error: No input for inference. Use '-i' or evaluation paths."); return

    # --- INFERENCE MODES ---
    mode = auto_detect_mode(args.input)
    os.makedirs(args.save_dir, exist_ok=True)
    FALLBACK_CLASS_MAP = {i: f"class_{i}" for i in range(200)} # Generic fallback for inference
    color_map = generate_color_map(len(FALLBACK_CLASS_MAP))
    engine_name = os.path.splitext(os.path.basename(args.engine_file))[0]
    
    if mode == "image":
        frame = cv2.imread(args.input)
        if frame is None: print(f"Error reading image: {args.input}"); return
        input_image_batch, input_size_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
        trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
        if trt_outputs:
            labels_out, boxes_out, scores_out = [trt_outputs.get(n, np.array([])) for n in ["labels", "boxes", "scores"]]
            if labels_out.ndim > 1: labels_out, boxes_out, scores_out = labels_out[0], boxes_out[0], scores_out[0]
            labels, boxes, scores = postprocess(labels_out, boxes_out, scores_out, args.threshold)
            annotated_frame = draw_detections_cv2(frame.copy(), labels, boxes, scores, FALLBACK_CLASS_MAP, color_map)
            if args.show:
                cv2.imshow("Inference", annotated_frame); cv2.waitKey(0)
            else:
                out_path = os.path.join(args.save_dir, f"{os.path.basename(args.input)}_{engine_name}.jpg")
                cv2.imwrite(out_path, annotated_frame); print(f"Saved to {out_path}")

    elif mode == "folder":
        if args.show: print("Warning: --show not supported for folder mode.")
        image_files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image_path in tqdm(image_files, desc="Processing folder"):
            frame = cv2.imread(image_path)
            if frame is None: continue
            input_image_batch, input_size_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
            trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
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
            input_image_batch, input_size_batch = preprocess_image(frame, INPUT_HEIGHT, INPUT_WIDTH)
            trt_outputs = infer_tensorrt(context, engine, inputs, outputs, stream, input_image_batch, input_size_batch)
            annotated_frame = frame.copy()
            if trt_outputs:
                labels_out, boxes_out, scores_out = [trt_outputs.get(n, np.array([])) for n in ["labels", "boxes", "scores"]]
                if labels_out.ndim > 1: labels_out, boxes_out, scores_out = labels_out[0], boxes_out[0], scores_out[0]
                labels, boxes, scores = postprocess(labels_out, boxes_out, scores_out, args.threshold)
                annotated_frame = draw_detections_cv2(annotated_frame, labels, boxes, scores, FALLBACK_CLASS_MAP, color_map)
            if args.show:
                cv2.imshow("Inference", annotated_frame)
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
    
    # --- Arguments ---
    parser.add_argument('--engine_file', type=str, required=True, help="Path to the TensorRT engine file.")
    
    # Inference Arguments
    inference_group = parser.add_argument_group('Inference Arguments')
    inference_group.add_argument('-i', '--input', type=str, help="Path to input for inference (image, video, folder, or camera index).")
    inference_group.add_argument('--save_dir', type=str, default='inference_results', help="Directory to save inference outputs.")
    inference_group.add_argument('--show', action='store_true', help="Display inference results in a window.")
    
    # Evaluation Arguments
    eval_group = parser.add_argument_group('Evaluation Arguments')
    eval_group.add_argument('--eval_json', type=str, help="Path to the COCO-style JSON annotation file.")
    eval_group.add_argument('--eval_root_dir', type=str, help="Root directory of the dataset images for evaluation.")
    eval_group.add_argument('--output_dir', type=str, default='evaluation_results', help="Directory to save all evaluation metric files.")

    # Common Parameters
    param_group = parser.add_argument_group('Common Parameters')
    param_group.add_argument('--threshold', type=float, default=0.4, help="Confidence threshold for detections.")
    param_group.add_argument('--iou_threshold', type=float, default=0.5, help="IoU threshold for a match during evaluation (for mAP).")
    
    args = parser.parse_args()
    
    if not args.eval_json and not args.input:
        parser.error("You must specify either inference input with '-i' or evaluation paths.")
    if args.eval_json and not args.eval_root_dir:
        parser.error("'--eval_root_dir' is required when using '--eval_json'.")

    main(args)