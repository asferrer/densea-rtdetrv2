#!/usr/bin/env python
"""
Unified evaluation script for RT-DETR-V2, inspired by best practices
from the ONNX and TensorRT inference scripts.

Features:
- Multi-format support: PyTorch (.pth), ONNX (.onnx) and TensorRT (.engine).
- Optimized inference engines, especially for TensorRT with memory management
  and asynchronous execution via PyCUDA.
- Performance benchmarking (latency and FPS).
- Complete metrics: mAP (using pycocotools), confusion matrix,
  classification report and false positive/negative analysis.
- Generation of images with detections for visual review.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import colorsys
import json
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
from scipy.optimize import linear_sum_assignment

# Configure matplotlib to use a non-interactive backend BEFORE importing pyplot
# This avoids Tkinter errors in headless environments or secondary threads
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Imports for Inference and Evaluation ---

# COCO import and evaluation block
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_TOOLS_AVAILABLE = True
except ImportError:
    COCO_TOOLS_AVAILABLE = False

# Add the parent directory to import internal modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

# --- Inference Engine Classes (Wrappers) ---

class PyTorchEngine(nn.Module):
    """Wrapper for the original PyTorch model."""
    def __init__(self, cfg, device):
        super().__init__()
        checkpoint = torch.load(cfg.resume, map_location=device)
        state = checkpoint.get('ema', {}).get('module', checkpoint.get('model'))
        
        self.model = cfg.model
        self.model.load_state_dict(state)
        self.model = self.model.to(device).eval()
        
        self.postprocessor = cfg.postprocessor.deploy()
        self.postprocessor = self.postprocessor.to(device).eval()
        print("PyTorch engine loaded successfully.")

    def forward(self, images, orig_target_sizes):
        with torch.no_grad():
            outputs = self.model(images)
            labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
        return labels, boxes, scores

class ONNXEngine:
    """Wrapper for ONNX Runtime inference."""
    def __init__(self, model_path, device_str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime is not installed. Install it with: pip install onnxruntime-gpu")

        print(f"Loading ONNX model from: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device_str else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.device = torch.device(device_str)
        print(f"ONNX model loaded. Using provider: {self.session.get_providers()[0]}")

    def __call__(self, images, orig_target_sizes):
        inputs = {
            self.input_names[0]: images.cpu().numpy(),
            self.input_names[1]: orig_target_sizes.cpu().numpy()
        }
        outputs = self.session.run(self.output_names, inputs)
        
        labels = torch.from_numpy(outputs[0]).to(self.device)
        boxes = torch.from_numpy(outputs[1]).to(self.device)
        scores = torch.from_numpy(outputs[2]).to(self.device)
        return labels, boxes, scores

class TensorRTEngine:
    """
    Wrapper for TensorRT inference.
    Handles engine loading, buffer allocation and asynchronous execution.
    """
    def __init__(self, engine_path, device_str):
        try:
            import tensorrt as trt
            # Imported here so the script does not fail if only ONNX or PyTorch is used
            import pycuda.driver as cuda
            import pycuda.autoinit  # crucial for automatic context management
            self.cuda = cuda
        except ImportError as e:
            raise ImportError(f"Missing dependency for TensorRT: {e}. Check the installation.")

        print(f"Loading TensorRT engine from: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("Could not deserialize the TensorRT engine. Check the file and its permissions.")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.device = torch.device(device_str)
        print("TensorRT engine loaded successfully.")

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = self.cuda.Stream()
        
        for binding_name in self.engine:
            binding_idx = self.engine.get_binding_index(binding_name)
            shape = tuple(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            size = trt.volume(shape)
            
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            buffer_info = {'host': host_mem, 'device': device_mem, 'shape': shape, 'name': binding_name}
            if self.engine.binding_is_input(binding_name):
                inputs.append(buffer_info)
            else:
                outputs.append(buffer_info)
                
        return inputs, outputs, bindings, stream

    def __call__(self, images, orig_target_sizes):
        np.copyto(self.inputs[0]['host'], images.cpu().numpy().ravel())
        np.copyto(self.inputs[1]['host'], orig_target_sizes.cpu().numpy().ravel())

        self.cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        batch_size = images.shape[0]
        # The output shapes may vary; make sure they match your model
        # Assuming a standard output of [batch, 300] for labels/scores and [batch, 300, 4] for boxes
        num_dets = self.outputs[0]['shape'][-1] if len(self.outputs[0]['shape']) > 1 else self.outputs[0]['shape'][0]
        
        labels = torch.from_numpy(self.outputs[0]['host']).reshape(batch_size, num_dets)
        boxes = torch.from_numpy(self.outputs[1]['host']).reshape(batch_size, num_dets, 4)
        scores = torch.from_numpy(self.outputs[2]['host']).reshape(batch_size, num_dets)

        return labels.to(self.device), boxes.to(self.device), scores.to(self.device)

# --- Helper functions and main loop (unchanged) ---

def generate_color_map_from_label_map(label_map: dict) -> dict:
    keys = sorted(label_map.keys())
    n = len(keys)
    color_map = {}
    for i, key in enumerate(keys):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        color_hex = '#{0:02x}{1:02x}{2:02x}'.format(int(r*255), int(g*255), int(b*255))
        color_map[key] = color_hex
    return color_map

def get_label_map(dataset) -> dict:
    label_map = {}
    try:
        categories = dataset.coco.dataset.get('categories', [])
        for cat in categories:
            label_map[int(cat['id'])] = cat['name']
    except Exception as e:
        print(f"Could not get the label mapping from the COCO dataset: {e}. Using an empty map.")
    return label_map

def compute_label_offset(pred_labels_tensor: torch.Tensor, label_map: dict) -> int:
    if label_map and pred_labels_tensor.numel() > 0:
        raw_pred_min = int(pred_labels_tensor.min().item())
        min_label_in_map = min(label_map.keys())
        return min_label_in_map - raw_pred_min
    return 0

def detect_minority_classes(class_gt_total: dict, label_map: dict, threshold_ratio: float = 0.5) -> list:
    """
    Dynamically detect minority classes based on the real dataset distribution.

    Args:
        class_gt_total: Dict {cat_id: count} with the total GT per class
        label_map: Dict {cat_id: cat_name} to map IDs to names
        threshold_ratio: Ratio relative to the mean. Classes with count < mean * ratio are minority classes.
                        Default 0.5 means classes with less than 50% of the mean.

    Returns:
        List of minority class names (empty if there are none)
    """
    if not class_gt_total:
        return []

    counts = list(class_gt_total.values())
    if not counts:
        return []

    mean_count = sum(counts) / len(counts)
    threshold = mean_count * threshold_ratio

    minority_classes = []
    for cat_id, count in class_gt_total.items():
        if count < threshold and count > 0:  # Only if it has at least 1 instance
            cat_name = label_map.get(cat_id, f"ID:{cat_id}")
            minority_classes.append(cat_name)

    return minority_classes

def draw_inferences(image: Image.Image, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor,
                     label_map=None, color_map=None, threshold: float = 0.5,
                     label_offset: int = 0, save_path: str = None, radius: int = 2):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    valid_indices = scores > threshold
    if valid_indices.sum() == 0:
        if save_path: image.save(save_path)
        return image

    valid_labels = labels[valid_indices]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    padding = 2
    for j, box in enumerate(valid_boxes):
        x0, y0, x1, y1 = map(int, box)
        pred_id = valid_labels[j].item() + label_offset
        color = color_map.get(pred_id, "black") if color_map is not None else "black"
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=color, width=2)
        label_text = label_map.get(pred_id, f"ID:{pred_id}")
        text = f"{label_text} {valid_scores[j].item():.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = bbox[3] - bbox[1]
        text_width = bbox[2] - bbox[0]
        text_position = (x0 + padding, y0 - text_height - padding if y0 > text_height + padding else y0 + padding)
        draw.rectangle([text_position[0], text_position[1], text_position[0] + text_width, text_position[1] + text_height], fill=color)
        draw.text(text_position, text, font=font, fill="white")

    if save_path:
        image.save(save_path)
    return image

def draw_ground_truth(image: Image.Image, annotations: list, label_map=None, color_map=None, radius: int = 2):
    """
    Draw the ground truth annotations on the image.
    'annotations' is expected to be a list of dicts with at least 'bbox' and 'category_id'.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    padding = 2
    for ann in annotations:
        if "bbox" not in ann or "category_id" not in ann:
            continue
        box = ann["bbox"]
        x0, y0, x1, y1 = map(int, box)
        cat_id = int(ann["category_id"])
        color = color_map.get(cat_id, "red") if color_map is not None else "red"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if label_map is not None:
            label_text = label_map.get(cat_id, str(cat_id))
        else:
            label_text = str(cat_id)
        text = f"GT: {label_text}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if y0 - text_height - padding >= 0:
            text_position = (x0 + padding, y0 - text_height - padding)
        else:
            text_position = (x0 + padding, y1 + padding)
        draw.text(text_position, text, font=font, fill=color, stroke_width=1, stroke_fill="black")
    return image

def save_debug_comparison(original_image: Image.Image, pred_labels: torch.Tensor, pred_boxes: torch.Tensor,
                            pred_scores: torch.Tensor, gt_annotations: list, label_map, color_map,
                            threshold: float, label_offset: int, out_path: str):
    """
    Generate a side-by-side comparison image for debugging:
      - Left: image with the predictions of the model.
      - Right: image with the ground truth annotations drawn.
    Saves the comparison image to 'out_path'.
    """
    pred_img = original_image.copy()
    gt_img = original_image.copy()

    pred_img = draw_inferences(pred_img, pred_labels, pred_boxes, pred_scores,
                               label_map=label_map, color_map=color_map, threshold=threshold,
                               label_offset=label_offset)

    gt_img = draw_ground_truth(gt_img, gt_annotations, label_map=label_map, color_map=color_map)

    width, height = original_image.size
    combined = Image.new('RGB', (width*2, height))
    combined.paste(pred_img, (0,0))
    combined.paste(gt_img, (width,0))
    combined.save(out_path)

def plot_val_dataset_distribution(dataset, label_map, args):
    """
    Generate a bar chart with the number of instances per class
    in the validation dataset, showing the exact value above each bar.
    """
    print("\n--- Generating validation dataset distribution ---")

    # Count the annotations of each category
    cat_counts = Counter()
    ann_list = dataset.coco.dataset.get('annotations', [])
    for ann in ann_list:
        cat_counts[ann['category_id']] += 1

    # Sort by category ID
    cat_ids = sorted(cat_counts.keys())
    cat_names = [label_map.get(cat_id, str(cat_id)) for cat_id in cat_ids]
    counts = [cat_counts[cat_id] for cat_id in cat_ids]

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cat_names, counts, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Classes')
    plt.ylabel('Number of instances')
    plt.title('Instance distribution per class (Validation)')

    # Show the exact value above each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(counts),
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=9
        )

    dist_path = os.path.join(args.base_eval_dir, "dataset_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    print(f"Per-class instance distribution saved to: {dist_path}")

def plot_score_distributions(score_distribution, label_map, output_dir, detection_threshold=0.5, minority_classes=None):
    """
    Generate score distribution histograms per class.
    Useful to analyze whether the model produces low-score predictions that get filtered out.

    Args:
        minority_classes: List of minority class names (detected dynamically).
                         If None or empty, no class is highlighted.
    """
    if minority_classes is None:
        minority_classes = []

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
        is_minority = cat_name in minority_classes

        if scores:
            # Filter very low scores for better visualization
            scores_filtered = [s for s in scores if s > 0.01]

            if scores_filtered:
                ax.hist(scores_filtered, bins=50, range=(0, 1), color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.axvline(x=detection_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={detection_threshold}')
                ax.axvline(x=0.25, color='orange', linestyle='--', linewidth=1.5, label='Threshold=0.25')

                # Statistics
                above_threshold = sum(1 for s in scores if s > detection_threshold)
                above_025 = sum(1 for s in scores if s > 0.25)
                above_01 = sum(1 for s in scores if s > 0.1)
                total = len(scores)

                # Highlight minority classes (only if any was detected)
                title_color = 'red' if is_minority else 'black'
                ax.set_title(f"{cat_name}\n>{detection_threshold}: {above_threshold} | >0.25: {above_025} | >0.1: {above_01}\nTotal: {total}",
                            fontsize=9, color=title_color, fontweight='bold' if is_minority else 'normal')
            else:
                ax.set_title(f"{cat_name}\nNo predictions >0.01", fontsize=9)
        else:
            ax.set_title(f"{cat_name}\nNo predictions", fontsize=9)

        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.set_xlim(0, 1)

    # Hide empty axes
    for idx in range(len(sorted_classes), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Score distribution per class (evaluation threshold: {detection_threshold})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Generate text report
    report = "Score distribution per class\n" + "=" * 70 + "\n"
    report += f"{'Class':<20} {'Total':>10} {'>0.5':>10} {'>0.25':>10} {'>0.1':>10} {'Mean':>10}\n"
    report += "-" * 70 + "\n"

    for cat_id, scores in sorted_classes:
        cat_name = label_map.get(cat_id, str(cat_id))
        if scores:
            above_05 = sum(1 for s in scores if s > 0.5)
            above_025 = sum(1 for s in scores if s > 0.25)
            above_01 = sum(1 for s in scores if s > 0.1)
            mean_score = np.mean(scores)
            report += f"{cat_name:<20} {len(scores):>10} {above_05:>10} {above_025:>10} {above_01:>10} {mean_score:>10.4f}\n"
        else:
            report += f"{cat_name:<20} {'0':>10} {'0':>10} {'0':>10} {'0':>10} {'N/A':>10}\n"

    report_path = os.path.join(output_dir, "score_distribution_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Score distribution saved to: {output_dir}/score_distributions.png")
    print(f"Score report saved to: {report_path}")

def iou(boxA, boxB):
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
    """
    Match detections to ground truth using the Hungarian Algorithm (optimal assignment).

    This method guarantees the globally optimal matching by maximizing the total IoU.
    It is the standard method used in DETR and modern end-to-end detectors.

    Args:
        gt_boxes: List of GT bounding boxes [x1, y1, x2, y2]
        gt_labels: List of GT labels
        pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        pred_labels: List of predicted labels
        pred_scores: List of confidence scores (unused in Hungarian, kept for compatibility)
        iou_thresh: Minimum IoU to consider a match valid

    Returns:
        matched_pairs: List of (gt_label, pred_label, iou_value) tuples for matched detections
        unmatched_gts: List of unmatched GT labels (false negatives)
        unmatched_preds: List of unmatched predicted labels (false positives)
    """
    if len(pred_boxes) == 0:
        unmatched_gts = list(gt_labels)
        return [], unmatched_gts, []

    if len(gt_boxes) == 0:
        unmatched_preds = list(pred_labels)
        return [], [], unmatched_preds

    # 1. Build the IoU matrix (pred_boxes x gt_boxes)
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = iou(pred_box, gt_box)

    # 2. Convert to a cost matrix (Hungarian minimizes, so use -IoU)
    cost_matrix = -iou_matrix

    # 3. Apply the Hungarian Algorithm to find the optimal matching
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # 4. Filter out matches below the IoU threshold
    matched_pairs = []
    matched_pred_indices = set()
    matched_gt_indices = set()

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_thresh:
            # Also include the IoU of the match
            matched_pairs.append((gt_labels[gt_idx], pred_labels[pred_idx], iou_matrix[pred_idx, gt_idx]))
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)

    # 5. Identify unmatched predictions and GT
    unmatched_preds = [pred_labels[i] for i in range(num_preds) if i not in matched_pred_indices]
    unmatched_gts = [gt_labels[j] for j in range(num_gts) if j not in matched_gt_indices]

    return matched_pairs, unmatched_gts, unmatched_preds


def match_detections_greedy(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """
    Match detections to ground truth using a score-based Greedy algorithm.

    This method sorts the predictions by score (highest to lowest) and assigns each one to the
    best available GT. It is faster than Hungarian but can produce suboptimal matchings.

    Useful if you want the evaluation to reflect a more "realistic" behavior where
    higher-confidence predictions take priority.

    Args:
        gt_boxes: List of GT bounding boxes [x1, y1, x2, y2]
        gt_labels: List of GT labels
        pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2]
        pred_labels: List of predicted labels
        pred_scores: List of confidence scores (USED for sorting)
        iou_thresh: Minimum IoU to consider a match valid

    Returns:
        matched_pairs: List of (gt_label, pred_label, iou_value) tuples for matched detections
        unmatched_gts: List of unmatched GT labels (false negatives)
        unmatched_preds: List of unmatched predicted labels (false positives)
    """
    if len(pred_boxes) == 0:
        unmatched_gts = list(gt_labels)
        return [], unmatched_gts, []

    if len(gt_boxes) == 0:
        unmatched_preds = list(pred_labels)
        return [], [], unmatched_preds

    # Sort predictions by score (highest to lowest)
    preds = sorted(zip(pred_boxes, pred_labels, pred_scores), key=lambda x: x[2], reverse=True)
    matched_pairs = []
    unmatched_preds = []
    assigned_gt = set()

    for p_box, p_label, p_score in preds:
        best_iou = -1
        best_gt_idx = -1

        # Find the best available GT for this prediction
        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in assigned_gt:
                continue
            current_iou = iou(p_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = idx

        # If a match by IoU is found
        if best_iou >= iou_thresh and best_gt_idx != -1:
            assigned_gt.add(best_gt_idx)
            # Store (gt_label, pred_label, iou) even if they differ (for the confusion matrix)
            matched_pairs.append((gt_labels[best_gt_idx], p_label, best_iou))
        else:
            unmatched_preds.append(p_label)

    # Unmatched ground truth = False Negatives
    unmatched_gts = [gt_labels[i] for i in range(len(gt_labels)) if i not in assigned_gt]
    return matched_pairs, unmatched_gts, unmatched_preds


def match_detections_in_image(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5, method='hungarian'):
    """
    Wrapper that calls the appropriate matching method.

    Args:
        method: 'hungarian' (optimal) or 'greedy' (score-based)
    """
    if method == 'hungarian':
        return match_detections_hungarian(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh)
    elif method == 'greedy':
        return match_detections_greedy(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh)
    else:
        raise ValueError(f"Unknown matching method: {method}. Use 'hungarian' or 'greedy'.")

def evaluate_on_dataset(args, cfg, device):
    if args.onnx:
        engine = ONNXEngine(args.onnx, args.device)
    elif args.tensorrt:
        engine = TensorRTEngine(args.tensorrt, args.device)
    else:
        cfg.resume = args.resume
        engine = PyTorchEngine(cfg, device)

    dataloader = cfg.val_dataloader
    dataset = dataloader.dataset
    label_map = get_label_map(dataset)
    color_map = generate_color_map_from_label_map(label_map)
    os.makedirs(args.base_eval_dir, exist_ok=True)

    # Report the evaluation configuration
    print("\n" + "=" * 70)
    print("EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"  Matching method:          {args.matching_method.upper()}")
    print(f"  IoU threshold:            {args.iou_threshold}")
    print(f"  Detection threshold:      {args.detection_threshold}")
    if args.matching_method == 'hungarian':
        print("  Note: Hungarian matching is the standard for DETR/end-to-end detectors")
    else:
        print("  Note: Greedy matching prioritizes higher-confidence detections")
    print("=" * 70 + "\n")

    # Generate the dataset distribution
    plot_val_dataset_distribution(dataset, label_map, args)

    all_y_true, all_y_pred = [], []
    coco_predictions = []
    latencies = []

    # Per-class counters for correct metrics
    # TP: GT matched with a prediction of the SAME class
    # FP: unmatched prediction OR matched with a GT of a DIFFERENT class
    # FN: unmatched GT OR matched with a prediction of a DIFFERENT class
    class_tp = Counter()  # True Positives per class
    class_fp = Counter()  # False Positives per class
    class_fn = Counter()  # False Negatives per class
    class_gt_total = Counter()  # Total GT per class

    # Diagnostic counters
    total_matched = 0
    total_fn = 0
    total_fp = 0
    total_gt = 0
    total_pred = 0

    # Additional localization and error metrics
    all_match_ious = []  # IoU of all matches (for Mean IoU)
    correct_class_matches = 0  # Matches with the correct class (localization + classification)
    wrong_class_matches = 0  # Matches with the wrong class (localization OK, classification FAIL)
    background_fps = 0  # Predictions without any match (phantom detections)

    # Score distribution per class (for threshold analysis)
    score_distribution = {cat_id: [] for cat_id in label_map.keys()}

    for batch in tqdm(dataloader, desc="Evaluating batches"):
        images, targets = batch
        images = images.to(device)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)

        start_time = time.perf_counter()
        labels_batch, boxes_batch, scores_batch = engine(images, orig_target_sizes)
        latencies.append(time.perf_counter() - start_time)

        for i in range(len(images)):
            target = targets[i]
            image_id = target['image_id'].item()

            # After the transforms, the annotations are in 'boxes' and 'labels'
            gt_boxes_tensor = target.get('boxes', torch.empty(0, 4))
            gt_labels_tensor = target.get('labels', torch.empty(0, dtype=torch.int64))

            # IMPORTANT: the GT boxes come from the dataloader in NORMALIZED CXCYWH format
            # (after the ConvertBoxes transform with fmt='cxcywh', normalize=True).
            # They must be converted to XYXY in original image coordinates
            # to compare them with the postprocessor predictions.
            # NOTE: orig_size is [W, H] per the postprocessor convention (see box_revert.py)
            orig_size = target.get('orig_size', torch.tensor([images[i].shape[2], images[i].shape[1]]))  # Default: [W, H]
            orig_w, orig_h = orig_size[0].item(), orig_size[1].item()  # orig_size is [W, H]

            # Convert GT boxes from normalized CXCYWH to XYXY in original pixels
            if len(gt_boxes_tensor) > 0:
                # gt_boxes_tensor is in normalized [cx, cy, w, h] format (0-1)
                cx = gt_boxes_tensor[:, 0]
                cy = gt_boxes_tensor[:, 1]
                w = gt_boxes_tensor[:, 2]
                h = gt_boxes_tensor[:, 3]

                # Convert to XYXY in original image coordinates
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                gt_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                gt_boxes = gt_boxes_xyxy.tolist()
            else:
                gt_boxes = []

            gt_labels = gt_labels_tensor.tolist() if len(gt_labels_tensor) > 0 else []

            raw_pred_labels = labels_batch[i].cpu()
            pred_boxes = boxes_batch[i].cpu()
            pred_scores = scores_batch[i].cpu()

            label_offset = compute_label_offset(raw_pred_labels, label_map)

            # Count ground truth
            total_gt += len(gt_labels)

            # Collect the score distribution per class (BEFORE filtering)
            for j in range(len(raw_pred_labels)):
                pred_class = raw_pred_labels[j].item() + label_offset
                if pred_class in score_distribution:
                    score_distribution[pred_class].append(pred_scores[j].item())

            if args.coco_eval and COCO_TOOLS_AVAILABLE:
                for j in range(len(raw_pred_labels)):
                    if pred_scores[j] > 0.001:
                        box = pred_boxes[j].tolist()
                        coco_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                        coco_predictions.append({
                            'image_id': image_id,
                            'category_id': raw_pred_labels[j].item() + label_offset,
                            'bbox': coco_box,
                            'score': pred_scores[j].item()
                        })

            filtered_indices = pred_scores > args.detection_threshold
            filtered_pred_boxes = pred_boxes[filtered_indices].tolist()
            filtered_pred_labels = (raw_pred_labels[filtered_indices] + label_offset).tolist()
            filtered_pred_scores = pred_scores[filtered_indices].tolist()

            # Count filtered predictions
            total_pred += len(filtered_pred_labels)

            matched, unmatched_gts, unmatched_preds = match_detections_in_image(
                gt_boxes, gt_labels, filtered_pred_boxes, filtered_pred_labels, filtered_pred_scores,
                args.iou_threshold, method=args.matching_method)

            # Update diagnostic counters
            total_matched += len(matched)
            total_fn += len(unmatched_gts)
            total_fp += len(unmatched_preds)

            # Count total GT per class
            for gt_lab in gt_labels:
                class_gt_total[gt_lab] += 1

            # Process matches (now including IoU)
            for match_tuple in matched:
                gt_lab, pred_lab, match_iou = match_tuple
                all_y_true.append(gt_lab)
                all_y_pred.append(pred_lab)
                all_match_ious.append(match_iou)

                if gt_lab == pred_lab:
                    # Correct match: TP for that class
                    class_tp[gt_lab] += 1
                    correct_class_matches += 1
                else:
                    # Wrong match: FN for the GT, FP for the prediction
                    class_fn[gt_lab] += 1
                    class_fp[pred_lab] += 1
                    wrong_class_matches += 1

            # Unmatched GT = FN
            for gt_lab in unmatched_gts:
                class_fn[gt_lab] += 1

            # Unmatched predictions = FP (phantom/background detections)
            for pred_lab in unmatched_preds:
                class_fp[pred_lab] += 1
                background_fps += 1

            # Debug mode: generate comparison images if prediction and ground truth differ
            if args.debug:
                num_gt = len(gt_labels)
                num_pred = (pred_scores > args.detection_threshold).sum().item()
                if num_pred != num_gt:
                    # Build the correct image path
                    img_info = dataset.coco.loadImgs(image_id)[0]
                    img_path = img_info['file_name']

                    # If the path does not exist, look the file up by name in the dataset directories
                    if not os.path.exists(img_path):
                        # Support both CocoDetection (img_folder) and MultiDirCocoDetection (image_dirs)
                        if hasattr(dataset, 'image_dirs'):
                            img_dirs = dataset.image_dirs if isinstance(dataset.image_dirs, list) else [dataset.image_dirs]
                        elif hasattr(dataset, 'img_folder'):
                            img_dirs = [dataset.img_folder]
                        else:
                            img_dirs = []
                        img_basename = os.path.basename(img_path)
                        found = False
                        for img_dir in img_dirs:
                            full_path = os.path.join(img_dir, img_basename)
                            if os.path.exists(full_path):
                                img_path = full_path
                                found = True
                                break
                        if not found:
                            print(f"Warning: image {img_basename} not found in any configured directory. Skipping debug for image_id={image_id}")
                            continue

                    original_image = Image.open(img_path).convert("RGB")
                    debug_out = os.path.join(args.base_eval_dir, "debug_images", f"debug_{image_id}.jpg")

                    # Convert gt_boxes and gt_labels to annotation format for draw_ground_truth
                    gt_annotations_dict = [{'bbox': box, 'category_id': label} for box, label in zip(gt_boxes, gt_labels)]

                    save_debug_comparison(original_image, raw_pred_labels, pred_boxes, pred_scores,
                                        gt_annotations_dict, label_map, color_map, args.detection_threshold,
                                        label_offset, debug_out)

            if args.draw:
                # Build the correct image path
                img_info = dataset.coco.loadImgs(image_id)[0]
                img_path = img_info['file_name']

                # If the path does not exist, look the file up by name in the dataset directories
                if not os.path.exists(img_path):
                    # Support both CocoDetection (img_folder) and MultiDirCocoDetection (image_dirs)
                    if hasattr(dataset, 'image_dirs'):
                        img_dirs = dataset.image_dirs if isinstance(dataset.image_dirs, list) else [dataset.image_dirs]
                    elif hasattr(dataset, 'img_folder'):
                        img_dirs = [dataset.img_folder]
                    else:
                        img_dirs = []
                    img_basename = os.path.basename(img_path)
                    found = False
                    for img_dir in img_dirs:
                        full_path = os.path.join(img_dir, img_basename)
                        if os.path.exists(full_path):
                            img_path = full_path
                            found = True
                            break
                    if not found:
                        print(f"Warning: image {img_basename} not found in any configured directory. Skipping draw for image_id={image_id}")
                        continue

                original_image = Image.open(img_path).convert("RGB")
                draw_inferences(original_image, raw_pred_labels, pred_boxes, pred_scores,
                                label_map, color_map, args.detection_threshold, label_offset,
                                save_path=os.path.join(args.base_eval_dir, "predicted_images", f"{image_id}_pred.jpg"))
                
    print("\n--- Performance Results ---")
    if len(latencies) > 1:
        avg_latency_ms = np.mean(latencies[1:]) * 1000
        avg_fps = 1.0 / np.mean(latencies[1:])
        print(f"Average latency: {avg_latency_ms:.2f} ms")
        print(f"Throughput (FPS): {avg_fps:.2f} FPS")
    else:
        print("Not enough measurements to report latency.")

    # --- Detect minority classes dynamically ---
    minority_classes = detect_minority_classes(class_gt_total, label_map, threshold_ratio=0.5)
    if minority_classes:
        print(f"\n--- Minority classes detected (< 50% of the mean): {', '.join(minority_classes)} ---")

    # --- Score Distribution per Class ---
    print("\n--- Generating per-class score distribution ---")
    plot_score_distributions(score_distribution, label_map, args.base_eval_dir, args.detection_threshold, minority_classes)

    print("\n--- Classification Metrics (IoU-based) ---")

    if label_map and any(label_map):
        class_names = [label_map.get(i, str(i)) for i in sorted(label_map.keys())]
        labels_to_report = sorted(list(set(all_y_true + all_y_pred)))

        # Create consistent names for target_names
        target_names = []
        for l in labels_to_report:
            target_names.append(label_map.get(l, f"Class_{l}"))

        try:
            report = classification_report(all_y_true, all_y_pred, labels=labels_to_report,
                                           target_names=target_names,
                                           zero_division=0, digits=4)
            print("\n" + "=" * 70)
            print("NOTE: This sklearn report ONLY considers matched detections.")
            print("It does NOT include undetected objects (FN) or phantom detections (FP).")
            print("For complete detection metrics, see 'PER-CLASS ANALYSIS' below.")
            print("=" * 70)
            print(report)

            # Save the classification report as a PNG image
            plt.figure(figsize=(10, 8))
            plt.text(0.01, 0.05, report, {'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.title("Classification Report (matched detections only, by IoU)\nNOTE: does NOT include undetected objects")
            report_path = os.path.join(args.base_eval_dir, "classification_report.png")
            plt.tight_layout()
            plt.savefig(report_path)
            plt.close()
            print(f"Classification report saved to: {report_path}")
        except Exception as e:
            print(f"Could not generate the classification report: {e}")

        # =====================================================================
        # COMPLETE DETECTION REPORT (includes undetected objects)
        # =====================================================================
        print("\n" + "=" * 70)
        print("COMPLETE DETECTION REPORT")
        print("(Includes undetected objects and phantom detections)")
        print("=" * 70)

        # Build the complete per-class report
        complete_report = "\n{:<20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}\n".format(
            "Class", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1-Score")
        complete_report += "-" * 94 + "\n"

        all_supports = []
        all_precisions_complete = []
        all_recalls_complete = []
        all_f1s_complete = []

        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            tp = class_tp.get(cat_id, 0)
            fp = class_fp.get(cat_id, 0)
            fn = class_fn.get(cat_id, 0)
            support = class_gt_total.get(cat_id, 0)  # Total GT of this class

            # Compute metrics using the correct formulas
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

            complete_report += "{:<20} {:>8} {:>8} {:>8} {:>8} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
                cat_name, support, tp, fp, fn, precision, recall, f1)

            all_supports.append(support)
            all_precisions_complete.append(precision)
            all_recalls_complete.append(recall)
            all_f1s_complete.append(f1)

        complete_report += "-" * 94 + "\n"

        # Compute averages
        total_support = sum(all_supports)

        # Macro average (equal weight per class)
        macro_prec = np.mean(all_precisions_complete)
        macro_rec = np.mean(all_recalls_complete)
        macro_f1_val = np.mean(all_f1s_complete)

        # Weighted average (weighted by support)
        if total_support > 0:
            weighted_prec = sum(p * s for p, s in zip(all_precisions_complete, all_supports)) / total_support
            weighted_rec = sum(r * s for r, s in zip(all_recalls_complete, all_supports)) / total_support
            weighted_f1_val = sum(f * s for f, s in zip(all_f1s_complete, all_supports)) / total_support
        else:
            weighted_prec = weighted_rec = weighted_f1_val = 0.0

        complete_report += "\n{:<20} {:>8} {:>8} {:>8} {:>8} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
            "macro avg", "", "", "", "", macro_prec, macro_rec, macro_f1_val)
        complete_report += "{:<20} {:>8} {:>8} {:>8} {:>8} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
            "weighted avg", total_support, sum(class_tp.values()), sum(class_fp.values()),
            sum(class_fn.values()), weighted_prec, weighted_rec, weighted_f1_val)

        print(complete_report)

        # Save the complete report
        complete_report_path = os.path.join(args.base_eval_dir, "complete_detection_report.txt")
        with open(complete_report_path, 'w', encoding='utf-8') as f:
            f.write("COMPLETE DETECTION REPORT\n")
            f.write("(Includes undetected objects and phantom detections)\n")
            f.write("=" * 94 + "\n")
            f.write("\nFormulas used:\n")
            f.write("  - Precision = TP / (TP + FP)\n")
            f.write("  - Recall = TP / (TP + FN)\n")
            f.write("  - F1 = 2 * (Precision * Recall) / (Precision + Recall)\n")
            f.write("\nWhere:\n")
            f.write("  - TP: Correct detection (IoU >= threshold AND correct class)\n")
            f.write("  - FP: Incorrect detection (no GT match) OR wrong class\n")
            f.write("  - FN: Undetected object OR detected with the wrong class\n")
            f.write("  - Support: Total number of GT objects of that class\n")
            f.write("=" * 94 + "\n")
            f.write(complete_report)
        print(f"Complete detection report saved to: {complete_report_path}")

        # Generate the confusion matrix with the correct names
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_to_report)

        # Compute a dynamic size based on the number of classes
        num_classes = len(labels_to_report)
        fig_size = max(10, num_classes * 0.8)

        # Confusion matrix with absolute values
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names,
                    cbar_kws={'label': 'Count'})
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Ground Truth", fontsize=12)
        plt.title("Confusion Matrix (Absolute Values)\n[Only detections matched by IoU]", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        cm_path = os.path.join(args.base_eval_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")

        # NORMALIZED confusion matrix (per row = per GT class)
        # Important for imbalanced datasets: shows proportions instead of absolute counts
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names,
                    cbar_kws={'label': 'Proportion'},
                    vmin=0, vmax=1)
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Ground Truth", fontsize=12)
        plt.title("Normalized Confusion Matrix (per row)\n[Shows the correct classification rate per class]", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        cm_norm_path = os.path.join(args.base_eval_dir, 'confusion_matrix_normalized.png')
        plt.tight_layout()
        plt.savefig(cm_norm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Normalized confusion matrix saved to: {cm_norm_path}")

        # Generate the False Positives and False Negatives report and chart
        print("\n--- Generating False Positives and False Negatives report ---")
        fp_fn_report = "False Positives and False Negatives Report\n"
        fp_fn_report += "=" * 70 + "\n"
        fp_fn_report += f"{'Category':<30} {'False Positives':>15} {'False Negatives':>15}\n"
        fp_fn_report += "-" * 70 + "\n"
        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            fp = class_fp.get(cat_id, 0)
            fn = class_fn.get(cat_id, 0)
            fp_fn_report += f"{cat_name} ({cat_id}){'':<20} {fp:>15} {fn:>15}\n"

        fp_fn_report_path = os.path.join(args.base_eval_dir, "fp_fn_report.txt")
        with open(fp_fn_report_path, "w") as f:
            f.write(fp_fn_report)
        print(f"FP/FN report saved to: {fp_fn_report_path}")

        # Plot the FP and FN counters per class
        classes = [label_map[k] for k in sorted(label_map.keys())]
        fp_counts = [class_fp.get(k, 0) for k in sorted(label_map.keys())]
        fn_counts = [class_fn.get(k, 0) for k in sorted(label_map.keys())]

        x = np.arange(len(classes))
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, fp_counts, width, label='False Positives', color='salmon')
        plt.bar(x + width/2, fn_counts, width, label='False Negatives', color='lightblue')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.xlabel("Classes")
        plt.ylabel("Count")
        plt.title("False Positives and False Negatives per Class")
        plt.legend()
        fp_fn_bar_path = os.path.join(args.base_eval_dir, "fp_fn_bar_chart.png")
        plt.tight_layout()
        plt.savefig(fp_fn_bar_path)
        plt.close()
        print(f"FP/FN chart saved to: {fp_fn_bar_path}")

    if args.coco_eval and COCO_TOOLS_AVAILABLE:
        print("\n--- COCO mAP Evaluation ---")
        if not coco_predictions:
            print("No predictions were generated for the COCO evaluation.")
        else:
            pred_json_path = os.path.join(args.base_eval_dir, "coco_predictions.json")
            with open(pred_json_path, 'w') as f:
                json.dump(coco_predictions, f)

            coco_dt = dataset.coco.loadRes(pred_json_path)
            coco_eval = COCOeval(dataset.coco, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Save the COCO results to a text file
            report_path = os.path.join(args.base_eval_dir, "coco_evaluation_report.txt")
            original_stdout = sys.stdout
            with open(report_path, 'w') as f:
                sys.stdout = f
                coco_eval.summarize()
            sys.stdout = original_stdout
            print(f"COCO evaluation report saved to: {report_path}")

            # --- Per-class AP (COCO) ---
            print("\n--- Per-class AP (COCO) ---")
            precisions = coco_eval.eval['precision']
            # precision has shape: [TxRxKxAxM]
            # T: IoU thresholds, R: recall thresholds, K: categories, A: areas, M: maxDets

            cat_ids = dataset.coco.getCatIds()
            cat_names_list = [dataset.coco.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]

            per_class_ap = {}
            per_class_ap50 = {}
            per_class_ap75 = {}
            per_class_report = "Per-class AP (COCO Metrics)\n" + "=" * 60 + "\n"
            per_class_report += f"{'Class':<20} {'AP@0.5:0.95':>12} {'AP@0.50':>12} {'AP@0.75':>12}\n"
            per_class_report += "-" * 60 + "\n"

            for idx, cat_id in enumerate(cat_ids):
                cat_name = cat_names_list[idx]

                # AP@0.50:0.95 (average over all IoU thresholds)
                precision_all = precisions[:, :, idx, 0, -1]
                precision_all = precision_all[precision_all > -1]
                ap = np.mean(precision_all) if precision_all.size else 0.0
                per_class_ap[cat_name] = ap

                # AP@0.50 (IoU threshold index 0)
                precision_50 = precisions[0, :, idx, 0, -1]
                precision_50 = precision_50[precision_50 > -1]
                ap50 = np.mean(precision_50) if precision_50.size else 0.0
                per_class_ap50[cat_name] = ap50

                # AP@0.75 (IoU threshold index 5)
                precision_75 = precisions[5, :, idx, 0, -1]
                precision_75 = precision_75[precision_75 > -1]
                ap75 = np.mean(precision_75) if precision_75.size else 0.0
                per_class_ap75[cat_name] = ap75

                per_class_report += f"{cat_name:<20} {ap:>12.4f} {ap50:>12.4f} {ap75:>12.4f}\n"

            print(per_class_report)

            # Save the per-class report
            per_class_path = os.path.join(args.base_eval_dir, "per_class_ap_report.txt")
            with open(per_class_path, 'w') as f:
                f.write(per_class_report)
            print(f"Per-class AP report saved to: {per_class_path}")

            # Per-class AP chart
            plt.figure(figsize=(14, 6))
            x = np.arange(len(cat_names_list))
            width = 0.25

            # Colors: red for minority classes (detected dynamically), blue for the rest
            colors_ap = ['#d62728' if name in minority_classes else '#1f77b4' for name in cat_names_list]

            bars1 = plt.bar(x - width, [per_class_ap[n] for n in cat_names_list], width, label='AP@0.5:0.95', color=colors_ap, alpha=0.9)
            bars2 = plt.bar(x, [per_class_ap50[n] for n in cat_names_list], width, label='AP@0.50', color='#2ca02c', alpha=0.7)
            bars3 = plt.bar(x + width, [per_class_ap75[n] for n in cat_names_list], width, label='AP@0.75', color='#ff7f0e', alpha=0.7)

            plt.xticks(x, cat_names_list, rotation=45, ha='right')
            plt.ylabel('Average Precision (AP)')
            plt.title('Per-class AP (COCO Metrics)')
            plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Target: 0.5')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(args.base_eval_dir, "per_class_ap.png"), dpi=150)
            plt.close()
            print(f"Per-class AP chart saved to: {args.base_eval_dir}/per_class_ap.png")

            # --- AP by Object Size (COCO) ---
            # Important to understand whether the model fails on small objects
            print("\n--- AP by Object Size (COCO) ---")
            # coco_eval.stats contains: [AP, AP50, AP75, AP_small, AP_medium, AP_large, AR1, AR10, AR100, AR_small, AR_medium, AR_large]
            stats = coco_eval.stats
            size_report = "AP and AR by Object Size\n" + "=" * 60 + "\n"
            size_report += f"{'Metric':<30} {'Value':>12}\n"
            size_report += "-" * 60 + "\n"
            size_report += f"{'AP (IoU=0.50:0.95)':<30} {stats[0]:>12.4f}\n"
            size_report += f"{'AP (IoU=0.50)':<30} {stats[1]:>12.4f}\n"
            size_report += f"{'AP (IoU=0.75)':<30} {stats[2]:>12.4f}\n"
            size_report += f"{'AP (small: area < 32²)':<30} {stats[3]:>12.4f}\n"
            size_report += f"{'AP (medium: 32² < area < 96²)':<30} {stats[4]:>12.4f}\n"
            size_report += f"{'AP (large: area > 96²)':<30} {stats[5]:>12.4f}\n"
            size_report += "-" * 60 + "\n"
            size_report += f"{'AR (maxDets=1)':<30} {stats[6]:>12.4f}\n"
            size_report += f"{'AR (maxDets=10)':<30} {stats[7]:>12.4f}\n"
            size_report += f"{'AR (maxDets=100)':<30} {stats[8]:>12.4f}\n"
            size_report += f"{'AR (small: area < 32²)':<30} {stats[9]:>12.4f}\n"
            size_report += f"{'AR (medium: 32² < area < 96²)':<30} {stats[10]:>12.4f}\n"
            size_report += f"{'AR (large: area > 96²)':<30} {stats[11]:>12.4f}\n"

            print(size_report)

            # Save the size report
            size_report_path = os.path.join(args.base_eval_dir, "ap_by_size_report.txt")
            with open(size_report_path, 'w') as f:
                f.write(size_report)
            print(f"AP-by-size report saved to: {size_report_path}")

            # AP/AR by size chart
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # AP by size
            sizes = ['Small\n(<32²)', 'Medium\n(32²-96²)', 'Large\n(>96²)']
            ap_by_size = [stats[3], stats[4], stats[5]]
            colors_size = ['#e74c3c' if v < 0.3 else '#f39c12' if v < 0.5 else '#27ae60' for v in ap_by_size]
            axes[0].bar(sizes, ap_by_size, color=colors_size, edgecolor='black')
            axes[0].set_ylabel('Average Precision (AP)')
            axes[0].set_title('AP by Object Size')
            axes[0].set_ylim(0, 1)
            axes[0].axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Target: 0.5')
            for i, v in enumerate(ap_by_size):
                axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

            # AR by size
            ar_by_size = [stats[9], stats[10], stats[11]]
            colors_ar = ['#e74c3c' if v < 0.3 else '#f39c12' if v < 0.5 else '#27ae60' for v in ar_by_size]
            axes[1].bar(sizes, ar_by_size, color=colors_ar, edgecolor='black')
            axes[1].set_ylabel('Average Recall (AR)')
            axes[1].set_title('AR by Object Size')
            axes[1].set_ylim(0, 1)
            axes[1].axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Target: 0.5')
            for i, v in enumerate(ar_by_size):
                axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(args.base_eval_dir, "ap_ar_by_size.png"), dpi=150)
            plt.close()
            print(f"AP/AR-by-size chart saved to: {args.base_eval_dir}/ap_ar_by_size.png")

            # Size diagnostics
            print("\n  Size Diagnostics:")
            if stats[3] < 0.2:
                print("  WARNING: AP_small is very low (<0.2). The model struggles with small objects.")
                print("          Consider: smaller anchor sizes, higher resolution, or zoom augmentation.")
            if stats[3] < stats[5] * 0.5:
                print(f"  WARNING: AP_small ({stats[3]:.3f}) << AP_large ({stats[5]:.3f})")
                print("          The model favors large objects. Review the training scale.")

    # --- DIAGNOSTIC SUMMARY ---
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # Compute the aggregated metrics correctly
    total_tp = sum(class_tp.values())
    total_fp_sum = sum(class_fp.values())
    total_fn_sum = sum(class_fn.values())
    total_gt_sum = sum(class_gt_total.values())

    # Global statistics
    print(f"\n{'GLOBAL STATISTICS':^70}")
    print("-" * 70)
    print(f"  Total Ground Truth:       {total_gt_sum:>8}")
    print(f"  Total predictions:       {total_pred:>8}")
    print(f"  True Positives (TP):      {total_tp:>8}")
    print(f"  False Negatives (FN):     {total_fn_sum:>8}  ({100*total_fn_sum/max(total_gt_sum,1):.1f}% of GT)")
    print(f"  False Positives (FP):     {total_fp_sum:>8}  ({100*total_fp_sum/max(total_pred,1):.1f}% of pred)")

    # Correct global metrics (micro-average)
    # Micro: sum all TP, FP, FN and compute global metrics
    micro_precision = total_tp / max(total_tp + total_fp_sum, 1)
    micro_recall = total_tp / max(total_tp + total_fn_sum, 1)
    micro_f1 = 2 * (micro_precision * micro_recall) / max(micro_precision + micro_recall, 1e-10)

    print(f"\n  {'MICRO-AVERAGE (weighted by instance):':^50}")
    print(f"  Precision (Micro):        {micro_precision:>8.2%}")
    print(f"  Recall (Micro):           {micro_recall:>8.2%}")
    print(f"  F1-Score (Micro):         {micro_f1:>8.4f}")

    # Macro-average: simple average of per-class metrics (equal weight per class)
    # Important for imbalanced datasets: gives equal importance to minority classes
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_weights = []  # For the weighted average

    for cat_id in sorted(label_map.keys()):
        tp = class_tp.get(cat_id, 0)
        fn = class_fn.get(cat_id, 0)
        fp = class_fp.get(cat_id, 0)
        gt_total = class_gt_total.get(cat_id, 0)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        class_weights.append(gt_total)

    # Macro: simple average (each class counts equally)
    macro_precision = np.mean(class_precisions)
    macro_recall = np.mean(class_recalls)
    macro_f1 = np.mean(class_f1s)

    print(f"\n  {'MACRO-AVERAGE (equal weight per class):':^50}")
    print(f"  Precision (Macro):        {macro_precision:>8.2%}")
    print(f"  Recall (Macro):           {macro_recall:>8.2%}")
    print(f"  F1-Score (Macro):         {macro_f1:>8.4f}")

    # Weighted: weighted by the number of GT of each class
    total_weight = sum(class_weights)
    if total_weight > 0:
        weighted_precision = sum(p * w for p, w in zip(class_precisions, class_weights)) / total_weight
        weighted_recall = sum(r * w for r, w in zip(class_recalls, class_weights)) / total_weight
        weighted_f1 = sum(f * w for f, w in zip(class_f1s, class_weights)) / total_weight
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    print(f"\n  {'WEIGHTED-AVERAGE (weighted by GT):':^50}")
    print(f"  Precision (Weighted):     {weighted_precision:>8.2%}")
    print(f"  Recall (Weighted):        {weighted_recall:>8.2%}")
    print(f"  F1-Score (Weighted):      {weighted_f1:>8.4f}")

    # G-Mean: geometric mean of recalls (very sensitive to poorly performing classes)
    # Best metric for imbalanced datasets
    if class_recalls and all(r > 0 for r in class_recalls):
        g_mean = np.exp(np.mean(np.log(class_recalls)))
    else:
        # If any recall is 0, G-Mean is 0
        g_mean = 0.0

    print(f"\n  {'G-MEAN (sensitive to minority classes):':^50}")
    print(f"  G-Mean:                   {g_mean:>8.4f}")
    if g_mean < macro_f1 * 0.8:
        print("  WARNING: G-Mean << Macro-F1 indicates a severe problem in some class")

    # === LOCALIZATION METRICS ===
    print(f"\n  {'LOCALIZATION METRICS:':^50}")
    print("-" * 70)

    # Localization Recall: % of GT with an IoU match (class-independent)
    localization_recall = total_matched / max(total_gt_sum, 1)
    print(f"  Localization Recall:      {localization_recall:>8.2%}  (GT with an IoU match)")

    # Mean IoU of the matches
    if all_match_ious:
        mean_iou = np.mean(all_match_ious)
        std_iou = np.std(all_match_ious)
        min_iou = np.min(all_match_ious)
        max_iou = np.max(all_match_ious)
        print(f"  Mean IoU (matches):       {mean_iou:>8.4f}  (std: {std_iou:.4f})")
        print(f"  IoU Range:                [{min_iou:.3f} - {max_iou:.3f}]")
    else:
        mean_iou = 0.0
        print(f"  Mean IoU (matches):       N/A (no matches)")

    # === ERROR BREAKDOWN ===
    print(f"\n  {'ERROR BREAKDOWN:':^50}")
    print("-" * 70)

    # Background FP Rate: predictions that do not match any GT
    background_fp_rate = background_fps / max(total_pred, 1)
    print(f"  Background FP:            {background_fps:>8}  ({background_fp_rate:.1%} of predictions)")
    print("                            -> 'Phantom' detections with no real object")

    # Misclassification Rate: matches with IoU but wrong class
    misclassification_rate = wrong_class_matches / max(total_matched, 1)
    print(f"  Misclassifications:       {wrong_class_matches:>8}  ({misclassification_rate:.1%} of matches)")
    print("                            -> Localizes well but classifies wrong")

    # Correct detections
    correct_detection_rate = correct_class_matches / max(total_matched, 1)
    print(f"  Correct detections:       {correct_class_matches:>8}  ({correct_detection_rate:.1%} of matches)")

    # Analysis of the predominant error type
    print(f"\n  {'ERROR DIAGNOSIS:':^50}")
    if total_fn_sum > background_fps and total_fn_sum > wrong_class_matches:
        print("  -> Main problem: FALSE NEGATIVES (objects are not detected)")
        print("     Fix: lower detection_threshold or train longer")
    elif background_fps > total_fn_sum and background_fps > wrong_class_matches:
        print("  -> Main problem: PHANTOM DETECTIONS (background FP)")
        print("     Fix: raise detection_threshold or improve NMS")
    elif wrong_class_matches > total_fn_sum and wrong_class_matches > background_fps:
        print("  -> Main problem: MISCLASSIFICATION (localizes but does not classify)")
        print("     Fix: improve classification features or class_weights")
    else:
        print("  -> Errors spread across FN, FP and misclassification")

    # Explanatory note about the metrics
    print(f"\n  {'INTERPRETATION OF AGGREGATE METRICS:':^50}")
    print("  - Micro: dominated by classes with more instances")
    print("  - Macro: equal weight per class (better for imbalanced data)")
    print("  - Weighted: compromise between Micro and Macro")
    print("  - G-Mean: very sensitive to classes with low recall")
    print("  - For minority classes, focus on Macro-F1 and G-Mean")

    # Per-class analysis (highlighting minority classes if any)
    header_suffix = " - Minority classes marked with >>" if minority_classes else ""
    print(f"\n{'PER-CLASS ANALYSIS (Detection Metrics)':^90}")
    if minority_classes:
        print(f"{'Minority classes: ' + ', '.join(minority_classes):^90}")
    print("-" * 90)
    print(f"{'Class':<20} {'GT':>6} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Status':>12}")
    print("-" * 90)

    # Metrics computed as follows:
    # - TP: correct detections (matched with the correct label)
    # - FN: undetected GT + GT matched with a prediction of a different class
    # - FP: unmatched predictions + predictions matched with a GT of a different class
    # - Recall = TP / (TP + FN)
    # - Precision = TP / (TP + FP)
    # - F1 = 2 * (Precision * Recall) / (Precision + Recall)

    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]

        # Counters already computed during the evaluation
        tp = class_tp.get(cat_id, 0)
        fn = class_fn.get(cat_id, 0)
        fp = class_fp.get(cat_id, 0)
        gt_total = class_gt_total.get(cat_id, 0)

        # Metrics
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

        # Determine the status based on the F1-score
        if f1 < 0.3:
            estado = "CRITICAL"
        elif f1 < 0.5:
            estado = "LOW"
        elif f1 < 0.7:
            estado = "MODERATE"
        else:
            estado = "GOOD"

        # Highlight minority classes (only if any was detected)
        prefix = ">> " if cat_name in minority_classes else "   "
        print(f"{prefix}{cat_name:<17} {gt_total:>6} {tp:>6} {fn:>6} {fp:>6} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f} {estado:>12}")

    print("-" * 70)

    # Recommendations based on the analysis
    print(f"\n{'RECOMMENDATIONS':^70}")
    print("-" * 70)

    # Identify problematic classes
    problematic_classes = []
    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]
        tp = class_tp.get(cat_id, 0)
        fn = class_fn.get(cat_id, 0)
        gt_total = class_gt_total.get(cat_id, 0)
        recall = tp / max(tp + fn, 1)
        if recall < 0.5 and gt_total > 10:  # Only if there are enough samples
            problematic_classes.append((cat_name, recall, gt_total))

    if problematic_classes:
        print("\n  Classes with low recall (< 50%):")
        for name, recall, gt in sorted(problematic_classes, key=lambda x: x[1]):
            print(f"    - {name}: {recall:.1%} recall ({gt} GT)")
        print("\n  Suggested actions:")
        print("    1. Increase class_weights in the loss function for these classes")
        print("    2. Use ClassBalancedSampler to oversample images with these classes")
        print("    3. Apply specific augmentations (class-aware augmentations)")
        print("    4. Consider lowering detection_threshold for these specific classes")
    else:
        print("\n  ✓ All classes have recall >= 50%")

    # Score distribution analysis for minority classes (only if any)
    if minority_classes:
        print(f"\n{'SCORE ANALYSIS (Minority Classes)':^70}")
        print("-" * 70)
        for cat_name in minority_classes:
            cat_id = next((k for k, v in label_map.items() if v == cat_name), None)
            if cat_id is not None and cat_id in score_distribution:
                scores = score_distribution[cat_id]
                if scores:
                    above_05 = sum(1 for s in scores if s > 0.5)
                    above_025 = sum(1 for s in scores if s > 0.25)
                    above_01 = sum(1 for s in scores if s > 0.1)
                    total = len(scores)
                    print(f"\n  {cat_name}:")
                    print(f"    Total predictions (before filtering): {total}")
                    print(f"    Scores > 0.5: {above_05} ({100*above_05/max(total,1):.1f}%)")
                    print(f"    Scores > 0.25: {above_025} ({100*above_025/max(total,1):.1f}%)")
                    print(f"    Scores > 0.1: {above_01} ({100*above_01/max(total,1):.1f}%)")
                    if total > 0:
                        print(f"    Mean score: {np.mean(scores):.3f}, median: {np.median(scores):.3f}")

    print("\n" + "=" * 70)

    # Save the diagnostic summary to a file
    diag_report_path = os.path.join(args.base_eval_dir, "diagnostic_summary.txt")
    with open(diag_report_path, 'w', encoding='utf-8') as f:
        f.write("DIAGNOSTIC SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write("GLOBAL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Ground Truth:       {total_gt_sum}\n")
        f.write(f"Total predictions:       {total_pred}\n")
        f.write(f"True Positives (TP):      {total_tp}\n")
        f.write(f"False Negatives (FN):     {total_fn_sum} ({100*total_fn_sum/max(total_gt_sum,1):.1f}% of GT)\n")
        f.write(f"False Positives (FP):     {total_fp_sum} ({100*total_fp_sum/max(total_pred,1):.1f}% of pred)\n\n")

        f.write("AGGREGATE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Micro-Precision:          {micro_precision:.2%}\n")
        f.write(f"Micro-Recall:             {micro_recall:.2%}\n")
        f.write(f"Micro-F1:                 {micro_f1:.4f}\n")
        f.write(f"Macro-Precision:          {macro_precision:.2%}\n")
        f.write(f"Macro-Recall:             {macro_recall:.2%}\n")
        f.write(f"Macro-F1:                 {macro_f1:.4f}\n")
        f.write(f"Weighted-F1:              {weighted_f1:.4f}\n")
        f.write(f"G-Mean:                   {g_mean:.4f}\n\n")

        f.write("LOCALIZATION METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Localization Recall:      {localization_recall:.2%}\n")
        f.write(f"Mean IoU (matches):       {mean_iou:.4f}\n\n")

        f.write("ERROR BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"Background FP:            {background_fps} ({background_fp_rate:.1%} of pred)\n")
        f.write(f"Misclassifications:       {wrong_class_matches} ({misclassification_rate:.1%} of matches)\n")
        f.write(f"Correct detections:       {correct_class_matches} ({correct_detection_rate:.1%} of matches)\n\n")

        f.write("PER CLASS\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Class':<20} {'GT':>6} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>8} {'Precision':>10} {'F1':>8}\n")
        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            tp = class_tp.get(cat_id, 0)
            fn = class_fn.get(cat_id, 0)
            fp = class_fp.get(cat_id, 0)
            gt_total = class_gt_total.get(cat_id, 0)
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            f.write(f"{cat_name:<20} {gt_total:>6} {tp:>6} {fn:>6} {fp:>6} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f}\n")

    print(f"Diagnostic summary saved to: {diag_report_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation script for RT-DETR-V2.")
    parser.add_argument('-c', '--config', required=True, help="Path to the YAML configuration file.")
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('-r', '--resume', type=str, help="Path to the PyTorch checkpoint (.pth).")
    model_group.add_argument('--onnx', type=str, help="Path to the ONNX model (.onnx).")
    model_group.add_argument('--tensorrt', type=str, help="Path to the TensorRT engine (.engine).")

    parser.add_argument('-d', '--device', type=str, default='cuda', help="Device (cpu or cuda).")
    parser.add_argument('-dt', '--detection_threshold', type=float, default=0.5, help="Confidence threshold for detections.")
    parser.add_argument('-iout', '--iou_threshold', type=float, default=0.5, help="IoU threshold for matching.")
    parser.add_argument('--matching_method', type=str, default='hungarian', choices=['hungarian', 'greedy'],
                        help="Matching method: 'hungarian' (optimal, DETR standard) or 'greedy' (faster, score-based).")
    parser.add_argument('--output_dir', type=str, default='eval_results', help="Base directory to save the results.")
    parser.add_argument('--coco_eval', action='store_true', default=True, help="Enable mAP evaluation with pycocotools.")
    parser.add_argument('--draw', action='store_true', help="Save images with the detections drawn.")
    parser.add_argument('--debug', action='store_true', help="Save debug comparison images of prediction vs. ground truth when they differ.")

    args = parser.parse_args()
    device = torch.device(args.device)

    model_path = args.resume or args.onnx or args.tensorrt
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    args.base_eval_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(args.base_eval_dir, exist_ok=True)

    cfg = YAMLConfig(args.config)

    if args.draw:
        os.makedirs(os.path.join(args.base_eval_dir, "predicted_images"), exist_ok=True)

    if args.debug:
        os.makedirs(os.path.join(args.base_eval_dir, "debug_images"), exist_ok=True)

    evaluate_on_dataset(args, cfg, device)

if __name__ == '__main__':
    main()