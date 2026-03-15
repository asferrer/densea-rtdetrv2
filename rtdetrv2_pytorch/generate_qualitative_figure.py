"""Generate qualitative results figure for the doctoral paper.

Loads COCO predictions from exp09 evaluation, finds images across
multiple dataset directories, selects representative successes and
failures, and creates a publication-quality grid figure.

Run inside the RT-DETRv2 Docker container:
  docker exec rt_detrv2 python /app/RT-DETR/generate_qualitative_figure.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


# --- Docker paths ---
SPLIT_PATH = "/app/RT-DETR/experiment_output/densea/datasets/exp07_real100/train.json"
# Use val.json from a known location - check if it exists in experiment_output
VAL_CANDIDATES = [
    "/app/RT-DETR/experiment_output/densea/datasets/exp07_real100/train.json",
]
PRED_PATH = (
    "/app/RT-DETR/experiment_output/densea/evaluation/"
    "exp09_real_synth_prop/best/coco_predictions.json"
)
IMAGE_DIRS = [
    "/app/RT-DETR/dataset/cleansea_dataset/CocoFormatDataset/train_coco",
    "/app/RT-DETR/dataset/cleansea_dataset/CocoFormatDataset/test_coco",
    "/app/RT-DETR/dataset/Ocean_garbage/train",
    "/app/RT-DETR/dataset/Ocean_garbage/valid",
    "/app/RT-DETR/dataset/Ocean_garbage/test",
    "/app/RT-DETR/dataset/Neural_Ocean/train",
    "/app/RT-DETR/dataset/Neural_Ocean/valid",
    "/app/RT-DETR/dataset/Neural_Ocean/test",
    "/app/RT-DETR/dataset/CleanSea/JPEGImages",
]
OUTPUT_PATH = "/app/RT-DETR/qualitative_examples.png"
SCORE_THRESHOLD = 0.5
CATEGORIES = {
    0: 'Bottle', 1: 'Can', 2: 'Fishing_Net', 3: 'Glove',
    4: 'Mask', 5: 'Metal_Debris', 6: 'Plastic_Debris', 7: 'Tire'
}
COLORS = {
    0: '#2196F3', 1: '#FF9800', 2: '#4CAF50', 3: '#9C27B0',
    4: '#F44336', 5: '#795548', 6: '#00BCD4', 7: '#607D8B'
}


def find_image(file_name):
    """Search for an image across multiple directories."""
    basename = os.path.basename(file_name)
    for d in IMAGE_DIRS:
        if not os.path.isdir(d):
            continue
        candidate = os.path.join(d, basename)
        if os.path.exists(candidate):
            return candidate
        full = os.path.join(d, file_name)
        if os.path.exists(full):
            return full
    return None


def find_val_json():
    """Find val.json - check multiple locations."""
    # First check if split_v8 is somewhere accessible
    candidates = [
        "/app/RT-DETR/split_v8/val.json",
        "/app/RT-DETR/splits/split_v8/val.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Search in experiment output for any val.json
    for root, dirs, files in os.walk("/app/RT-DETR/experiment_output"):
        if "val.json" in files:
            path = os.path.join(root, "val.json")
            print(f"Found val.json at: {path}")
            return path

    # Search in dataset directory
    for root, dirs, files in os.walk("/app/RT-DETR/dataset"):
        if "val.json" in files:
            path = os.path.join(root, "val.json")
            print(f"Found val.json at: {path}")
            return path

    return None


def load_data(val_path):
    """Load val.json and predictions."""
    with open(val_path) as f:
        val_data = json.load(f)
    with open(PRED_PATH) as f:
        predictions = json.load(f)

    id_to_img = {img['id']: img for img in val_data['images']}
    id_to_annots = defaultdict(list)
    for ann in val_data['annotations']:
        id_to_annots[ann['image_id']].append(ann)

    id_to_preds = defaultdict(list)
    for pred in predictions:
        if pred['score'] >= SCORE_THRESHOLD:
            id_to_preds[pred['image_id']].append(pred)

    return id_to_img, id_to_annots, id_to_preds


def score_image_success(annots, preds):
    """Score how good an image is as a success example."""
    if not annots or not preds:
        return -1

    gt_cats = set(a['category_id'] for a in annots)
    pred_cats = set(p['category_id'] for p in preds)
    avg_score = np.mean([p['score'] for p in preds])

    # Prefer: diverse classes, many objects, high confidence
    diversity = len(gt_cats)
    coverage = len(pred_cats & gt_cats) / max(len(gt_cats), 1)
    n_objects = min(len(annots), 8)  # Cap at 8

    return avg_score * coverage * diversity * (1 + 0.1 * n_objects)


def score_image_failure(annots, preds):
    """Score how informative a failure case is."""
    if not annots:
        return -1

    gt_cats = set(a['category_id'] for a in annots)
    pred_cats = set(p['category_id'] for p in preds)
    n_gt = len(annots)
    n_pred = len(preds)

    # Count missed objects (FN) and wrong class predictions
    missed = max(0, n_gt - n_pred)
    wrong_cats = pred_cats - gt_cats

    # Prefer: interesting failures (some hits, some misses)
    if n_pred == 0:
        return missed * 0.5  # Total miss is less interesting
    return missed + len(wrong_cats) * 2


def select_images(id_to_img, id_to_annots, id_to_preds):
    """Select representative success and failure images."""
    success_scores = []
    failure_scores = []

    for img_id, img_info in id_to_img.items():
        annots = id_to_annots[img_id]
        preds = id_to_preds[img_id]

        if not annots:
            continue

        img_path = find_image(img_info['file_name'])
        if img_path is None:
            continue

        s_score = score_image_success(annots, preds)
        f_score = score_image_failure(annots, preds)

        success_scores.append((s_score, img_id, img_path))
        failure_scores.append((f_score, img_id, img_path))

    # Sort and take top 4 each
    success_scores.sort(reverse=True)
    failure_scores.sort(reverse=True)

    successes = [(s[1], s[2]) for s in success_scores[:4]]
    failures = [(f[1], f[2]) for f in failure_scores[:4]]

    return successes, failures


def draw_detections(ax, img_path, preds, annots, title=""):
    """Draw bounding boxes on an image."""
    img = Image.open(img_path).convert('RGB')
    ax.imshow(img)

    # Draw ground truth in dashed white
    for ann in annots:
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.2,
            edgecolor='white', facecolor='none', linestyle='--', alpha=0.5
        )
        ax.add_patch(rect)

    # Draw predictions with colored boxes
    for pred in preds:
        x, y, w, h = pred['bbox']
        cat_id = pred['category_id']
        score = pred['score']
        color = COLORS.get(cat_id, '#FFFFFF')
        label = f"{CATEGORIES[cat_id]} {score:.0%}"

        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2,
            edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x, y - 4, label,
            fontsize=5, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.85)
        )

    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.axis('off')


def main():
    print("Finding val.json...")
    val_path = find_val_json()
    if val_path is None:
        print("ERROR: Could not find val.json!")
        print("Available dirs in experiment_output:")
        for item in os.listdir("/app/RT-DETR/experiment_output"):
            print(f"  {item}")
        return
    print(f"Using val.json: {val_path}")

    print("Loading data...")
    id_to_img, id_to_annots, id_to_preds = load_data(val_path)
    print(f"  Images: {len(id_to_img)}, with predictions: {len(id_to_preds)}")

    print("Checking image accessibility...")
    found = 0
    total = min(50, len(id_to_img))
    for i, (img_id, img_info) in enumerate(id_to_img.items()):
        if i >= total:
            break
        if find_image(img_info['file_name']) is not None:
            found += 1
    print(f"  Found {found}/{total} sampled images")

    if found == 0:
        print("ERROR: No images found! Checking directories...")
        for d in IMAGE_DIRS:
            if os.path.isdir(d):
                n = len(os.listdir(d))
                print(f"  {d}: {n} files")
            else:
                print(f"  {d}: NOT FOUND")
        return

    print("Selecting images...")
    successes, failures = select_images(id_to_img, id_to_annots, id_to_preds)
    print(f"  Successes: {len(successes)}, Failures: {len(failures)}")

    n_success = len(successes)
    n_failure = len(failures)
    n_cols = max(n_success, n_failure, 1)

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), dpi=300)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Top row: successes
    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_success:
            img_id, img_path = successes[i]
            n_det = len(id_to_preds[img_id])
            n_gt = len(id_to_annots[img_id])
            cats = set(CATEGORIES[p['category_id']]
                       for p in id_to_preds[img_id])
            draw_detections(
                ax, img_path,
                id_to_preds[img_id], id_to_annots[img_id],
                title=f"{n_det}/{n_gt} detected | {', '.join(sorted(cats))}"
            )
        else:
            ax.axis('off')

    # Bottom row: failures
    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_failure:
            img_id, img_path = failures[i]
            n_det = len(id_to_preds[img_id])
            n_gt = len(id_to_annots[img_id])
            missed = n_gt - n_det
            draw_detections(
                ax, img_path,
                id_to_preds[img_id], id_to_annots[img_id],
                title=f"{n_det}/{n_gt} detected | {missed} missed"
            )
        else:
            ax.axis('off')

    fig.text(0.01, 0.73, '(a) Successful\n     detections',
             fontsize=9, fontweight='bold', rotation=90, va='center')
    fig.text(0.01, 0.27, '(b) Failure\n     cases',
             fontsize=9, fontweight='bold', rotation=90, va='center')

    plt.tight_layout(rect=[0.04, 0.01, 0.99, 0.99])
    plt.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"Figure saved to: {OUTPUT_PATH}")
    plt.close()


if __name__ == '__main__':
    main()
