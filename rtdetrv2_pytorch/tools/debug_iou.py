#!/usr/bin/env python
"""
Debug de IoU: Compara predicciones con ground truth para entender por qué mAP es bajo.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from src.core import YAMLConfig

def compute_iou(box1, box2):
    """Calcula IoU entre dos bboxes en formato XYXY."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def main():
    config_path = 'configs/rtdetrv2/rtdetrv2_r50vd_synthetic_overfit_test.yml'
    checkpoint_path = 'output/synthetic_overfit_test/best.pth'
    json_path = 'split_sintetico_simple/overfit_test.json'

    print("=" * 80)
    print("DEBUG DE IoU - COMPARACIÓN PREDICCIONES VS GROUND TRUTH")
    print("=" * 80)

    # Cargar ground truth
    with open(json_path) as f:
        gt_data = json.load(f)

    # Crear lookup de ground truth
    gt_by_image = {}
    for ann in gt_data['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        # Convertir bbox de XYWH a XYXY
        x, y, w, h = ann['bbox']
        gt_by_image[img_id].append({
            'bbox': [x, y, x + w, y + h],  # XYXY
            'category_id': ann['category_id']
        })

    # Cargar modelo
    cfg = YAMLConfig(config_path)
    model = cfg.model
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    postprocessor = cfg.postprocessor
    dataset = cfg.val_dataloader.dataset

    print(f"\nAnalizando {min(10, len(dataset))} muestras...")

    total_matches = 0
    total_gt = 0
    total_preds = 0

    for idx in range(min(10, len(dataset))):
        img_tensor, target = dataset[idx]

        # Obtener image_id real
        image_id = target['image_id']
        if isinstance(image_id, torch.Tensor):
            image_id = image_id.item()

        # Inferencia
        img_input = img_tensor.unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model(img_input)

        # Postprocess
        orig_size = target['orig_size'].unsqueeze(0).cuda()
        results = postprocessor(outputs, orig_size)
        result = results[0]

        # Filtrar predicciones por score
        mask = result['scores'] > 0.3
        pred_boxes = result['boxes'][mask].cpu()
        pred_labels = result['labels'][mask].cpu()
        pred_scores = result['scores'][mask].cpu()

        # Ground truth para esta imagen
        gt_list = gt_by_image.get(image_id, [])

        print(f"\n{'='*60}")
        print(f"Image ID: {image_id}")
        print(f"orig_size (H, W): {target['orig_size'].tolist()}")
        print(f"Ground Truth: {len(gt_list)} objetos")
        print(f"Predicciones (score > 0.3): {len(pred_boxes)}")

        if len(gt_list) == 0:
            print("  ⚠️ No hay ground truth para esta imagen!")
            continue

        # Comparar cada predicción con cada ground truth
        for i, (pbox, plabel, pscore) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            pbox = pbox.tolist()
            plabel = plabel.item()
            pscore = pscore.item()

            best_iou = 0
            best_gt_idx = -1
            best_gt_label = -1

            for j, gt in enumerate(gt_list):
                iou = compute_iou(pbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    best_gt_label = gt['category_id']

            match_str = "✅ MATCH" if best_iou > 0.5 and plabel == best_gt_label else "❌"
            label_match = "✅" if plabel == best_gt_label else f"❌ (pred={plabel}, gt={best_gt_label})"

            if i < 5:  # Solo mostrar primeras 5
                print(f"  Pred {i}: score={pscore:.2f}, label={plabel}, bbox={[round(x, 1) for x in pbox]}")
                print(f"    Best IoU: {best_iou:.3f} with GT {best_gt_idx}, label match: {label_match} {match_str}")

            if best_iou > 0.5 and plabel == best_gt_label:
                total_matches += 1

        total_gt += len(gt_list)
        total_preds += len(pred_boxes)

        # Mostrar GT
        print(f"\n  Ground Truth:")
        for j, gt in enumerate(gt_list):
            if j < 5:
                print(f"    GT {j}: label={gt['category_id']}, bbox={[round(x, 1) for x in gt['bbox']]}")

    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    print(f"Total GT objects: {total_gt}")
    print(f"Total predictions (score > 0.3): {total_preds}")
    print(f"Matches (IoU > 0.5 AND label match): {total_matches}")
    print(f"Recall: {total_matches/total_gt:.3f}" if total_gt > 0 else "N/A")
    print(f"Precision: {total_matches/total_preds:.3f}" if total_preds > 0 else "N/A")

if __name__ == '__main__':
    main()
