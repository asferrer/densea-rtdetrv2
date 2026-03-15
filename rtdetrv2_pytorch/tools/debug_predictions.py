#!/usr/bin/env python
"""
Script para debuggear predicciones del modelo durante overfitting.
Visualiza qué está prediciendo el modelo y por qué el mAP es 0.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from src.core import YAMLConfig

def visualize_predictions(config_path, checkpoint_path, num_samples=5):
    """Visualiza predicciones del modelo."""

    # Cargar config y modelo
    cfg = YAMLConfig(config_path)
    model = cfg.model
    model.eval()

    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    # Obtener dataset
    dataset = cfg.val_dataloader.dataset

    print(f"Dataset size: {len(dataset)}")
    print(f"Generando {num_samples} visualizaciones...")

    # Procesar algunas muestras
    for idx in range(min(num_samples, len(dataset))):
        print(f"\nProcessing sample {idx}...")

        # Cargar imagen y target
        img_tensor, target = dataset[idx]

        # Preparar para el modelo
        img_input = img_tensor.unsqueeze(0).cuda()

        # Inferencia
        with torch.no_grad():
            outputs = model(img_input)

        # Obtener predicciones
        # El modelo RT-DETR devuelve (boxes, scores, labels)
        pred_boxes = outputs['pred_boxes'][0]  # [num_queries, 4]
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_scores = pred_logits.sigmoid().max(dim=-1)[0]
        pred_labels = pred_logits.sigmoid().argmax(dim=-1)

        # Filtrar por score
        threshold = 0.5 # Muy bajo para ver TODAS las predicciones
        mask = pred_scores > threshold

        print(f"  Total queries: {len(pred_scores)}")
        print(f"  Predictions > {threshold}: {mask.sum().item()}")
        print(f"  Max score: {pred_scores.max().item():.4f}")
        print(f"  Mean score: {pred_scores.mean().item():.4f}")

        if mask.sum() > 0:
            print(f"  Top 5 scores: {pred_scores.topk(5)[0].cpu().numpy()}")

        # Visualizar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Convertir imagen a numpy
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # Ground truth
        ax1.imshow(img_np)
        ax1.set_title(f'Ground Truth (Sample {idx})')

        if 'boxes' in target:
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            # Convertir de CXCYWH normalizado a XYXY pixels
            h, w = img_np.shape[:2]
            for box, label in zip(gt_boxes, gt_labels):
                cx, cy, bw, bh = box
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                box_w = bw * w
                box_h = bh * h

                rect = patches.Rectangle(
                    (x1, y1), box_w, box_h,
                    linewidth=2, edgecolor='green', facecolor='none'
                )
                ax1.add_patch(rect)
                ax1.text(x1, y1-5, f'GT: {label.item()}', color='green',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Predictions
        ax2.imshow(img_np)
        ax2.set_title(f'Predictions (Sample {idx}) - Threshold={threshold}')

        if mask.sum() > 0:
            filtered_boxes = pred_boxes[mask].cpu()  # ← Mover a CPU
            filtered_scores = pred_scores[mask].cpu()
            filtered_labels = pred_labels[mask].cpu()

            # Convertir de CXCYWH normalizado a XYXY pixels
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                cx, cy, bw, bh = box
                x1 = float((cx - bw/2) * w)  # ← Convertir a float
                y1 = float((cy - bh/2) * h)
                box_w = float(bw * w)
                box_h = float(bh * h)

                rect = patches.Rectangle(
                    (x1, y1), box_w, box_h,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax2.add_patch(rect)
                ax2.text(x1, y1-5, f'P: {label.item()} ({score:.3f})', color='red',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax2.text(w/2, h/2, 'NO PREDICTIONS!',
                    ha='center', va='center', fontsize=20, color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        ax1.axis('off')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(f'debug_predictions_sample_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved: debug_predictions_sample_{idx}.png")

if __name__ == '__main__':
    config_path = 'configs/rtdetrv2/rtdetrv2_r50vd_synthetic_overfit_test.yml'
    checkpoint_path = 'output/synthetic_overfit_test/best.pth'

    print("=" * 80)
    print("DEBUGGER DE PREDICCIONES - TEST DE OVERFITTING")
    print("=" * 80)
    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("")

    visualize_predictions(config_path, checkpoint_path, num_samples=25)

    print("\n" + "=" * 80)
    print("COMPLETADO")
    print("Revisa las imágenes generadas: debug_predictions_sample_*.png")
    print("=" * 80)
