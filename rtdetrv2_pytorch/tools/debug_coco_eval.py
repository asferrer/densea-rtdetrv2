#!/usr/bin/env python
"""
Script para debuggear la discrepancia entre visualizaciones y métricas COCO.
Verifica si el problema es de mapping de categorías o formato de bboxes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.core import YAMLConfig

def debug_evaluation(config_path, checkpoint_path, json_path):
    """Debug completo del proceso de evaluación."""

    print("=" * 80)
    print("DEBUG DE EVALUACIÓN COCO")
    print("=" * 80)

    # 1. Verificar categorías en el JSON de ground truth
    print("\n[1] CATEGORÍAS EN EL GROUND TRUTH JSON")
    print("-" * 40)

    coco_gt = COCO(json_path)
    gt_cat_ids = coco_gt.getCatIds()
    gt_categories = coco_gt.loadCats(gt_cat_ids)

    print(f"Category IDs en el JSON: {gt_cat_ids}")
    print("Detalle de categorías:")
    for cat in gt_categories:
        print(f"  ID: {cat['id']} -> {cat['name']}")

    # 2. Cargar modelo y verificar predicciones
    print("\n[2] PREDICCIONES DEL MODELO")
    print("-" * 40)

    cfg = YAMLConfig(config_path)
    model = cfg.model
    model.eval()

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()

    # Obtener postprocessor
    postprocessor = cfg.postprocessor
    print(f"Postprocessor num_classes: {postprocessor.num_classes}")
    print(f"Postprocessor remap_mscoco_category: {postprocessor.remap_mscoco_category}")

    # Cargar una muestra
    dataset = cfg.val_dataloader.dataset
    img_tensor, target = dataset[0]

    print(f"\nTarget keys: {target.keys()}")
    print(f"Target labels: {target.get('labels', 'N/A')}")
    print(f"Target orig_size: {target.get('orig_size', 'N/A')}")

    # Inferencia
    img_input = img_tensor.unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = model(img_input)

    # Aplicar postprocessor
    orig_target_sizes = target['orig_size'].unsqueeze(0).cuda()
    results = postprocessor(outputs, orig_target_sizes)

    result = results[0]
    print(f"\nResultados del postprocessor:")
    print(f"  Labels (category IDs predichos): {result['labels'][:10]}...")  # Primeros 10
    print(f"  Scores: {result['scores'][:10]}...")
    print(f"  Boxes shape: {result['boxes'].shape}")

    # Filtrar por score > 0.5
    mask = result['scores'] > 0.5
    if mask.sum() > 0:
        filtered_labels = result['labels'][mask].cpu().numpy()
        filtered_scores = result['scores'][mask].cpu().numpy()
        print(f"\n  Predicciones con score > 0.5:")
        print(f"    Labels: {filtered_labels}")
        print(f"    Scores: {filtered_scores}")

    # 3. Comparar IDs de categorías
    print("\n[3] COMPARACIÓN DE CATEGORY IDs")
    print("-" * 40)

    pred_labels_unique = result['labels'].unique().cpu().numpy()
    print(f"Category IDs que predice el modelo: {sorted(pred_labels_unique)}")
    print(f"Category IDs en el ground truth: {sorted(gt_cat_ids)}")

    # Verificar si coinciden
    pred_set = set(pred_labels_unique.tolist())
    gt_set = set(gt_cat_ids)

    if pred_set.issubset(gt_set):
        print("✅ Los IDs predichos están dentro de los IDs del ground truth")
    else:
        missing = pred_set - gt_set
        print(f"❌ PROBLEMA: El modelo predice IDs que NO existen en el ground truth: {missing}")
        print("   Esto causaría que TODAS las predicciones sean ignoradas!")

    # 4. Simular evaluación COCO manual
    print("\n[4] SIMULACIÓN DE EVALUACIÓN COCO")
    print("-" * 40)

    # Obtener algunas muestras
    num_samples = 10
    all_predictions = []

    print(f"Procesando {num_samples} muestras...")

    for idx in range(min(num_samples, len(dataset))):
        img_tensor, target = dataset[idx]
        img_input = img_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = model(img_input)

        orig_target_sizes = target['orig_size'].unsqueeze(0).cuda()
        results = postprocessor(outputs, orig_target_sizes)
        result = results[0]

        # Filtrar por score
        mask = result['scores'] > 0.3  # Umbral bajo para debug
        if mask.sum() > 0:
            boxes = result['boxes'][mask].cpu()
            scores = result['scores'][mask].cpu()
            labels = result['labels'][mask].cpu()

            # Convertir de XYXY a XYWH (formato COCO)
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width = xmax - xmin
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height = ymax - ymin

            image_id = target.get('image_id', idx)
            if isinstance(image_id, torch.Tensor):
                image_id = image_id.item()

            for box, score, label in zip(boxes_xywh, scores, labels):
                pred = {
                    "image_id": int(image_id),
                    "category_id": int(label.item()),
                    "bbox": box.tolist(),
                    "score": float(score.item())
                }
                all_predictions.append(pred)

    print(f"\nTotal predicciones generadas: {len(all_predictions)}")

    if len(all_predictions) > 0:
        print(f"Muestra de predicciones:")
        for pred in all_predictions[:5]:
            print(f"  image_id={pred['image_id']}, cat_id={pred['category_id']}, "
                  f"score={pred['score']:.3f}, bbox={[round(x, 1) for x in pred['bbox']]}")

        # Verificar category_ids en predicciones vs ground truth
        pred_cat_ids = set(p['category_id'] for p in all_predictions)
        print(f"\nCategory IDs en predicciones: {sorted(pred_cat_ids)}")
        print(f"Category IDs en ground truth: {sorted(gt_cat_ids)}")

        overlap = pred_cat_ids & gt_set
        if len(overlap) == 0:
            print("\n" + "!" * 60)
            print("! PROBLEMA CRÍTICO: NO hay overlap entre IDs predichos y GT!")
            print("! Esto explica por qué mAP = 0")
            print("!" * 60)
        else:
            print(f"\nOverlap de IDs: {sorted(overlap)}")

    # 5. Verificar ground truth
    print("\n[5] VERIFICACIÓN DE GROUND TRUTH")
    print("-" * 40)

    # Ver algunas anotaciones del ground truth
    img_ids = coco_gt.getImgIds()[:5]
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        print(f"Image ID {img_id}: {len(anns)} anotaciones")
        for ann in anns[:2]:
            print(f"  cat_id={ann['category_id']}, bbox={[round(x, 1) for x in ann['bbox']]}")

    print("\n" + "=" * 80)
    print("FIN DEL DEBUG")
    print("=" * 80)

if __name__ == '__main__':
    config_path = 'configs/rtdetrv2/rtdetrv2_r50vd_synthetic_overfit_test.yml'
    checkpoint_path = 'output/synthetic_overfit_test/best.pth'
    json_path = 'split_sintetico_simple/overfit_test.json'

    debug_evaluation(config_path, checkpoint_path, json_path)
