"""
Validador de Pipeline de Dataset para RT-DETRv2.

Verifica que el dataset carga correctamente en el pipeline de RT-DETRv2,
incluyendo transformaciones, formato de tensores y compatibilidad con el modelo.

Uso:
    python tools/validate_dataset_pipeline.py \
        -c configs/rtdetrv2/rtdetrv2_r50vd_synthetic_overfit_test.yml \
        --num-samples 100 \
        --check-collate
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Agregar path del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core import YAMLConfig


def validate_tensor(tensor, name, expected_dtype=None, value_range=None, allow_empty=False):
    """Valida un tensor y retorna lista de errores encontrados."""
    errors = []

    if tensor is None:
        if not allow_empty:
            errors.append(f"{name}: Es None (no permitido)")
        return errors

    if not isinstance(tensor, torch.Tensor):
        errors.append(f"{name}: No es un tensor (tipo: {type(tensor).__name__})")
        return errors

    # Verificar dtype
    if expected_dtype and tensor.dtype != expected_dtype:
        errors.append(f"{name}: dtype incorrecto ({tensor.dtype} vs esperado {expected_dtype})")

    # Verificar NaN/Inf
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        errors.append(f"{name}: Contiene {nan_count} valores NaN")

    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        errors.append(f"{name}: Contiene {inf_count} valores Inf")

    # Verificar rango de valores
    if value_range and tensor.numel() > 0:
        min_val, max_val = value_range
        actual_min = tensor.min().item()
        actual_max = tensor.max().item()

        # Tolerancia de 0.05 para permitir valores ligeramente fuera de rango
        # debido a transformaciones de augmentación (rotación, interpolación, etc.)
        tolerance = 0.05

        if actual_min < min_val - tolerance:
            errors.append(f"{name}: Valor min ({actual_min:.4f}) < rango esperado ({min_val})")
        if actual_max > max_val + tolerance:
            errors.append(f"{name}: Valor max ({actual_max:.4f}) > rango esperado ({max_val})")

    return errors


def validate_sample(img, target, idx, num_classes):
    """Valida una muestra individual del dataset."""
    errors = []
    warnings = []
    stats = {}

    # Validar imagen
    if img is None:
        errors.append(f"Sample {idx}: Imagen es None")
        return errors, warnings, stats

    img_errors = validate_tensor(img, f"Sample {idx} imagen",
                                  expected_dtype=torch.float32,
                                  value_range=(0.0, 1.0))
    errors.extend(img_errors)

    # Verificar shape de imagen
    if len(img.shape) != 3:
        errors.append(f"Sample {idx}: Shape de imagen incorrecto ({img.shape}), esperado [C, H, W]")
    elif img.shape[0] not in [1, 3]:
        errors.append(f"Sample {idx}: Canales de imagen incorrectos ({img.shape[0]}), esperado 1 o 3")

    stats['img_shape'] = tuple(img.shape) if img is not None else None
    stats['img_dtype'] = str(img.dtype) if img is not None else None

    # Validar target
    if target is None:
        warnings.append(f"Sample {idx}: Target es None (imagen sin anotaciones)")
        return errors, warnings, stats

    # Validar boxes
    if 'boxes' not in target:
        errors.append(f"Sample {idx}: Target no contiene 'boxes'")
    else:
        boxes = target['boxes']

        # Convertir BoundingBoxes a tensor si es necesario
        if hasattr(boxes, 'data'):
            boxes_tensor = boxes.data if hasattr(boxes, 'data') else torch.tensor(boxes)
        else:
            boxes_tensor = boxes

        box_errors = validate_tensor(boxes_tensor, f"Sample {idx} boxes",
                                      expected_dtype=torch.float32,
                                      value_range=(0.0, 1.0))  # Normalizado
        errors.extend(box_errors)

        if boxes_tensor.numel() > 0:
            if len(boxes_tensor.shape) != 2 or boxes_tensor.shape[1] != 4:
                errors.append(f"Sample {idx}: Shape de boxes incorrecto ({boxes_tensor.shape}), esperado [N, 4]")

            # Verificar formato CXCYWH (centro + dimensiones)
            # En CXCYWH normalizado: cx, cy en [0,1], w, h > 0 y <= 1
            if boxes_tensor.shape[0] > 0:
                cx, cy, w, h = boxes_tensor[:, 0], boxes_tensor[:, 1], boxes_tensor[:, 2], boxes_tensor[:, 3]

                if (w <= 0).any() or (h <= 0).any():
                    errors.append(f"Sample {idx}: Boxes con width/height <= 0")

                if (w > 1.0 + 1e-6).any() or (h > 1.0 + 1e-6).any():
                    errors.append(f"Sample {idx}: Boxes con width/height > 1.0 (no normalizado)")

        stats['num_boxes'] = boxes_tensor.shape[0] if boxes_tensor.numel() > 0 else 0
        stats['boxes_shape'] = tuple(boxes_tensor.shape)

    # Validar labels
    if 'labels' not in target:
        errors.append(f"Sample {idx}: Target no contiene 'labels'")
    else:
        labels = target['labels']

        label_errors = validate_tensor(labels, f"Sample {idx} labels",
                                        expected_dtype=torch.int64)
        errors.extend(label_errors)

        if labels.numel() > 0:
            min_label = labels.min().item()
            max_label = labels.max().item()

            if min_label < 0:
                errors.append(f"Sample {idx}: Labels con valores negativos (min={min_label})")

            if max_label >= num_classes:
                errors.append(f"Sample {idx}: Labels fuera de rango (max={max_label}, num_classes={num_classes})")

        stats['labels_range'] = (labels.min().item(), labels.max().item()) if labels.numel() > 0 else None

    # Verificar consistencia boxes/labels
    if 'boxes' in target and 'labels' in target:
        boxes_tensor = target['boxes'].data if hasattr(target['boxes'], 'data') else target['boxes']
        if boxes_tensor.shape[0] != target['labels'].shape[0]:
            errors.append(f"Sample {idx}: Mismatch boxes ({boxes_tensor.shape[0]}) vs labels ({target['labels'].shape[0]})")

    return errors, warnings, stats


def validate_collate(dataloader, num_batches=5):
    """Valida que el collate_fn funciona correctamente."""
    errors = []
    batch_stats = []

    print(f"\nValidando collate_fn con {num_batches} batches...")

    for i, (images, targets) in enumerate(dataloader):
        if i >= num_batches:
            break

        batch_info = {
            'batch_idx': i,
            'images_shape': tuple(images.shape),
            'num_targets': len(targets)
        }

        # Verificar images batch
        if len(images.shape) != 4:
            errors.append(f"Batch {i}: Shape de images incorrecto ({images.shape}), esperado [B, C, H, W]")

        if images.dtype != torch.float32:
            errors.append(f"Batch {i}: dtype de images incorrecto ({images.dtype}), esperado float32")

        if images.min() < -1e-6 or images.max() > 1.0 + 1e-6:
            errors.append(f"Batch {i}: Valores de images fuera de [0, 1]: [{images.min():.4f}, {images.max():.4f}]")

        # Verificar targets
        if len(targets) != images.shape[0]:
            errors.append(f"Batch {i}: Mismatch batch_size images ({images.shape[0]}) vs targets ({len(targets)})")

        total_boxes = sum(t['boxes'].shape[0] if 'boxes' in t else 0 for t in targets)
        batch_info['total_boxes'] = total_boxes

        batch_stats.append(batch_info)
        print(f"  Batch {i}: images={images.shape}, targets={len(targets)}, total_boxes={total_boxes}")

    return errors, batch_stats


def main():
    parser = argparse.ArgumentParser(description='Validar pipeline de dataset RT-DETRv2')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Archivo de configuración YAML')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Número de muestras a validar')
    parser.add_argument('--check-collate', action='store_true',
                        help='También validar collate_fn con batches')
    parser.add_argument('--num-batches', type=int, default=5,
                        help='Número de batches para validar collate')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both',
                        help='Split a validar')
    parser.add_argument('--verbose', action='store_true',
                        help='Mostrar información detallada')
    # Parámetros opcionales para sobrescribir paths del dataset
    parser.add_argument('--train-json', type=str, default=None,
                        help='Sobrescribir path al JSON de entrenamiento')
    parser.add_argument('--val-json', type=str, default=None,
                        help='Sobrescribir path al JSON de validación')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Sobrescribir directorio de imágenes')
    args = parser.parse_args()

    print("=" * 80)
    print("VALIDADOR DE PIPELINE DE DATASET RT-DETRv2")
    print("=" * 80)
    print(f"\nConfig: {args.config}")
    print(f"Samples a validar: {args.num_samples}")

    # Preparar kwargs para sobrescribir configuración
    config_overrides = {}
    if args.train_json:
        config_overrides['train_dataloader'] = {'dataset': {'ann_file': args.train_json}}
        print(f"Sobrescribiendo train JSON: {args.train_json}")
    if args.val_json:
        config_overrides['val_dataloader'] = {'dataset': {'ann_file': args.val_json}}
        print(f"Sobrescribiendo val JSON: {args.val_json}")
    if args.image_dir:
        if 'train_dataloader' not in config_overrides:
            config_overrides['train_dataloader'] = {'dataset': {}}
        if 'val_dataloader' not in config_overrides:
            config_overrides['val_dataloader'] = {'dataset': {}}
        config_overrides['train_dataloader']['dataset']['img_folder'] = args.image_dir
        config_overrides['val_dataloader']['dataset']['img_folder'] = args.image_dir
        print(f"Sobrescribiendo image dir: {args.image_dir}")

    # Cargar configuración con sobrescrituras
    cfg = YAMLConfig(args.config, **config_overrides)
    num_classes = cfg.yaml_cfg.get('num_classes', 6)
    print(f"Número de clases: {num_classes}")

    all_errors = []
    all_warnings = []

    splits_to_check = []
    if args.split in ['train', 'both']:
        splits_to_check.append(('train', cfg.train_dataloader.dataset))
    if args.split in ['val', 'both']:
        splits_to_check.append(('val', cfg.val_dataloader.dataset))

    for split_name, dataset in splits_to_check:
        print(f"\n{'─' * 80}")
        print(f"VALIDANDO SPLIT: {split_name.upper()}")
        print(f"{'─' * 80}")
        print(f"Dataset size: {len(dataset)}")

        split_errors = []
        split_warnings = []
        all_stats = []

        # Seleccionar índices a validar
        num_to_check = min(args.num_samples, len(dataset))
        indices = np.random.choice(len(dataset), num_to_check, replace=False)
        # Convertir a int de Python (torchvision COCO dataset no acepta numpy.int64)
        indices = [int(idx) for idx in indices]

        print(f"\nValidando {num_to_check} muestras...")

        for i, idx in enumerate(indices):
            try:
                img, target = dataset[idx]
                errors, warnings, stats = validate_sample(img, target, idx, num_classes)
                split_errors.extend(errors)
                split_warnings.extend(warnings)
                all_stats.append(stats)

                if args.verbose and (errors or warnings):
                    if errors:
                        print(f"  [{idx}] ERRORES: {errors}")
                    if warnings:
                        print(f"  [{idx}] WARNINGS: {warnings}")

            except Exception as e:
                split_errors.append(f"Sample {idx}: Excepción al cargar: {str(e)}")
                if args.verbose:
                    import traceback
                    print(f"\n  ERROR en sample {idx}:")
                    traceback.print_exc()

            # Progress
            if (i + 1) % 20 == 0:
                print(f"  Procesadas {i + 1}/{num_to_check} muestras...")

        # Estadísticas agregadas
        if all_stats:
            valid_stats = [s for s in all_stats if s.get('num_boxes') is not None]
            if valid_stats:
                avg_boxes = np.mean([s['num_boxes'] for s in valid_stats])
                max_boxes = max([s['num_boxes'] for s in valid_stats])
                min_boxes = min([s['num_boxes'] for s in valid_stats])

                print(f"\nEstadísticas de {split_name}:")
                print(f"  Boxes por imagen: min={min_boxes}, max={max_boxes}, avg={avg_boxes:.2f}")

                # Verificar shapes consistentes
                shapes = set(s.get('img_shape') for s in valid_stats if s.get('img_shape'))
                if len(shapes) > 1:
                    print(f"  WARNING: Múltiples shapes de imagen: {shapes}")

        # Resumen del split
        print(f"\nResumen {split_name}:")
        print(f"  Errores: {len(split_errors)}")
        print(f"  Warnings: {len(split_warnings)}")

        if split_errors:
            print(f"\n  Primeros 10 errores:")
            for err in split_errors[:10]:
                print(f"    - {err}")

        all_errors.extend(split_errors)
        all_warnings.extend(split_warnings)

    # Validar collate si se solicita
    if args.check_collate:
        print(f"\n{'─' * 80}")
        print("VALIDANDO COLLATE_FN")
        print(f"{'─' * 80}")

        for split_name in ['train', 'val']:
            if args.split not in [split_name, 'both']:
                continue

            print(f"\n{split_name.upper()} DataLoader:")
            try:
                if split_name == 'train':
                    dataloader = cfg.train_dataloader
                else:
                    dataloader = cfg.val_dataloader

                collate_errors, batch_stats = validate_collate(dataloader, args.num_batches)
                all_errors.extend(collate_errors)

                if collate_errors:
                    print(f"  Errores de collate: {len(collate_errors)}")
                    for err in collate_errors:
                        print(f"    - {err}")
                else:
                    print(f"  Collate OK - {len(batch_stats)} batches validados")

            except Exception as e:
                all_errors.append(f"Collate {split_name}: Excepción: {str(e)}")
                print(f"  ERROR: {e}")

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    if not all_errors:
        print("\n[OK] VALIDACIÓN EXITOSA")
        print("El dataset carga correctamente en el pipeline RT-DETRv2.")
        print("Criterios cumplidos:")
        print("  - Todas las muestras cargan sin errores")
        print("  - Tensores con shapes y dtypes correctos")
        print("  - Valores de imagen en rango [0, 1]")
        print("  - Boxes en formato CXCYWH normalizado [0, 1]")
        print("  - Labels en rango válido [0, num_classes)")
        return 0
    else:
        print(f"\n[FALLO] VALIDACIÓN CON ERRORES")
        print(f"Total errores: {len(all_errors)}")
        print(f"Total warnings: {len(all_warnings)}")
        print("\nErrores encontrados:")
        for err in all_errors[:20]:
            print(f"  - {err}")
        if len(all_errors) > 20:
            print(f"  ... y {len(all_errors) - 20} errores más")
        return 1


if __name__ == '__main__':
    sys.exit(main())
