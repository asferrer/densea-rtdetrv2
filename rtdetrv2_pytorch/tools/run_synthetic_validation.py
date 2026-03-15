#!/usr/bin/env python
"""
Script Maestro de Validacion de Dataset Sintetico para RT-DETRv2.

Ejecuta todas las fases de validacion en orden:

FASE 1: Validacion de Integridad (Bloqueante)
  1.1 Verificar indexacion de categorias
  1.2 Validar formato COCO e imagenes
  1.3 Validar carga en pipeline RT-DETR
  1.4 Detectar imagenes duplicadas entre splits (train/val/test)
  1.5 Detectar imagenes corruptas

FASE 2: Validacion de Calidad (Informativa)
  2.1 Analisis estadistico del dataset
  2.2 Detectar padding en bboxes
  2.3 Auditoria visual (genera muestras)
  2.4 Comparacion de calidad con dataset real (requiere --real-json)
  2.5 Analisis de domain gap (requiere --real-image-dir)

FASE 3: Test de Entrenabilidad (Critica)
  3.1 Crear subset de overfitting
  3.2 Entrenar test de overfitting

FASE 4: Test de Convergencia (Final)
  4.1 Entrenar en dataset completo

Uso:
    # Validar y entrenar (todas las fases, sin comparacion con real)
    python tools/run_synthetic_validation.py --all \\
        --train-json split_sintetico_taco/train.json \\
        --val-json split_sintetico_taco/val.json \\
        --image-dir dataset/dataset_sintetico_taco/images

    # Validacion completa con comparacion vs dataset real
    python tools/run_synthetic_validation.py --all \\
        --train-json split_sintetico_taco/train.json \\
        --val-json split_sintetico_taco/val.json \\
        --image-dir dataset/dataset_sintetico_taco/images \\
        --real-json dataset/real/train.json \\
        --real-image-dir dataset/real/images

    # Solo validacion (fases 1 y 2, sin entrenamiento)
    python tools/run_synthetic_validation.py --phases 1 2 \\
        --train-json split_sintetico_simple/train.json \\
        --val-json split_sintetico_simple/val.json \\
        --image-dir dataset/dataset_sintetico_simple/images

    # Modo rapido (menos muestras, menos epocas)
    python tools/run_synthetic_validation.py --all --quick \\
        --train-json split_sintetico_simple/train.json \\
        --val-json split_sintetico_simple/val.json \\
        --image-dir dataset/dataset_sintetico_simple/images
"""

import os
import sys
import argparse
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime


# Colores para output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_phase(phase_num, text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}[FASE {phase_num}] {text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'─' * 70}{Colors.ENDC}")


def print_step(step, text):
    print(f"\n{Colors.YELLOW}[{step}] {text}{Colors.ENDC}")


def print_success(text):
    print(f"{Colors.GREEN}[OK] {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.RED}[ERROR] {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.YELLOW}[WARNING] {text}{Colors.ENDC}")


def run_command(cmd, description, cwd=None, check=True, realtime=False):
    """Ejecuta un comando y muestra el resultado."""
    print(f"\n  Ejecutando: {' '.join(cmd[:5])}...")

    try:
        if realtime:
            # Ejecutar con output en tiempo real
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            for line in process.stdout:
                print(f"  {line}", end='')
                output_lines.append(line)

            process.wait()

            if process.returncode == 0:
                print_success(description)
                return True, ''.join(output_lines)
            else:
                print_error(f"{description} - codigo de retorno: {process.returncode}")
                return False, ''.join(output_lines)
        else:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=check
            )

            if result.returncode == 0:
                print_success(description)
                return True, result.stdout
            else:
                print_error(f"{description} - codigo de retorno: {result.returncode}")
                if result.stdout:
                    print(f"\n  --- Output del script ---")
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-50:]:
                        print(f"  {line}")
                    print(f"  --- Fin output ---\n")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:500]}")
                return False, result.stderr

    except subprocess.CalledProcessError as e:
        print_error(f"{description} - excepcion: {e}")
        return False, str(e)
    except FileNotFoundError as e:
        print_error(f"Comando no encontrado: {e}")
        return False, str(e)


def get_dataset_name(train_json_path):
    """Extrae un nombre identificador del dataset basado en el path."""
    path = Path(train_json_path)
    parent = path.parent.name
    if parent.startswith('split_'):
        return parent.replace('split_', '')
    return parent


def check_duplicate_images_across_splits(json_paths: list, base_dir: Path) -> dict:
    """
    Detecta imagenes duplicadas entre diferentes splits (train/val/test).

    Args:
        json_paths: Lista de paths a los archivos JSON de cada split
        base_dir: Directorio base del proyecto

    Returns:
        dict con resultados: {'duplicates': [...], 'total_checked': int, 'passed': bool}
    """
    from collections import defaultdict

    image_to_splits = defaultdict(list)
    split_names = []

    for json_path in json_paths:
        full_path = base_dir / json_path
        if not full_path.exists():
            continue

        split_name = Path(json_path).stem  # train, val, test
        split_names.append(split_name)

        with open(full_path, 'r') as f:
            data = json.load(f)

        for img in data.get('images', []):
            file_name = img['file_name']
            image_to_splits[file_name].append(split_name)

    # Encontrar duplicados (imagenes que aparecen en mas de un split)
    duplicates = []
    for file_name, splits in image_to_splits.items():
        if len(splits) > 1:
            duplicates.append({
                'file_name': file_name,
                'splits': splits
            })

    return {
        'duplicates': duplicates,
        'total_checked': len(image_to_splits),
        'splits_checked': split_names,
        'passed': len(duplicates) == 0
    }


def find_image_in_dirs(file_name: str, image_dirs: list) -> Path:
    """
    Busca una imagen en una lista de directorios.

    Args:
        file_name: Nombre del archivo de imagen
        image_dirs: Lista de directorios donde buscar

    Returns:
        Path al archivo encontrado, o None si no se encuentra
    """
    # Extraer solo el nombre base en caso de que file_name tenga subdirectorios
    base_name = os.path.basename(file_name)

    for img_dir in image_dirs:
        # Intentar primero con el path completo tal como viene en el JSON
        full_path = img_dir / file_name
        if full_path.exists():
            return full_path

        # Intentar solo con el nombre base
        base_path = img_dir / base_name
        if base_path.exists():
            return base_path

    return None


def check_corrupted_images(json_path: Path, image_dirs: list, num_samples: int = None) -> dict:
    """
    Detecta imagenes corruptas que no se pueden abrir o leer.

    Args:
        json_path: Path al archivo JSON con anotaciones
        image_dirs: Lista de directorios de imagenes donde buscar
        num_samples: Numero de imagenes a verificar (None = todas)

    Returns:
        dict con resultados: {'corrupted': [...], 'total_checked': int, 'passed': bool}
    """
    from PIL import Image
    import random

    if not json_path.exists():
        return {'corrupted': [], 'total_checked': 0, 'passed': False, 'error': 'JSON no existe'}

    with open(json_path, 'r') as f:
        data = json.load(f)

    images = data.get('images', [])

    # Limitar muestras si se especifica
    if num_samples and len(images) > num_samples:
        images = random.sample(images, num_samples)

    corrupted = []
    checked = 0

    for img_info in images:
        file_name = img_info['file_name']
        img_path = find_image_in_dirs(file_name, image_dirs)
        checked += 1

        if img_path is None:
            corrupted.append({
                'file_name': file_name,
                'error': f'Archivo no existe en ninguno de los directorios: {[str(d) for d in image_dirs]}'
            })
            continue

        try:
            # Intentar abrir y verificar la imagen
            with Image.open(img_path) as img:
                # Forzar carga completa para detectar corrupcion
                img.load()

                # Verificar que las dimensiones coincidan con el JSON
                if 'width' in img_info and 'height' in img_info:
                    if img.size[0] != img_info['width'] or img.size[1] != img_info['height']:
                        corrupted.append({
                            'file_name': file_name,
                            'error': f"Dimension mismatch: JSON={img_info['width']}x{img_info['height']}, Real={img.size[0]}x{img.size[1]}"
                        })
        except Exception as e:
            corrupted.append({
                'file_name': file_name,
                'error': str(e)
            })

    return {
        'corrupted': corrupted,
        'total_checked': checked,
        'passed': len(corrupted) == 0
    }


def generate_overfit_config(args, base_dir, overfit_json_path):
    """Genera configuracion dinamica para test de overfitting."""

    dataset_name = get_dataset_name(args.train_json)

    # Paths relativos desde configs/rtdetrv2/generated/
    config = {
        '__include__': [
            '../../runtime.yml',
            '../include/rtdetrv2_r50vd.yml',
        ],
        'output_dir': f'./output/{dataset_name}_overfit_test',
        'task': 'detection',
        'evaluator': {
            'type': 'CocoEvaluator',
            'iou_types': ['bbox']
        },
        'num_classes': args.num_classes,
        'remap_mscoco_category': False,

        # Modelo
        'PResNet': {
            'depth': 50,
            'variant': 'd',
            'freeze_at': 0,
            'freeze_norm': False,
            'pretrained': True,
            'return_idx': [1, 2, 3],
            'num_stages': 4
        },
        'HybridEncoder': {
            'in_channels': [512, 1024, 2048],
            'hidden_dim': 256,
            'expansion': 1.0
        },
        'RTDETRTransformerv2': {
            'num_layers': 6,
            'num_queries': 150,
            'num_denoising': 100,
            'label_noise_ratio': 0.3,
            'box_noise_scale': 0.7,
            'num_points': [4, 4, 4],
            'cross_attn_method': 'default',
            'query_select_method': 'default'
        },
        'RTDETRPostProcessor': {
            'num_top_queries': 150
        },
        'RTDETRCriterionv2': {
            'weight_dict': {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
            'losses': ['vfl', 'boxes'],
            'alpha': 0.75,
            'gamma': 0.5,
            'matcher': {
                'type': 'HungarianMatcher',
                'weight_dict': {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                'alpha': 0.25,
                'gamma': 0.5
            }
        },

        # Entrenamiento
        'epoches': 50 if not args.quick else 20,
        'optimizer': {
            'type': 'AdamW',
            'lr': 0.0005,
            'betas': [0.9, 0.999],
            'weight_decay': 0.0001
        },
        'lr_scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': 50 if not args.quick else 20,
            'eta_min': 0.000001
        },
        'lr_warmup_scheduler': {
            'type': 'LinearWarmup',
            'warmup_duration': 100
        },
        'use_amp': True,
        'use_ema': False,
        'clip_max_norm': 1.0,
        'use_focal_loss': True,
        'early_stopping_patience': None,
        'print_freq': 10,
        'checkpoint_freq': 10,

        # DataLoaders
        'train_dataloader': {
            'type': 'DataLoader',
            'dataset': {
                'type': 'CocoDetection',
                'img_folder': f'./{args.image_dir[0]}',  # Usar primer directorio
                'ann_file': f'./{overfit_json_path}',
                'return_masks': False,
                'remap_mscoco_category': False,
                'transforms': {
                    'type': 'Compose',
                    'ops': [
                        {'type': 'ToImage'},
                        {'type': 'Resize', 'size': [640, 640]},
                        {'type': 'ToDtype', 'dtype': 'torch.float32', 'scale': True},
                        {'type': 'SanitizeBoundingBoxes', 'min_size': 1},
                        {'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True}
                    ]
                }
            },
            'shuffle': True,
            'total_batch_size': 16,
            'num_workers': 8,
            'drop_last': True,
            'collate_fn': {'type': 'BatchImageCollateFuncion'}
        },
        'val_dataloader': {
            'type': 'DataLoader',
            'dataset': {
                'type': 'CocoDetection',
                'img_folder': f'./{args.image_dir[0]}',  # Usar primer directorio
                'ann_file': f'./{overfit_json_path}',
                'return_masks': False,
                'remap_mscoco_category': False,
                'transforms': {
                    'type': 'Compose',
                    'ops': [
                        {'type': 'ToImage'},
                        {'type': 'Resize', 'size': [640, 640]},
                        {'type': 'ToDtype', 'dtype': 'torch.float32', 'scale': True},
                        {'type': 'SanitizeBoundingBoxes', 'min_size': 1},
                        {'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True}
                    ]
                }
            },
            'shuffle': False,
            'total_batch_size': 8,
            'num_workers': 4,
            'drop_last': False,
            'collate_fn': {'type': 'BatchImageCollateFuncion'}
        }
    }

    # Guardar config
    config_dir = base_dir / 'configs' / 'rtdetrv2' / 'generated'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f'{dataset_name}_overfit_test.yml'

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Config generada: {config_path}")
    return config_path


def generate_convergence_config(args, base_dir):
    """Genera configuracion dinamica para test de convergencia."""

    dataset_name = get_dataset_name(args.train_json)

    # Paths relativos desde configs/rtdetrv2/generated/
    config = {
        '__include__': [
            '../../runtime.yml',
            '../include/rtdetrv2_r50vd.yml',
        ],
        'output_dir': f'./output/{dataset_name}_convergence_test',
        'task': 'detection',
        'evaluator': {
            'type': 'CocoEvaluator',
            'iou_types': ['bbox']
        },
        'num_classes': args.num_classes,
        'remap_mscoco_category': False,

        # Modelo
        'PResNet': {
            'depth': 50,
            'variant': 'd',
            'freeze_at': 2,
            'freeze_norm': True,
            'pretrained': True,
            'return_idx': [1, 2, 3],
            'num_stages': 4
        },
        'HybridEncoder': {
            'in_channels': [512, 1024, 2048],
            'hidden_dim': 256,
            'expansion': 1.0
        },
        'RTDETRTransformerv2': {
            'num_layers': 6,
            'num_queries': 150,
            'num_denoising': 100,
            'label_noise_ratio': 0.3,
            'box_noise_scale': 0.7,
            'num_points': [4, 4, 4],
            'cross_attn_method': 'default',
            'query_select_method': 'default'
        },
        'RTDETRPostProcessor': {
            'num_top_queries': 150
        },
        'RTDETRCriterionv2': {
            'weight_dict': {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
            'losses': ['vfl', 'boxes'],
            'alpha': 0.75,
            'gamma': 2.0,
            'matcher': {
                'type': 'HungarianMatcher',
                'weight_dict': {'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
                'alpha': 0.25,
                'gamma': 2.0
            }
        },

        # Entrenamiento
        'epoches': 20 if not args.quick else 10,
        'optimizer': {
            'type': 'AdamW',
            'params': [
                {'params': '^(?=.*backbone)(?!.*norm).*$', 'lr': 0.000005},
                {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$', 'weight_decay': 0.0}
            ],
            'lr': 0.00005,
            'betas': [0.9, 0.999],
            'weight_decay': 0.05
        },
        'lr_scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': 20 if not args.quick else 10,
            'eta_min': 0.000001
        },
        'lr_warmup_scheduler': {
            'type': 'LinearWarmup',
            'warmup_duration': 500
        },
        'use_amp': True,
        'use_ema': True,
        'ema': {
            'type': 'ModelEMA',
            'decay': 0.9998,
            'warmups': 500
        },
        'clip_max_norm': 1.0,
        'use_focal_loss': True,
        'early_stopping_patience': None,
        'print_freq': 50,
        'checkpoint_freq': 5,

        # DataLoaders
        'train_dataloader': {
            'type': 'DataLoader',
            'dataset': {
                'type': 'CocoDetection',
                'img_folder': f'./{args.image_dir[0]}',  # Usar primer directorio
                'ann_file': f'./{args.train_json}',
                'return_masks': False,
                'remap_mscoco_category': False,
                'transforms': {
                    'type': 'Compose',
                    'ops': [
                        {'type': 'ToImage'},
                        {'type': 'RandomHorizontalFlip', 'p': 0.5},
                        {'type': 'RandomRotation', 'degrees': 10, 'p': 0.3},
                        {'type': 'RandomZoomOut', 'fill': 0, 'p': 0.2},
                        {'type': 'Resize', 'size': [640, 640]},
                        {'type': 'ToDtype', 'dtype': 'torch.float32', 'scale': True},
                        {'type': 'RandomPhotometricDistort', 'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.05, 'p': 0.4},
                        {'type': 'RandomGaussianBlur', 'kernel_size': 3, 'sigma': [0.1, 1.5], 'p': 0.2},
                        {'type': 'SanitizeBoundingBoxes', 'min_size': 1},
                        {'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True}
                    ]
                }
            },
            'shuffle': True,
            'total_batch_size': 32,
            'num_workers': 12,
            'drop_last': True,
            'collate_fn': {'type': 'BatchImageCollateFuncion', 'scales': [640]}
        },
        'val_dataloader': {
            'type': 'DataLoader',
            'dataset': {
                'type': 'CocoDetection',
                'img_folder': f'./{args.image_dir[0]}',  # Usar primer directorio
                'ann_file': f'./{args.val_json}',
                'return_masks': False,
                'remap_mscoco_category': False,
                'transforms': {
                    'type': 'Compose',
                    'ops': [
                        {'type': 'ToImage'},
                        {'type': 'Resize', 'size': [640, 640]},
                        {'type': 'ToDtype', 'dtype': 'torch.float32', 'scale': True},
                        {'type': 'SanitizeBoundingBoxes', 'min_size': 1},
                        {'type': 'ConvertBoxes', 'fmt': 'cxcywh', 'normalize': True}
                    ]
                }
            },
            'shuffle': False,
            'total_batch_size': 16,
            'num_workers': 8,
            'drop_last': False,
            'collate_fn': {'type': 'BatchImageCollateFuncion'}
        }
    }

    # Guardar config
    config_dir = base_dir / 'configs' / 'rtdetrv2' / 'generated'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f'{dataset_name}_convergence_test.yml'

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  Config generada: {config_path}")
    return config_path


def phase1_integrity(args, base_dir):
    """FASE 1: Validacion de Integridad."""
    print_phase(1, "VALIDACION DE INTEGRIDAD")
    results = {'passed': True, 'steps': {}}

    # 1.1 Verificar indexacion de categorias
    print_step("1.1", "Verificando indexacion de categorias")

    train_json = base_dir / args.train_json
    val_json = base_dir / args.val_json

    if not train_json.exists():
        print_error(f"No existe: {train_json}")
        results['passed'] = False
        results['steps']['1.1'] = False
    else:
        with open(train_json, 'r') as f:
            data = json.load(f)

        categories = data.get('categories', [])
        cat_ids = [c['id'] for c in categories]

        # Detectar numero de clases automaticamente si no se especifico
        if args.num_classes is None:
            args.num_classes = len(categories)
            print(f"  Detectadas {args.num_classes} clases automaticamente")

        if 0 in cat_ids:
            print_success(f"Categorias 0-indexed: {cat_ids[:10]}{'...' if len(cat_ids) > 10 else ''}")
            results['steps']['1.1'] = True
        else:
            print_warning(f"Categorias NO son 0-indexed: {cat_ids[:10]}")
            print("  Ejecutar: python tools/convert_to_0indexed.py si es necesario")
            results['steps']['1.1'] = False
            results['passed'] = False

    # 1.2 Validar formato COCO e imagenes
    print_step("1.2", "Validando formato COCO e imagenes")

    # Construir lista de directorios de imagenes
    image_dirs = [base_dir / img_dir for img_dir in args.image_dir]

    if train_json.exists():
        with open(train_json, 'r') as f:
            data = json.load(f)

        images = data.get('images', [])
        annotations = data.get('annotations', [])

        print(f"  Imagenes en JSON: {len(images)}")
        print(f"  Anotaciones: {len(annotations)}")
        if len(image_dirs) > 1:
            print(f"  Buscando en {len(image_dirs)} directorios")

        # Verificar que las imagenes existen (buscando en todos los directorios)
        missing = 0
        missing_files = []
        sample_check = min(100, len(images))

        for img in images[:sample_check]:
            img_path = find_image_in_dirs(img['file_name'], image_dirs)
            if img_path is None:
                missing += 1
                if len(missing_files) < 5:
                    missing_files.append(img['file_name'])

        if missing == 0:
            print_success(f"Verificadas {sample_check} imagenes - todas existen")
            results['steps']['1.2'] = True
        else:
            print_error(f"{missing}/{sample_check} imagenes no encontradas")
            for mf in missing_files:
                print(f"    - {mf}")
            if missing > 5:
                print(f"    ... y {missing - 5} mas")
            results['steps']['1.2'] = False
            results['passed'] = False
    else:
        results['steps']['1.2'] = False
        results['passed'] = False

    # 1.3 Validar carga en pipeline
    print_step("1.3", "Validando carga en pipeline RT-DETR")

    if not args.skip_pipeline:
        # Para el pipeline, usar el primer directorio (el principal)
        # El pipeline de RT-DETR no soporta multiples directorios nativamente
        cmd = [
            sys.executable, 'tools/validate_dataset_pipeline.py',
            '-c', str(args.config_pipeline),
            '--num-samples', str(50 if args.quick else 100),
            '--split', 'train',
            '--train-json', str(args.train_json),
            '--image-dir', str(args.image_dir[0])  # Usar primer directorio
        ]

        if len(args.image_dir) > 1:
            print_warning(f"Pipeline usa solo el primer directorio: {args.image_dir[0]}")

        success, output = run_command(
            cmd,
            "Validacion de pipeline",
            cwd=str(base_dir),
            check=False
        )
        results['steps']['1.3'] = success
        if not success:
            results['passed'] = False
    else:
        print("  (Saltado con --skip-pipeline)")
        results['steps']['1.3'] = None

    # 1.4 Detectar imagenes duplicadas entre splits
    print_step("1.4", "Detectando imagenes duplicadas entre splits")

    # Construir lista de JSONs a verificar
    json_paths = [args.train_json, args.val_json]

    # Intentar detectar test.json automaticamente
    train_path = Path(args.train_json)
    test_json_path = train_path.parent / 'test.json'
    if (base_dir / test_json_path).exists():
        json_paths.append(str(test_json_path))
        print(f"  Verificando splits: train, val, test")
    else:
        print(f"  Verificando splits: train, val (no se encontro test.json)")

    dup_results = check_duplicate_images_across_splits(json_paths, base_dir)

    if dup_results['passed']:
        print_success(f"No se encontraron duplicados en {dup_results['total_checked']} imagenes")
        results['steps']['1.4'] = True
    else:
        num_dups = len(dup_results['duplicates'])
        print_error(f"Se encontraron {num_dups} imagenes duplicadas entre splits!")

        # Mostrar algunos ejemplos
        for dup in dup_results['duplicates'][:5]:
            print(f"    - {dup['file_name']} aparece en: {', '.join(dup['splits'])}")
        if num_dups > 5:
            print(f"    ... y {num_dups - 5} mas")

        print_warning("IMPORTANTE: Las imagenes duplicadas entre splits causan data leakage!")
        print("  Ejecutar herramienta de split para regenerar train/val/test sin duplicados")
        results['steps']['1.4'] = False
        results['passed'] = False

    # 1.5 Detectar imagenes corruptas
    print_step("1.5", "Detectando imagenes corruptas")

    train_json = base_dir / args.train_json
    # image_dirs ya esta definido arriba en paso 1.2

    # En modo quick, verificar menos imagenes
    num_samples = 200 if args.quick else None  # None = todas

    corrupt_results = check_corrupted_images(train_json, image_dirs, num_samples)

    if corrupt_results['passed']:
        print_success(f"Verificadas {corrupt_results['total_checked']} imagenes - ninguna corrupta")
        results['steps']['1.5'] = True
    else:
        num_corrupt = len(corrupt_results['corrupted'])
        print_error(f"Se encontraron {num_corrupt} imagenes corruptas o con problemas!")

        # Mostrar algunos ejemplos
        for corrupt in corrupt_results['corrupted'][:10]:
            print(f"    - {corrupt['file_name']}: {corrupt['error']}")
        if num_corrupt > 10:
            print(f"    ... y {num_corrupt - 10} mas")

        print_warning("Las imagenes corruptas causaran errores durante el entrenamiento")
        print("  Regenerar o eliminar las imagenes problematicas antes de entrenar")
        results['steps']['1.5'] = False
        results['passed'] = False

    return results


def phase2_quality(args, base_dir):
    """FASE 2: Validacion de Calidad."""
    print_phase(2, "VALIDACION DE CALIDAD DE ANOTACIONES")
    results = {'passed': True, 'steps': {}}

    dataset_name = get_dataset_name(args.train_json)
    output_dir = base_dir / 'validation_output' / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 Analisis estadistico
    print_step("2.1", "Analisis estadistico del dataset")

    train_json = base_dir / args.train_json
    if train_json.exists():
        with open(train_json, 'r') as f:
            data = json.load(f)

        annotations = data.get('annotations', [])
        images = data.get('images', [])
        categories = {c['id']: c['name'] for c in data.get('categories', [])}

        from collections import Counter
        class_counts = Counter(ann['category_id'] for ann in annotations)

        print(f"\n  Imagenes: {len(images)}")
        print(f"  Anotaciones: {len(annotations)}")
        print(f"  Objetos/imagen: {len(annotations)/len(images):.2f}")
        print(f"\n  Distribucion de clases (top 10):")
        for cat_id, count in class_counts.most_common(10):
            cat_name = categories.get(cat_id, f'Unknown_{cat_id}')
            pct = 100 * count / len(annotations)
            print(f"    {cat_name}: {count} ({pct:.1f}%)")

        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n  Ratio de desbalance: {imbalance:.1f}:1")

        if imbalance > 10:
            print_warning(f"Alto desbalance de clases ({imbalance:.0f}:1)")
        else:
            print_success("Distribucion de clases aceptable")

        results['steps']['2.1'] = True
    else:
        results['steps']['2.1'] = False

    # 2.2 Detectar padding en bboxes
    print_step("2.2", "Detectando padding en bounding boxes")

    if not args.skip_bbox_check:
        train_json_abs = base_dir / args.train_json
        # Usar el primer directorio de imagenes para bbox check
        image_dir_abs = base_dir / args.image_dir[0]

        if not train_json_abs.exists():
            print_error(f"JSON no encontrado: {train_json_abs}")
            results['steps']['2.2'] = False
        elif not image_dir_abs.exists():
            print_error(f"Directorio de imagenes no encontrado: {image_dir_abs}")
            results['steps']['2.2'] = False
        else:
            if len(args.image_dir) > 1:
                print_warning(f"Bbox check usa solo el primer directorio: {args.image_dir[0]}")

            cmd = [
                sys.executable, 'tools/validate_bbox_tightness.py',
                '--json', str(train_json_abs),
                '--image-dir', str(image_dir_abs),
                '--num-samples', str(200 if args.quick else 500),
                '--output-dir', str(output_dir / 'bbox_tightness')
            ]

            success, output = run_command(
                cmd,
                "Analisis de padding en bboxes",
                cwd=str(base_dir),
                check=False
            )
            results['steps']['2.2'] = success
    else:
        print("  (Saltado con --skip-bbox-check)")
        results['steps']['2.2'] = None

    # 2.3 Auditoria visual
    print_step("2.3", "Generando muestras visuales")

    vis_dir = output_dir / 'visual_samples'
    vis_dir.mkdir(exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
        import random

        train_json = base_dir / args.train_json
        # Construir lista de directorios de imagenes
        image_dirs = [base_dir / img_dir for img_dir in args.image_dir]

        with open(train_json, 'r') as f:
            data = json.load(f)

        images = {img['id']: img for img in data['images']}
        categories = {c['id']: c['name'] for c in data['categories']}

        from collections import defaultdict
        anns_by_img = defaultdict(list)
        for ann in data['annotations']:
            anns_by_img[ann['image_id']].append(ann)

        num_samples = 10 if args.quick else 20
        sample_ids = random.sample(list(anns_by_img.keys()), min(num_samples, len(anns_by_img)))

        colors = plt.cm.tab20.colors

        for i, img_id in enumerate(sample_ids):
            img_info = images[img_id]
            # Buscar imagen en todos los directorios
            img_path = find_image_in_dirs(img_info['file_name'], image_dirs)

            if img_path is None:
                continue

            img = Image.open(img_path)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(img)

            for ann in anns_by_img[img_id]:
                x, y, w, h = ann['bbox']
                cat_id = ann['category_id']
                color = colors[cat_id % len(colors)]

                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)

                cat_name = categories.get(cat_id, f'Class_{cat_id}')
                ax.text(x, y - 5, cat_name, fontsize=8, color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.set_title(f"Sample {i+1}: {img_info['file_name']}")
            ax.axis('off')

            plt.tight_layout()
            plt.savefig(vis_dir / f'sample_{i:03d}.png', dpi=100, bbox_inches='tight')
            plt.close()

        print_success(f"Generadas {len(sample_ids)} visualizaciones en {vis_dir}")
        results['steps']['2.3'] = True

    except Exception as e:
        print_error(f"Error generando visualizaciones: {e}")
        results['steps']['2.3'] = False

    # 2.4 Comparacion de calidad con dataset real
    print_step("2.4", "Comparacion de calidad con dataset real")

    if args.real_json:
        real_json_abs = base_dir / args.real_json
        train_json_abs = base_dir / args.train_json

        if not real_json_abs.exists():
            print_error(f"JSON real no encontrado: {real_json_abs}")
            results['steps']['2.4'] = False
        else:
            cmd = [
                sys.executable, 'tools/validate_synthetic_quality.py',
                '--synthetic-ann', str(train_json_abs),
                '--real-ann', str(real_json_abs),
                '--output-dir', str(output_dir / 'quality_comparison')
            ]

            # Añadir directorios de imagenes si existen (usar el primero)
            if args.image_dir:
                img_dir_abs = base_dir / args.image_dir[0]
                if img_dir_abs.exists():
                    cmd.extend(['--synthetic-img', str(img_dir_abs)])

            if args.real_image_dir:
                # Usar el primer directorio de imagenes reales
                real_img_dir_abs = base_dir / args.real_image_dir[0]
                if real_img_dir_abs.exists():
                    cmd.extend(['--real-img', str(real_img_dir_abs)])
                if len(args.real_image_dir) > 1:
                    print_warning(f"Quality comparison usa solo el primer directorio real: {args.real_image_dir[0]}")

            success, output = run_command(
                cmd,
                "Comparacion de calidad sintetico vs real",
                cwd=str(base_dir),
                check=False
            )
            results['steps']['2.4'] = success

            if success:
                print(f"  Reporte guardado en: {output_dir / 'quality_comparison'}")
    else:
        print("  (Saltado - no se proporciono --real-json)")
        results['steps']['2.4'] = None

    # 2.5 Analisis de domain gap
    print_step("2.5", "Analisis de domain gap (estadisticas de imagen)")

    if args.real_image_dir:
        # Usar el primer directorio de cada tipo
        real_img_dir_abs = base_dir / args.real_image_dir[0]
        synth_img_dir_abs = base_dir / args.image_dir[0]

        if not real_img_dir_abs.exists():
            print_error(f"Directorio de imagenes reales no encontrado: {real_img_dir_abs}")
            results['steps']['2.5'] = False
        elif not synth_img_dir_abs.exists():
            print_error(f"Directorio de imagenes sinteticas no encontrado: {synth_img_dir_abs}")
            results['steps']['2.5'] = False
        else:
            if len(args.image_dir) > 1:
                print_warning(f"Domain gap usa solo el primer directorio sintetico: {args.image_dir[0]}")
            if len(args.real_image_dir) > 1:
                print_warning(f"Domain gap usa solo el primer directorio real: {args.real_image_dir[0]}")

            num_samples = 200 if args.quick else 500

            cmd = [
                sys.executable, 'tools/analyze_domain_gap.py',
                '--synthetic-dir', str(synth_img_dir_abs),
                '--real-dir', str(real_img_dir_abs),
                '--output-dir', str(output_dir / 'domain_gap_analysis'),
                '--num-samples', str(num_samples)
            ]

            success, output = run_command(
                cmd,
                "Analisis de domain gap",
                cwd=str(base_dir),
                check=False
            )
            results['steps']['2.5'] = success

            if success:
                print(f"  Reporte guardado en: {output_dir / 'domain_gap_analysis'}")

                # Mostrar resumen del domain gap si esta disponible
                gap_summary_path = output_dir / 'domain_gap_analysis' / 'domain_gap_summary.json'
                if gap_summary_path.exists():
                    try:
                        with open(gap_summary_path, 'r') as f:
                            gap_data = json.load(f)
                        if 'overall_gap_score' in gap_data:
                            score = gap_data['overall_gap_score']
                            if score < 0.3:
                                print_success(f"Domain gap bajo: {score:.2f} (< 0.3)")
                            elif score < 0.6:
                                print_warning(f"Domain gap moderado: {score:.2f} (0.3 - 0.6)")
                            else:
                                print_error(f"Domain gap alto: {score:.2f} (> 0.6)")
                    except Exception as e:
                        pass  # Si no se puede leer el resumen, continuar
    else:
        print("  (Saltado - no se proporciono --real-image-dir)")
        results['steps']['2.5'] = None

    return results


def phase3_overfitting(args, base_dir):
    """FASE 3: Test de Entrenabilidad (Overfitting)."""
    print_phase(3, "TEST DE ENTRENABILIDAD (OVERFITTING)")
    results = {'passed': True, 'steps': {}}

    # 3.1 Crear subset de overfitting
    print_step("3.1", "Creando subset para test de overfitting")

    train_json_path = Path(args.train_json)
    split_dir = train_json_path.parent
    overfit_json_path = split_dir / 'overfit_test.json'

    # Usar path absoluto para el script
    train_json_abs = base_dir / args.train_json
    overfit_json_abs = base_dir / overfit_json_path

    cmd = [
        sys.executable, 'tools/test_synthetic_overfitting.py',
        '--input-json', str(train_json_abs),
        '--output-json', str(overfit_json_abs),
        '--num-images', str(50 if args.quick else 100),
        '--seed', '42'
    ]

    success, output = run_command(
        cmd,
        "Creacion de subset de overfitting",
        cwd=str(base_dir),
        check=False
    )
    results['steps']['3.1'] = success

    if not success:
        results['passed'] = False
        return results

    # 3.2 Entrenar test de overfitting
    print_step("3.2", "Ejecutando test de overfitting")

    # Generar config dinamica
    config_path = generate_overfit_config(args, base_dir, overfit_json_path)

    if args.skip_training:
        print("  (Saltado con --skip-training)")
        print(f"\n  Para ejecutar manualmente:")
        print(f"    python tools/train.py -c {config_path} -d cuda --seed 42")
        results['steps']['3.2'] = None
    else:
        epochs = 20 if args.quick else 50
        print(f"  Entrenando por {epochs} epocas...")
        print(f"  Criterios de exito:")
        print(f"    - Epoca 10: mAP@0.5 > 0.1")
        print(f"    - Epoca 30: mAP@0.5 > 0.5")
        print(f"    - Epoca 50: mAP@0.5 > 0.8")

        cmd = [
            sys.executable, 'tools/train.py',
            '-c', str(config_path),
            '-d', 'cuda',
            '--seed', '42'
        ]

        success, output = run_command(
            cmd,
            "Entrenamiento de overfitting",
            cwd=str(base_dir),
            check=False,
            realtime=True
        )
        results['steps']['3.2'] = success

        if not success:
            results['passed'] = False

    return results


def phase4_convergence(args, base_dir):
    """FASE 4: Test de Convergencia."""
    print_phase(4, "TEST DE CONVERGENCIA")
    results = {'passed': True, 'steps': {}}

    print_step("4.1", "Ejecutando test de convergencia")

    # Generar config dinamica
    config_path = generate_convergence_config(args, base_dir)

    if args.skip_training:
        print("  (Saltado con --skip-training)")
        print(f"\n  Para ejecutar manualmente:")
        print(f"    python tools/train.py -c {config_path} -d cuda --seed 42")
        results['steps']['4.1'] = None
    else:
        epochs = 10 if args.quick else 20
        print(f"  Entrenando por {epochs} epocas en dataset completo...")
        print(f"  Criterios de exito:")
        print(f"    - Epoca 5:  Loss bajando")
        print(f"    - Epoca 10: Loss < 12, mAP > 0.01")
        print(f"    - Epoca 20: Loss < 10, mAP > 0.05")

        cmd = [
            sys.executable, 'tools/train.py',
            '-c', str(config_path),
            '-d', 'cuda',
            '--seed', '42'
        ]

        success, output = run_command(
            cmd,
            "Entrenamiento de convergencia",
            cwd=str(base_dir),
            check=False,
            realtime=True
        )
        results['steps']['4.1'] = success

        if not success:
            results['passed'] = False

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validacion completa de dataset sintetico para RT-DETRv2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Validar y entrenar (todas las fases, un directorio de imagenes)
  python tools/run_synthetic_validation.py --all \\
      --train-json split_sintetico_taco/train.json \\
      --val-json split_sintetico_taco/val.json \\
      --image-dir dataset/dataset_sintetico_taco/images

  # Con MULTIPLES directorios de imagenes - Forma 1: todos en una linea
  python tools/run_synthetic_validation.py --all \\
      --train-json annotations.json \\
      --val-json annotations_val.json \\
      --image-dir dataset/images_part1 dataset/images_part2 dataset/images_part3

  # Con MULTIPLES directorios de imagenes - Forma 2: flag repetido
  python tools/run_synthetic_validation.py --all \\
      --train-json annotations.json \\
      --val-json annotations_val.json \\
      --image-dir dataset/images_part1 \\
      --image-dir dataset/images_part2 \\
      --image-dir dataset/images_part3

  # Validacion COMPLETA con comparacion vs dataset real
  python tools/run_synthetic_validation.py --all \\
      --train-json split_sintetico_taco/train.json \\
      --val-json split_sintetico_taco/val.json \\
      --image-dir dataset/dataset_sintetico_taco/images \\
      --real-json dataset/real/train.json \\
      --real-image-dir dataset/real/images

  # Con MULTIPLES directorios para imagenes reales (flag repetido)
  python tools/run_synthetic_validation.py --all \\
      --train-json split_sintetico/train.json \\
      --val-json split_sintetico/val.json \\
      --image-dir dataset/sintetico/images \\
      --real-json dataset/real/combined.json \\
      --real-image-dir dataset/real/cleansea \\
      --real-image-dir dataset/real/ocean_garbage \\
      --real-image-dir dataset/real/neural_ocean

  # Solo validacion (fases 1 y 2, sin entrenamiento)
  python tools/run_synthetic_validation.py --phases 1 2 \\
      --train-json split_sintetico_simple/train.json \\
      --val-json split_sintetico_simple/val.json \\
      --image-dir dataset/dataset_sintetico_simple/images

  # Modo rapido con entrenamiento
  python tools/run_synthetic_validation.py --all --quick \\
      --train-json split_sintetico_simple/train.json \\
      --val-json split_sintetico_simple/val.json \\
      --image-dir dataset/dataset_sintetico_simple/images

  # Generar configs sin ejecutar entrenamientos
  python tools/run_synthetic_validation.py --all --skip-training \\
      --train-json split_sintetico_taco/train.json \\
      --val-json split_sintetico_taco/val.json \\
      --image-dir dataset/dataset_sintetico_taco/images
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Ejecutar todas las fases')
    parser.add_argument('--phases', nargs='+', type=int, choices=[1, 2, 3, 4],
                        help='Fases especificas a ejecutar')
    parser.add_argument('--quick', action='store_true',
                        help='Modo rapido (menos muestras, menos epocas)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Saltar entrenamientos (solo generar configs)')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Saltar validacion de pipeline')
    parser.add_argument('--skip-bbox-check', action='store_true',
                        help='Saltar verificacion de padding en bboxes')

    # Paths dataset sintetico (obligatorios)
    parser.add_argument('--train-json', type=str, required=True,
                        help='Path al JSON de entrenamiento sintetico (relativo a base_dir)')
    parser.add_argument('--val-json', type=str, required=True,
                        help='Path al JSON de validacion sintetico (relativo a base_dir)')
    parser.add_argument('--image-dir', type=str, nargs='+', action='extend', required=True,
                        help='Directorio(s) de imagenes sinteticas. Se puede usar de dos formas: '
                             '1) --image-dir dir1 dir2 dir3 (todos en una linea), o '
                             '2) --image-dir dir1 --image-dir dir2 (flag repetido). '
                             'Las imagenes se buscan en orden en cada directorio (relativo a base_dir)')

    # Paths dataset real (opcionales, para comparacion)
    parser.add_argument('--real-json', type=str, default=None,
                        help='Path al JSON del dataset real para comparacion (opcional)')
    parser.add_argument('--real-image-dir', type=str, nargs='+', action='extend', default=None,
                        help='Directorio(s) de imagenes reales para analisis de domain gap. '
                             'Se puede usar de dos formas: '
                             '1) --real-image-dir dir1 dir2 (todos en una linea), o '
                             '2) --real-image-dir dir1 --real-image-dir dir2 (flag repetido). '
                             'Las imagenes se buscan en orden (opcional)')

    # Otros parametros
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Numero de clases (detectado automaticamente si no se especifica)')
    parser.add_argument('--config-pipeline', type=str,
                        default='configs/rtdetrv2/rtdetrv2_r50vd_synthetic_6cls.yml',
                        help='Config para validacion de pipeline')

    args = parser.parse_args()

    # Determinar fases a ejecutar
    if args.all:
        phases = [1, 2, 3, 4]
    elif args.phases:
        phases = args.phases
    else:
        phases = [1, 2]  # Por defecto solo validacion

    # Directorio base
    base_dir = Path(__file__).parent.parent.resolve()

    dataset_name = get_dataset_name(args.train_json)

    print_header(f"VALIDACION DE DATASET SINTETICO: {dataset_name}")
    print(f"Directorio base: {base_dir}")
    print(f"\n{Colors.BOLD}Dataset Sintetico:{Colors.ENDC}")
    print(f"  Train JSON: {args.train_json}")
    print(f"  Val JSON:   {args.val_json}")
    if len(args.image_dir) == 1:
        print(f"  Image dir:  {args.image_dir[0]}")
    else:
        print(f"  Image dirs ({len(args.image_dir)}):")
        for img_dir in args.image_dir:
            print(f"    - {img_dir}")

    if args.real_json or args.real_image_dir:
        print(f"\n{Colors.BOLD}Dataset Real (para comparacion):{Colors.ENDC}")
        print(f"  Real JSON:      {args.real_json or '(no proporcionado)'}")
        if args.real_image_dir:
            if len(args.real_image_dir) == 1:
                print(f"  Real Image dir: {args.real_image_dir[0]}")
            else:
                print(f"  Real Image dirs ({len(args.real_image_dir)}):")
                for img_dir in args.real_image_dir:
                    print(f"    - {img_dir}")
        else:
            print(f"  Real Image dir: (no proporcionado)")
    else:
        print(f"\n{Colors.YELLOW}Nota: No se proporciono dataset real. Fases 2.4 y 2.5 seran saltadas.{Colors.ENDC}")
        print(f"      Usa --real-json y --real-image-dir para habilitar comparacion.")

    print(f"\n{Colors.BOLD}Configuracion:{Colors.ENDC}")
    print(f"  Fases a ejecutar: {phases}")
    print(f"  Modo rapido: {'Si' if args.quick else 'No'}")
    print(f"  Ejecutar entrenamientos: {'No' if args.skip_training else 'Si'}")

    all_results = {}

    # Ejecutar fases
    if 1 in phases:
        all_results[1] = phase1_integrity(args, base_dir)

        if not all_results[1]['passed']:
            print_error("\nFASE 1 FALLIDA - Corregir errores antes de continuar")
            if 2 in phases or 3 in phases or 4 in phases:
                print_warning("Saltando fases restantes debido a errores de integridad")
                phases = [p for p in phases if p == 1]

    if 2 in phases:
        all_results[2] = phase2_quality(args, base_dir)

    if 3 in phases:
        all_results[3] = phase3_overfitting(args, base_dir)

    if 4 in phases:
        all_results[4] = phase4_convergence(args, base_dir)

    # Resumen final
    print_header("RESUMEN DE VALIDACION")

    for phase_num, results in sorted(all_results.items()):
        status = "PASS" if results['passed'] else "FAIL"
        color = Colors.GREEN if results['passed'] else Colors.RED
        print(f"\n{color}FASE {phase_num}: {status}{Colors.ENDC}")

        for step, passed in results['steps'].items():
            if passed is True:
                print(f"  [{Colors.GREEN}OK{Colors.ENDC}] Paso {step}")
            elif passed is False:
                print(f"  [{Colors.RED}FAIL{Colors.ENDC}] Paso {step}")
            else:
                print(f"  [{Colors.YELLOW}SKIP{Colors.ENDC}] Paso {step}")

    # Veredicto final
    all_passed = all(r['passed'] for r in all_results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}VALIDACION COMPLETADA EXITOSAMENTE{Colors.ENDC}")
        print(f"\nEl dataset {dataset_name} ha pasado las validaciones.")

        # Mostrar paths de output
        print(f"\nOutputs generados:")
        print(f"  - Visualizaciones: validation_output/{dataset_name}/")
        if 3 in phases or 4 in phases:
            print(f"  - Configs: configs/rtdetrv2/generated/")
            print(f"  - Checkpoints: output/{dataset_name}_*/")
    else:
        print(f"{Colors.RED}{Colors.BOLD}VALIDACION CON ERRORES{Colors.ENDC}")
        print("\nRevisar los errores reportados arriba y corregir antes de entrenar.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
