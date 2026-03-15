"""
Validador de Ajuste de Bounding Boxes (Detección de Padding).

Analiza los bounding boxes del dataset sintético para detectar padding excesivo,
es decir, espacio vacío entre el borde del objeto y el borde de la bbox.

Este es un problema crítico identificado históricamente en datasets sintéticos
que causa que el modelo aprenda a detectar el padding en lugar de los objetos.

Uso:
    python tools/validate_bbox_tightness.py \
        --json split_sintetico_v2/train.json \
        --image-dir dataset/dataset_sintetico_v2/images \
        --num-samples 500 \
        --output-dir validation_output/bbox_tightness
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import warnings

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore')

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("WARNING: OpenCV no disponible, usando método alternativo")


def load_coco_annotations(json_path):
    """Carga anotaciones COCO."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_bbox_region(image, bbox, margin=0):
    """Extrae la región de una imagen correspondiente a un bbox.

    Args:
        image: numpy array [H, W, C]
        bbox: [x, y, w, h] en formato COCO
        margin: margen adicional a extraer alrededor del bbox

    Returns:
        region: numpy array de la región extraída
    """
    x, y, w, h = [int(v) for v in bbox]
    H, W = image.shape[:2]

    # Aplicar margen
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(W, x + w + margin)
    y2 = min(H, y + h + margin)

    return image[y1:y2, x1:x2]


def analyze_edge_content(region, edge_width=5):
    """Analiza el contenido de los bordes de una región.

    Detecta si los bordes contienen contenido similar al fondo (padding)
    o contenido diferente (objeto bien ajustado).

    Args:
        region: numpy array [H, W, C]
        edge_width: ancho de los bordes a analizar en píxeles

    Returns:
        dict con métricas de análisis
    """
    if region.size == 0:
        return {'valid': False}

    h, w = region.shape[:2]

    if h < edge_width * 3 or w < edge_width * 3:
        # Región muy pequeña para analizar
        return {'valid': False, 'reason': 'too_small'}

    # Convertir a escala de grises si es necesario
    if len(region.shape) == 3:
        gray = np.mean(region, axis=2)
    else:
        gray = region

    # Extraer bordes y centro
    top_edge = gray[:edge_width, :]
    bottom_edge = gray[-edge_width:, :]
    left_edge = gray[:, :edge_width]
    right_edge = gray[:, -edge_width:]

    center = gray[edge_width:-edge_width, edge_width:-edge_width]

    if center.size == 0:
        return {'valid': False, 'reason': 'no_center'}

    # Calcular estadísticas
    edge_pixels = np.concatenate([
        top_edge.flatten(),
        bottom_edge.flatten(),
        left_edge.flatten(),
        right_edge.flatten()
    ])

    edge_mean = np.mean(edge_pixels)
    edge_std = np.std(edge_pixels)
    center_mean = np.mean(center)
    center_std = np.std(center)

    # Calcular contraste borde-centro
    edge_center_diff = abs(edge_mean - center_mean)

    # Calcular varianza en los bordes (bordes uniformes = padding)
    edge_uniformity = 1.0 / (edge_std + 1e-6)  # Mayor valor = más uniforme

    # Detectar si los bordes son muy uniformes (posible padding)
    is_edge_uniform = edge_std < 20  # Umbral empírico

    # Calcular gradiente en los bordes (objetos reales tienen gradientes)
    if HAS_CV2:
        sobel_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        edge_gradient = np.mean([
            gradient_magnitude[:edge_width, :].mean(),
            gradient_magnitude[-edge_width:, :].mean(),
            gradient_magnitude[:, :edge_width].mean(),
            gradient_magnitude[:, -edge_width:].mean()
        ])
    else:
        # Método alternativo sin OpenCV
        edge_gradient = edge_std * 2  # Aproximación

    return {
        'valid': True,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'center_mean': center_mean,
        'center_std': center_std,
        'edge_center_diff': edge_center_diff,
        'edge_uniformity': edge_uniformity,
        'is_edge_uniform': is_edge_uniform,
        'edge_gradient': edge_gradient,
        'region_size': (h, w)
    }


def calculate_padding_score(metrics):
    """Calcula un score de padding (0-1, mayor = más probable padding).

    Args:
        metrics: dict de métricas del análisis de bordes

    Returns:
        float: score de padding [0, 1]
    """
    if not metrics.get('valid', False):
        return None

    score = 0.0

    # Factor 1: Bordes uniformes (poco contraste interno)
    if metrics['edge_std'] < 15:
        score += 0.3
    elif metrics['edge_std'] < 25:
        score += 0.15

    # Factor 2: Poco contraste borde-centro
    if metrics['edge_center_diff'] < 20:
        score += 0.3
    elif metrics['edge_center_diff'] < 40:
        score += 0.15

    # Factor 3: Bajo gradiente en bordes
    if metrics['edge_gradient'] < 10:
        score += 0.4
    elif metrics['edge_gradient'] < 25:
        score += 0.2

    return min(1.0, score)


def visualize_suspicious_bbox(image, bbox, metrics, output_path):
    """Genera visualización de un bbox sospechoso."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Imagen completa con bbox
        axes[0].imshow(image)
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title('Imagen con BBox')
        axes[0].axis('off')

        # Región extraída
        region = extract_bbox_region(image, bbox)
        axes[1].imshow(region)
        axes[1].set_title(f'Región BBox\n{region.shape[1]}x{region.shape[0]}')
        axes[1].axis('off')

        # Métricas
        metrics_text = f"""
Padding Score: {metrics.get('score', 'N/A'):.3f}
Edge Std: {metrics.get('edge_std', 'N/A'):.1f}
Center Std: {metrics.get('center_std', 'N/A'):.1f}
Edge-Center Diff: {metrics.get('edge_center_diff', 'N/A'):.1f}
Edge Gradient: {metrics.get('edge_gradient', 'N/A'):.1f}
"""
        axes[2].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                     verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_title('Métricas de Análisis')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"  Error generando visualización: {e}")


def main():
    parser = argparse.ArgumentParser(description='Validar ajuste de bounding boxes (detectar padding)')
    parser.add_argument('--json', type=str, required=True,
                        help='Archivo de anotaciones COCO JSON')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='Directorio de imágenes')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Número de bboxes a analizar')
    parser.add_argument('--output-dir', type=str, default='validation_output/bbox_tightness',
                        help='Directorio de salida para visualizaciones')
    parser.add_argument('--padding-threshold', type=float, default=0.6,
                        help='Umbral de score para considerar padding sospechoso')
    parser.add_argument('--edge-width', type=int, default=5,
                        help='Ancho de borde a analizar en píxeles')
    parser.add_argument('--visualize-suspicious', type=int, default=20,
                        help='Número de bboxes sospechosos a visualizar')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para reproducibilidad')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 80)
    print("VALIDADOR DE AJUSTE DE BOUNDING BOXES (DETECCIÓN DE PADDING)")
    print("=" * 80)
    print(f"\nArchivo JSON: {args.json}")
    print(f"Directorio de imágenes: {args.image_dir}")
    print(f"Muestras a analizar: {args.num_samples}")
    print(f"Umbral de padding: {args.padding_threshold}")

    # Cargar datos
    data = load_coco_annotations(args.json)
    images_dict = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']

    print(f"\nDataset: {len(images_dict)} imágenes, {len(annotations)} anotaciones")

    # Seleccionar muestras aleatorias
    num_to_analyze = min(args.num_samples, len(annotations))
    selected_indices = np.random.choice(len(annotations), num_to_analyze, replace=False)

    # Análisis
    results = []
    suspicious_bboxes = []
    class_stats = defaultdict(lambda: {'total': 0, 'suspicious': 0})

    print(f"\nAnalizando {num_to_analyze} bounding boxes...")

    for i, idx in enumerate(tqdm(selected_indices, desc="Analizando")):
        ann = annotations[idx]
        img_info = images_dict.get(ann['image_id'])

        if img_info is None:
            continue

        # Cargar imagen
        img_path = os.path.join(args.image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            continue

        bbox = ann['bbox']
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, f'Unknown_{cat_id}')

        # Extraer región y analizar
        region = extract_bbox_region(image, bbox)
        if region.size == 0:
            continue

        metrics = analyze_edge_content(region, edge_width=args.edge_width)

        if not metrics.get('valid', False):
            continue

        padding_score = calculate_padding_score(metrics)
        metrics['score'] = padding_score

        result = {
            'ann_id': ann['id'],
            'image_id': ann['image_id'],
            'image_file': img_info['file_name'],
            'bbox': bbox,
            'category': cat_name,
            'padding_score': padding_score,
            'metrics': metrics
        }
        results.append(result)

        # Estadísticas por clase
        class_stats[cat_name]['total'] += 1

        if padding_score >= args.padding_threshold:
            class_stats[cat_name]['suspicious'] += 1
            suspicious_bboxes.append(result)

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generar visualizaciones de casos sospechosos
    if suspicious_bboxes and args.visualize_suspicious > 0:
        print(f"\nGenerando visualizaciones de {min(len(suspicious_bboxes), args.visualize_suspicious)} casos sospechosos...")
        vis_dir = output_dir / 'suspicious_samples'
        vis_dir.mkdir(exist_ok=True)

        # Ordenar por score descendente
        suspicious_bboxes.sort(key=lambda x: x['padding_score'], reverse=True)

        for i, result in enumerate(suspicious_bboxes[:args.visualize_suspicious]):
            img_path = os.path.join(args.image_dir, result['image_file'])
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
                vis_path = vis_dir / f"suspicious_{i:03d}_score{result['padding_score']:.2f}.png"
                visualize_suspicious_bbox(image, result['bbox'], result['metrics'], vis_path)
            except Exception as e:
                print(f"  Error en visualización {i}: {e}")

    # Estadísticas globales
    if results:
        scores = [r['padding_score'] for r in results if r['padding_score'] is not None]

        print("\n" + "=" * 80)
        print("RESULTADOS DEL ANÁLISIS")
        print("=" * 80)

        print(f"\nBBoxes analizados: {len(results)}")
        print(f"BBoxes con padding sospechoso (score >= {args.padding_threshold}): {len(suspicious_bboxes)}")
        print(f"Porcentaje sospechoso: {100 * len(suspicious_bboxes) / len(results):.2f}%")

        print(f"\nDistribución de Padding Scores:")
        print(f"  Min:     {min(scores):.3f}")
        print(f"  Max:     {max(scores):.3f}")
        print(f"  Media:   {np.mean(scores):.3f}")
        print(f"  Mediana: {np.median(scores):.3f}")
        print(f"  Std:     {np.std(scores):.3f}")

        # Distribución por rangos
        ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        print(f"\nDistribución por rangos de score:")
        for low, high in ranges:
            count = sum(1 for s in scores if low <= s < high)
            pct = 100 * count / len(scores)
            bar = '█' * int(pct / 2)
            status = "⚠️" if low >= args.padding_threshold else ""
            print(f"  [{low:.1f}-{high:.1f}): {count:5d} ({pct:5.1f}%) {bar} {status}")

        # Por clase
        print(f"\nAnálisis por clase:")
        print(f"  {'Clase':<20} {'Total':>8} {'Sospechosos':>12} {'%':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*8}")
        for cat_name in sorted(class_stats.keys()):
            stats = class_stats[cat_name]
            pct = 100 * stats['suspicious'] / stats['total'] if stats['total'] > 0 else 0
            flag = "⚠️" if pct > 10 else ""
            print(f"  {cat_name:<20} {stats['total']:>8} {stats['suspicious']:>12} {pct:>7.1f}% {flag}")

        # Guardar resultados
        results_file = output_dir / 'bbox_tightness_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_analyzed': len(results),
                    'suspicious_count': len(suspicious_bboxes),
                    'suspicious_percentage': 100 * len(suspicious_bboxes) / len(results),
                    'threshold': args.padding_threshold,
                    'score_mean': np.mean(scores),
                    'score_std': np.std(scores)
                },
                'class_stats': dict(class_stats),
                'suspicious_bboxes': [
                    {
                        'ann_id': r['ann_id'],
                        'image_file': r['image_file'],
                        'category': r['category'],
                        'padding_score': r['padding_score'],
                        'bbox': r['bbox']
                    }
                    for r in suspicious_bboxes
                ]
            }, f, indent=2)
        print(f"\nResultados guardados en: {results_file}")

        # Veredicto final
        suspicious_pct = 100 * len(suspicious_bboxes) / len(results)
        print("\n" + "=" * 80)
        print("VEREDICTO")
        print("=" * 80)

        if suspicious_pct < 5:
            print(f"\n[OK] DATASET APROBADO")
            print(f"Solo {suspicious_pct:.1f}% de bboxes tienen padding sospechoso (<5% umbral)")
            return 0
        elif suspicious_pct < 10:
            print(f"\n[WARNING] DATASET CON ADVERTENCIAS")
            print(f"{suspicious_pct:.1f}% de bboxes tienen padding sospechoso")
            print("Revisar manualmente las visualizaciones generadas")
            return 0
        else:
            print(f"\n[FALLO] DATASET CON PROBLEMAS DE PADDING")
            print(f"{suspicious_pct:.1f}% de bboxes tienen padding sospechoso (>10%)")
            print("Esto puede afectar severamente el entrenamiento del modelo")
            print("Revisar el proceso de generación de anotaciones sintéticas")
            return 1
    else:
        print("\nNo se pudieron analizar suficientes bboxes")
        return 1


if __name__ == '__main__':
    sys.exit(main())
