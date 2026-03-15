"""
Script de diagnóstico profundo para dataset sintético RT-DETRv2.

Analiza:
1. Visualización de anotaciones sobre imágenes
2. Distribución de clases
3. Tamaños de bounding boxes
4. Detección de posibles problemas de etiquetado
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_coco_json(json_path: str) -> dict:
    """Carga el archivo JSON de anotaciones COCO."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_image(file_name: str, image_dirs: list) -> str:
    """Busca una imagen en los directorios dados."""
    for img_dir in image_dirs:
        full_path = os.path.join(img_dir, file_name)
        if os.path.exists(full_path):
            return full_path
    return None


def analyze_bbox_sizes(coco_data: dict):
    """Analiza la distribución de tamaños de bounding boxes."""
    areas = []
    widths = []
    heights = []
    aspect_ratios = []

    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        if w > 0 and h > 0:
            areas.append(w * h)
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)

    print("\n" + "="*60)
    print("ANÁLISIS DE TAMAÑOS DE BOUNDING BOXES")
    print("="*60)

    # Categorías de tamaño según COCO
    small = sum(1 for a in areas if a < 32*32)
    medium = sum(1 for a in areas if 32*32 <= a < 96*96)
    large = sum(1 for a in areas if a >= 96*96)

    print(f"\nDistribución por tamaño (COCO):")
    print(f"  Small  (<32x32):   {small:6d} ({100*small/len(areas):.1f}%)")
    print(f"  Medium (32-96):    {medium:6d} ({100*medium/len(areas):.1f}%)")
    print(f"  Large  (>96x96):   {large:6d} ({100*large/len(areas):.1f}%)")

    print(f"\nEstadísticas de área:")
    print(f"  Mínimo:    {min(areas):10.1f} px² ({np.sqrt(min(areas)):.1f}x{np.sqrt(min(areas)):.1f})")
    print(f"  Máximo:    {max(areas):10.1f} px²")
    print(f"  Promedio:  {np.mean(areas):10.1f} px²")
    print(f"  Mediana:   {np.median(areas):10.1f} px²")

    print(f"\nEstadísticas de dimensiones:")
    print(f"  Width:  min={min(widths):.1f}, max={max(widths):.1f}, mean={np.mean(widths):.1f}")
    print(f"  Height: min={min(heights):.1f}, max={max(heights):.1f}, mean={np.mean(heights):.1f}")

    # Objetos muy pequeños (potencialmente problemáticos)
    very_small = sum(1 for a in areas if a < 16*16)
    if very_small > 0:
        print(f"\n⚠️  ADVERTENCIA: {very_small} objetos muy pequeños (<16x16 = 256px²)")
        print(f"   Estos objetos pueden ser filtrados por SanitizeBoundingBoxes")

    return {
        'areas': areas,
        'widths': widths,
        'heights': heights,
        'aspect_ratios': aspect_ratios
    }


def analyze_class_distribution(coco_data: dict):
    """Analiza la distribución de clases."""
    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    class_counts = Counter()

    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        class_counts[cat_names.get(cat_id, f'ID:{cat_id}')] += 1

    print("\n" + "="*60)
    print("DISTRIBUCIÓN DE CLASES")
    print("="*60)

    total = sum(class_counts.values())
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {cls:20s}: {count:6d} ({pct:5.1f}%) {bar}")

    # Detectar desbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    ratio = max_count / min_count if min_count > 0 else float('inf')

    if ratio > 10:
        print(f"\n⚠️  ADVERTENCIA: Desbalance de clases severo (ratio {ratio:.1f}:1)")

    return class_counts


def visualize_samples_detailed(coco_data: dict, image_dirs: list, num_samples: int = 9,
                               save_path: str = None, check_bbox_quality: bool = True):
    """Visualiza muestras detalladas con diagnóstico de calidad."""

    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    img_id_to_info = {img['id']: img for img in coco_data['images']}

    # Seleccionar imágenes con anotaciones
    valid_img_ids = [img_id for img_id in img_to_anns.keys()]
    if not valid_img_ids:
        print("No hay imágenes con anotaciones")
        return

    sample_ids = random.sample(valid_img_ids, min(num_samples, len(valid_img_ids)))

    # Colores por clase
    colors = plt.cm.tab10(np.linspace(0, 1, len(cat_names)))
    cat_colors = {cat_id: colors[i % len(colors)] for i, cat_id in enumerate(cat_names.keys())}

    # Crear figura
    cols = 3
    rows = (len(sample_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    axes = np.array(axes).flatten()

    quality_issues = []

    for idx, img_id in enumerate(sample_ids):
        ax = axes[idx]
        img_info = img_id_to_info[img_id]
        file_name = img_info['file_name']

        # Buscar imagen
        img_path = find_image(file_name, image_dirs)
        if not img_path:
            ax.set_title(f'NO ENCONTRADA: {file_name}')
            ax.axis('off')
            quality_issues.append(f"Imagen no encontrada: {file_name}")
            continue

        # Cargar imagen
        img = Image.open(img_path)
        img_w, img_h = img.size
        ax.imshow(img)

        # Verificar dimensiones
        expected_w, expected_h = img_info.get('width', img_w), img_info.get('height', img_h)
        if img_w != expected_w or img_h != expected_h:
            quality_issues.append(f"Dimensiones incorrectas: {file_name} (JSON: {expected_w}x{expected_h}, Real: {img_w}x{img_h})")

        # Dibujar anotaciones
        anns = img_to_anns[img_id]
        for ann in anns:
            x, y, w, h = ann['bbox']
            cat_id = ann['category_id']
            color = cat_colors.get(cat_id, 'red')

            # Verificar bbox dentro de imagen
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                quality_issues.append(f"BBox fuera de imagen: {file_name}, bbox=[{x:.0f},{y:.0f},{w:.0f},{h:.0f}]")

            # Verificar bbox muy pequeño
            if w * h < 100:
                quality_issues.append(f"BBox muy pequeño: {file_name}, área={w*h:.0f}px²")

            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            label = cat_names.get(cat_id, f'ID:{cat_id}')
            ax.text(
                x, y - 3,
                f'{label} ({w:.0f}x{h:.0f})',
                color='white',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
            )

        ax.set_title(f'{file_name}\n{len(anns)} anotaciones, {img_w}x{img_h}px')
        ax.axis('off')

    # Ocultar ejes vacíos
    for idx in range(len(sample_ids), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualizaciones guardadas en: {save_path}")
    else:
        plt.show()

    # Reportar problemas de calidad
    if quality_issues:
        print("\n" + "="*60)
        print("⚠️  PROBLEMAS DE CALIDAD DETECTADOS")
        print("="*60)
        for issue in quality_issues[:20]:  # Mostrar primeros 20
            print(f"  • {issue}")
        if len(quality_issues) > 20:
            print(f"  ... y {len(quality_issues) - 20} problemas más")

    return quality_issues


def compare_train_val(train_json: str, val_json: str, image_dirs: list):
    """Compara distribuciones entre train y val."""
    train_data = load_coco_json(train_json)
    val_data = load_coco_json(val_json)

    print("\n" + "="*60)
    print("COMPARACIÓN TRAIN vs VAL")
    print("="*60)

    # Categorías
    train_cats = {cat['id']: cat['name'] for cat in train_data['categories']}
    val_cats = {cat['id']: cat['name'] for cat in val_data['categories']}

    print(f"\nCategorías Train: {list(train_cats.values())}")
    print(f"Categorías Val:   {list(val_cats.values())}")

    if train_cats != val_cats:
        print("⚠️  ADVERTENCIA: Las categorías no coinciden entre train y val!")

    # Conteo por clase
    train_counts = Counter(train_cats.get(ann['category_id'], 'UNK')
                          for ann in train_data['annotations'])
    val_counts = Counter(val_cats.get(ann['category_id'], 'UNK')
                        for ann in val_data['annotations'])

    print("\nDistribución por clase:")
    print(f"{'Clase':<20} {'Train':>8} {'Val':>8} {'Ratio':>8}")
    print("-" * 48)

    all_classes = set(train_counts.keys()) | set(val_counts.keys())
    for cls in sorted(all_classes):
        t = train_counts.get(cls, 0)
        v = val_counts.get(cls, 0)
        ratio = t / v if v > 0 else float('inf')
        print(f"{cls:<20} {t:>8} {v:>8} {ratio:>7.1f}x")

    # Verificar que val usa imágenes del mismo directorio
    print("\nVerificando acceso a imágenes de validación...")
    found = 0
    missing = 0
    for img in val_data['images'][:100]:  # Verificar primeras 100
        if find_image(img['file_name'], image_dirs):
            found += 1
        else:
            missing += 1
            if missing <= 3:
                print(f"  No encontrada: {img['file_name']}")

    print(f"  Encontradas: {found}/100, Faltantes: {missing}/100")
    if missing > 0:
        print("  ⚠️  CRÍTICO: Algunas imágenes de validación no se encuentran!")


def main():
    parser = argparse.ArgumentParser(description='Diagnóstico profundo de dataset sintético')
    parser.add_argument('--train-json', type=str, required=True, help='JSON de entrenamiento')
    parser.add_argument('--val-json', type=str, default=None, help='JSON de validación (opcional)')
    parser.add_argument('--image-dirs', type=str, nargs='+', required=True, help='Directorios de imágenes')
    parser.add_argument('--visualize', type=int, default=9, help='Número de muestras a visualizar')
    parser.add_argument('--save-dir', type=str, default='./diagnostic_output', help='Directorio para guardar resultados')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Cargando JSON de entrenamiento: {args.train_json}")
    train_data = load_coco_json(args.train_json)

    print(f"\n📊 Dataset: {len(train_data['images'])} imágenes, {len(train_data['annotations'])} anotaciones")
    print(f"   Categorías: {[c['name'] for c in train_data['categories']]}")

    # Análisis de tamaños
    analyze_bbox_sizes(train_data)

    # Análisis de clases
    analyze_class_distribution(train_data)

    # Visualización detallada
    print("\n\nGenerando visualizaciones de muestras...")
    visualize_samples_detailed(
        train_data,
        args.image_dirs,
        num_samples=args.visualize,
        save_path=os.path.join(args.save_dir, 'train_samples.png')
    )

    # Comparación train/val si se proporciona
    if args.val_json:
        compare_train_val(args.train_json, args.val_json, args.image_dirs)

        val_data = load_coco_json(args.val_json)
        print("\n\nGenerando visualizaciones de validación...")
        visualize_samples_detailed(
            val_data,
            args.image_dirs,
            num_samples=args.visualize,
            save_path=os.path.join(args.save_dir, 'val_samples.png')
        )

    print(f"\n✅ Diagnóstico completado. Resultados guardados en: {args.save_dir}")


if __name__ == '__main__':
    main()
