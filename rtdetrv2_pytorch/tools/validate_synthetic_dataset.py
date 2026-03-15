"""
Script de validación para dataset sintético RT-DETRv2.

Verifica:
1. Que todas las imágenes referenciadas en el JSON existen
2. Que los bounding boxes tienen formato correcto (COCO: [x, y, width, height])
3. Que los category_ids son válidos
4. Estadísticas del dataset
5. Visualización de muestras aleatorias con anotaciones
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_coco_json(json_path: str) -> dict:
    """Carga el archivo JSON de anotaciones COCO."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_images_exist(coco_data: dict, image_dirs: list) -> dict:
    """Verifica que todas las imágenes del JSON existan en los directorios."""
    results = {
        'total': len(coco_data['images']),
        'found': 0,
        'missing': [],
        'found_paths': {}
    }

    for img_info in coco_data['images']:
        file_name = img_info['file_name']
        found = False

        for img_dir in image_dirs:
            full_path = os.path.join(img_dir, file_name)
            if os.path.exists(full_path):
                results['found'] += 1
                results['found_paths'][img_info['id']] = full_path
                found = True
                break

        if not found:
            results['missing'].append({
                'id': img_info['id'],
                'file_name': file_name
            })

    return results


def validate_bounding_boxes(coco_data: dict) -> dict:
    """Valida el formato de los bounding boxes."""
    results = {
        'total': len(coco_data['annotations']),
        'valid': 0,
        'invalid': [],
        'warnings': [],
        'size_stats': {
            'areas': [],
            'widths': [],
            'heights': []
        }
    }

    # Crear mapeo de image_id a dimensiones
    img_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    for ann in coco_data['annotations']:
        bbox = ann.get('bbox', [])
        img_id = ann['image_id']
        ann_id = ann['id']

        # Verificar formato: [x, y, width, height]
        if len(bbox) != 4:
            results['invalid'].append({
                'id': ann_id,
                'image_id': img_id,
                'reason': f'bbox debe tener 4 elementos, tiene {len(bbox)}'
            })
            continue

        x, y, w, h = bbox

        # Verificar valores positivos
        if w <= 0 or h <= 0:
            results['invalid'].append({
                'id': ann_id,
                'image_id': img_id,
                'reason': f'width ({w}) o height ({h}) no positivo',
                'bbox': bbox
            })
            continue

        # Verificar que bbox está dentro de la imagen
        if img_id in img_dims:
            img_w, img_h = img_dims[img_id]
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                results['warnings'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'reason': f'bbox fuera de límites: bbox={bbox}, img_size=({img_w}, {img_h})'
                })

        # Estadísticas
        area = w * h
        results['size_stats']['areas'].append(area)
        results['size_stats']['widths'].append(w)
        results['size_stats']['heights'].append(h)
        results['valid'] += 1

    return results


def validate_categories(coco_data: dict) -> dict:
    """Valida los category_ids de las anotaciones."""
    valid_cat_ids = {cat['id'] for cat in coco_data['categories']}
    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

    results = {
        'categories': coco_data['categories'],
        'valid_ids': sorted(valid_cat_ids),
        'category_counts': Counter(),
        'invalid_annotations': []
    }

    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        if cat_id in valid_cat_ids:
            results['category_counts'][cat_names[cat_id]] += 1
        else:
            results['invalid_annotations'].append({
                'id': ann['id'],
                'image_id': ann['image_id'],
                'invalid_category_id': cat_id
            })

    return results


def calculate_statistics(coco_data: dict) -> dict:
    """Calcula estadísticas generales del dataset."""
    # Anotaciones por imagen
    anns_per_image = Counter(ann['image_id'] for ann in coco_data['annotations'])

    stats = {
        'num_images': len(coco_data['images']),
        'num_annotations': len(coco_data['annotations']),
        'num_categories': len(coco_data['categories']),
        'avg_annotations_per_image': len(coco_data['annotations']) / len(coco_data['images']) if coco_data['images'] else 0,
        'max_annotations_per_image': max(anns_per_image.values()) if anns_per_image else 0,
        'min_annotations_per_image': min(anns_per_image.values()) if anns_per_image else 0,
        'images_without_annotations': len(coco_data['images']) - len(anns_per_image)
    }

    return stats


def visualize_samples(coco_data: dict, image_paths: dict, num_samples: int = 6, save_path: str = None):
    """Visualiza muestras aleatorias con sus anotaciones."""
    # Filtrar imágenes que tienen anotaciones y paths válidos
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    valid_img_ids = [img_id for img_id in img_to_anns.keys() if img_id in image_paths]

    if not valid_img_ids:
        print("No hay imágenes válidas para visualizar")
        return

    # Seleccionar muestras aleatorias
    sample_ids = random.sample(valid_img_ids, min(num_samples, len(valid_img_ids)))

    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    colors = plt.cm.tab10(np.linspace(0, 1, len(cat_names)))
    cat_colors = {cat_id: colors[i % len(colors)] for i, cat_id in enumerate(cat_names.keys())}

    # Crear figura
    cols = 3
    rows = (len(sample_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for idx, img_id in enumerate(sample_ids):
        ax = axes[idx]
        img_path = image_paths[img_id]

        # Cargar imagen
        img = Image.open(img_path)
        ax.imshow(img)

        # Dibujar anotaciones
        for ann in img_to_anns[img_id]:
            x, y, w, h = ann['bbox']
            cat_id = ann['category_id']
            color = cat_colors.get(cat_id, 'red')

            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x, y - 5,
                cat_names.get(cat_id, f'ID:{cat_id}'),
                color='white',
                fontsize=8,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
            )

        ax.set_title(f'ID: {img_id} ({len(img_to_anns[img_id])} anns)')
        ax.axis('off')

    # Ocultar ejes vacíos
    for idx in range(len(sample_ids), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    else:
        plt.show()


def print_report(stats: dict, img_results: dict, bbox_results: dict, cat_results: dict):
    """Imprime el reporte de validación."""
    print("\n" + "="*60)
    print("REPORTE DE VALIDACIÓN - DATASET SINTÉTICO")
    print("="*60)

    # Estadísticas generales
    print("\n📊 ESTADÍSTICAS GENERALES")
    print("-"*40)
    print(f"  Imágenes totales:        {stats['num_images']}")
    print(f"  Anotaciones totales:     {stats['num_annotations']}")
    print(f"  Categorías:              {stats['num_categories']}")
    print(f"  Anns/imagen (promedio):  {stats['avg_annotations_per_image']:.2f}")
    print(f"  Anns/imagen (máx):       {stats['max_annotations_per_image']}")
    print(f"  Anns/imagen (mín):       {stats['min_annotations_per_image']}")
    print(f"  Imágenes sin anns:       {stats['images_without_annotations']}")

    # Validación de imágenes
    print("\n🖼️  VALIDACIÓN DE IMÁGENES")
    print("-"*40)
    print(f"  Encontradas:  {img_results['found']}/{img_results['total']}")
    if img_results['missing']:
        print(f"  ❌ Faltantes:  {len(img_results['missing'])}")
        for m in img_results['missing'][:5]:
            print(f"     - {m['file_name']}")
        if len(img_results['missing']) > 5:
            print(f"     ... y {len(img_results['missing']) - 5} más")
    else:
        print("  ✅ Todas las imágenes encontradas")

    # Validación de bounding boxes
    print("\n📦 VALIDACIÓN DE BOUNDING BOXES")
    print("-"*40)
    print(f"  Válidos:      {bbox_results['valid']}/{bbox_results['total']}")
    if bbox_results['invalid']:
        print(f"  ❌ Inválidos:  {len(bbox_results['invalid'])}")
        for inv in bbox_results['invalid'][:3]:
            print(f"     - Ann {inv['id']}: {inv['reason']}")
    else:
        print("  ✅ Todos los bboxes válidos")

    if bbox_results['warnings']:
        print(f"  ⚠️  Advertencias: {len(bbox_results['warnings'])}")

    if bbox_results['size_stats']['areas']:
        areas = bbox_results['size_stats']['areas']
        print(f"\n  Estadísticas de tamaño:")
        print(f"    Área promedio:  {np.mean(areas):.1f} px²")
        print(f"    Área mínima:    {min(areas):.1f} px²")
        print(f"    Área máxima:    {max(areas):.1f} px²")

        # Objetos muy pequeños (< 32x32 = 1024 px²)
        small = sum(1 for a in areas if a < 1024)
        print(f"    Objetos pequeños (<32x32): {small} ({100*small/len(areas):.1f}%)")

    # Validación de categorías
    print("\n🏷️  VALIDACIÓN DE CATEGORÍAS")
    print("-"*40)
    print(f"  IDs válidos: {cat_results['valid_ids']}")
    print("\n  Distribución:")
    for cat, count in sorted(cat_results['category_counts'].items(), key=lambda x: -x[1]):
        print(f"    {cat:20s}: {count:6d}")

    if cat_results['invalid_annotations']:
        print(f"\n  ❌ Anotaciones con categoría inválida: {len(cat_results['invalid_annotations'])}")
    else:
        print("\n  ✅ Todas las categorías válidas")

    print("\n" + "="*60)

    # Resumen final
    all_ok = (
        len(img_results['missing']) == 0 and
        len(bbox_results['invalid']) == 0 and
        len(cat_results['invalid_annotations']) == 0
    )

    if all_ok:
        print("✅ VALIDACIÓN EXITOSA - El dataset está correctamente formateado")
    else:
        print("❌ VALIDACIÓN CON ERRORES - Revisa los problemas indicados arriba")

    print("="*60 + "\n")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Validar dataset sintético para RT-DETRv2')
    parser.add_argument('--json', type=str, required=True, help='Ruta al archivo JSON de anotaciones')
    parser.add_argument('--image-dirs', type=str, nargs='+', required=True, help='Directorios de imágenes')
    parser.add_argument('--visualize', type=int, default=0, help='Número de muestras a visualizar (0=ninguna)')
    parser.add_argument('--save-vis', type=str, default=None, help='Ruta para guardar visualización')
    args = parser.parse_args()

    print(f"\nCargando JSON: {args.json}")
    coco_data = load_coco_json(args.json)

    print(f"Buscando imágenes en: {args.image_dirs}")

    # Validaciones
    img_results = validate_images_exist(coco_data, args.image_dirs)
    bbox_results = validate_bounding_boxes(coco_data)
    cat_results = validate_categories(coco_data)
    stats = calculate_statistics(coco_data)

    # Reporte
    all_ok = print_report(stats, img_results, bbox_results, cat_results)

    # Visualización
    if args.visualize > 0:
        print(f"\nGenerando visualización de {args.visualize} muestras...")
        visualize_samples(
            coco_data,
            img_results['found_paths'],
            num_samples=args.visualize,
            save_path=args.save_vis
        )

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
