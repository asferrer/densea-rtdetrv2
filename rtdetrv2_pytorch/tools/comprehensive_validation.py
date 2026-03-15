"""
Validación COMPLETA de dataset sintético para debugging cuando el modelo no aprende.

Este script verifica TODO lo que puede estar mal:
1. Formato COCO correcto
2. Imágenes legibles
3. Bboxes válidos y alineados
4. Distribuciones estadísticas
5. Calidad visual
"""

import os
import sys
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def find_image(file_name, image_dirs):
    """Busca una imagen en múltiples directorios."""
    for img_dir in image_dirs:
        full_path = os.path.join(img_dir, file_name)
        if os.path.exists(full_path):
            return full_path
    return None


def test_1_load_json(json_path):
    """TEST 1: ¿El JSON se puede cargar?"""
    print("\n" + "="*70)
    print("TEST 1: CARGA DE JSON")
    print("="*70)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON cargado correctamente")
        print(f"   Imágenes: {len(data.get('images', []))}")
        print(f"   Anotaciones: {len(data.get('annotations', []))}")
        print(f"   Categorías: {len(data.get('categories', []))}")
        return data, []
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None, [f"No se puede cargar JSON: {e}"]


def test_2_structure(data):
    """TEST 2: ¿Tiene la estructura COCO correcta?"""
    print("\n" + "="*70)
    print("TEST 2: ESTRUCTURA COCO")
    print("="*70)

    issues = []

    required = ['images', 'annotations', 'categories']
    for key in required:
        if key not in data:
            print(f"❌ Falta clave: {key}")
            issues.append(f"Falta clave '{key}'")
        else:
            print(f"✅ Clave '{key}' presente")

    if not issues:
        if len(data['images']) == 0:
            issues.append("Sin imágenes")
        if len(data['annotations']) == 0:
            issues.append("Sin anotaciones")
        if len(data['categories']) == 0:
            issues.append("Sin categorías")

    return issues


def test_3_categories(data):
    """TEST 3: ¿Las categorías son válidas?"""
    print("\n" + "="*70)
    print("TEST 3: CATEGORÍAS")
    print("="*70)

    issues = []
    cat_ids = set()

    for cat in data['categories']:
        if 'id' not in cat or 'name' not in cat:
            issues.append(f"Categoría sin id/name: {cat}")
            continue

        if cat['id'] in cat_ids:
            issues.append(f"ID duplicado: {cat['id']}")
        cat_ids.add(cat['id'])

        print(f"✅ ID {cat['id']}: {cat['name']}")

    # Verificar consecutividad desde 0
    expected = set(range(len(data['categories'])))
    if cat_ids != expected:
        print(f"⚠️  IDs no consecutivos desde 0")
        print(f"   Encontrados: {sorted(cat_ids)}")
        print(f"   Esperados: {sorted(expected)}")
        issues.append("IDs no consecutivos desde 0")

    return issues


def test_4_images(data, image_dirs):
    """TEST 4: ¿Todas las imágenes existen y son legibles?"""
    print("\n" + "="*70)
    print("TEST 4: IMÁGENES")
    print("="*70)

    issues = []
    missing = []
    corrupt = []
    valid = 0

    print(f"Verificando {len(data['images'])} imágenes...")

    for i, img_info in enumerate(data['images']):
        if i % 1000 == 0 and i > 0:
            print(f"  Procesadas: {i}/{len(data['images'])}")

        file_name = img_info.get('file_name', '?')
        img_path = find_image(file_name, image_dirs)

        if img_path is None:
            missing.append(file_name)
            continue

        try:
            img = Image.open(img_path)
            img.verify()
            valid += 1
        except Exception as e:
            corrupt.append((file_name, str(e)))

    print(f"\n✅ Válidas: {valid}/{len(data['images'])}")

    if missing:
        print(f"❌ Faltantes: {len(missing)}")
        print(f"   Ejemplos: {missing[:3]}")
        issues.append(f"{len(missing)} imágenes faltantes")

    if corrupt:
        print(f"❌ Corruptas: {len(corrupt)}")
        print(f"   Ejemplos: {corrupt[:3]}")
        issues.append(f"{len(corrupt)} imágenes corruptas")

    return issues


def test_5_annotations(data):
    """TEST 5: ¿Las anotaciones son válidas?"""
    print("\n" + "="*70)
    print("TEST 5: ANOTACIONES")
    print("="*70)

    issues = []
    img_ids = {img['id'] for img in data['images']}
    cat_ids = {cat['id'] for cat in data['categories']}

    invalid_img_refs = 0
    invalid_cat_refs = 0
    invalid_bboxes = 0
    zero_area = 0

    for ann in data['annotations']:
        # Image ID válido
        if ann.get('image_id') not in img_ids:
            invalid_img_refs += 1

        # Category ID válido
        if ann.get('category_id') not in cat_ids:
            invalid_cat_refs += 1

        # Bbox válido
        bbox = ann.get('bbox', [])
        if len(bbox) != 4:
            invalid_bboxes += 1
            continue

        x, y, w, h = bbox
        if x < 0 or y < 0 or w < 0 or h < 0:
            invalid_bboxes += 1

        if w * h == 0:
            zero_area += 1

    print(f"✅ Total anotaciones: {len(data['annotations'])}")

    if invalid_img_refs > 0:
        print(f"❌ Referencias a image_id inválido: {invalid_img_refs}")
        issues.append(f"{invalid_img_refs} refs a image_id inválido")

    if invalid_cat_refs > 0:
        print(f"❌ Referencias a category_id inválido: {invalid_cat_refs}")
        issues.append(f"{invalid_cat_refs} refs a category_id inválido")

    if invalid_bboxes > 0:
        print(f"❌ Bboxes inválidos: {invalid_bboxes}")
        issues.append(f"{invalid_bboxes} bboxes inválidos")

    if zero_area > 0:
        print(f"⚠️  Bboxes con área cero: {zero_area}")
        issues.append(f"{zero_area} bboxes con área cero")

    return issues


def test_6_visual_alignment(data, image_dirs, num_samples=12, save_path='validation_visual.png'):
    """TEST 6: ¿Los bboxes están visualmente alineados con los objetos?"""
    print("\n" + "="*70)
    print("TEST 6: ALINEACIÓN VISUAL")
    print("="*70)

    cat_names = {cat['id']: cat['name'] for cat in data['categories']}

    # Agrupar anotaciones por imagen
    anns_by_img = defaultdict(list)
    for ann in data['annotations']:
        anns_by_img[ann['image_id']].append(ann)

    # Seleccionar muestras
    img_ids = list(anns_by_img.keys())
    samples = random.sample(img_ids, min(num_samples, len(img_ids)))

    # Crear grid
    rows = (num_samples + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, img_id in enumerate(samples):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Encontrar info de imagen
        img_info = next((img for img in data['images'] if img['id'] == img_id), None)
        if not img_info:
            continue

        img_path = find_image(img_info['file_name'], image_dirs)
        if not img_path:
            ax.text(0.5, 0.5, 'No encontrada', ha='center', va='center')
            continue

        try:
            img = Image.open(img_path)
            ax.imshow(img)

            # Dibujar bboxes
            for ann in anns_by_img[img_id]:
                x, y, w, h = ann['bbox']
                cat_name = cat_names.get(ann['category_id'], 'UNK')
                color = colors[ann['category_id'] % 10]

                rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                        edgecolor=color, facecolor='none')
                ax.add_patch(rect)

                ax.text(x, max(0, y-5), cat_name, color='white', fontsize=10,
                       weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

            ax.set_title(f"{img_info['file_name'][:40]}...\n{len(anns_by_img[img_id])} objetos",
                        fontsize=9)
            ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')

    # Ocultar axes vacíos
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('VERIFICAR: ¿Los bboxes (cajas de colores) cubren correctamente los objetos?',
                 fontsize=14, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualización guardada: {save_path}")
    print(f"\n   ⚠️  IMPORTANTE: ABRE Y REVISA ESTA IMAGEN")
    print(f"   Si los bboxes NO cubren bien los objetos, ESE es el problema.")

    return []


def test_7_distribution(data):
    """TEST 7: ¿La distribución de clases es razonable?"""
    print("\n" + "="*70)
    print("TEST 7: DISTRIBUCIÓN DE CLASES")
    print("="*70)

    cat_names = {cat['id']: cat['name'] for cat in data['categories']}
    class_counts = Counter(cat_names[ann['category_id']]
                          for ann in data['annotations']
                          if ann['category_id'] in cat_names)

    total = sum(class_counts.values())

    print(f"\n{'Clase':<20} {'Cantidad':>10} {'Porcentaje':>12}")
    print("-" * 44)

    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = 100 * count / total
        print(f"{cls:<20} {count:>10} {pct:>11.1f}%")

    # Detectar desbalances severos
    percentages = [100 * c / total for c in class_counts.values()]
    if max(percentages) > 50:
        print(f"\n⚠️  Una clase domina >50% del dataset")
    elif max(percentages) / min(percentages) > 10:
        print(f"\n⚠️  Desbalance de clases >10x entre mayor y menor")

    return []


def test_8_size_distribution(data):
    """TEST 8: ¿Los tamaños de objetos son razonables?"""
    print("\n" + "="*70)
    print("TEST 8: DISTRIBUCIÓN DE TAMAÑOS")
    print("="*70)

    img_dims = {img['id']: (img['width'], img['height']) for img in data['images']}

    sizes = []
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id in img_dims:
            img_w, img_h = img_dims[img_id]
            x, y, w, h = ann['bbox']
            rel_area = (w * h) / (img_w * img_h)
            sizes.append(rel_area)

    if not sizes:
        print("❌ No se pudo calcular tamaños")
        return ["No se pudo calcular tamaños"]

    print(f"\nÁrea relativa (bbox_area / image_area):")
    print(f"  Media:   {np.mean(sizes):.4f}")
    print(f"  Mediana: {np.median(sizes):.4f}")
    print(f"  Min:     {np.min(sizes):.6f}")
    print(f"  Max:     {np.max(sizes):.4f}")

    # Categorías COCO
    small = sum(1 for s in sizes if s < (32*32)/(640*640))
    medium = sum(1 for s in sizes if (32*32)/(640*640) <= s < (96*96)/(640*640))
    large = sum(1 for s in sizes if s >= (96*96)/(640*640))

    print(f"\nCategorías COCO (asumiendo img 640x640):")
    print(f"  Small:  {small:5d} ({100*small/len(sizes):5.1f}%)")
    print(f"  Medium: {medium:5d} ({100*medium/len(sizes):5.1f}%)")
    print(f"  Large:  {large:5d} ({100*large/len(sizes):5.1f}%)")

    return []


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Validación COMPLETA de dataset sintético para debugging'
    )
    parser.add_argument('--json', required=True, help='JSON de anotaciones')
    parser.add_argument('--image-dirs', nargs='+', required=True,
                       help='Directorios de imágenes')
    parser.add_argument('--num-visual', type=int, default=12,
                       help='Muestras para validación visual')
    args = parser.parse_args()

    all_issues = []

    # Ejecutar tests
    data, issues = test_1_load_json(args.json)
    all_issues.extend(issues)
    if data is None:
        return 1

    all_issues.extend(test_2_structure(data))
    all_issues.extend(test_3_categories(data))
    all_issues.extend(test_4_images(data, args.image_dirs))
    all_issues.extend(test_5_annotations(data))
    all_issues.extend(test_6_visual_alignment(data, args.image_dirs, args.num_visual))
    all_issues.extend(test_7_distribution(data))
    all_issues.extend(test_8_size_distribution(data))

    # Reporte final
    print("\n" + "="*70)
    print("REPORTE FINAL")
    print("="*70)

    if not all_issues:
        print("\n🎉 DATASET FORMALMENTE VÁLIDO")
        print("\nPero si el modelo NO aprende, el problema es de CALIDAD VISUAL:")
        print("  1. Revisa la imagen 'validation_visual.png'")
        print("  2. Verifica que los bboxes cubran correctamente los objetos")
        print("  3. Verifica que los objetos sintéticos se vean REALISTAS")
        print("  4. Busca artefactos: halos, bordes artificiales, iluminación inconsistente")
        print("\nSi los objetos se ven artificiales, el problema es el pipeline de generación.")
    else:
        print(f"\n❌ ENCONTRADOS {len(all_issues)} PROBLEMAS:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("\n⚠️  CORRIGE ESTOS PROBLEMAS ANTES DE ENTRENAR")

    return 0 if not all_issues else 1


if __name__ == '__main__':
    sys.exit(main())
