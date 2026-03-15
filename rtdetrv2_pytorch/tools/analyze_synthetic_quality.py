"""
Análisis profundo de calidad del dataset sintético para RT-DETRv2.

Investiga por qué el modelo puede estar aprendiendo lentamente:
1. Distribución espacial de objetos (¿siempre en el centro?)
2. Variabilidad de tamaños por clase
3. Análisis de colores/brillo de las imágenes
4. Aspect ratios de los bboxes
5. Densidad de objetos por imagen
6. Comparación visual sintético vs real (si hay datos reales)
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
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_image(file_name: str, image_dirs: list) -> str:
    for img_dir in image_dirs:
        full_path = os.path.join(img_dir, file_name)
        if os.path.exists(full_path):
            return full_path
    return None


def analyze_spatial_distribution(coco_data: dict):
    """Analiza dónde están posicionados los objetos en las imágenes."""
    print("\n" + "="*60)
    print("1. DISTRIBUCIÓN ESPACIAL DE OBJETOS")
    print("="*60)

    # Mapear image_id a dimensiones
    img_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    # Calcular posiciones normalizadas del centro de cada bbox
    center_x_norm = []
    center_y_norm = []

    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_dims:
            continue

        img_w, img_h = img_dims[img_id]
        x, y, w, h = ann['bbox']

        # Centro normalizado [0, 1]
        cx = (x + w/2) / img_w
        cy = (y + h/2) / img_h

        center_x_norm.append(cx)
        center_y_norm.append(cy)

    # Estadísticas
    print(f"\nPosición X del centro (normalizada 0-1):")
    print(f"  Media:    {np.mean(center_x_norm):.3f} (ideal: ~0.5)")
    print(f"  Std:      {np.std(center_x_norm):.3f} (ideal: >0.2)")
    print(f"  Min-Max:  [{min(center_x_norm):.3f}, {max(center_x_norm):.3f}]")

    print(f"\nPosición Y del centro (normalizada 0-1):")
    print(f"  Media:    {np.mean(center_y_norm):.3f} (ideal: ~0.5)")
    print(f"  Std:      {np.std(center_y_norm):.3f} (ideal: >0.2)")
    print(f"  Min-Max:  [{min(center_y_norm):.3f}, {max(center_y_norm):.3f}]")

    # Detectar problemas
    if np.std(center_x_norm) < 0.15 or np.std(center_y_norm) < 0.15:
        print("\n⚠️  PROBLEMA: Baja variabilidad espacial")
        print("   Los objetos están muy concentrados en una zona de la imagen.")
        print("   Esto dificulta que el modelo aprenda a detectar en otras posiciones.")

    # Análisis por cuadrantes
    quadrants = {'TL': 0, 'TR': 0, 'BL': 0, 'BR': 0}
    for cx, cy in zip(center_x_norm, center_y_norm):
        if cx < 0.5 and cy < 0.5:
            quadrants['TL'] += 1
        elif cx >= 0.5 and cy < 0.5:
            quadrants['TR'] += 1
        elif cx < 0.5 and cy >= 0.5:
            quadrants['BL'] += 1
        else:
            quadrants['BR'] += 1

    total = sum(quadrants.values())
    print(f"\nDistribución por cuadrantes:")
    for q, count in quadrants.items():
        pct = 100 * count / total
        expected = 25
        diff = abs(pct - expected)
        flag = "⚠️" if diff > 10 else "✓"
        print(f"  {q}: {pct:5.1f}% {flag}")

    return center_x_norm, center_y_norm


def analyze_size_per_class(coco_data: dict):
    """Analiza la variabilidad de tamaños por clase."""
    print("\n" + "="*60)
    print("2. VARIABILIDAD DE TAMAÑOS POR CLASE")
    print("="*60)

    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    img_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    sizes_per_class = defaultdict(list)

    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        x, y, w, h = ann['bbox']

        if img_id in img_dims:
            img_w, img_h = img_dims[img_id]
            # Tamaño relativo a la imagen
            rel_area = (w * h) / (img_w * img_h)
            sizes_per_class[cat_names.get(cat_id, f'ID:{cat_id}')].append(rel_area)

    print(f"\n{'Clase':<20} {'N':>6} {'Media':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8}")
    print("-" * 70)

    low_variability_classes = []
    for cls in sorted(sizes_per_class.keys()):
        sizes = sizes_per_class[cls]
        mean = np.mean(sizes)
        std = np.std(sizes)
        cv = std / mean if mean > 0 else 0  # Coeficiente de variación

        flag = ""
        if cv < 0.3:
            flag = " ⚠️ Baja variabilidad"
            low_variability_classes.append(cls)

        print(f"{cls:<20} {len(sizes):>6} {mean:>8.4f} {std:>8.4f} {cv:>8.2f} {min(sizes):>8.4f} {max(sizes):>8.4f}{flag}")

    if low_variability_classes:
        print(f"\n⚠️  PROBLEMA: Clases con baja variabilidad de tamaño:")
        print(f"   {', '.join(low_variability_classes)}")
        print("   El modelo puede tener dificultad con objetos de tamaños diferentes.")

    return sizes_per_class


def analyze_aspect_ratios(coco_data: dict):
    """Analiza los aspect ratios de los bounding boxes."""
    print("\n" + "="*60)
    print("3. ASPECT RATIOS DE BOUNDING BOXES")
    print("="*60)

    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    ar_per_class = defaultdict(list)

    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        if h > 0:
            ar = w / h
            cat_id = ann['category_id']
            ar_per_class[cat_names.get(cat_id, f'ID:{cat_id}')].append(ar)

    print(f"\n{'Clase':<20} {'Media AR':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 62)

    for cls in sorted(ar_per_class.keys()):
        ars = ar_per_class[cls]
        print(f"{cls:<20} {np.mean(ars):>10.2f} {np.std(ars):>10.2f} {min(ars):>10.2f} {max(ars):>10.2f}")

    # Detectar aspect ratios extremos
    all_ars = [ar for ars in ar_per_class.values() for ar in ars]
    extreme = sum(1 for ar in all_ars if ar < 0.2 or ar > 5)
    if extreme > 0:
        print(f"\n⚠️  {extreme} bboxes con aspect ratio extremo (<0.2 o >5)")

    return ar_per_class


def analyze_objects_per_image(coco_data: dict):
    """Analiza la densidad de objetos por imagen."""
    print("\n" + "="*60)
    print("4. DENSIDAD DE OBJETOS POR IMAGEN")
    print("="*60)

    anns_per_image = Counter(ann['image_id'] for ann in coco_data['annotations'])
    counts = list(anns_per_image.values())

    print(f"\nObjetos por imagen:")
    print(f"  Media:   {np.mean(counts):.2f}")
    print(f"  Mediana: {np.median(counts):.1f}")
    print(f"  Std:     {np.std(counts):.2f}")
    print(f"  Min:     {min(counts)}")
    print(f"  Max:     {max(counts)}")

    # Distribución
    bins = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    print(f"\nDistribución:")
    for i in range(len(bins)):
        if i == 0:
            count = sum(1 for c in counts if c == 1)
            print(f"  1 objeto:      {count:5d} ({100*count/len(counts):.1f}%)")
        elif i < len(bins) - 1:
            count = sum(1 for c in counts if bins[i] <= c < bins[i+1])
            print(f"  {bins[i]}-{bins[i+1]-1} objetos:  {count:5d} ({100*count/len(counts):.1f}%)")
        else:
            count = sum(1 for c in counts if c >= bins[i])
            print(f"  {bins[i]}+ objetos:   {count:5d} ({100*count/len(counts):.1f}%)")

    # Imágenes sin anotaciones
    imgs_with_anns = set(anns_per_image.keys())
    all_imgs = set(img['id'] for img in coco_data['images'])
    imgs_without = len(all_imgs - imgs_with_anns)

    if imgs_without > 0:
        print(f"\n⚠️  {imgs_without} imágenes sin anotaciones")

    return counts


def analyze_image_properties(coco_data: dict, image_dirs: list, num_samples: int = 100):
    """Analiza propiedades visuales de las imágenes (brillo, contraste)."""
    print("\n" + "="*60)
    print("5. PROPIEDADES VISUALES DE LAS IMÁGENES")
    print("="*60)

    # Seleccionar muestras aleatorias
    sample_imgs = random.sample(coco_data['images'], min(num_samples, len(coco_data['images'])))

    brightness_values = []
    contrast_values = []

    for img_info in sample_imgs:
        img_path = find_image(img_info['file_name'], image_dirs)
        if not img_path:
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            # Brillo (media de todos los canales)
            brightness = np.mean(img_array)
            brightness_values.append(brightness)

            # Contraste (desviación estándar)
            contrast = np.std(img_array)
            contrast_values.append(contrast)
        except Exception as e:
            continue

    if brightness_values:
        print(f"\nBrillo (0-255):")
        print(f"  Media:   {np.mean(brightness_values):.1f}")
        print(f"  Std:     {np.std(brightness_values):.1f}")
        print(f"  Min-Max: [{min(brightness_values):.1f}, {max(brightness_values):.1f}]")

        print(f"\nContraste (std de píxeles):")
        print(f"  Media:   {np.mean(contrast_values):.1f}")
        print(f"  Std:     {np.std(contrast_values):.1f}")
        print(f"  Min-Max: [{min(contrast_values):.1f}, {max(contrast_values):.1f}]")

        # Detectar problemas
        if np.std(brightness_values) < 20:
            print("\n⚠️  PROBLEMA: Baja variabilidad de brillo")
            print("   Las imágenes tienen condiciones de iluminación muy similares.")

        if np.mean(contrast_values) < 40:
            print("\n⚠️  PROBLEMA: Bajo contraste promedio")
            print("   Las imágenes pueden ser difíciles de distinguir del fondo.")

    return brightness_values, contrast_values


def analyze_bbox_tightness(coco_data: dict, image_dirs: list, num_samples: int = 20):
    """Verifica si los bboxes están bien ajustados (no muy sueltos)."""
    print("\n" + "="*60)
    print("6. ANÁLISIS DE AJUSTE DE BOUNDING BOXES")
    print("="*60)

    print("\nEste análisis requiere inspección visual.")
    print("Generando muestras para revisión manual...")

    # Crear figura con muestras
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    img_id_to_info = {img['id']: img for img in coco_data['images']}
    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Seleccionar imágenes con diferentes números de objetos
    valid_ids = list(img_to_anns.keys())
    sample_ids = random.sample(valid_ids, min(num_samples, len(valid_ids)))

    return sample_ids


def analyze_class_cooccurrence(coco_data: dict):
    """Analiza qué clases aparecen juntas en las mismas imágenes."""
    print("\n" + "="*60)
    print("7. CO-OCURRENCIA DE CLASES")
    print("="*60)

    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Clases por imagen
    classes_per_image = defaultdict(set)
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        cat_name = cat_names.get(ann['category_id'], 'UNK')
        classes_per_image[img_id].add(cat_name)

    # Contar co-ocurrencias
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for img_id, classes in classes_per_image.items():
        for c1 in classes:
            for c2 in classes:
                cooccurrence[c1][c2] += 1

    # Mostrar matriz
    all_classes = sorted(cat_names.values())
    print(f"\nMatriz de co-ocurrencia (diagonal = apariciones totales):")
    print(f"\n{'':15}", end='')
    for c in all_classes:
        print(f"{c[:8]:>10}", end='')
    print()

    for c1 in all_classes:
        print(f"{c1:15}", end='')
        for c2 in all_classes:
            count = cooccurrence[c1][c2]
            print(f"{count:>10}", end='')
        print()

    # Detectar clases que nunca aparecen juntas
    print("\nClases que NUNCA aparecen juntas:")
    found_isolated = False
    for i, c1 in enumerate(all_classes):
        for c2 in all_classes[i+1:]:
            if cooccurrence[c1][c2] == 0:
                print(f"  - {c1} + {c2}")
                found_isolated = True

    if not found_isolated:
        print("  (Todas las clases co-ocurren al menos una vez)")

    # Detectar imágenes con una sola clase
    single_class_imgs = sum(1 for classes in classes_per_image.values() if len(classes) == 1)
    pct = 100 * single_class_imgs / len(classes_per_image)
    print(f"\nImágenes con una sola clase: {single_class_imgs} ({pct:.1f}%)")

    if pct > 80:
        print("⚠️  PROBLEMA: Mayoría de imágenes tienen solo una clase")
        print("   El modelo puede tener dificultad distinguiendo clases.")

    return cooccurrence


def generate_diagnostic_visualization(coco_data: dict, image_dirs: list, save_path: str):
    """Genera visualización completa del diagnóstico."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    cat_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    img_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}

    # 1. Heatmap de posiciones
    ax = axes[0, 0]
    center_x, center_y = [], []
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in img_dims:
            img_w, img_h = img_dims[img_id]
            x, y, w, h = ann['bbox']
            center_x.append((x + w/2) / img_w)
            center_y.append((y + h/2) / img_h)

    ax.hist2d(center_x, center_y, bins=20, cmap='hot')
    ax.set_title('Distribución espacial de objetos')
    ax.set_xlabel('X normalizado')
    ax.set_ylabel('Y normalizado')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()

    # 2. Distribución de tamaños
    ax = axes[0, 1]
    areas = []
    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        areas.append(np.sqrt(w * h))  # Lado equivalente
    ax.hist(areas, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title('Distribución de tamaños (lado equivalente)')
    ax.set_xlabel('Tamaño (px)')
    ax.set_ylabel('Frecuencia')
    ax.axvline(32, color='r', linestyle='--', label='Small/Medium (32px)')
    ax.axvline(96, color='g', linestyle='--', label='Medium/Large (96px)')
    ax.legend()

    # 3. Aspect ratios
    ax = axes[0, 2]
    ars = []
    for ann in coco_data['annotations']:
        x, y, w, h = ann['bbox']
        if h > 0:
            ars.append(w / h)
    ax.hist(ars, bins=50, edgecolor='black', alpha=0.7, range=(0, 5))
    ax.set_title('Distribución de aspect ratios')
    ax.set_xlabel('Aspect ratio (w/h)')
    ax.set_ylabel('Frecuencia')
    ax.axvline(1, color='r', linestyle='--', label='Cuadrado')
    ax.legend()

    # 4. Objetos por imagen
    ax = axes[1, 0]
    anns_per_image = Counter(ann['image_id'] for ann in coco_data['annotations'])
    counts = list(anns_per_image.values())
    ax.hist(counts, bins=range(1, max(counts)+2), edgecolor='black', alpha=0.7)
    ax.set_title('Objetos por imagen')
    ax.set_xlabel('Número de objetos')
    ax.set_ylabel('Número de imágenes')

    # 5. Distribución de clases
    ax = axes[1, 1]
    class_counts = Counter(cat_names.get(ann['category_id'], 'UNK')
                          for ann in coco_data['annotations'])
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]
    bars = ax.bar(range(len(classes)), counts, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_title('Distribución de clases')
    ax.set_ylabel('Número de instancias')

    # 6. Tamaño relativo por clase
    ax = axes[1, 2]
    sizes_by_class = defaultdict(list)
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in img_dims:
            img_w, img_h = img_dims[img_id]
            x, y, w, h = ann['bbox']
            rel_size = np.sqrt((w * h) / (img_w * img_h))
            cat_name = cat_names.get(ann['category_id'], 'UNK')
            sizes_by_class[cat_name].append(rel_size)

    data = [sizes_by_class[c] for c in sorted(sizes_by_class.keys())]
    ax.boxplot(data, labels=sorted(sizes_by_class.keys()))
    ax.set_xticklabels(sorted(sizes_by_class.keys()), rotation=45, ha='right')
    ax.set_title('Tamaño relativo por clase')
    ax.set_ylabel('Tamaño relativo (sqrt)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualizaciones guardadas en: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Análisis profundo de calidad del dataset sintético')
    parser.add_argument('--json', type=str, required=True, help='JSON de anotaciones')
    parser.add_argument('--image-dirs', type=str, nargs='+', required=True, help='Directorios de imágenes')
    parser.add_argument('--save-dir', type=str, default='./diagnostic_output', help='Directorio para guardar')
    parser.add_argument('--num-visual-samples', type=int, default=100, help='Muestras para análisis visual')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Cargando: {args.json}")
    coco_data = load_coco_json(args.json)

    print(f"\nDataset: {len(coco_data['images'])} imágenes, {len(coco_data['annotations'])} anotaciones")
    print(f"Categorías: {[c['name'] for c in coco_data['categories']]}")

    # Ejecutar análisis
    analyze_spatial_distribution(coco_data)
    analyze_size_per_class(coco_data)
    analyze_aspect_ratios(coco_data)
    analyze_objects_per_image(coco_data)
    analyze_image_properties(coco_data, args.image_dirs, args.num_visual_samples)
    analyze_class_cooccurrence(coco_data)

    # Generar visualización
    print("\n\nGenerando visualizaciones...")
    generate_diagnostic_visualization(
        coco_data,
        args.image_dirs,
        os.path.join(args.save_dir, 'quality_analysis.png')
    )

    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"\nResultados guardados en: {args.save_dir}")
    print("\nRECOMENDACIONES si el modelo sigue sin aprender bien:")
    print("1. Si la distribución espacial está muy concentrada: diversificar posiciones")
    print("2. Si hay baja variabilidad de tamaños: añadir objetos de diferentes escalas")
    print("3. Si el brillo/contraste es muy uniforme: variar condiciones de iluminación")
    print("4. Si las clases no co-ocurren: crear imágenes con múltiples clases")
    print("5. Verificar que los bboxes están bien ajustados a los objetos")


if __name__ == '__main__':
    main()
