"""
Script de validación de calidad de dataset sintético.

Analiza:
1. Distribución de tamaños de bounding boxes
2. Aspect ratios de objetos
3. Densidad de objetos por imagen
4. Detección de bboxes anómalas (muy pequeñas, muy grandes, fuera de imagen)
5. Distribución de clases
6. Overlap/oclusión entre objetos
7. Comparación con dataset real
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from PIL import Image
from tqdm import tqdm


def load_coco_dataset(ann_file):
    """Carga annotations COCO."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data


def compute_iou(box1, box2):
    """Calcula IoU entre dos boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def analyze_dataset(ann_file, img_folder=None, name="Dataset"):
    """Analiza un dataset COCO."""

    print(f"\n{'='*80}")
    print(f"ANÁLISIS DE CALIDAD: {name}")
    print(f"{'='*80}\n")

    data = load_coco_dataset(ann_file)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    annotations = data['annotations']

    # Estadísticas básicas
    print(f"📊 ESTADÍSTICAS BÁSICAS")
    print(f"{'─'*80}")
    print(f"  Imágenes:     {len(images)}")
    print(f"  Anotaciones:  {len(annotations)}")
    print(f"  Clases:       {len(categories)}")
    print(f"  Objetos/img:  {len(annotations)/len(images):.2f} promedio")

    # Agrupar anotaciones por imagen
    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann['image_id']].append(ann)

    # 1. Distribución de objetos por imagen
    objects_per_image = [len(anns) for anns in anns_by_image.values()]

    print(f"\n📦 DISTRIBUCIÓN DE OBJETOS POR IMAGEN")
    print(f"{'─'*80}")
    print(f"  Min:     {min(objects_per_image)}")
    print(f"  Max:     {max(objects_per_image)}")
    print(f"  Media:   {np.mean(objects_per_image):.2f}")
    print(f"  Mediana: {np.median(objects_per_image):.0f}")
    print(f"  StdDev:  {np.std(objects_per_image):.2f}")

    # 2. Distribución de clases
    class_counts = Counter([ann['category_id'] for ann in annotations])

    print(f"\n🏷️  DISTRIBUCIÓN DE CLASES")
    print(f"{'─'*80}")
    total_anns = len(annotations)
    for cat_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        cat_name = categories.get(cat_id, f"Unknown_{cat_id}")
        percentage = (count / total_anns) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {cat_name:20s} {count:6d} ({percentage:5.1f}%) {bar}")

    # Calcular desbalance
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n  Ratio de desbalance: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        print(f"  ⚠️  WARNING: Alto desbalance de clases (>{imbalance_ratio:.0f}:1)")

    # 3. Análisis de tamaños de bounding boxes
    box_areas = []
    box_widths = []
    box_heights = []
    aspect_ratios = []
    relative_areas = []  # Área relativa al tamaño de imagen

    anomalies = {
        'very_small': [],      # Área < 32x32 píxeles
        'very_large': [],      # Área > 80% de imagen
        'extreme_aspect': [],  # Aspect ratio > 5:1 o < 1:5
        'out_of_bounds': [],   # Box fuera de límites de imagen
        'zero_area': []        # Área = 0
    }

    for ann in tqdm(annotations, desc="Analizando bounding boxes"):
        x, y, w, h = ann['bbox']
        area = w * h

        # Obtener tamaño de imagen
        img_id = ann['image_id']
        if img_id in images:
            img_w = images[img_id].get('width', 640)
            img_h = images[img_id].get('height', 640)
            img_area = img_w * img_h

            # Área relativa
            rel_area = area / img_area if img_area > 0 else 0
            relative_areas.append(rel_area)

            # Detectar anomalías
            if area < 32*32:  # Muy pequeño
                anomalies['very_small'].append((ann['id'], area, categories.get(ann['category_id'])))

            if rel_area > 0.8:  # Muy grande (>80% de imagen)
                anomalies['very_large'].append((ann['id'], rel_area, categories.get(ann['category_id'])))

            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                anomalies['out_of_bounds'].append((ann['id'], (x, y, w, h), (img_w, img_h)))

        if area == 0:
            anomalies['zero_area'].append(ann['id'])
            continue

        box_areas.append(area)
        box_widths.append(w)
        box_heights.append(h)

        aspect_ratio = w / h if h > 0 else 0
        aspect_ratios.append(aspect_ratio)

        if aspect_ratio > 5 or aspect_ratio < 0.2:
            anomalies['extreme_aspect'].append((ann['id'], aspect_ratio, categories.get(ann['category_id'])))

    # Estadísticas de tamaños
    print(f"\n📏 TAMAÑOS DE BOUNDING BOXES")
    print(f"{'─'*80}")
    print(f"  Área (píxeles²):")
    print(f"    Min:     {min(box_areas):.0f}")
    print(f"    Max:     {max(box_areas):.0f}")
    print(f"    Media:   {np.mean(box_areas):.0f}")
    print(f"    Mediana: {np.median(box_areas):.0f}")

    print(f"\n  Ancho (píxeles):")
    print(f"    Min:     {min(box_widths):.0f}")
    print(f"    Max:     {max(box_widths):.0f}")
    print(f"    Media:   {np.mean(box_widths):.0f}")

    print(f"\n  Alto (píxeles):")
    print(f"    Min:     {min(box_heights):.0f}")
    print(f"    Max:     {max(box_heights):.0f}")
    print(f"    Media:   {np.mean(box_heights):.0f}")

    print(f"\n  Aspect Ratio (W/H):")
    print(f"    Min:     {min(aspect_ratios):.2f}")
    print(f"    Max:     {max(aspect_ratios):.2f}")
    print(f"    Media:   {np.mean(aspect_ratios):.2f}")

    print(f"\n  Área Relativa (% de imagen):")
    print(f"    Min:     {min(relative_areas)*100:.2f}%")
    print(f"    Max:     {max(relative_areas)*100:.2f}%")
    print(f"    Media:   {np.mean(relative_areas)*100:.2f}%")

    # 4. Reportar anomalías
    print(f"\n⚠️  ANOMALÍAS DETECTADAS")
    print(f"{'─'*80}")

    total_anomalies = sum(len(v) for v in anomalies.values())

    if anomalies['very_small']:
        print(f"  📦 Objetos muy pequeños (<32x32px): {len(anomalies['very_small'])}")
        print(f"     Ejemplos:")
        for ann_id, area, cat_name in anomalies['very_small'][:5]:
            print(f"       - ID {ann_id}: {cat_name} ({area:.0f}px²)")

    if anomalies['very_large']:
        print(f"  📦 Objetos muy grandes (>80% imagen): {len(anomalies['very_large'])}")
        print(f"     Ejemplos:")
        for ann_id, rel_area, cat_name in anomalies['very_large'][:5]:
            print(f"       - ID {ann_id}: {cat_name} ({rel_area*100:.1f}%)")

    if anomalies['extreme_aspect']:
        print(f"  📐 Aspect ratios extremos (>5:1 o <1:5): {len(anomalies['extreme_aspect'])}")
        print(f"     Ejemplos:")
        for ann_id, ar, cat_name in anomalies['extreme_aspect'][:5]:
            print(f"       - ID {ann_id}: {cat_name} (AR={ar:.2f})")

    if anomalies['out_of_bounds']:
        print(f"  🚫 Boxes fuera de límites: {len(anomalies['out_of_bounds'])}")
        print(f"     Ejemplos:")
        for ann_id, bbox, img_size in anomalies['out_of_bounds'][:5]:
            print(f"       - ID {ann_id}: bbox={bbox}, img_size={img_size}")

    if anomalies['zero_area']:
        print(f"  ❌ Boxes con área cero: {len(anomalies['zero_area'])}")

    if total_anomalies == 0:
        print(f"  ✅ No se detectaron anomalías")
    else:
        print(f"\n  Total de anomalías: {total_anomalies} ({total_anomalies/len(annotations)*100:.2f}%)")

    # 5. Análisis de overlap/oclusión
    print(f"\n🔄 ANÁLISIS DE OCLUSIÓN/OVERLAP")
    print(f"{'─'*80}")

    overlap_stats = []
    high_overlap_images = []

    for img_id, img_anns in tqdm(list(anns_by_image.items())[:500], desc="Calculando IoU"):
        if len(img_anns) < 2:
            continue

        # Calcular IoU entre todos los pares
        ious = []
        for i, ann1 in enumerate(img_anns):
            for ann2 in img_anns[i+1:]:
                iou = compute_iou(ann1['bbox'], ann2['bbox'])
                if iou > 0:
                    ious.append(iou)

        if ious:
            max_iou = max(ious)
            avg_iou = np.mean(ious)
            overlap_stats.append((img_id, max_iou, avg_iou, len(ious)))

            if max_iou > 0.5:  # Alto overlap
                high_overlap_images.append((img_id, max_iou))

    if overlap_stats:
        max_ious = [s[1] for s in overlap_stats]
        avg_ious = [s[2] for s in overlap_stats]

        print(f"  IoU máximo por imagen:")
        print(f"    Media:   {np.mean(max_ious):.3f}")
        print(f"    Max:     {max(max_ious):.3f}")
        print(f"    Mediana: {np.median(max_ious):.3f}")

        print(f"\n  Imágenes con alto overlap (IoU>0.5): {len(high_overlap_images)}")
        if high_overlap_images:
            print(f"  Ejemplos:")
            for img_id, max_iou in sorted(high_overlap_images, key=lambda x: x[1], reverse=True)[:5]:
                img_name = images[img_id].get('file_name', f'img_{img_id}')
                print(f"    - {img_name}: IoU={max_iou:.3f}")

    # 6. Verificar imágenes físicas (si img_folder provisto)
    if img_folder:
        print(f"\n🖼️  VERIFICACIÓN DE IMÁGENES FÍSICAS")
        print(f"{'─'*80}")

        img_folder = Path(img_folder)
        missing_images = []
        corrupt_images = []
        size_mismatches = []

        for img_id, img_info in tqdm(list(images.items())[:500], desc="Verificando imágenes"):
            img_path = img_folder / img_info['file_name']

            if not img_path.exists():
                missing_images.append(img_info['file_name'])
                continue

            try:
                with Image.open(img_path) as pil_img:
                    w, h = pil_img.size

                    # Verificar que tamaño coincida con annotation
                    ann_w = img_info.get('width', w)
                    ann_h = img_info.get('height', h)

                    if w != ann_w or h != ann_h:
                        size_mismatches.append((img_info['file_name'], (w, h), (ann_w, ann_h)))

            except Exception as e:
                corrupt_images.append((img_info['file_name'], str(e)))

        print(f"  Imágenes faltantes: {len(missing_images)}")
        if missing_images[:3]:
            print(f"    Ejemplos: {', '.join(missing_images[:3])}")

        print(f"  Imágenes corruptas: {len(corrupt_images)}")
        if corrupt_images[:3]:
            for fname, err in corrupt_images[:3]:
                print(f"    - {fname}: {err}")

        print(f"  Tamaños inconsistentes: {len(size_mismatches)}")
        if size_mismatches[:3]:
            for fname, real_size, ann_size in size_mismatches[:3]:
                print(f"    - {fname}: real={real_size}, annotation={ann_size}")

    return {
        'num_images': len(images),
        'num_annotations': len(annotations),
        'objects_per_image': np.mean(objects_per_image),
        'class_counts': class_counts,
        'imbalance_ratio': imbalance_ratio,
        'box_areas': box_areas,
        'box_widths': box_widths,
        'box_heights': box_heights,
        'aspect_ratios': aspect_ratios,
        'relative_areas': relative_areas,
        'anomalies': {k: len(v) for k, v in anomalies.items()},
        'overlap_stats': overlap_stats
    }


def compare_datasets(synth_stats, real_stats):
    """Compara estadísticas de datasets sintético vs real."""

    print(f"\n{'='*80}")
    print(f"COMPARACIÓN SINTÉTICO vs REAL")
    print(f"{'='*80}\n")

    print(f"📊 ESTADÍSTICAS COMPARATIVAS")
    print(f"{'─'*80}")
    print(f"{'Métrica':<30} {'Sintético':>15} {'Real':>15} {'Ratio':>10}")
    print(f"{'─'*80}")

    # Comparar tamaños
    synth_mean_area = np.mean(synth_stats['box_areas'])
    real_mean_area = np.mean(real_stats['box_areas'])
    print(f"{'Área promedio bbox (px²)':<30} {synth_mean_area:>15.0f} {real_mean_area:>15.0f} {synth_mean_area/real_mean_area:>10.2f}x")

    synth_mean_obj = synth_stats['objects_per_image']
    real_mean_obj = real_stats['objects_per_image']
    print(f"{'Objetos por imagen':<30} {synth_mean_obj:>15.2f} {real_mean_obj:>15.2f} {synth_mean_obj/real_mean_obj:>10.2f}x")

    synth_imb = synth_stats['imbalance_ratio']
    real_imb = real_stats['imbalance_ratio']
    print(f"{'Desbalance de clases':<30} {synth_imb:>15.1f}:1 {real_imb:>15.1f}:1 {synth_imb/real_imb:>10.2f}x")

    # Aspect ratios
    synth_mean_ar = np.mean(synth_stats['aspect_ratios'])
    real_mean_ar = np.mean(real_stats['aspect_ratios'])
    print(f"{'Aspect ratio promedio':<30} {synth_mean_ar:>15.2f} {real_mean_ar:>15.2f} {synth_mean_ar/real_mean_ar:>10.2f}x")

    # Área relativa
    synth_mean_rel = np.mean(synth_stats['relative_areas']) * 100
    real_mean_rel = np.mean(real_stats['relative_areas']) * 100
    print(f"{'Área relativa promedio (%)':<30} {synth_mean_rel:>15.2f} {real_mean_rel:>15.2f} {synth_mean_rel/real_mean_rel:>10.2f}x")

    # Anomalías
    print(f"\n⚠️  ANOMALÍAS")
    print(f"{'─'*80}")
    synth_anom_total = sum(synth_stats['anomalies'].values())
    real_anom_total = sum(real_stats['anomalies'].values())
    synth_anom_pct = synth_anom_total / synth_stats['num_annotations'] * 100
    real_anom_pct = real_anom_total / real_stats['num_annotations'] * 100

    print(f"{'Total anomalías (%)':<30} {synth_anom_pct:>15.2f}% {real_anom_pct:>15.2f}% {synth_anom_pct/real_anom_pct if real_anom_pct > 0 else 0:>10.2f}x")

    for key in synth_stats['anomalies'].keys():
        synth_val = synth_stats['anomalies'][key]
        real_val = real_stats['anomalies'][key]
        print(f"  {key:<28} {synth_val:>15d} {real_val:>15d}")

    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES")
    print(f"{'─'*80}")

    issues = []

    if synth_mean_area / real_mean_area > 1.5:
        issues.append("⚠️  Objetos sintéticos son significativamente MÁS GRANDES que reales")
        issues.append("   → Reducir escala de objetos en generación sintética")
    elif synth_mean_area / real_mean_area < 0.7:
        issues.append("⚠️  Objetos sintéticos son significativamente MÁS PEQUEÑOS que reales")
        issues.append("   → Aumentar escala de objetos en generación sintética")

    if synth_mean_obj / real_mean_obj > 1.5:
        issues.append("⚠️  Demasiados objetos por imagen en sintéticos")
        issues.append("   → Reducir densidad de objetos en escenas sintéticas")
    elif synth_mean_obj / real_mean_obj < 0.5:
        issues.append("⚠️  Muy pocos objetos por imagen en sintéticos")
        issues.append("   → Aumentar densidad de objetos en escenas sintéticas")

    if synth_imb / real_imb > 1.5:
        issues.append("⚠️  Desbalance de clases MAYOR en sintéticos que en reales")
        issues.append("   → Balancear mejor distribución de clases en generación")

    if synth_anom_pct > real_anom_pct * 2:
        issues.append("⚠️  MUCHAS más anomalías en sintéticos que en reales")
        issues.append("   → Revisar proceso de generación y exportación de annotations")

    if not issues:
        print("  ✅ Distribuciones estadísticas son consistentes entre sintético y real")
    else:
        for issue in issues:
            print(f"  {issue}")


def plot_distributions(synth_stats, real_stats, output_dir):
    """Genera gráficos comparativos."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparación Dataset Sintético vs Real', fontsize=16, fontweight='bold')

    # 1. Distribución de áreas
    ax = axes[0, 0]
    ax.hist(synth_stats['box_areas'], bins=50, alpha=0.5, label='Sintético', density=True)
    ax.hist(real_stats['box_areas'], bins=50, alpha=0.5, label='Real', density=True)
    ax.set_xlabel('Área bbox (px²)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Áreas de Bounding Boxes')
    ax.legend()
    ax.set_xlim(0, np.percentile(synth_stats['box_areas'] + real_stats['box_areas'], 95))

    # 2. Distribución de aspect ratios
    ax = axes[0, 1]
    ax.hist(synth_stats['aspect_ratios'], bins=50, alpha=0.5, label='Sintético', density=True)
    ax.hist(real_stats['aspect_ratios'], bins=50, alpha=0.5, label='Real', density=True)
    ax.set_xlabel('Aspect Ratio (W/H)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Aspect Ratios')
    ax.legend()
    ax.set_xlim(0, 5)

    # 3. Distribución de áreas relativas
    ax = axes[0, 2]
    synth_rel = [x*100 for x in synth_stats['relative_areas']]
    real_rel = [x*100 for x in real_stats['relative_areas']]
    ax.hist(synth_rel, bins=50, alpha=0.5, label='Sintético', density=True)
    ax.hist(real_rel, bins=50, alpha=0.5, label='Real', density=True)
    ax.set_xlabel('Área Relativa (% de imagen)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Áreas Relativas')
    ax.legend()
    ax.set_xlim(0, 50)

    # 4. Distribución de anchos
    ax = axes[1, 0]
    ax.hist(synth_stats['box_widths'], bins=50, alpha=0.5, label='Sintético', density=True)
    ax.hist(real_stats['box_widths'], bins=50, alpha=0.5, label='Real', density=True)
    ax.set_xlabel('Ancho bbox (px)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Anchos')
    ax.legend()
    ax.set_xlim(0, np.percentile(synth_stats['box_widths'] + real_stats['box_widths'], 95))

    # 5. Distribución de altos
    ax = axes[1, 1]
    ax.hist(synth_stats['box_heights'], bins=50, alpha=0.5, label='Sintético', density=True)
    ax.hist(real_stats['box_heights'], bins=50, alpha=0.5, label='Real', density=True)
    ax.set_xlabel('Alto bbox (px)')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Altos')
    ax.legend()
    ax.set_xlim(0, np.percentile(synth_stats['box_heights'] + real_stats['box_heights'], 95))

    # 6. Comparación de clases
    ax = axes[1, 2]
    synth_total = sum(synth_stats['class_counts'].values())
    real_total = sum(real_stats['class_counts'].values())

    # Normalizar a porcentajes
    synth_pcts = {k: v/synth_total*100 for k, v in synth_stats['class_counts'].items()}
    real_pcts = {k: v/real_total*100 for k, v in real_stats['class_counts'].items()}

    all_classes = sorted(set(synth_pcts.keys()) | set(real_pcts.keys()))
    x = np.arange(len(all_classes))
    width = 0.35

    synth_vals = [synth_pcts.get(c, 0) for c in all_classes]
    real_vals = [real_pcts.get(c, 0) for c in all_classes]

    ax.bar(x - width/2, synth_vals, width, label='Sintético', alpha=0.8)
    ax.bar(x + width/2, real_vals, width, label='Real', alpha=0.8)
    ax.set_xlabel('Clase')
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title('Distribución de Clases (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'synthetic_vs_real_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Gráficos guardados en: {output_dir / 'synthetic_vs_real_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description='Validar calidad de dataset sintético')
    parser.add_argument('--synthetic-ann', required=True, help='Annotation file sintético')
    parser.add_argument('--real-ann', required=True, help='Annotation file real')
    parser.add_argument('--synthetic-img', help='Carpeta de imágenes sintéticas (opcional)')
    parser.add_argument('--real-img', help='Carpeta de imágenes reales (opcional)')
    parser.add_argument('--output-dir', default='./dataset_quality_report', help='Directorio de salida')

    args = parser.parse_args()

    # Analizar sintético
    synth_stats = analyze_dataset(args.synthetic_ann, args.synthetic_img, name="SINTÉTICO")

    # Analizar real
    real_stats = analyze_dataset(args.real_ann, args.real_img, name="REAL")

    # Comparar
    compare_datasets(synth_stats, real_stats)

    # Generar gráficos
    plot_distributions(synth_stats, real_stats, args.output_dir)

    print(f"\n{'='*80}")
    print(f"✅ ANÁLISIS COMPLETADO")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
