"""
Visualiza muestras aleatorias del dataset con bounding boxes.

Útil para auditoría visual de:
- Calidad de imágenes sintéticas
- Alineación de bounding boxes
- Categorías correctas
- Detección de artefactos
"""

import json
import random
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from collections import defaultdict


# Colores por clase (para visualización)
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
]


def load_coco_dataset(ann_file):
    """Carga annotations COCO."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data


def visualize_samples(ann_file, img_folder, output_dir, num_samples=50, seed=42):
    """Visualiza muestras aleatorias con bounding boxes."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_coco_dataset(ann_file)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    # Agrupar anotaciones por imagen
    anns_by_image = defaultdict(list)
    for ann in data['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    # Seleccionar muestras aleatorias
    random.seed(seed)
    sample_ids = random.sample(list(anns_by_image.keys()), min(num_samples, len(anns_by_image)))

    print(f"Visualizando {len(sample_ids)} muestras aleatorias...")
    print(f"Salida: {output_dir}")

    img_folder = Path(img_folder)

    # Estadísticas de visualización
    stats = {
        'missing_images': [],
        'corrupt_images': [],
        'successful': 0,
        'total_objects': 0,
        'class_counts': defaultdict(int)
    }

    for idx, img_id in enumerate(sample_ids):
        img_info = images[img_id]
        img_path = img_folder / img_info['file_name']

        if not img_path.exists():
            stats['missing_images'].append(img_info['file_name'])
            print(f"⚠️  Imagen no encontrada: {img_info['file_name']}")
            continue

        try:
            # Cargar imagen
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            # Obtener anotaciones
            img_anns = anns_by_image[img_id]

            # Crear figura
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.imshow(img_array)

            # Dibujar bounding boxes
            for ann in img_anns:
                x, y, w, h = ann['bbox']
                cat_id = ann['category_id']
                cat_name = categories.get(cat_id, f"Unknown_{cat_id}")

                # Color por clase
                color = COLORS[cat_id % len(COLORS)]

                # Dibujar rectángulo
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none',
                    linestyle='-'
                )
                ax.add_patch(rect)

                # Etiqueta
                label = f"{cat_name}"
                ax.text(
                    x, y - 5,
                    label,
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
                )

                stats['class_counts'][cat_name] += 1
                stats['total_objects'] += 1

            # Título con info
            title = f"{img_info['file_name']}\n"
            title += f"Objetos: {len(img_anns)} | "
            title += f"Resolución: {img.width}x{img.height}"

            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.axis('off')

            # Guardar
            output_path = output_dir / f"sample_{idx:03d}_{img_info['file_name']}"
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()

            stats['successful'] += 1

            if (idx + 1) % 10 == 0:
                print(f"  Procesadas: {idx + 1}/{len(sample_ids)}")

        except Exception as e:
            stats['corrupt_images'].append((img_info['file_name'], str(e)))
            print(f"❌ Error procesando {img_info['file_name']}: {e}")

    # Reporte final
    print(f"\n{'='*80}")
    print(f"REPORTE DE VISUALIZACIÓN")
    print(f"{'='*80}\n")

    print(f"✅ Imágenes procesadas exitosamente: {stats['successful']}")
    print(f"❌ Imágenes faltantes: {len(stats['missing_images'])}")
    print(f"❌ Imágenes corruptas: {len(stats['corrupt_images'])}")
    print(f"\n📦 Total de objetos visualizados: {stats['total_objects']}")

    print(f"\n🏷️  Distribución de clases:")
    for cat_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat_name:20s} {count:5d}")

    if stats['missing_images']:
        print(f"\n⚠️  Imágenes faltantes:")
        for fname in stats['missing_images'][:10]:
            print(f"  - {fname}")
        if len(stats['missing_images']) > 10:
            print(f"  ... y {len(stats['missing_images'])-10} más")

    if stats['corrupt_images']:
        print(f"\n❌ Imágenes corruptas:")
        for fname, err in stats['corrupt_images'][:5]:
            print(f"  - {fname}: {err}")

    print(f"\n✅ Visualizaciones guardadas en: {output_dir}")
    print(f"{'='*80}\n")


def create_comparison_grid(synth_ann, synth_img, real_ann, real_img, output_dir, num_pairs=5, seed=42):
    """Crea grid comparativo sintético vs real lado a lado."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datasets
    synth_data = load_coco_dataset(synth_ann)
    real_data = load_coco_dataset(real_ann)

    synth_images = {img['id']: img for img in synth_data['images']}
    real_images = {img['id']: img for img in real_data['images']}

    synth_cats = {cat['id']: cat['name'] for cat in synth_data['categories']}
    real_cats = {cat['id']: cat['name'] for cat in real_data['categories']}

    # Agrupar anotaciones
    synth_anns = defaultdict(list)
    for ann in synth_data['annotations']:
        synth_anns[ann['image_id']].append(ann)

    real_anns = defaultdict(list)
    for ann in real_data['annotations']:
        real_anns[ann['image_id']].append(ann)

    # Seleccionar muestras aleatorias
    random.seed(seed)
    synth_samples = random.sample(list(synth_anns.keys()), min(num_pairs, len(synth_anns)))
    real_samples = random.sample(list(real_anns.keys()), min(num_pairs, len(real_anns)))

    synth_img = Path(synth_img)
    real_img = Path(real_img)

    print(f"Creando grid comparativo ({num_pairs} pares)...")

    for idx in range(num_pairs):
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'Comparación Sintético vs Real - Muestra {idx+1}', fontsize=16, fontweight='bold')

        # Sintético
        synth_id = synth_samples[idx]
        synth_info = synth_images[synth_id]
        synth_path = synth_img / synth_info['file_name']

        if synth_path.exists():
            synth_img_data = np.array(Image.open(synth_path).convert('RGB'))
            axes[0].imshow(synth_img_data)

            for ann in synth_anns[synth_id]:
                x, y, w, h = ann['bbox']
                cat_name = synth_cats.get(ann['category_id'], "Unknown")
                color = COLORS[ann['category_id'] % len(COLORS)]

                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                axes[0].add_patch(rect)
                axes[0].text(x, y-5, cat_name, color='white', fontsize=9, fontweight='bold',
                           bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1))

            axes[0].set_title(f"SINTÉTICO\n{synth_info['file_name']}\nObjetos: {len(synth_anns[synth_id])}")
        else:
            axes[0].text(0.5, 0.5, 'Imagen no encontrada', ha='center', va='center', fontsize=14)
            axes[0].set_title("SINTÉTICO (faltante)")

        axes[0].axis('off')

        # Real
        real_id = real_samples[idx]
        real_info = real_images[real_id]
        real_path = real_img / real_info['file_name']

        if real_path.exists():
            real_img_data = np.array(Image.open(real_path).convert('RGB'))
            axes[1].imshow(real_img_data)

            for ann in real_anns[real_id]:
                x, y, w, h = ann['bbox']
                cat_name = real_cats.get(ann['category_id'], "Unknown")
                color = COLORS[ann['category_id'] % len(COLORS)]

                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                axes[1].add_patch(rect)
                axes[1].text(x, y-5, cat_name, color='white', fontsize=9, fontweight='bold',
                           bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1))

            axes[1].set_title(f"REAL\n{real_info['file_name']}\nObjetos: {len(real_anns[real_id])}")
        else:
            axes[1].text(0.5, 0.5, 'Imagen no encontrada', ha='center', va='center', fontsize=14)
            axes[1].set_title("REAL (faltante)")

        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'comparison_{idx:02d}.png', dpi=100, bbox_inches='tight')
        plt.close()

    print(f"✅ Comparaciones guardadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualizar muestras del dataset')
    parser.add_argument('--ann-file', required=True, help='Annotation file')
    parser.add_argument('--img-folder', required=True, help='Carpeta de imágenes')
    parser.add_argument('--output-dir', default='./dataset_visualization', help='Directorio de salida')
    parser.add_argument('--num-samples', type=int, default=50, help='Número de muestras a visualizar')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reproducibilidad')

    # Opcional: comparación sintético vs real
    parser.add_argument('--compare', action='store_true', help='Crear comparación sintético vs real')
    parser.add_argument('--real-ann', help='Annotation file real (para comparación)')
    parser.add_argument('--real-img', help='Carpeta de imágenes reales (para comparación)')
    parser.add_argument('--num-pairs', type=int, default=10, help='Número de pares a comparar')

    args = parser.parse_args()

    # Visualizar muestras
    visualize_samples(
        args.ann_file,
        args.img_folder,
        args.output_dir,
        args.num_samples,
        args.seed
    )

    # Comparación opcional
    if args.compare:
        if not args.real_ann or not args.real_img:
            print("⚠️  Para comparación, especifica --real-ann y --real-img")
        else:
            comparison_dir = Path(args.output_dir) / 'comparison'
            create_comparison_grid(
                args.ann_file,
                args.img_folder,
                args.real_ann,
                args.real_img,
                comparison_dir,
                args.num_pairs,
                args.seed
            )


if __name__ == '__main__':
    main()
