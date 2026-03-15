"""
Comparación visual lado a lado: imágenes sintéticas vs reales.

Ayuda a identificar diferencias de dominio que pueden afectar el aprendizaje.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

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


def get_image_stats(img_path: str) -> dict:
    """Calcula estadísticas de una imagen."""
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img)

    return {
        'size': img.size,
        'brightness': np.mean(arr),
        'contrast': np.std(arr),
        'r_mean': np.mean(arr[:,:,0]),
        'g_mean': np.mean(arr[:,:,1]),
        'b_mean': np.mean(arr[:,:,2]),
        'r_std': np.std(arr[:,:,0]),
        'g_std': np.std(arr[:,:,1]),
        'b_std': np.std(arr[:,:,2]),
    }


def compare_datasets(synth_json: str, synth_dirs: list,
                     real_json: str, real_dirs: list,
                     num_samples: int = 50):
    """Compara estadísticas entre datasets sintético y real."""

    print("="*60)
    print("COMPARACIÓN ESTADÍSTICA: SINTÉTICO vs REAL")
    print("="*60)

    synth_data = load_coco_json(synth_json)
    real_data = load_coco_json(real_json)

    # Muestrear imágenes
    synth_samples = random.sample(synth_data['images'], min(num_samples, len(synth_data['images'])))
    real_samples = random.sample(real_data['images'], min(num_samples, len(real_data['images'])))

    synth_stats = []
    real_stats = []

    print("\nAnalizando imágenes sintéticas...")
    for img_info in synth_samples:
        path = find_image(img_info['file_name'], synth_dirs)
        if path:
            try:
                synth_stats.append(get_image_stats(path))
            except:
                pass

    print("Analizando imágenes reales...")
    for img_info in real_samples:
        path = find_image(img_info['file_name'], real_dirs)
        if path:
            try:
                real_stats.append(get_image_stats(path))
            except:
                pass

    if not synth_stats or not real_stats:
        print("ERROR: No se pudieron cargar suficientes imágenes")
        return

    # Comparar estadísticas
    metrics = ['brightness', 'contrast', 'r_mean', 'g_mean', 'b_mean', 'r_std', 'g_std', 'b_std']

    print(f"\n{'Métrica':<15} {'Sintético':>12} {'Real':>12} {'Diferencia':>12} {'Status':>10}")
    print("-" * 65)

    differences = {}
    for metric in metrics:
        synth_val = np.mean([s[metric] for s in synth_stats])
        real_val = np.mean([s[metric] for s in real_stats])
        diff = synth_val - real_val
        pct_diff = abs(diff / real_val * 100) if real_val != 0 else 0

        differences[metric] = {'synth': synth_val, 'real': real_val, 'diff': diff, 'pct': pct_diff}

        status = "✓" if pct_diff < 20 else "⚠️" if pct_diff < 40 else "❌"
        print(f"{metric:<15} {synth_val:>12.1f} {real_val:>12.1f} {diff:>+12.1f} {status:>10}")

    # Diagnóstico
    print("\n" + "="*60)
    print("DIAGNÓSTICO DE DOMAIN GAP")
    print("="*60)

    issues = []

    # Brillo
    if abs(differences['brightness']['pct']) > 30:
        issues.append(f"Brillo muy diferente: sintético={differences['brightness']['synth']:.0f}, real={differences['brightness']['real']:.0f}")

    # Contraste
    if abs(differences['contrast']['pct']) > 30:
        issues.append(f"Contraste muy diferente: sintético={differences['contrast']['synth']:.0f}, real={differences['contrast']['real']:.0f}")

    # Balance de color
    synth_rgb = [differences['r_mean']['synth'], differences['g_mean']['synth'], differences['b_mean']['synth']]
    real_rgb = [differences['r_mean']['real'], differences['g_mean']['real'], differences['b_mean']['real']]

    color_diff = np.sqrt(sum((s-r)**2 for s,r in zip(synth_rgb, real_rgb)))
    if color_diff > 30:
        issues.append(f"Balance de color muy diferente (distancia RGB: {color_diff:.1f})")
        issues.append(f"  Sintético RGB: ({synth_rgb[0]:.0f}, {synth_rgb[1]:.0f}, {synth_rgb[2]:.0f})")
        issues.append(f"  Real RGB:      ({real_rgb[0]:.0f}, {real_rgb[1]:.0f}, {real_rgb[2]:.0f})")

    if issues:
        print("\n⚠️  PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"   • {issue}")
        print("\nEl domain gap visual puede estar afectando el aprendizaje.")
        print("Considera ajustar la generación sintética o usar data augmentation más agresivo.")
    else:
        print("\n✅ Las estadísticas visuales son similares entre datasets.")
        print("El problema puede estar en detalles de textura o realismo que no capturan estas métricas.")

    return differences


def visualize_side_by_side(synth_json: str, synth_dirs: list,
                           real_json: str, real_dirs: list,
                           save_path: str, num_pairs: int = 6):
    """Genera visualización lado a lado."""

    synth_data = load_coco_json(synth_json)
    real_data = load_coco_json(real_json)

    # Mapear anotaciones
    synth_anns = {}
    for ann in synth_data['annotations']:
        if ann['image_id'] not in synth_anns:
            synth_anns[ann['image_id']] = []
        synth_anns[ann['image_id']].append(ann)

    real_anns = {}
    for ann in real_data['annotations']:
        if ann['image_id'] not in real_anns:
            real_anns[ann['image_id']] = []
        real_anns[ann['image_id']].append(ann)

    synth_cat_names = {cat['id']: cat['name'] for cat in synth_data['categories']}
    real_cat_names = {cat['id']: cat['name'] for cat in real_data['categories']}

    # Seleccionar imágenes con anotaciones
    synth_with_anns = [img for img in synth_data['images'] if img['id'] in synth_anns]
    real_with_anns = [img for img in real_data['images'] if img['id'] in real_anns]

    synth_samples = random.sample(synth_with_anns, min(num_pairs, len(synth_with_anns)))
    real_samples = random.sample(real_with_anns, min(num_pairs, len(real_with_anns)))

    # Crear figura
    fig, axes = plt.subplots(num_pairs, 2, figsize=(16, 5*num_pairs))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i in range(num_pairs):
        # Sintético
        ax_synth = axes[i, 0]
        synth_img_info = synth_samples[i]
        synth_path = find_image(synth_img_info['file_name'], synth_dirs)

        if synth_path:
            img = Image.open(synth_path)
            ax_synth.imshow(img)

            for ann in synth_anns.get(synth_img_info['id'], []):
                x, y, w, h = ann['bbox']
                cat_name = synth_cat_names.get(ann['category_id'], 'UNK')
                color = colors[ann['category_id'] % 10]

                rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                         edgecolor=color, facecolor='none')
                ax_synth.add_patch(rect)
                ax_synth.text(x, y-3, cat_name, color='white', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

        ax_synth.set_title(f'SINTÉTICO: {synth_img_info["file_name"]}', fontsize=10)
        ax_synth.axis('off')

        # Real
        ax_real = axes[i, 1]
        real_img_info = real_samples[i]
        real_path = find_image(real_img_info['file_name'], real_dirs)

        if real_path:
            img = Image.open(real_path)
            ax_real.imshow(img)

            for ann in real_anns.get(real_img_info['id'], []):
                x, y, w, h = ann['bbox']
                cat_name = real_cat_names.get(ann['category_id'], 'UNK')
                color = colors[ann['category_id'] % 10]

                rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                         edgecolor=color, facecolor='none')
                ax_real.add_patch(rect)
                ax_real.text(x, y-3, cat_name, color='white', fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

        ax_real.set_title(f'REAL: {real_img_info["file_name"]}', fontsize=10)
        ax_real.axis('off')

    plt.suptitle('Comparación: Sintético (izquierda) vs Real (derecha)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualizaciones guardadas en: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Comparar dataset sintético vs real')
    parser.add_argument('--synth-json', type=str, required=True, help='JSON sintético')
    parser.add_argument('--synth-dirs', type=str, nargs='+', required=True, help='Dirs imágenes sintéticas')
    parser.add_argument('--real-json', type=str, required=True, help='JSON real')
    parser.add_argument('--real-dirs', type=str, nargs='+', required=True, help='Dirs imágenes reales')
    parser.add_argument('--save-dir', type=str, default='./diagnostic_output', help='Directorio de salida')
    parser.add_argument('--num-samples', type=int, default=50, help='Muestras para estadísticas')
    parser.add_argument('--num-visual', type=int, default=6, help='Pares para visualización')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Comparación estadística
    compare_datasets(
        args.synth_json, args.synth_dirs,
        args.real_json, args.real_dirs,
        args.num_samples
    )

    # Visualización lado a lado
    print("\n\nGenerando comparación visual...")
    visualize_side_by_side(
        args.synth_json, args.synth_dirs,
        args.real_json, args.real_dirs,
        os.path.join(args.save_dir, 'synthetic_vs_real.png'),
        args.num_visual
    )

    print("\n" + "="*60)
    print("SIGUIENTE PASO")
    print("="*60)
    print("\nRevisa la imagen 'synthetic_vs_real.png' y observa:")
    print("1. ¿Los objetos sintéticos se ven realistas?")
    print("2. ¿Los fondos son similares?")
    print("3. ¿Las condiciones de iluminación son comparables?")
    print("4. ¿Los bboxes están bien ajustados en ambos casos?")


if __name__ == '__main__':
    main()
