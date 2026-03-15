"""
Test rápido: ¿El modelo puede al menos sobreajustarse a un subset pequeño de datos sintéticos?

Si el modelo NO puede sobreajustarse a 100 imágenes en 50 épocas, hay un problema fundamental
con los datos o la arquitectura.

Si SÍ puede sobreajustarse, el problema es de generalización/domain gap.
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path


def create_overfit_subset(
    original_json: str,
    output_json: str,
    num_images: int = 100
):
    """Crea un subset pequeño para test de overfitting."""

    with open(original_json, 'r') as f:
        data = json.load(f)

    # Seleccionar imágenes aleatorias
    selected_images = random.sample(data['images'], min(num_images, len(data['images'])))
    selected_ids = {img['id'] for img in selected_images}

    # Filtrar anotaciones
    selected_anns = [ann for ann in data['annotations'] if ann['image_id'] in selected_ids]

    # Crear nuevo JSON
    subset_data = {
        'images': selected_images,
        'annotations': selected_anns,
        'categories': data['categories']
    }

    # Guardar
    with open(output_json, 'w') as f:
        json.dump(subset_data, f)

    print(f"Subset creado: {len(selected_images)} imágenes, {len(selected_anns)} anotaciones")
    print(f"Guardado en: {output_json}")

    return subset_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=str, required=True)
    parser.add_argument('--output-json', type=str, required=True)
    parser.add_argument('--num-images', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    create_overfit_subset(args.input_json, args.output_json, args.num_images)

    print("\n" + "="*60)
    print("INSTRUCCIONES PARA TEST DE OVERFITTING")
    print("="*60)
    print("""
1. Crea una configuración temporal que use este subset para train Y val
2. Entrena por 50 épocas sin data augmentation
3. Observa si el loss baja a ~0 y el mAP sube a >0.8

Si el modelo puede sobreajustarse:
  → El problema es domain gap / generalización
  → Solución: Mejorar generación sintética o usar mixto real+sintético

Si el modelo NO puede sobreajustarse:
  → Hay un problema fundamental con los datos
  → Revisar: bboxes mal alineados, objetos invisibles, ruido en etiquetas
""")


if __name__ == '__main__':
    main()
