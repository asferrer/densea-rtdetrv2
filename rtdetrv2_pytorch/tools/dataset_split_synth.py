#!/usr/bin/env python3
import json
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import copy

def plot_class_distribution(split_name, coco_subset, cat_id_to_name, categories, output_dir):
    """
    Calcula y guarda un gráfico de barras con el número total de instancias
    por clase para el split dado.
    """
    # Inicializar conteos
    all_cat_names = sorted([cat['name'] for cat in categories])
    counts = {name: 0 for name in all_cat_names}

    # Contar instancias
    for ann in coco_subset.get("annotations", []):
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id)
        if cat_name in counts:
            counts[cat_name] += 1

    names = list(counts.keys())
    values = np.array(list(counts.values()))

    # Configuración del gráfico
    title = f"Distribución de instancias por clase en {split_name}"

    fig, ax = plt.subplots(figsize=(14, 8))

    # Crear las barras
    bar_real = ax.bar(names, values, label='Instancias', color='dodgerblue')

    ax.set_title(title)
    ax.set_xlabel("Clases")
    ax.set_ylabel("Número de instancias")

    # Añadir el número total sobre cada barra
    for i, total in enumerate(values):
        if total > 0:
            ax.annotate(f"{int(total)}",
                        xy=(i, total),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Guardar el gráfico
    plot_filename = os.path.join(output_dir, f"{split_name.lower()}_distribution.png")
    plt.savefig(plot_filename)
    plt.close()


def filter_coco_by_categories(coco_data, allowed_class_ids):
    """
    Filtra un dataset COCO para mantener solo las imágenes y anotaciones
    correspondientes a un conjunto de IDs de categoría permitidos.
    """
    # Filtrar anotaciones
    annotations_filtered = [
        ann for ann in coco_data.get("annotations", [])
        if ann["category_id"] in allowed_class_ids
    ]
    
    # Obtener los IDs de las imágenes que tienen esas anotaciones
    image_ids_with_allowed_anns = {ann["image_id"] for ann in annotations_filtered}
    
    # Filtrar imágenes
    images_filtered = [
        img for img in coco_data.get("images", [])
        if img["id"] in image_ids_with_allowed_anns
    ]

    # Crear una copia filtrada del dataset
    coco_filtered = copy.deepcopy(coco_data)
    coco_filtered["annotations"] = annotations_filtered
    coco_filtered["images"] = images_filtered
    
    # Mantener solo las categorías permitidas en la lista de categorías
    coco_filtered["categories"] = [
        cat for cat in coco_data.get("categories", [])
        if cat["id"] in allowed_class_ids
    ]
    
    return coco_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Script para dividir un dataset COCO. El dataset de entrenamiento se toma de --train_input, "
                    "mientras que --real_input se divide en conjuntos de validación y prueba."
    )
    # Argumentos modificados para mayor claridad
    parser.add_argument("--real_input", type=str, required=True,
                        help="Ruta al archivo COCO JSON para los conjuntos de validación y prueba (datos reales).")
    parser.add_argument("--train_input", type=str, required=True,
                        help="Ruta al archivo COCO JSON que se usará como conjunto de entrenamiento.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directorio de salida para los JSON y las gráficas.")
    parser.add_argument("--test_size", type=float, default=0.5,
                        help="Fracción de --real_input para el conjunto de test (default: 0.5, es decir, 50% de los datos reales).")
    parser.add_argument("--val_size", type=float, default=0.5,
                        help="Fracción de --real_input para el conjunto de validación (default: 0.5, es decir, 50% de los datos reales).")
    parser.add_argument("--seed", type=int, default=21,
                        help="Semilla para aleatorización (default: 21).")
    parser.add_argument("--min_instances", type=int, default=5,
                        help="Número mínimo de instancias en --real_input por clase para ser considerada (default: 5).")
    parser.add_argument("--exclude_classes", type=str, nargs='*', default=[],
                        help="Lista de nombres de clases a excluir del dataset final.")
    
    args = parser.parse_args()

    # --- Cargar datasets ---
    print(f"Cargando dataset real (para val/test) desde: {args.real_input}")
    with open(args.real_input, 'r') as f:
        coco_real = json.load(f)

    print(f"Cargando dataset de entrenamiento desde: {args.train_input}")
    with open(args.train_input, 'r') as f:
        coco_train_source = json.load(f)

    # --- Determinar clases permitidas basado en el dataset real ---
    categories_real = coco_real["categories"]
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories_real}
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}

    # 1. Filtrado por mínimo de instancias en el dataset real
    instance_counts = {}
    for ann in coco_real["annotations"]:
        cat_id = ann["category_id"]
        instance_counts[cat_id] = instance_counts.get(cat_id, 0) + 1

    ids_to_remove_min_instances = {cat_id for cat_id, count in instance_counts.items() if count < args.min_instances}
    if ids_to_remove_min_instances:
        print("Se eliminarán las siguientes clases por no cumplir con el mínimo de instancias en el dataset real:")
        for cat_id in sorted(list(ids_to_remove_min_instances)):
            print(f" - {cat_id_to_name.get(cat_id, f'ID Desconocido {cat_id}')} (id: {cat_id}, instancias: {instance_counts.get(cat_id, 0)})")

    # 2. Filtrado por clases excluidas explícitamente
    ids_to_remove_excluded = {cat_name_to_id[name] for name in args.exclude_classes if name in cat_name_to_id}
    if ids_to_remove_excluded:
        print("Se excluirán las siguientes clases por petición del usuario:")
        for cat_id in sorted(list(ids_to_remove_excluded)):
             print(f" - {cat_id_to_name.get(cat_id, f'ID Desconocido {cat_id}')} (id: {cat_id})")

    # 3. Combinar y obtener IDs finales de clases permitidas
    all_ids_to_remove = ids_to_remove_min_instances.union(ids_to_remove_excluded)
    all_original_ids = {cat['id'] for cat in categories_real}
    allowed_class_ids = all_original_ids - all_ids_to_remove
    
    allowed_categories = [cat for cat in categories_real if cat["id"] in allowed_class_ids]
    
    print(f"\nTotal clases permitidas tras filtrado: {len(allowed_categories)}")

    # --- Filtrar ambos datasets para que solo contengan las clases permitidas ---
    coco_real_filtered = filter_coco_by_categories(coco_real, allowed_class_ids)
    coco_train = filter_coco_by_categories(coco_train_source, allowed_class_ids)
    
    # --- Dividir el dataset real en Validación y Test ---
    total_split_fraction = args.val_size + args.test_size
    if not (0 < total_split_fraction <= 1.0):
        raise ValueError(f"La suma de --val_size y --test_size debe estar entre 0 y 1, pero es {total_split_fraction}")

    test_fraction_of_real = args.test_size / total_split_fraction
    
    images_real = coco_real_filtered["images"]
    annotations_real = coco_real_filtered["annotations"]

    image_ids_real = [img["id"] for img in images_real]
    id_to_index_real = {img_id: idx for idx, img_id in enumerate(image_ids_real)}
    n_images_real = len(images_real)
    
    allowed_cat_ids_list = sorted(list(allowed_class_ids))
    allowed_cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(allowed_cat_ids_list)}
    n_categories = len(allowed_cat_ids_list)

    Y = np.zeros((n_images_real, n_categories), dtype=int)
    for ann in annotations_real:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id in id_to_index_real and cat_id in allowed_cat_id_to_index:
            i = id_to_index_real[img_id]
            j = allowed_cat_id_to_index[cat_id]
            Y[i, j] = 1

    # Añadir columna para imágenes sin anotaciones
    Y_dummy = np.concatenate([Y, np.zeros((n_images_real, 1), dtype=int)], axis=1)
    for i in range(n_images_real):
        if np.sum(Y[i]) == 0:
            Y_dummy[i, -1] = 1

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_fraction_of_real, random_state=args.seed)
    X_dummy = np.zeros((n_images_real, 1))
    
    if n_images_real > 0:
        val_idx, test_idx = next(msss.split(X_dummy, Y_dummy))
        val_ids = {image_ids_real[i] for i in val_idx}
        test_ids = {image_ids_real[i] for i in test_idx}
    else:
        val_ids, test_ids = set(), set()


    def build_coco_subset(image_id_set, base_coco):
        subset_images = [img for img in base_coco["images"] if img["id"] in image_id_set]
        subset_img_ids = {img['id'] for img in subset_images}
        subset_annotations = [ann for ann in base_coco["annotations"] if ann["image_id"] in subset_img_ids]
        return {
            "images": subset_images, 
            "annotations": subset_annotations, 
            "categories": base_coco["categories"], 
            "info": base_coco.get("info", {}), 
            "licenses": base_coco.get("licenses", [])
        }

    coco_val = build_coco_subset(val_ids, coco_real_filtered)
    coco_test = build_coco_subset(test_ids, coco_real_filtered)
    
    print("\nForzando la lista de categorías maestras en todos los splits para garantizar la consistencia.")
    master_categories = allowed_categories  # Esta es la lista de categorías filtrada del dataset real
    coco_train["categories"] = master_categories
    coco_val["categories"] = master_categories
    coco_test["categories"] = master_categories
    
    # --- Guardar archivos y generar gráficas ---
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        json.dump(coco_train, f, indent=4)
    with open(os.path.join(args.output_dir, "val.json"), "w") as f:
        json.dump(coco_val, f, indent=4)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        json.dump(coco_test, f, indent=4)

    print("\nDivisión completada:")
    print(f"  Train: {len(coco_train['images'])} imágenes (desde {os.path.basename(args.train_input)})")
    print(f"  Validación: {len(coco_val['images'])} imágenes (desde {os.path.basename(args.real_input)})")
    print(f"  Test: {len(coco_test['images'])} imágenes (desde {os.path.basename(args.real_input)})")

    # Mapeo final de ID a nombre para las gráficas
    final_cat_id_to_name = {cat["id"]: cat["name"] for cat in allowed_categories}

    print("\nGenerando gráficas de distribución de clases...")
    plot_class_distribution("Train", coco_train, final_cat_id_to_name, allowed_categories, args.output_dir)
    plot_class_distribution("Validación", coco_val, final_cat_id_to_name, allowed_categories, args.output_dir)
    plot_class_distribution("Test", coco_test, final_cat_id_to_name, allowed_categories, args.output_dir)
    print(f"Gráficas guardadas en: {args.output_dir}")

if __name__ == "__main__":
    main()