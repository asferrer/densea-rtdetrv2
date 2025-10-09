#!/usr/bin/env python3
import json
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import copy

def plot_class_distribution(split_name, coco_subset, cat_id_to_name, categories, output_dir, synthetic_img_ids=None, balanced=False):
    """
    Calcula y guarda un gráfico de barras con el número total de instancias
    por clase para el split dado. Si el split está balanceado, muestra
    barras apiladas para diferenciar entre datos reales y sintéticos.
    """
    if synthetic_img_ids is None:
        synthetic_img_ids = set()

    # Inicializar conteos
    all_cat_names = sorted([cat['name'] for cat in categories])
    real_counts = {name: 0 for name in all_cat_names}
    synthetic_counts = {name: 0 for name in all_cat_names}

    # Contar instancias reales y sintéticas
    for ann in coco_subset["annotations"]:
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id)
        if cat_name:
            if ann['image_id'] in synthetic_img_ids:
                synthetic_counts[cat_name] += 1
            else:
                real_counts[cat_name] += 1

    names = list(real_counts.keys())
    real_values = np.array(list(real_counts.values()))
    synthetic_values = np.array(list(synthetic_counts.values()))

    # Configuración del gráfico
    title = f"Distribución de instancias por clase en {split_name}"
    if balanced:
        title += " (Balanceado)"

    fig, ax = plt.subplots(figsize=(14, 8))

    # Crear las barras
    if balanced:
        # Barras apiladas si hay datos sintéticos
        bar_real = ax.bar(names, real_values, label='Real', color='dodgerblue')
        bar_synthetic = ax.bar(names, synthetic_values, bottom=real_values, label='Sintético', color='darkorange')
        ax.legend()
    else:
        # Barras normales si no está balanceado
        bar_real = ax.bar(names, real_values, label='Real', color='dodgerblue')

    ax.set_title(title)
    ax.set_xlabel("Clases")
    ax.set_ylabel("Número de instancias")

    # Añadir el número total sobre cada barra
    total_values = real_values + synthetic_values
    for i, total in enumerate(total_values):
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


def merge_and_reid_datasets(base_coco, synthetic_coco, max_img_id, max_ann_id):
    """
    Fusiona un dataset sintético en uno base, reetiquetando IDs para evitar colisiones.
    Devuelve el dataset fusionado, los nuevos IDs máximos y los IDs de las imágenes sintéticas añadidas.
    """
    merged_coco = copy.deepcopy(base_coco)
    synthetic_to_add = copy.deepcopy(synthetic_coco)
    
    # 1. Fusionar categorías
    base_cat_ids = {cat['id'] for cat in merged_coco['categories']}
    for cat in synthetic_to_add.get('categories', []):
        if cat['id'] not in base_cat_ids:
            merged_coco['categories'].append(cat)
            base_cat_ids.add(cat['id'])

    # 2. Reetiquetar y fusionar imágenes y anotaciones
    img_id_map = {}
    new_synthetic_img_ids = set()
    next_img_id = max_img_id + 1
    next_ann_id = max_ann_id + 1

    for img in synthetic_to_add.get('images', []):
        old_img_id = img['id']
        new_img_id = next_img_id
        img_id_map[old_img_id] = new_img_id
        new_synthetic_img_ids.add(new_img_id)
        img['id'] = new_img_id
        merged_coco['images'].append(img)
        next_img_id += 1

    for ann in synthetic_to_add.get('annotations', []):
        old_img_id = ann['image_id']
        if old_img_id in img_id_map:
            ann['id'] = next_ann_id
            ann['image_id'] = img_id_map[old_img_id]
            merged_coco['annotations'].append(ann)
            next_ann_id += 1

    return merged_coco, next_img_id - 1, next_ann_id - 1, new_synthetic_img_ids


def main():
    parser = argparse.ArgumentParser(
        description="Script para dividir un dataset COCO, filtrar clases y opcionalmente balancear particiones con datos sintéticos."
    )
    # Argumentos originales
    parser.add_argument("--input", type=str, required=True,
                        help="Ruta al archivo COCO JSON principal.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directorio de salida para los JSON y las gráficas.")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Fracción de imágenes para el conjunto de test (default: 0.15).")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fracción de imágenes para el conjunto de validación (default: 0.15).")
    parser.add_argument("--seed", type=int, default=21,
                        help="Semilla para aleatorización (default: 21).")
    parser.add_argument("--min_instances", type=int, default=5,
                        help="Número mínimo de instancias por clase para ser considerada (default: 5).")
    
    # Nuevos argumentos para el balanceo con datos sintéticos
    parser.add_argument("--synthetic_input", type=str,
                        help="Ruta opcional al archivo COCO JSON con datos sintéticos.")
    parser.add_argument("--balance_splits", type=str, nargs='+', choices=['train', 'val', 'test'], default=[],
                        help="Especifica qué particiones balancear con el dataset sintético (e.g., train val).")
    
    args = parser.parse_args()

    # --- Cargar datasets ---
    print(f"Cargando dataset principal desde: {args.input}")
    with open(args.input, 'r') as f:
        coco = json.load(f)

    coco_synthetic = None
    if args.synthetic_input:
        if not args.balance_splits:
            print("Aviso: Se ha proporcionado un dataset sintético pero no se ha especificado qué particiones balancear con --balance_splits. No se realizará ningún balanceo.")
        else:
            print(f"Cargando dataset sintético desde: {args.synthetic_input}")
            with open(args.synthetic_input, 'r') as f:
                coco_synthetic = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # --- Filtrado por mínimo de instancias (sobre el dataset principal) ---
    instance_counts = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        instance_counts[cat_id] = instance_counts.get(cat_id, 0) + 1

    allowed_class_ids = {cat_id for cat_id, count in instance_counts.items() if count >= args.min_instances}
    
    removed_classes = [cat for cat in categories if cat["id"] not in allowed_class_ids]
    if removed_classes:
        print("Se eliminarán las siguientes clases del dataset principal por no cumplir con el mínimo de instancias:")
        for cat in removed_classes:
            print(f" - {cat['name']} (id: {cat['id']}, instancias: {instance_counts.get(cat['id'], 0)})")
    
    allowed_categories = [cat for cat in categories if cat["id"] in allowed_class_ids]
    annotations_filtered = [ann for ann in annotations if ann["category_id"] in allowed_class_ids]
    images_filtered = images

    print(f"Total imágenes originales: {len(images_filtered)}")
    print(f"Total clases permitidas: {len(allowed_categories)}")

    # --- Preparar la matriz multi-etiqueta para el split ---
    image_ids = [img["id"] for img in images_filtered]
    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    n_images = len(images_filtered)
    allowed_cat_ids = [cat["id"] for cat in allowed_categories]
    allowed_cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(allowed_cat_ids)}
    n_categories = len(allowed_categories)

    Y = np.zeros((n_images, n_categories), dtype=int)
    for ann in annotations_filtered:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if img_id in id_to_index and cat_id in allowed_cat_id_to_index:
            i = id_to_index[img_id]
            j = allowed_cat_id_to_index[cat_id]
            Y[i, j] = 1

    Y_dummy = np.concatenate([Y, np.zeros((n_images, 1), dtype=int)], axis=1)
    for i in range(n_images):
        if np.sum(Y[i]) == 0:
            Y_dummy[i, -1] = 1

    # --- Realizar el split multi-etiqueta ---
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    X_dummy = np.zeros((n_images, 1))
    train_val_idx, test_idx = next(msss.split(X_dummy, Y_dummy))
    val_fraction = args.val_size / (1 - args.test_size)
    X_dummy_tv = np.zeros((len(train_val_idx), 1))
    Y_tv = Y_dummy[train_val_idx]
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=args.seed)
    train_idx_rel, val_idx_rel = next(msss_val.split(X_dummy_tv, Y_tv))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    train_ids = {image_ids[i] for i in train_idx}
    val_ids = {image_ids[i] for i in val_idx}
    test_ids = {image_ids[i] for i in test_idx}

    def build_coco_subset(image_id_set, base_coco):
        subset_images = [img for img in base_coco["images"] if img["id"] in image_id_set]
        subset_img_ids = {img['id'] for img in subset_images}
        subset_annotations = [ann for ann in base_coco["annotations"] if ann["image_id"] in subset_img_ids]
        return {
            "images": subset_images,
            "annotations": subset_annotations,
            "categories": base_coco["categories"],
            "info": base_coco.get("info", {}),
            "licenses": base_coco.get("licenses", []),
        }

    coco_filtered = {
        "images": images_filtered,
        "annotations": annotations_filtered,
        "categories": allowed_categories,
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
    }

    coco_train = build_coco_subset(train_ids, coco_filtered)
    coco_val = build_coco_subset(val_ids, coco_filtered)
    coco_test = build_coco_subset(test_ids, coco_filtered)

    # --- Balanceo con datos sintéticos (si se ha solicitado) ---
    splits_data = {'train': coco_train, 'val': coco_val, 'test': coco_test}
    balanced_flags = {'train': False, 'val': False, 'test': False}
    all_synthetic_img_ids = set()

    if coco_synthetic and args.balance_splits:
        print("\n--- Iniciando proceso de balanceo ---")
        max_img_id = max(img['id'] for img in coco['images']) if coco.get('images') else 0
        max_ann_id = max(ann['id'] for ann in coco['annotations']) if coco.get('annotations') else 0

        for split_name in args.balance_splits:
            if split_name in splits_data:
                print(f"Balanceando la partición '{split_name}'...")
                original_split_data = splits_data[split_name]
                
                balanced_split, new_max_img_id, new_max_ann_id, new_synth_ids = merge_and_reid_datasets(
                    original_split_data, coco_synthetic, max_img_id, max_ann_id
                )
                
                splits_data[split_name] = balanced_split
                all_synthetic_img_ids.update(new_synth_ids)
                max_img_id = new_max_img_id
                max_ann_id = new_max_ann_id
                balanced_flags[split_name] = True
                print(f"Partición '{split_name}' balanceada.")
    
    coco_train, coco_val, coco_test = splits_data['train'], splits_data['val'], splits_data['test']

    # --- Guardar archivos y generar gráficas ---
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        json.dump(coco_train, f, indent=4)
    with open(os.path.join(args.output_dir, "val.json"), "w") as f:
        json.dump(coco_val, f, indent=4)
    with open(os.path.join(args.output_dir, "test.json"), "w") as f:
        json.dump(coco_test, f, indent=4)

    print("\nDivisión completada:")
    print(f"  Train: {len(coco_train['images'])} imágenes")
    print(f"  Validación: {len(coco_val['images'])} imágenes")
    print(f"  Test: {len(coco_test['images'])} imágenes")

    # Unir todas las categorías para el mapeo
    all_categories = coco_train['categories'] + coco_val['categories'] + coco_test['categories']
    final_categories_map = {cat['id']: cat for cat in all_categories}
    final_categories = list(final_categories_map.values())
    cat_id_to_name = {cat["id"]: cat["name"] for cat in final_categories}

    print("\nGenerando gráficas de distribución de clases...")
    plot_class_distribution("Train", coco_train, cat_id_to_name, final_categories, args.output_dir, 
                            synthetic_img_ids=all_synthetic_img_ids, balanced=balanced_flags['train'])
    plot_class_distribution("Validación", coco_val, cat_id_to_name, final_categories, args.output_dir, 
                            synthetic_img_ids=all_synthetic_img_ids, balanced=balanced_flags['val'])
    plot_class_distribution("Test", coco_test, cat_id_to_name, final_categories, args.output_dir, 
                            synthetic_img_ids=all_synthetic_img_ids, balanced=balanced_flags['test'])
    print(f"Gráficas guardadas en: {args.output_dir}")

if __name__ == "__main__":
    main()