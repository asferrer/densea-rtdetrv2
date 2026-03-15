import json
from collections import defaultdict

def get_dataset_summary(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    return {
        'n_images': len(images),
        'n_annotations': len(annotations),
        'n_categories': len(categories),
        'categories': {c['id']: c['name'] for c in categories},
        'images': images,
        'annotations': annotations
    }

# Get all data
sintetico_train = get_dataset_summary(r'split_sintetico/train.json')
sintetico_val = get_dataset_summary(r'split_sintetico/val.json')
v7_train = get_dataset_summary(r'split_v7/train.json')
v8_train = get_dataset_summary(r'split_v8/train.json')

print("="*100)
print("COMPARACION GENERAL DE DATASETS")
print("="*100)
print(f"\n{'Metric':<30} | {'Sintetico Train':<20} | {'Sintetico Val':<20} | {'v7 Train':<20} | {'v8 Train':<20}")
print("-" * 100)

print(f"{'Imágenes':<30} | {sintetico_train['n_images']:<20} | {sintetico_val['n_images']:<20} | {v7_train['n_images']:<20} | {v8_train['n_images']:<20}")
print(f"{'Anotaciones':<30} | {sintetico_train['n_annotations']:<20} | {sintetico_val['n_annotations']:<20} | {v7_train['n_annotations']:<20} | {v8_train['n_annotations']:<20}")
print(f"{'Categorías':<30} | {sintetico_train['n_categories']:<20} | {sintetico_val['n_categories']:<20} | {v7_train['n_categories']:<20} | {v8_train['n_categories']:<20}")
print(f"{'Anotaciones por imagen':<30} | {sintetico_train['n_annotations']/sintetico_train['n_images']:<20.2f} | {sintetico_val['n_annotations']/sintetico_val['n_images']:<20.2f} | {v7_train['n_annotations']/v7_train['n_images']:<20.2f} | {v8_train['n_annotations']/v8_train['n_images']:<20.2f}")

print("\n" + "="*100)
print("ANALISIS DE CATEGORIAS")
print("="*100)

print(f"\nCategorías en split_sintetico:")
for cat_id in sorted(sintetico_train['categories'].keys()):
    print(f"  {cat_id}: {sintetico_train['categories'][cat_id]}")

print(f"\nCategorías en split_v7 y split_v8:")
for cat_id in sorted(v7_train['categories'].keys()):
    v8_name = v8_train['categories'].get(cat_id, 'MISSING')
    match = "OK" if v7_train['categories'].get(cat_id) == v8_name else "DIFERENTE"
    print(f"  {cat_id}: v7='{v7_train['categories'].get(cat_id)}' vs v8='{v8_name}' [{match}]")

print(f"\n*** PROBLEMA CRITICO: split_sintetico tiene SOLO 6 categorías (faltan 'Mask' e 'Tire')")
print(f"    Esto significa que las anotaciones de Mask (ID 4) y Tire (ID 7) del dataset real fueron ELIMINADAS")

print("\n" + "="*100)
print("ESTADISTICAS DE BBOXES")
print("="*100)

def get_bbox_stats(annotations):
    areas = []
    widths = []
    heights = []
    
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            areas.append(w * h)
            widths.append(w)
            heights.append(h)
    
    if areas:
        return {
            'min_area': min(areas),
            'max_area': max(areas),
            'avg_area': sum(areas) / len(areas),
            'min_width': min(widths),
            'max_width': max(widths),
            'min_height': min(heights),
            'max_height': max(heights),
        }
    return {}

sintetico_train_stats = get_bbox_stats(sintetico_train['annotations'])
sintetico_val_stats = get_bbox_stats(sintetico_val['annotations'])
v7_train_stats = get_bbox_stats(v7_train['annotations'])
v8_train_stats = get_bbox_stats(v8_train['annotations'])

print(f"\n{'Métrica':<25} | {'Sintetico Train':<25} | {'Sintetico Val':<25} | {'v7 Train':<25} | {'v8 Train':<25}")
print("-" * 110)
print(f"{'Min bbox area':<25} | {sintetico_train_stats['min_area']:<25.0f} | {sintetico_val_stats['min_area']:<25.0f} | {v7_train_stats['min_area']:<25.0f} | {v8_train_stats['min_area']:<25.0f}")
print(f"{'Max bbox area':<25} | {sintetico_train_stats['max_area']:<25.0f} | {sintetico_val_stats['max_area']:<25.0f} | {v7_train_stats['max_area']:<25.0f} | {v8_train_stats['max_area']:<25.0f}")
print(f"{'Avg bbox area':<25} | {sintetico_train_stats['avg_area']:<25.0f} | {sintetico_val_stats['avg_area']:<25.0f} | {v7_train_stats['avg_area']:<25.0f} | {v8_train_stats['avg_area']:<25.0f}")

print("\n" + "="*100)
print("RESUMEN DE PROBLEMAS ENCONTRADOS")
print("="*100)

print("""
split_sintetico/train.json:
  [CRITICO] Faltan 2 categorías respecto a v7/v8: 'Mask' (ID 4) y 'Tire' (ID 7)
  [IMPORTANTE] 30 bboxes exceden límites de imagen (overflow)
  [LEVE] Huecos en IDs de imágenes (715 gaps)

split_sintetico/val.json:
  [CRITICO] Faltan 2 categorías respecto a v7/v8: 'Mask' (ID 4) y 'Tire' (ID 7)
  [IMPORTANTE] 3 bboxes exceden límites de imagen
  [LEVE] Huecos en IDs de imágenes (397 gaps)

split_v7/train.json:
  [IMPORTANTE] 80 bboxes exceden límites de imagen
  [LEVE] Huecos en IDs de imágenes (3392 gaps)
  [LEVE] Desbalance severo de clases (ratio 56.93x)
  [LEVE] Mezcla de rutas: con "JPEGImages/" y sin prefijo
  
split_v8/train.json:
  [IMPORTANTE] 73 bboxes exceden límites de imagen
  [LEVE] Huecos en IDs de imágenes (3450 gaps)
  [LEVE] Desbalance severo de clases (ratio 46.43x)
  [LEVE] Mezcla de rutas: con "JPEGImages/" y sin prefijo
""")

