import json
from collections import defaultdict
import sys

def analyze_dataset(json_path, dataset_name):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {dataset_name}")
    print(f"{'='*80}\n")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Basic counts
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    n_images = len(images)
    n_annotations = len(annotations)
    n_categories = len(categories)
    
    print(f"CONTEOS BASICOS:")
    print(f"  Imágenes: {n_images}")
    print(f"  Anotaciones: {n_annotations}")
    print(f"  Categorías: {n_categories}\n")
    
    # Category analysis
    print(f"CATEGORIAS:")
    cat_ids = set()
    for cat in categories:
        cat_ids.add(cat['id'])
        print(f"  ID {cat['id']:2d}: {cat['name']}")
    
    print(f"\n  Rango: {min(cat_ids) if cat_ids else 'N/A'} - {max(cat_ids) if cat_ids else 'N/A'}")
    
    # Check for missing IDs
    if cat_ids:
        expected = set(range(min(cat_ids), max(cat_ids)+1))
        missing = expected - cat_ids
        if missing:
            print(f"  ALERTA: IDs faltantes: {missing}")
    
    # Image analysis
    print(f"\nANALISIS DE IMAGENES:")
    image_ids = set()
    image_map = {}
    
    for img in images:
        image_ids.add(img['id'])
        image_map[img['id']] = img
    
    # Check for gaps
    if image_ids:
        sorted_ids = sorted(image_ids)
        gaps = []
        for i in range(len(sorted_ids)-1):
            if sorted_ids[i+1] - sorted_ids[i] > 1:
                gaps.append((sorted_ids[i], sorted_ids[i+1]))
        if gaps:
            print(f"  HUECOS EN IDS: {len(gaps)} detected")
            for start, end in gaps[:3]:
                missing = end - start - 1
                print(f"    Entre {start} y {end}: faltan {missing} IDs")
    
    # Annotation analysis
    print(f"\nANALISIS DE ANOTACIONES:")
    
    bbox_problems = {
        'out_of_bounds': 0,
        'negative': 0,
        'zero_area': 0,
        'very_small': 0,
        'invalid_ref': 0
    }
    
    cat_counts = defaultdict(int)
    annotations_per_image = defaultdict(int)
    bbox_areas = []
    
    for ann in annotations:
        img_id = ann.get('image_id')
        cat_id = ann.get('category_id')
        bbox = ann.get('bbox', [])
        
        cat_counts[cat_id] += 1
        annotations_per_image[img_id] += 1
        
        if cat_id not in cat_ids:
            bbox_problems['invalid_ref'] += 1
        
        if len(bbox) == 4:
            x, y, w, h = bbox
            
            if x < 0 or y < 0 or w < 0 or h < 0:
                bbox_problems['negative'] += 1
            
            area = w * h
            if area == 0:
                bbox_problems['zero_area'] += 1
            elif area < 100:
                bbox_problems['very_small'] += 1
            
            bbox_areas.append(area)
            
            if img_id in image_map:
                img_w = image_map[img_id].get('width', 0)
                img_h = image_map[img_id].get('height', 0)
                
                if x + w > img_w or y + h > img_h:
                    bbox_problems['out_of_bounds'] += 1
    
    print(f"  Problemas detectados:")
    for ptype, count in bbox_problems.items():
        if count > 0:
            print(f"    {ptype}: {count}")
    
    if all(v == 0 for v in bbox_problems.values()):
        print(f"    Ninguno detectado")
    
    # Statistics
    if bbox_areas:
        print(f"\n  Estadísticas de áreas:")
        print(f"    Min: {min(bbox_areas):.1f}")
        print(f"    Max: {max(bbox_areas):.1f}")
        print(f"    Promedio: {sum(bbox_areas)/len(bbox_areas):.1f}")
    
    # Class distribution
    print(f"\nDISTRIBUCION DE CLASES:")
    print(f"  ID | Nombre               | Count | % ")
    print(f"  {'-'*50}")
    
    for cat_id in sorted(cat_counts.keys()):
        count = cat_counts[cat_id]
        pct = 100 * count / n_annotations if n_annotations > 0 else 0
        cat_name = next((c['name'] for c in categories if c['id'] == cat_id), 'UNKNOWN')
        print(f"  {cat_id:2d} | {cat_name:20s} | {count:5d} | {pct:6.2f}%")
    
    # Imbalance check
    if cat_counts:
        min_count = min(cat_counts.values())
        max_count = max(cat_counts.values())
        if min_count > 0:
            ratio = max_count / min_count
            print(f"\n  Ratio desbalance (max/min): {ratio:.2f}x")
            if ratio > 10:
                print(f"  ALERTA: Desbalance SEVERO")
    
    # Images without annotations
    images_without = len([i for i in image_ids if annotations_per_image[i] == 0])
    print(f"\nIMAGENES SIN ANOTACIONES: {images_without}")
    
    print()

# Main
analyze_dataset(r'split_sintetico/train.json', 'split_sintetico/train.json')
analyze_dataset(r'split_sintetico/val.json', 'split_sintetico/val.json')
analyze_dataset(r'split_v7/train.json', 'split_v7/train.json (REFERENCE)')
analyze_dataset(r'split_v8/train.json', 'split_v8/train.json (REFERENCE)')
