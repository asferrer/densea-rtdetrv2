import json

def find_bbox_issues(json_path, dataset_name):
    print(f"\n{'='*80}")
    print(f"BUSCANDO PROBLEMAS ESPECIFICOS EN: {dataset_name}")
    print(f"{'='*80}\n")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])
    
    # Find out of bounds
    print("1. BBOXES QUE EXCEDEN LIMITES DE IMAGEN:")
    out_of_bounds = []
    
    for ann in annotations:
        img_id = ann.get('image_id')
        bbox = ann.get('bbox', [])
        
        if img_id in images and len(bbox) == 4:
            x, y, w, h = bbox
            img_w = images[img_id]['width']
            img_h = images[img_id]['height']
            
            if x + w > img_w or y + h > img_h:
                out_of_bounds.append({
                    'ann_id': ann['id'],
                    'image_id': img_id,
                    'file': images[img_id]['file_name'],
                    'bbox': bbox,
                    'img_dims': (img_w, img_h),
                    'overflow': (x + w - img_w, y + h - img_h)
                })
    
    if out_of_bounds:
        print(f"   Total encontrados: {len(out_of_bounds)}")
        for i, issue in enumerate(out_of_bounds[:5]):
            print(f"\n   Ejemplo {i+1}:")
            print(f"     Anotación ID: {issue['ann_id']}")
            print(f"     Imagen ID: {issue['image_id']} ({issue['file']})")
            print(f"     Dimensiones imagen: {issue['img_dims']}")
            print(f"     Bbox: {issue['bbox']} [x, y, w, h]")
            print(f"     Overflow (W, H): {issue['overflow']}")
        if len(out_of_bounds) > 5:
            print(f"   ... y {len(out_of_bounds)-5} más")
    else:
        print(f"   Ninguno encontrado")
    
    # Check file_name format
    print(f"\n\n2. ANALISIS DE RUTAS (file_name):")
    prefixes = {}
    for img in data.get('images', []):
        fname = img.get('file_name', '')
        prefix = fname.split('/')[0] if '/' in fname else 'sin_prefijo'
        if prefix not in prefixes:
            prefixes[prefix] = 0
        prefixes[prefix] += 1
    
    print(f"   Prefijos encontrados:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True):
        print(f"     {prefix}: {count} imágenes")
    
    # Sample file names
    print(f"\n   Ejemplos de file_name:")
    for img in list(data.get('images', []))[:5]:
        print(f"     {img['file_name']}")
    
    # Check very small bboxes
    print(f"\n\n3. BBOXES CON AREA MUY PEQUEÑA (<100):")
    very_small = []
    
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            area = w * h
            if 0 < area < 100:
                very_small.append({
                    'ann_id': ann['id'],
                    'image_id': ann['image_id'],
                    'bbox': bbox,
                    'area': area
                })
    
    if very_small:
        print(f"   Total encontrados: {len(very_small)}")
        for issue in very_small[:5]:
            print(f"     Ann {issue['ann_id']}: bbox={issue['bbox']}, area={issue['area']:.0f}")
    else:
        print(f"   Ninguno encontrado")
    
    # Check zero area
    print(f"\n\n4. BBOXES CON AREA CERO:")
    zero_area = []
    
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            if w * h == 0:
                zero_area.append({'ann_id': ann['id'], 'bbox': bbox})
    
    if zero_area:
        print(f"   Total encontrados: {len(zero_area)}")
        for issue in zero_area[:5]:
            print(f"     Ann {issue['ann_id']}: bbox={issue['bbox']}")
    else:
        print(f"   Ninguno encontrado")
    
    # Check negative coordinates
    print(f"\n\n5. BBOXES CON VALORES NEGATIVOS:")
    negative = []
    
    for ann in annotations:
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w, h = bbox
            if x < 0 or y < 0 or w < 0 or h < 0:
                negative.append({'ann_id': ann['id'], 'bbox': bbox})
    
    if negative:
        print(f"   Total encontrados: {len(negative)}")
        for issue in negative[:5]:
            print(f"     Ann {issue['ann_id']}: bbox={issue['bbox']}")
    else:
        print(f"   Ninguno encontrado")

# Run analysis
find_bbox_issues(r'split_sintetico/train.json', 'split_sintetico/train.json')
find_bbox_issues(r'split_sintetico/val.json', 'split_sintetico/val.json')
find_bbox_issues(r'split_v7/train.json', 'split_v7/train.json')
