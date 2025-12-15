# tools/create_calibration_set.py

import json
import os
import shutil
import argparse
from collections import defaultdict
from tqdm import tqdm
import unicodedata

def normalize_caseless(text):
    """Normalizes a string for caseless comparison."""
    return unicodedata.normalize("NFKD", text.casefold())

def get_origin_from_path(relative_path, real_path, data_root):
    """
    Determines the dataset origin. Prefers using the real path structure if possible.
    """
    try:
        rel_to_root = os.path.relpath(real_path, data_root)
        parts = os.path.normpath(rel_to_root).split(os.sep)
        if parts:
            return parts[0]
    except (ValueError, IndexError):
        parts = os.path.normpath(relative_path).split(os.sep)
        if parts:
            return parts[0]
    return 'unknown'

def create_image_index(data_root):
    """
    Walks through the data_root and creates a comprehensive, case-insensitive index
    of every image file, mapping each basename to a list of its full paths.
    """
    print(f"Indexing all images under '{data_root}'. This may take a moment...")
    image_index = defaultdict(list)
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    file_list = [os.path.join(root, filename) for root, _, files in os.walk(data_root) for filename in files]

    for full_path in tqdm(file_list, desc="Scanning files"):
        filename = os.path.basename(full_path)
        if filename.lower().endswith(supported_extensions):
            caseless_basename = normalize_caseless(filename)
            image_index[caseless_basename].append(full_path)
            
    if not image_index:
        print(f"❌ Error: No images found in '{data_root}'. Please check the path.")
    else:
        print(f"✅ Found and indexed {sum(len(v) for v in image_index.values())} image files ({len(image_index)} unique basenames).")
    return image_index

def find_best_path(json_path, candidates):
    """
    Given a list of candidate paths, finds the best match based on the
    path provided in the JSON file using a suffix matching approach.
    """
    if len(candidates) == 1:
        return candidates[0]
    
    best_match = candidates[0]
    max_match_len = 0
    
    # Normalize the json path once, using forward slashes
    json_path_norm = os.path.normpath(json_path).replace('\\', '/')
    json_parts = json_path_norm.split('/')

    for real_path in candidates:
        real_path_norm = os.path.normpath(real_path).replace('\\', '/')
        real_parts = real_path_norm.split('/')
        
        # Check how many parts match from the end of the lists
        match_len = 0
        for j_part, r_part in zip(reversed(json_parts), reversed(real_parts)):
            if j_part == r_part:
                match_len += 1
            else:
                break
        
        if match_len > max_match_len:
            max_match_len = match_len
            best_match = real_path

    return best_match

def select_calibration_images(json_path, data_root, num_images, output_dir):
    """
    Selects a balanced subset for INT8 calibration. It intelligently finds images
    by basename, and robustly handles path mismatches and duplicates.
    """
    print(f"Loading validation data from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file '{json_path}' was not found.")
        return

    # --- 1. Index all available images ---
    image_index = create_image_index(data_root)
    if not image_index: return

    # --- 2. Link JSON data to real file paths ---
    print("Linking JSON entries to physical image files...")
    images_info = {img['id']: img for img in coco_data['images']}
    cat_info = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    image_to_cats = defaultdict(set)
    image_to_origin = {}
    resolved_paths = {}
    unfound_json_files = []

    for img_id, img_data in tqdm(images_info.items(), desc="Resolving paths"):
        json_filename = os.path.basename(img_data['file_name'])
        caseless_basename = normalize_caseless(json_filename)
        
        candidate_paths = image_index.get(caseless_basename)
        
        if candidate_paths:
            best_path = find_best_path(img_data['file_name'], candidate_paths)
            resolved_paths[img_id] = best_path
        else:
            unfound_json_files.append(json_filename)
    
    if unfound_json_files:
        print(f"\n⚠️ Warning: Could not find {len(unfound_json_files)} images listed in the JSON file anywhere inside '{data_root}'.")
        if len(unfound_json_files) < 20:
            print("Missing files sample:", unfound_json_files[:20])

    for ann in coco_data['annotations']:
        image_to_cats[ann['image_id']].add(ann['category_id'])
        
    for img_id, img_data in images_info.items():
        if img_id in resolved_paths:
            real_path = resolved_paths[img_id]
            image_to_origin[img_id] = get_origin_from_path(img_data['file_name'], real_path, data_root)

    # --- 3. Iterative Selection Logic ---
    print(f"\nSelecting {num_images} balanced images from the {len(resolved_paths)} found images...")
    
    selected_cat_counts = defaultdict(int)
    selected_origin_counts = defaultdict(int)
    
    available_image_ids = set(resolved_paths.keys())
    selected_image_ids = []

    pbar = tqdm(total=min(num_images, len(available_image_ids)), desc="Selecting Images")

    for _ in range(num_images):
        if not available_image_ids:
            print("\nWarning: Ran out of available images.")
            break

        best_image_id, max_score = -1, -1
        for img_id in available_image_ids:
            score = 0
            if img_id in image_to_cats:
                for cat_id in image_to_cats[img_id]:
                    score += 1.0 / (selected_cat_counts[cat_id] + 1)
            
            origin = image_to_origin.get(img_id, 'unknown')
            score += 1.0 / (selected_origin_counts[origin] + 1)
            
            if score > max_score:
                max_score, best_image_id = score, img_id

        if best_image_id != -1:
            selected_image_ids.append(best_image_id)
            available_image_ids.remove(best_image_id)
            for cat_id in image_to_cats[best_image_id]:
                selected_cat_counts[cat_id] += 1
            selected_origin_counts[image_to_origin.get(best_image_id, 'unknown')] += 1
            pbar.update(1)
    pbar.close()

    # --- 4. Copy Files ---
    print(f"\nCopying {len(selected_image_ids)} selected images to '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)
    
    for img_id in tqdm(selected_image_ids, desc="Copying Files"):
        src_path = resolved_paths[img_id]
        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)
            
    # --- 5. Final Summary ---
    print("\n--- Calibration Set Summary ---")
    print(f"Total images selected: {len(selected_image_ids)}")
    print("\nCategory Distribution:")
    for cat_id, count in sorted(selected_cat_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {cat_info.get(cat_id, 'Unknown')}: {count} images")
    print("\nDataset Origin Distribution:")
    for origin, count in sorted(selected_origin_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {origin}: {count} images")
    print(f"\n✅ Calibration set created successfully in '{output_dir}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a balanced calibration set for INT8 quantization. Intelligently finds images within the data root.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--json_path', type=str, required=True, help="Path to the COCO-style validation JSON file.")
    parser.add_argument('--data_root', type=str, required=True, help="Path to the root directory to search for all images.")
    parser.add_argument('--output_dir', type=str, default='calibration_set', help="Directory to save the selected calibration images.")
    parser.add_argument('--num_images', type=int, default=250, help="Number of images to select for the set.")
    args = parser.parse_args()
    select_calibration_images(args.json_path, args.data_root, args.num_images, args.output_dir)