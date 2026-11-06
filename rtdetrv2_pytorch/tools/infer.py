#!/usr/bin/env python
"""
Inference script for RT-DETR-V2 that automatically detects the input mode
(image, video or camera/stream) from the given input.
Output is not shown live; the inference result (with detections drawn) is saved
automatically under a name combining the input name and the config file name.
If nothing is detected in a frame, the original frame is saved.
"""

import os
import sys
import argparse
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import colorsys
from tqdm import tqdm

# Add the parent directory so YAMLConfig and other internal modules can be imported.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

########################
# Visualization helpers
########################

def generate_color_map_from_label_map(label_map: dict) -> dict:
    keys = sorted(label_map.keys())
    n = len(keys)
    color_map = {}
    for i, key in enumerate(keys):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        color_hex = '#{0:02x}{1:02x}{2:02x}'.format(int(r*255), int(g*255), int(b*255))
        color_map[key] = color_hex
    return color_map

def get_label_map(dataset) -> dict:
    label_map = {}
    try:
        categories = dataset.coco.dataset.get('categories', [])
        for cat in categories:
            # Convert the key to int for consistency
            label_map[int(cat['id'])] = cat['name']
    except Exception as e:
        print(f"Could not get the label mapping: {e}")
    return label_map

def compute_label_offset(pred_labels_tensor: torch.Tensor, label_map: dict) -> int:
    """
    Dynamically determine the offset to apply to the predicted labels so they are consistent
    with the label mapping. If the model already produces labels starting at 0, returns 0.
    """
    if pred_labels_tensor.numel() > 0:
        raw_pred_min = int(pred_labels_tensor.min().item())
        min_label = min(label_map.keys()) if label_map else 0
        offset = min_label - raw_pred_min
        return offset
    return 0

def draw_inferences_pil(image: Image.Image, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor,
                         label_map=None, color_map=None, threshold: float = 0.5, 
                         label_offset: int = 0, radius: int = 2):
    """
    Draw the inferences on a PIL image in a minimalist way.
    If nothing is detected (all scores <= threshold), the original image is returned.
    The 'label_offset' parameter adjusts the labels when needed.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    valid_indices = scores > threshold
    # If there are no valid detections, return the original image
    if valid_indices.sum() == 0:
        return image

    valid_labels = labels[valid_indices]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    padding = 2
    for j, box in enumerate(valid_boxes):
        x0, y0, x1, y1 = map(int, box)
        # Apply the offset; the model produces 0-indexed labels, so 1 is not added
        pred_id = valid_labels[j].item() + label_offset
        color = color_map.get(pred_id, "black") if color_map is not None else "black"
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=color, width=1)
        if label_map is not None:
            label_text = label_map.get(pred_id, str(pred_id))
        else:
            label_text = str(pred_id)
        text = f"{label_text} {valid_scores[j].item():.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if y0 - text_height - padding >= 0:
            text_position = (x0 + padding, y0 - text_height - padding)
        else:
            text_position = (x0 + padding, y1 + padding)
        draw.text(text_position, text, font=font, fill=color, stroke_width=1, stroke_fill="black")
    return image

########################
# PIL <-> OpenCV conversion helpers
########################

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

########################
# ModelWrapper class
########################

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super(ModelWrapper, self).__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs  # Expected (labels, boxes, scores)

########################
# Inference on a single frame
########################

def inference_on_frame(frame: np.ndarray, model, transform, device, label_map, color_map, threshold=0.5):
    """
    Run inference on a frame. The frame is converted to PIL, the inference transform is applied,
    the offset is computed (0 here, since the model produces labels starting at 0)
    and the detections are drawn on the PIL image.
    The annotated frame is returned at the original size.
    """
    pil_img = cv2_to_pil(frame)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    orig_size = torch.tensor([pil_img.width, pil_img.height])[None].to(device)
    with torch.no_grad():
        labels, boxes, scores = model(input_tensor, orig_size)
    # Compute offset (should be 0 if the model already produces 0-indexed labels)
    label_offset = compute_label_offset(labels[0].cpu(), label_map)
    annotated_pil = draw_inferences_pil(pil_img, labels[0], boxes[0], scores[0],
                                         label_map=label_map, color_map=color_map, threshold=threshold,
                                         label_offset=label_offset)
    annotated_pil = annotated_pil.resize((pil_img.width, pil_img.height))
    annotated_frame = pil_to_cv2(annotated_pil)
    return annotated_frame

########################
# Automatic mode detection
########################

def auto_detect_mode(input_source: str) -> str:
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_exts = ['.mp4', '.avi', '.mkv', '.mov']
    try:
        ext = os.path.splitext(input_source)[1].lower()
        if ext in image_exts:
            return "image"
        elif ext in video_exts:
            return "video"
        else:
            return "image"
    except:
        try:
            int(input_source)
            return "camera"
        except ValueError:
            if input_source.startswith("rtsp://") or input_source.startswith("http://") or input_source.startswith("https://"):
                return "camera"
            else:
                return "camera"

########################
# Main function
########################

def main():
    parser = argparse.ArgumentParser(
        description="Inference script for images, video or camera (RTSP/webcam) that saves the result automatically."
    )
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument('-r', '--resume', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('-d', '--device', type=str, default='cuda', help="Device: cpu or cuda.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the image/video file, or index/URL for a camera (RTSP/webcam).")
    parser.add_argument('--save_dir', type=str, default='inference_results',
                        help="Base directory to save the results.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold to filter detections.")
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # Detect the mode automatically
    mode = auto_detect_mode(args.input)
    print(f"Detected mode: {mode}")
    
    # Load configuration and model
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location=device)
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    model_instance = cfg.model
    model_instance.load_state_dict(state)
    model = ModelWrapper(cfg).to(device)
    model.eval()
    
    # Get label map and color map (making sure keys are integers)
    label_map = get_label_map(cfg.val_dataloader.dataset)
    color_map = generate_color_map_from_label_map(label_map)
    
    # Define the input transform to match the model input
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    # Create the directory to save the results
    os.makedirs(args.save_dir, exist_ok=True)
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    
    if mode == "image":
        pil_img = Image.open(args.input).convert('RGB')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        orig_size = torch.tensor([pil_img.width, pil_img.height])[None].to(device)
        with torch.no_grad():
            labels, boxes, scores = model(input_tensor, orig_size)
        # Compute offset (should be 0)
        label_offset = compute_label_offset(labels[0].cpu(), label_map)
        annotated_img = draw_inferences_pil(pil_img, labels[0], boxes[0], scores[0],
                                             label_map=label_map, color_map=color_map, threshold=args.threshold,
                                             label_offset=label_offset)
        out_filename = f"{input_basename}_{config_name}.jpg"
        out_path = os.path.join(args.save_dir, out_filename)
        annotated_img.save(out_path)
        print(f"Inferred image saved to: {out_path}")
    elif mode in ["video", "camera"]:
        if mode == "video":
            cap = cv2.VideoCapture(args.input)
        else:
            try:
                cam_index = int(args.input)
                cap = cv2.VideoCapture(cam_index)
            except ValueError:
                cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print("Error: could not open the video/camera source.")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25
        out_filename = f"{input_basename}_{config_name}.mp4" if mode == "video" else f"camera_{config_name}.mp4"
        out_path = os.path.join(args.save_dir, out_filename)
        out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if mode == "video" else None
        pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame = inference_on_frame(frame, model, transform, device, label_map, color_map, threshold=args.threshold)
            annotated_frame = cv2.resize(annotated_frame, (width, height))
            out_writer.write(annotated_frame)
            pbar.update(1)
        pbar.close()
        cap.release()
        out_writer.release()
        print(f"Inferred video saved to: {out_path}")

if __name__ == '__main__':
    main()
