#!/usr/bin/env python
"""
Script de evaluación unificado para RT-DETR-V2, inspirado en las mejores
prácticas de los scripts de inferencia para ONNX y TensorRT.

Funcionalidades:
- Soporte multi-formato: PyTorch (.pth), ONNX (.onnx), y TensorRT (.engine).
- Motores de inferencia optimizados, especialmente para TensorRT con gestión
  de memoria y ejecución asíncrona vía PyCUDA.
- Benchmarking de rendimiento (latencia y FPS).
- Métricas completas: mAP (usando pycocotools), matriz de confusión,
  reporte de clasificación y análisis de falsos positivos/negativos.
- Generación de imágenes con detecciones para revisión visual.
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import colorsys
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.optimize import linear_sum_assignment

# --- Importaciones para Inferencia y Evaluación ---

# Bloque para la importación y evaluación COCO
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_TOOLS_AVAILABLE = True
except ImportError:
    COCO_TOOLS_AVAILABLE = False

# Agregar directorio padre para importar módulos internos
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

# --- Clases de Motores de Inferencia (Wrappers) ---

class PyTorchEngine(nn.Module):
    """Wrapper para el modelo PyTorch original."""
    def __init__(self, cfg, device):
        super().__init__()
        checkpoint = torch.load(cfg.resume, map_location=device)
        state = checkpoint.get('ema', {}).get('module', checkpoint.get('model'))
        
        self.model = cfg.model
        self.model.load_state_dict(state)
        self.model = self.model.to(device).eval()
        
        self.postprocessor = cfg.postprocessor.deploy()
        self.postprocessor = self.postprocessor.to(device).eval()
        print("Motor PyTorch cargado exitosamente.")

    def forward(self, images, orig_target_sizes):
        with torch.no_grad():
            outputs = self.model(images)
            labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
        return labels, boxes, scores

class ONNXEngine:
    """Wrapper para la inferencia con ONNX Runtime."""
    def __init__(self, model_path, device_str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime no está instalado. Instálalo con: pip install onnxruntime-gpu")

        print(f"Cargando modelo ONNX desde: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device_str else ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.device = torch.device(device_str)
        print(f"Modelo ONNX cargado. Usando proveedor: {self.session.get_providers()[0]}")

    def __call__(self, images, orig_target_sizes):
        inputs = {
            self.input_names[0]: images.cpu().numpy(),
            self.input_names[1]: orig_target_sizes.cpu().numpy()
        }
        outputs = self.session.run(self.output_names, inputs)
        
        labels = torch.from_numpy(outputs[0]).to(self.device)
        boxes = torch.from_numpy(outputs[1]).to(self.device)
        scores = torch.from_numpy(outputs[2]).to(self.device)
        return labels, boxes, scores

class TensorRTEngine:
    """
    Wrapper para la inferencia con TensorRT.
    Gestiona la carga del motor, la asignación de buffers y la ejecución asíncrona.
    """
    def __init__(self, engine_path, device_str):
        try:
            import tensorrt as trt
            # Se importan aquí para que el script no falle si solo se usa ONNX o PyTorch
            import pycuda.driver as cuda
            import pycuda.autoinit  # crucial para la gestión automática del contexto
            self.cuda = cuda
        except ImportError as e:
            raise ImportError(f"Dependencia faltante para TensorRT: {e}. Revisa la instalación.")

        print(f"Cargando motor de TensorRT desde: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("No se pudo deserializar el motor de TensorRT. Verifica el archivo y los permisos.")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.device = torch.device(device_str)
        print("Motor de TensorRT cargado exitosamente.")

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = self.cuda.Stream()
        
        for binding_name in self.engine:
            binding_idx = self.engine.get_binding_index(binding_name)
            shape = tuple(self.context.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            size = trt.volume(shape)
            
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            buffer_info = {'host': host_mem, 'device': device_mem, 'shape': shape, 'name': binding_name}
            if self.engine.binding_is_input(binding_name):
                inputs.append(buffer_info)
            else:
                outputs.append(buffer_info)
                
        return inputs, outputs, bindings, stream

    def __call__(self, images, orig_target_sizes):
        np.copyto(self.inputs[0]['host'], images.cpu().numpy().ravel())
        np.copyto(self.inputs[1]['host'], orig_target_sizes.cpu().numpy().ravel())

        self.cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        for out in self.outputs:
            self.cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        
        self.stream.synchronize()
        
        batch_size = images.shape[0]
        # La forma de las salidas puede variar, asegúrate de que coincida con tu modelo
        # Asumiendo una salida estándar de [batch, 300] para etiquetas/scores y [batch, 300, 4] para cajas
        num_dets = self.outputs[0]['shape'][-1] if len(self.outputs[0]['shape']) > 1 else self.outputs[0]['shape'][0]
        
        labels = torch.from_numpy(self.outputs[0]['host']).reshape(batch_size, num_dets)
        boxes = torch.from_numpy(self.outputs[1]['host']).reshape(batch_size, num_dets, 4)
        scores = torch.from_numpy(self.outputs[2]['host']).reshape(batch_size, num_dets)

        return labels.to(self.device), boxes.to(self.device), scores.to(self.device)

# --- Funciones auxiliares y bucle principal (sin cambios) ---

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
            label_map[int(cat['id'])] = cat['name']
    except Exception as e:
        print(f"No se pudo obtener el mapeo de etiquetas del dataset COCO: {e}. Se usará un mapa vacío.")
    return label_map

def compute_label_offset(pred_labels_tensor: torch.Tensor, label_map: dict) -> int:
    if label_map and pred_labels_tensor.numel() > 0:
        raw_pred_min = int(pred_labels_tensor.min().item())
        min_label_in_map = min(label_map.keys())
        return min_label_in_map - raw_pred_min
    return 0

def draw_inferences(image: Image.Image, labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor,
                     label_map=None, color_map=None, threshold: float = 0.5,
                     label_offset: int = 0, save_path: str = None, radius: int = 2):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    valid_indices = scores > threshold
    if valid_indices.sum() == 0:
        if save_path: image.save(save_path)
        return image

    valid_labels = labels[valid_indices]
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    padding = 2
    for j, box in enumerate(valid_boxes):
        x0, y0, x1, y1 = map(int, box)
        pred_id = valid_labels[j].item() + label_offset
        color = color_map.get(pred_id, "black") if color_map is not None else "black"
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=color, width=2)
        label_text = label_map.get(pred_id, f"ID:{pred_id}")
        text = f"{label_text} {valid_scores[j].item():.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_height = bbox[3] - bbox[1]
        text_width = bbox[2] - bbox[0]
        text_position = (x0 + padding, y0 - text_height - padding if y0 > text_height + padding else y0 + padding)
        draw.rectangle([text_position[0], text_position[1], text_position[0] + text_width, text_position[1] + text_height], fill=color)
        draw.text(text_position, text, font=font, fill="white")

    if save_path:
        image.save(save_path)
    return image

def draw_ground_truth(image: Image.Image, annotations: list, label_map=None, color_map=None, radius: int = 2):
    """
    Dibuja las anotaciones ground truth sobre la imagen.
    Se espera que 'annotations' sea una lista de diccionarios con al menos 'bbox' y 'category_id'.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    padding = 2
    for ann in annotations:
        if "bbox" not in ann or "category_id" not in ann:
            continue
        box = ann["bbox"]
        x0, y0, x1, y1 = map(int, box)
        cat_id = int(ann["category_id"])
        color = color_map.get(cat_id, "red") if color_map is not None else "red"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        if label_map is not None:
            label_text = label_map.get(cat_id, str(cat_id))
        else:
            label_text = str(cat_id)
        text = f"GT: {label_text}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if y0 - text_height - padding >= 0:
            text_position = (x0 + padding, y0 - text_height - padding)
        else:
            text_position = (x0 + padding, y1 + padding)
        draw.text(text_position, text, font=font, fill=color, stroke_width=1, stroke_fill="black")
    return image

def save_debug_comparison(original_image: Image.Image, pred_labels: torch.Tensor, pred_boxes: torch.Tensor,
                            pred_scores: torch.Tensor, gt_annotations: list, label_map, color_map,
                            threshold: float, label_offset: int, out_path: str):
    """
    Genera una imagen comparativa para debug:
      - Izquierda: Imagen con inferencias predichas por el modelo.
      - Derecha: Imagen con las anotaciones ground truth dibujadas.
    Guarda la imagen comparativa en 'out_path'.
    """
    pred_img = original_image.copy()
    gt_img = original_image.copy()

    pred_img = draw_inferences(pred_img, pred_labels, pred_boxes, pred_scores,
                               label_map=label_map, color_map=color_map, threshold=threshold,
                               label_offset=label_offset)

    gt_img = draw_ground_truth(gt_img, gt_annotations, label_map=label_map, color_map=color_map)

    width, height = original_image.size
    combined = Image.new('RGB', (width*2, height))
    combined.paste(pred_img, (0,0))
    combined.paste(gt_img, (width,0))
    combined.save(out_path)

def plot_val_dataset_distribution(dataset, label_map, args):
    """
    Genera una gráfica de barras con el número de instancias por clase
    en el dataset de validación, mostrando además el valor exacto sobre cada barra.
    """
    print("\n--- Generando distribución del dataset de validación ---")

    # Contar cuántas anotaciones hay por cada categoría
    cat_counts = Counter()
    ann_list = dataset.coco.dataset.get('annotations', [])
    for ann in ann_list:
        cat_counts[ann['category_id']] += 1

    # Ordenar por ID de categoría
    cat_ids = sorted(cat_counts.keys())
    cat_names = [label_map.get(cat_id, str(cat_id)) for cat_id in cat_ids]
    counts = [cat_counts[cat_id] for cat_id in cat_ids]

    # Graficar
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cat_names, counts, color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Clases')
    plt.ylabel('Número de instancias')
    plt.title('Distribución de instancias por clase (Validación)')

    # Mostrar valor exacto sobre cada barra
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01 * max(counts),
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=9
        )

    dist_path = os.path.join(args.base_eval_dir, "dataset_distribution.png")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    print(f"Distribución de instancias por clase guardada en: {dist_path}")

def plot_score_distributions(score_distribution, label_map, output_dir, detection_threshold=0.5):
    """
    Genera histogramas de distribución de scores por clase.
    Útil para analizar si el modelo genera predicciones con scores bajos que se filtran.
    """
    num_classes = len(score_distribution)
    cols = 4
    rows = (num_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if num_classes > 1 else [axes]

    sorted_classes = sorted(score_distribution.items())

    for idx, (cat_id, scores) in enumerate(sorted_classes):
        if idx >= len(axes):
            break
        ax = axes[idx]
        cat_name = label_map.get(cat_id, str(cat_id))

        if scores:
            # Filtrar scores muy bajos para mejor visualización
            scores_filtered = [s for s in scores if s > 0.01]

            if scores_filtered:
                ax.hist(scores_filtered, bins=50, range=(0, 1), color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
                ax.axvline(x=detection_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={detection_threshold}')
                ax.axvline(x=0.25, color='orange', linestyle='--', linewidth=1.5, label='Threshold=0.25')

                # Estadísticas
                above_threshold = sum(1 for s in scores if s > detection_threshold)
                above_025 = sum(1 for s in scores if s > 0.25)
                above_01 = sum(1 for s in scores if s > 0.1)
                total = len(scores)

                # Resaltar clases minoritarias
                title_color = 'red' if cat_name in ['Can', 'Metal_Debris'] else 'black'
                ax.set_title(f"{cat_name}\n>{detection_threshold}: {above_threshold} | >0.25: {above_025} | >0.1: {above_01}\nTotal: {total}",
                            fontsize=9, color=title_color, fontweight='bold' if cat_name in ['Can', 'Metal_Debris'] else 'normal')
            else:
                ax.set_title(f"{cat_name}\nSin predicciones >0.01", fontsize=9)
        else:
            ax.set_title(f"{cat_name}\nSin predicciones", fontsize=9)

        ax.set_xlabel('Score', fontsize=8)
        ax.set_ylabel('Frecuencia', fontsize=8)
        ax.set_xlim(0, 1)

    # Ocultar ejes vacíos
    for idx in range(len(sorted_classes), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Distribución de Scores por Clase (Threshold de evaluación: {detection_threshold})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Generar reporte de texto
    report = "Distribución de Scores por Clase\n" + "=" * 70 + "\n"
    report += f"{'Clase':<20} {'Total':>10} {'>0.5':>10} {'>0.25':>10} {'>0.1':>10} {'Media':>10}\n"
    report += "-" * 70 + "\n"

    for cat_id, scores in sorted_classes:
        cat_name = label_map.get(cat_id, str(cat_id))
        if scores:
            above_05 = sum(1 for s in scores if s > 0.5)
            above_025 = sum(1 for s in scores if s > 0.25)
            above_01 = sum(1 for s in scores if s > 0.1)
            mean_score = np.mean(scores)
            report += f"{cat_name:<20} {len(scores):>10} {above_05:>10} {above_025:>10} {above_01:>10} {mean_score:>10.4f}\n"
        else:
            report += f"{cat_name:<20} {'0':>10} {'0':>10} {'0':>10} {'0':>10} {'N/A':>10}\n"

    report_path = os.path.join(output_dir, "score_distribution_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Distribución de scores guardada en: {output_dir}/score_distributions.png")
    print(f"Reporte de scores guardado en: {report_path}")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

def match_detections_hungarian(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """
    Empareja detecciones con ground truth usando el Hungarian Algorithm (optimal assignment).

    Este método garantiza el matching óptimo global maximizando la suma total de IoU.
    Es el método estándar usado en DETR y detectores modernos end-to-end.

    Args:
        gt_boxes: Lista de bounding boxes GT [x1, y1, x2, y2]
        gt_labels: Lista de etiquetas GT
        pred_boxes: Lista de bounding boxes predichas [x1, y1, x2, y2]
        pred_labels: Lista de etiquetas predichas
        pred_scores: Lista de scores de confianza (no se usa en Hungarian, pero se mantiene por compatibilidad)
        iou_thresh: Umbral mínimo de IoU para considerar un match válido

    Returns:
        matched_pairs: Lista de tuplas (gt_label, pred_label) para detecciones emparejadas
        unmatched_gts: Lista de etiquetas GT sin emparejar (falsos negativos)
        unmatched_preds: Lista de etiquetas predichas sin emparejar (falsos positivos)
    """
    if len(pred_boxes) == 0:
        unmatched_gts = list(gt_labels)
        return [], unmatched_gts, []

    if len(gt_boxes) == 0:
        unmatched_preds = list(pred_labels)
        return [], [], unmatched_preds

    # 1. Construir matriz de IoU (pred_boxes x gt_boxes)
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)
    iou_matrix = np.zeros((num_preds, num_gts))

    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = iou(pred_box, gt_box)

    # 2. Convertir a matriz de costos (Hungarian minimiza, así que usamos -IoU)
    cost_matrix = -iou_matrix

    # 3. Aplicar Hungarian Algorithm para encontrar el matching óptimo
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # 4. Filtrar matches que no superen el threshold de IoU
    matched_pairs = []
    matched_pred_indices = set()
    matched_gt_indices = set()

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_thresh:
            matched_pairs.append((gt_labels[gt_idx], pred_labels[pred_idx]))
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)

    # 5. Identificar predicciones y GT sin emparejar
    unmatched_preds = [pred_labels[i] for i in range(num_preds) if i not in matched_pred_indices]
    unmatched_gts = [gt_labels[j] for j in range(num_gts) if j not in matched_gt_indices]

    return matched_pairs, unmatched_gts, unmatched_preds


def match_detections_greedy(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5):
    """
    Empareja detecciones con ground truth usando un algoritmo Greedy basado en scores.

    Este método ordena las predicciones por score (mayor a menor) y asigna cada una al
    mejor GT disponible. Es más rápido que Hungarian pero puede producir matchings subóptimos.

    Útil si quieres que la evaluación refleje un comportamiento más "realista" donde las
    predicciones con mayor confianza tienen prioridad.

    Args:
        gt_boxes: Lista de bounding boxes GT [x1, y1, x2, y2]
        gt_labels: Lista de etiquetas GT
        pred_boxes: Lista de bounding boxes predichas [x1, y1, x2, y2]
        pred_labels: Lista de etiquetas predichas
        pred_scores: Lista de scores de confianza (SE USA para ordenar)
        iou_thresh: Umbral mínimo de IoU para considerar un match válido

    Returns:
        matched_pairs: Lista de tuplas (gt_label, pred_label) para detecciones emparejadas
        unmatched_gts: Lista de etiquetas GT sin emparejar (falsos negativos)
        unmatched_preds: Lista de etiquetas predichas sin emparejar (falsos positivos)
    """
    if len(pred_boxes) == 0:
        unmatched_gts = list(gt_labels)
        return [], unmatched_gts, []

    if len(gt_boxes) == 0:
        unmatched_preds = list(pred_labels)
        return [], [], unmatched_preds

    # Ordenar predicciones por score (mayor a menor)
    preds = sorted(zip(pred_boxes, pred_labels, pred_scores), key=lambda x: x[2], reverse=True)
    matched_pairs = []
    unmatched_preds = []
    assigned_gt = set()

    for p_box, p_label, p_score in preds:
        best_iou = -1
        best_gt_idx = -1

        # Buscar el mejor GT disponible para esta predicción
        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in assigned_gt:
                continue
            current_iou = iou(p_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = idx

        # Si encontramos un match por IoU
        if best_iou >= iou_thresh and best_gt_idx != -1:
            assigned_gt.add(best_gt_idx)
            # Guardamos (gt_label, pred_label) aunque no coincidan (para confusion matrix)
            matched_pairs.append((gt_labels[best_gt_idx], p_label))
        else:
            unmatched_preds.append(p_label)

    # Ground truth sin emparejar = Falsos Negativos
    unmatched_gts = [gt_labels[i] for i in range(len(gt_labels)) if i not in assigned_gt]
    return matched_pairs, unmatched_gts, unmatched_preds


def match_detections_in_image(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh=0.5, method='hungarian'):
    """
    Wrapper que llama al método de matching apropiado.

    Args:
        method: 'hungarian' (óptimo) o 'greedy' (basado en scores)
    """
    if method == 'hungarian':
        return match_detections_hungarian(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh)
    elif method == 'greedy':
        return match_detections_greedy(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thresh)
    else:
        raise ValueError(f"Método de matching desconocido: {method}. Use 'hungarian' o 'greedy'.")

def evaluate_on_dataset(args, cfg, device):
    if args.onnx:
        engine = ONNXEngine(args.onnx, args.device)
    elif args.tensorrt:
        engine = TensorRTEngine(args.tensorrt, args.device)
    else:
        cfg.resume = args.resume
        engine = PyTorchEngine(cfg, device)

    dataloader = cfg.val_dataloader
    dataset = dataloader.dataset
    label_map = get_label_map(dataset)
    color_map = generate_color_map_from_label_map(label_map)
    os.makedirs(args.base_eval_dir, exist_ok=True)

    # Informar configuración de evaluación
    print("\n" + "=" * 70)
    print("CONFIGURACIÓN DE EVALUACIÓN")
    print("=" * 70)
    print(f"  Método de matching:       {args.matching_method.upper()}")
    print(f"  IoU threshold:            {args.iou_threshold}")
    print(f"  Detection threshold:      {args.detection_threshold}")
    if args.matching_method == 'hungarian':
        print("  Nota: Hungarian matching es el estándar para detectores DETR/end-to-end")
    else:
        print("  Nota: Greedy matching prioriza detecciones con mayor confianza")
    print("=" * 70 + "\n")

    # Generar distribución del dataset
    plot_val_dataset_distribution(dataset, label_map, args)

    all_y_true, all_y_pred = [], []
    fp_counter, fn_counter = Counter(), Counter()
    coco_predictions = []
    latencies = []

    # Contadores de diagnóstico
    total_matched = 0
    total_fn = 0
    total_fp = 0
    total_gt = 0
    total_pred = 0

    # Distribución de scores por clase (para análisis de umbral)
    score_distribution = {cat_id: [] for cat_id in label_map.keys()}

    for batch in tqdm(dataloader, desc="Evaluando batches"):
        images, targets = batch
        images = images.to(device)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)

        start_time = time.perf_counter()
        labels_batch, boxes_batch, scores_batch = engine(images, orig_target_sizes)
        latencies.append(time.perf_counter() - start_time)

        for i in range(len(images)):
            target = targets[i]
            image_id = target['image_id'].item()

            # Después de las transformaciones, las anotaciones están en 'boxes' y 'labels'
            gt_boxes_tensor = target.get('boxes', torch.empty(0, 4))
            gt_labels_tensor = target.get('labels', torch.empty(0, dtype=torch.int64))

            # IMPORTANTE: Las cajas GT vienen en las coordenadas del tensor redimensionado (640x640),
            # pero necesitamos escalarlas de vuelta al tamaño original para compararlas con las predicciones
            # que el modelo devuelve en coordenadas originales
            orig_size = target.get('orig_size', torch.tensor([images[i].shape[1], images[i].shape[2]]))
            current_h, current_w = images[i].shape[1], images[i].shape[2]
            orig_h, orig_w = orig_size[0].item(), orig_size[1].item()

            # Escalar las cajas GT de vuelta al tamaño original
            if len(gt_boxes_tensor) > 0:
                scale_h = orig_h / current_h
                scale_w = orig_w / current_w
                gt_boxes_tensor_scaled = gt_boxes_tensor.clone()
                gt_boxes_tensor_scaled[:, 0] *= scale_w  # x1
                gt_boxes_tensor_scaled[:, 1] *= scale_h  # y1
                gt_boxes_tensor_scaled[:, 2] *= scale_w  # x2
                gt_boxes_tensor_scaled[:, 3] *= scale_h  # y2
                gt_boxes = gt_boxes_tensor_scaled.tolist()
            else:
                gt_boxes = []

            gt_labels = gt_labels_tensor.tolist() if len(gt_labels_tensor) > 0 else []

            raw_pred_labels = labels_batch[i].cpu()
            pred_boxes = boxes_batch[i].cpu()
            pred_scores = scores_batch[i].cpu()

            label_offset = compute_label_offset(raw_pred_labels, label_map)

            # Contar ground truth
            total_gt += len(gt_labels)

            # Recolectar distribución de scores por clase (ANTES del filtrado)
            for j in range(len(raw_pred_labels)):
                pred_class = raw_pred_labels[j].item() + label_offset
                if pred_class in score_distribution:
                    score_distribution[pred_class].append(pred_scores[j].item())

            if args.coco_eval and COCO_TOOLS_AVAILABLE:
                for j in range(len(raw_pred_labels)):
                    if pred_scores[j] > 0.001:
                        box = pred_boxes[j].tolist()
                        coco_box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                        coco_predictions.append({
                            'image_id': image_id,
                            'category_id': raw_pred_labels[j].item() + label_offset,
                            'bbox': coco_box,
                            'score': pred_scores[j].item()
                        })

            filtered_indices = pred_scores > args.detection_threshold
            filtered_pred_boxes = pred_boxes[filtered_indices].tolist()
            filtered_pred_labels = (raw_pred_labels[filtered_indices] + label_offset).tolist()
            filtered_pred_scores = pred_scores[filtered_indices].tolist()

            # Contar predicciones filtradas
            total_pred += len(filtered_pred_labels)

            matched, unmatched_gts, unmatched_preds = match_detections_in_image(
                gt_boxes, gt_labels, filtered_pred_boxes, filtered_pred_labels, filtered_pred_scores,
                args.iou_threshold, method=args.matching_method)

            # Actualizar contadores de diagnóstico
            total_matched += len(matched)
            total_fn += len(unmatched_gts)
            total_fp += len(unmatched_preds)

            for gt_lab, pred_lab in matched:
                all_y_true.append(gt_lab)
                all_y_pred.append(pred_lab)
            for gt_lab in unmatched_gts:
                fn_counter[gt_lab] += 1
            for pred_lab in unmatched_preds:
                fp_counter[pred_lab] += 1

            # Modo debug: generar imágenes comparativas si hay diferencias entre predicción y ground truth
            if args.debug:
                num_gt = len(gt_labels)
                num_pred = (pred_scores > args.detection_threshold).sum().item()
                if num_pred != num_gt:
                    # Construir ruta correcta de la imagen
                    img_info = dataset.coco.loadImgs(image_id)[0]
                    img_path = img_info['file_name']

                    # Si la ruta no existe, buscar el archivo por nombre en los directorios del dataset
                    if not os.path.exists(img_path):
                        img_dirs = dataset.image_dirs if isinstance(dataset.image_dirs, list) else [dataset.image_dirs]
                        img_basename = os.path.basename(img_path)
                        found = False
                        for img_dir in img_dirs:
                            full_path = os.path.join(img_dir, img_basename)
                            if os.path.exists(full_path):
                                img_path = full_path
                                found = True
                                break
                        if not found:
                            print(f"Advertencia: No se encontró la imagen {img_basename} en ningún directorio configurado. Saltando debug para image_id={image_id}")
                            continue

                    original_image = Image.open(img_path).convert("RGB")
                    debug_out = os.path.join(args.base_eval_dir, "debug_images", f"debug_{image_id}.jpg")

                    # Convertir gt_boxes y gt_labels a formato de anotaciones para draw_ground_truth
                    gt_annotations_dict = [{'bbox': box, 'category_id': label} for box, label in zip(gt_boxes, gt_labels)]

                    save_debug_comparison(original_image, raw_pred_labels, pred_boxes, pred_scores,
                                        gt_annotations_dict, label_map, color_map, args.detection_threshold,
                                        label_offset, debug_out)

            if args.draw:
                # Construir ruta correcta de la imagen
                img_info = dataset.coco.loadImgs(image_id)[0]
                img_path = img_info['file_name']

                # Si la ruta no existe, buscar el archivo por nombre en los directorios del dataset
                if not os.path.exists(img_path):
                    img_dirs = dataset.image_dirs if isinstance(dataset.image_dirs, list) else [dataset.image_dirs]
                    img_basename = os.path.basename(img_path)
                    found = False
                    for img_dir in img_dirs:
                        full_path = os.path.join(img_dir, img_basename)
                        if os.path.exists(full_path):
                            img_path = full_path
                            found = True
                            break
                    if not found:
                        print(f"Advertencia: No se encontró la imagen {img_basename} en ningún directorio configurado. Saltando draw para image_id={image_id}")
                        continue

                original_image = Image.open(img_path).convert("RGB")
                draw_inferences(original_image, raw_pred_labels, pred_boxes, pred_scores,
                                label_map, color_map, args.detection_threshold, label_offset,
                                save_path=os.path.join(args.base_eval_dir, "predicted_images", f"{image_id}_pred.jpg"))
                
    print("\n--- Resultados de Rendimiento ---")
    if len(latencies) > 1:
        avg_latency_ms = np.mean(latencies[1:]) * 1000
        avg_fps = 1.0 / np.mean(latencies[1:])
        print(f"Latencia Promedio: {avg_latency_ms:.2f} ms")
        print(f"Throughput (FPS): {avg_fps:.2f} FPS")
    else:
        print("No hay suficientes mediciones para reportar latencia.")

    # --- Distribución de Scores por Clase ---
    print("\n--- Generando distribución de scores por clase ---")
    plot_score_distributions(score_distribution, label_map, args.base_eval_dir, args.detection_threshold)

    print("\n--- Métricas de Clasificación (basado en IoU) ---")

    if label_map and any(label_map):
        class_names = [label_map.get(i, str(i)) for i in sorted(label_map.keys())]
        labels_to_report = sorted(list(set(all_y_true + all_y_pred)))

        # Crear nombres consistentes para target_names
        target_names = []
        for l in labels_to_report:
            target_names.append(label_map.get(l, f"Clase_{l}"))

        try:
            report = classification_report(all_y_true, all_y_pred, labels=labels_to_report,
                                           target_names=target_names,
                                           zero_division=0, digits=4)
            print(report)

            # Guardar el reporte de clasificación como imagen PNG
            plt.figure(figsize=(10, 8))
            plt.text(0.01, 0.05, report, {'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.title("Reporte de Clasificación (Detecciones emparejadas)")
            report_path = os.path.join(args.base_eval_dir, "classification_report.png")
            plt.tight_layout()
            plt.savefig(report_path)
            plt.close()
            print(f"Reporte de clasificación guardado en: {report_path}")
        except Exception as e:
            print(f"No se pudo generar el reporte de clasificación: {e}")

        # Generar matriz de confusión con nombres correctos
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_to_report)

        # Calcular tamaño dinámico basado en número de clases
        num_classes = len(labels_to_report)
        fig_size = max(10, num_classes * 0.8)

        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names,
                    cbar_kws={'label': 'Cantidad'})
        plt.xlabel("Predicción", fontsize=12)
        plt.ylabel("Real (Ground Truth)", fontsize=12)
        plt.title("Matriz de Confusión", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        cm_path = os.path.join(args.base_eval_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Matriz de confusión guardada en: {cm_path}")

        # Generar reporte y gráfica de Falsos Positivos y Falsos Negativos
        print("\n--- Generando reporte de Falsos Positivos y Falsos Negativos ---")
        fp_fn_report = "Reporte de Falsos Positivos y Falsos Negativos (No Detecciones)\n"
        fp_fn_report += "=" * 70 + "\n"
        fp_fn_report += f"{'Categoría':<30} {'Falsos Positivos':>15} {'Falsos Negativos':>15}\n"
        fp_fn_report += "-" * 70 + "\n"
        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            fp = fp_counter.get(cat_id, 0)
            fn = fn_counter.get(cat_id, 0)
            fp_fn_report += f"{cat_name} ({cat_id}){'':<20} {fp:>15} {fn:>15}\n"

        fp_fn_report_path = os.path.join(args.base_eval_dir, "fp_fn_report.txt")
        with open(fp_fn_report_path, "w") as f:
            f.write(fp_fn_report)
        print(f"Reporte de FP/FN guardado en: {fp_fn_report_path}")

        # Graficar los contadores de FP y FN por clase
        classes = [label_map[k] for k in sorted(label_map.keys())]
        fp_counts = [fp_counter.get(k, 0) for k in sorted(label_map.keys())]
        fn_counts = [fn_counter.get(k, 0) for k in sorted(label_map.keys())]

        x = np.arange(len(classes))
        width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, fp_counts, width, label='Falsos Positivos', color='salmon')
        plt.bar(x + width/2, fn_counts, width, label='Falsos Negativos', color='lightblue')
        plt.xticks(x, classes, rotation=45, ha='right')
        plt.xlabel("Clases")
        plt.ylabel("Cantidad")
        plt.title("Falsos Positivos y Falsos Negativos por Clase")
        plt.legend()
        fp_fn_bar_path = os.path.join(args.base_eval_dir, "fp_fn_bar_chart.png")
        plt.tight_layout()
        plt.savefig(fp_fn_bar_path)
        plt.close()
        print(f"Gráfica de FP/FN guardada en: {fp_fn_bar_path}")

    if args.coco_eval and COCO_TOOLS_AVAILABLE:
        print("\n--- Evaluación COCO mAP ---")
        if not coco_predictions:
            print("No se generaron predicciones para la evaluación COCO.")
        else:
            pred_json_path = os.path.join(args.base_eval_dir, "coco_predictions.json")
            with open(pred_json_path, 'w') as f:
                json.dump(coco_predictions, f)

            coco_dt = dataset.coco.loadRes(pred_json_path)
            coco_eval = COCOeval(dataset.coco, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            # Guardar los resultados de COCO en un archivo de texto
            report_path = os.path.join(args.base_eval_dir, "coco_evaluation_report.txt")
            original_stdout = sys.stdout
            with open(report_path, 'w') as f:
                sys.stdout = f
                coco_eval.summarize()
            sys.stdout = original_stdout
            print(f"Reporte de evaluación COCO guardado en: {report_path}")

            # --- AP por Clase (COCO) ---
            print("\n--- AP por Clase (COCO) ---")
            precisions = coco_eval.eval['precision']
            # precision tiene forma: [TxRxKxAxM]
            # T: IoU thresholds, R: recall thresholds, K: categories, A: areas, M: maxDets

            cat_ids = dataset.coco.getCatIds()
            cat_names_list = [dataset.coco.loadCats(cat_id)[0]['name'] for cat_id in cat_ids]

            per_class_ap = {}
            per_class_ap50 = {}
            per_class_ap75 = {}
            per_class_report = "AP por Clase (COCO Metrics)\n" + "=" * 60 + "\n"
            per_class_report += f"{'Clase':<20} {'AP@0.5:0.95':>12} {'AP@0.50':>12} {'AP@0.75':>12}\n"
            per_class_report += "-" * 60 + "\n"

            for idx, cat_id in enumerate(cat_ids):
                cat_name = cat_names_list[idx]

                # AP@0.50:0.95 (promedio sobre todos los IoU thresholds)
                precision_all = precisions[:, :, idx, 0, -1]
                precision_all = precision_all[precision_all > -1]
                ap = np.mean(precision_all) if precision_all.size else 0.0
                per_class_ap[cat_name] = ap

                # AP@0.50 (IoU threshold index 0)
                precision_50 = precisions[0, :, idx, 0, -1]
                precision_50 = precision_50[precision_50 > -1]
                ap50 = np.mean(precision_50) if precision_50.size else 0.0
                per_class_ap50[cat_name] = ap50

                # AP@0.75 (IoU threshold index 5)
                precision_75 = precisions[5, :, idx, 0, -1]
                precision_75 = precision_75[precision_75 > -1]
                ap75 = np.mean(precision_75) if precision_75.size else 0.0
                per_class_ap75[cat_name] = ap75

                per_class_report += f"{cat_name:<20} {ap:>12.4f} {ap50:>12.4f} {ap75:>12.4f}\n"

            print(per_class_report)

            # Guardar reporte por clase
            per_class_path = os.path.join(args.base_eval_dir, "per_class_ap_report.txt")
            with open(per_class_path, 'w') as f:
                f.write(per_class_report)
            print(f"Reporte AP por clase guardado en: {per_class_path}")

            # Gráfica de AP por clase
            plt.figure(figsize=(14, 6))
            x = np.arange(len(cat_names_list))
            width = 0.25

            # Colores: rojo para Can y Metal_Debris, azul para el resto
            colors_ap = ['#d62728' if name in ['Can', 'Metal_Debris'] else '#1f77b4' for name in cat_names_list]

            bars1 = plt.bar(x - width, [per_class_ap[n] for n in cat_names_list], width, label='AP@0.5:0.95', color=colors_ap, alpha=0.9)
            bars2 = plt.bar(x, [per_class_ap50[n] for n in cat_names_list], width, label='AP@0.50', color='#2ca02c', alpha=0.7)
            bars3 = plt.bar(x + width, [per_class_ap75[n] for n in cat_names_list], width, label='AP@0.75', color='#ff7f0e', alpha=0.7)

            plt.xticks(x, cat_names_list, rotation=45, ha='right')
            plt.ylabel('Average Precision (AP)')
            plt.title('AP por Clase (COCO Metrics)')
            plt.axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Objetivo: 0.5')
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(args.base_eval_dir, "per_class_ap.png"), dpi=150)
            plt.close()
            print(f"Gráfica AP por clase guardada en: {args.base_eval_dir}/per_class_ap.png")

    # --- RESUMEN DIAGNÓSTICO ---
    print("\n" + "=" * 70)
    print("RESUMEN DIAGNÓSTICO")
    print("=" * 70)

    # Estadísticas globales
    print(f"\n{'ESTADÍSTICAS GLOBALES':^70}")
    print("-" * 70)
    print(f"  Total Ground Truth:       {total_gt:>8}")
    print(f"  Total Predicciones:       {total_pred:>8}")
    print(f"  Total Emparejados:        {total_matched:>8}")
    print(f"  Falsos Negativos (FN):    {total_fn:>8}  ({100*total_fn/max(total_gt,1):.1f}% de GT)")
    print(f"  Falsos Positivos (FP):    {total_fp:>8}  ({100*total_fp/max(total_pred,1):.1f}% de pred)")

    # Calcular recall y precision globales
    global_recall = total_matched / max(total_gt, 1)
    global_precision = total_matched / max(total_pred, 1)
    print(f"\n  Recall Global:            {global_recall:>8.2%}")
    print(f"  Precision Global:         {global_precision:>8.2%}")

    # Análisis por clase (especialmente las minoritarias)
    print(f"\n{'ANÁLISIS POR CLASE (Métricas de Detección)':^90}")
    print("-" * 90)
    print(f"{'Clase':<20} {'GT':>6} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Estado':>12}")
    print("-" * 90)

    # Calcular TP por clase desde all_y_true y all_y_pred (matched pairs)
    class_tp_count = Counter()
    for yt, yp in zip(all_y_true, all_y_pred):
        if yt == yp:  # Match correcto (mismo label)
            class_tp_count[yt] += 1

    # IMPORTANTE: Para cada clase calculamos:
    # - TP: Detecciones correctas (matched con label correcto)
    # - FN: Ground truth sin detectar
    # - FP: Predicciones incorrectas (matched con label incorrecto + sin match)
    # - Total GT = TP + FN (no solo los matched!)
    # - Total Pred = TP + FP
    # - Recall = TP / (TP + FN) = TP / Total_GT
    # - Precision = TP / (TP + FP) = TP / Total_Pred
    # - F1 = 2 * (Precision * Recall) / (Precision + Recall)

    minority_classes = ['Can', 'Metal_Debris']
    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]

        # Contadores
        tp = class_tp_count.get(cat_id, 0)
        fn = fn_counter.get(cat_id, 0)
        fp = fp_counter.get(cat_id, 0)

        # CORRECCIÓN: Total GT incluye tanto TP como FN
        total_gt = tp + fn
        total_pred = tp + fp

        # Métricas
        recall = tp / max(total_gt, 1)
        precision = tp / max(total_pred, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

        # Determinar estado basado en F1-score
        if f1 < 0.3:
            estado = "⚠️  CRÍTICO"
        elif f1 < 0.5:
            estado = "⚠️  BAJO"
        elif f1 < 0.7:
            estado = "MODERADO"
        else:
            estado = "✓ BUENO"

        # Resaltar clases minoritarias
        prefix = ">> " if cat_name in minority_classes else "   "
        print(f"{prefix}{cat_name:<17} {total_gt:>6} {tp:>6} {fn:>6} {fp:>6} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f} {estado:>12}")

    print("-" * 70)

    # Recomendaciones basadas en el análisis
    print(f"\n{'RECOMENDACIONES':^70}")
    print("-" * 70)

    # Identificar clases problemáticas
    problematic_classes = []
    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]
        tp = class_tp_count.get(cat_id, 0)
        fn = fn_counter.get(cat_id, 0)
        total_gt = tp + fn
        recall = tp / max(total_gt, 1)
        if recall < 0.5 and total_gt > 10:  # Solo si hay suficientes muestras
            problematic_classes.append((cat_name, recall, total_gt))

    if problematic_classes:
        print("\n  Clases con bajo recall (< 50%):")
        for name, recall, gt in sorted(problematic_classes, key=lambda x: x[1]):
            print(f"    - {name}: {recall:.1%} recall ({gt} GT)")
        print("\n  Acciones sugeridas:")
        print("    1. Aumentar class_weights en la función de pérdida para estas clases")
        print("    2. Usar ClassBalancedSampler para sobremuestrear imágenes con estas clases")
        print("    3. Aplicar augmentaciones específicas (class-aware augmentations)")
        print("    4. Considerar reducir detection_threshold para estas clases específicas")
    else:
        print("\n  ✓ Todas las clases tienen recall >= 50%")

    # Análisis de distribución de scores para clases problemáticas
    print(f"\n{'ANÁLISIS DE SCORES (Clases Minoritarias)':^70}")
    print("-" * 70)
    for cat_name in minority_classes:
        cat_id = next((k for k, v in label_map.items() if v == cat_name), None)
        if cat_id is not None and cat_id in score_distribution:
            scores = score_distribution[cat_id]
            if scores:
                above_05 = sum(1 for s in scores if s > 0.5)
                above_025 = sum(1 for s in scores if s > 0.25)
                above_01 = sum(1 for s in scores if s > 0.1)
                total = len(scores)
                print(f"\n  {cat_name}:")
                print(f"    Total predicciones (antes de filtrar): {total}")
                print(f"    Scores > 0.5: {above_05} ({100*above_05/max(total,1):.1f}%)")
                print(f"    Scores > 0.25: {above_025} ({100*above_025/max(total,1):.1f}%)")
                print(f"    Scores > 0.1: {above_01} ({100*above_01/max(total,1):.1f}%)")
                if total > 0:
                    print(f"    Score medio: {np.mean(scores):.3f}, mediana: {np.median(scores):.3f}")

    print("\n" + "=" * 70)

    # Guardar resumen diagnóstico en archivo
    diag_report_path = os.path.join(args.base_eval_dir, "diagnostic_summary.txt")
    with open(diag_report_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DIAGNÓSTICO\n")
        f.write("=" * 70 + "\n\n")
        f.write("ESTADÍSTICAS GLOBALES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Ground Truth:       {total_gt}\n")
        f.write(f"Total Predicciones:       {total_pred}\n")
        f.write(f"Total Emparejados:        {total_matched}\n")
        f.write(f"Falsos Negativos (FN):    {total_fn} ({100*total_fn/max(total_gt,1):.1f}% de GT)\n")
        f.write(f"Falsos Positivos (FP):    {total_fp} ({100*total_fp/max(total_pred,1):.1f}% de pred)\n")
        f.write(f"Recall Global:            {global_recall:.2%}\n")
        f.write(f"Precision Global:         {global_precision:.2%}\n\n")

        f.write("POR CLASE (Métricas Corregidas)\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Clase':<20} {'GT':>6} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>8} {'Precision':>10} {'F1':>8}\n")
        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            tp = class_tp_count.get(cat_id, 0)
            fn = fn_counter.get(cat_id, 0)
            fp = fp_counter.get(cat_id, 0)
            total_gt = tp + fn
            total_pred = tp + fp
            recall = tp / max(total_gt, 1)
            precision = tp / max(total_pred, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            f.write(f"{cat_name:<20} {total_gt:>6} {tp:>6} {fn:>6} {fp:>6} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f}\n")

    print(f"Resumen diagnóstico guardado en: {diag_report_path}")

def main():
    parser = argparse.ArgumentParser(description="Script de evaluación unificado para RT-DETR-V2.")
    parser.add_argument('-c', '--config', required=True, help="Ruta al archivo de configuración YAML.")
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('-r', '--resume', type=str, help="Ruta al checkpoint de PyTorch (.pth).")
    model_group.add_argument('--onnx', type=str, help="Ruta al modelo ONNX (.onnx).")
    model_group.add_argument('--tensorrt', type=str, help="Ruta al motor de TensorRT (.engine).")

    parser.add_argument('-d', '--device', type=str, default='cuda', help="Dispositivo (cpu o cuda).")
    parser.add_argument('-dt', '--detection_threshold', type=float, default=0.5, help="Umbral de confianza para detecciones.")
    parser.add_argument('-iout', '--iou_threshold', type=float, default=0.5, help="Umbral de IoU para emparejamiento.")
    parser.add_argument('--matching_method', type=str, default='hungarian', choices=['hungarian', 'greedy'],
                        help="Método de matching: 'hungarian' (óptimo, estándar DETR) o 'greedy' (más rápido, basado en scores).")
    parser.add_argument('--output_dir', type=str, default='eval_results', help="Directorio base para guardar resultados.")
    parser.add_argument('--coco_eval', action='store_true', help="Activar evaluación de mAP con pycocotools.")
    parser.add_argument('--draw', action='store_true', help="Guardar imágenes con detecciones dibujadas.")
    parser.add_argument('--debug', action='store_true', help="Guardar imágenes de debug comparativas entre predicción y ground truth cuando hay diferencias.")

    args = parser.parse_args()
    device = torch.device(args.device)

    model_path = args.resume or args.onnx or args.tensorrt
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    args.base_eval_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(args.base_eval_dir, exist_ok=True)

    cfg = YAMLConfig(args.config)

    if args.draw:
        os.makedirs(os.path.join(args.base_eval_dir, "predicted_images"), exist_ok=True)

    if args.debug:
        os.makedirs(os.path.join(args.base_eval_dir, "debug_images"), exist_ok=True)

    evaluate_on_dataset(args, cfg, device)

if __name__ == '__main__':
    main()