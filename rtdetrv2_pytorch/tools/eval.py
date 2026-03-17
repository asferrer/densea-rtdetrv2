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

# Optimización de memoria PyTorch para Jetson (debe ir ANTES de import torch)
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,garbage_collection_threshold:0.6'

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import colorsys
import gc
import json
from collections import Counter

# --- Imports opcionales (el script no crashea si faltan) ---
try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

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
    """Wrapper para la inferencia con ONNX Runtime.

    Optimizado para Jetson Orin Nano (8GB RAM compartida CPU/GPU):
    - Limita arena GPU para dejar memoria a PyTorch y al sistema
    - Usa búsqueda heurística de cuDNN (evita allocations de benchmarking)
    - Soporta fallback automático a CPU si CUDA no tiene memoria
    """
    def __init__(self, model_path, device_str):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime no está instalado. Instálalo con: pip install onnxruntime-gpu")

        print(f"Cargando modelo ONNX desde: {model_path}")

        # Session options para uso eficiente de memoria
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = True        # Reusar patrones de memoria
        sess_options.enable_cpu_mem_arena = False      # No pre-alocar arena CPU (ahorra RAM)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        if 'cuda' in device_str:
            # Liberar cache de PyTorch ANTES de inicializar ONNX Runtime
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            providers = [
                ('CUDAExecutionProvider', {
                    # Limitar arena GPU a 1GB (Jetson tiene 8GB compartidos)
                    'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
                    # Alocar solo lo necesario (no redondear a potencia de 2)
                    'arena_extend_strategy': 'kSameAsRequested',
                    # Heurística en vez de exhaustiva (evita allocations temporales grandes)
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    # No usar workspace máximo de cuDNN
                    'cudnn_conv_use_max_workspace': '0',
                    'do_copy_in_default_stream': '1',
                }),
                ('CPUExecutionProvider', {}),
            ]
        else:
            providers = [('CPUExecutionProvider', {})]

        try:
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        except Exception as e:
            print(f"Error creando sesión con providers solicitados: {e}")
            print("Intentando con CPUExecutionProvider...")
            self.session = ort.InferenceSession(
                model_path, sess_options=sess_options,
                providers=[('CPUExecutionProvider', {})]
            )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Determinar device real según el proveedor activo (puede caer a CPU por OOM)
        active_provider = self.session.get_providers()[0]
        if 'CUDA' in active_provider:
            self.device = torch.device(device_str)
        else:
            print("ONNX Runtime usando CPU. Evaluación continuará en CPU (más lento pero estable).")
            self.device = torch.device('cpu')
        print(f"Modelo ONNX cargado. Usando proveedor: {active_provider}")

    def __call__(self, images, orig_target_sizes):
        # Cast orig_target_sizes a int32 (el ONNX fue exportado con int32)
        sizes_np = orig_target_sizes.cpu().numpy().astype(np.int32)
        inputs = {
            self.input_names[0]: images.cpu().numpy(),
            self.input_names[1]: sizes_np
        }
        outputs = self.session.run(self.output_names, inputs)

        labels = torch.from_numpy(outputs[0]).to(self.device)
        boxes = torch.from_numpy(outputs[1]).to(self.device)
        scores = torch.from_numpy(outputs[2]).to(self.device)

        # Asegurar dimensión de batch (algunos modelos ONNX la omiten)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        return labels, boxes, scores

class TensorRTEngine:
    """
    Wrapper para la inferencia con TensorRT 10.x.
    Usa buffers PyTorch CUDA (en vez de PyCUDA) para evitar conflictos de streams
    con los aux streams internos de TRT 10.x en Jetson.
    """

    # Mapeo TRT dtype → torch dtype
    _TRT_TO_TORCH = None

    @staticmethod
    def _trt_dtype_to_torch(trt, trt_dtype):
        if TensorRTEngine._TRT_TO_TORCH is None:
            TensorRTEngine._TRT_TO_TORCH = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32:   torch.int32,
                trt.int64:   torch.int64,
                trt.int8:    torch.int8,
                trt.bool:    torch.bool,
            }
        return TensorRTEngine._TRT_TO_TORCH.get(trt_dtype, torch.float32)

    def __init__(self, engine_path, device_str):
        try:
            import tensorrt as trt
            self.trt = trt
        except ImportError as e:
            raise ImportError(f"Dependencia faltante para TensorRT: {e}. Revisa la instalación.")

        # Liberar cache de PyTorch ANTES de alocar buffers TensorRT
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        self.device = torch.device(device_str)
        self.stream = torch.cuda.Stream(device=self.device)

        print(f"Cargando motor de TensorRT desde: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if not self.engine:
            raise RuntimeError("No se pudo deserializar el motor de TensorRT. Verifica el archivo y los permisos.")

        self.context = self.engine.create_execution_context()
        self.tensors = self._allocate_buffers()
        print("Motor de TensorRT cargado exitosamente.")

    def _allocate_buffers(self):
        """Asigna buffers como tensores PyTorch CUDA (compatibles con TRT streams)."""
        tensors = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            torch_dtype = self._trt_dtype_to_torch(self.trt, trt_dtype)
            mode = self.engine.get_tensor_mode(name)

            # Manejar dimensiones dinámicas (-1) con valores esperados
            if -1 in shape:
                if name == "images":
                    shape = (1, 3, 640, 640)
                elif name == "orig_target_sizes":
                    shape = (1, 2)
                elif "labels" in name or "scores" in name:
                    shape = (1, 300)
                elif "boxes" in name:
                    shape = (1, 300, 4)

            # Configurar shapes de entrada para dimensiones dinámicas
            if mode == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)

            # Alocar tensor GPU con torch (comparte CUDA context con TRT)
            tensor = torch.empty(shape, dtype=torch_dtype, device=self.device).contiguous()
            self.context.set_tensor_address(name, tensor.data_ptr())

            tensors[name] = {
                'tensor': tensor,
                'shape': shape,
                'name': name,
                'dtype': torch_dtype,
                'is_input': mode == self.trt.TensorIOMode.INPUT,
            }

        return tensors

    def warmup(self, n=50):
        """Ejecuta inferencias de warmup para estabilizar GPU clocks y JIT en Jetson."""
        with torch.cuda.stream(self.stream):
            for _ in range(n):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
        print(f"[TRT] Warmup completado ({n} iteraciones)")

    def __call__(self, images, orig_target_sizes):
        # Copiar inputs a los buffers GPU con cast de dtype correcto
        img_buf = self.tensors["images"]
        size_buf = self.tensors["orig_target_sizes"]

        with torch.cuda.stream(self.stream):
            img_buf['tensor'].copy_(images[:1])  # batch=1
            size_buf['tensor'].copy_(orig_target_sizes[:1].to(size_buf['dtype']))

            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()

        # Leer outputs directamente de los tensores GPU
        labels = self.tensors['labels']['tensor'].clone()
        boxes = self.tensors['boxes']['tensor'].clone()
        scores = self.tensors['scores']['tensor'].clone()

        # Asegurar dimensión de batch
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if boxes.dim() == 2:
            boxes = boxes.unsqueeze(0)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        return labels, boxes, scores

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

def detect_minority_classes(class_gt_total: dict, label_map: dict, threshold_ratio: float = 0.5) -> list:
    """
    Detecta dinámicamente las clases minoritarias basándose en la distribución real del dataset.

    Args:
        class_gt_total: Diccionario {cat_id: count} con el total de GT por clase
        label_map: Diccionario {cat_id: cat_name} para mapear IDs a nombres
        threshold_ratio: Ratio respecto a la media. Clases con count < mean * ratio son minoritarias.
                        Default 0.5 significa clases con menos del 50% de la media.

    Returns:
        Lista de nombres de clases minoritarias (vacía si no hay ninguna)
    """
    if not class_gt_total:
        return []

    counts = list(class_gt_total.values())
    if not counts:
        return []

    mean_count = sum(counts) / len(counts)
    threshold = mean_count * threshold_ratio

    minority_classes = []
    for cat_id, count in class_gt_total.items():
        if count < threshold and count > 0:  # Solo si tiene al menos 1 instancia
            cat_name = label_map.get(cat_id, f"ID:{cat_id}")
            minority_classes.append(cat_name)

    return minority_classes

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

def plot_score_distributions(score_distribution, label_map, output_dir, detection_threshold=0.5, minority_classes=None):
    """
    Genera histogramas de distribución de scores por clase.
    Útil para analizar si el modelo genera predicciones con scores bajos que se filtran.

    Args:
        minority_classes: Lista de nombres de clases minoritarias (detectadas dinámicamente).
                         Si es None o vacía, no se resalta ninguna clase.
    """
    if minority_classes is None:
        minority_classes = []

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
        is_minority = cat_name in minority_classes

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

                # Resaltar clases minoritarias (solo si hay alguna detectada)
                title_color = 'red' if is_minority else 'black'
                ax.set_title(f"{cat_name}\n>{detection_threshold}: {above_threshold} | >0.25: {above_025} | >0.1: {above_01}\nTotal: {total}",
                            fontsize=9, color=title_color, fontweight='bold' if is_minority else 'normal')
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
        matched_pairs: Lista de tuplas (gt_label, pred_label, iou_value) para detecciones emparejadas
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
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy es necesario para match_detections_hungarian. Instálalo con: pip install scipy")
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)

    # 4. Filtrar matches que no superen el threshold de IoU
    matched_pairs = []
    matched_pred_indices = set()
    matched_gt_indices = set()

    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_thresh:
            # Ahora incluimos el IoU del match
            matched_pairs.append((gt_labels[gt_idx], pred_labels[pred_idx], iou_matrix[pred_idx, gt_idx]))
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
        matched_pairs: Lista de tuplas (gt_label, pred_label, iou_value) para detecciones emparejadas
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
            # Guardamos (gt_label, pred_label, iou) aunque no coincidan (para confusion matrix)
            matched_pairs.append((gt_labels[best_gt_idx], p_label, best_iou))
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
    if PLOTTING_AVAILABLE:
        plot_val_dataset_distribution(dataset, label_map, args)
    else:
        print("matplotlib no disponible. Saltando gráfica de distribución.")

    all_y_true, all_y_pred = [], []
    coco_predictions = []
    latencies = []

    # Contadores por clase para métricas correctas
    # TP: GT emparejado con predicción de la MISMA clase
    # FP: Predicción sin emparejar O emparejada con GT de clase DIFERENTE
    # FN: GT sin emparejar O emparejado con predicción de clase DIFERENTE
    class_tp = Counter()  # True Positives por clase
    class_fp = Counter()  # False Positives por clase
    class_fn = Counter()  # False Negatives por clase
    class_gt_total = Counter()  # Total GT por clase

    # Contadores de diagnóstico
    total_matched = 0
    total_fn = 0
    total_fp = 0
    total_gt = 0
    total_pred = 0

    # Métricas adicionales de localización y errores
    all_match_ious = []  # IoU de todos los matches (para Mean IoU)
    correct_class_matches = 0  # Matches con clase correcta (localización + clasificación)
    wrong_class_matches = 0  # Matches con clase incorrecta (localización OK, clasificación FAIL)
    background_fps = 0  # Predicciones sin ningún match (detecciones fantasma)

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

            # IMPORTANTE: Las cajas GT vienen del dataloader en formato CXCYWH NORMALIZADO
            # (después de la transformación ConvertBoxes con fmt='cxcywh', normalize=True)
            # Necesitamos convertirlas a XYXY en coordenadas de la imagen original
            # para compararlas con las predicciones del postprocessor.
            # NOTA: orig_size es [W, H] según la convención del postprocessor (ver box_revert.py)
            orig_size = target.get('orig_size', torch.tensor([images[i].shape[2], images[i].shape[1]]))  # Default: [W, H]
            orig_w, orig_h = orig_size[0].item(), orig_size[1].item()  # orig_size es [W, H]

            # Convertir cajas GT de CXCYWH normalizado a XYXY en píxeles originales
            if len(gt_boxes_tensor) > 0:
                # gt_boxes_tensor está en formato [cx, cy, w, h] normalizado (0-1)
                cx = gt_boxes_tensor[:, 0]
                cy = gt_boxes_tensor[:, 1]
                w = gt_boxes_tensor[:, 2]
                h = gt_boxes_tensor[:, 3]

                # Convertir a XYXY en coordenadas de la imagen original
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                gt_boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                gt_boxes = gt_boxes_xyxy.tolist()
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

            # Contar GT total por clase
            for gt_lab in gt_labels:
                class_gt_total[gt_lab] += 1

            # Procesar matches (ahora incluyen IoU)
            for match_tuple in matched:
                gt_lab, pred_lab, match_iou = match_tuple
                all_y_true.append(gt_lab)
                all_y_pred.append(pred_lab)
                all_match_ious.append(match_iou)

                if gt_lab == pred_lab:
                    # Match correcto: TP para esa clase
                    class_tp[gt_lab] += 1
                    correct_class_matches += 1
                else:
                    # Match incorrecto: FN para GT, FP para predicción
                    class_fn[gt_lab] += 1
                    class_fp[pred_lab] += 1
                    wrong_class_matches += 1

            # GT sin emparejar = FN
            for gt_lab in unmatched_gts:
                class_fn[gt_lab] += 1

            # Predicciones sin emparejar = FP (detecciones fantasma/background)
            for pred_lab in unmatched_preds:
                class_fp[pred_lab] += 1
                background_fps += 1

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
                        # Soportar tanto CocoDetection (img_folder) como MultiDirCocoDetection (image_dirs)
                        if hasattr(dataset, 'image_dirs'):
                            img_dirs = dataset.image_dirs if isinstance(dataset.image_dirs, list) else [dataset.image_dirs]
                        elif hasattr(dataset, 'img_folder'):
                            img_dirs = [dataset.img_folder]
                        else:
                            img_dirs = []
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
                    # Soportar tanto CocoDetection (img_folder) como MultiDirCocoDetection (image_dirs)
                    if hasattr(dataset, 'image_dirs'):
                        img_dirs = dataset.image_dirs if isinstance(dataset.image_dirs, list) else [dataset.image_dirs]
                    elif hasattr(dataset, 'img_folder'):
                        img_dirs = [dataset.img_folder]
                    else:
                        img_dirs = []
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

    # --- Detectar clases minoritarias dinámicamente ---
    minority_classes = detect_minority_classes(class_gt_total, label_map, threshold_ratio=0.5)
    if minority_classes:
        print(f"\n--- Clases minoritarias detectadas (< 50% de la media): {', '.join(minority_classes)} ---")

    # --- Distribución de Scores por Clase ---
    if PLOTTING_AVAILABLE:
        print("\n--- Generando distribución de scores por clase ---")
        plot_score_distributions(score_distribution, label_map, args.base_eval_dir, args.detection_threshold, minority_classes)

    print("\n--- Métricas de Clasificación (basado en IoU) ---")

    if label_map and any(label_map):
        class_names = [label_map.get(i, str(i)) for i in sorted(label_map.keys())]
        labels_to_report = sorted(list(set(all_y_true + all_y_pred)))

        # Crear nombres consistentes para target_names
        target_names = []
        for l in labels_to_report:
            target_names.append(label_map.get(l, f"Clase_{l}"))

        try:
            if not labels_to_report:
                raise ValueError("No hay detecciones emparejadas — el modelo no produjo detecciones válidas")
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn no disponible")
            report = classification_report(all_y_true, all_y_pred, labels=labels_to_report,
                                           target_names=target_names,
                                           zero_division=0, digits=4)
            print("\n" + "=" * 70)
            print("NOTA: Este reporte de sklearn SOLO considera detecciones emparejadas.")
            print("NO incluye objetos no detectados (FN) ni detecciones fantasma (FP).")
            print("Para métricas completas de detección, ver 'ANÁLISIS POR CLASE' más abajo.")
            print("=" * 70)
            print(report)

            # Guardar el reporte de clasificación como imagen PNG
            plt.figure(figsize=(10, 8))
            plt.text(0.01, 0.05, report, {'fontsize': 10}, fontproperties='monospace')
            plt.axis('off')
            plt.title("Reporte de Clasificación (Solo detecciones emparejadas por IoU)\nNOTA: NO incluye objetos no detectados")
            report_path = os.path.join(args.base_eval_dir, "classification_report.png")
            plt.tight_layout()
            plt.savefig(report_path)
            plt.close()
            print(f"Reporte de clasificación guardado en: {report_path}")
        except Exception as e:
            print(f"No se pudo generar el reporte de clasificación: {e}")

        # =====================================================================
        # REPORTE DE DETECCIÓN COMPLETO (incluye objetos no detectados)
        # =====================================================================
        print("\n" + "=" * 70)
        print("REPORTE DE DETECCIÓN COMPLETO")
        print("(Incluye objetos no detectados y detecciones fantasma)")
        print("=" * 70)

        # Construir reporte completo por clase
        complete_report = "\n{:<20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}\n".format(
            "Clase", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1-Score")
        complete_report += "-" * 94 + "\n"

        all_supports = []
        all_precisions_complete = []
        all_recalls_complete = []
        all_f1s_complete = []

        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            tp = class_tp.get(cat_id, 0)
            fp = class_fp.get(cat_id, 0)
            fn = class_fn.get(cat_id, 0)
            support = class_gt_total.get(cat_id, 0)  # Total GT de esta clase

            # Calcular métricas usando las fórmulas correctas
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

            complete_report += "{:<20} {:>8} {:>8} {:>8} {:>8} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
                cat_name, support, tp, fp, fn, precision, recall, f1)

            all_supports.append(support)
            all_precisions_complete.append(precision)
            all_recalls_complete.append(recall)
            all_f1s_complete.append(f1)

        complete_report += "-" * 94 + "\n"

        # Calcular promedios
        total_support = sum(all_supports)

        # Macro average (igual peso por clase)
        macro_prec = np.mean(all_precisions_complete)
        macro_rec = np.mean(all_recalls_complete)
        macro_f1_val = np.mean(all_f1s_complete)

        # Weighted average (ponderado por support)
        if total_support > 0:
            weighted_prec = sum(p * s for p, s in zip(all_precisions_complete, all_supports)) / total_support
            weighted_rec = sum(r * s for r, s in zip(all_recalls_complete, all_supports)) / total_support
            weighted_f1_val = sum(f * s for f, s in zip(all_f1s_complete, all_supports)) / total_support
        else:
            weighted_prec = weighted_rec = weighted_f1_val = 0.0

        complete_report += "\n{:<20} {:>8} {:>8} {:>8} {:>8} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
            "macro avg", "", "", "", "", macro_prec, macro_rec, macro_f1_val)
        complete_report += "{:<20} {:>8} {:>8} {:>8} {:>8} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(
            "weighted avg", total_support, sum(class_tp.values()), sum(class_fp.values()),
            sum(class_fn.values()), weighted_prec, weighted_rec, weighted_f1_val)

        print(complete_report)

        # Guardar reporte completo
        complete_report_path = os.path.join(args.base_eval_dir, "complete_detection_report.txt")
        with open(complete_report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE DETECCIÓN COMPLETO\n")
            f.write("(Incluye objetos no detectados y detecciones fantasma)\n")
            f.write("=" * 94 + "\n")
            f.write("\nFórmulas utilizadas:\n")
            f.write("  - Precision = TP / (TP + FP)\n")
            f.write("  - Recall = TP / (TP + FN)\n")
            f.write("  - F1 = 2 * (Precision * Recall) / (Precision + Recall)\n")
            f.write("\nDonde:\n")
            f.write("  - TP: Detección correcta (IoU >= threshold Y clase correcta)\n")
            f.write("  - FP: Detección incorrecta (sin GT match) O clase incorrecta\n")
            f.write("  - FN: Objeto no detectado O detectado con clase incorrecta\n")
            f.write("  - Support: Total de objetos GT de esa clase\n")
            f.write("=" * 94 + "\n")
            f.write(complete_report)
        print(f"Reporte de detección completo guardado en: {complete_report_path}")

        # Generar matriz de confusión con nombres correctos
        if SKLEARN_AVAILABLE and PLOTTING_AVAILABLE and labels_to_report:
            cm = confusion_matrix(all_y_true, all_y_pred, labels=labels_to_report)

            # Calcular tamaño dinámico basado en número de clases
            num_classes = len(labels_to_report)
            fig_size = max(10, num_classes * 0.8)

            # Matriz de confusión con valores absolutos
            plt.figure(figsize=(fig_size, fig_size))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names,
                        yticklabels=target_names,
                        cbar_kws={'label': 'Cantidad'})
            plt.xlabel("Predicción", fontsize=12)
            plt.ylabel("Real (Ground Truth)", fontsize=12)
            plt.title("Matriz de Confusión (Valores Absolutos)\n[Solo detecciones emparejadas por IoU]", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            cm_path = os.path.join(args.base_eval_dir, 'confusion_matrix.png')
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Matriz de confusión guardada en: {cm_path}")

            # Matriz de confusión NORMALIZADA (por fila = por clase GT)
            # Importante para datasets desbalanceados: muestra proporciones en lugar de absolutos
            cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

            plt.figure(figsize=(fig_size, fig_size))
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                        xticklabels=target_names,
                        yticklabels=target_names,
                        cbar_kws={'label': 'Proporción'},
                        vmin=0, vmax=1)
            plt.xlabel("Predicción", fontsize=12)
            plt.ylabel("Real (Ground Truth)", fontsize=12)
            plt.title("Matriz de Confusión Normalizada (por fila)\n[Muestra tasa de clasificación correcta por clase]", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            cm_norm_path = os.path.join(args.base_eval_dir, 'confusion_matrix_normalized.png')
            plt.tight_layout()
            plt.savefig(cm_norm_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Matriz de confusión normalizada guardada en: {cm_norm_path}")
        else:
            print("sklearn o matplotlib no disponibles. Saltando matrices de confusión.")

        # Generar reporte y gráfica de Falsos Positivos y Falsos Negativos
        print("\n--- Generando reporte de Falsos Positivos y Falsos Negativos ---")
        fp_fn_report = "Reporte de Falsos Positivos y Falsos Negativos\n"
        fp_fn_report += "=" * 70 + "\n"
        fp_fn_report += f"{'Categoría':<30} {'Falsos Positivos':>15} {'Falsos Negativos':>15}\n"
        fp_fn_report += "-" * 70 + "\n"
        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            fp = class_fp.get(cat_id, 0)
            fn = class_fn.get(cat_id, 0)
            fp_fn_report += f"{cat_name} ({cat_id}){'':<20} {fp:>15} {fn:>15}\n"

        fp_fn_report_path = os.path.join(args.base_eval_dir, "fp_fn_report.txt")
        with open(fp_fn_report_path, "w") as f:
            f.write(fp_fn_report)
        print(f"Reporte de FP/FN guardado en: {fp_fn_report_path}")

        # Graficar los contadores de FP y FN por clase
        if PLOTTING_AVAILABLE:
            classes = [label_map[k] for k in sorted(label_map.keys())]
            fp_counts = [class_fp.get(k, 0) for k in sorted(label_map.keys())]
            fn_counts = [class_fn.get(k, 0) for k in sorted(label_map.keys())]

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
            if PLOTTING_AVAILABLE:
                # Gráfica de AP por clase
                plt.figure(figsize=(14, 6))
                x = np.arange(len(cat_names_list))
                width = 0.25

                # Colores: rojo para clases minoritarias (detectadas dinámicamente), azul para el resto
                colors_ap = ['#d62728' if name in minority_classes else '#1f77b4' for name in cat_names_list]

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

            # --- AP por Tamaño de Objeto (COCO) ---
            # Importante para entender si el modelo falla con objetos pequeños
            print("\n--- AP por Tamaño de Objeto (COCO) ---")
            # coco_eval.stats contiene: [AP, AP50, AP75, AP_small, AP_medium, AP_large, AR1, AR10, AR100, AR_small, AR_medium, AR_large]
            stats = coco_eval.stats
            size_report = "AP y AR por Tamaño de Objeto\n" + "=" * 60 + "\n"
            size_report += f"{'Métrica':<30} {'Valor':>12}\n"
            size_report += "-" * 60 + "\n"
            size_report += f"{'AP (IoU=0.50:0.95)':<30} {stats[0]:>12.4f}\n"
            size_report += f"{'AP (IoU=0.50)':<30} {stats[1]:>12.4f}\n"
            size_report += f"{'AP (IoU=0.75)':<30} {stats[2]:>12.4f}\n"
            size_report += f"{'AP (small: area < 32²)':<30} {stats[3]:>12.4f}\n"
            size_report += f"{'AP (medium: 32² < area < 96²)':<30} {stats[4]:>12.4f}\n"
            size_report += f"{'AP (large: area > 96²)':<30} {stats[5]:>12.4f}\n"
            size_report += "-" * 60 + "\n"
            size_report += f"{'AR (maxDets=1)':<30} {stats[6]:>12.4f}\n"
            size_report += f"{'AR (maxDets=10)':<30} {stats[7]:>12.4f}\n"
            size_report += f"{'AR (maxDets=100)':<30} {stats[8]:>12.4f}\n"
            size_report += f"{'AR (small: area < 32²)':<30} {stats[9]:>12.4f}\n"
            size_report += f"{'AR (medium: 32² < area < 96²)':<30} {stats[10]:>12.4f}\n"
            size_report += f"{'AR (large: area > 96²)':<30} {stats[11]:>12.4f}\n"

            print(size_report)

            # Guardar reporte de tamaño
            size_report_path = os.path.join(args.base_eval_dir, "ap_by_size_report.txt")
            with open(size_report_path, 'w') as f:
                f.write(size_report)
            print(f"Reporte AP por tamaño guardado en: {size_report_path}")

            if PLOTTING_AVAILABLE:
                # Gráfica de AP/AR por tamaño
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # AP por tamaño
                sizes = ['Small\n(<32²)', 'Medium\n(32²-96²)', 'Large\n(>96²)']
                ap_by_size = [stats[3], stats[4], stats[5]]
                colors_size = ['#e74c3c' if v < 0.3 else '#f39c12' if v < 0.5 else '#27ae60' for v in ap_by_size]
                axes[0].bar(sizes, ap_by_size, color=colors_size, edgecolor='black')
                axes[0].set_ylabel('Average Precision (AP)')
                axes[0].set_title('AP por Tamaño de Objeto')
                axes[0].set_ylim(0, 1)
                axes[0].axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Objetivo: 0.5')
                for i, v in enumerate(ap_by_size):
                    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

                # AR por tamaño
                ar_by_size = [stats[9], stats[10], stats[11]]
                colors_ar = ['#e74c3c' if v < 0.3 else '#f39c12' if v < 0.5 else '#27ae60' for v in ar_by_size]
                axes[1].bar(sizes, ar_by_size, color=colors_ar, edgecolor='black')
                axes[1].set_ylabel('Average Recall (AR)')
                axes[1].set_title('AR por Tamaño de Objeto')
                axes[1].set_ylim(0, 1)
                axes[1].axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Objetivo: 0.5')
                for i, v in enumerate(ar_by_size):
                    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

                plt.tight_layout()
                plt.savefig(os.path.join(args.base_eval_dir, "ap_ar_by_size.png"), dpi=150)
                plt.close()
                print(f"Gráfica AP/AR por tamaño guardada en: {args.base_eval_dir}/ap_ar_by_size.png")

            # Diagnóstico de tamaño
            print("\n  Diagnóstico por Tamaño:")
            if stats[3] < 0.2:
                print("  ALERTA: AP_small muy bajo (<0.2). El modelo tiene problemas con objetos pequeños.")
                print("          Considerar: anchor sizes más pequeños, higher resolution, o augmentation de zoom.")
            if stats[3] < stats[5] * 0.5:
                print(f"  ALERTA: AP_small ({stats[3]:.3f}) << AP_large ({stats[5]:.3f})")
                print("          El modelo favorece objetos grandes. Revisar escala de entrenamiento.")

    # --- RESUMEN DIAGNÓSTICO ---
    print("\n" + "=" * 70)
    print("RESUMEN DIAGNÓSTICO")
    print("=" * 70)

    # Calcular métricas agregadas correctamente
    total_tp = sum(class_tp.values())
    total_fp_sum = sum(class_fp.values())
    total_fn_sum = sum(class_fn.values())
    total_gt_sum = sum(class_gt_total.values())

    # Estadísticas globales
    print(f"\n{'ESTADÍSTICAS GLOBALES':^70}")
    print("-" * 70)
    print(f"  Total Ground Truth:       {total_gt_sum:>8}")
    print(f"  Total Predicciones:       {total_pred:>8}")
    print(f"  True Positives (TP):      {total_tp:>8}")
    print(f"  False Negatives (FN):     {total_fn_sum:>8}  ({100*total_fn_sum/max(total_gt_sum,1):.1f}% de GT)")
    print(f"  False Positives (FP):     {total_fp_sum:>8}  ({100*total_fp_sum/max(total_pred,1):.1f}% de pred)")

    # Métricas globales correctas (micro-average)
    # Micro: suma todos los TP, FP, FN y calcula métricas globales
    micro_precision = total_tp / max(total_tp + total_fp_sum, 1)
    micro_recall = total_tp / max(total_tp + total_fn_sum, 1)
    micro_f1 = 2 * (micro_precision * micro_recall) / max(micro_precision + micro_recall, 1e-10)

    print(f"\n  {'MICRO-AVERAGE (pondera por instancia):':^50}")
    print(f"  Precision (Micro):        {micro_precision:>8.2%}")
    print(f"  Recall (Micro):           {micro_recall:>8.2%}")
    print(f"  F1-Score (Micro):         {micro_f1:>8.4f}")

    # Macro-average: promedio simple de métricas por clase (igual peso a cada clase)
    # Importante para datasets desbalanceados: da igual importancia a clases minoritarias
    class_precisions = []
    class_recalls = []
    class_f1s = []
    class_weights = []  # Para weighted average

    for cat_id in sorted(label_map.keys()):
        tp = class_tp.get(cat_id, 0)
        fn = class_fn.get(cat_id, 0)
        fp = class_fp.get(cat_id, 0)
        gt_total = class_gt_total.get(cat_id, 0)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

        class_precisions.append(precision)
        class_recalls.append(recall)
        class_f1s.append(f1)
        class_weights.append(gt_total)

    # Macro: promedio simple (cada clase cuenta igual)
    macro_precision = np.mean(class_precisions)
    macro_recall = np.mean(class_recalls)
    macro_f1 = np.mean(class_f1s)

    print(f"\n  {'MACRO-AVERAGE (igual peso por clase):':^50}")
    print(f"  Precision (Macro):        {macro_precision:>8.2%}")
    print(f"  Recall (Macro):           {macro_recall:>8.2%}")
    print(f"  F1-Score (Macro):         {macro_f1:>8.4f}")

    # Weighted: ponderado por número de GT de cada clase
    total_weight = sum(class_weights)
    if total_weight > 0:
        weighted_precision = sum(p * w for p, w in zip(class_precisions, class_weights)) / total_weight
        weighted_recall = sum(r * w for r, w in zip(class_recalls, class_weights)) / total_weight
        weighted_f1 = sum(f * w for f, w in zip(class_f1s, class_weights)) / total_weight
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    print(f"\n  {'WEIGHTED-AVERAGE (ponderado por GT):':^50}")
    print(f"  Precision (Weighted):     {weighted_precision:>8.2%}")
    print(f"  Recall (Weighted):        {weighted_recall:>8.2%}")
    print(f"  F1-Score (Weighted):      {weighted_f1:>8.4f}")

    # G-Mean: Geometric Mean de recalls (muy sensible a clases con bajo rendimiento)
    # Mejor métrica para datasets desbalanceados
    if class_recalls and all(r > 0 for r in class_recalls):
        g_mean = np.exp(np.mean(np.log(class_recalls)))
    else:
        # Si algún recall es 0, G-Mean es 0
        g_mean = 0.0

    print(f"\n  {'G-MEAN (sensible a clases minoritarias):':^50}")
    print(f"  G-Mean:                   {g_mean:>8.4f}")
    if g_mean < macro_f1 * 0.8:
        print("  ALERTA: G-Mean << Macro-F1 indica problema severo en alguna clase")

    # === MÉTRICAS DE LOCALIZACIÓN ===
    print(f"\n  {'MÉTRICAS DE LOCALIZACIÓN:':^50}")
    print("-" * 70)

    # Localization Recall: % de GT que tienen match por IoU (independiente de clase)
    localization_recall = total_matched / max(total_gt_sum, 1)
    print(f"  Localization Recall:      {localization_recall:>8.2%}  (GT con match por IoU)")

    # Mean IoU de matches
    if all_match_ious:
        mean_iou = np.mean(all_match_ious)
        std_iou = np.std(all_match_ious)
        min_iou = np.min(all_match_ious)
        max_iou = np.max(all_match_ious)
        print(f"  Mean IoU (matches):       {mean_iou:>8.4f}  (std: {std_iou:.4f})")
        print(f"  IoU Range:                [{min_iou:.3f} - {max_iou:.3f}]")
    else:
        mean_iou = 0.0
        print(f"  Mean IoU (matches):       N/A (sin matches)")

    # === DESGLOSE DE ERRORES ===
    print(f"\n  {'DESGLOSE DE ERRORES:':^50}")
    print("-" * 70)

    # Background FP Rate: predicciones que no matchean con ningún GT
    background_fp_rate = background_fps / max(total_pred, 1)
    print(f"  Background FP:            {background_fps:>8}  ({background_fp_rate:.1%} de predicciones)")
    print("                            -> Detecciones 'fantasma' sin objeto real")

    # Misclassification Rate: matches con IoU pero clase incorrecta
    misclassification_rate = wrong_class_matches / max(total_matched, 1)
    print(f"  Misclassifications:       {wrong_class_matches:>8}  ({misclassification_rate:.1%} de matches)")
    print("                            -> Localiza bien pero clasifica mal")

    # Correct detections
    correct_detection_rate = correct_class_matches / max(total_matched, 1)
    print(f"  Detecciones Correctas:    {correct_class_matches:>8}  ({correct_detection_rate:.1%} de matches)")

    # Análisis de tipo de error predominante
    print(f"\n  {'DIAGNÓSTICO DE ERRORES:':^50}")
    if total_fn_sum > background_fps and total_fn_sum > wrong_class_matches:
        print("  -> Problema principal: FALSOS NEGATIVOS (no detecta objetos)")
        print("     Solución: Reducir detection_threshold o aumentar training")
    elif background_fps > total_fn_sum and background_fps > wrong_class_matches:
        print("  -> Problema principal: DETECCIONES FANTASMA (FP de background)")
        print("     Solución: Aumentar detection_threshold o mejorar NMS")
    elif wrong_class_matches > total_fn_sum and wrong_class_matches > background_fps:
        print("  -> Problema principal: MISCLASSIFICACIÓN (localiza pero no clasifica)")
        print("     Solución: Mejorar features de clasificación o class_weights")
    else:
        print("  -> Errores distribuidos entre FN, FP y misclassificación")

    # Nota explicativa sobre las métricas
    print(f"\n  {'INTERPRETACIÓN MÉTRICAS AGREGADAS:':^50}")
    print("  - Micro: Dominado por clases con más instancias")
    print("  - Macro: Igual peso a cada clase (mejor para desbalanceados)")
    print("  - Weighted: Compromiso entre Micro y Macro")
    print("  - G-Mean: Muy sensible a clases con recall bajo")
    print("  - Para clases minoritarias, enfocarse en Macro-F1 y G-Mean")

    # Análisis por clase (resaltando minoritarias si las hay)
    header_suffix = " - Minoritarias marcadas con >>" if minority_classes else ""
    print(f"\n{'ANÁLISIS POR CLASE (Métricas de Detección)':^90}")
    if minority_classes:
        print(f"{'Clases minoritarias: ' + ', '.join(minority_classes):^90}")
    print("-" * 90)
    print(f"{'Clase':<20} {'GT':>6} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Estado':>12}")
    print("-" * 90)

    # Métricas calculadas correctamente:
    # - TP: Detecciones correctas (matched con label correcto)
    # - FN: GT sin detectar + GT emparejado con predicción de clase diferente
    # - FP: Predicciones sin emparejar + predicciones emparejadas con GT de clase diferente
    # - Recall = TP / (TP + FN)
    # - Precision = TP / (TP + FP)
    # - F1 = 2 * (Precision * Recall) / (Precision + Recall)

    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]

        # Contadores ya calculados correctamente durante la evaluación
        tp = class_tp.get(cat_id, 0)
        fn = class_fn.get(cat_id, 0)
        fp = class_fp.get(cat_id, 0)
        gt_total = class_gt_total.get(cat_id, 0)

        # Métricas
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)

        # Determinar estado basado en F1-score
        if f1 < 0.3:
            estado = "CRITICO"
        elif f1 < 0.5:
            estado = "BAJO"
        elif f1 < 0.7:
            estado = "MODERADO"
        else:
            estado = "BUENO"

        # Resaltar clases minoritarias (solo si hay alguna detectada)
        prefix = ">> " if cat_name in minority_classes else "   "
        print(f"{prefix}{cat_name:<17} {gt_total:>6} {tp:>6} {fn:>6} {fp:>6} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f} {estado:>12}")

    print("-" * 70)

    # Recomendaciones basadas en el análisis
    print(f"\n{'RECOMENDACIONES':^70}")
    print("-" * 70)

    # Identificar clases problemáticas
    problematic_classes = []
    for cat_id in sorted(label_map.keys()):
        cat_name = label_map[cat_id]
        tp = class_tp.get(cat_id, 0)
        fn = class_fn.get(cat_id, 0)
        gt_total = class_gt_total.get(cat_id, 0)
        recall = tp / max(tp + fn, 1)
        if recall < 0.5 and gt_total > 10:  # Solo si hay suficientes muestras
            problematic_classes.append((cat_name, recall, gt_total))

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

    # Análisis de distribución de scores para clases minoritarias (solo si las hay)
    if minority_classes:
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
        f.write(f"Total Ground Truth:       {total_gt_sum}\n")
        f.write(f"Total Predicciones:       {total_pred}\n")
        f.write(f"True Positives (TP):      {total_tp}\n")
        f.write(f"False Negatives (FN):     {total_fn_sum} ({100*total_fn_sum/max(total_gt_sum,1):.1f}% de GT)\n")
        f.write(f"False Positives (FP):     {total_fp_sum} ({100*total_fp_sum/max(total_pred,1):.1f}% de pred)\n\n")

        f.write("MÉTRICAS AGREGADAS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Micro-Precision:          {micro_precision:.2%}\n")
        f.write(f"Micro-Recall:             {micro_recall:.2%}\n")
        f.write(f"Micro-F1:                 {micro_f1:.4f}\n")
        f.write(f"Macro-Precision:          {macro_precision:.2%}\n")
        f.write(f"Macro-Recall:             {macro_recall:.2%}\n")
        f.write(f"Macro-F1:                 {macro_f1:.4f}\n")
        f.write(f"Weighted-F1:              {weighted_f1:.4f}\n")
        f.write(f"G-Mean:                   {g_mean:.4f}\n\n")

        f.write("MÉTRICAS DE LOCALIZACIÓN\n")
        f.write("-" * 70 + "\n")
        f.write(f"Localization Recall:      {localization_recall:.2%}\n")
        f.write(f"Mean IoU (matches):       {mean_iou:.4f}\n\n")

        f.write("DESGLOSE DE ERRORES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Background FP:            {background_fps} ({background_fp_rate:.1%} de pred)\n")
        f.write(f"Misclassifications:       {wrong_class_matches} ({misclassification_rate:.1%} de matches)\n")
        f.write(f"Detecciones Correctas:    {correct_class_matches} ({correct_detection_rate:.1%} de matches)\n\n")

        f.write("POR CLASE\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Clase':<20} {'GT':>6} {'TP':>6} {'FN':>6} {'FP':>6} {'Recall':>8} {'Precision':>10} {'F1':>8}\n")
        for cat_id in sorted(label_map.keys()):
            cat_name = label_map[cat_id]
            tp = class_tp.get(cat_id, 0)
            fn = class_fn.get(cat_id, 0)
            fp = class_fp.get(cat_id, 0)
            gt_total = class_gt_total.get(cat_id, 0)
            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
            f.write(f"{cat_name:<20} {gt_total:>6} {tp:>6} {fn:>6} {fp:>6} {recall:>8.1%} {precision:>10.1%} {f1:>8.3f}\n")

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
    parser.add_argument('--coco_eval', action='store_true', default=True, help="Activar evaluación de mAP con pycocotools.")
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