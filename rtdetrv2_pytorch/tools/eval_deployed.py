# tools/eval_deployed.py

import os
import sys
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

# Añadir el directorio raíz del proyecto al path para poder importar 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import YAMLConfig
from src.solver import TASKS

# --- Clases de Motores de Inferencia ---

class ONNXEngine:
    """Wrapper para la inferencia con ONNX Runtime."""
    def __init__(self, model_path, device):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime no está instalado. Instálalo con: pip install onnxruntime-gpu")

        print(f"Cargando modelo ONNX desde: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"Modelo ONNX cargado. Usando proveedor: {self.session.get_providers()[0]}")

    def __call__(self, images_tensor):
        """La entrada es solo el tensor de imágenes, como en el modelo original."""
        inputs = {self.input_names[0]: images_tensor.cpu().numpy()}
        # El modelo ONNX exportado ya tiene el post-procesador integrado,
        # por lo que no necesita el `orig_target_sizes`.
        outputs = self.session.run(self.output_names, inputs)
        # La salida ya está post-procesada, devolvemos un diccionario simple
        # que el bucle de evaluación puede usar.
        return {
            'labels': torch.from_numpy(outputs[0]),
            'boxes': torch.from_numpy(outputs[1]),
            'scores': torch.from_numpy(outputs[2])
        }

class TensorRTEngine:
    """Wrapper para la inferencia con un motor de TensorRT."""
    def __init__(self, engine_path, device):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT o PyCUDA no están instalados.")

        print(f"Cargando motor de TensorRT desde: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError("No se pudo deserializar el motor de TensorRT.")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        self.device = device
        print("Motor de TensorRT cargado exitosamente.")

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def __call__(self, images_tensor):
        """El modelo exportado con post-procesador solo necesita el tensor de imágenes."""
        # Se asume que el batch size es 1
        np.copyto(self.inputs[0]['host'], images_tensor.cpu().numpy().ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # El segundo input `orig_target_sizes` también debe ser alimentado.
        # Lo creamos aquí ya que no es parte de la entrada del modelo PyTorch.
        h, w = images_tensor.shape[2:]
        orig_target_sizes = torch.tensor([[h, w]], dtype=torch.int64)
        np.copyto(self.inputs[1]['host'], orig_target_sizes.numpy().ravel())
        cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()

        num_dets = 300 # Número de detecciones de salida de RT-DETR
        labels = torch.from_numpy(self.outputs[0]['host']).reshape(1, num_dets)
        boxes = torch.from_numpy(self.outputs[1]['host']).reshape(1, num_dets, 4)
        scores = torch.from_numpy(self.outputs[2]['host']).reshape(1, num_dets)
        
        return {'labels': labels.to(self.device), 'boxes': boxes.to(self.device), 'scores': scores.to(self.device)}

# --- Bucle de Evaluación Principal ---

def main(args):
    cfg = YAMLConfig(args.config)
    
    # --- 1. Construir el Dataloader y el Evaluador usando la lógica original ---
    # Esto asegura que los datos y las métricas sean 100% consistentes.
    val_loader = TASKS.build(cfg.data['val'])
    coco_evaluator = val_loader.dataset.get_evaluator()
    
    # --- 2. Cargar el Modelo Desplegado ---
    if args.onnx:
        engine = ONNXEngine(args.onnx, args.device)
    elif args.tensorrt:
        engine = TensorRTEngine(args.tensorrt, args.device)
    else:
        raise ValueError("Debes especificar un modelo con --onnx o --tensorrt.")

    # --- 3. Bucle de Evaluación ---
    print("\nIniciando evaluación en el conjunto de test...")
    latencies = []
    device = torch.device(args.device)

    for i, (samples, targets) in enumerate(tqdm(val_loader, desc="Evaluando")):
        samples = samples.to(device)
        
        start_time = time.perf_counter()
        outputs = engine(samples)
        end_time = time.perf_counter()
        
        latencies.append(end_time - start_time)

        # Formatear resultados para el evaluador de COCO.
        # El post-procesador ya no es necesario aquí porque está integrado
        # en el grafo ONNX/TensorRT exportado.
        res = {target['image_id'].item(): output for target, output in zip(targets, [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(outputs['scores'], outputs['labels'], outputs['boxes'])])}
        
        if coco_evaluator:
            coco_evaluator.update(res)

    # --- 4. Mostrar Resultados ---
    if latencies:
        print("\n--- Resultados del Benchmark de Rendimiento ---")
        avg_latency_ms = np.mean(latencies) * 1000
        avg_fps = 1.0 / np.mean(latencies)
        print(f"Latencia Promedio: {avg_latency_ms:.2f} ms")
        print(f"Throughput (FPS): {avg_fps:.2f} FPS")

    if coco_evaluator:
        print("\n--- Resultados de Precisión (Métricas COCO) ---")
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluación de modelos ONNX y TensorRT.")
    parser.add_argument('-c', '--config', required=True, type=str, help="Ruta al archivo de configuración YAML.")
    parser.add_argument('--onnx', type=str, help="Ruta al archivo del modelo ONNX.")
    parser.add_argument('--tensorrt', type=str, help="Ruta al archivo del motor de TensorRT (.engine).")
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help="Dispositivo a usar (ej. 'cuda:0').")
    
    args = parser.parse_args()

    if not (args.onnx or args.tensorrt):
        parser.error("Se requiere al menos uno: --onnx o --tensorrt")
    if args.onnx and args.tensorrt:
        parser.error("Proporciona solo un tipo de modelo a la vez: --onnx o --tensorrt")

    main(args)