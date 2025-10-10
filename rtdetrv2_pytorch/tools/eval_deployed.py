# tools/eval_deployed.py

import os
import sys
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

# Añadir el directorio raíz al path para importar módulos de src
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from src.core import YAMLConfig
    from src.data.dataloader import create_dataloader
    from src.data.dataset.coco_eval import CocoEvaluator
    from src.data.transforms import Compose
except ImportError as e:
    print(f"Error: No se pudieron importar los módulos necesarios. Asegúrate de que tu entorno esté configurado correctamente. {e}")
    sys.exit(1)

# === Clases de Inferencia ===
# Se definen clases separadas para cada motor para mantener el código limpio.

class ONNXEvaluator:
    """Clase para evaluar un modelo ONNX usando ONNX Runtime."""
    def __init__(self, model_path, device):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime no está instalado. Por favor, instálalo con: pip install onnxruntime-gpu")

        print(f"Cargando modelo ONNX desde: {model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in device else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        print(f"Modelo ONNX cargado. Usando proveedor: {self.session.get_providers()[0]}")

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, images_tensor, orig_target_sizes_tensor):
        """Ejecuta la inferencia en un lote de imágenes."""
        inputs = {
            self.input_names[0]: images_tensor.cpu().numpy(),
            self.input_names[1]: orig_target_sizes_tensor.cpu().numpy()
        }
        outputs = self.session.run(self.output_names, inputs)
        
        # Convertir las salidas de numpy a tensores de PyTorch para compatibilidad
        # con el resto del pipeline.
        return {
            'labels': torch.from_numpy(outputs[0]),
            'boxes': torch.from_numpy(outputs[1]),
            'scores': torch.from_numpy(outputs[2])
        }

class TensorRTEvaluator:
    """Clase para evaluar un motor de TensorRT."""
    def __init__(self, engine_path, device):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT o PyCUDA no están instalados. Asegúrate de que estén configurados en tu sistema.")

        print(f"Cargando motor de TensorRT desde: {engine_path}")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if not self.engine:
            raise RuntimeError("No se pudo deserializar el motor de TensorRT.")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        print("Motor de TensorRT cargado exitosamente.")

    def _allocate_buffers(self):
        """Prepara los buffers de memoria en la GPU para la inferencia."""
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        return inputs, outputs, bindings, stream

    def __call__(self, images_tensor, orig_target_sizes_tensor):
        """Ejecuta la inferencia en un lote de imágenes."""
        # Copiar datos de entrada al buffer de la GPU
        np.copyto(self.inputs[0]['host'], images_tensor.cpu().numpy().ravel())
        np.copyto(self.inputs[1]['host'], orig_target_sizes_tensor.cpu().numpy().ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # Ejecutar la inferencia
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copiar resultados de vuelta a la CPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        # Reformatear la salida para que coincida con la estructura esperada
        # (Esto asume el orden de salida: labels, boxes, scores)
        # Las formas exactas deben ajustarse según la salida de tu modelo.
        # Por ejemplo, si el batch size es 1:
        num_dets = 300 # O el número de detecciones que produce tu modelo
        
        labels = torch.from_numpy(self.outputs[0]['host']).reshape(1, num_dets)
        boxes = torch.from_numpy(self.outputs[1]['host']).reshape(1, num_dets, 4)
        scores = torch.from_numpy(self.outputs[2]['host']).reshape(1, num_dets)

        return {'labels': labels, 'boxes': boxes, 'scores': scores}


# === Bucle de Evaluación Principal ===

def main(args):
    """
    Función principal que carga el modelo y ejecuta el bucle de evaluación.
    """
    cfg = YAMLConfig(args.config)
    
    # --- 1. Cargar el Dataloader ---
    # Asegúrate de que el dataloader NO use `dist.get_rank()` ni `dist.get_world_size()`
    # ya que esta evaluación se ejecuta en un solo proceso.
    val_loader_cfg = cfg.data['val']['loader']
    val_loader_cfg['shuffle'] = False
    val_loader_cfg['drop_last'] = False
    val_dataset_cfg = cfg.data['val']['dataset']
    
    # Usar Compose directamente en lugar de Presets para evitar lógicas de entrenamiento
    transform = Compose(cfg.data['val']['transforms'])
    val_dataset_cfg['transform'] = transform
    
    val_loader = create_dataloader(val_dataset_cfg, val_loader_cfg, 0, 1)[0]
    
    # --- 2. Inicializar el Evaluador de COCO ---
    # Esto es idéntico al `eval.py` original para asegurar métricas comparables.
    coco_gt = val_loader.dataset.coco
    coco_evaluator = CocoEvaluator(coco_gt, ('bbox', ))

    # --- 3. Cargar el Modelo Desplegado ---
    if args.onnx:
        model_path = args.onnx
        evaluator = ONNXEvaluator(model_path, args.device)
    elif args.tensorrt:
        model_path = args.tensorrt
        evaluator = TensorRTEvaluator(model_path, args.device)
    else:
        raise ValueError("Debes especificar un modelo con --onnx o --tensorrt.")

    # --- 4. Bucle de Evaluación ---
    print("\nIniciando evaluación en el conjunto de test...")
    latencies = []
    
    for samples, targets in tqdm(val_loader, desc="Evaluando"):
        # Mover los datos al dispositivo es gestionado por los evaluadores si es necesario.
        # Aquí los mantenemos en la CPU para pasarlos.
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = evaluator(samples, orig_target_sizes)
        
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

        # Formatear los resultados para el evaluador de COCO
        res = {target['image_id'].item(): output for target, output in zip(targets, [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(outputs['scores'], outputs['labels'], outputs['boxes'])])}
        
        coco_evaluator.update(res)

    # --- 5. Mostrar Resultados ---
    print("\n--- Resultados del Benchmark de Rendimiento ---")
    avg_latency_ms = np.mean(latencies) * 1000
    avg_fps = 1.0 / np.mean(latencies)
    print(f"Latencia Promedio: {avg_latency_ms:.2f} ms")
    print(f"Throughput (FPS): {avg_fps:.2f} FPS")

    print("\n--- Resultados de Precisión (Métricas COCO) ---")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluación de modelos ONNX y TensorRT en el conjunto de datos COCO.")
    parser.add_argument('-c', '--config', required=True, type=str, help="Ruta al archivo de configuración YAML (el mismo usado para entrenar/evaluar).")
    parser.add_argument('--onnx', type=str, help="Ruta al archivo del modelo ONNX.")
    parser.add_argument('--tensorrt', type=str, help="Ruta al archivo del motor de TensorRT (.engine).")
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help="Dispositivo a usar (ej. 'cuda:0' o 'cpu').")
    
    args = parser.parse_args()

    if not (args.onnx or args.tensorrt):
        parser.error("Se requiere al menos uno de los argumentos: --onnx o --tensorrt")
    if args.onnx and args.tensorrt:
        parser.error("Por favor, proporciona solo un tipo de modelo a la vez: --onnx o --tensorrt")

    main(args)