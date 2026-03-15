# Experimentos: Análisis de Datos Sintéticos vs Reales

Sistema automatizado para evaluar el impacto de diferentes proporciones de datos sintéticos y reales en el entrenamiento de RT-DETRv2 para detección de desechos marinos.

---

## Objetivo

Responder las siguientes preguntas mediante experimentación sistemática:

1. **¿El dato sintético puede mitigar el desbalance de clases?** (Can y Metal_Debris tienen <2% de muestras)
2. **¿Cuál es el rendimiento de un modelo entrenado 100% con datos sintéticos sobre datos reales?**
3. **¿Cuál es la proporción óptima de datos sintéticos/reales?**

---

## 9 Experimentos Configurados

| # | Nombre | Sintético | Real | Estrategia | Objetivo |
|---|--------|-----------|------|------------|----------|
| 1 | `exp01_synth100` | 100% | 0% | - | Baseline sintético puro |
| 2 | `exp02_synth100_real10` | 100% | 10% | Stratified | Mínimo aporte real |
| 3 | `exp03_synth100_real25` | 100% | 25% | Stratified | 1/4 de datos reales |
| 4 | `exp04_synth100_real50` | 100% | 50% | Stratified | Mitad y mitad |
| 5 | `exp05_synth100_real75` | 100% | 75% | Stratified | 3/4 de datos reales |
| 6 | `exp06_synth100_real100` | 100% | 100% | - | Todo combinado |
| 7 | `exp07_real100` | 0% | 100% | - | **Baseline real (referencia)** |
| 8 | `exp08_real_synth_minor` | Solo minoritarias | 100% | Minoritarias | Balancear Can + Metal_Debris |
| 9 | `exp09_real_synth_prop` | Proporcional | 100% | Inverso | Más sintético donde menos real |

### Detalle de Experimentos Especiales

**Experimento 8** - Sintético solo para clases minoritarias:
- Usa 100% de datos reales
- Añade datos sintéticos SOLO de Can y Metal_Debris
- Objetivo: Balancear las clases más escasas sin contaminar las demás

**Experimento 9** - Sintético proporcional inverso:
- Usa 100% de datos reales
- Añade más datos sintéticos para clases con menos datos reales
- Distribución: Metal_Debris (+4500) > Can (+4200) > Glove (+2000) > ... > Bottle (+0)

---

## Uso Rápido

### Ejecutar Pipeline Completo

```bash
# Dentro del contenedor Docker
cd /app/RT-DETR

# Ejecutar todo automáticamente
python experiments/run_full_pipeline.py
```

### Ejecución por Fases

```bash
# Fase 1: Preparar datasets
python experiments/prepare_datasets.py

# Fase 2: Generar configuraciones YAML
python experiments/generate_configs.py

# Fase 3: Entrenar y evaluar
python experiments/run_experiments.py

# Fase 4: Recolectar resultados
python experiments/collect_results.py
```

---

## Opciones del Pipeline

### Ejecutar experimentos específicos

```bash
# Solo experimentos 1, 7, 8 y 9 (los más importantes)
python experiments/run_full_pipeline.py -e 1 7 8 9
```

### Retomar ejecución interrumpida

```bash
# Continuar desde donde se quedó
python experiments/run_experiments.py --resume

# O desde una fase específica
python experiments/run_full_pipeline.py --start-phase 3
```

### Saltar fases ya completadas

```bash
# Si los datasets ya están preparados
python experiments/run_full_pipeline.py --skip-prepare

# Si los configs ya están generados
python experiments/run_full_pipeline.py --skip-prepare --skip-configs
```

### Modo prueba (dry-run)

```bash
# Ver qué se ejecutaría sin hacer nada
python experiments/run_full_pipeline.py --dry-run
```

### Ver estado actual

```bash
# Ver qué experimentos están completados
python experiments/run_experiments.py --status
```

---

## Estructura de Salidas

```
experiment_output/
├── datasets/                          # Datasets generados
│   ├── exp01_synth100/
│   │   └── train.json                 # Anotaciones COCO
│   ├── exp02_synth100_real10/
│   │   └── train.json
│   └── ...
│
├── configs/                           # Configuraciones YAML
│   ├── exp01_synth100.yml
│   ├── exp02_synth100_real10.yml
│   └── ...
│
├── training/                          # Outputs de entrenamiento
│   ├── exp01_synth100/
│   │   ├── best.pth                   # Mejor checkpoint
│   │   ├── last.pth                   # Último checkpoint
│   │   ├── train.log                  # Log de entrenamiento
│   │   └── events.out.tfevents.*      # TensorBoard
│   └── ...
│
├── evaluation/                        # Resultados de evaluación
│   ├── exp01_synth100/
│   │   ├── coco_evaluation_report.txt
│   │   ├── per_class_ap_report.txt
│   │   └── predictions.json
│   └── ...
│
└── results/                           # Resultados agregados
    ├── comparison_table.csv           # Tabla comparativa
    ├── mAP_vs_real_ratio.png          # Gráfico de progresión
    ├── minority_class_analysis.png    # Análisis Can/Metal
    ├── per_class_comparison.png       # Comparación por clase
    └── EXPERIMENT_REPORT.md           # Reporte final
```

---

## Configuración

### Actualizar rutas de datasets

Editar `experiments/config.py`:

```python
# Dataset sintético (actualizar cuando esté listo)
SYNTHETIC_TRAIN_JSON = "/app/RT-DETR/split_sintetico_densea/train.json"
SYNTHETIC_IMAGES_DIR = "./dataset/split_sintetico_densea/images/"
```

### Modificar hiperparámetros de entrenamiento

Los hiperparámetros están en `experiments/config.py`:

```python
MAX_EPOCHS = 200              # Épocas máximas
EARLY_STOPPING_PATIENCE = 20  # Paciencia para early stopping
BATCH_SIZE = 32               # Tamaño de batch
SEED = 21                     # Semilla para reproducibilidad
```

### Personalizar template de configuración

Editar `experiments/config_template.yml` para modificar:
- Transforms de data augmentation
- Arquitectura del modelo
- Scheduler de learning rate

---

## Requisitos

### Datasets necesarios

1. **Datos reales**: `split_v8/train.json` (ya existente)
2. **Datos sintéticos**: `split_sintetico_densea/train.json` (en generación)
3. **Test set**: `split_v8/test.json` (ya existente)

### Dependencias Python

```
torch >= 2.0
torchvision
pycocotools
pyyaml
pandas (opcional, para visualizaciones)
matplotlib (opcional, para gráficos)
```

---

## Tiempo Estimado

| Fase | Duración estimada |
|------|-------------------|
| Preparar datasets | ~5 minutos |
| Generar configs | ~1 minuto |
| Entrenar 1 experimento | 4-8 horas (GPU RTX) |
| **Total 9 experimentos** | **36-72 horas** |
| Recolectar resultados | ~2 minutos |

> **Nota**: El early stopping puede reducir significativamente el tiempo si el modelo converge antes de las 200 épocas.

---

## Resultados Esperados

### Hipótesis a validar

| Hipótesis | Métrica clave |
|-----------|---------------|
| Sintético puro tiene domain gap | mAP exp01 << mAP exp07 |
| Añadir real mejora progresivamente | mAP exp02 < exp03 < ... < exp07 |
| Exp 8/9 mejoran clases minoritarias | AP_Can y AP_Metal más altos |
| Existe ratio óptimo | Punto de inflexión en curva mAP vs % real |

### Tabla de referencia (valores esperados)

| Experimento | mAP esperado | AP_Can | AP_Metal |
|-------------|--------------|--------|----------|
| exp01 (100% sint) | 0.01-0.10 | ~0.05 | ~0.03 |
| exp07 (100% real) | **0.50-0.55** | ~0.15 | ~0.10 |
| exp08 (real+minor) | 0.52-0.58 | **~0.30** | **~0.28** |

---

## Troubleshooting

### Error: "Synthetic dataset not found"

El dataset sintético no existe todavía. Actualiza la ruta en `config.py` cuando esté listo.

### Error: OOM (Out of Memory)

Reduce el batch size en `config.py`:
```python
BATCH_SIZE = 16  # Reducir de 32 a 16
```

### Experimento falla pero pipeline continúa

El sistema está diseñado para continuar con el siguiente experimento si uno falla. Revisa los logs en `experiment_output/training/expXX/train.log`.

### Retomar después de interrupción

```bash
python experiments/run_experiments.py --resume
```

---

## Archivos del Módulo

| Archivo | Descripción |
|---------|-------------|
| `config.py` | Configuración centralizada (rutas, experimentos, hiperparámetros) |
| `prepare_datasets.py` | Genera datasets balanceados para cada experimento |
| `generate_configs.py` | Genera archivos YAML de configuración |
| `run_experiments.py` | Ejecuta entrenamiento y evaluación |
| `collect_results.py` | Agrega métricas y genera visualizaciones |
| `run_full_pipeline.py` | **Orquestador principal** (ejecuta todo) |
| `config_template.yml` | Template base para configuraciones YAML |

---

## Contacto

Para dudas o problemas, revisar los logs en `experiment_output/`.
