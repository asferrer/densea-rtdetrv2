"""
Herramienta para analizar el domain gap entre datasets sintéticos y reales.

Analiza:
- Distribución de colores (RGB, HSV)
- Brillo y contraste
- Estadísticas por canal
- Frecuencias espaciales (nitidez)
- Histogramas y momentos estadísticos

Uso:
    python tools/analyze_domain_gap.py \
        --synthetic-dir ./dataset/dataset_sintetico/images \
        --real-dir ./dataset/cleansea_dataset/CocoFormatDataset/train_coco \
        --output-dir ./domain_gap_analysis \
        --num-samples 500

NOTA: Este script usa cálculo incremental de estadísticas para ser eficiente
en memoria, procesando las imágenes una a una sin cargarlas todas en RAM.
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class OnlineStatistics:
    """
    Calcula media y desviación estándar de forma incremental (Welford's algorithm).
    Esto permite procesar millones de valores sin almacenarlos todos en memoria.
    """
    def __init__(self, n_channels: int = 3):
        self.n = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.M2 = np.zeros(n_channels, dtype=np.float64)
        self.min_val = np.full(n_channels, np.inf, dtype=np.float64)
        self.max_val = np.full(n_channels, -np.inf, dtype=np.float64)
        # Para percentiles, usamos reservoirs sampling o histogramas
        self.histogram_bins = 256
        self.histograms = np.zeros((n_channels, self.histogram_bins), dtype=np.int64)

    def update(self, values: np.ndarray):
        """
        Actualiza estadísticas con nuevos valores.
        values: array de shape (N, n_channels) con valores en [0, 255]
        """
        for val in values:
            self.n += 1
            delta = val - self.mean
            self.mean += delta / self.n
            delta2 = val - self.mean
            self.M2 += delta * delta2

            # Min/Max
            self.min_val = np.minimum(self.min_val, val)
            self.max_val = np.maximum(self.max_val, val)

    def update_batch(self, values: np.ndarray):
        """
        Actualiza estadísticas con un batch de valores (más eficiente).
        values: array de shape (N, n_channels) con valores en [0, 255]
        """
        if len(values) == 0:
            return

        # Calcular estadísticas del batch
        batch_n = len(values)
        batch_mean = values.mean(axis=0)
        batch_var = values.var(axis=0)

        # Combinar con estadísticas existentes (parallel algorithm)
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_var * batch_n
        else:
            new_n = self.n + batch_n
            delta = batch_mean - self.mean
            self.mean = (self.n * self.mean + batch_n * batch_mean) / new_n
            self.M2 += batch_var * batch_n + delta ** 2 * self.n * batch_n / new_n
            self.n = new_n

        # Min/Max
        self.min_val = np.minimum(self.min_val, values.min(axis=0))
        self.max_val = np.maximum(self.max_val, values.max(axis=0))

        # Actualizar histogramas para percentiles
        for ch in range(values.shape[1]):
            hist, _ = np.histogram(values[:, ch], bins=self.histogram_bins, range=(0, 256))
            self.histograms[ch] += hist

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.zeros_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1))

    def get_percentile(self, p: float) -> np.ndarray:
        """Calcula percentil aproximado usando histogramas."""
        result = np.zeros(len(self.histograms))
        for ch in range(len(self.histograms)):
            cumsum = np.cumsum(self.histograms[ch])
            target = p / 100.0 * cumsum[-1]
            idx = np.searchsorted(cumsum, target)
            result[ch] = idx
        return result


class DomainGapAnalyzer:
    """Analiza diferencias estadísticas entre datasets sintéticos y reales."""

    def __init__(self, synthetic_dir: str, real_dir: str, num_samples: int = 500):
        self.synthetic_dir = Path(synthetic_dir)
        self.real_dir = Path(real_dir)
        self.num_samples = num_samples

        self.synthetic_stats = {}
        self.real_stats = {}
        self.gap_report = {}

    def get_image_files(self, directory: Path, max_images: int) -> List[Path]:
        """Obtiene lista de archivos de imagen sin cargarlos."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in directory.rglob('*') if f.suffix.lower() in extensions]

        # Muestreo aleatorio si hay demasiadas imágenes
        if len(image_files) > max_images:
            np.random.seed(42)
            indices = np.random.choice(len(image_files), max_images, replace=False)
            image_files = [image_files[i] for i in indices]

        return image_files

    def analyze_images_streaming(self, image_files: List[Path], dataset_name: str) -> Dict:
        """
        Análisis completo de un dataset usando streaming (memoria eficiente).
        Procesa una imagen a la vez sin cargar todas en RAM.
        """
        print(f"\n{'='*60}")
        print(f"Analizando dataset: {dataset_name}")
        print(f"{'='*60}")

        # Estadísticas online
        rgb_stats = OnlineStatistics(n_channels=3)
        hsv_stats = OnlineStatistics(n_channels=3)

        brightness_values = []
        contrast_values = []
        sharpness_values = []

        # Procesar imagen por imagen
        print(f"Procesando {len(image_files)} imágenes...")
        for img_path in tqdm(image_files, desc=f"Analizando {dataset_name}"):
            try:
                # Cargar imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Submuestreo de píxeles para estadísticas de color
                # Tomar solo 10,000 píxeles aleatorios por imagen para ahorrar memoria
                h, w = img_rgb.shape[:2]
                total_pixels = h * w
                sample_size = min(10000, total_pixels)

                if total_pixels > sample_size:
                    # Muestreo aleatorio de píxeles
                    flat_img = img_rgb.reshape(-1, 3)
                    indices = np.random.choice(total_pixels, sample_size, replace=False)
                    sampled_pixels = flat_img[indices].astype(np.float64)
                else:
                    sampled_pixels = img_rgb.reshape(-1, 3).astype(np.float64)

                # Actualizar estadísticas RGB
                rgb_stats.update_batch(sampled_pixels)

                # Convertir a HSV y actualizar estadísticas
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                if total_pixels > sample_size:
                    flat_hsv = img_hsv.reshape(-1, 3)
                    sampled_hsv = flat_hsv[indices].astype(np.float64)
                else:
                    sampled_hsv = img_hsv.reshape(-1, 3).astype(np.float64)

                hsv_stats.update_batch(sampled_hsv)

                # Brillo y contraste (sobre imagen completa, pero solo guardamos scalar)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness_values.append(float(gray.mean()))
                contrast_values.append(float(gray.std()))

                # Nitidez (solo para primeras 100 imágenes para velocidad)
                if len(sharpness_values) < 100:
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    sharpness_values.append(float(laplacian.var()))

                # Liberar memoria explícitamente
                del img, img_rgb, img_hsv, gray, sampled_pixels
                if 'sampled_hsv' in dir():
                    del sampled_hsv

            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue

        # Forzar garbage collection
        gc.collect()

        # Compilar estadísticas
        stats = {
            'rgb': {
                'mean_rgb': rgb_stats.mean.tolist(),
                'std_rgb': rgb_stats.std.tolist(),
                'median_rgb': rgb_stats.get_percentile(50).tolist(),
                'min_rgb': rgb_stats.min_val.tolist(),
                'max_rgb': rgb_stats.max_val.tolist(),
                'percentile_25_rgb': rgb_stats.get_percentile(25).tolist(),
                'percentile_75_rgb': rgb_stats.get_percentile(75).tolist(),
            },
            'hsv': {
                'mean_hsv': hsv_stats.mean.tolist(),
                'std_hsv': hsv_stats.std.tolist(),
                'median_hsv': hsv_stats.get_percentile(50).tolist(),
            },
            'brightness_contrast': {
                'mean_brightness': float(np.mean(brightness_values)) if brightness_values else 0,
                'std_brightness': float(np.std(brightness_values)) if brightness_values else 0,
                'mean_contrast': float(np.mean(contrast_values)) if contrast_values else 0,
                'std_contrast': float(np.std(contrast_values)) if contrast_values else 0,
            },
            'sharpness': {
                'mean_sharpness': float(np.mean(sharpness_values)) if sharpness_values else 0,
                'std_sharpness': float(np.std(sharpness_values)) if sharpness_values else 0,
            },
            'num_images_processed': rgb_stats.n // 10000 if rgb_stats.n > 0 else len(image_files),  # Aproximado
            'num_pixels_sampled': int(rgb_stats.n),
        }

        print(f"  Píxeles analizados: {rgb_stats.n:,}")
        print(f"  Media RGB: R={stats['rgb']['mean_rgb'][0]:.1f}, G={stats['rgb']['mean_rgb'][1]:.1f}, B={stats['rgb']['mean_rgb'][2]:.1f}")
        print(f"  Brillo promedio: {stats['brightness_contrast']['mean_brightness']:.1f}")

        return stats

    def compute_gap(self) -> Dict:
        """Calcula las diferencias entre sintético y real."""
        gap = {}

        # Gap RGB
        synth_rgb = np.array(self.synthetic_stats['rgb']['mean_rgb'])
        real_rgb = np.array(self.real_stats['rgb']['mean_rgb'])
        gap['rgb_mean_diff'] = (synth_rgb - real_rgb).tolist()
        gap['rgb_mean_diff_magnitude'] = float(np.linalg.norm(synth_rgb - real_rgb))

        # Gap brillo
        synth_bright = self.synthetic_stats['brightness_contrast']['mean_brightness']
        real_bright = self.real_stats['brightness_contrast']['mean_brightness']
        gap['brightness_diff'] = synth_bright - real_bright
        gap['brightness_ratio'] = synth_bright / real_bright if real_bright > 0 else 0

        # Gap contraste
        synth_contrast = self.synthetic_stats['brightness_contrast']['mean_contrast']
        real_contrast = self.real_stats['brightness_contrast']['mean_contrast']
        gap['contrast_diff'] = synth_contrast - real_contrast
        gap['contrast_ratio'] = synth_contrast / real_contrast if real_contrast > 0 else 0

        # Gap nitidez
        synth_sharp = self.synthetic_stats['sharpness']['mean_sharpness']
        real_sharp = self.real_stats['sharpness']['mean_sharpness']
        gap['sharpness_diff'] = synth_sharp - real_sharp
        gap['sharpness_ratio'] = synth_sharp / real_sharp if real_sharp > 0 else 0

        # Score general de gap (0-1, donde 0 = sin gap)
        rgb_gap_normalized = gap['rgb_mean_diff_magnitude'] / 255.0  # Normalizar a [0, 1]
        brightness_gap_normalized = abs(gap['brightness_diff']) / 255.0
        contrast_gap_normalized = abs(gap['contrast_diff']) / 128.0

        gap['overall_gap_score'] = float(
            0.5 * rgb_gap_normalized +
            0.3 * brightness_gap_normalized +
            0.2 * contrast_gap_normalized
        )

        return gap

    def generate_correction_parameters(self) -> Dict:
        """Genera parámetros de corrección para cerrar el gap."""
        gap = self.gap_report

        # Calcular factores de corrección
        synth_bright = self.synthetic_stats['brightness_contrast']['mean_brightness']
        real_bright = self.real_stats['brightness_contrast']['mean_brightness']
        brightness_factor = real_bright / synth_bright if synth_bright > 0 else 1.0

        synth_rgb = np.array(self.synthetic_stats['rgb']['mean_rgb'])
        real_rgb = np.array(self.real_stats['rgb']['mean_rgb'])

        # Factores por canal (R, G, B)
        rgb_factors = real_rgb / (synth_rgb + 1e-6)

        corrections = {
            'brightness_multiplier': float(brightness_factor),
            'rgb_channel_multipliers': rgb_factors.tolist(),
            'target_mean_rgb': real_rgb.tolist(),
            'target_brightness': float(self.real_stats['brightness_contrast']['mean_brightness']),
            'target_contrast': float(self.real_stats['brightness_contrast']['mean_contrast']),

            # Para SyntheticDomainAdaptation transform
            'recommended_transform_params': {
                'brightness_factor': [
                    float(max(0.5, brightness_factor - 0.1)),
                    float(min(1.5, brightness_factor + 0.1))
                ],
                'channel_adjustments': {
                    'red': float(rgb_factors[0]),
                    'green': float(rgb_factors[1]),
                    'blue': float(rgb_factors[2]),
                },
                'application_probability': 0.7
            }
        }

        return corrections

    def plot_comparison(self, output_dir: Path):
        """Genera gráficos de comparación."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Comparación RGB
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Mean RGB
        ax = axes[0, 0]
        channels = ['R', 'G', 'B']
        x = np.arange(len(channels))
        width = 0.35

        synth_rgb = self.synthetic_stats['rgb']['mean_rgb']
        real_rgb = self.real_stats['rgb']['mean_rgb']

        ax.bar(x - width/2, synth_rgb, width, label='Sintético', alpha=0.8)
        ax.bar(x + width/2, real_rgb, width, label='Real', alpha=0.8)
        ax.set_ylabel('Valor Promedio')
        ax.set_title('Comparación RGB Promedio')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Brightness comparison
        ax = axes[0, 1]
        datasets = ['Sintético', 'Real']
        brightness = [
            self.synthetic_stats['brightness_contrast']['mean_brightness'],
            self.real_stats['brightness_contrast']['mean_brightness']
        ]
        ax.bar(datasets, brightness, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
        ax.set_ylabel('Brillo Promedio')
        ax.set_title('Comparación de Brillo')
        ax.grid(True, alpha=0.3, axis='y')

        # Contrast comparison
        ax = axes[1, 0]
        contrast = [
            self.synthetic_stats['brightness_contrast']['mean_contrast'],
            self.real_stats['brightness_contrast']['mean_contrast']
        ]
        ax.bar(datasets, contrast, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
        ax.set_ylabel('Contraste Promedio')
        ax.set_title('Comparación de Contraste')
        ax.grid(True, alpha=0.3, axis='y')

        # Sharpness comparison
        ax = axes[1, 1]
        sharpness = [
            self.synthetic_stats['sharpness']['mean_sharpness'],
            self.real_stats['sharpness']['mean_sharpness']
        ]
        ax.bar(datasets, sharpness, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
        ax.set_ylabel('Nitidez (Laplacian Variance)')
        ax.set_title('Comparación de Nitidez')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / 'domain_gap_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\nGráficos guardados en: {output_dir / 'domain_gap_comparison.png'}")
        plt.close()

    def print_report(self):
        """Imprime reporte detallado."""
        print("\n" + "="*80)
        print("REPORTE DE DOMAIN GAP - SINTÉTICO vs REAL")
        print("="*80)

        print("\n📊 ESTADÍSTICAS RGB PROMEDIO:")
        print(f"  Sintético: R={self.synthetic_stats['rgb']['mean_rgb'][0]:.1f}, "
              f"G={self.synthetic_stats['rgb']['mean_rgb'][1]:.1f}, "
              f"B={self.synthetic_stats['rgb']['mean_rgb'][2]:.1f}")
        print(f"  Real:      R={self.real_stats['rgb']['mean_rgb'][0]:.1f}, "
              f"G={self.real_stats['rgb']['mean_rgb'][1]:.1f}, "
              f"B={self.real_stats['rgb']['mean_rgb'][2]:.1f}")
        print(f"  Gap:       R={self.gap_report['rgb_mean_diff'][0]:+.1f}, "
              f"G={self.gap_report['rgb_mean_diff'][1]:+.1f}, "
              f"B={self.gap_report['rgb_mean_diff'][2]:+.1f}")

        print("\n💡 BRILLO:")
        print(f"  Sintético: {self.synthetic_stats['brightness_contrast']['mean_brightness']:.1f}")
        print(f"  Real:      {self.real_stats['brightness_contrast']['mean_brightness']:.1f}")
        print(f"  Gap:       {self.gap_report['brightness_diff']:+.1f} "
              f"({self.gap_report['brightness_ratio']:.2f}x)")

        print("\n🎨 CONTRASTE:")
        print(f"  Sintético: {self.synthetic_stats['brightness_contrast']['mean_contrast']:.1f}")
        print(f"  Real:      {self.real_stats['brightness_contrast']['mean_contrast']:.1f}")
        print(f"  Gap:       {self.gap_report['contrast_diff']:+.1f} "
              f"({self.gap_report['contrast_ratio']:.2f}x)")

        print("\n🔍 NITIDEZ:")
        print(f"  Sintético: {self.synthetic_stats['sharpness']['mean_sharpness']:.1f}")
        print(f"  Real:      {self.real_stats['sharpness']['mean_sharpness']:.1f}")
        print(f"  Gap:       {self.gap_report['sharpness_diff']:+.1f} "
              f"({self.gap_report['sharpness_ratio']:.2f}x)")

        print("\n📏 SCORE DE GAP GENERAL:")
        score = self.gap_report.get('overall_gap_score', 0)
        if score < 0.3:
            status = "✅ BAJO"
        elif score < 0.6:
            status = "⚠️  MODERADO"
        else:
            status = "❌ ALTO"
        print(f"  Score: {score:.3f} ({status})")

        print("\n" + "="*80)

    def run_analysis(self, output_dir: str) -> Dict:
        """Ejecuta análisis completo."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Obtener lista de archivos (sin cargar imágenes)
        print("\n🔍 Buscando imágenes sintéticas...")
        synthetic_files = self.get_image_files(self.synthetic_dir, self.num_samples)
        print(f"  Encontradas: {len(synthetic_files)} imágenes")

        print("\n🔍 Buscando imágenes reales...")
        real_files = self.get_image_files(self.real_dir, self.num_samples)
        print(f"  Encontradas: {len(real_files)} imágenes")

        if len(synthetic_files) == 0 or len(real_files) == 0:
            raise ValueError("No se encontraron imágenes. Verifica las rutas.")

        # Analizar datasets usando streaming (memoria eficiente)
        self.synthetic_stats = self.analyze_images_streaming(synthetic_files, "SINTÉTICO")

        # Garbage collection entre datasets
        gc.collect()

        self.real_stats = self.analyze_images_streaming(real_files, "REAL")

        # Calcular gap
        print("\n📏 Calculando domain gap...")
        self.gap_report = self.compute_gap()

        # Generar correcciones
        print("🔧 Generando parámetros de corrección...")
        corrections = self.generate_correction_parameters()

        # Guardar resultados
        results = {
            'synthetic_stats': self.synthetic_stats,
            'real_stats': self.real_stats,
            'gap_report': self.gap_report,
            'correction_parameters': corrections,
            'overall_gap_score': self.gap_report.get('overall_gap_score', 0),
        }

        # Guardar análisis completo
        with open(output_path / 'domain_gap_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Guardar solo el summary para el script de validación
        summary = {
            'overall_gap_score': self.gap_report.get('overall_gap_score', 0),
            'rgb_gap_magnitude': self.gap_report.get('rgb_mean_diff_magnitude', 0),
            'brightness_ratio': self.gap_report.get('brightness_ratio', 1),
            'contrast_ratio': self.gap_report.get('contrast_ratio', 1),
        }
        with open(output_path / 'domain_gap_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Resultados guardados en: {output_path}")

        # Generar visualizaciones
        print("\n📊 Generando gráficos...")
        self.plot_comparison(output_path)

        # Imprimir reporte
        self.print_report()

        # Imprimir recomendaciones
        self.print_recommendations(corrections)

        return results

    def print_recommendations(self, corrections: Dict):
        """Imprime recomendaciones de corrección."""
        print("\n" + "="*80)
        print("🎯 RECOMENDACIONES DE CORRECCIÓN")
        print("="*80)

        rec = corrections['recommended_transform_params']

        print("\n1️⃣  CORRECCIÓN EN LA HERRAMIENTA DE GENERACIÓN:")
        print("   Aplicar estos ajustes DURANTE la generación (preferido):")
        print(f"   - Multiplicar brillo por: {corrections['brightness_multiplier']:.3f}")
        print(f"   - Ajustar canal R por: {rec['channel_adjustments']['red']:.3f}")
        print(f"   - Ajustar canal G por: {rec['channel_adjustments']['green']:.3f}")
        print(f"   - Ajustar canal B por: {rec['channel_adjustments']['blue']:.3f}")

        print("\n2️⃣  CORRECCIÓN EN EL ENTRENAMIENTO:")
        print("   Si no puedes modificar la generación, usa esto en synthetic_6cls_fixed.yml:")
        print(f"""
   - {{type: SyntheticDomainAdaptation,
      brightness_factor: {rec['brightness_factor']},
      red_adjustment: {rec['channel_adjustments']['red']:.3f},
      green_adjustment: {rec['channel_adjustments']['green']:.3f},
      blue_adjustment: {rec['channel_adjustments']['blue']:.3f},
      p: {rec['application_probability']}}}
        """)

        print("\n3️⃣  ORDEN DE PRIORIDAD:")
        print("   ✅ MEJOR: Corregir en generación sintética")
        print("   ⚠️  ALTERNATIVA: Corregir en augmentations de entrenamiento")
        print("   ❌ EVITAR: No corregir (gap demasiado grande)")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analiza domain gap entre datasets sintéticos y reales'
    )
    parser.add_argument(
        '--synthetic-dir',
        type=str,
        required=True,
        help='Directorio con imágenes sintéticas'
    )
    parser.add_argument(
        '--real-dir',
        type=str,
        required=True,
        help='Directorio con imágenes reales'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./domain_gap_analysis',
        help='Directorio para guardar resultados'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=500,
        help='Número de imágenes a muestrear de cada dataset'
    )

    args = parser.parse_args()

    analyzer = DomainGapAnalyzer(
        synthetic_dir=args.synthetic_dir,
        real_dir=args.real_dir,
        num_samples=args.num_samples
    )

    analyzer.run_analysis(args.output_dir)


if __name__ == '__main__':
    main()
