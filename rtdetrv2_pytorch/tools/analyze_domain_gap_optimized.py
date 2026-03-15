"""
Versión OPTIMIZADA de análisis de domain gap - uso eficiente de memoria.

Procesa imágenes en streaming sin cargar todo en memoria.

Uso:
    python tools/analyze_domain_gap_optimized.py \
        --synthetic-dir ./dataset/dataset_sintetico/images \
        --real-dir ./dataset/cleansea_dataset/CocoFormatDataset/train_coco \
        --output-dir ./domain_gap_analysis \
        --num-samples 500
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class DomainGapAnalyzerOptimized:
    """Analiza diferencias estadísticas entre datasets - OPTIMIZADO para memoria."""

    def __init__(self, synthetic_dir: str, real_dir: str, num_samples: int = 500):
        self.synthetic_dir = Path(synthetic_dir)
        self.real_dir = Path(real_dir)
        self.num_samples = num_samples

        self.synthetic_stats = {}
        self.real_stats = {}
        self.gap_report = {}

    def get_image_files(self, directory: Path, max_images: int) -> List[Path]:
        """Obtiene lista de archivos de imagen (sin cargarlas)."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in directory.rglob('*') if f.suffix.lower() in extensions]

        # Muestreo aleatorio si hay demasiadas
        if len(image_files) > max_images:
            np.random.seed(42)
            image_files = list(np.random.choice(image_files, max_images, replace=False))

        return image_files

    def compute_statistics_streaming(self, image_files: List[Path]) -> Dict:
        """
        Calcula estadísticas procesando imágenes una por una (streaming).
        Usa algoritmo de Welford para media/varianza online.
        """
        # Acumuladores para estadísticas online
        n = 0
        mean_rgb = np.zeros(3)
        m2_rgb = np.zeros(3)  # Para varianza

        all_brightness = []
        all_contrast = []
        all_sharpness = []

        # HSV acumuladores
        mean_hsv = np.zeros(3)
        m2_hsv = np.zeros(3)

        # Percentiles (necesitamos guardar muestras)
        rgb_samples = []
        max_samples_percentile = min(10000, len(image_files) * 100)  # Limitar samples

        print(f"Procesando {len(image_files)} imágenes en streaming...")

        sharpness_count = 0

        for idx, img_path in enumerate(tqdm(image_files, desc="Analizando")):
            try:
                # Leer imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Redimensionar si es muy grande (para ahorrar memoria)
                h, w = img_rgb.shape[:2]
                if max(h, w) > 1024:
                    scale = 1024 / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_rgb = cv2.resize(img_rgb, (new_w, new_h))

                # Actualizar estadísticas RGB (algoritmo de Welford)
                pixels = img_rgb.reshape(-1, 3).astype(np.float64)

                for pixel in pixels[::10]:  # Muestreo 1 de cada 10 pixels para velocidad
                    n += 1
                    delta = pixel - mean_rgb
                    mean_rgb += delta / n
                    delta2 = pixel - mean_rgb
                    m2_rgb += delta * delta2

                # Guardar muestras para percentiles (limitado)
                if len(rgb_samples) < max_samples_percentile:
                    sample_pixels = pixels[::50]  # 1 de cada 50
                    rgb_samples.extend(sample_pixels[:min(100, len(sample_pixels))])

                # Brillo y contraste
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                all_brightness.append(gray.mean())
                all_contrast.append(gray.std())

                # Nitidez (solo primeras 100 imágenes para velocidad)
                if sharpness_count < 100:
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    all_sharpness.append(laplacian.var())
                    sharpness_count += 1

                # HSV (muestreo)
                if idx % 5 == 0:  # 1 de cada 5 imágenes para HSV
                    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                    hsv_pixels = hsv.reshape(-1, 3).astype(np.float64)
                    for pixel_hsv in hsv_pixels[::20]:
                        delta_hsv = pixel_hsv - mean_hsv
                        mean_hsv += delta_hsv / (n / 20)  # Aproximación

                # Liberar memoria
                del img, img_rgb, pixels, gray

            except Exception as e:
                print(f"\nError procesando {img_path}: {e}")
                continue

        # Calcular varianza final
        if n > 1:
            variance_rgb = m2_rgb / (n - 1)
            std_rgb = np.sqrt(variance_rgb)
        else:
            std_rgb = np.zeros(3)

        # Calcular percentiles de muestras guardadas
        rgb_samples = np.array(rgb_samples)

        stats = {
            'rgb': {
                'mean_rgb': mean_rgb.tolist(),
                'std_rgb': std_rgb.tolist(),
                'median_rgb': np.median(rgb_samples, axis=0).tolist() if len(rgb_samples) > 0 else mean_rgb.tolist(),
                'min_rgb': rgb_samples.min(axis=0).tolist() if len(rgb_samples) > 0 else [0, 0, 0],
                'max_rgb': rgb_samples.max(axis=0).tolist() if len(rgb_samples) > 0 else [255, 255, 255],
                'percentile_25_rgb': np.percentile(rgb_samples, 25, axis=0).tolist() if len(rgb_samples) > 0 else mean_rgb.tolist(),
                'percentile_75_rgb': np.percentile(rgb_samples, 75, axis=0).tolist() if len(rgb_samples) > 0 else mean_rgb.tolist(),
            },
            'hsv': {
                'mean_hsv': mean_hsv.tolist(),
                'std_hsv': [0, 0, 0],  # Simplificado
                'median_hsv': mean_hsv.tolist(),
            },
            'brightness_contrast': {
                'mean_brightness': float(np.mean(all_brightness)),
                'std_brightness': float(np.std(all_brightness)),
                'mean_contrast': float(np.mean(all_contrast)),
                'std_contrast': float(np.std(all_contrast)),
            },
            'sharpness': {
                'mean_sharpness': float(np.mean(all_sharpness)) if all_sharpness else 0.0,
                'std_sharpness': float(np.std(all_sharpness)) if all_sharpness else 0.0,
            }
        }

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

        return gap

    def generate_correction_parameters(self) -> Dict:
        """Genera parámetros de corrección para cerrar el gap."""
        gap = self.gap_report

        # Calcular factores de corrección
        brightness_factor = self.real_stats['brightness_contrast']['mean_brightness'] / \
                          self.synthetic_stats['brightness_contrast']['mean_brightness']

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

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Mean RGB
        ax = axes[0, 0]
        channels = ['R', 'G', 'B']
        x = np.arange(len(channels))
        width = 0.35

        synth_rgb = self.synthetic_stats['rgb']['mean_rgb']
        real_rgb = self.real_stats['rgb']['mean_rgb']

        ax.bar(x - width/2, synth_rgb, width, label='Sintético', alpha=0.8, color='#ff7f0e')
        ax.bar(x + width/2, real_rgb, width, label='Real', alpha=0.8, color='#2ca02c')
        ax.set_ylabel('Valor Promedio')
        ax.set_title('Comparación RGB Promedio')
        ax.set_xticks(x)
        ax.set_xticklabels(channels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Brightness
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

        # Contrast
        ax = axes[1, 0]
        contrast = [
            self.synthetic_stats['brightness_contrast']['mean_contrast'],
            self.real_stats['brightness_contrast']['mean_contrast']
        ]
        ax.bar(datasets, contrast, color=['#ff7f0e', '#2ca02c'], alpha=0.8)
        ax.set_ylabel('Contraste Promedio')
        ax.set_title('Comparación de Contraste')
        ax.grid(True, alpha=0.3, axis='y')

        # Sharpness
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
        plt.savefig(output_dir / 'domain_gap_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráficos guardados en: {output_dir / 'domain_gap_comparison.png'}")
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

        print("\n" + "="*80)

    def print_recommendations(self, corrections: Dict):
        """Imprime recomendaciones de corrección."""
        print("\n" + "="*80)
        print("🎯 PARÁMETROS DE CORRECCIÓN RECOMENDADOS")
        print("="*80)

        rec = corrections['recommended_transform_params']

        print("\n📝 Para aplicar con correct_synthetic_images.py:")
        print(f"""
python tools/correct_synthetic_images.py \\
    --input-dir ./dataset/dataset_sintetico/images \\
    --output-dir ./dataset/dataset_sintetico_corrected/images \\
    --brightness-factor {corrections['brightness_multiplier']:.3f} \\
    --red-factor {rec['channel_adjustments']['red']:.3f} \\
    --green-factor {rec['channel_adjustments']['green']:.3f} \\
    --blue-factor {rec['channel_adjustments']['blue']:.3f} \\
    --copy-annotations
        """)

        print("\n" + "="*80)

    def run_analysis(self, output_dir: str) -> Dict:
        """Ejecuta análisis completo."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Obtener listas de archivos
        print("\n🔍 Buscando imágenes sintéticas...")
        synthetic_files = self.get_image_files(self.synthetic_dir, self.num_samples)
        print(f"   Encontradas: {len(synthetic_files)} imágenes")

        print("\n🔍 Buscando imágenes reales...")
        real_files = self.get_image_files(self.real_dir, self.num_samples)
        print(f"   Encontradas: {len(real_files)} imágenes")

        if len(synthetic_files) == 0 or len(real_files) == 0:
            raise ValueError("No se pudieron encontrar imágenes. Verifica las rutas.")

        # Analizar ambos datasets (streaming - sin cargar todo en memoria)
        print("\n" + "="*60)
        print("Analizando dataset: SINTÉTICO")
        print("="*60)
        self.synthetic_stats = self.compute_statistics_streaming(synthetic_files)

        print("\n" + "="*60)
        print("Analizando dataset: REAL")
        print("="*60)
        self.real_stats = self.compute_statistics_streaming(real_files)

        # Calcular gap
        print("\n📏 Calculando domain gap...")
        self.gap_report = self.compute_gap()

        # Generar correcciones
        print("\n🔧 Generando parámetros de corrección...")
        corrections = self.generate_correction_parameters()

        # Guardar resultados
        results = {
            'synthetic_stats': self.synthetic_stats,
            'real_stats': self.real_stats,
            'gap_report': self.gap_report,
            'correction_parameters': corrections,
        }

        with open(output_path / 'domain_gap_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Resultados guardados en: {output_path / 'domain_gap_analysis.json'}")

        # Generar visualizaciones
        print("\n📊 Generando gráficos...")
        self.plot_comparison(output_path)

        # Imprimir reporte
        self.print_report()

        # Imprimir recomendaciones
        self.print_recommendations(corrections)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Analiza domain gap entre datasets (OPTIMIZADO para memoria)'
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

    analyzer = DomainGapAnalyzerOptimized(
        synthetic_dir=args.synthetic_dir,
        real_dir=args.real_dir,
        num_samples=args.num_samples
    )

    analyzer.run_analysis(args.output_dir)


if __name__ == '__main__':
    main()
