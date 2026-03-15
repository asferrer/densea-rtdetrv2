"""
Post-procesamiento de imágenes sintéticas para cerrar domain gap.

Aplica correcciones de color/brillo basadas en análisis de domain gap
para acercar imágenes sintéticas a distribución real.

Uso:
    # Usando parámetros auto-detectados
    python tools/correct_synthetic_images.py \
        --input-dir ./dataset/dataset_sintetico/images \
        --output-dir ./dataset/dataset_sintetico_corrected/images \
        --gap-analysis ./domain_gap_analysis/domain_gap_analysis.json

    # Usando parámetros manuales
    python tools/correct_synthetic_images.py \
        --input-dir ./dataset/dataset_sintetico/images \
        --output-dir ./dataset/dataset_sintetico_corrected/images \
        --brightness-factor 0.85 \
        --red-factor 0.92 \
        --green-factor 0.95 \
        --blue-factor 0.88
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class SyntheticImageCorrector:
    """Aplica correcciones de color/brillo a imágenes sintéticas."""

    def __init__(
        self,
        brightness_factor: float = 1.0,
        red_factor: float = 1.0,
        green_factor: float = 1.0,
        blue_factor: float = 1.0,
        contrast_adjustment: float = 0.0,
    ):
        """
        Args:
            brightness_factor: Multiplicador de brillo global (< 1 oscurece, > 1 aclara)
            red_factor: Multiplicador para canal rojo
            green_factor: Multiplicador para canal verde
            blue_factor: Multiplicador para canal azul
            contrast_adjustment: Ajuste de contraste (-100 a 100)
        """
        self.brightness_factor = brightness_factor
        self.channel_factors = np.array([red_factor, green_factor, blue_factor])
        self.contrast_adjustment = contrast_adjustment

    @classmethod
    def from_gap_analysis(cls, gap_analysis_path: str) -> 'SyntheticImageCorrector':
        """Crea corrector desde archivo de análisis de domain gap."""
        with open(gap_analysis_path, 'r') as f:
            data = json.load(f)

        corrections = data['correction_parameters']
        rec = corrections['recommended_transform_params']

        return cls(
            brightness_factor=corrections['brightness_multiplier'],
            red_factor=rec['channel_adjustments']['red'],
            green_factor=rec['channel_adjustments']['green'],
            blue_factor=rec['channel_adjustments']['blue'],
        )

    def correct_image(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica correcciones a una imagen.

        Args:
            img: Imagen en formato RGB, uint8 [0, 255]

        Returns:
            Imagen corregida en formato RGB, uint8 [0, 255]
        """
        # Convertir a float para operaciones
        img_float = img.astype(np.float32)

        # Aplicar correcciones por canal
        img_corrected = img_float * self.channel_factors.reshape(1, 1, 3)

        # Aplicar brillo global
        img_corrected = img_corrected * self.brightness_factor

        # Aplicar contraste si es necesario
        if abs(self.contrast_adjustment) > 0.01:
            mean = img_corrected.mean()
            img_corrected = mean + (img_corrected - mean) * (1 + self.contrast_adjustment / 100)

        # Clip y convertir de vuelta a uint8
        img_corrected = np.clip(img_corrected, 0, 255).astype(np.uint8)

        return img_corrected

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        show_preview: bool = False
    ) -> Dict[str, int]:
        """
        Procesa todas las imágenes de un directorio.

        Args:
            input_dir: Directorio con imágenes sintéticas originales
            output_dir: Directorio para guardar imágenes corregidas
            show_preview: Si True, muestra preview de algunas correcciones

        Returns:
            Diccionario con estadísticas de procesamiento
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in input_dir.rglob('*') if f.suffix.lower() in extensions]

        stats = {
            'total': len(image_files),
            'processed': 0,
            'errors': 0,
        }

        print(f"\n🔧 Procesando {len(image_files)} imágenes...")
        print(f"   Entrada: {input_dir}")
        print(f"   Salida: {output_dir}")
        print(f"\n   Parámetros de corrección:")
        print(f"   - Brillo: {self.brightness_factor:.3f}x")
        print(f"   - Canal R: {self.channel_factors[0]:.3f}x")
        print(f"   - Canal G: {self.channel_factors[1]:.3f}x")
        print(f"   - Canal B: {self.channel_factors[2]:.3f}x")
        print()

        preview_shown = False

        for img_path in tqdm(image_files, desc="Corrigiendo imágenes"):
            try:
                # Leer imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    stats['errors'] += 1
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Aplicar correcciones
                img_corrected = self.correct_image(img_rgb)

                # Guardar manteniendo estructura de directorios
                relative_path = img_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                img_corrected_bgr = cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img_corrected_bgr)

                stats['processed'] += 1

                # Mostrar preview para primera imagen
                if show_preview and not preview_shown:
                    self._show_preview(img_rgb, img_corrected, img_path.name)
                    preview_shown = True

            except Exception as e:
                print(f"\n❌ Error procesando {img_path}: {e}")
                stats['errors'] += 1

        return stats

    def _show_preview(self, original: np.ndarray, corrected: np.ndarray, filename: str):
        """Muestra preview de corrección."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(original)
        axes[0].set_title(f'Original - {filename}')
        axes[0].axis('off')

        axes[1].imshow(corrected)
        axes[1].set_title(f'Corregido - {filename}')
        axes[1].axis('off')

        # Agregar estadísticas
        orig_mean = original.mean(axis=(0, 1))
        corr_mean = corrected.mean(axis=(0, 1))

        info_text = (
            f"RGB Promedio Original: ({orig_mean[0]:.1f}, {orig_mean[1]:.1f}, {orig_mean[2]:.1f})\n"
            f"RGB Promedio Corregido: ({corr_mean[0]:.1f}, {corr_mean[1]:.1f}, {corr_mean[2]:.1f})"
        )
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()


def copy_annotations(input_dir: Path, output_dir: Path):
    """Copia archivos de anotaciones al directorio de salida."""
    json_files = list(input_dir.parent.rglob('*.json'))

    for json_file in json_files:
        if json_file.parent == input_dir:
            continue  # Skip si está dentro del directorio de imágenes

        # Copiar al mismo nivel relativo
        relative_path = json_file.relative_to(input_dir.parent)
        output_path = output_dir.parent / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy2(json_file, output_path)
        print(f"📄 Copiado: {json_file.name} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Corrige imágenes sintéticas para reducir domain gap'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directorio con imágenes sintéticas originales'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directorio para guardar imágenes corregidas'
    )
    parser.add_argument(
        '--gap-analysis',
        type=str,
        default=None,
        help='Ruta al archivo domain_gap_analysis.json (auto-detecta parámetros)'
    )
    parser.add_argument(
        '--brightness-factor',
        type=float,
        default=None,
        help='Multiplicador de brillo (ej: 0.85 para oscurecer 15%%)'
    )
    parser.add_argument(
        '--red-factor',
        type=float,
        default=None,
        help='Multiplicador para canal rojo'
    )
    parser.add_argument(
        '--green-factor',
        type=float,
        default=None,
        help='Multiplicador para canal verde'
    )
    parser.add_argument(
        '--blue-factor',
        type=float,
        default=None,
        help='Multiplicador para canal azul'
    )
    parser.add_argument(
        '--contrast-adjustment',
        type=float,
        default=0.0,
        help='Ajuste de contraste (-100 a 100)'
    )
    parser.add_argument(
        '--show-preview',
        action='store_true',
        help='Mostrar preview de corrección'
    )
    parser.add_argument(
        '--copy-annotations',
        action='store_true',
        help='Copiar archivos de anotaciones JSON al directorio de salida'
    )

    args = parser.parse_args()

    # Crear corrector
    if args.gap_analysis:
        print(f"📊 Cargando parámetros desde: {args.gap_analysis}")
        corrector = SyntheticImageCorrector.from_gap_analysis(args.gap_analysis)
    elif all(x is not None for x in [args.brightness_factor, args.red_factor,
                                       args.green_factor, args.blue_factor]):
        print("🔧 Usando parámetros manuales")
        corrector = SyntheticImageCorrector(
            brightness_factor=args.brightness_factor,
            red_factor=args.red_factor,
            green_factor=args.green_factor,
            blue_factor=args.blue_factor,
            contrast_adjustment=args.contrast_adjustment,
        )
    else:
        parser.error(
            "Debes proporcionar --gap-analysis O todos los parámetros manuales "
            "(--brightness-factor, --red-factor, --green-factor, --blue-factor)"
        )

    # Procesar imágenes
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    stats = corrector.process_directory(input_dir, output_dir, args.show_preview)

    # Copiar anotaciones si se solicita
    if args.copy_annotations:
        print("\n📋 Copiando archivos de anotaciones...")
        copy_annotations(input_dir, output_dir)

    # Imprimir resumen
    print("\n" + "="*60)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("="*60)
    print(f"Total de imágenes: {stats['total']}")
    print(f"Procesadas exitosamente: {stats['processed']}")
    print(f"Errores: {stats['errors']}")
    print(f"\n📁 Imágenes corregidas guardadas en: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
