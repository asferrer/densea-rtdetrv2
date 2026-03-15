"""
Extrae parámetros de corrección de domain_gap_analysis.json y genera config YAML.

Uso:
    python tools/extract_correction_params.py \
        --gap-analysis ./domain_gap_analysis/domain_gap_analysis.json \
        --output ./domain_gap_correction_params.yml
"""

import argparse
import json
from pathlib import Path


def extract_yaml_config(gap_analysis_path: str, output_path: str = None):
    """Extrae parámetros y genera configuración YAML."""

    # Cargar análisis
    with open(gap_analysis_path, 'r') as f:
        data = json.load(f)

    corrections = data['correction_parameters']
    rec = corrections['recommended_transform_params']

    # Calcular rangos para parámetros (±10% de variación)
    brightness = corrections['brightness_multiplier']
    r_factor = rec['channel_adjustments']['red']
    g_factor = rec['channel_adjustments']['green']
    b_factor = rec['channel_adjustments']['blue']

    # Crear rangos con variación
    def make_range(value, variation=0.05):
        """Crea rango [value-variation, value+variation] clipeado a [0.5, 1.5]."""
        low = max(0.5, value - variation)
        high = min(1.5, value + variation)
        return [round(low, 3), round(high, 3)]

    brightness_range = make_range(brightness, 0.05)
    red_range = make_range(r_factor, 0.05)
    green_range = make_range(g_factor, 0.05)
    blue_range = make_range(b_factor, 0.05)

    # Generar configuración YAML
    yaml_config = f"""# ============================================================================
# PARÁMETROS DE CORRECCIÓN DE DOMAIN GAP
# Generados automáticamente desde: {Path(gap_analysis_path).name}
# ============================================================================

# Estadísticas de Domain Gap:
# Sintético RGB: [{data['synthetic_stats']['rgb']['mean_rgb'][0]:.1f}, {data['synthetic_stats']['rgb']['mean_rgb'][1]:.1f}, {data['synthetic_stats']['rgb']['mean_rgb'][2]:.1f}]
# Real RGB:      [{data['real_stats']['rgb']['mean_rgb'][0]:.1f}, {data['real_stats']['rgb']['mean_rgb'][1]:.1f}, {data['real_stats']['rgb']['mean_rgb'][2]:.1f}]
# Gap RGB:       [{data['gap_report']['rgb_mean_diff'][0]:+.1f}, {data['gap_report']['rgb_mean_diff'][1]:+.1f}, {data['gap_report']['rgb_mean_diff'][2]:+.1f}]
#
# Sintético Brillo: {data['synthetic_stats']['brightness_contrast']['mean_brightness']:.1f}
# Real Brillo:      {data['real_stats']['brightness_contrast']['mean_brightness']:.1f}
# Gap Brillo:       {data['gap_report']['brightness_diff']:+.1f} ({data['gap_report']['brightness_ratio']:.2f}x)

# ============================================================================
# CONFIGURACIÓN PARA synthetic_6cls_fixed.yml
# ============================================================================

# OPCIÓN 1: Copia esto en transforms → ops (RECOMENDADO)
# Reemplaza la línea actual de SyntheticDomainAdaptation con:

- {{type: SyntheticDomainAdaptation,
   brightness_factor: {brightness_range},    # Brillo: {brightness:.3f} ±5%
   red_adjustment: {red_range},              # Canal R: {r_factor:.3f} ±5%
   green_adjustment: {green_range},          # Canal G: {g_factor:.3f} ±5%
   blue_adjustment: {blue_range},            # Canal B: {b_factor:.3f} ±5%
   p: 0.7}}                                  # Aplicar al 70% de imágenes

# ============================================================================
# OPCIÓN 2: Parámetros exactos (sin variación aleatoria)
# ============================================================================

- {{type: SyntheticDomainAdaptation,
   brightness_factor: [{brightness:.3f}, {brightness:.3f}],
   red_adjustment: [{r_factor:.3f}, {r_factor:.3f}],
   green_adjustment: [{g_factor:.3f}, {g_factor:.3f}],
   blue_adjustment: [{b_factor:.3f}, {b_factor:.3f}],
   p: 1.0}}  # Aplicar siempre (100%)

# ============================================================================
# PARÁMETROS PARA correct_synthetic_images.py (si prefieres post-procesar)
# ============================================================================

python tools/correct_synthetic_images.py \\
    --input-dir ./dataset/dataset_sintetico/images \\
    --output-dir ./dataset/dataset_sintetico_corrected/images \\
    --brightness-factor {brightness:.3f} \\
    --red-factor {r_factor:.3f} \\
    --green-factor {g_factor:.3f} \\
    --blue-factor {b_factor:.3f} \\
    --copy-annotations

# ============================================================================
# VALIDACIÓN
# ============================================================================
# Después de aplicar estos parámetros, re-analizar gap:
#
# python tools/analyze_domain_gap_optimized.py \\
#     --synthetic-dir ./dataset/dataset_sintetico/images \\
#     --real-dir ./dataset/cleansea_dataset/CocoFormatDataset/train_coco \\
#     --output-dir ./domain_gap_analysis_v2 \\
#     --num-samples 300
#
# Objetivo: RGB mean diff < 10 por canal
"""

    # Guardar o imprimir
    if output_path:
        with open(output_path, 'w') as f:
            f.write(yaml_config)
        print(f"\n✅ Parámetros guardados en: {output_path}")
        print(f"\n📋 Copia la sección 'OPCIÓN 1' en tu archivo de configuración:")
        print(f"    {Path(output_path).absolute()}")
    else:
        print(yaml_config)

    # También imprimir resumen
    print("\n" + "="*80)
    print("📊 RESUMEN DE PARÁMETROS DE CORRECCIÓN")
    print("="*80)
    print(f"\nBrightness: {brightness_range[0]:.3f} - {brightness_range[1]:.3f} (centro: {brightness:.3f})")
    print(f"Red:        {red_range[0]:.3f} - {red_range[1]:.3f} (centro: {r_factor:.3f})")
    print(f"Green:      {green_range[0]:.3f} - {green_range[1]:.3f} (centro: {g_factor:.3f})")
    print(f"Blue:       {blue_range[0]:.3f} - {blue_range[1]:.3f} (centro: {b_factor:.3f})")

    print(f"\n💡 Interpretación:")
    if brightness < 0.9:
        print(f"  - Sintéticos son más BRILLANTES → oscurecer {(1-brightness)*100:.1f}%")
    elif brightness > 1.1:
        print(f"  - Sintéticos son más OSCUROS → aclarar {(brightness-1)*100:.1f}%")
    else:
        print(f"  - Brillo está bien balanceado")

    rgb_gap = data['gap_report']['rgb_mean_diff']
    if abs(rgb_gap[0]) > 15:
        print(f"  - Canal ROJO: gap de {rgb_gap[0]:+.1f} (ajustar con factor {r_factor:.3f})")
    if abs(rgb_gap[1]) > 15:
        print(f"  - Canal VERDE: gap de {rgb_gap[1]:+.1f} (ajustar con factor {g_factor:.3f})")
    if abs(rgb_gap[2]) > 15:
        print(f"  - Canal AZUL: gap de {rgb_gap[2]:+.1f} (ajustar con factor {b_factor:.3f})")

    print("\n" + "="*80)

    return {
        'brightness_range': brightness_range,
        'red_range': red_range,
        'green_range': green_range,
        'blue_range': blue_range,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extrae parámetros de corrección de domain gap analysis'
    )
    parser.add_argument(
        '--gap-analysis',
        type=str,
        required=True,
        help='Ruta a domain_gap_analysis.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Archivo de salida para parámetros YAML (opcional)'
    )

    args = parser.parse_args()

    if not Path(args.gap_analysis).exists():
        print(f"❌ Error: Archivo no encontrado: {args.gap_analysis}")
        print(f"\n💡 Primero ejecuta:")
        print(f"   python tools/analyze_domain_gap_optimized.py \\")
        print(f"       --synthetic-dir ./dataset/dataset_sintetico/images \\")
        print(f"       --real-dir ./dataset/cleansea_dataset/CocoFormatDataset/train_coco \\")
        print(f"       --output-dir ./domain_gap_analysis")
        return

    extract_yaml_config(args.gap_analysis, args.output)


if __name__ == '__main__':
    main()
