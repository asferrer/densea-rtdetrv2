#!/bin/bash
###############################################################################
# Script Helper para Validar Datasets Sintéticos
#
# Uso:
#   ./tools/validate_dataset.sh dataset_sintetico_simple
#   ./tools/validate_dataset.sh dataset_sintetico_v2
#   ./tools/validate_dataset.sh path/to/custom/dataset
#
# El script asume la siguiente estructura:
#   - Imágenes: dataset/<nombre>/images/
#   - JSON train: split_<nombre>/train.json
#   - JSON val: split_<nombre>/val.json
#
# Si tu estructura es diferente, usa directamente:
#   python tools/run_synthetic_validation.py con los parámetros adecuados
###############################################################################

set -e  # Exit on error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_usage() {
    echo -e "${BLUE}Uso:${NC}"
    echo "  $0 <dataset_name> [opciones]"
    echo ""
    echo -e "${BLUE}Ejemplos:${NC}"
    echo "  $0 dataset_sintetico_simple"
    echo "  $0 dataset_sintetico_v2"
    echo "  $0 dataset_sintetico_simple --quick"
    echo "  $0 dataset_sintetico_simple --phases 1"
    echo ""
    echo -e "${BLUE}Opciones adicionales:${NC}"
    echo "  --quick              Modo rápido (menos muestras)"
    echo "  --phases 1 2 3 4     Fases específicas a ejecutar"
    echo "  --skip-pipeline      Saltar validación de pipeline"
    echo "  --skip-bbox-check    Saltar verificación de padding"
    echo ""
    echo -e "${BLUE}Estructura esperada:${NC}"
    echo "  dataset/<nombre>/images/           <- Imágenes"
    echo "  split_<nombre>/train.json          <- JSON entrenamiento"
    echo "  split_<nombre>/val.json            <- JSON validación"
}

function print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

function print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Verificar argumentos
if [ $# -lt 1 ]; then
    print_error "Debes especificar el nombre del dataset"
    echo ""
    print_usage
    exit 1
fi

DATASET_NAME="$1"
shift  # Remove first argument

# Detectar automáticamente los paths
if [[ "$DATASET_NAME" == dataset_* ]]; then
    # Formato: dataset_sintetico_simple -> split_sintetico_simple
    SPLIT_NAME=$(echo "$DATASET_NAME" | sed 's/^dataset_/split_/')
else
    # Si no empieza con "dataset_", asumir que es el nombre completo
    SPLIT_NAME="$DATASET_NAME"
    DATASET_NAME="dataset_$DATASET_NAME"
fi

IMAGE_DIR="dataset/${DATASET_NAME}/images"
TRAIN_JSON="${SPLIT_NAME}/train.json"
VAL_JSON="${SPLIT_NAME}/val.json"

# Verificar que los paths existen
print_info "Verificando estructura del dataset..."
echo "  Dataset: $DATASET_NAME"
echo "  Split: $SPLIT_NAME"
echo "  Imágenes: $IMAGE_DIR"
echo "  Train JSON: $TRAIN_JSON"
echo "  Val JSON: $VAL_JSON"
echo ""

ERRORS=0

if [ ! -d "$IMAGE_DIR" ]; then
    print_error "Directorio de imágenes no encontrado: $IMAGE_DIR"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "$TRAIN_JSON" ]; then
    print_error "JSON de entrenamiento no encontrado: $TRAIN_JSON"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "$VAL_JSON" ]; then
    print_error "JSON de validación no encontrado: $VAL_JSON"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    print_error "No se encontraron todos los archivos necesarios"
    print_info "Estructura esperada:"
    echo "  $IMAGE_DIR/"
    echo "  $TRAIN_JSON"
    echo "  $VAL_JSON"
    echo ""
    print_info "Si tu estructura es diferente, usa directamente:"
    echo "  python tools/run_synthetic_validation.py \\"
    echo "      --train-json path/to/train.json \\"
    echo "      --val-json path/to/val.json \\"
    echo "      --image-dir path/to/images"
    exit 1
fi

print_success "Todos los archivos encontrados"
echo ""

# Construir comando
CMD="python tools/run_synthetic_validation.py"
CMD="$CMD --train-json $TRAIN_JSON"
CMD="$CMD --val-json $VAL_JSON"
CMD="$CMD --image-dir $IMAGE_DIR"

# Agregar opciones adicionales
if [ $# -eq 0 ]; then
    # Por defecto, ejecutar fases 1 y 2
    CMD="$CMD --phases 1 2"
else
    # Pasar todas las opciones adicionales
    CMD="$CMD $@"
fi

# Mostrar comando
print_info "Ejecutando validación..."
echo ""
echo -e "${BLUE}Comando:${NC}"
echo "  $CMD"
echo ""

# Ejecutar
$CMD

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    print_success "Validación completada"
else
    print_error "Validación falló (código: $exit_code)"
fi

exit $exit_code
