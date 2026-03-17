#!/bin/bash
################################################################################
#  RT-DETRv2 DenSea Exp06 — Pipeline completo: Export + Build + Eval
#  Jetson Orin Nano Super (8GB) — TensorRT 10.3
#
#  USO:  chmod +x run_full_eval.sh && ./run_full_eval.sh
#  NOTA: Ejecutar desde terminal independiente (sin VSCode) para max memoria
################################################################################

set -uo pipefail

# Cachear credenciales sudo para toda la ejecucion (evita pedir password a mitad)
echo "Solicitando permisos sudo para toda la sesion..."
sudo -v
# Mantener sudo vivo en background durante todo el script
( while true; do sudo -n true; sleep 55; done ) 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null' EXIT

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_step()  { echo -e "\n${CYAN}================================================================${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}================================================================${NC}"; }
log_ok()    { echo -e "${GREEN}[OK] $1${NC}"; }
log_warn()  { echo -e "${YELLOW}[WARN] $1${NC}"; }
log_fail()  { echo -e "${RED}[FAIL] $1${NC}"; }

# ============================================================================
# CONFIGURACION
# ============================================================================

WORKDIR="/home/saflex/Desktop/RT-DETRv2-Densea/rtdetrv2_pytorch"
CONFIG="experiment_output/densea/configs/exp06_synth100_real100_eval.yml"
CHECKPOINT="rtdetrv2_r50vd_densea_exp06.pth"
ENGINE_DIR=".."
ENGINE_PREFIX="rtdetrv2_r50vd_densea_exp06"
ONNX_RAW="../${ENGINE_PREFIX}.onnx"
ONNX_OUTPUT="../${ENGINE_PREFIX}_simplified.onnx"
EVAL_DIR="eval_results"
CONDA_ENV="densea_env"
WORKSPACE_MB=2048
OPT_LEVEL=5
INT8_CALIB_IMAGES=100
TIMING_CACHE="timing_cache.bin"
CALIB_DIR="calibration_set"

# Fichero de log con timestamp — captura TODA la salida (stdout + stderr)
LOGFILE="/home/saflex/Desktop/RT-DETRv2-Densea/eval_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "=== Log iniciado: $(date) ==="
echo "=== Logfile: $LOGFILE ==="

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

drop_caches() {
    # Liberar cache del kernel + esperar a que el driver CUDA libere memoria GPU
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    sync
    sleep 3
    echo "Memoria tras limpieza:"
    free -h | grep Mem
}

run_python() {
    conda run -n "$CONDA_ENV" --no-capture-output python "$@"
}

check_file() {
    if [ -f "$1" ]; then
        log_ok "$1 ($(ls -lh "$1" | awk '{print $5}'))"
        return 0
    else
        log_fail "No se genero: $1"
        return 1
    fi
}

# ============================================================================
# PASO 0: PREPARACION DEL SISTEMA
# ============================================================================

log_step "PASO 0/8: Preparacion del sistema — liberando memoria maxima"

# Matar procesos pesados no esenciales para liberar RAM
echo "Matando procesos pesados (VSCode, node, snap, tracker, etc.)..."
pkill -f code 2>/dev/null || true
pkill -f "node " 2>/dev/null || true
pkill -f snap-store 2>/dev/null || true
pkill -f tracker-miner 2>/dev/null || true
pkill -f tracker-extract 2>/dev/null || true
pkill -f evolution-data 2>/dev/null || true
pkill -f goa-daemon 2>/dev/null || true
pkill -f gnome-software 2>/dev/null || true
sleep 2

# Parar servicios no esenciales
for svc in bluetooth cups cups-browsed avahi-daemon ModemManager wpa_supplicant snapd fwupd power-profiles-daemon thermald; do
    sudo systemctl stop "$svc" 2>/dev/null || true
done

# Maximo rendimiento GPU
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true

drop_caches

echo ""
free -h
echo ""
echo "Objetivo: >5GB libres para builds TensorRT"

# ============================================================================
# PASO 1: LIMPIAR ARTEFACTOS ANTERIORES
# ============================================================================

log_step "PASO 1/8: Limpieza de artefactos anteriores"

cd "$WORKDIR"

rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_fp32.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_fp16.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_fp32_bs1.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_fp16_bs1.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_int8.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_int8_bs1.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_int8_entropy.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_int8_minmax.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_int8_classic.engine"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_simplified.onnx"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_simplified_int8.onnx"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}_simplified_named.onnx"
rm -f "${ENGINE_DIR}/${ENGINE_PREFIX}.onnx"
rm -f "${ENGINE_DIR}/calib_entropy.cache" "${ENGINE_DIR}/calib_minmax.cache"
rm -f calibration.cache
rm -f ../model.onnx
rm -f model.onnx model_simplified.onnx
rm -rf "${EVAL_DIR}"

log_ok "Limpieza completada"

# ============================================================================
# PASO 2: EXPORTAR ONNX
# ============================================================================

log_step "PASO 2/8: Exportar ONNX (RepVgg fusion + Int32 + static batch)"

run_python tools/export_onnx.py \
    -c "$CONFIG" \
    -r "$CHECKPOINT" \
    -o "$ONNX_RAW" \
    --static-batch \
    --fuse-layers \
    --simplify \
    --check

echo ""
check_file "$ONNX_RAW"
check_file "$ONNX_OUTPUT" || exit 1

# ============================================================================
# PASO 3: BUILD TENSORRT FP32
# ============================================================================

log_step "PASO 3/8: Build TensorRT FP32"
drop_caches

FP32_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_fp32.engine"
run_python references/deploy/build_engine.py \
    --onnx "$ONNX_OUTPUT" \
    --output "$FP32_ENGINE" \
    --workspace_mb "$WORKSPACE_MB" \
    --opt_level "$OPT_LEVEL" \
    --timing_cache "$TIMING_CACHE"

check_file "$FP32_ENGINE"

# ============================================================================
# PASO 4: BUILD TENSORRT FP16
# ============================================================================

log_step "PASO 4/8: Build TensorRT FP16"
drop_caches

FP16_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_fp16.engine"
run_python references/deploy/build_engine.py \
    --onnx "$ONNX_OUTPUT" \
    --output "$FP16_ENGINE" \
    --fp16 \
    --workspace_mb "$WORKSPACE_MB" \
    --opt_level "$OPT_LEVEL" \
    --timing_cache "$TIMING_CACHE"

check_file "$FP16_ENGINE"

# ============================================================================
# PASO 5: BUILD TENSORRT INT8 (entropy) — nvidia-modelopt explicit Q/DQ
# ============================================================================

log_step "PASO 5/8: Build TensorRT INT8 (entropy, modelopt Q/DQ)"
drop_caches

INT8_ENTROPY_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_int8_entropy.engine"
run_python references/deploy/build_int8_engine.py \
    --onnx "$ONNX_OUTPUT" \
    --output "$INT8_ENTROPY_ENGINE" \
    --calib_dir "$CALIB_DIR" \
    --max_images "$INT8_CALIB_IMAGES" \
    --calibrate_method entropy \
    --workspace "$WORKSPACE_MB" \
    --opt_level 3 \
    --timing_cache "$TIMING_CACHE" || log_warn "Build INT8 entropy fallo, continuando..."

check_file "$INT8_ENTROPY_ENGINE" || true

# ============================================================================
# PASO 6: BUILD TENSORRT INT8 (minmax) — nvidia-modelopt explicit Q/DQ
# ============================================================================

log_step "PASO 6/8: Build TensorRT INT8 (minmax, modelopt Q/DQ)"
drop_caches

INT8_MINMAX_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_int8_minmax.engine"
run_python references/deploy/build_int8_engine.py \
    --onnx "$ONNX_OUTPUT" \
    --output "$INT8_MINMAX_ENGINE" \
    --calib_dir "$CALIB_DIR" \
    --max_images "$INT8_CALIB_IMAGES" \
    --calibrate_method minmax \
    --workspace "$WORKSPACE_MB" \
    --opt_level 3 \
    --timing_cache "$TIMING_CACHE" || log_warn "Build INT8 minmax fallo, continuando..."

check_file "$INT8_MINMAX_ENGINE" || true

# ============================================================================
# RESUMEN DE BUILDS
# ============================================================================

log_step "Resumen de modelos generados"
echo ""
ls -lhS "${ENGINE_DIR}/${ENGINE_PREFIX}"*.engine "${ENGINE_DIR}/${ENGINE_PREFIX}"*.onnx 2>/dev/null || echo "No se encontraron modelos"
echo ""

# ============================================================================
# PASO 7: EVALUACIONES (5 modelos)
# ============================================================================

EVAL_FAILED=0

# --- EVAL 1: ONNX ---
log_step "PASO 7/8: Evaluacion ONNX (1/5)"
drop_caches
if run_python tools/eval.py -c "$CONFIG" --onnx "$ONNX_OUTPUT" -d cuda --output_dir "${EVAL_DIR}/onnx"; then
    log_ok "Evaluacion ONNX completada"
else
    log_fail "Evaluacion ONNX fallo"
    EVAL_FAILED=$((EVAL_FAILED + 1))
fi

# --- EVAL 2: TensorRT FP32 ---
log_step "Evaluacion TensorRT FP32 (2/5)"
drop_caches
if [ -f "$FP32_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$FP32_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_fp32"; then
    log_ok "Evaluacion TRT FP32 completada"
else
    log_warn "Evaluacion TRT FP32 saltada o fallo"
    EVAL_FAILED=$((EVAL_FAILED + 1))
fi

# --- EVAL 3: TensorRT FP16 ---
log_step "Evaluacion TensorRT FP16 (3/5)"
drop_caches
if [ -f "$FP16_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$FP16_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_fp16"; then
    log_ok "Evaluacion TRT FP16 completada"
else
    log_warn "Evaluacion TRT FP16 saltada o fallo"
    EVAL_FAILED=$((EVAL_FAILED + 1))
fi

# --- EVAL 4: TensorRT INT8 entropy ---
log_step "Evaluacion TensorRT INT8 entropy (4/5)"
drop_caches
if [ -f "$INT8_ENTROPY_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$INT8_ENTROPY_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_int8_entropy"; then
    log_ok "Evaluacion TRT INT8 entropy completada"
else
    log_warn "Evaluacion TRT INT8 entropy saltada o fallo"
    EVAL_FAILED=$((EVAL_FAILED + 1))
fi

# --- EVAL 5: TensorRT INT8 minmax ---
log_step "Evaluacion TensorRT INT8 minmax (5/5)"
drop_caches
if [ -f "$INT8_MINMAX_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$INT8_MINMAX_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_int8_minmax"; then
    log_ok "Evaluacion TRT INT8 minmax completada"
else
    log_warn "Evaluacion TRT INT8 minmax saltada o fallo"
    EVAL_FAILED=$((EVAL_FAILED + 1))
fi

# ============================================================================
# PASO 8: RESUMEN COMPARATIVO
# ============================================================================

log_step "PASO 8/8: Resumen comparativo"

# Header
echo ""
printf "%-22s  %-10s  %-12s  %s\n" "MODELO" "TAMANO" "mAP@50:95" "NOTAS"
printf "%-22s  %-10s  %-12s  %s\n" "----------------------" "----------" "------------" "-----"

for d in "${EVAL_DIR}"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")

    # Buscar mAP en los ficheros de resultados
    map=$(grep -rh "Average Precision.*IoU=0.50:0.95.*area=   all.*maxDets=100" "$d" 2>/dev/null | head -1 | awk '{print $NF}')

    # Tamano del modelo
    case "$name" in
        onnx)             sz=$(ls -lh "$ONNX_OUTPUT" 2>/dev/null | awk '{print $5}') ;;
        trt_fp32)         sz=$(ls -lh "$FP32_ENGINE" 2>/dev/null | awk '{print $5}') ;;
        trt_fp16)         sz=$(ls -lh "$FP16_ENGINE" 2>/dev/null | awk '{print $5}') ;;
        trt_int8_entropy) sz=$(ls -lh "$INT8_ENTROPY_ENGINE" 2>/dev/null | awk '{print $5}') ;;
        trt_int8_minmax)  sz=$(ls -lh "$INT8_MINMAX_ENGINE" 2>/dev/null | awk '{print $5}') ;;
        *)                sz="-" ;;
    esac

    printf "%-22s  %-10s  %-12s\n" "$name" "${sz:--}" "${map:--}"
done

echo ""
echo "Resultados detallados en: ${WORKDIR}/${EVAL_DIR}/*/"
echo ""

if [ "$EVAL_FAILED" -gt 0 ]; then
    log_warn "${EVAL_FAILED} evaluacion(es) fallaron. Revisar logs arriba."
else
    log_ok "Todas las evaluaciones completadas exitosamente!"
fi

echo ""
echo "Pipeline finalizado: $(date)"
