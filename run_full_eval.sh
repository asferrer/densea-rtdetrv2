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
EVAL_OPT_DIR="eval_results_optimized"
CONDA_ENV="densea_env"
WORKSPACE_MB=2048
OPT_LEVEL=5
INT8_CALIB_IMAGES=100
TIMING_CACHE="timing_cache.bin"
CALIB_DIR="calibration_set"

# Dataset paths (para evaluacion optimizada)
EVAL_JSON="split_v8/test.json"
EVAL_IMG_DIR="dataset/test_images"

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

# Pregunta interactiva: si el fichero existe, da opcion de saltar o regenerar.
# Uso: ask_skip "descripcion" fichero1 [fichero2 ...]
# Retorna 0 si se debe SALTAR (ficheros existen y usuario quiere reutilizar),
# retorna 1 si se debe EJECUTAR (ficheros no existen o usuario quiere regenerar).
ask_skip() {
    local description="$1"
    shift
    local all_exist=true
    for f in "$@"; do
        if [ ! -f "$f" ]; then
            all_exist=false
            break
        fi
    done

    if [ "$all_exist" = false ]; then
        return 1  # Ejecutar: algun fichero no existe
    fi

    echo ""
    echo -e "${YELLOW}Los siguientes ficheros ya existen para '${description}':${NC}"
    for f in "$@"; do
        echo "  - $f ($(ls -lh "$f" 2>/dev/null | awk '{print $5}'))"
    done
    echo ""
    echo -n -e "${YELLOW}Reutilizar ficheros existentes? [S/n] (auto-SI en 15s): ${NC}"

    local answer
    if read -t 15 -r answer; then
        case "$answer" in
            [nN]|[nN][oO])
                echo "Regenerando..."
                for f in "$@"; do rm -f "$f"; done
                return 1  # Ejecutar
                ;;
            *)
                echo "Reutilizando ficheros existentes, saltando fase."
                return 0  # Saltar
                ;;
        esac
    else
        echo ""
        echo "Timeout — reutilizando ficheros existentes."
        return 0  # Saltar
    fi
}

# ============================================================================
# PASO 0: PREPARACION DEL SISTEMA
# ============================================================================

log_step "PASO 0/9: Preparacion del sistema — liberando memoria maxima"

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
# PASO 1: EXPORTAR ONNX
# ============================================================================

cd "$WORKDIR"

log_step "PASO 1/9: Exportar ONNX (RepVgg fusion + Int32 + static batch)"

if ask_skip "ONNX export" "$ONNX_RAW" "$ONNX_OUTPUT"; then
    log_ok "ONNX export saltado (ficheros existentes)"
else
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
fi

# ============================================================================
# PASO 2: BUILD TENSORRT FP32
# ============================================================================

FP32_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_fp32.engine"

log_step "PASO 2/9: Build TensorRT FP32"

if ask_skip "TensorRT FP32 engine" "$FP32_ENGINE"; then
    log_ok "Build FP32 saltado (engine existente)"
else
    drop_caches
    run_python references/deploy/build_engine.py \
        --onnx "$ONNX_OUTPUT" \
        --output "$FP32_ENGINE" \
        --workspace_mb "$WORKSPACE_MB" \
        --opt_level "$OPT_LEVEL" \
        --timing_cache "$TIMING_CACHE"

    check_file "$FP32_ENGINE"
fi

# ============================================================================
# PASO 3: BUILD TENSORRT FP16
# ============================================================================

FP16_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_fp16.engine"

log_step "PASO 3/9: Build TensorRT FP16"

if ask_skip "TensorRT FP16 engine" "$FP16_ENGINE"; then
    log_ok "Build FP16 saltado (engine existente)"
else
    drop_caches
    run_python references/deploy/build_engine.py \
        --onnx "$ONNX_OUTPUT" \
        --output "$FP16_ENGINE" \
        --fp16 \
        --workspace_mb "$WORKSPACE_MB" \
        --opt_level "$OPT_LEVEL" \
        --timing_cache "$TIMING_CACHE"

    check_file "$FP16_ENGINE"
fi

# ============================================================================
# PASO 4: BUILD TENSORRT INT8 (entropy) — nvidia-modelopt explicit Q/DQ
# ============================================================================

INT8_ENTROPY_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_int8_entropy.engine"

log_step "PASO 4/9: Build TensorRT INT8 (entropy, modelopt Q/DQ)"

if ask_skip "TensorRT INT8 entropy engine" "$INT8_ENTROPY_ENGINE"; then
    log_ok "Build INT8 entropy saltado (engine existente)"
else
    drop_caches
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
fi

# ============================================================================
# PASO 5: BUILD TENSORRT INT8 (minmax) — nvidia-modelopt explicit Q/DQ
# ============================================================================

INT8_MINMAX_ENGINE="${ENGINE_DIR}/${ENGINE_PREFIX}_int8_minmax.engine"

log_step "PASO 5/9: Build TensorRT INT8 (minmax, modelopt Q/DQ)"

if ask_skip "TensorRT INT8 minmax engine" "$INT8_MINMAX_ENGINE"; then
    log_ok "Build INT8 minmax saltado (engine existente)"
else
    drop_caches
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
fi

# ============================================================================
# RESUMEN DE BUILDS
# ============================================================================

log_step "Resumen de modelos generados"
echo ""
ls -lhS "${ENGINE_DIR}/${ENGINE_PREFIX}"*.engine "${ENGINE_DIR}/${ENGINE_PREFIX}"*.onnx 2>/dev/null || echo "No se encontraron modelos"
echo ""

# ============================================================================
# PASO 6: EVALUACIONES ESTANDAR (5 modelos con eval.py)
# ============================================================================

EVAL_FAILED=0

# --- EVAL 1: ONNX ---
log_step "PASO 6/9: Evaluacion ONNX (1/5)"

if ask_skip "Evaluacion ONNX" "${EVAL_DIR}/onnx/evaluation_summary.txt"; then
    log_ok "Evaluacion ONNX saltada (resultados existentes)"
else
    drop_caches
    if run_python tools/eval.py -c "$CONFIG" --onnx "$ONNX_OUTPUT" -d cuda --output_dir "${EVAL_DIR}/onnx"; then
        log_ok "Evaluacion ONNX completada"
    else
        log_fail "Evaluacion ONNX fallo"
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi
fi

# --- EVAL 2: TensorRT FP32 ---
log_step "Evaluacion TensorRT FP32 (2/5)"

if ask_skip "Evaluacion TRT FP32" "${EVAL_DIR}/trt_fp32/evaluation_summary.txt"; then
    log_ok "Evaluacion TRT FP32 saltada (resultados existentes)"
else
    drop_caches
    if [ -f "$FP32_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$FP32_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_fp32"; then
        log_ok "Evaluacion TRT FP32 completada"
    else
        log_warn "Evaluacion TRT FP32 saltada o fallo"
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi
fi

# --- EVAL 3: TensorRT FP16 ---
log_step "Evaluacion TensorRT FP16 (3/5)"

if ask_skip "Evaluacion TRT FP16" "${EVAL_DIR}/trt_fp16/evaluation_summary.txt"; then
    log_ok "Evaluacion TRT FP16 saltada (resultados existentes)"
else
    drop_caches
    if [ -f "$FP16_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$FP16_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_fp16"; then
        log_ok "Evaluacion TRT FP16 completada"
    else
        log_warn "Evaluacion TRT FP16 saltada o fallo"
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi
fi

# --- EVAL 4: TensorRT INT8 entropy ---
log_step "Evaluacion TensorRT INT8 entropy (4/5)"

if ask_skip "Evaluacion TRT INT8 entropy" "${EVAL_DIR}/trt_int8_entropy/evaluation_summary.txt"; then
    log_ok "Evaluacion TRT INT8 entropy saltada (resultados existentes)"
else
    drop_caches
    if [ -f "$INT8_ENTROPY_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$INT8_ENTROPY_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_int8_entropy"; then
        log_ok "Evaluacion TRT INT8 entropy completada"
    else
        log_warn "Evaluacion TRT INT8 entropy saltada o fallo"
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi
fi

# --- EVAL 5: TensorRT INT8 minmax ---
log_step "Evaluacion TensorRT INT8 minmax (5/5)"

if ask_skip "Evaluacion TRT INT8 minmax" "${EVAL_DIR}/trt_int8_minmax/evaluation_summary.txt"; then
    log_ok "Evaluacion TRT INT8 minmax saltada (resultados existentes)"
else
    drop_caches
    if [ -f "$INT8_MINMAX_ENGINE" ] && run_python tools/eval.py -c "$CONFIG" --tensorrt "$INT8_MINMAX_ENGINE" -d cuda --output_dir "${EVAL_DIR}/trt_int8_minmax"; then
        log_ok "Evaluacion TRT INT8 minmax completada"
    else
        log_warn "Evaluacion TRT INT8 minmax saltada o fallo"
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi
fi

# ============================================================================
# PASO 7: EVALUACIONES OPTIMIZADAS (CUDA graphs + GPU timing + power)
# ============================================================================

log_step "PASO 7/9: Evaluaciones optimizadas — CUDA graphs, latencia GPU, consumo"

OPTIMIZED_SCRIPT="references/deploy/rtdetrv2_tensorrt_optimized.py"

# Verificar que existen los ficheros necesarios
if [ ! -f "$OPTIMIZED_SCRIPT" ]; then
    log_warn "Script optimizado no encontrado: $OPTIMIZED_SCRIPT — saltando"
elif [ ! -f "$EVAL_JSON" ]; then
    log_warn "Anotaciones no encontradas: $EVAL_JSON — saltando evaluaciones optimizadas"
elif [ ! -d "$EVAL_IMG_DIR" ]; then
    log_warn "Directorio de imagenes no encontrado: $EVAL_IMG_DIR — saltando evaluaciones optimizadas"
else
    # Lista de engines a evaluar: nombre_corto engine_path
    declare -a OPT_ENGINES=(
        "trt_fp32:${FP32_ENGINE}"
        "trt_fp16:${FP16_ENGINE}"
        "trt_int8_entropy:${INT8_ENTROPY_ENGINE}"
        "trt_int8_minmax:${INT8_MINMAX_ENGINE}"
    )

    OPT_IDX=0
    OPT_TOTAL=${#OPT_ENGINES[@]}

    for entry in "${OPT_ENGINES[@]}"; do
        OPT_IDX=$((OPT_IDX + 1))
        name="${entry%%:*}"
        engine_path="${entry#*:}"
        out_dir="${EVAL_OPT_DIR}/${name}"

        echo ""
        echo -e "${CYAN}--- Evaluacion optimizada ${OPT_IDX}/${OPT_TOTAL}: ${name} ---${NC}"

        if [ ! -f "$engine_path" ]; then
            log_warn "Engine no existe: $engine_path — saltando $name"
            continue
        fi

        if ask_skip "Evaluacion optimizada ${name}" "${out_dir}/performance_stats.json"; then
            log_ok "Evaluacion optimizada ${name} saltada (resultados existentes)"
            continue
        fi

        drop_caches

        if run_python "$OPTIMIZED_SCRIPT" \
            --engine_file "$engine_path" \
            --eval_json "$EVAL_JSON" \
            --eval_root_dir "$EVAL_IMG_DIR" \
            --output_dir "$out_dir" \
            --threshold 0.001; then
            log_ok "Evaluacion optimizada ${name} completada"
        else
            log_warn "Evaluacion optimizada ${name} fallo"
            EVAL_FAILED=$((EVAL_FAILED + 1))
        fi
    done
fi

# ============================================================================
# PASO 8: RESUMEN COMPARATIVO
# ============================================================================

log_step "PASO 8/9: Resumen comparativo — Evaluacion estandar"

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

# ============================================================================
# PASO 9: RESUMEN RENDIMIENTO OPTIMIZADO
# ============================================================================

log_step "PASO 9/9: Resumen rendimiento optimizado (CUDA graphs + GPU timing)"

echo ""
printf "%-22s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s\n" \
    "MODELO" "mAP" "FPS(wall)" "FPS(GPU)" "Lat.p95" "Lat.p99" "Power(W)"
printf "%-22s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s\n" \
    "----------------------" "----------" "----------" "----------" "----------" "----------" "----------"

for d in "${EVAL_OPT_DIR}"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    json_file="${d}performance_stats.json"

    if [ ! -f "$json_file" ]; then
        printf "%-22s  %-10s\n" "$name" "(sin datos)"
        continue
    fi

    # Extraer metricas del JSON con python
    read -r ap fps_wall fps_gpu p95 p99 power <<< "$(run_python -c "
import json, sys
with open('$json_file') as f:
    d = json.load(f)
cm = d.get('coco_metrics', {})
wt = d.get('wall_time_ms', {})
gt = d.get('gpu_time_ms', {})
fp = d.get('fps', {})
gm = d.get('gpu_monitoring', {})
pw = gm.get('avg_power_w', 'N/A')
pw_str = f'{pw:.1f}' if isinstance(pw, (int, float)) else 'N/A'
print(f\"{cm.get('AP', 0):.4f} {fp.get('from_wall_mean', 0):.1f} {fp.get('from_gpu_mean', 0):.1f} {wt.get('p95', 0):.2f} {wt.get('p99', 0):.2f} {pw_str}\")
" 2>/dev/null)" || true

    printf "%-22s  %-10s  %-10s  %-10s  %-10s  %-10s  %-10s\n" \
        "$name" "${ap:--}" "${fps_wall:--}" "${fps_gpu:--}" "${p95:--}" "${p99:--}" "${power:--}"
done

echo ""
echo "Resultados estandar en:   ${WORKDIR}/${EVAL_DIR}/*/"
echo "Resultados optimizados en: ${WORKDIR}/${EVAL_OPT_DIR}/*/"
echo ""

if [ "$EVAL_FAILED" -gt 0 ]; then
    log_warn "${EVAL_FAILED} evaluacion(es) fallaron. Revisar logs arriba."
else
    log_ok "Todas las evaluaciones completadas exitosamente!"
fi

echo ""
echo "Pipeline finalizado: $(date)"
