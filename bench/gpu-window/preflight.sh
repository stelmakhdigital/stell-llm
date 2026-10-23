#!/usr/bin/env bash
# preflight.sh — диагностика окна GPU (ничего не измеряет, ничего не ломает).
# Выход: 0 — окно ЧИСТОЕ (можно мерить), 1 — окно занято/неполное (мерить
# нельзя: чужие процессы или мало карт), 2 — среда не готова (нет nvidia-smi).
# Снапшот окружения пишет в $1 (дефолт: ./gpu-window-snapshot).
set -uo pipefail

OUT="${1:-./gpu-window-snapshot}"
mkdir -p "$OUT"
S="$OUT/environment.txt"
: > "$S"

log() { echo "$*" | tee -a "$S"; }
snap() { echo "\$ $*" | tee -a "$S"; "$@" 2>&1 | tee -a "$S"; }

log "=== GPU-окно: снапшот окружения $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
snap git rev-parse HEAD
snap git status --short

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "PREFLIGHT: nvidia-smi недоступен — это не GPU-сервер?"
    exit 2
fi

snap nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
snap nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m > "$OUT/topology.txt" 2>&1 || log "topo -m недоступен"
nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1 || true

# --- Карты: ≥4 V100 (прочие карты — не участники, не мешают) ---------------
# Правило (по факту машины): не-V100 карты (RTX 2060 и т.п.) в окне НЕ
# участвуют и из критериев исключены — только V100-SXM2 считаются.
GPU_LIST=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null)
V100_IDX=$(echo "$GPU_LIST" | awk -F', ' '$2 ~ /V100/ {print $1}')
V100_COUNT=$(echo "$V100_IDX" | grep -c . || true)
OTHER=$(echo "$GPU_LIST" | awk -F', ' '$2 !~ /V100/ {print $1" ("$2")"}')
[ -n "$OTHER" ] && log "Карты-не-участники (исключены из критериев): $OTHER"
log "Карт V100 (участники): $V100_COUNT"
if [ "$V100_COUNT" -lt 4 ]; then
    log "PREFLIGHT: НЕДостаточно V100 (нужно ≥4 для TP4) — окно НЕЧИСТОЕ"
    exit 1
fi

# --- Чужие процессы на V100 (co-tenancy) ------------------------------------
# Учитываются ТОЛЬКО процессы на V100 (не-V100 карты игнорируются).
FOREIGN=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | grep -v "^\s*$" | while IFS=', ' read -r uuid pid pname mem; do
        idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
              | awk -F', ' -v u="$uuid" '$2 == u {print $1}')
        for vi in $V100_IDX; do [ "$idx" = "$vi" ] && echo "gpu$idx pid=$pid $pname ($mem)"; done
    done)
if [ -n "$FOREIGN" ]; then
    log "PREFLIGHT: на GPU ЕСТЬ вычислительные процессы — окно ЗАНЯТО:"
    log "$FOREIGN"
    log "ЧУЖУЮ ПАМЯТЬ НЕ ТРОГАТЬ (kill/очистка запрещены). Окно откроется"
    log "после их завершения. Замеры под co-tenancy честной пометкой НЕ снимают"
    log "pending GPU-верификации (сравнение с чистым окном некорректно)."
    exit 1
fi

# --- Свободная VRAM на каждой V100 ------------------------------------------
LOWFREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$2 < 30000 {print $1" ("$2" MiB свободно)"}' \
    | while read -r line; do
        idx=${line%% *}
        for vi in $V100_IDX; do [ "$idx" = "$vi" ] && echo "$line"; done
    done)
if [ -n "$LOWFREE" ]; then
    log "PREFLIGHT: карты со свободной VRAM < 30 GiB: $LOWFREE — окно НЕЧИСТОЕ"
    exit 1
fi

# --- NVMe под layout ---------------------------------------------------------
FREE_GB=$(df --output=avail -BG . 2>/dev/null | tail -1 | tr -dc '0-9')
[ -n "$FREE_GB" ] || FREE_GB=$(df -P . 2>/dev/null | tail -1 | awk '{print int($4/1048576)}')
log "Свободно на диске репо: ${FREE_GB} GiB"
if [ "${FREE_GB:-0}" -lt 20 ]; then
    log "PREFLIGHT: < 20 GiB на диске — конвертация layout (~16 GiB) может не пройти."
    log "Удалите старые layout (bench/results/*/logs/…stelltier) или расчистите место."
    exit 1
fi

# --- Сборка/тулчейн ----------------------------------------------------------
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -1 | tee -a "$S" \
    || log "nvcc не в PATH (сборка STELL_CUDA=ON потребует CUDA toolkit)"
ls /usr/lib/x86_64-linux-gnu/libnccl* >/dev/null 2>&1 || \
    find / -maxdepth 4 -name "libnccl.so*" 2>/dev/null | head -1 | grep -q . \
    || log "ВНИМАНИЕ: libnccl не найден быстро — bench_scaling проверит точнее"

log "PREFLIGHT: ОКНО ЧИСТОЕ — можно запускать run-gpu-window.sh"
exit 0
