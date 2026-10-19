#!/usr/bin/env bash
# bench_p95_nvme.sh — [R2] (раунд 3, роль tiers): cold-замер p95-пауз слоёв
# на РЕАЛЬНОМ NVMe. Cold-эвикция page cache — userspace-механизм
# posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) по файлам тира (planner [Q3]):
# unprivileged, blast radius = только файлы тира (не трогает page cache
# co-tenancy-соседей, в отличие от глобального drop_caches).
#
# Путь sudo//proc/sys/vm/drop_caches УДАЛЁН (spec: только user-level;
# DROP_CACHE мог оставить только помеченный fallback «тёплый замер» —
# см. DROP_CACHE_WARM ниже).
#
# ============ ТОЧНАЯ КОМАНДА ПРОГОНА (worker, ПОСЛЕ [R1] и merge) ============
#   # 0) окружение (в клоне ~/Code/stell-llm после git pull):
#   export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
#   export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
#   cmake -S . -B build/cpu -G Ninja && cmake --build build/cpu
#   # 1) прогон (файл layout — на корневом NVMe; ~/tiers_bench — СВОЯ dir):
#   ./src/tiers/bench_p95_nvme.sh build/cpu ~/tiers_bench
#   #    опционально под CUDA-конфигурацией: build/cuda вместо build/cpu
#   #    (DMA-путь в контуре bench; VRAM-тир — единицы MiB, co-tenancy
#   #    безопасно; strata на V100 НЕ трогать).
#   # 2) результат: лог + JSON-сводка в $DATA_DIR (см. поля ниже).
# ============================================================================
#
# ОЖИДАЕМЫЕ ПОЛЯ ОТЧЁТА (verdict-секция скрипта; сохранить лог в
# bench/results/ вместе с p95_*.log):
#   - p50 / p95 / p99 пауз слоёв (bench-side двойной замер, режимы
#     prefetch и cold), мс;
#   - verdict: достигнута ли цель p95 ≤ 5 мс (PLAN.md §4) — по режиму
#     prefetch (рабочая конфигурация engine с router-lookahead);
#   - профиль остатка без CRC-once (raw-паузы по проходам: проход 1 —
#     холодное NVMe-чтение, проходы 2+ — RAM/VRAM-хиты; CRC из hot path
#     удалён в [R1] — если p95 > 5 мс, остаток доминируется чтением/
#     DMA/memcpy, а не CRC);
#   - пометка co-tenancy: NVMe и GPU общие с серверной нагрузкой
#     (strata держит 21–30 ГБ VRAM — НЕ трогать); замер — в окне с помехами,
#     если nvidia-smi показывает занятую VRAM.
#
# Использование: bench_p95_nvme.sh <build_dir> <data_dir>
#   build_dir — каталог сборки с bench_tiers_pause (дефолт build/cpu);
#   data_dir  — каталог ДЛЯ ФАЙЛА LAYOUT НА РЕАЛЬНОМ NVMe (дефолт /tmp —
#               ВНИМАНИЕ: /tmp может быть tmpfs! Указывайте каталог на
#               корневом NVMe, напр. ~/tiers_bench).
#
# Env-флаги:
#   VERIFY_CRC=1 — прогнать дополнительный проход с STELL_TIER_VERIFY_CRC=1
#                  (сравнение «CRC-once в hot path» vs дефолт off, [R1]).
#   DROP_CACHE_WARM=1 — fallback: пометить прогон «тёплый (page cache)»
#                  явно (сброс кеша требует root и затрагивает весь сервер —
#                  удалён по spec; флаг оставлен только для честной пометки
#                  тёплого замера в отчёте).
#   PASSES, N_GROUPS, LAYERS, SLOT — параметры прогона (ниже).

set -euo pipefail

BUILD_DIR="${1:-build/cpu}"
DATA_DIR="${2:-/tmp/tiers_bench_nvme}"

N_GROUPS="${N_GROUPS:-32}"            # 32 группы = 16 слоёв × 2 группы
LAYERS="${LAYERS:-16}"
SLOT="${SLOT:-$((16*1024*1024))}"     # 16 MiB слот (кратен 4096)
PASSES="${PASSES:-4}"
THRESHOLD_MS="5.0"                    # цель PLAN.md §4: p95 ≤ 5 мс

BIN=""
for cand in "$BUILD_DIR/bench_tiers_pause" "$BUILD_DIR/tests/tiers/bench_tiers_pause"; do
    [ -x "$cand" ] && BIN="$cand" && break
done
[ -n "$BIN" ] || { echo "ERROR: нет bench_tiers_pause в $BUILD_DIR — соберите: cmake --build $BUILD_DIR"; exit 2; }

mkdir -p "$DATA_DIR"
LAYOUT="$DATA_DIR/pause_layout.stelltier"

# Честная пометка носителя: layout обязан лежать на реальном NVMe.
FSType=$(df --output=fstype "$DATA_DIR" 2>/dev/null | tail -1)
if [ "$FSType" = "tmpfs" ] || [ "$FSType" = "ramfs" ]; then
    echo "ВНИМАНИЕ: $DATA_DIR — $FSType (RAM), НЕ реальный NVMe!"
    echo "  Укажите каталог на NVMe (напр. ~/tiers_bench) — иначе замер некорректен."
    exit 2
fi
echo "носитель: $DATA_DIR ($FSType) — должен быть реальным NVMe"

# Co-tenancy фиксация: свободная VRAM/чужие процессы (если GPU видны).
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "co-tenancy: $(nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null | head -4 || echo 'nvidia-smi недоступен')"
fi

# --- cold-эвикция page cache: posix_fadvise(POSIX_FADV_DONTNEED) по файлам
# тира (unprivileged; [Q3] planner round 2, зафиксировано в [R2] round 3).
EVICT_MODE="cold (posix_fadvise DONTNEED)"
evict_tier_pages() {
    if [ "${DROP_CACHE_WARM:-0}" = "1" ]; then
        EVICT_MODE="тёплый (page cache): fallback-пометка DROP_CACHE_WARM"
        return 1
    fi
    if command -v python3 >/dev/null 2>&1; then
        if python3 - "$LAYOUT" <<'PY' 2>/dev/null
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
PY
        then
            EVICT_MODE="cold (posix_fadvise POSIX_FADV_DONTNEED по файлу тира)"
            return 0
        fi
        echo "posix_fadvise не удался — прогон помечен «тёплый (page cache)»"
        EVICT_MODE="тёплый (page cache): fadvise недоступен"
        return 1
    fi
    echo "python3 недоступен — fadvise невозможен; прогон помечен «тёплый (page cache)»"
    EVICT_MODE="тёплый (page cache): нет python3 для fadvise"
    return 1
}

# Fallback-пометка «тёплый замер» (spec round 3: DROP_CACHE-путь удалён;
# осталась только честная пометка тёплого прогона — сам evict_tier_pages
# при DROP_CACHE_WARM=1 эвикию НЕ выполняет и помечает прогон тёплым).
if [ "${DROP_CACHE_WARM:-0}" = "1" ]; then
    echo "fallback: DROP_CACHE_WARM=1 — прогон помечен «тёплый (page cache)»"
    echo "  (сброс page cache через /proc/sys/vm/drop_caches требует root и"
    echo "   затрагивает весь сервер — удалён по spec; см. [Q3])"
fi

run_mode() { # $1 = prefetch|cold, $2 = метка лога
    echo "== режим $1 ($2) =="
    BENCH_DEBUG=1 "$BIN" "$LAYOUT" "$N_GROUPS" "$SLOT" "$LAYERS" "$PASSES" "$1" 2>&1 \
        | grep -E 'bench-side pause|raw pauses|^bench_pause' | tee "$2"
}

# Warmup: 1 проход — прогрев page cache И creation layout. CRC-once больше
# НЕ доминирует проход 1: payload-CRC из hot path удалён ([R1], дефолт off) —
# холодная пауза прохода 1 = честное NVMe-чтение (профиль остатка без CRC).
"$BIN" "$LAYOUT" "$N_GROUPS" "$SLOT" "$LAYERS" 1 prefetch >/dev/null
echo "warmup: layout $LAYOUT готов (${N_GROUPS} групп × ${SLOT} байт); CRC-once в hot path off ([R1])"

# --- cold-проход prefetch (цель p95 ≤ 5 мс — по этому режиму) ---------------
evict_tier_pages || true
echo "замер: $EVICT_MODE"
run_mode prefetch "$DATA_DIR/p95_prefetch.log"

# --- cold-проход cold (без lookahead: честная холодная пауза слоя) ----------
evict_tier_pages || true
run_mode cold "$DATA_DIR/p95_cold.log"

# --- опционально: verify=on — сравнение с CRC-once в hot path ([R1]) --------
if [ "${VERIFY_CRC:-0}" = "1" ]; then
    evict_tier_pages || true
    echo "== режим prefetch (STELL_TIER_VERIFY_CRC=1: CRC-once в hot path, для сравнения) =="
    STELL_TIER_VERIFY_CRC=1 BENCH_DEBUG=1 "$BIN" "$LAYOUT" "$N_GROUPS" "$SLOT" \
        "$LAYERS" "$PASSES" prefetch 2>&1 \
        | grep -E 'bench-side pause|^bench_pause' | tee "$DATA_DIR/p95_prefetch_crc_on.log"
fi

# Вердикт по bench-side p95 (независимый двойной замер, не stats-снапшот).
P50_PREFETCH=$(grep 'bench-side pause' "$DATA_DIR/p95_prefetch.log" | grep -oE 'p50=[0-9.]+' | cut -d= -f2 | head -1)
P95_PREFETCH=$(grep 'bench-side pause' "$DATA_DIR/p95_prefetch.log" | grep -oE 'p95=[0-9.]+' | cut -d= -f2 | head -1)
P99_PREFETCH=$(grep 'bench-side pause' "$DATA_DIR/p95_prefetch.log" | grep -oE 'p99=[0-9.]+' | cut -d= -f2 | head -1)
P95_COLD=$(grep 'bench-side pause' "$DATA_DIR/p95_cold.log" | grep -oE 'p95=[0-9.]+' | cut -d= -f2 | head -1)

echo
echo "================= ВЕРДИКТ [R2] (цель PLAN.md §4: p95 ≤ ${THRESHOLD_MS} мс) ================="
printf 'p50/p95/p99 prefetch: %s / %s / %s мс\np95 cold:             %s мс\n' \
    "${P50_PREFETCH:-?}" "${P95_PREFETCH:-?}" "${P99_PREFETCH:-?}" "${P95_COLD:-?}"
printf 'режим замера:         %s\n' "$EVICT_MODE"
VERDICT="FAIL"
# Вердикт — по prefetch-режиму (рабочая конфигурация engine с
# router-lookahead); cold приводится рядом честно (холодная пауза —
# деградация, не steady-state).
if [ -n "${P95_PREFETCH:-}" ] && awk -v a="${P95_PREFETCH}" -v t="$THRESHOLD_MS" 'BEGIN { exit !(a+0 <= t+0) }'; then
    VERDICT="PASS"
fi
echo "verdict: p95 (prefetch) ${VERDICT} (≤ ${THRESHOLD_MS} мс)"
echo
echo "--- обязательные поля отчёта в bench/results/ (co-tenancy-прогон) ---"
echo "  1) p50/p95/p99 (prefetch+cold) — выше;"
echo "  2) verdict против цели p95 ≤ 5 мс — ${VERDICT};"
echo "  3) профиль остатка БЕЗ CRC-once: raw-паузы в ${DATA_DIR}/p95_*.log"
echo "     (строка 'raw pauses'); CRC из hot path удалён ([R1]) — если p95 > 5 мс,"
echo "     остаток = NVMe-чтение + DMA/memcpy RAM→VRAM + планирование, НЕ CRC;"
echo "     для сравнения с CRC-once: VERIFY_CRC=1 → p95_prefetch_crc_on.log;"
echo "  4) co-tenancy-пометка: NVMe/GPU общие (strata НЕ трогать) — см. nvidia-smi выше."
exit 0
