#!/usr/bin/env bash
# bench_p95_nvme.sh — R7 (раунд 2 продолжение, роль tiers): замер p95-пауз
# слоёв на РЕАЛЬНОМ NVMe как отдельный шаг worker'а после merge.
#
# Что меряет: паузы pin_layer (ADR-08 §2 PauseStats) на контуре
# TierManager (VRAM-slots эмуляция + pinned RAM + NVMe io_uring, stell-tier
# layout ADR-12) — bench_tiers_pause (tests/tiers/bench_pause.cpp), режимы:
#   prefetch — router-lookahead: prefetch следующего слоя до паузы;
#   cold     — без lookahead: честная холодная загрузка NVMe→RAM→VRAM.
# Вердикт — цель PLAN.md §4: p95 ≤ 5 мс.
#
# Использование (на сервере, в клоне ~/Code/stell-llm ПОСЛЕ pull):
#   export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
#   cmake -S . -B build/cpu -G Ninja && cmake --build build/cpu
#   ./src/tiers/bench_p95_nvme.sh build/cpu ~/tiers_bench
#
# Аргументы: bench_p95_nvme.sh <build_dir> <data_dir>
#   build_dir — каталог сборки с bench_tiers_pause (дефолт build/cpu);
#   data_dir  — каталог ДЛЯ ФАЙЛА LAYOUT НА РЕАЛЬНОМ NVMe (дефолт /tmp —
#               ВНИМАНИЕ: /tmp может быть tmpfs! На сервере указывайте
#               каталог на корневом NVMe, напр. ~/tiers_bench).
#
# Env-флаги:
#   DROP_CACHE=1 — сбросить page cache перед прогонами (sync + drop_caches,
#                  нужен root): без этого чтения через buffered fd могут
#                  обслуживаться page cache → замер «тёплого» NVMe
#                  (фиксируется честно в отчёте).
#   PASSES, N_GROUPS, LAYERS — параметры прогона (см. ниже).
#
# Co-tenancy (spec): NVMe общий с серверной нагрузкой; strata на V100
# НЕ трогать. Перед прогоном печатается свободная VRAM (nvidia-smi), если
# доступна — для фиксации co-tenancy в отчёте.

set -euo pipefail

BUILD_DIR="${1:-build/cpu}"
DATA_DIR="${2:-/tmp/tiers_bench_nvme}"

N_GROUPS="${N_GROUPS:-32}"            # 32 группы = 16 слоёв × 2 группы (как в round-2 §4)
LAYERS="${LAYERS:-16}"
SLOT="${SLOT:-$((16*1024*1024))}" # 16 MiB слот (кратен 4096)
PASSES="${PASSES:-4}"
THRESHOLD_MS="5.0"                # цель PLAN.md §4: p95 ≤ 5 мс

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

drop_caches() {
    if [ "${DROP_CACHE:-0}" = "1" ]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches 2>/dev/null \
            && echo "page cache сброшен (честный холодный NVMe)" \
            || echo "drop_caches не удался (нужен root) — замер «тёплого» NVMe (page cache), пометить в отчёте"
    else
        echo "page cache НЕ сброшен (DROP_CACHE=1 чтобы сбросить; нужен root)"
    fi
}

run_mode() { # $1 = prefetch|cold, $2 = метка лога
    echo "== режим $1 ($2) =="
    "$BIN" "$LAYOUT" "$N_GROUPS" "$SLOT" "$LAYERS" "$PASSES" "$1" 2>&1 \
        | grep -E 'bench-side pause|^bench_pause' | tee "$2"
}

# Warmup: 1 проход — прогрев page cache и CRC-once (ADR-12 §2.3: CRC
# считается один раз за жизнь файла в кеше tiers, чтобы не попадать в p95).
"$BIN" "$LAYOUT" "$N_GROUPS" "$SLOT" "$LAYERS" 1 prefetch >/dev/null
echo "warmup: layout $LAYOUT готов (${N_GROUPS} групп × ${SLOT} байт)"
echo

drop_caches
run_mode prefetch "$DATA_DIR/p95_prefetch.log"

drop_caches
run_mode cold "$DATA_DIR/p95_cold.log"

# Вердикт по bench-side p95 (независимый двойной замер, не stats-снапшот).
P95_PREFETCH=$(grep 'bench-side pause' "$DATA_DIR/p95_prefetch.log" | grep -oE 'p95=[0-9.]+' | cut -d= -f2 | head -1)
P95_COLD=$(grep 'bench-side pause' "$DATA_DIR/p95_cold.log" | grep -oE 'p95=[0-9.]+' | cut -d= -f2 | head -1)
echo
echo "================= ВЕРДИКТ R7 (цель PLAN.md §4: p95 ≤ ${THRESHOLD_MS} мс) ================="
printf 'p95 prefetch: %s мс\np95 cold:     %s мс\n' "${P95_PREFETCH:-?}" "${P95_COLD:-?}"
VERDICT="FAIL"
# Вердикт — по prefetch-режиму (рабочая конфигурация engine с
# router-lookahead); cold приводится рядом честно (холодная пауза —
# деградация, не steady-state).
if [ -n "${P95_PREFETCH:-}" ] && awk -v a="${P95_PREFETCH}" -v t="$THRESHOLD_MS" 'BEGIN { exit !(a+0 <= t+0) }'; then
    VERDICT="PASS"
fi
echo "verdict: p95 (prefetch) ${VERDICT} (≤ ${THRESHOLD_MS} мс)"
echo "Помнить при интерпретации: co-tenancy NVMe/GPU (см. выше), page cache"
echo "(DROP_CACHE=1 — честный холодный прогон), CRC-once — только в warmup/проходе 1."
exit 0
