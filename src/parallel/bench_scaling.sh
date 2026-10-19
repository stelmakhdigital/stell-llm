#!/usr/bin/env bash
# bench_scaling.sh — [R8] раунд 3 (поток C, parallel): scaling-бенчмарк
# TP1/TP2/TP4 dense по контракту PLAN.md §5 (ориентир TP4 ≥3.2× от TP1).
#
# Что меряет: latency allreduce F32 (медиана по 100 итерациям) на матрице
# размеров 1 … 262144 floats на TP1/TP2/TP4 через NCCL-backend
# (бинарник test_parallel_gpu, bench-фаза оркестратора [R3]).
# Метрики:
#   - латентность allreduce по размерам (худший ранок — консервативно);
#   - «allreduce на токен»: латентность allreduce вектора d_model
#     (D_MODEL floats; dense TP: до 2 allreduce на слой на токен);
#   - scaling-фактор TP1/TPn по каждому размеру.
# Вердикт по ориентиру §5 (TP4 ≥3.2×) — ИТОГОВО считается на end-to-end
# decode (tok/s); этот скрипт даёт allreduce-компоненту, что честно
# помечается в отчёте.
#
# Использование (на сервере, В GIT-КЛОНЕ — протокол server-env «прогоны —
# только через git-клон», scp-артефакты запрещены; strata НЕ трогать):
#   export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
#   cmake -S . -B build/gpu -G Ninja -DSTELL_CUDA=ON && cmake --build build/gpu
#   ./src/parallel/bench_scaling.sh build/gpu bench/results
#
# Окно исполнения: ≥4 V100-SXM2 без помех (после освобождения сервера).
# До окна — скрипт пишет ЧЕСТНЫЙ pending с причиной (ко-тенантность,
# недоступные карты, отсутствие бинарника) в отчёт и завершается с 0;
# результаты не выдумываются. Замеры на RTX 2060 (GPU4) или смешанных
# парах без NVLink с целями §5 НЕ сравниваются (server-env).
#
# Env-переменные:
#   STELL_SCALING_TP1/TP2/TP4 — карты шагов (дефолт 0 / 0,1 / 0,1,2,3);
#   D_MODEL                   — ширина «токена» для метрики на токен (4096);
#   STELL_SCALING_TIMEOUT_S   — внешний тайм-аут шага (600с; страховка —
#                               внутри теста есть свой watchdog [R3]).

set -euo pipefail

BUILD_DIR="${1:-build/gpu}"
RESULTS_DIR="${2:-bench/results}"
D_MODEL="${D_MODEL:-4096}"
STEP_TIMEOUT_S="${STELL_SCALING_TIMEOUT_S:-600}"
THRESHOLD="3.2"    # ориентир PLAN.md §5: TP4 ≥3.2× от TP1

# --- Прогоны — только через git-клон (протокол server-env) -------------------
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: текущий каталог не git-клон. Прогоны — только через git-клон"
    echo "  (server-env [R6]: scp-копирование артефактов запрещено)."
    exit 2
fi

# --- Бинарник GPU-теста (сборка STELL_CUDA=ON + libnccl) ----------------------
BIN=""
for cand in "$BUILD_DIR/test_parallel_gpu" "$BUILD_DIR/tests/parallel/test_parallel_gpu"; do
    [ -x "$cand" ] && BIN="$cand" && break
done
if [ -z "$BIN" ]; then
    mkdir -p "$RESULTS_DIR"
    REPORT="$RESULTS_DIR/scaling_tp_$(date +%Y%m%d_%H%M%S).md"
    {
        echo "# Scaling-бенчмарк TP1/TP2/TP4 dense — [R8] round 3 (T-20261017200354)"
        echo
        echo "## ВЕРДИКТ: PENDING (честно, с причиной)"
        echo
        echo "Причина: бинарник test_parallel_gpu не найден в \`$BUILD_DIR\` —"
        echo "сборка STELL_CUDA=ON + libnccl не выполнена. Исполнение отложено, не молчание:"
        echo "  cmake -S . -B $BUILD_DIR -G Ninja -DSTELL_CUDA=ON && cmake --build $BUILD_DIR"
    } > "$REPORT"
    echo "PENDING: нет test_parallel_gpu в $BUILD_DIR — причина записана в $REPORT"
    exit 0
fi

TP1_DEV="${STELL_SCALING_TP1:-0}"
TP2_DEV="${STELL_SCALING_TP2:-0,1}"
TP4_DEV="${STELL_SCALING_TP4:-0,1,2,3}"

mkdir -p "$RESULTS_DIR"
REPORT="$RESULTS_DIR/scaling_tp_$(date +%Y%m%d_%H%M%S).md"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Co-tenancy фиксация (GPU заняты чужим процессом strata? — в отчёт).
COTENANCE="$(nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo 'nvidia-smi недоступен')"
echo "co-tenancy: $COTENANCE"

# Разбор лога шага: NCCL-BENCH строки → size<TAB>латентность худшего ранка.
# ВНИМАНИЕ: формат «size=%6zu» содержит пробелы-заполнители внутри токена,
# поэтому size извлекается из всей строки, а не по позициям полей.
parse_tp() { # $1 = лог, $2 = tsv-выход
    awk '
        /^NCCL-BENCH: size=/ {
            size = $0;
            sub(/^.*size= */, "", size);   # всё до числа
            sub(/[^0-9].*/, "", size);     # хвост после числа
            max = 0;
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^rank[0-9]+=/) {
                    v = $i; sub(/^[^=]*=/, "", v); sub(/[^0-9.].*/, "", v);
                    if (v + 0 > max + 0) max = v + 0;
                }
            }
            if (max > 0 && size != "") printf "%s\t%.1f\n", size, max;
        }' "$1" | sort -n > "$2"
}

# Один шаг матрицы: честный pending при любой неудаче (co-tenancy и т.п.).
run_step() { # $1 = имя шага (tp1|tp2|tp4), $2 = карты
    local name="$1" devs="$2" log="$WORK/$1.log"
    echo "== шаг $name (карты $devs) =="
    local rc=0
    if command -v timeout >/dev/null 2>&1; then
        STELL_GPU_DEVICES="$devs" STELL_GPU_BENCH=1 \
            timeout "$STEP_TIMEOUT_S" "$BIN" >"$log" 2>&1 || rc=$?
    else
        STELL_GPU_DEVICES="$devs" STELL_GPU_BENCH=1 \
            "$BIN" >"$log" 2>&1 || rc=$?
    fi
    if [ "$rc" != 0 ]; then
        echo "pending:$name: прогон завершился аварийно (exit=$rc — co-tenancy/OOM/зависание); хвост лога:" | tee "$WORK/$name.status"
        tail -5 "$log"
        return 1
    fi
    parse_tp "$log" "$WORK/$name.tsv"
    if [ ! -s "$WORK/$name.tsv" ]; then
        echo "pending:$name: латентностей в логе нет (bench-фаза отложена самим тестом — co-tenancy)" > "$WORK/$name.status"
        tail -5 "$log"
        return 1
    fi
    echo "pending:$name: ок ($(wc -l < "$WORK/$name.tsv") размеров)" > "$WORK/$name.status"
    return 0
}

TP1_OK=1; TP2_OK=1; TP4_OK=1
run_step tp1 "$TP1_DEV" || TP1_OK=0
run_step tp2 "$TP2_DEV" || TP2_OK=0
run_step tp4 "$TP4_DEV" || TP4_OK=0

# --- Таблицы шагов (только реально измеренное) -------------------------------
TABLES=""
SCALING_BLOCK=""
SCALING_ROWS=""
TOKEN_BLOCK=""
for s in tp1 tp2 tp4; do
    if [ -s "$WORK/$s.tsv" ]; then
        TABLES="$TABLES
### шаг $s — латентность allreduce (худший ранок), µs

\`\`\`
size_floats   lat_worst_us
$(awk -F'\t' '{ printf "%12s  %14.1f\n", $1, $2 }' "$WORK/$s.tsv")
\`\`\`
"
    fi
done

# --- Scaling-факторы TP1→TP4 (по общим размерам) ------------------------------
SCALING_BLOCK=""
TOKEN_BLOCK=""
if [ "$TP1_OK" = 1 ] && [ "$TP4_OK" = 1 ]; then
    SCALING_BLOCK="$(join -t "$(printf '\t')" "$WORK/tp1.tsv" "$WORK/tp4.tsv")"
    if [ -n "$SCALING_BLOCK" ]; then
        SCALING_ROWS="$(printf '%s\n' "$SCALING_BLOCK" | awk -F'\t' -v th="$THRESHOLD" '
            BEGIN { ok = 0; n = 0 }
            {
                lat1 = $2; lat4 = $3;
                r = (lat1 > 0) ? lat1 / lat4 : 0;
                n++;
                if (r + 0 >= th + 0) ok++;
                printf "%12s  %12.1f  %12.1f  %9.2f\n", $1, lat1, lat4, r;
            }
            END { printf "  (ориентир ≥%sх: на %.0f%% общих размеров)\n", th, (n > 0) ? ok * 100 / n : 0 }')"
        # «allreduce на токен»: вектор d_model.
        if grep -q "^${D_MODEL}$(printf '\t')" "$WORK/tp1.tsv" 2>/dev/null; then
            L1="$(awk -F'\t' -v dm="$D_MODEL" '$1==dm { print $2 }' "$WORK/tp1.tsv")"
            TOKEN_BLOCK="«allreduce на токен» (вектор d_model=$D_MODEL floats; до 2×/слой):
- TP1: ${L1} µs"
            for s in tp2 tp4; do
                if [ -s "$WORK/$s.tsv" ] && grep -q "^${D_MODEL}$(printf '\t')" "$WORK/$s.tsv" 2>/dev/null; then
                    LN="$(awk -F'\t' -v dm="$D_MODEL" '$1==dm { print $2 }' "$WORK/$s.tsv")"
                    FACT="$(awk -v a="$L1" -v b="$LN" 'BEGIN { printf "%.2f", (b > 0) ? a / b : 0 }')"
                    TOKEN_BLOCK="$TOKEN_BLOCK
- TP${s#tp}: ${LN} µs (фактор ${FACT}× от TP1)"
                fi
            done
        fi
    fi
fi

# --- Отчёт ---------------------------------------------------------------------
{
    echo "# Scaling-бенчмарк TP1/TP2/TP4 dense — [R8] round 3 (T-20261017200354)"
    echo
    echo "- дата: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- клон: \`$(git rev-parse --show-toplevel)\` @ \`$(git rev-parse --short HEAD)\`"
    echo "- co-tenancy: \`$COTENANCE\`"
    echo "- шаги: TP1=$TP1_DEV, TP2=$TP2_DEV, TP4=$TP4_DEV; D_MODEL=$D_MODEL (floats/токен);"
    echo "  тайм-аут шага ${STEP_TIMEOUT_S}с; латентность = худший ранок, медиана 100 итераций"
    echo "- сравнение с целями §5 допустимо ТОЛЬКО на V100-SXM2-парах по NVLink"
    echo "  в окне без co-tenancy (server-env; RTX 2060/смешанные пары — не сравнивать)"
    echo
    if [ "$TP1_OK" = 0 ] || [ "$TP2_OK" = 0 ] || [ "$TP4_OK" = 0 ]; then
        echo "## ВЕРДИКТ: PENDING (честно, с причиной)"
        echo
        echo "Исполнение отложено до ≥4 свободных V100-SXM2 без помех (strata держит"
        echo "3 из 4 V100). Результаты не выдумываются; измеренные шаги приведены честно."
        echo
        for s in tp1 tp2 tp4; do
            st="$(cat "$WORK/$s.status" 2>/dev/null || echo "статус неизвестен")"
            echo "- $s: $st"
        done
        echo
    fi
    [ -n "$TABLES" ] && { printf '%s\n' "$TABLES"; echo; }
    if [ -n "$SCALING_ROWS" ]; then
        echo "### scaling TP1 → TP4 (по общим размерам), ориентир §5 ≥${THRESHOLD}×"
        echo
        echo '```'
        printf '%12s  %12s  %12s  %9s\n' "size_floats" "lat_tp1" "lat_tp4" "factor"
        printf '%s\n' "$SCALING_ROWS"
        echo '```'
        echo
    fi
    if [ -n "$TOKEN_BLOCK" ]; then
        printf '%s\n' "$TOKEN_BLOCK"
        echo
    fi
    if [ "$TP1_OK" = 1 ] && [ "$TP2_OK" = 1 ] && [ "$TP4_OK" = 1 ]; then
        echo "## ВЕРДИКТ: ИЗМЕРЕНО (co-tenancy-пометка обязательна)"
        echo
        echo "Ориентир §5 (TP4 dense ≥3.2× от TP1) — ИТОГОВЫЙ вердикт считается на"
        echo "end-to-end decode (tok/s); выше — allreduce-компонента."
        echo
    fi
    echo "Скрипт: src/parallel/bench_scaling.sh (только git-клон; strata не трогается;"
    echo "зависимости: nvidia-smi/awk/join/timeout — user-level)."
} > "$REPORT"

echo
echo "отчёт: $REPORT"
if [ "$TP1_OK" = 0 ] || [ "$TP2_OK" = 0 ] || [ "$TP4_OK" = 0 ]; then
    echo "итог: ЧЕСТНЫЙ PENDING (причина в отчёте) — исполнение отложено до окна"
else
    echo "итог: шаги измерены — см. $REPORT (вердикт §5 — на end-to-end decode)"
fi
exit 0
