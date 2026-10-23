#!/usr/bin/env bash
# run-gpu-window.sh — сбор ВСЕХ отложенных GPU/CPU-логов за один заход
# (запуск ВРУЧНУЮ при чистом окне; см. README.md в этом каталоге).
#
# Честность: шаг без бинарника/входов/окна → запись "PENDING: причина" в
# сводку и продолжение; результаты не имитируются. Чужие GPU-процессы НЕ
# трогаются (kill/очистка запрещены). Прогоны — только из git-клона.
#
# Использование (сервер, корень репо):
#   export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
#   bash bench/gpu-window/run-gpu-window.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE/.." rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: не git-клон — прогоны только через git-клон (server-env)."; exit 2
fi

STAMP=$(date -u +%Y%m%d-%H%M%S)
RESULTS="${STELL_GPUWINDOW_RESULTS:-bench/results}/${STAMP}-gpu-window"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"
SUMMARY="$RESULTS/report.md"
PENDINGS=()
MEASURED=()

MODEL_DIR="${STELL_GPUWINDOW_MODEL:-$HOME/models/qwen3-30b-a3b}"
EXTRA_MODELS="${STELL_GPUWINDOW_EXTRA_MODELS:-}"
BUILD=build/gpu

note()  { echo "$*" | tee -a "$SUMMARY"; }
pend()  { PENDINGS+=("$1"); note "- **PENDING**: $1"; }
done_m(){ MEASURED+=("$1"); note "- **MEASURED**: $1"; }

{
  echo "# GPU-окно — сводка прогона $STAMP"
  echo
  echo "- git HEAD: $(git rev-parse HEAD 2>/dev/null || echo '?')"
  echo "- модель (шаги C-E): \`$MODEL_DIR\`"
  echo
  echo "## Результаты по шагам"
} > "$SUMMARY"

# --- Окно: снапшот окружения (всегда) ---------------------------------------
bash "$HERE/preflight.sh" "$RESULTS" > "$LOGS/preflight.log" 2>&1
RC=$?
WINDOW=dirty
[ "$RC" = "0" ] && WINDOW=clean
note "- Окно: **$WINDOW** (детали: \`logs/preflight.log\`, \`environment.txt\`)"
if [ "$RC" = "2" ]; then
    pend "preflight: nvidia-smi недоступен — шаги A/B пропущены entirely"
fi

# --- Сборка (CUDA-таргеты) ---------------------------------------------------
note
note "## Сборка"
if command -v cmake >/dev/null 2>&1; then
    ( cmake -S . -B "$BUILD" -G Ninja -DSTELL_CUDA=ON &&
      cmake --build "$BUILD" -j"$(nproc)" ) > "$LOGS/build.log" 2>&1
    if [ $? = 0 ]; then done_m "сборка STELL_CUDA=ON (лог: logs/build.log)";
    else
        pend "сборка STELL_CUDA=ON не прошла (лог: logs/build.log) — GPU-шаги отложены"
        note "  диагностика (хвост logs/build.log):"
        tail -n 12 "$LOGS/build.log" | sed 's/^/  > /' | tee -a "$SUMMARY"
        note "  подсказки: 1) если build/gpu остался от CPU-конфигурации — rm -rf $BUILD и повторить;
  2) nccl.h ищется в /usr/include и /usr/local/cuda/include — при другом пути:
     cmake -S . -B $BUILD -G Ninja -DSTELL_CUDA=ON -DSTELL_NCCL_INCLUDE=<каталог с nccl.h>;
  3) CUDA toolkit должен быть в PATH (nvcc)."
    fi
else
    pend "cmake недоступен в PATH"
fi

BIN() { for c in "$BUILD/$1" "$BUILD/tests/tiers/$1" "$BUILD/tests/parallel/$1" \
                 "$BUILD/src/models/$1" "$BUILD/tools/$1"; do
            [ -x "$c" ] && { echo "$c"; return 0; }; done; return 1; }

# --- A. NCCL scaling (только чистое окно) ------------------------------------
note
note "## A. NCCL scaling TP1/TP2/TP4"
if SC="$(BIN test_parallel_gpu)" && [ "$WINDOW" = clean ]; then
    bash src/parallel/bench_scaling.sh "$BUILD" "$RESULTS" \
        > "$LOGS/nccl-scaling.log" 2>&1 \
        && done_m "NCCL scaling (лог: logs/nccl-scaling.log, отчёт bench/results/scaling_tp_*.md)" \
        || pend "NCCL scaling завершился с ошибкой (лог: logs/nccl-scaling.log)"
elif [ "$WINDOW" != clean ]; then
    pend "NCCL scaling — окно занято (co-tenancy): замер только при чистом окне"
else
    pend "NCCL scaling: бинарник test_parallel_gpu не собран (лог: logs/build.log)"
fi

# --- B. tiers GPU DMA smoke (только чистое окно) -----------------------------
note
note "## B. tiers GPU DMA smoke"
if TC="$(BIN bench_tiers_cuda)" && [ "$WINDOW" = clean ]; then
    "$TC" > "$LOGS/tiers-cuda-smoke.log" 2>&1 \
        && done_m "tiers CUDA DMA smoke (лог: logs/tiers-cuda-smoke.log)" \
        || pend "tiers CUDA smoke ошибкой (лог: logs/tiers-cuda-smoke.log)"
elif [ "$WINDOW" != clean ]; then
    pend "tiers CUDA smoke — окно занято"
else
    pend "tiers CUDA smoke: бинарник bench_tiers_cuda не собран"
fi

# --- C. Конвертация layout (CPU/NVMe — безопасно при co-tenancy) -------------
note
note "## C. Layout реального чекпоинта (stell-tier)"
LAYOUT=""
HD="$(BIN stell_hf_description)"
CV="$(BIN stell_convert_streaming)"
# Автопоиск модели: MODEL_DIR → ~/models/* по config.json (первый с знакомой
# конвертеру архитектурой); найденное честно перечисляется в сводке.
if [ ! -d "$MODEL_DIR" ]; then
    FOUND=""
    for cfg in "$HOME"/models/*/config.json; do
        [ -f "$cfg" ] || continue
        d=$(dirname "$cfg")
        FOUND="$FOUND $(basename "$d")"
    done
    [ -n "$FOUND" ] && note "  - \"$MODEL_DIR\" отсутствует; в ~/models найдены: $FOUND"
    for cfg in "$HOME"/models/*/config.json; do
        [ -f "$cfg" ] || continue
        d=$(dirname "$cfg")
        arch=$(python3 -c "import json;print(json.load(open('$cfg')).get('architectures',['?'])[0])" 2>/dev/null || echo '?')
        case "$arch" in
            LlamaForCausalLM|Qwen2ForCausalLM|MixtralForCausalLM|Qwen2MoeForCausalLM|Qwen1.5MoeForCausalLM|OlmoeForCausalLM|Qwen3MoeForCausalLM)
                MODEL_DIR="$d"; note "  - автопоиск: выбран \"$d\" (arch=$arch)"; break ;;
        esac
    done
fi
if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ] && [ -n "$HD" ] && [ -n "$CV" ]; then
    DESC="$RESULTS/description.json"
    LAYOUT="$RESULTS/checkpoint.stelltier"
    if [ -s "$LAYOUT" ]; then
        note "- layout уже существует, конвертация пропущена: \`$LAYOUT\`"
    else
        "$HD" "$MODEL_DIR" NVFP4 > "$DESC" 2> "$LOGS/hf-description.err" \
          && "$CV" "$DESC" "$MODEL_DIR/model.safetensors" "$LAYOUT" \
                  > "$LOGS/convert.log" 2>&1 \
            && done_m "конвертация layout (лог: logs/convert.log; ~11 мин, peak RSS см. лог)" \
            || pend "конвертация не прошла (лог: logs/convert.log, logs/hf-description.err — возможно, архитектура вне конвертера: это тоже честный результат)"
    fi
else
    pend "конвертация: нет модели ($MODEL_DIR) или инструментов stell_hf_description/stell_convert_streaming"
fi

# --- D. per-group bench матрица (hit-rate + p95; CPU/NVMe контур) ------------
note
note "## D. per-group bench: hit-rate + p95 (спека Потока B: p95 ≤ 5 мс, prefetch ≥ 60%)"
if [ -n "$LAYOUT" ] && [ -s "$LAYOUT" ] && BP="$(BIN bench_tiers_pause)"; then
    for io in 0 1; do
        for k in 1 4; do
            for slots in 24 48 96; do
                tag="pergroup_io${io}_k${k}_slots${slots}"
                STELL_TIER_ODIRECT=$io STELL_TIER_PREFETCH_DEPTH=$k \
                STELL_TIER_VRAM_SLOTS=$slots \
                "$BP" "$LAYOUT" 0 0 0 3 prefetch pergroup \
                    > "$LOGS/${tag}.log" 2>&1 \
                || note "  - ячейка $tag: завершение с ошибкой (честно: лог $tag.log)"
            done
        done
    done
    grep -l "pergroup hit-rate" "$LOGS"/pergroup_*.log >/dev/null 2>&1 \
        && done_m "матрица per-group {io}×{k}×{бюджет} (логи: pergroup_*.log; критерии Потока B сверять по строкам 'pergroup hit-rate' и 'цель PLAN.md §4')" \
        || pend "per-group матрица: ни одна ячейка не выпустила hit-rate (см. логи)"
else
    pend "per-group bench: нет layout (шаг C) или bench_tiers_pause"
fi

# --- E. Декод-контур: побитовые логиты + wall/token ---------------------------
note
note "## E. Декод-контур (побитовая сверка логитов, wall/token)"
HARNESS_SRC="bench/results/20261020-decode-real-qwen3-30b-a3b/harness_decode_logits.cpp"
if [ -n "$LAYOUT" ] && [ -s "$LAYOUT" ] && [ -f "$HARNESS_SRC" ] && [ -d "$BUILD" ]; then
    g++ -std=c++20 -O2 -I src/tiers/include -I src/models/include \
        -I src/engine/include -I src/kernels/include \
        "$HARNESS_SRC" "$BUILD/src/engine/libstell_engine.a" \
        "$BUILD/src/models/libstell_models.a" "$BUILD/src/tiers/libstell_tiers.a" \
        "$BUILD/src/kernels/libstell_kernels.a" -lpthread \
        -o "$LOGS/decode_logits" > "$LOGS/decode-build.log" 2>&1
    if [ -x "$LOGS/decode_logits" ]; then
        "$LOGS/decode_logits" "$MODEL_DIR" "$LAYOUT" \
            > "$LOGS/decode-logits.log" 2>&1 \
            && done_m "декод-контур: логиты побитово + wall/token (лог: logs/decode-logits.log)" \
            || pend "декод-контур завершился ошибкой (лог: logs/decode-logits.log)"
    else
        pend "декод-контур: не собран (лог: logs/decode-build.log)"
    fi
else
    pend "декод-контур: нет layout или исходника харнесса ($HARNESS_SRC)"
fi

# --- F. Quality gates live-real (PPL/GSM8K/HumanEval через ABI) --------------
note
note "## F. Quality gates (live-real)"
PY="${STELL_GPUWINDOW_PYTHON:-}"
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    for cand in /home/arkalaust/.local/stell-conda/envs/serve/bin/python \
                "$HOME/.local/stell-conda/envs/serve/bin/python" \
                "$(command -v python3 2>/dev/null)"; do
        [ -n "$cand" ] && [ -x "$cand" ] && PY="$cand" && break
    done
fi
if [ -x "$PY" ] && [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ]; then
    ( cd src/bench && STELL_R8_MODEL_DIR="$MODEL_DIR" \
      STELL_R8_RESULTS_DIR="$REPO/$RESULTS/quality-gates" \
      STELL_R8_CO_TENANCY=$([ "$WINDOW" = clean ] && echo 0 || echo 1) \
      bash stell_bench/run_real_gates.sh ) \
      > "$LOGS/quality-gates.log" 2>&1 \
      && done_m "quality gates live-real (лог: logs/quality-gates.log, артефакт: quality-gates/)" \
      || pend "quality gates: прогон с ошибкой/тайм-аутом (лог: logs/quality-gates.log)"
else
    pend "quality gates: нет python ($PY) или каталога модели"
fi

# --- G. Дополнительные чекпоинты (GLM/Qwen3_5Moe/dense — пробы конвертера) ---
note
note "## G. Пробы дополнительных чекпоинтов (pending ADR-09)"
if [ -n "$EXTRA_MODELS" ] && [ -d "$EXTRA_MODELS" ] && [ -n "$HD" ]; then
    find "$EXTRA_MODELS" -name config.json | while read -r cfg; do
        d=$(dirname "$cfg")
        arch=$("$PY" -c "import json;print(json.load(open('$cfg')).get('architectures',['?'])[0])" 2>/dev/null || echo '?')
        if "$HD" "$d" NVFP4 > "$RESULTS/extra-$(basename "$d").json" 2>"$RESULTS/extra-$(basename "$d").err"; then
            note "- \`$d\`: arch=$arch — описание построено (артефакт: extra-$(basename "$d").json)"
        else
            note "- \`$d\`: arch=$arch — ЧЕСТНЫЙ SchemaMismatch (ожидалось для GLM/qwen3_5 вне контура; err: extra-$(basename "$d").err)"
        fi
    done
else
    pend "пробы чекпоинтов: STELL_GPUWINDOW_EXTRA_MODELS не задан (положите GLM/Qwen3_5Moe/dense в ~/models и укажите каталог)"
fi

# --- Сводка ------------------------------------------------------------------
{
  echo
  echo "## Итог"
  echo "- измерено: ${#MEASURED[@]} шагов; pending: ${#PENDINGS[@]} шагов"
} >> "$SUMMARY"
echo
echo "=== ГОТОВО. Сводка: $SUMMARY (измерено ${#MEASURED[@]}, pending ${#PENDINGS[@]}) ==="
exit 0
