#!/usr/bin/env bash
# R8 (round 5): обёртка серверного прогона quality-gates «реальная модель».
#
# Назначение ([R8] planner round 5, [Q1]): единый протокол запуска/перезапуска
# прогона на реальных весах с честным замером wall-времени до/после [R1]
# (движковая оптимизация positions:"last") — обе цифры фиксируются в артефакте.
# Замер — date-диффом (date +%s.%N до/после, разница awk'ом).
#
# Запуск (worker, на сервере, из корня репо):
#   bash src/bench/stell_bench/run_real_gates.sh
#
# Управление (env; все необязательны):
#   STELL_R8_MODEL_DIR     каталог модели (description.json ADR-09 + веса
#                          [+ tokenizer.json]); по умолчанию
#                          "$HOME/models/tiny-qwen2moe"; пустая строка →
#                          прогон без источника = честный pending runner'а.
#   STELL_R8_RESULTS_DIR   каталог артефакта; по умолчанию
#                          bench/results/<дата>-quality-gates-real-<имя модели>
#                          (имя — basename каталога модели, иначе "pending").
#   STELL_R8_TIMEOUT       wall-бюджет timeout(1) в секундах; по умолчанию 4800.
#                          Прогон до [R1] был УБИТ этим бюджетом (rc=124,
#                          ~4797 с, O(n²) движок) — см. REPORT-ROUND5.md.
#   STELL_R8_CO_TENANCY    1 → --co-tenancy (честная пометка совместной машины
#                          в provenance).
#   STELL_R8_ENGINE_LIB    путь к libstell_core.so (по умолчанию — выбор
#                          движка: resolve_engine_lib, build/cpu).
#   STELL_R8_WALL_PHASE    "before" | "after" — фаза относительно [R1]
#                          (попадает в provenance.wall_phase).
#   STELL_R8_WALL_BEFORE   wall-время прогона фазы "before" в секундах — для
#                          артефакта фазы "after" (обе цифры в provenance).
#   STELL_R8_DRY_RUN       1 → только напечатать итоговую команду и выйти.
#
# Перезапуск точки регрессии после перегенерации описания ([R2] round 6,
# протокол для worker на сервере):
#   Описание модели (description.json ADR-09) перегенерирует models/worker
#   с полем dtype_orig (полный provenance точек регрессии — правило
#   docs/design/server-env.md); побитовая сверка слотов/логитов до/после —
#   обязательна (конверсия не менялась, поле информационное). Затем точка
#   регрессии R8 перезапускается этой же обёрткой точечной командой:
#
#     STELL_R8_RESULTS_DIR=bench/results/20261020-quality-gates-real-r5\
#       STELL_R8_CO_TENANCY=1 STELL_R8_WALL_PHASE=after\
#       STELL_R8_WALL_BEFORE=4800.0\
#       bash src/bench/stell_bench/run_real_gates.sh
#
#   - STELL_R8_RESULTS_DIR закреплён за существующим каталогом артефакта
#     (иначе обёртка по умолчанию создаст каталог с сегодняшней датой);
#   - wall-поля сохраняются честно: wall_phase="after" и
#     wall_before_seconds=4800.0 (цифра фазы "before", убитой timeout'ом)
#     переезжают в новый provenance; wall_seconds нового прогона замеряется
#     заново (новый запуск — новая честная цифра, НЕ задним числом);
#   - dtype_orig в новом report.json будет заполнен из перегенерированного
#     описания; до перегенерации runner честно пишет dtype_orig=None плюс
#     пометку dtype_orig_note «описание перегенерировать» ([R2] round 6).
#   - co-tenancy-пометка сохраняется (--co-tenancy — машина осталась общей).
#
# Артефакты в STELL_R8_RESULTS_DIR:
#   report.json — штатный отчёт runner'а (пишется при завершении прогона);
#                 обёртка дополняет provenance wall-полями (wall_seconds,
#                 wall_phase, wall_before_seconds) — обе цифры до/после [R1]
#                 живут в самой точке регрессии;
#   wall.json   — provenance обёртки: wall_seconds, exit_code, timeout_s,
#                 killed_by_timeout (rc=124), model_dir, engine_lib,
#                 git_commit, hostname, date_utc, co_tenancy, wall_phase,
#                 wall_before_seconds. Пишется ВСЕГДА — в том числе при
#                 таймауте/ошибке: прогон до [R1] был убит по timeout БЕЗ
#                 report.json, поэтому wall-факт обязан переживать отказ.
#
# Веса в git не попадают: каталог модели — вне репо; в results-dir пишутся
# только отчёты (как и у runner'а).

set -u

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)   # src/bench/stell_bench → корень

model_dir="${STELL_R8_MODEL_DIR-$HOME/models/tiny-qwen2moe}"
dry_run="${STELL_R8_DRY_RUN:-0}"
results_dir="${STELL_R8_RESULTS_DIR:-}"
if [ -z "$results_dir" ]; then
    today=$(date +%Y%m%d)
    if [ -n "$model_dir" ] && [ -d "$model_dir" ]; then
        slug=$(basename "$model_dir")
        results_dir="$repo_root/bench/results/${today}-quality-gates-real-${slug}"
    else
        results_dir="$repo_root/bench/results/${today}-quality-gates-real-pending"
    fi
fi
if [ "$dry_run" != "1" ]; then
    mkdir -p "$results_dir"
fi

timeout_s="${STELL_R8_TIMEOUT:-4800}"
engine_lib="${STELL_R8_ENGINE_LIB:-}"

# --- итоговая команда runner'а -------------------------------------------
cmd=(python3 -m stell_bench.quality_gates_runner --real --live)
if [ -n "$model_dir" ]; then
    export STELL_R8_REAL_MODEL_DIR="$model_dir"
    cmd+=(--real-model-dir "$model_dir")
else
    unset STELL_R8_REAL_MODEL_DIR || true
fi
if [ -n "$engine_lib" ]; then
    cmd+=(--engine-lib "$engine_lib")
fi
co_tenancy=0
if [ "${STELL_R8_CO_TENANCY:-0}" = "1" ]; then
    cmd+=(--co-tenancy)
    co_tenancy=1
fi
wall_phase="${STELL_R8_WALL_PHASE:-}"
if [ -n "$wall_phase" ]; then
    cmd+=(--wall-phase "$wall_phase")
fi
wall_before="${STELL_R8_WALL_BEFORE:-}"
if [ -n "$wall_before" ]; then
    cmd+=(--wall-before-seconds "$wall_before")
fi
cmd+=(--results-dir "$results_dir")

if [ "${STELL_R8_DRY_RUN:-0}" = "1" ]; then
    printf 'PYTHONPATH=%s\n' "$repo_root/python:$repo_root/src/bench"
    printf 'timeout %s' "$timeout_s"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

# --- wall-замер (date-дифф) и запуск с бюджетом timeout(1) ---------------
export PYTHONPATH="$repo_root/python:$repo_root/src/bench"
t0=$(date +%s.%N)
timeout "$timeout_s" "${cmd[@]}" </dev/null
rc=$?
t1=$(date +%s.%N)

wall_seconds=$(awk "BEGIN{printf \"%.3f\", $t1 - $t0}")
[ "$rc" = "124" ] && killed_by_timeout=1 || killed_by_timeout=0

commit=$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || echo "не-git")
host=$(hostname 2>/dev/null || echo "unknown")
date_utc=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)

python3 - "$results_dir/wall.json" <<PYEOF
import json, sys

model_dir = "$model_dir" if "$model_dir" != "" else None
engine_lib = "$engine_lib" if "$engine_lib" != "" else None
wall_phase = "$wall_phase" if "$wall_phase" != "" else None
wall_before = None
if "$wall_before" != "":
    wall_before = float("$wall_before")

json.dump({
    "wrapper": "src/bench/stell_bench/run_real_gates.sh",
    "model_dir": model_dir,
    "engine_lib": engine_lib,
    "git_commit": "$commit",
    "hostname": "$host",
    "date_utc": "$date_utc",
    "co_tenancy": bool($co_tenancy),
    "timeout_s": int("$timeout_s"),
    "exit_code": $rc,
    "killed_by_timeout": bool($killed_by_timeout),
    "wall_seconds": float("$wall_seconds"),
    "wall_phase": wall_phase,
    "wall_before_seconds": wall_before,
    "results_dir": "$results_dir",
}, open(sys.argv[1], "w"), indent=2, ensure_ascii=False)

# Wall-поля честно вплетаются и в штатный report.json (коммитимый артефакт):
# provenance дополняется wall_seconds/wall_phase/wall_before_seconds —
# чтобы обе цифры до/после [R1] жили в самой точке регрессии. Если report.json
# нет (прогон убит timeout'ом до записи) — остаётся только wall.json (raw).
import os

report = os.path.join("$results_dir", "report.json")
wall_fields = {}
if float("$wall_seconds") is not None:
    wall_fields["wall_seconds"] = float("$wall_seconds")
if wall_phase:
    wall_fields["wall_phase"] = wall_phase
if wall_before is not None:
    wall_fields["wall_before_seconds"] = wall_before
if os.path.isfile(report) and wall_fields:
    try:
        with open(report, "r", encoding="utf-8") as f:
            doc = json.load(f)
        prov = doc.get("provenance") or {}
        prov.update(wall_fields)
        doc["provenance"] = prov
        with open(report, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    except Exception as e:  # честно: не смогли патчить — raw wall.json остаётся
        sys.stderr.write(f"wall-патч report.json не удался: {e}\n")
PYEOF

echo "R8 wrapper: rc=$rc, wall=${wall_seconds}с, артефакты: $results_dir"
exit "$rc"
