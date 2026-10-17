"""Runner quality-gates (PLAN.md §5): GSM8K-32, HumanEval-32, PPL (round 2).

Round 2: датасеты РЕАЛЬНЫЕ (см. stell_bench.fetch_quality_gates —
фиксированная выборка 32 задач seed=42 из открытых датасетов OpenAI), но
прогона нет — честный pending:

- нет модели/весов (реестр и лоадеры — поток D, runner поверх реальных
  весов — поток A);
- ABI ADR-11 round 1 не отдаёт логитов (нужны для PPL) и полных генераций
  для проверки HumanEval/GSM8K — расхождение контракта, зафиксировано в
  отчёте round 2; до появления символа логитов в ABI гейты не вычислимы.

Запуск: python -m stell_bench.quality_gates_runner [--engine-lib PATH]
[--results-dir DIR]; отчёт — bench/results/local/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .quality_gates import GateResult, gsm8k_32, humaneval_32, ppl_text
from .report import results_local_dir_for, write_report

_ABI_DIVERGENCE = ("ABI ADR-11 round 1 не отдаёт логитов (PPL) и полных "
                   "генераций для проверки — расхождение контракта, "
                   "зафиксировано в отчёте round 2; гейты вычислимы только "
                   "после расширения ABI (новый символ — ABI-совместимо) и "
                   "появления весов/модели")


class _GatesSnapshot:
    """Обёртка отчёта write_report: снапшот гейтов (pending-статус + значения)."""

    def __init__(self, gates: List[GateResult]) -> None:
        self.gates = gates

    def to_dict(self) -> Dict[str, Any]:
        return {"pending": all(g.pending for g in self.gates),
                "gates": [g.to_dict() for g in self.gates]}


_NO_MODEL = "нет реальной модели/весов (реестр/лоадеры — поток D, runner — поток A)"


def run_gates() -> List[GateResult]:
    """Снимок гейтов: датасеты реальны, значения — честный pending (round 2)."""
    return [
        GateResult("quality-gate", "gsm8k_32", pending=True, value=None,
                   items=len(gsm8k_32()), unit="score",
                   notes=[_NO_MODEL, _ABI_DIVERGENCE]),
        GateResult("quality-gate", "humaneval_32", pending=True, value=None,
                   items=len(humaneval_32()), unit="score",
                   notes=[_NO_MODEL, _ABI_DIVERGENCE]),
        GateResult("quality-gate", "ppl_fixed_text", pending=True, value=None,
                   items=len(ppl_text().split()), unit="ppl",
                   notes=[_NO_MODEL, _ABI_DIVERGENCE]),
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stell_bench.quality_gates_runner")
    parser.add_argument("--engine-lib", default=None,
                        help="libstell_core.so (реальный движок или mock)")
    parser.add_argument("--results-dir", default=None,
                        help="переопределить каталог отчёта (для тестов)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "bench"))
    gates = run_gates()

    snapshot = _GatesSnapshot(gates)
    out_dir = (Path(args.results_dir) if args.results_dir
               else Path(results_local_dir_for("quality-gates-local")))
    write_report(snapshot, out_dir, log_lines=[
        "quality-gates: датасеты реальные (32 задачи, seed=42); прогон — "
        "pending=True (нет модели/весов; ABI без логитов — см. notes)",
    ])
    print(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False))
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
