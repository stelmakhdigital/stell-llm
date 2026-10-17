"""Runner калибровщика на CPU-замерах (round 2, задача calibrate↔bench).

Прогоняет арбитр (policy.py) на замерах, снятых через mock-движок (C ABI
ADR-11), и пишет отчёт в bench/results/local/ (gitignored). Честность:

- декод-таймер — синтетическая модель декода (sleep по заявленной tpot
  mock-движка, замеренная timed_call через ABI-контур submit/poll/step);
  это НЕ GPU-честные числа: они проверяют логику A/B-арбитра, а не железо;
- draft_accept_rate, PPL (f16/e4m3), ram/nvme-чтение, max_group_bytes —
  синтетические входы (нет модели/весов; ABI ADR-11 не отдаёт логитов —
  расхождение контракта, зафиксировано в отчёте round 2);
- результат помечен pending=True до появления реальной модели на GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]

MAX_POLL_ROUNDS = 1024


@dataclass
class CalibrationRun:
    decision: Dict[str, Any]
    measurements: Dict[str, float]
    notes: List[str]
    pending: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision, "measurements": self.measurements,
                "notes": self.notes, "pending": self.pending}


def decode_run(backend, tpot_ms: float, n_tokens: int = 16,
               prompt: str = "calibrate-measurement") -> int:
    """Один декод-прогон через движок (C ABI, ADR-11 §1..§3): submit → poll
    до finish; на каждый token — синтетическая пауза tpot_ms (модель декода).
    Возвращает число декодированных токенов."""
    request = {
        "session": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                    "seed": 0, "max_tokens": n_tokens, "stop": []},
        "prompt": prompt,
        "stream": False,
        "mock": {"token_count": n_tokens},  # mock: ровно n токенов → finish stop
    }
    sid = backend.submit(request)
    count = 0
    empty = 0
    rounds = 0
    while rounds < MAX_POLL_ROUNDS:
        rounds += 1
        chunk = backend.poll(sid)
        if not chunk:
            empty += 1
            backend.step()  # real: шаг декода; mock: no-op
            if empty >= 32:
                break
            continue
        empty = 0
        for ev in chunk:
            if ev.get("type") == "token":
                time.sleep(tpot_ms / 1000.0)  # синтетическая пауза декода
                count += 1
            if ev.get("type") == "finish":
                return count
    raise RuntimeError("calibration decode run: no finish event")


def measure_tok_s(backend, tpot_ms: float, n_tokens: int = 16,
                  repeats: int = 3) -> float:
    """tok/s по тёплым повторам (timed_call из round 1): warmup + repeats."""
    from .harness import timed_call

    elapsed = timed_call(lambda: decode_run(backend, tpot_ms, n_tokens),
                         repeats=repeats)
    return n_tokens / elapsed if elapsed > 0 else 0.0


def make_backend(lib_path: Optional[str], tpot_ms: float):
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(lib_path)
    backend.create({
        "model": "mock-model",
        "mock": {"ttft_ms": 1.0, "tpot_ms": tpot_ms},
    })
    return backend


def run_calibration(lib_path: Optional[str] = None,
                    n_tokens: int = 16) -> CalibrationRun:
    """A/B-замеры фич (спек-драфт/KV-квант/NVMe-тир) → решение арбитра."""
    from stell_calibrate.config import CalibrationConfig
    from stell_calibrate.policy import decide

    notes: List[str] = [
        "CPU-прогон на mock-движке (C ABI ADR-11): таймер декода — "
        "синтетическая модель (sleep по заявленной tpot mock), НЕ GPU-честно",
        "draft_accept_rate, PPL f16/e4m3, ram/nvme-чтение, max_group_bytes — "
        "синтетические входы: нет модели/весов; ABI ADR-11 не отдаёт логитов "
        "(расхождение контракта — зафиксировано в отчёте round 2)",
        "решение арбитра валидно как логика вкл/выкл по замерам; значения "
        "на реальном железе — задача прогона bench на целевом сервере",
    ]

    measurements: Dict[str, float] = {}

    # --- декод: базовый контур (tpot 14 мс mock) ----------------------------
    base = make_backend(lib_path, 14.0)
    try:
        measurements["decode_tok_s"] = round(measure_tok_s(
            base, 14.0, n_tokens), 3)
    finally:
        base.close()

    # --- спек-драфт: контур с драфтером (tpot 10 мс mock) -------------------
    spec = make_backend(lib_path, 10.0)
    try:
        measurements["spec_decode_tok_s"] = round(measure_tok_s(
            spec, 10.0, n_tokens), 3)
    finally:
        spec.close()
    measurements["draft_accept_rate"] = 0.7  # синтетический вход (нет драфтера)

    # --- KV-квант E4M3 vs F16 (tpot 12 мс против базового 14 мс) ------------
    measurements["decode_tok_s_kv_f16"] = measurements["decode_tok_s"]
    kv = make_backend(lib_path, 12.0)
    try:
        measurements["decode_tok_s_kv_e4m3"] = round(measure_tok_s(
            kv, 12.0, n_tokens), 3)
    finally:
        kv.close()
    # PPL — синтетический вход (нет логитов в ABI)
    measurements["ppl_kv_f16"] = 10.0
    measurements["ppl_kv_e4m3"] = 10.05

    # --- NVMe-тир и граница тиров (синтетические входы) ---------------------
    measurements["ram_read_tok_s"] = 200.0
    measurements["nvme_read_tok_s"] = 80.0
    measurements["max_group_bytes"] = 3 * (1 << 30)

    decision = decide(CalibrationConfig(), measurements).to_dict()
    return CalibrationRun(decision=decision, measurements=measurements,
                          notes=notes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stell_calibrate")
    parser.add_argument("--engine-lib", default=None)
    parser.add_argument("--results-dir", default=None,
                        help="переопределить каталог отчёта (для тестов)")
    parser.add_argument("--n-tokens", type=int, default=16)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_REPO_ROOT / "src" / "calibrate"))
    sys.path.insert(0, str(_REPO_ROOT / "python"))
    sys.path.insert(0, str(_REPO_ROOT / "src" / "bench"))

    run = run_calibration(args.engine_lib, n_tokens=args.n_tokens)

    if args.results_dir:
        out_dir = Path(args.results_dir)
    else:
        from stell_bench.report import results_local_dir_for

        out_dir = Path(results_local_dir_for("calibrate-local"))
    from stell_bench.report import write_report

    write_report(run, out_dir, log_lines=[
        f"calibrate: decision={json.dumps(run.decision)}",
        f"measurements={json.dumps(run.measurements)}",
        "pending=True (CPU-mock, синтетические входы; см. notes)",
    ])
    print(json.dumps(run.to_dict(), indent=2, ensure_ascii=False))
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
