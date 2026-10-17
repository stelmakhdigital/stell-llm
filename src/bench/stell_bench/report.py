"""Запись отчётов по контракту PLAN.md §5: bench/results/<date>-<slug>/
(report.json + run.log). Локальные прогоны без модели — в bench/results/local/
(игнорируется git; финальные bench/results коммитятся)."""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESULTS = _REPO_ROOT / "bench" / "results"


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:32] or "bench"


def write_report(result, out_dir: Optional[Path] = None,
                 log_lines: Optional[list] = None) -> Path:
    """Пишет report.json (+ run.log при --report логе) в results-каталог."""
    d = Path(out_dir) if out_dir else _RESULTS
    d.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    (d / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if log_lines:
        (d / "run.log").write_text("\n".join(log_lines) + "\n")
    return d


def results_dir_for(slug: str) -> Path:
    date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    return _RESULTS / f"{date}-{_slugify(slug)}"


def results_local_dir_for(slug: str) -> Path:
    """Локальные прогоны (без GPU/модели) — в игнорируемом bench/results/local/."""
    return _RESULTS / "local" / results_dir_for(slug).name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stell_bench")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--engine-lib", default=None)
    parser.add_argument("--prompt", default="Hello, world!")
    parser.add_argument("--slug", default=None)
    parser.add_argument("--report", action="store_true",
                        help="писать run.log рядом с report.json")
    parser.add_argument("--pending", action="store_true", default=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_REPO_ROOT / "src" / "bench"))
    sys.path.insert(0, str(_REPO_ROOT / "python"))

    from stell_bench import BenchRunner, BenchmarkProtocol
    from stell_bench.report import results_dir_for, results_local_dir_for
    from stell_serve.app import default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(args.engine_lib)
    backend.create(default_engine_cfg(args.model))
    runner = BenchRunner(
        backend,
        BenchmarkProtocol(),
        pending=True,
        notes=["прогон без реальной модели: mock-движок, значения не GPU-честные"],
    )
    result = runner.run_benchmark(args.prompt)
    out = (results_local_dir_for(args.slug or f"{args.model}-local")
           if args.pending else results_dir_for(args.slug or f"{args.model}"))
    lines = [
        f"model={args.model} backend={getattr(backend, '_lib_path', '?')}",
        f"medians={json.dumps(result.medians)}",
        "pending=True (CPU-local, без GPU-модели)",
    ]
    write_report(result, out, log_lines=lines if args.report else None)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
