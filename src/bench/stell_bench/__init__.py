"""stell_bench — бенчмарк-харнесс по контракту PLAN.md §5.

Метрики раздельно: TTFT, prefill tok/s, target-only decode tok/s,
spec-decode tok/s, TPOT. Протокол single-request: T=1, top-p=0.95,
top-k=20, seed=0, до 256 выходных токенов. Медиана 3 тёплых прогонов
после 1 холодного. Фиксация: KV-dtype, batch, memory-utilization,
версия CUDA/драйвера, GPU-часы, co-tenancy. Результаты —
bench/results/<date>-<slug>/ с логом --report.
"""

from .aggregate import RunMetrics, median, aggregate_runs
from .protocol import BenchmarkProtocol, capture_env, BENCH_SESSION
from .harness import BenchRunner, run_benchmark
from .ppl_gate import SPEC_PPL_DELTA_FIELD, ppl_delta, ppl_from_logprobs
from .report import write_report

__all__ = ["RunMetrics", "median", "aggregate_runs", "BenchmarkProtocol",
           "capture_env", "BENCH_SESSION", "BenchRunner", "run_benchmark",
           "write_report", "SPEC_PPL_DELTA_FIELD", "ppl_delta",
           "ppl_from_logprobs"]
