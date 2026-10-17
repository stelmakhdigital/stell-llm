"""Агрегация прогонов: медиана 3 тёплых после 1 холодного (PLAN.md §5)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class RunMetrics:
    ttft_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    spec_decode_tok_s: float
    tpot_ms: float
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


_MEDIAN_FIELDS = ("ttft_ms", "prefill_tok_s", "decode_tok_s",
                  "spec_decode_tok_s", "tpot_ms")


def aggregate_runs(runs: Iterable[RunMetrics]) -> Dict[str, float]:
    """Медианы по списку тёплых прогонов (холодный отбрасывается вызывающим)."""
    runs = list(runs)
    if not runs:
        raise ValueError("no warm runs to aggregate")
    out: Dict[str, float] = {f: median([getattr(r, f) for r in runs])
                             for f in _MEDIAN_FIELDS}
    out["prompt_tokens"] = median(
        [float(r.prompt_tokens) for r in runs]) if runs else 0.0
    out["completion_tokens"] = median(
        [float(r.completion_tokens) for r in runs]) if runs else 0.0
    return out
