"""Прогон бенчмарка через бэкенд движка (ADR-11): холодный + тёплые прогоны.

Runner измеряет по каждому прогону: TTFT (submit → первый token),
prefill tok/s (prompt_tokens / TTFT), target-only decode tok/s,
spec-decode tok/s (из снапшота engine.features.spec_draft=True — иначе 0),
TPOT. На CPU без GPU прогоны на mock честно помечаются pending=true.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .aggregate import RunMetrics, aggregate_runs
from .protocol import BenchmarkProtocol, capture_env


@dataclass
class BenchResult:
    medians: Dict[str, float]
    runs: List[RunMetrics]
    env: Dict[str, Any]
    protocol: Dict[str, Any]
    pending: bool = True          # True = прогоны без реальной модели/GPU
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "medians": self.medians,
            "runs": [r.to_dict() for r in self.runs],
            "env": self.env,
            "protocol": self.protocol,
            "pending": self.pending,
            "notes": self.notes,
        }


class BenchRunner:
    """Харнесс: один успешный старт на версию (§5), 1 холодный + N тёплых."""

    def __init__(self, backend, protocol: Optional[BenchmarkProtocol] = None,
                 pending: bool = True, notes: Optional[List[str]] = None) -> None:
        self.backend = backend
        self.protocol = protocol or BenchmarkProtocol()
        self.protocol.validate()
        self.pending = pending
        self.notes = notes or []

    def run_single(self, prompt: str) -> RunMetrics:
        """Single-request прогон: T=1, top-p=0.95, top-k=20, seed=0 (§5)."""
        request = {
            "session": dict(self.protocol.session),
            "messages": [{"role": "user", "content": prompt}],
            "prompt": prompt,
            "stream": False,
        }
        t0 = time.perf_counter()
        sid = self.backend.submit(request)
        ttft_ms = None
        prompt_tokens = 0
        completion_tokens = 0
        empty_streak = 0
        finish = None
        while empty_streak < 32:
            events = self.backend.poll(sid)
            if not events:
                empty_streak += 1
                continue
            empty_streak = 0
            for ev in events:
                if ev.get("type") == "finish":
                    finish = ev
                if ev.get("type") == "token" and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
        if finish is None:
            raise RuntimeError("bench run: no finish event")
        usage = finish.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_ms = (time.perf_counter() - t0) * 1000.0
        ttft = ttft_ms if ttft_ms is not None else 0.0
        decode_ms = max(total_ms - ttft, 0.0)
        prefill_tok_s = (prompt_tokens / (ttft / 1000.0)) if ttft > 0 else 0.0
        decode_tok_s = (completion_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        tpot_ms = (decode_ms / completion_tokens) if completion_tokens else 0.0
        spec_tok_s = 0.0
        try:
            snap = self.backend.metrics()
            if snap.get("features", {}).get("spec_draft"):
                spec_tok_s = snap.get("latency", {}).get("tok_s", {}).get("spec_decode", 0.0)
        except Exception:  # метрики не обязательны для прогона
            pass
        return RunMetrics(
            ttft_ms=ttft, prefill_tok_s=prefill_tok_s, decode_tok_s=decode_tok_s,
            spec_decode_tok_s=spec_tok_s, tpot_ms=tpot_ms,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        )

    def run_benchmark(self, prompt: str) -> BenchResult:
        """1 холодный (дискерт) + 3 тёплых, медианы по тёплым (§5)."""
        self.run_single(prompt)  # холодный
        warm = [self.run_single(prompt) for _ in range(self.protocol.warm_runs)]
        medians = aggregate_runs(warm)
        env = capture_env(gpu_hours=None)
        env["co_tenancy"] = self.protocol.co_tenancy
        return BenchResult(
            medians=medians, runs=warm, env=env,
            protocol={
                "name": self.protocol.name,
                "session": dict(self.protocol.session),
                "cold_runs": self.protocol.cold_runs,
                "warm_runs": self.protocol.warm_runs,
                "aggregation": self.protocol.aggregation,
            },
            pending=self.pending,
            notes=list(self.notes),
        )


def run_benchmark(backend, prompt: str,
                  protocol: Optional[BenchmarkProtocol] = None,
                  pending: bool = True) -> BenchResult:
    return BenchRunner(backend, protocol, pending).run_benchmark(prompt)
