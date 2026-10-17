"""Serve-side метрики и логи (ADR-11 §4): счётчики, наблюдаемые TTFT/TPOT,
tok/s, кольцевой буфер логов для дашборда. Численные форматы — p50/p95/p99
как в ADR-08 §2."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _percentile(sorted_values: List[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return round(sorted_values[idx], 3)


class ServeMetrics:
    """Состояние control-plane: счётчики запросов, наблюдаемая латентность,
    лог-кольцо (дашборд рендерит 1:1)."""

    def __init__(self, log_size: int = 100) -> None:
        self.started = time.monotonic()
        self.requests_total = 0
        self.errors_total = 0
        self.cancelled_total = 0
        self.ttft_ms: List[float] = []
        self.tpot_ms: List[float] = []
        self.decode_tok_s: List[float] = []
        self.logs: deque = deque(maxlen=log_size)

    def log(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.logs.append(f"{stamp} {message}")

    def record_request(
        self,
        ttft_ms: float,
        tpot_ms: float,
        decode_tok_s: float,
        completion_tokens: int,
    ) -> None:
        self.requests_total += 1
        self.ttft_ms.append(ttft_ms)
        self.tpot_ms.append(tpot_ms)
        self.decode_tok_s.append(decode_tok_s)

    def record_error(self) -> None:
        self.errors_total += 1

    def record_cancelled(self) -> None:
        self.cancelled_total += 1

    def snapshot(self) -> Dict[str, Any]:
        ttft = sorted(self.ttft_ms)
        tpot = sorted(self.tpot_ms)
        tps = sorted(self.decode_tok_s)
        return {
            "requests_total": self.requests_total,
            "errors_total": self.errors_total,
            "cancelled_total": self.cancelled_total,
            "uptime_s": round(time.monotonic() - self.started, 3),
            "latency": {
                "ttft_ms": {"p50": _percentile(ttft, 0.5),
                            "p95": _percentile(ttft, 0.95),
                            "p99": _percentile(ttft, 0.99)},
                "tpot_ms": {"p50": _percentile(tpot, 0.5),
                            "p95": _percentile(tpot, 0.95),
                            "p99": _percentile(tpot, 0.99)},
                "decode_tok_s": {"p50": _percentile(tps, 0.5),
                                 "p95": _percentile(tps, 0.95),
                                 "p99": _percentile(tps, 0.99)},
            },
            "logs": list(self.logs),
        }
