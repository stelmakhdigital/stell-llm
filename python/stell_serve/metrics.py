"""Serve-side метрики и логи (ADR-11 §4): счётчики, наблюдаемые TTFT/TPOT,
tok/s, кольцевой буфер логов для дашборда. Численные форматы — p50/p95/p99
как в ADR-08 §2."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Строгий формат logits-снапшота /metrics (ADR-11 §4, round 3, [R5] —
# решение планировщика [Q5]): статистика вызовов `stell_logits` БЕЗ payload
# (массивы логитов в /metrics не передаются никогда — размер). Ровно 4 поля;
# дашборд рендерит строго эти 4 KV.
LOGITS_SNAPSHOT_FIELDS = (
    "requests_total",     # всего вызовов stell_logits
    "errors_total",       # из них ошибочных (out_json с {"error"…}, rc=-1)
    "last_n_vocab",       # n_vocab последнего вызова
    "last_positions",     # n_positions последнего вызова
)


def is_valid_logits_snapshot(obj: Any) -> bool:
    """Проверка строгого формата logits-снапшота ([R5]): ровно 4 поля
    LOGITS_SNAPSHOT_FIELDS; счётчики — числа, `last_*` — числа или `null`
    (до первого успешного вызова — «честно: нет данных», ADR-11 §4 round 3;
    payload-массивы/объекты в /metrics запрещены контрактом)."""
    if not (isinstance(obj, dict) and set(obj) == set(LOGITS_SNAPSHOT_FIELDS)):
        return False
    for key, value in obj.items():
        if key in ("requests_total", "errors_total"):
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
                return False
        else:  # last_n_vocab, last_positions — число или null
            if value is not None and not (isinstance(value, (int, float, str))
                                          and not isinstance(value, bool)):
                return False
    return True


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
