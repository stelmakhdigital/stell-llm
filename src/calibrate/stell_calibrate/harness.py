"""Замер-харнесс калибровщика: A/B-замеры через бэкенд движка + таймер.

На CPU (без GPU) харнесс гоняется на синтетических замерах — юнит-тесты
политики не требуют железа. Реальные замеры (драфт on/off, KV F16/E4M3,
RAM/NVMe-чтение) выполняются тем же кодом на целевом сервере.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .config import CalibrationConfig
from .policy import CalibrationDecision, decide


@dataclass
class Measurement:
    name: str
    value: float
    unit: str = "tok/s"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "unit": self.unit,
                "notes": self.notes}


def timed_call(fn: Callable[[], Any], repeats: int = 3) -> float:
    """Среднее время (сек) по repeats запускам; первый запуск прогревочный."""
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


class CalibrationHarness:
    """Арбитр-харнесс: измеряет A/B и передаёт замеры политике."""

    def __init__(self, cfg: Optional[CalibrationConfig] = None,
                 measurements: Optional[Dict[str, float]] = None) -> None:
        self.cfg = cfg or CalibrationConfig()
        self.measurements: Dict[str, float] = dict(measurements or {})

    def measure(self, name: str, fn: Callable[[], Any],
                unit: str = "tok/s", scale: float = 1.0,
                repeats: int = 3) -> Measurement:
        """Измеряет fn (среднее по тёплым повторам) и регистрирует замер."""
        elapsed = timed_call(fn, repeats)
        value = scale / elapsed if elapsed > 0 else 0.0
        self.measurements[name] = value
        return Measurement(name, value, unit)

    def calibrate(self) -> CalibrationDecision:
        return decide(self.cfg, self.measurements)
