"""Фиксированный протокол измерений (PLAN.md §5) + снапшот окружения."""

from __future__ import annotations

import datetime
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# single-request контракт PLAN.md §5
BENCH_SESSION = {
    "temperature": 1.0,  # T=1 (допущение зафиксировано: T=1 — temperature 1.0)
    "top_p": 0.95,
    "top_k": 20,
    "seed": 0,
    "max_tokens": 256,   # до 256 выходных токенов
    "stop": [],
}


@dataclass
class BenchmarkProtocol:
    name: str = "plan5-single-request"
    session: Dict[str, Any] = field(default_factory=lambda: dict(BENCH_SESSION))
    cold_runs: int = 1      # один холодный прогон (дискерт)
    warm_runs: int = 3      # медиана 3 тёплых прогонов
    aggregation: str = "median"
    max_output_tokens: int = 256
    co_tenancy: bool = False  # совмещение с чужими процессами на GPU

    def validate(self) -> None:
        if self.warm_runs < 1:
            raise ValueError("warm_runs >= 1 required")
        if self.cold_runs != 1:
            raise ValueError("контракт §5: 1 холодный прогон")
        s = self.session
        if (s.get("temperature"), s.get("top_p"), s.get("top_k"), s.get("seed")) != (
                1.0, 0.95, 20, 0):
            raise ValueError("контракт §5: T=1, top-p=0.95, top-k=20, seed=0")
        if s.get("max_tokens", 0) > 256:
            raise ValueError("контракт §5: до 256 выходных токенов")


def capture_env(kv_dtype: str = "F16", batch: int = 1,
                memory_utilization: float = 0.9,
                gpu_hours: Optional[float] = None) -> Dict[str, Any]:
    """Фиксация окружения по §5: KV-dtype, batch, memory-utilization,
    версии CUDA/драйвера, GPU-часы, co-tenancy, git-ревизия, хост."""
    cuda = driver = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,compute_capacity",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            driver = out.stdout.splitlines()[0].strip()
            cuda = "cuda (см. nvidia-smi)"
    except (OSError, subprocess.TimeoutExpired):
        pass  # локальная машина без GPU — честно None
    git_rev = None
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            git_rev = out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "kv_dtype": kv_dtype,
        "batch": batch,
        "memory_utilization": memory_utilization,
        "cuda_version": cuda,
        "driver_version": driver,
        "gpu_hours": gpu_hours,
        "hostname": platform.node(),
        "python": platform.python_version(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "git_rev": git_rev,
    }
