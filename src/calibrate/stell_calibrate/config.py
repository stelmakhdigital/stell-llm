"""Конфигурация калибровщика: пороги арбитра (вкл/выкл по замеру)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class CalibrationConfig:
    # спекулятивный драфт: включается, если spec-decode быстрее target-only
    spec_draft_min_speedup: float = 1.05
    spec_draft_min_accept: float = 0.5  # доля принятых драфт-токенов
    # KV-квант E4M3: включается, если быстрее и PPL не хуже порога
    kv_quant_min_speedup: float = 1.03
    kv_quant_max_ppl_penalty_pct: float = 1.0  # допустимый рост PPL, %
    # NVMe-тир: включается, если его чтение ≥ доли RAM-тира
    tier_nvme_min_ram_ratio: float = 0.25
    # граница тиров: тензорные группы меньше boundary держатся в VRAM
    tier_boundary_bytes: int = 1 << 30  # 1 GiB по умолчанию

    def validate(self) -> None:
        if self.spec_draft_min_speedup <= 1.0:
            raise ValueError("spec_draft_min_speedup must be > 1.0")
        if not 0.0 < self.spec_draft_min_accept <= 1.0:
            raise ValueError("spec_draft_min_accept must be in (0, 1]")
        if self.kv_quant_min_speedup <= 1.0:
            raise ValueError("kv_quant_min_speedup must be > 1.0")
        if self.kv_quant_max_ppl_penalty_pct < 0:
            raise ValueError("kv_quant_max_ppl_penalty_pct must be >= 0")
        if self.tier_nvme_min_ram_ratio <= 0:
            raise ValueError("tier_nvme_min_ram_ratio must be > 0")
        if self.tier_boundary_bytes < 0:
            raise ValueError("tier_boundary_bytes must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_draft_min_speedup": self.spec_draft_min_speedup,
            "spec_draft_min_accept": self.spec_draft_min_accept,
            "kv_quant_min_speedup": self.kv_quant_min_speedup,
            "kv_quant_max_ppl_penalty_pct": self.kv_quant_max_ppl_penalty_pct,
            "tier_nvme_min_ram_ratio": self.tier_nvme_min_ram_ratio,
            "tier_boundary_bytes": self.tier_boundary_bytes,
        }
