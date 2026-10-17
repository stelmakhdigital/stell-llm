"""Политика арбитра: решения вкл/выкл по замерам (PLAN.md §4, принцип 4).

Политика — чистая функция от конфига и синтетических/реальных замеров;
юнит-тесты гоняют её на синтетике (см. tests/serve/test_calibrate.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import CalibrationConfig


@dataclass
class CalibrationDecision:
    spec_draft: bool = False
    kv_quant: bool = False
    kv_dtype: str = "F16"  # ADR-04: базово F16; E4M3 — по калибровке
    nvme_tier: bool = False
    tier_boundary_bytes: int = 1 << 30
    reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_draft": self.spec_draft,
            "kv_quant": self.kv_quant,
            "kv_dtype": self.kv_dtype,
            "nvme_tier": self.nvme_tier,
            "tier_boundary_bytes": self.tier_boundary_bytes,
            "reasons": dict(self.reasons),
        }


def decide(cfg: CalibrationConfig, m: Dict[str, Optional[float]]) -> CalibrationDecision:
    """Решения по замерам (m — dict с опциональными ключами):
      decode_tok_s, spec_decode_tok_s, draft_accept_rate,
      decode_tok_s_kv_f16, decode_tok_s_kv_e4m3,
      ppl_kv_f16, ppl_kv_e4m3,
      ram_read_tok_s, nvme_read_tok_s,
      max_group_bytes (кандидат на выгрузку в NVMe).
    Отсутствующие замеры трактуются консервативно (фича выключена)."""
    cfg.validate()
    d = CalibrationDecision(tier_boundary_bytes=cfg.tier_boundary_bytes)

    # --- спекулятивный драфт ------------------------------------------------
    base = m.get("decode_tok_s")
    spec = m.get("spec_decode_tok_s")
    accept = m.get("draft_accept_rate")
    if base and spec and accept is not None:
        speedup = spec / base
        if speedup >= cfg.spec_draft_min_speedup and accept >= cfg.spec_draft_min_accept:
            d.spec_draft = True
            d.reasons["spec_draft"] = (
                f"speedup={speedup:.3f} >= {cfg.spec_draft_min_speedup}, "
                f"accept={accept:.3f} >= {cfg.spec_draft_min_accept}")
        else:
            d.reasons["spec_draft"] = (
                f"недостаточный выигрыш: speedup={speedup:.3f}, accept={accept:.3f}")
    else:
        d.reasons["spec_draft"] = "нет замеров — фича выключена"

    # --- KV-квант E4M3 -------------------------------------------------------
    f16 = m.get("decode_tok_s_kv_f16")
    e4m3 = m.get("decode_tok_s_kv_e4m3")
    ppl_f16 = m.get("ppl_kv_f16")
    ppl_e4m3 = m.get("ppl_kv_e4m3")
    if f16 and e4m3 and ppl_f16 and ppl_e4m3:
        speedup = e4m3 / f16
        penalty_pct = 100.0 * (ppl_e4m3 - ppl_f16) / ppl_f16
        if speedup >= cfg.kv_quant_min_speedup and penalty_pct <= cfg.kv_quant_max_ppl_penalty_pct:
            d.kv_quant = True
            d.kv_dtype = "E4M3"
            d.reasons["kv_quant"] = (
                f"speedup={speedup:.3f} >= {cfg.kv_quant_min_speedup}, "
                f"penalty={penalty_pct:.2f}% <= {cfg.kv_quant_max_ppl_penalty_pct}%")
        else:
            d.reasons["kv_quant"] = (
                f"отклонено: speedup={speedup:.3f}, penalty={penalty_pct:.2f}%")
    else:
        d.reasons["kv_quant"] = "нет замеров — F16 (ADR-04)"

    # --- NVMe-тир и граница тиров -------------------------------------------
    ram = m.get("ram_read_tok_s")
    nvme = m.get("nvme_read_tok_s")
    if ram and nvme:
        ratio = nvme / ram
        if ratio >= cfg.tier_nvme_min_ram_ratio:
            d.nvme_tier = True
            d.reasons["tier"] = f"nvme/ram={ratio:.3f} >= {cfg.tier_nvme_min_ram_ratio}"
        else:
            d.reasons["tier"] = f"NVMe слишком медленный: ratio={ratio:.3f}"
    else:
        d.reasons["tier"] = "нет замеров — NVMe-тир выключен"

    group = m.get("max_group_bytes")
    if group:
        # граница: группы размером <= boundary держим в VRAM, остальные — выше
        d.tier_boundary_bytes = max(cfg.tier_boundary_bytes, int(group))
        d.reasons["boundary"] = f"max_group_bytes={int(group)}"
    return d
