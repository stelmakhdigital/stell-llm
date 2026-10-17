"""Юнит-тесты калибровщика: политика на синтетических замерах, конфиг,
замер-харнесс."""

from __future__ import annotations

import pytest

from stell_calibrate import CalibrationConfig, CalibrationHarness, decide


GOOD = {
    "decode_tok_s": 50.0, "spec_decode_tok_s": 60.0, "draft_accept_rate": 0.7,
    "decode_tok_s_kv_f16": 50.0, "decode_tok_s_kv_e4m3": 55.0,
    "ppl_kv_f16": 10.0, "ppl_kv_e4m3": 10.05,
    "ram_read_tok_s": 200.0, "nvme_read_tok_s": 80.0,
    "max_group_bytes": 3 * (1 << 30),
}


def test_config_validation():
    cfg = CalibrationConfig()
    cfg.validate()  # не кидает
    bad = CalibrationConfig(spec_draft_min_speedup=1.0)
    with pytest.raises(ValueError):
        bad.validate()


def test_spec_draft_enabled_on_speedup():
    d = decide(CalibrationConfig(), GOOD)
    assert d.spec_draft is True
    assert "spec_draft" in d.reasons and "speedup" in d.reasons["spec_draft"]


def test_spec_draft_disabled_on_low_accept():
    d = decide(CalibrationConfig(), {**GOOD, "draft_accept_rate": 0.2})
    assert d.spec_draft is False


def test_spec_draft_disabled_without_measurements():
    d = decide(CalibrationConfig(), {})
    assert d.spec_draft is False
    assert d.kv_quant is False
    assert d.kv_dtype == "F16"  # ADR-04 базово F16
    assert d.nvme_tier is False


def test_kv_quant_enabled_on_speedup_and_penalty():
    d = decide(CalibrationConfig(), GOOD)
    assert d.kv_quant is True
    assert d.kv_dtype == "E4M3"


def test_kv_quant_disabled_on_ppl_penalty():
    d = decide(CalibrationConfig(), {**GOOD, "ppl_kv_e4m3": 12.0})
    assert d.kv_quant is False
    assert d.kv_dtype == "F16"


def test_nvme_tier_and_boundary():
    d = decide(CalibrationConfig(), GOOD)
    assert d.nvme_tier is True  # 80/200 = 0.4 >= 0.25
    assert d.tier_boundary_bytes >= 3 * (1 << 30)


def test_nvme_tier_disabled_when_slow():
    d = decide(CalibrationConfig(), {**GOOD, "nvme_read_tok_s": 10.0})
    assert d.nvme_tier is False


def test_harness_measure_and_calibrate():
    h = CalibrationHarness(CalibrationConfig())
    m = h.measure("decode_tok_s", lambda: None, scale=100.0)
    assert m.value > 0
    d = h.calibrate()
    assert d.spec_draft is False  # не хватает замеров — консервативный арбитр
    assert d.to_dict()["kv_dtype"] == "F16"
