"""Тесты конфигурации control-plane (python/stell_serve/config.py, round 2)."""

from __future__ import annotations

import json

import pytest

from stell_serve.config import ServeConfig, from_dict, load_config


def test_defaults_are_valid():
    cfg = ServeConfig()
    cfg.validate()  # не кидает
    assert cfg.port == 8080  # ADR-06
    assert cfg.cors_origins == ["*"]
    assert cfg.model == "mock-model"
    assert cfg.engine_lib is None  # → разрешение через STELL_ENGINE_LIB/build


def test_validate_port_range():
    for bad in (0, -1, 70000, 65536):
        cfg = ServeConfig(port=bad)
        with pytest.raises(ValueError):
            cfg.validate()
    cfg = ServeConfig(port=65535)
    cfg.validate()


def test_validate_cors_origins():
    with pytest.raises(ValueError):
        ServeConfig(cors_origins=[]).validate()
    with pytest.raises(ValueError):
        ServeConfig(cors_origins=[None]).validate()
    ServeConfig(cors_origins=["http://localhost:3000", "*"]).validate()


def test_validate_model_and_host():
    with pytest.raises(ValueError):
        ServeConfig(model="").validate()
    with pytest.raises(ValueError):
        ServeConfig(host="").validate()


def test_from_dict_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        from_dict({"portt": 8080})


def test_from_dict_full_roundtrip():
    data = {
        "host": "127.0.0.1", "port": 9090, "model": "m",
        "engine_lib": "/tmp/lib.so", "cors_origins": ["http://a"],
        "ui_dir": "/tmp/ui", "log_metrics": True,
        "calibrate": {"spec_draft_min_speedup": 1.2},
    }
    cfg = from_dict(data)
    cfg.validate()
    assert cfg.to_dict() == data


def test_calibrate_flags_validation():
    cfg = ServeConfig(calibrate={"spec_draft_min_speedup": 1.0})
    with pytest.raises(ValueError):  # порог арбитра < 1 — ошибка (policy)
        cfg.validate()
    ok = ServeConfig(calibrate={"tier_boundary_bytes": 5 * (1 << 30)})
    ok.validate()
    cc = ok.calibration_config()
    assert cc.tier_boundary_bytes == 5 * (1 << 30)
    with pytest.raises(ValueError, match="unknown calibrate keys"):
        ServeConfig(calibrate={"bogus": 1}).validate()


def test_calibration_config_mapping():
    cfg = ServeConfig(calibrate={"kv_quant_min_speedup": 1.5,
                                 "kv_quant_max_ppl_penalty_pct": 0.5})
    cc = cfg.calibration_config()
    assert cc.kv_quant_min_speedup == 1.5
    assert cc.kv_quant_max_ppl_penalty_pct == 0.5
    # дефолты арбитра сохраняются для неуказанных порогов
    assert cc.spec_draft_min_speedup == 1.05
    assert cc.tier_nvme_min_ram_ratio == 0.25  # дефолт арбитра сохранён


def test_load_config_from_file(tmp_path):
    p = tmp_path / "serve.json"
    p.write_text(json.dumps({"port": 9090, "cors_origins": ["http://x"],
                             "calibrate": {"spec_draft_min_accept": 0.6}}))
    cfg = load_config(str(p))
    cfg.validate()
    assert cfg.port == 9090
    assert cfg.cors_origins == ["http://x"]
    assert cfg.calibrate["spec_draft_min_accept"] == 0.6
    # остальные поля — дефолты
    assert cfg.model == "mock-model" and cfg.host == "0.0.0.0"


def test_load_config_env_override(tmp_path, monkeypatch):
    p = tmp_path / "env-serve.json"
    p.write_text(json.dumps({"port": 9100, "log_metrics": True}))
    monkeypatch.setenv("STELL_SERVE_CONFIG", str(p))
    cfg = load_config(None)
    assert cfg.port == 9100 and cfg.log_metrics is True


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.json"))


def test_load_config_invalid_file_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"port": 999999}))
    with pytest.raises(ValueError):
        load_config(str(p))


def test_package_config_json_is_valid():
    """Коммитнутый config.json рядом с пакетом — валидный дефолтный конфиг."""
    from pathlib import Path

    import stell_serve

    p = Path(stell_serve.__file__).resolve().parent / "config.json"
    cfg = load_config(str(p))
    cfg.validate()
    assert cfg.port == 8080
    assert cfg.calibrate["tier_boundary_bytes"] == 1 << 30
