"""Конфигурация control-plane (ADR-06: порт 8080; round 2, задача T-…).

Источники (в порядке приоритета): аргументы CLI → файл конфигурации
(STELL_SERVE_CONFIG или python/stell_serve/config.json) → значения по
умолчанию. Формат файла — JSON. Неизвестные ключи — ошибка валидации
(честная защита от опечаток), пороги калибровщика делегируются
CalibrationConfig.validate() (src/calibrate, арбитр-политика).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Порты и пути по умолчанию
DEFAULT_PORT = 8080  # ADR-06
_CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

# Допустимые ключи блока calibrate (пороги арбитра, src/calibrate/policy.py)
_CALIBRATE_KEYS = frozenset({
    "spec_draft_min_speedup",
    "spec_draft_min_accept",
    "kv_quant_min_speedup",
    "kv_quant_max_ppl_penalty_pct",
    "tier_nvme_min_ram_ratio",
    "tier_boundary_bytes",
})


@dataclass
class ServeConfig:
    """Конфигурация одного процесса control-plane (ADR-06, ADR-11)."""

    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    model: str = "mock-model"
    # путь к libstell_core.so / libstell_mock_core.so; None → разрешение
    # через STELL_ENGINE_LIB / build/cpu (см. engine.resolve_engine_lib)
    engine_lib: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    ui_dir: Optional[str] = None
    log_metrics: bool = False
    # пороги калибровщика-арбитра (фичи-флаги: спек-драфт/KV-квант/NVMe-тир)
    calibrate: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be in 1..65535, got {self.port}")
        if not isinstance(self.host, str) or not self.host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a non-empty string")
        if self.engine_lib is not None and not isinstance(self.engine_lib, str):
            raise ValueError("engine_lib must be a string path or null")
        if not self.cors_origins:
            raise ValueError("cors_origins must be a non-empty list")
        for o in self.cors_origins:
            if not isinstance(o, str):
                raise ValueError("cors_origins entries must be strings")
        if self.ui_dir is not None and not isinstance(self.ui_dir, str):
            raise ValueError("ui_dir must be a string path or null")
        if not isinstance(self.log_metrics, bool):
            raise ValueError("log_metrics must be a boolean")
        self._validate_calibrate()

    def _validate_calibrate(self) -> None:
        unknown = set(self.calibrate) - _CALIBRATE_KEYS
        if unknown:
            raise ValueError(
                f"unknown calibrate keys: {sorted(unknown)}; allowed: "
                f"{sorted(_CALIBRATE_KEYS)}")
        if self.calibrate:
            self.calibration_config()  # делегирует пороги арбитру (ValueError)

    def calibration_config(self):
        """Пороги арбитра → CalibrationConfig (src/calibrate, политика round 1)."""
        from stell_calibrate.config import CalibrationConfig  # тот же репо

        cc = CalibrationConfig(**dict(self.calibrate))
        cc.validate()  # пороги арбитра обязаны быть корректны (round 1 политика)
        return cc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "model": self.model,
            "engine_lib": self.engine_lib,
            "cors_origins": list(self.cors_origins),
            "ui_dir": self.ui_dir,
            "log_metrics": self.log_metrics,
            "calibrate": dict(self.calibrate),
        }


_KNOWN_KEYS = frozenset(ServeConfig().to_dict())


def from_dict(data: Dict[str, Any]) -> ServeConfig:
    """ServeConfig из dict; неизвестные ключи — ValueError."""
    unknown = set(data) - _KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}; "
                         f"allowed: {sorted(_KNOWN_KEYS)}")
    cfg = ServeConfig(
        host=data.get("host", "0.0.0.0"),
        port=data.get("port", DEFAULT_PORT),
        model=data.get("model", "mock-model"),
        engine_lib=data.get("engine_lib"),
        cors_origins=list(data.get("cors_origins", ["*"])),
        ui_dir=data.get("ui_dir"),
        log_metrics=bool(data.get("log_metrics", False)),
        calibrate=dict(data.get("calibrate", {})),
    )
    return cfg


def load_config(path: Optional[str] = None) -> ServeConfig:
    """Загрузка: явный путь (обязателен к существованию) → STELL_SERVE_CONFIG →
    config.json рядом с пакетом (если есть) → значения по умолчанию.
    Валидация обязательна (ValueError)."""
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        cfg = from_dict(json.loads(p.read_text(encoding="utf-8")))
        cfg.validate()
        return cfg
    for cand in (os.environ.get("STELL_SERVE_CONFIG"),
                 str(_CONFIG_FILE) if _CONFIG_FILE.exists() else None):
        if not cand:
            continue
        p = Path(cand)
        if not p.exists():
            continue
        cfg = from_dict(json.loads(p.read_text(encoding="utf-8")))
        cfg.validate()
        return cfg
    cfg = ServeConfig()
    cfg.validate()
    return cfg


def default_config() -> ServeConfig:
    """Значения по умолчанию без чтения файлов (для тестов/CI)."""
    return ServeConfig()
