"""Строгий формат logits-снапшота /metrics ([R5] round 3, ADR-11 §4):
`{"requests_total","errors_total","last_n_vocab","last_positions"}` — все
4 поля присутствуют, payload (массивы логитов) в /metrics ОТСУТСТВУЕТ.

Контракт проверяется на РЕАЛЬНОМ движке (ABI-копия/сборка); поле появится
после merge core-4 (сейчас stell_metrics отдаёт min-снапшот без logits) —
поэтому живой тест — честный SKIP по наличию поля и включится после merge
автоматически. Чистая валидация формата (is_valid_logits_snapshot) покрыта
синтетикой здесь же — контракт проверен, даже пока движок SKIP-ится.
"""

from __future__ import annotations

import pytest

from stell_serve.metrics import LOGITS_SNAPSHOT_FIELDS, is_valid_logits_snapshot

try:
    from .conftest import ROOT as _ROOT  # noqa: F401
except ImportError:  # pragma: no cover
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[2]


# --------------------- чистая валидация формата ([R5]) -----------------------

def test_fields_constant_is_strict():
    assert LOGITS_SNAPSHOT_FIELDS == ("requests_total", "errors_total",
                                      "last_n_vocab", "last_positions")


def test_valid_snapshot_accepts_exact_four_numeric_fields():
    snap = {"requests_total": 7, "errors_total": 1,
            "last_n_vocab": 16, "last_positions": 5}
    assert is_valid_logits_snapshot(snap) is True
    # числа с плавающей точкой допустимы (счётчики — целые, но формат — число)
    assert is_valid_logits_snapshot({**snap, "last_n_vocab": 16.0}) is True


def test_invalid_snapshot_missing_field():
    snap = {"requests_total": 0, "errors_total": 0, "last_n_vocab": 16}
    assert is_valid_logits_snapshot(snap) is False       # нет last_positions


def test_invalid_snapshot_extra_field():
    snap = {"requests_total": 0, "errors_total": 0, "last_n_vocab": 16,
            "last_positions": 1, "logits": [0.1, 0.2]}
    assert is_valid_logits_snapshot(snap) is False       # 5-е поле запрещено


def test_invalid_snapshot_payload_arrays_forbidden():
    """Payload логитов в /metrics ОТСУТСТВУЕТ: массивы/объекты — нарушение."""
    base = {"requests_total": 0, "errors_total": 0,
            "last_n_vocab": 16, "last_positions": 1}
    assert is_valid_logits_snapshot({**base, "last_n_vocab": [16]}) is False
    assert is_valid_logits_snapshot({**base, "last_positions": {"p": 1}}) is False
    assert is_valid_logits_snapshot({**base, "requests_total": None}) is False
    assert is_valid_logits_snapshot({**base, "errors_total": "3"}) is False
    assert is_valid_logits_snapshot({**base, "last_positions": True}) is False


def test_invalid_snapshot_not_a_dict():
    assert is_valid_logits_snapshot(None) is False
    assert is_valid_logits_snapshot([1, 2, 3, 4]) is False
    assert is_valid_logits_snapshot("requests_total=0") is False


# --------- контракт на реальном движке (честный SKIP до merge core-4) -------

def test_real_engine_metrics_logits_snapshot_strict(real_client):
    """/metrics РЕАЛЬНОГО движка содержит logits-поле строго 4 полей без
    массивов. Поле появится после merge core-4 ([R5]) — до того честный
    SKIP по наличию поля (включится после merge автоматически)."""
    snap = real_client.get("/metrics").json()
    if "logits" not in snap:
        pytest.skip("logits-снапшота в stell_metrics ещё нет (merge core-4, "
                    "[R5]) — честный SKIP по наличию поля, включится после "
                    "merge автоматически")
    logits = snap["logits"]
    # ровно 4 поля, все присутствуют (строгое множество)
    assert set(logits) == set(LOGITS_SNAPSHOT_FIELDS)
    # payload в /metrics отсутствует: никакие значения не массивы/объекты
    for key, value in logits.items():
        assert not isinstance(value, (list, dict)), \
            f"logits.{key}: payload ({type(value).__name__}) запрещён в /metrics"
    assert is_valid_logits_snapshot(logits), \
        f"формат logits-снапшота нарушен: {logits}"


def test_real_engine_metrics_payload_absence_probe(real_client):
    """/metrics реального движка не содержит массивов логитов нигде в
    logits-контексте: если поле появится — в нём нет payload (и нет
    верхнеуровневого ключа-массива логитов)."""
    snap = real_client.get("/metrics").json()
    if "logits" not in snap:
        pytest.skip("см. test_real_engine_metrics_logits_snapshot_strict "
                    "(merge core-4, [R5])")
    # ключи-«ловушки» payload-контракта не должны появиться на верхнем уровне
    for trap in ("logits_payload", "logits_values", "logits_rows"):
        assert trap not in snap
    assert not isinstance(snap["logits"], (list, tuple))
