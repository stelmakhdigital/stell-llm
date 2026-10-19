"""Биндинг stell_logits ([R1-интеграция], планировщик Q1; расширение ABI round 2).

Контракт (планировщик T-20261017200354 Q1, конвенция ADR-11 §1):
    int stell_logits(stell_engine_t* e, const char* request_json, char** out_json);
    request: {"prompt_tokens":[…], "positions":"last"|"all"} (по умолчанию "last")
    out:     {"logits":[[f32,…],…], "n_positions":P, "n_vocab":V}
    возврат: 0 = успех, -1 = ошибка (out_json = {"error":…});
    строки malloc + free() вызывающим.

Символ добавляет поток core ([R1]). Пока его нет в собранной .so
(ABI-копия/реальный движок pre-merge) — честный SKIP; тест включается
после merge core автоматически (классификация — по наличию символа).

Для тестов САМОГО биндинга (подписи ctypes, malloc/free, JSON-конвенция)
используется stub-.so с минимальным ABI + stell_logits (C-исходник и
фикстуры stub_logits_lib/stub_backend — в conftest.py, там же их
использует живой runner quality-gates).
"""

from __future__ import annotations

import json

import pytest

try:
    from .conftest import ROOT as _ROOT
except ImportError:  # pragma: no cover - прямой запуск без пакета
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[2]


def test_kind_classified_by_symbol_presence(stub_backend):
    """Классификация доступности logits — по наличию символа в .so:
    stub экспортирует stell_logits → supports_logits True (kind="real")."""
    assert stub_backend.kind == "real"
    assert stub_backend.supports_logits is True


def test_logits_last_position(stub_backend):
    payload = stub_backend.logits([1, 2, 3])
    assert payload["n_positions"] == 1
    assert payload["n_vocab"] == 4
    # pass-through prompt_tokens: значения выводятся из n_tokens = 3
    assert payload["logits"] == [[30, 31, 32, 33]]


def test_logits_all_positions(stub_backend):
    payload = stub_backend.logits([5, 6, 7, 8], positions="all")
    assert payload["n_positions"] == 4
    assert payload["n_vocab"] == 4
    assert payload["logits"] == [
        [40, 41, 42, 43], [41, 42, 43, 44],
        [42, 43, 44, 45], [43, 44, 45, 46],
    ]


def test_logits_error_contract_rc_minus_one(stub_backend):
    """0/-1 + {"error":…} в out_json (конвенция ADR-11 §1)."""
    from stell_serve.engine import EngineError

    with pytest.raises(EngineError, match="empty prompt_tokens"):
        stub_backend.logits([])


def test_logits_rejects_unknown_positions(stub_backend):
    from stell_serve.engine import EngineError

    with pytest.raises(EngineError, match="positions"):
        stub_backend.logits([1], positions="middle")


def test_logits_malloc_free_convention_smoke(stub_backend):
    """Строки движка — malloc, освобождает вызывающий: много повторных
    вызовов не должны течь/падать (smoke, без valgrind)."""
    for _ in range(64):
        payload = stub_backend.logits([9, 8, 7], positions="all")
        assert payload["n_positions"] == 3


def test_mock_has_no_logits_symbol(mock_engine_lib):
    """Mock (pre-merge контур) символ не экспортирует: supports_logits False,
    logits() — честная EngineError (не молчаливый нуль)."""
    from stell_serve.engine import CtypesEngineBackend, EngineError

    backend = CtypesEngineBackend(mock_engine_lib)
    try:
        assert backend.supports_logits is False
        with pytest.raises(EngineError, match="stell_logits"):
            backend.logits([1, 2, 3])
    finally:
        backend.close()


# --------------------- реальный движок (после merge core) --------------------

@pytest.fixture()
def real_logits_backend(real_engine_lib):
    """Реальный движок над mini_dense (runner:"ops", safetensors) — той же
    моделью, что и golden-фикстура (аналог test_capi_logits в tests/core)."""
    from stell_serve.engine import CtypesEngineBackend

    fx = _ROOT / "tests" / "models" / "golden" / "logits_mini_dense.json"
    profile = json.loads(fx.read_text()).get("profile", "")
    wf = "safetensors" if profile.upper() == "F16" else "nvfp4w4a16"
    fixtures = _ROOT / "tests" / "models" / "fixtures"
    cfg = {
        "runner": "ops",
        "engine": {"max_slots": 4},
        "model": {
            "description": str(fixtures / "mini_dense.json"),
            "weights": str(fixtures / "mini_dense.safetensors"),
            "weights_format": wf,
            "tokenizer": "byte",
        },
    }
    backend = CtypesEngineBackend(real_engine_lib)
    backend.create(cfg)
    yield backend
    backend.close()


def test_real_engine_logits_against_golden(real_engine_lib, real_logits_backend):
    """[R1] реальный движок: логиты mini_dense на prompt из golden-фикстуры
    совпадают с tests/models/golden/logits_mini_dense.json (rel ≤ 1e-6).
    Честный SKIP, пока символа нет в собранной .so (включится после merge
    core автоматически — классификация по наличию символа)."""
    if not real_logits_backend.supports_logits:
        pytest.skip("stell_logits ещё не в собранной .so — включится после "
                    "merge core [R1] (классификация по наличию символа)")
    golden = json.loads(
        (_ROOT / "tests" / "models" / "golden" / "logits_mini_dense.json")
        .read_text())
    prompt = golden["prompt_tokens"]
    payload = real_logits_backend.logits(prompt, positions="all")
    rows = payload["logits"]
    assert payload["n_positions"] == len(prompt)
    assert payload["n_vocab"] == 16
    # последняя позиция ("all") == logits_ref (логиты последней позиции)
    ref = golden["logits_ref"]
    last = rows[-1]
    assert len(last) == len(ref)
    worst = max(abs(a - b) / max(abs(a), abs(b), 1.0) for a, b in zip(ref, last))
    assert worst <= 1e-6, f"max_rel={worst}"
