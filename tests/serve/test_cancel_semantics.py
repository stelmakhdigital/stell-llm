"""[R2-интеграция] Семантика stell_cancel по ADR-11 §1 (планировщик Q2,
round 2): **0 = успех, -1 = сессия не найдена/уже закрыта**.

Проверяется полный контур отмены:
  - raw ABI: cancel активной сессии → 0; несуществующей → -1; уже
    закрытой (после finish) → -1;
  - биндинг: активная → без ошибки; неизвестная → EngineError;
  - следующий poll → событие finish reason="cancelled", затем сессия
    закрыта (poll → 0 событий);
  - счётчик `cancelled_total` инкрементируется (engine-блок метрик mock и
    serve-секция /metrics).

Реальный движок движется параллельно ([R2], merge core): его
legacy-сборка (1/0, scheduler.cancel → bool) распознаётся биндингом по
отсутствию stell_logits; тесты контракта 0/-1 на реальном движке
честно SKIP-ятся до merge core и включаются автоматически (гейт — по
наличию символа stell_logits, оба изменения входят в один merge).
"""

from __future__ import annotations

import ctypes

import pytest

try:
    from .conftest import ROOT  # noqa: F401
except ImportError:  # pragma: no cover - прямой запуск без пакета
    pass


def _raw_cancel(backend, session_id: int) -> int:
    """Raw-вызов stell_cancel (минуя интерпретацию биндинга)."""
    return backend.lib.stell_cancel(backend._require_handle(),
                                    ctypes.c_uint64(session_id))


@pytest.fixture()
def mock_backend(mock_engine_lib):
    from stell_serve.app import default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(mock_engine_lib)
    backend.create(default_engine_cfg("mock-model"))
    yield backend
    backend.close()


def _submit(backend, max_tokens: int = 50) -> int:
    return backend.submit({
        "session": {"max_tokens": max_tokens, "temperature": 1.0,
                    "top_p": 0.95, "top_k": 20, "seed": 0, "stop": []},
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })


# ------------------------------- raw ABI (mock) ------------------------------

def test_raw_cancel_active_returns_zero(mock_backend):
    sid = _submit(mock_backend)
    assert _raw_cancel(mock_backend, sid) == 0


def test_raw_cancel_unknown_returns_minus_one(mock_backend):
    assert _raw_cancel(mock_backend, 9999) == -1


def test_raw_cancel_after_finish_returns_minus_one(mock_backend):
    """Закрытая сессия (finish отдан) — cancel → -1 (не найдена/закрыта)."""
    sid = _submit(mock_backend, max_tokens=1)
    while True:
        chunk = mock_backend.poll(sid)
        if any(e.get("type") == "finish" for e in chunk):
            break
    assert _raw_cancel(mock_backend, sid) == -1


# --------------------------- биндинг + poll (mock) ---------------------------

def test_cancel_active_ok_unknown_raises(mock_backend):
    from stell_serve.engine import EngineError

    sid = _submit(mock_backend)
    mock_backend.cancel(sid)  # 0 → ок, без исключения
    with pytest.raises(EngineError):
        mock_backend.cancel(9999)  # -1 → EngineError


def test_cancel_then_poll_yields_cancelled_finish_and_closes(mock_backend):
    """Отмена активной сессии → следующий poll даёт закрытие сессии
    (finish reason="cancelled"), затем poll возвращает 0 событий."""
    sid = _submit(mock_backend)
    mock_backend.cancel(sid)
    events = mock_backend.poll(sid)
    fins = [e for e in events if e.get("type") == "finish"]
    assert len(fins) == 1
    assert fins[0]["reason"] == "cancelled"
    assert fins[0]["usage"]["completion_tokens"] == 0
    # сессия закрыта: дальше событий нет (mock: poll → -1 → биндинг даёт [])
    assert mock_backend.poll(sid) == []


def test_cancelled_total_increments_in_engine_metrics(mock_backend):
    sid = _submit(mock_backend)
    before = mock_backend.metrics()["engine"]["cancelled_total"]
    mock_backend.cancel(sid)
    after = mock_backend.metrics()["engine"]["cancelled_total"]
    assert after == before + 1


def test_cancelled_twice_counts_once(mock_backend):
    """Повторная отмена закрытой сессии (после poll finish) → -1:
    счётчик не растёт."""
    sid = _submit(mock_backend)
    mock_backend.cancel(sid)
    mock_backend.poll(sid)          # finish cancelled → сессия закрыта
    from stell_serve.engine import EngineError

    with pytest.raises(EngineError):
        mock_backend.cancel(sid)
    assert mock_backend.metrics()["engine"]["cancelled_total"] == 1


# ------------------------------ через /metrics -------------------------------

def test_api_serve_cancelled_total_increments(app):
    """POST cancel-эндпоинта → serve-счётчик cancelled_total в /metrics."""
    from fastapi.testclient import TestClient

    client = TestClient(app)
    sid = app.state.engine.submit({
        "session": {"max_tokens": 50}, "messages": [{"role": "user",
                                                     "content": "hi"}],
        "stream": False})
    before = client.get("/metrics").json()["serve"]["cancelled_total"]
    resp = client.post(f"/v1/chat/completions/cancel/{sid}")
    assert resp.status_code == 200
    after = client.get("/metrics").json()["serve"]["cancelled_total"]
    assert after == before + 1


# --------------------- реальный движок (после merge core) --------------------

def _assert_cancel_contract_real(backend):
    """Контракт 0/-1 на реальном движке (активируется по наличию
    stell_logits — маркер merge core [R1]+[R2])."""
    from stell_serve.engine import EngineError

    sid = backend.submit({"session": {"max_tokens": 50}, "prompt": "hi",
                          "stream": False})
    # активная сессия → 0 (биндинг: без исключения)
    backend.cancel(sid)
    # следующий poll → закрытие сессии
    events = backend.poll(sid)
    fins = [e for e in events if e.get("type") == "finish"]
    assert fins and fins[0]["reason"] == "cancelled"
    assert backend.poll(sid) == []
    # несуществующая сессия → -1 → EngineError
    with pytest.raises(EngineError):
        backend.cancel(9999)
    # уже закрытая сессия → -1 → EngineError
    with pytest.raises(EngineError):
        backend.cancel(sid)


def test_real_engine_cancel_contract(real_engine_lib):
    from stell_serve.app import default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(real_engine_lib)
    backend.create(default_engine_cfg("engine-demo"))
    try:
        if not backend.supports_logits:
            pytest.skip("реальный движок: контракт cancel 0/-1 включится "
                        "после merge core [R2] (гейт — наличие "
                        "stell_logits, оба изменения — один merge)")
        _assert_cancel_contract_real(backend)
    finally:
        backend.close()
