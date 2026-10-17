"""Контрактные тесты РЕАЛЬНОГО движка (src/engine, поток A) через C ABI
(ADR-11 §1..§4, round 2 вклейка). Прогон на ABI-копии src/engine
(libstell_core_serve.so из src/serve/CMakeLists.txt) — тот же исходник
capi.cpp/scheduler.cpp, что и таргет stell_core потока core; когда core
вклеит SHARED libstell_core.so в build/cpu/src/engine/, эти же тесты
покрываются через STELL_ENGINE_LIB (см. engine.resolve_engine_lib).

Особенности реального движка против mock (зафиксированы в отчёте round 2):
  - token-события несут только token_id (engine не владеет детокенизатором) —
    serve подставляет текст decimal-детокенизатором;
  - шаги декода ведутся вызовами stell_step (однопоточный CPU-контур);
  - cancel: 1 = ок, 0 = сессия неизвестна (scheduler.cancel → bool);
    у mock: 0 ок / <0 неизвестная — расхождение, обработано в биндинге.
"""

from __future__ import annotations

import re

import pytest

from stell_serve.engine import CtypesEngineBackend, EngineError, decimal_detokenizer

ROOT = None  # ROOT импортируется из conftest ниже
try:
    from .conftest import ROOT as _CONFTEST_ROOT
    ROOT = _CONFTEST_ROOT
except ImportError:  # pragma: no cover
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def real_backend(real_engine_lib):
    from stell_serve.app import default_engine_cfg

    backend = CtypesEngineBackend(real_engine_lib)
    backend.create(default_engine_cfg("engine-demo"))
    yield backend
    backend.close()


def drain(backend, sid):
    events = []
    empty = 0
    rounds = 0
    while empty < 32 and rounds < 1024:
        rounds += 1
        chunk = backend.poll(sid)
        if not chunk:
            empty += 1
            backend.step()  # однопоточный CPU-контур: шаг ведёт serve
            continue
        empty = 0
        events.extend(chunk)
        if any(e.get("type") == "finish" for e in chunk):
            break
    return events


# ------------------------------- ABI ----------------------------------------

def test_abi_symbols_match_master_capi_h(real_engine_lib, real_backend):
    """ABI-совместимость с авторитетным capi.h из master (src/engine):
    каждый объявленный в заголовке символ доступен из .so (включая
    stell_step — ABI-совместимое расширение round 1)."""
    header = (ROOT / "src" / "engine" / "include" / "stell" / "engine" /
              "capi.h")
    declared = set(re.findall(r"\b(stell_[a-z_]+)\s*\(", header.read_text()))
    assert {"stell_engine_create", "stell_engine_destroy", "stell_submit",
            "stell_poll", "stell_cancel", "stell_metrics", "stell_step"} <= declared
    lib = real_backend.lib
    for symbol in sorted(declared):
        assert hasattr(lib, symbol), f"ABI: нет символа {symbol} (master capi.h)"


def test_backend_kind_is_real(real_backend):
    assert real_backend.kind == "real"
    assert real_backend.supports_step is True


# --------------------------- submit / poll / step ---------------------------

def test_submit_poll_step_tokens(real_backend):
    sid = real_backend.submit({
        "session": {"max_tokens": 4, "temperature": 1.0, "top_p": 0.95,
                    "top_k": 20, "seed": 0, "stop": []},
        "prompt": "abc",  # → prompt_tokens = UTF-8-байты (3 байта)
        "stream": False,
    })
    assert sid > 0
    events = drain(real_backend, sid)
    tokens = [e for e in events if e["type"] == "token"]
    finishes = [e for e in events if e["type"] == "finish"]
    assert len(tokens) == 4
    assert [t["token_id"] for t in tokens] == [0, 1, 2, 3]
    # детокенизатор serve (decimal): token_id → текст; контракт не меняется
    assert [t["text"] for t in tokens] == ["0", "1", "2", "3"]
    fin = finishes[0]
    assert fin["reason"] == "length"  # max_tokens исчерпан (demo без eos)
    assert fin["usage"]["prompt_tokens"] == 3  # байты промпта
    assert fin["usage"]["completion_tokens"] == 4
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    # после finish poll = 0 событий (scheduler удаляет запись)
    assert real_backend.poll(sid) == []


def test_submit_requires_prompt(real_backend):
    # пустой prompt → engine отказ: empty prompt_tokens (ADR-11 §2)
    with pytest.raises(EngineError):
        real_backend.submit({"session": {"max_tokens": 2}, "prompt": "",
                             "stream": False})


def test_cancel_yields_cancelled(real_backend):
    sid = real_backend.submit({"session": {"max_tokens": 50}, "prompt": "hi",
                               "stream": False})
    real_backend.cancel(sid)
    events = real_backend.poll(sid)
    fin = [e for e in events if e["type"] == "finish"][0]
    assert fin["reason"] == "cancelled"
    assert fin["usage"]["completion_tokens"] == 0
    assert real_backend.poll(sid) == []


def test_cancel_unknown_session(real_backend):
    with pytest.raises(EngineError):
        real_backend.cancel(9999)


def test_slot_overflow(real_backend):
    with pytest.raises(EngineError):
        for _ in range(32):
            real_backend.submit({"session": {"max_tokens": 1}, "prompt": "x",
                                 "stream": False})


def test_metrics_snapshot_engine_block(real_backend):
    snap = real_backend.metrics()
    assert set(snap) >= {"engine"}
    engine = snap["engine"]
    assert engine["state"] == "serving"
    assert engine["slots_total"] == 16
    assert engine["batch_size_max"] == 16
    # у реального движка нет tiers/latency/features (min-снапшот round 1) —
    # /metrics честно отдаёт только engine-блок + serve-секцию (см. API-тест)


# --------------------------- через FastAPI-приложение -----------------------

def test_api_chat_completions_real_engine(real_client):
    resp = real_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "abc"}], "max_tokens": 4})
    assert resp.status_code == 200
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["message"]["content"] == "0123"
    assert body["usage"]["prompt_tokens"] == 9  # render_prompt: "user: abc" → 9 байт
    assert body["usage"]["completion_tokens"] == 4


def test_api_chat_completions_stream_real_engine(real_client):
    resp = real_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "ab"}], "max_tokens": 3,
        "stream": True})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "data: [DONE]" in text
    deltas = [line for line in text.splitlines() if line.startswith("data: {")]
    contents = re.findall(r'"delta":\s*\{"content":\s*"\d"', "\n".join(deltas))
    assert len(contents) == 3  # 3 token-чанка: "0","1","2"
    assert '"finish_reason": "length"' in text


def test_api_messages_real_engine(real_client):
    resp = real_client.post("/v1/messages", json={
        "messages": [{"role": "user", "content": "abc"}], "max_tokens": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "max_tokens"
    texts = [b for b in body["content"] if b.get("type") == "text"]
    assert texts[0]["text"] == "0123"


def test_api_metrics_real_engine(real_client):
    resp = real_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2})
    assert resp.status_code == 200
    snap = real_client.get("/metrics").json()
    # passthrough engine-блока (у реального движка — min-снапшот)
    assert snap["engine"]["state"] == "serving"
    assert snap["engine"]["slots_total"] == 16
    assert "tiers" not in snap and "latency" not in snap  # честно: нет данных
    # serve-секция как у mock (ADR-11 §4)
    assert snap["serve"]["requests_total"] == 1
    assert snap["serve"]["latency"]["ttft_ms"]["p50"] is not None
    assert real_client.get("/health").json()["status"] == "ok"


def test_api_cancelled_total_real_engine(real_app):
    # cancel endpoint → движок cancel (scheduler) + serve-счётчик cancelled_total
    # (сессия должна быть активной: на закрытой — движок вернёт «неизвестна»)
    from fastapi.testclient import TestClient

    client = TestClient(real_app)
    sid = real_app.state.engine.submit({
        "session": {"max_tokens": 50}, "prompt": "hi", "stream": False})
    resp = client.post(f"/v1/chat/completions/cancel/{sid}")
    assert resp.status_code == 200
    snap = client.get("/metrics").json()
    assert snap["serve"]["cancelled_total"] == 1
    # расхождение ABI/метрик (зафиксировано в отчёте round 2): mock публикует
    # engine.cancelled_total (additive-поле mock); РЕАЛЬНЫЙ движок round 1
    # (min-снапшот src/engine/capi.cpp) его не отдаёт — служебный счётчик serve
    # единственный источник факта отмены в /metrics.
    assert "cancelled_total" not in snap["engine"]


def test_decimal_detokenizer_is_deterministic():
    assert [decimal_detokenizer(i) for i in range(4)] == ["0", "1", "2", "3"]


def test_dashboard_serves_on_real_engine(real_client):
    """Дашборд соответствует /metrics реального движка (снапшот round 2):
    у реального движка нет tiers/latency — app.js рендерит через optional
    chaining (tiers?.tier_bytes), /metrics честно отдаёт только engine+serve."""
    resp = real_client.get("/ui/index.html")
    assert resp.status_code == 200 and "stell-llm" in resp.text
    app_js = real_client.get("/ui/app.js")
    assert app_js.status_code == 200
    assert "/metrics" in app_js.text
    assert "tiers?.tier_bytes" in app_js.text          # отсутствие tiers не роняет UI
    assert real_client.get("/ui/style.css").status_code == 200


def test_resolve_engine_lib_paths(real_engine_lib, monkeypatch):
    from pathlib import Path

    from stell_serve.engine import (_DEFAULT_MOCK, _REAL_CANDIDATES,
                                    default_engine_lib, resolve_engine_lib)

    # явный путь mock → kind mock
    assert resolve_engine_lib(str(_DEFAULT_MOCK))[1] == "mock"
    # env → реальный движок (STELL_ENGINE_LIB)
    monkeypatch.setenv("STELL_ENGINE_LIB", str(real_engine_lib))
    path, kind = resolve_engine_lib()
    assert kind == "real" and path == Path(real_engine_lib)
    # дефолт (без env): реальный кандидат (таргет core в build/cpu/src/engine/)
    # имеет приоритет над локальной ABI-копией; последняя — фолббек
    monkeypatch.delenv("STELL_ENGINE_LIB", raising=False)
    assert default_engine_lib() == real_engine_lib
    assert _REAL_CANDIDATES[0] == Path(_REAL_CANDIDATES[0])
    assert _DEFAULT_MOCK.name == "libstell_mock_core.so"
