"""Контрактные тесты C ABI mock-движка (ADR-11 §1..§4): ctypes/dlopen,
JSON через границу, коды возврата, владение памятью."""

from __future__ import annotations

import pytest

from stell_serve.engine import CtypesEngineBackend, EngineError


def make_backend(lib, cfg=None):
    backend = CtypesEngineBackend(lib)
    backend.create(cfg or {"model": "m1", "mock": {"token_text": " x"}})
    return backend


def submit_and_poll(backend, request):
    sid = backend.submit(request)
    events = []
    empty = 0
    while empty < 8:
        chunk = backend.poll(sid)
        if not chunk:
            empty += 1
            continue
        empty = 0
        events.extend(chunk)
        if any(e.get("type") == "finish" for e in chunk):
            break
    return sid, events


def test_create_fail(mock_engine_lib):
    backend = CtypesEngineBackend(mock_engine_lib)
    with pytest.raises(EngineError):
        backend.create({"model": "m1", "mock": {"fail_create": True}})


def test_submit_poll_finish_usage(mock_engine_lib):
    backend = make_backend(mock_engine_lib)
    sid, events = submit_and_poll(backend, {
        "session": {"max_tokens": 5, "temperature": 1.0, "top_p": 0.95,
                    "top_k": 20, "seed": 0, "stop": []},
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    })
    assert sid > 0
    tokens = [e for e in events if e["type"] == "token"]
    finishes = [e for e in events if e["type"] == "finish"]
    assert len(tokens) == 5
    assert len(finishes) == 1
    fin = finishes[0]
    assert fin["reason"] == "length"  # max_tokens исчерпан до естественного stop
    assert fin["usage"]["completion_tokens"] == 5
    assert fin["usage"]["prompt_tokens"] > 0
    # seq монотонен в рамках сессии (ADR-11 §3)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    # после finish сессия закрыта: poll возвращает 0 событий
    assert backend.poll(sid) == []


def test_stop_reason_natural_stop(mock_engine_lib):
    backend = make_backend(mock_engine_lib)
    _, events = submit_and_poll(backend, {
        "session": {"max_tokens": 3}, "messages": [], "stream": False,
        # mock: естественная остановка до лимита → reason stop (ADR-11 §3)
        "mock": {"token_count": 2},
    })
    fin = [e for e in events if e["type"] == "finish"][0]
    assert fin["reason"] == "stop"
    assert fin["usage"]["completion_tokens"] == 2


def test_tool_call_event(mock_engine_lib):
    backend = make_backend(mock_engine_lib)
    _, events = submit_and_poll(backend, {
        "session": {"max_tokens": 4},
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{"name": "get_weather", "schema": {"type": "object"}}],
        "stream": True,
    })
    tool = [e for e in events if e["type"] == "tool_call"]
    assert len(tool) == 1
    assert tool[0]["name"] == "get_weather"
    assert tool[0]["arguments"] == "{\"city\":\"Moscow\"}"
    assert tool[0]["id"].startswith("call_")
    fin = [e for e in events if e["type"] == "finish"][0]
    assert fin["reason"] == "tool_call"


def test_cancel_yields_cancelled_finish(mock_engine_lib):
    backend = make_backend(mock_engine_lib)
    sid = backend.submit({"session": {"max_tokens": 10}, "messages": []})
    assert backend.cancel(sid) is None
    events = backend.poll(sid)
    fin = [e for e in events if e["type"] == "finish"][0]
    assert fin["reason"] == "cancelled"
    assert backend.poll(sid) == []


def test_cancel_unknown_session(mock_engine_lib):
    backend = make_backend(mock_engine_lib)
    with pytest.raises(EngineError):
        backend.cancel(9999)


def test_slot_overflow(mock_engine_lib):
    backend = CtypesEngineBackend(mock_engine_lib)
    backend.create({"model": "m1", "mock": {"slots_total": 2}})
    assert backend.submit({"session": {"max_tokens": 1}}) > 0
    assert backend.submit({"session": {"max_tokens": 1}}) > 0
    with pytest.raises(EngineError, match="slot_overflow"):
        backend.submit({"session": {"max_tokens": 1}})


def test_metrics_snapshot(mock_engine_lib):
    backend = make_backend(mock_engine_lib)
    snap = backend.metrics()
    for key in ("engine", "latency", "tiers", "parallel", "features", "build"):
        assert key in snap
    assert snap["engine"]["slots_total"] == 16
    assert "p50" in snap["latency"]["ttft_ms"] and "p95" in snap["latency"]["tpot_ms"]
    assert snap["tiers"]["prefetch"]["hit_rate"] == 0.63
    assert len(snap["tiers"]["tier_bytes"]) == 3
    backend.close()
