"""Контракты /v1/chat/completions (OpenAI-совместимый): не-стриминг,
стриминг SSE, tool-calls, finish_reason маппинг, CORS."""

from __future__ import annotations

import json

from stell_serve.app import create_app
from stell_serve.engine import CtypesEngineBackend

from fastapi.testclient import TestClient


def make_client(lib, cfg=None):
    backend = CtypesEngineBackend(lib)
    cfg = cfg or {"model": "mock-model",
                  "mock": {"token_text": " x", "ttft_ms": 1.0, "tpot_ms": 14.0}}
    application = create_app(backend, engine_cfg=cfg)
    return TestClient(application), backend


def sse_events(text):
    """Разбор SSE-кадров: [event, json-data]."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        data = None
        for line in block.splitlines():
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
        if data is None:
            continue
        if data == "[DONE]":
            out.append(("done", "[DONE]"))
        else:
            out.append(("data", json.loads(data)))
    return out


def test_nonstream_completion(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/chat/completions", json={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == " x" * 5
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + 5


def test_nonstream_tool_calls(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather?"}],
        "max_tokens": 4,
        "tools": [{"type": "function",
                   "function": {"name": "get_weather",
                                "parameters": {"type": "object"}}}],
    })
    assert resp.status_code == 200
    msg = resp.json()["choices"][0]
    assert msg["finish_reason"] == "tool_calls"
    assert msg["message"]["content"] is None
    tc = msg["message"]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Moscow"}
    assert tc["id"].startswith("call_")


def test_streaming_chunks_and_usage(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4, "stream": True,
    })
    assert resp.status_code == 200
    events = sse_events(resp.text)
    chunks = [d for kind, d in events if kind == "data"]
    assert events[-1][0] == "done" and events[-1][1] == "[DONE]"
    token_chunks = [c for c in chunks
                    if c.get("choices") and c["choices"][0].get("delta", {}).get("content")]
    assert len(token_chunks) == 4
    text = "".join(c["choices"][0]["delta"]["content"] for c in token_chunks)
    assert text == " x" * 4
    # терминальный chunk: finish_reason (ADR-11 §3 finish)
    finish_chunks = [c for c in chunks
                     if c.get("choices") and c["choices"][0].get("finish_reason")]
    assert finish_chunks[-1]["choices"][0]["finish_reason"] == "length"
    # usage-чанк (обязателен по ADR-11 §3 finish.usage)
    usage_chunks = [c for c in chunks if c.get("usage")]
    assert usage_chunks[-1]["usage"]["completion_tokens"] == 4


def test_streaming_tool_calls(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather?"}],
        "max_tokens": 2, "stream": True,
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    })
    events = sse_events(resp.text)
    chunks = [d for kind, d in events if kind == "data"]
    tool_deltas = [c["choices"][0]["delta"]["tool_calls"]
                   for c in chunks
                   if c.get("choices") and c["choices"][0].get("delta", {}).get("tool_calls")]
    assert len(tool_deltas) == 1
    tc = tool_deltas[0][0]
    assert tc["index"] == 0
    assert tc["function"]["name"] == "get_weather"
    finish = [c for c in chunks
              if c.get("choices") and c["choices"][0].get("finish_reason") == "tool_calls"]
    assert finish


def test_engine_error_maps_to_502(mock_engine_lib):
    client, _ = make_client(
        mock_engine_lib, {"model": "m1", "mock": {"fail_submit": True}})
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 502
    assert resp.json()["detail"]["code"] == "engine_error"


def test_cors_preflight(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.options("/v1/chat/completions", headers={
        "Origin": "http://dashboard.example",
        "Access-Control-Request-Method": "POST",
    })
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "*"


def test_cors_simple_request_header(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
        headers={"Origin": "http://dashboard.example"})
    assert resp.headers["access-control-allow-origin"] == "*"
