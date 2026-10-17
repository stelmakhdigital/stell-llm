"""Контракты /v1/messages (Anthropic-совместимый): не-стриминг, SSE,
tool_use, stop_reason маппинг."""

from __future__ import annotations

import json

from stell_serve.app import create_app
from stell_serve.engine import CtypesEngineBackend

from fastapi.testclient import TestClient

_CFG = {"model": "mock-model",
        "mock": {"token_text": " x", "ttft_ms": 1.0, "tpot_ms": 14.0}}


def make_client(lib):
    backend = CtypesEngineBackend(lib)
    application = create_app(backend, engine_cfg=dict(_CFG))
    return TestClient(application), backend


def sse_events(text):
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if data:
            out.append((event, json.loads(data)))
    return out

def test_nonstream_message(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/messages", json={
        "model": "mock-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0] == {"type": "text", "text": " x" * 4}
    assert body["stop_reason"] == "max_tokens"  # reason=length → max_tokens
    assert body["usage"]["output_tokens"] == 4
    assert body["usage"]["input_tokens"] > 0


def test_nonstream_tool_use(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/messages", json={
        "messages": [{"role": "user", "content": "weather?"}],
        "max_tokens": 4,
        "tools": [{"name": "get_weather",
                   "description": "погода",
                   "input_schema": {"type": "object"}}],
    })
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    tool_use = [b for b in body["content"] if b["type"] == "tool_use"]
    assert len(tool_use) == 1
    assert tool_use[0]["name"] == "get_weather"
    assert tool_use[0]["input"] == {"city": "Moscow"}
    assert tool_use[0]["id"].startswith("call_")  # id движка (ADR-11 §3)


def test_stream_message_sequence(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/messages", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 3, "stream": True,
    })
    assert resp.status_code == 200
    events = sse_events(resp.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "message_start"
    assert kinds.count("content_block_delta") == 3
    assert kinds[-2] == "message_delta"
    assert kinds[-1] == "message_stop"
    deltas = [d for e, d in events if e == "content_block_delta"]
    text = "".join(d["delta"]["text"] for d in deltas)
    assert text == " x" * 3
    delta_final = dict(events[-2][1])
    assert delta_final["delta"]["stop_reason"] == "max_tokens"
    assert delta_final["usage"]["output_tokens"] == 3
    start = dict(events[0][1])
    assert start["message"]["role"] == "assistant"


def test_stream_tool_use(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    resp = client.post("/v1/messages", json={
        "messages": [{"role": "user", "content": "weather?"}],
        "max_tokens": 2, "stream": True,
        "tools": [{"name": "get_weather", "input_schema": {}}],
    })
    events = sse_events(resp.text)
    kinds = [e for e, _ in events]
    # блок tool_use: content_block_start с tool_use + input_json_delta + stop
    starts = [d for e, d in events if e == "content_block_start"]
    tool_blocks = [d for d in starts if d["content_block"]["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["content_block"]["name"] == "get_weather"
    deltas = [d for e, d in events if e == "content_block_delta"
              and d["delta"]["type"] == "input_json_delta"]
    assert deltas and deltas[0]["delta"]["partial_json"] == "{\"city\":\"Moscow\"}"
    delta_final = dict(events[-2][1])
    assert delta_final["delta"]["stop_reason"] == "tool_use"


def test_engine_error_maps_to_502(mock_engine_lib):
    client, _ = make_client(mock_engine_lib)
    # отдельное приложение с fail_submit
    backend = CtypesEngineBackend(mock_engine_lib)
    app = create_app(backend, engine_cfg={
        "model": "m1", "mock": {"fail_submit": True}})
    client2 = TestClient(app)
    resp = client2.post("/v1/messages", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert resp.status_code == 502
