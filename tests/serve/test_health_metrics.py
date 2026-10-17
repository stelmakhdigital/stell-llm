"""Контракты /health, /metrics и дашборда /ui (ADR-11 §4, ADR-06)."""

from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["engine"]) >= {"state", "sessions_active", "slots_busy",
                                   "slots_total"}
    assert body["engine"]["slots_total"] == 16
    assert body["serve"]["requests_total"] == 0


def test_metrics_passthrough_and_serve(client):
    # один не-стриминг запрос, чтобы serve-метрики наполнились
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    assert resp.status_code == 200
    resp = client.get("/metrics")
    assert resp.status_code == 200
    snap = resp.json()
    # passthrough tier-статистики движка (ADR-11 §4)
    assert snap["tiers"]["prefetch"]["hit_rate"] == 0.63
    assert set(snap["tiers"]["layer_pause_ms"]) == {"p50", "p95", "p99"}
    assert len(snap["tiers"]["tier_bytes"]) == 3
    # латентность движка: TTFT/TPOT/tok/s
    assert snap["latency"]["ttft_ms"]["p50"] == 1.0
    assert snap["latency"]["tok_s"]["decode"] > 0
    # serve-секция: счётчики + наблюдаемые TTFT/TPOT/tok/s + логи
    assert snap["serve"]["requests_total"] == 1
    assert snap["serve"]["latency"]["ttft_ms"]["p50"] is not None
    assert snap["serve"]["latency"]["tpot_ms"]["p95"] is not None
    assert snap["serve"]["latency"]["decode_tok_s"]["p50"] is not None
    assert snap["serve"]["logs"]
    # features/build passthrough
    assert snap["features"]["kv_quant"] == "F16"
    assert "version" in snap["build"]


def test_cancelled_total_serve_and_engine(client, app):
    # уточнение ADR-11 round 2: факт отмены — в /metrics (не в семантике
    # finish_reason); счётчики: serve.cancelled_total + engine passthrough
    # (additive-поле mock-движка)
    sid = app.state.engine.submit({"session": {"max_tokens": 10},
                                   "messages": [], "stream": False})
    assert client.post(f"/v1/chat/completions/cancel/{sid}").status_code == 200
    snap = client.get("/metrics").json()
    assert snap["serve"]["cancelled_total"] == 1
    assert snap["engine"]["cancelled_total"] == 1


def test_metrics_before_requests(client):
    snap = client.get("/metrics").json()
    assert snap["serve"]["requests_total"] == 0
    assert snap["serve"]["latency"]["ttft_ms"]["p50"] is None


def test_dashboard_ui(client):
    resp = client.get("/ui/index.html")
    assert resp.status_code == 200
    assert "stell-llm" in resp.text
    assert client.get("/ui/app.js").status_code == 200
    assert client.get("/ui/style.css").status_code == 200
