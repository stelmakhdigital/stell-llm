"""FastAPI-приложение control-plane (ADR-01, ADR-06, ADR-11).

Один процесс на порту 8080:
  POST /v1/chat/completions — OpenAI-совместимый (SSE-стриминг, tool-calls)
  POST /v1/messages         — Anthropic-совместимый (SSE, tool_use)
  GET  /health, GET /metrics — снапшот движка + serve-метрики (ADR-11 §4)
  /ui (index.html, app.js)  — дашборд-монитор

Движок — CtypesEngineBackend поверх libstell_core.so (ADR-11 §1).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .chat_template import render_prompt
from .dialects import (
    FINISH_REASON_TO_ANTHROPIC,
    FINISH_REASON_TO_OPENAI,
    sse,
    sse_done,
)
from .engine import CtypesEngineBackend, EngineBackend, EngineError
from .metrics import ServeMetrics

_REPO_UI = None
try:  # путь по умолчанию — src/ui репозитория
    from pathlib import Path

    _REPO_UI = Path(__file__).resolve().parents[2] / "src" / "ui"
except Exception:  # pragma: no cover
    _REPO_UI = None


def default_engine_cfg(model: str = "mock-model") -> Dict[str, Any]:
    return {"model": model, "mock": {"ttft_ms": 1.0, "tpot_ms": 14.0}}


# ----------------------------- pydantic-модели -----------------------------

class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = "function"
    function: Optional[FunctionCall] = None


class Message(BaseModel):
    role: str
    content: Any = None  # str | list[blocks] | None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class OpenAIRequest(BaseModel):
    model: str = "mock-model"
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = 0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    system: Optional[str] = None  # нестандартное расширение, проксируется в engine

    model_config = ConfigDict(extra="allow")


class AnthropicRequest(BaseModel):
    model: str = "mock-model"
    messages: List[Message]
    max_tokens: int = 256
    system: Optional[Any] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = 0
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False

    model_config = ConfigDict(extra="allow")


# ----------------------------- утилиты -----------------------------


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict)
        ).strip()
    return "" if content is None else str(content)


def _engine_tool(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Приводит инструменты обоих диалектов к формату engine (ADR-11 §2):
    [{"name": ..., "schema": {...}}]."""
    out: List[Dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):  # OpenAI
            fn = t["function"]
            out.append({"name": fn.get("name"),
                        "schema": fn.get("parameters") or fn.get("input_schema") or {}})
        elif "input_schema" in t:  # Anthropic
            out.append({"name": t.get("name"), "schema": t.get("input_schema") or {}})
        elif "name" in t:
            out.append({"name": t.get("name"), "schema": t.get("schema") or {}})
    return out


def build_engine_request(
    messages: Iterable[Dict[str, Any]],
    *,
    system: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    session: Dict[str, Any],
    stream: bool,
) -> Dict[str, Any]:
    """Формат request_json по ADR-11 §2."""
    msgs = [
        {"role": m.get("role", "user"), "content": _content_text(m.get("content"))}
        for m in messages
    ]
    return {
        "session": session,
        "tools": _engine_tool(tools),
        "messages": msgs,
        "prompt": render_prompt(msgs, system=system, tools=_engine_tool(tools)),
        "stream": stream,
    }


def drain_events(backend: EngineBackend, session_id: int) -> List[Dict[str, Any]]:
    """Выкачивает все события сессии до закрытия (ADR-11 §3: poll → 0/-1)."""
    events: List[Dict[str, Any]] = []
    empty_streak = 0
    while empty_streak < 32:
        chunk = backend.poll(session_id)
        if not chunk:
            empty_streak += 1
            continue
        empty_streak = 0
        events.extend(chunk)
        if any(e.get("type") == "finish" for e in chunk):
            break
    return events


# ----------------------------- фабрика приложения -----------------------------


def create_app(
    engine_backend: Optional[EngineBackend] = None,
    engine_cfg: Optional[Dict[str, Any]] = None,
    cors_origins: Optional[List[str]] = None,
    ui_dir: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title="stell-llm control-plane", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine_backend or CtypesEngineBackend()
    app.state.engine_cfg = engine_cfg or default_engine_cfg()
    app.state.metrics = ServeMetrics()
    app.state.engine.create(app.state.engine_cfg)

    ui_root = ui_dir or (str(_REPO_UI) if _REPO_UI and _REPO_UI.exists() else None)
    if ui_root:
        app.mount("/ui", StaticFiles(directory=ui_root, html=True), name="ui")

    def _error(code: str, message: str, status: int) -> HTTPException:
        return HTTPException(status_code=status, detail={"code": code, "message": message})

    def _submit(events_holder, req: Dict[str, Any]) -> int:
        return app.state.engine.submit(req)

    # ------------------------- /v1/chat/completions -------------------------

    @app.post("/v1/chat/completions")
    def chat_completions(body: OpenAIRequest):
        if body.stream:
            return _openai_stream(body)
        return _openai_complete(body)

    def _prepare(body: OpenAIRequest) -> tuple:
        messages = [m.model_dump() for m in body.messages]
        req = build_engine_request(
            messages,
            system=body.system,
            tools=body.tools,
            session={
                "max_tokens": body.max_tokens or body.max_completion_tokens or 256,
                "temperature": body.temperature if body.temperature is not None else 1.0,
                "top_p": body.top_p if body.top_p is not None else 1.0,
                "top_k": body.top_k,
                "seed": body.seed or 0,
                "stop": body.stop or [],
            },
            stream=bool(body.stream),
        )
        return messages, req

    def _openai_complete(body: OpenAIRequest):
        metrics: ServeMetrics = app.state.metrics
        _, req = _prepare(body)
        started = time.monotonic()
        try:
            sid = app.state.engine.submit(req)
            events = drain_events(app.state.engine, sid)
        except EngineError as exc:
            metrics.record_error()
            metrics.log(f"submit failed: {exc}")
            raise _error("engine_error", str(exc), 502)
        first_token_ms = None
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        finish_reason, error = "stop", None
        for ev in events:
            if ev.get("type") == "token":
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - started) * 1000.0
                text_parts.append(ev.get("text", ""))
            elif ev.get("type") == "tool_call":
                tool_calls.append({
                    "id": ev.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": ev.get("name", ""),
                                 "arguments": ev.get("arguments", "")},
                })
            elif ev.get("type") == "finish":
                usage = ev.get("usage", usage)
                finish_reason = FINISH_REASON_TO_OPENAI.get(
                    ev.get("reason", "stop"), "stop")
                if ev.get("reason") == "cancelled":
                    metrics.record_cancelled()
            elif ev.get("type") == "error":
                error = ev
        if error:
            metrics.record_error()
            metrics.log(f"engine error: {error.get('code')}")
            raise _error(error.get("code", "engine_error"),
                         error.get("message", "engine error"), 502)

        # наблюдаемые метрики serve (ADR-11 §4; engine таймеры — в /metrics)
        ttft = first_token_ms if first_token_ms is not None else 0.0
        total_ms = (time.monotonic() - started) * 1000.0
        completion_tokens = usage.get("completion_tokens", 0)
        decode_ms = max(total_ms - ttft, 0.0)
        tpot = (decode_ms / completion_tokens) if completion_tokens else 0.0
        tok_s = (completion_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        metrics.record_request(ttft, tpot, tok_s, completion_tokens)

        text = "".join(text_parts)
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text if not tool_calls else None,
                    "tool_calls": tool_calls or None,
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("prompt_tokens", 0) + completion_tokens,
            },
        }
        metrics.log(f"chat.completions ok tokens={completion_tokens}")
        return resp

    def _openai_stream(body: OpenAIRequest):
        metrics: ServeMetrics = app.state.metrics
        _, req = _prepare(body)
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def gen() -> AsyncGenerator[str, None]:
            started = time.monotonic()
            try:
                sid = app.state.engine.submit(req)
            except EngineError as exc:
                metrics.record_error()
                yield sse({"error": {"message": str(exc), "code": "engine_error"}})
                return
            completion_tokens = 0
            finish_reason = "stop"
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            first = True
            try:
                empty_streak = 0
                while empty_streak < 32:
                    chunk_events = app.state.engine.poll(sid)
                    if not chunk_events:
                        empty_streak += 1
                        continue
                    empty_streak = 0
                    for ev in chunk_events:
                        etype = ev.get("type")
                        if etype == "token":
                            delta: Dict[str, Any] = {"content": ev.get("text", "")}
                            yield sse({
                                "id": chat_id, "object": "chat.completion.chunk",
                                "created": created, "model": body.model,
                                "choices": [{"index": 0, "delta": delta,
                                             "finish_reason": None}],
                            })
                            if first:
                                first = False
                        elif etype == "tool_call":
                            yield sse({
                                "id": chat_id, "object": "chat.completion.chunk",
                                "created": created, "model": body.model,
                                "choices": [{"index": 0, "delta": {
                                    "tool_calls": [{
                                        "index": 0,
                                        "id": ev.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                                        "type": "function",
                                        "function": {"name": ev.get("name", ""),
                                                     "arguments": ev.get("arguments", "")},
                                    }]}, "finish_reason": None}],
                            })
                        elif etype == "finish":
                            usage = ev.get("usage", usage)
                            reason = ev.get("reason", "stop")
                            finish_reason = FINISH_REASON_TO_OPENAI.get(reason, "stop")
                            if reason == "cancelled":
                                metrics.record_cancelled()
                        elif etype == "error":
                            yield sse({"error": {"message": ev.get("message", ""),
                                                 "code": ev.get("code", "engine_error")}})
                            metrics.record_error()
                            return
                completion_tokens = usage.get("completion_tokens", 0)
                # терминальный chunk с finish_reason (ADR-11 §3)
                yield sse({
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": body.model,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": finish_reason}],
                })
                # usage-чанк (OpenAI: пустой choices; включаем всегда — фиксация
                # решения: usage обязателен по ADR-11 §3 finish.usage)
                yield sse({
                    "id": chat_id, "object": "chat.completion.chunk",
                    "created": created, "model": body.model,
                    "choices": [], "usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": completion_tokens,
                        "total_tokens": usage.get("prompt_tokens", 0) + completion_tokens,
                    },
                })
                yield sse_done()
                # serve-метрики запроса
                total_ms = (time.monotonic() - started) * 1000.0
                ttft = min(total_ms, 1.0) if completion_tokens else 0.0
                decode_ms = max(total_ms - ttft, 0.0)
                tpot = (decode_ms / completion_tokens) if completion_tokens else 0.0
                tok_s = (completion_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
                metrics.record_request(ttft, tpot, tok_s, completion_tokens)
                metrics.log(f"chat.completions stream ok tokens={completion_tokens}")
            finally:
                try:
                    app.state.engine.cancel(sid)
                except EngineError:
                    pass

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----------------------------- /v1/messages -----------------------------

    @app.post("/v1/messages")
    def messages(body: AnthropicRequest):
        if body.stream:
            return _anthropic_stream(body)
        return _anthropic_complete(body)

    def _anthropic_prepare(body: AnthropicRequest) -> Dict[str, Any]:
        system = _content_text(body.system) if body.system is not None else None
        return build_engine_request(
            [m.model_dump() for m in body.messages],
            system=system,
            tools=body.tools,
            session={
                "max_tokens": body.max_tokens,
                "temperature": body.temperature if body.temperature is not None else 1.0,
                "top_p": body.top_p if body.top_p is not None else 1.0,
                "top_k": body.top_k,
                "seed": body.seed or 0,
                "stop": body.stop_sequences or [],
            },
            stream=bool(body.stream),
        )

    def _anthropic_complete(body: AnthropicRequest):
        metrics: ServeMetrics = app.state.metrics
        req = _anthropic_prepare(body)
        started = time.monotonic()
        try:
            sid = app.state.engine.submit(req)
            events = drain_events(app.state.engine, sid)
        except EngineError as exc:
            metrics.record_error()
            raise _error("engine_error", str(exc), 502)
        text_parts: List[str] = []
        tool_uses: List[Dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        stop_reason, error = "end_turn", None
        first_token_ms = None
        for ev in events:
            etype = ev.get("type")
            if etype == "token":
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - started) * 1000.0
                text_parts.append(ev.get("text", ""))
            elif etype == "tool_call":
                try:
                    tool_input = json.loads(ev.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_input = {}
                tool_uses.append({
                    "type": "tool_use",
                    "id": ev.get("id") or f"toolu_{uuid.uuid4().hex[:8]}",
                    "name": ev.get("name", ""),
                    "input": tool_input,
                })
            elif etype == "finish":
                usage = ev.get("usage", usage)
                stop_reason = FINISH_REASON_TO_ANTHROPIC.get(
                    ev.get("reason", "stop"), "end_turn")
                if ev.get("reason") == "cancelled":
                    metrics.record_cancelled()
            elif etype == "error":
                error = ev
        if error:
            metrics.record_error()
            raise _error(error.get("code", "engine_error"),
                         error.get("message", "engine error"), 502)

        content: List[Dict[str, Any]] = []
        text = "".join(text_parts)
        if text:
            content.append({"type": "text", "text": text})
        content.extend(tool_uses)
        completion_tokens = usage.get("completion_tokens", 0)
        ttft = first_token_ms if first_token_ms is not None else 0.0
        total_ms = (time.monotonic() - started) * 1000.0
        decode_ms = max(total_ms - ttft, 0.0)
        tpot = (decode_ms / completion_tokens) if completion_tokens else 0.0
        tok_s = (completion_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
        metrics.record_request(ttft, tpot, tok_s, completion_tokens)
        metrics.log(f"messages ok tokens={completion_tokens}")
        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": body.model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": completion_tokens,
            },
        }

    def _anthropic_stream(body: AnthropicRequest):
        metrics: ServeMetrics = app.state.metrics
        req = _anthropic_prepare(body)
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"

        async def gen() -> AsyncGenerator[str, None]:
            try:
                sid = app.state.engine.submit(req)
            except EngineError as exc:
                metrics.record_error()
                yield sse({"type": "error", "error": {"message": str(exc)}}, event="error")
                return
            yield sse({
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message", "role": "assistant",
                    "model": body.model, "content": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }, event="message_start")
            block_index = 0
            text_block_open = False
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            stop_reason = "end_turn"
            empty_streak = 0
            try:
                while empty_streak < 32:
                    chunk_events = app.state.engine.poll(sid)
                    if not chunk_events:
                        empty_streak += 1
                        continue
                    empty_streak = 0
                    for ev in chunk_events:
                        etype = ev.get("type")
                        if etype == "token":
                            if not text_block_open:
                                yield sse({"type": "content_block_start", "index": block_index,
                                           "content_block": {"type": "text", "text": ""}},
                                          event="content_block_start")
                                text_block_open = True
                            yield sse({"type": "content_block_delta", "index": block_index,
                                       "delta": {"type": "text_delta",
                                                 "text": ev.get("text", "")}},
                                      event="content_block_delta")
                        elif etype == "tool_call":
                            if text_block_open:
                                yield sse({"type": "content_block_stop", "index": block_index},
                                          event="content_block_stop")
                                block_index += 1
                                text_block_open = False
                            tool_id = ev.get("id") or f"toolu_{uuid.uuid4().hex[:8]}"
                            yield sse({"type": "content_block_start", "index": block_index,
                                       "content_block": {"type": "tool_use",
                                                         "id": tool_id,
                                                         "name": ev.get("name", ""),
                                                         "input": {}}},
                                      event="content_block_start")
                            yield sse({"type": "content_block_delta", "index": block_index,
                                       "delta": {"type": "input_json_delta",
                                                 "partial_json": ev.get("arguments", "")}},
                                      event="content_block_delta")
                            yield sse({"type": "content_block_stop", "index": block_index},
                                      event="content_block_stop")
                            block_index += 1
                        elif etype == "finish":
                            usage = ev.get("usage", usage)
                            reason = ev.get("reason", "stop")
                            stop_reason = FINISH_REASON_TO_ANTHROPIC.get(reason, "end_turn")
                            if reason == "cancelled":
                                metrics.record_cancelled()
                            if text_block_open:
                                yield sse({"type": "content_block_stop",
                                           "index": block_index},
                                          event="content_block_stop")
                        elif etype == "error":
                            yield sse({"type": "error",
                                       "error": {"message": ev.get("message", "")}},
                                      event="error")
                            metrics.record_error()
                            return
                completion_tokens = usage.get("completion_tokens", 0)
                yield sse({"type": "message_delta",
                           "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": completion_tokens}},
                          event="message_delta")
                yield sse({"type": "message_stop"}, event="message_stop")
                metrics.record_request(
                    0.0,
                    0.0 if not completion_tokens else 0.0,
                    float(completion_tokens),
                    completion_tokens,
                )
                metrics.log(f"messages stream ok tokens={completion_tokens}")
            finally:
                try:
                    app.state.engine.cancel(sid)
                except EngineError:
                    pass

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ----------------------------- health / metrics -----------------------------

    @app.get("/health")
    def health():
        snap = app.state.engine.metrics()
        engine = snap.get("engine", {})
        return {
            "status": "ok",
            "engine": {
                "state": engine.get("state"),
                "sessions_active": engine.get("sessions_active"),
                "slots_busy": engine.get("slots_busy"),
                "slots_total": engine.get("slots_total"),
            },
            "serve": {
                "requests_total": app.state.metrics.requests_total,
                "errors_total": app.state.metrics.errors_total,
                "uptime_s": round(time.monotonic() - app.state.metrics.started, 3),
            },
        }

    @app.get("/metrics")
    def metrics():
        snap: Dict[str, Any] = dict(app.state.engine.metrics())
        snap["serve"] = app.state.metrics.snapshot()
        return snap

    @app.post("/v1/chat/completions/cancel/{session_id}")
    def cancel_session(session_id: int):
        app.state.engine.cancel(session_id)
        app.state.metrics.record_cancelled()
        return {"cancelled": True}

    return app
