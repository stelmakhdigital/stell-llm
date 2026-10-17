"""Маппинг finish.reason (ADR-11 §3) в диалекты API и построители SSE.

Engine не знает API-диалектов (ADR-11 §3): маппинг — на стороне serve.
  reason      OpenAI finish_reason   Anthropic stop_reason
  stop        stop                   end_turn
  length      length                 max_tokens
  tool_call   tool_calls             tool_use
  cancelled   stop (+флаг в метриках)   end_turn (+флаг в метриках)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

FINISH_REASON_TO_OPENAI = {
    "stop": "stop",
    "length": "length",
    "tool_call": "tool_calls",
    # в OpenAI нет терминальной отмены; честный фолббек — stop
    "cancelled": "stop",
}

FINISH_REASON_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_call": "tool_use",
    # в Anthropic нет терминальной отмены; честный фолббек — end_turn
    "cancelled": "end_turn",
}

TERMINAL_TYPES = {"finish", "error"}
EVENT_TYPES = {"token", "tool_call", "tool_result", "finish", "error"}


def sse(data: Dict[str, Any], event: Optional[str] = None) -> str:
    """Формат SSE-кадра: [event: name]\\ndata: {json}\\n\\n.

    Ручное форматирование выбрано вместо sse-starlette, потому что оба
    диалекта требуют точного wire-формата (OpenAI: data: {...} + data: [DONE];
    Anthropic: именованные event:). sse-starlette остаётся зависимостью
    для будущих keepalive-политик.
    """
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"
