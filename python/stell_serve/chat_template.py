"""Chat-шаблон serve (ADR-11 §2): serve применяет шаблон модели и передаёт
готовую строку/токены; engine не знает ни OpenAI, ни Anthropic.

Текущая реализация — детерминированный линейный шаблон (заглушка до
появления реестра моделей потоков D; подмена на jinja-шаблоны модели
происходит здесь, контракт с engine не меняется).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def render_prompt(
    messages: Iterable[Dict[str, Any]],
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Собирает плоский промпт из сообщений (роль: текст), system-блока
    и описания инструментов. Детерминирован — удобно для тестов."""
    parts: List[str] = []
    if system:
        parts.append(f"system: {system}")
    if tools:
        names = ", ".join(t.get("name", "?") for t in tools)
        parts.append(f"tools: {names}")
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            ).strip()
        else:
            text = str(content or "")
        parts.append(f"{role}: {text}")
    return "\n".join(parts)
