"""stell_serve — Python control-plane stell-llm (ADR-01, ADR-06, ADR-11).

Один процесс на порту 8080: /v1/chat/completions (OpenAI), /v1/messages
(Anthropic), /health, /metrics, /ui (дашборд). Движок — libstell_core.so
через ctypes (ADR-11); для локальных тестов без GPU — mock-движок
src/serve/mock (тот же C ABI).
"""

__version__ = "0.1.0"
