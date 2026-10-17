# stell-llm control-plane

Тонкий Python-слой (ADR-01): FastAPI-приложение `stell_serve`, один процесс
на порту 8080 (ADR-06):

- `POST /v1/chat/completions` — OpenAI-совместимый (SSE-стриминг, tool-calls);
- `POST /v1/messages` — Anthropic-совместимый;
- `GET /health`, `GET /metrics` — снапшот движка + serve-метрики (ADR-11 §4);
- `/ui` — дашборд-монитор (статика из `src/ui`).

Движок подключается строго через C ABI (ADR-11): `libstell_core.so` через
ctypes. Для локальных тестов без GPU используется mock-движок
`src/serve/mock` (тот же ABI), собираемый в `build/cpu`:

```bash
export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
cmake -S . -B build/cpu -G Ninja -DSTELL_CUDA=OFF
cmake --build build/cpu --target stell_mock_core
```

Запуск (Python-префикс `~/.local/stell-conda/envs/serve`):

```bash
PYTHONPATH=python ~/.local/stell-conda/envs/serve/bin/python -m stell_serve --port 8080
```

Путь к .so: `--engine-lib` или переменная `STELL_ENGINE_LIB`; по умолчанию —
mock в `build/cpu/src/serve/libstell_mock_core.so`.

Зависимости: `pip install -r requirements.txt` (в префикс serve, не в
envs/stell и не в системные места).

## Замечания по решению SSE

sse-starlette установлен как зависимость, но wire-формат SSE форматируется
вручную (`dialects.sse`): оба диалекта требуют точного формата кадра
(OpenAI: `data: {...}` + `data: [DONE]`; Anthropic: именованные `event:`),
которым ручной генератор даёт полный контроль. sse-starlette оставлен для
будущих keepalive/heartbeat политик.

## Статус

Реализовано и покрыто тестами (`tests/serve/`): контракты обоих API,
стриминг, tool-calls, CORS, health/metrics, биндинг ABI. Реальный
`libstell_core.so` от потока A пока не существует — все прогоны на mock.
