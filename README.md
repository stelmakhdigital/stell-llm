# stell-llm

Собственный рантайм инференса LLM для NVIDIA Tesla V100 (SM70, SXM2, NVLink).

Идея: **любая модель** (dense и MoE) + **tiering весов VRAM → RAM → NVMe** (как в project-maya)
+ **Volta-ядра и NVLink-параллелизм** уровня 1Cat-vLLM, без недостатков обоих прототипов.

Дорожная карта, архитектура и критерии приёмки: [PLAN.md](PLAN.md).

Статус: **раунд 1 выполнен** — CPU-контуры всех модулей реализованы и протестированы,
интерфейсные контракты зафиксированы и уточнены (ADR-08…ADR-12); CUDA SM70-конфигурация
собрана на целевом сервере. **Не сделано (честно): GPU-прогоны ядер/комм, golden-тесты
против HF-референсов, интеграционные срезы (замена stub-адаптеров core на реальные
модули), quality-gates на реальных весах.** Это большие многораундовые задачи (PLAN.md §4).

## Документация

- ADR (архитектурные решения и контракты модулей): [docs/adr/index.md](docs/adr/index.md)
  — ADR-01…ADR-07 (принятые технологические решения) и **ADR-08…ADR-12** — контракты
  `engine↔tiers`, `engine↔models`, `engine↔parallel`, `engine↔serve`, `tiers↔models`.
  Уточнения раунда 1 (задача T-20261017200354) — разделы «Уточнение (round 1)»
  в конце ADR-08…ADR-12; формат stell-tier layout v1 не менялся.
- Дизайн-документы (заготовки-оглавления по потокам A–E, с уточнениями контрактов):
  [docs/design/README.md](docs/design/README.md).

## Что реализовано в master после раунда 1

Честный статус по модулям (все CPU-пути проходят тесты локально, без GPU):

- **engine (поток A)** — планировщик continuous batching, пулы слотов (≥8 сессий, LIFO),
  спекулятивный контур (draft + FP32-верификация + автоотключение), KV-stream,
  ExecutionPlan (`layout_version`), decode-executor (CUDA Graph replay / eager-фолббек,
  CPU), C ABI `capi.h` (`stell_submit/poll/cancel/step/metrics`), минимальный JSON.
- **kernels (поток A)** — CPU-референсы: attention FP32 QK/PV, MoE gather/scatter,
  dequant NVFP4/W4A16, KV-квант E4M3, драфт; SM70-ядра `.cu` и dispatch; операторные
  replay-тесты против независимых референсов.
- **tiers (поток B)** — Tier Manager: VRAM-slots (+ DMA под STELL_CUDA), pinned RAM (LRU),
  NVMe-тир io_uring (ABI-подмножество без liburing + fallback), stell-tier layout
  writer/reader (ADR-12), политика LRU/LCU + router-статистика, prefetch-предиктор,
  метрики hit-rate/пауз p50/p95/p99.
- **models (поток D)** — JSON-схема описания 1.0 (строгий парсер), реестр 3 архитектуры
  (1 dense + 2 MoE), лоадеры safetensors/GGUF-v3/NVFP4-W4A16/stell-tier, фабрика,
  конвертер в stell-tier (writer у models, reader у tiers — ADR-12 §4).
- **parallel (поток C)** — комм-паттерны (AllReduce/AllGather/ExpertAllToAll/Chain/KvStream),
  кодек F16/E4M3 с детерминированной редукцией, план деления ранков/слотов,
  CPU-эмулятор; NCCL-заготовка под STELL_CUDA; golden TP2/TP4 == TP1 на CPU.
- **serve + ui + calibrate + bench (поток E)** — FastAPI: `/v1/chat/completions`
  (OpenAI) и `/v1/messages` (Anthropic), SSE, tool-calls, CORS, `/health`, `/metrics`,
  cancel; ctypes-биндинг C ABI; mock-движок CPU-fallback (тот же ABI); дашборд
  (`src/ui`: VRAM-карта, паузы слоёв, hit-rate, сессии, логи); калибровщик-арбитр
  (вкл/выкл по замерам); bench-контракт PLAN.md §5 + каркасы quality-gates
  (GSM8K-32/HumanEval-32/PPL).

Тесты после последнего интеграционного прогона (раунд 2, T-20261017200354):
`ctest 40/40` (parallel/tiers/core/models), `pytest 75/75` (serve); чистая
сборка CPU без предупреждений. Счётчики раунда 1 (`ctest 29/29`, `pytest 40/40`)
устарели — см. worker-отчёты раунда 2 (`.pi/pipeline/T-20261017200354/`).

## Pending (честно, не сделано)

- **GPU-прогоны** — CUDA SM70-конфигурация собрана на сервере (`ssh -p 2214 192.168.1.114`,
  nvcc 12.8), но ядра/комм на GPU не запускались и не замерялись (сервер совмещён
  с чужим процессом: свободная VRAM ~2–11 GB на V100 — не мешать ему).
- **Golden-логиты против HF-референсов** — каркас и схема фикстур готовы
  (`tests/models/golden/`, честный SKIP без фикстур); нужны веса и прогон на сервере.
- **Интеграционные срезы 1→4** (PLAN.md §4) — замена stub-адаптеров core на реальные
  tiers/models/parallel, MoE > VRAM через tier-контур, TP4 ≥3.2×, quality-gates.
- **p95 ≤ 5 мс на реальном NVMe** и hit-rate на реальных весах — замеряется в срезе 3.

## Сборка

- **CPU-конфигурация (локально, без GPU)** — обязательный путь CI и юнит-тестов
  (ADR-07); тулчейн — user-level (`~/.local/stell-conda/envs/stell`):
  ```bash
  export PATH=~/.local/stell-conda/envs/stell/bin:$PATH
  export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++
  cmake -S . -B build/cpu -G Ninja -DSTELL_CUDA=OFF
  cmake --build build/cpu -j
  ctest --test-dir build/cpu --output-on-failure
  ```
  Python-тесты serve (префикс `~/.local/stell-conda/envs/serve`):
  ```bash
  PYTHONPATH=python ~/.local/stell-conda/envs/serve/bin/python -m pytest tests/serve -q
  ```
- **CUDA-конфигурация (целевой сервер, SM70)** — только на сервере, с учётом
  совмещения (чужой процесс держит ~21–30 GB VRAM на каждой V100 — не мешать ему):
  ```bash
  cmake -S . -B build/cuda -G Ninja -DSTELL_CUDA=ON
  cmake --build build/cuda -j
  ```
  Архитектура SM70 задаётся внутри `CMakeLists.txt` (`CMAKE_CUDA_ARCHITECTURES 70`);
  NCCL-путь parallel собирается только при наличии `libnccl` (ADR-10 §4).

## Запуск API и дашборда

Один Python-процесс (FastAPI, тонкий контроль-плейн) на порту 8080 с in-process
движком `libstell_core.so` через ctypes (ADR-11): `/v1/chat/completions`,
`/v1/messages`, `/health`, `/metrics`, `/ui` (дашборд). Раунд 1 работает на
mock-движке CPU-fallback (тот же C ABI); реальный движок вклеивается на
интеграционных срезах.

```bash
cmake --build build/cpu --target stell_mock_core
PYTHONPATH=python ~/.local/stell-conda/envs/serve/bin/python \
  python -m stell_serve --port 8080
```

Путь к .so: `--engine-lib` или `STELL_ENGINE_LIB` (по умолчанию — mock в
`build/cpu/src/serve/libstell_mock_core.so`). Зависимости: `pip install -r
python/requirements.txt`. Подробности — [python/README.md](python/README.md).

## Известные ограничения

- GPU-прогоны, golden-HF, интеграционные срезы, quality-gates на реальных весах —
  pending (см. выше); всё, что не помечено как реализованное, — целевая
  архитектура [docs/design/](docs/design/README.md).
- Отклонение, найденное в раунде кларификаций (ADR-12 «Уточнение»): reader/writer
  stell-tier в `src/tiers/src/layout.cpp` считают `header_crc32` по байтам
  [0x00..0x28), канон ADR-12 — [0x00..0x2C) (как у конвертера models).
  Исправление — round 2 (роль tiers); до исправления конвертированные файлы
  не скармливать NVMe-тиру.
- Multi-GPU CUDA-прогоны — только на целевом сервере, с ограничением по
  совмещению с чужим процессом.
