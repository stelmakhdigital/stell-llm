# Дизайн-документы (stell-llm)

Этот каталог — оглавление дизайн-документов по разделу 2 PLAN.md
(целевая архитектура). Пока кода нет, документы — **заготовки-оглавления**:
здесь фиксируется состав и границы каждого модуля по PLAN.md §2/§4,
но не описывается нереализованное как готовое.

Документы наполняются соответствующими ролями (core/tiers/parallel/models/serve)
по мере готовности модулей; контракты на стыках — только через ADR-08…ADR-12.

## Состав (план)

- [engine.md](engine.md) — планировщик continuous batching (≥8 сессий), пулы слотов,
  CUDA Graphs, спекулятивный контур (поток A, роль `core`).
- [kernels.md](kernels.md) — SM70-attention (FP32 QK/PV), MoE gather/scatter,
  dequant NVFP4/W4A16, KV-квант E4M3, драфтер-ядра (поток A, роль `core`).
- [tiers.md](tiers.md) — Tier Manager: VRAM-slots / pinned RAM / NVMe (io_uring),
  политика LRU/LCU, router-lookahead prefetch, метрики (поток B, роль `tiers`).
- [models.md](models.md) — декларативные описания архитектур, реестр dense/MoE,
  лоадеры safetensors/GGUF/NVFP4-W4A16/stell-tier, конвертеры, golden-тесты
  (поток D, роль `models`).
- [parallel.md](parallel.md) — NCCL TP/EP/PP по NVLink, комбинированные схемы,
  KV-streaming, CPU-эмуляция комм-паттернов (поток C, роль `parallel`).
- [serve.md](serve.md) — OpenAI/Anthropic-совместимый API, SSE, tool-calls, CORS,
  health/метрики, дашборд, калибровщик, bench + quality-gates
  (поток E, роль `serve`).

## Ссылки

- План разработки и критерии приёмки: [../../PLAN.md](../../PLAN.md)
- Контракты модулей: [../adr/index.md](../adr/index.md) (ADR-08…ADR-12)
- Сборка и запуск: [../../README.md](../../README.md)

## Интеграционные срезы (PLAN.md §4)

1. A×D — dense-модель в 1 GPU, оба API живых.
2. +C — TP4 dense по NVLink.
3. +B — MoE с tiering (модель > VRAM).
4. +E-финал — спекуляция, калибровка, дашборд, quality-gates.
