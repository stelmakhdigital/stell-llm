# ADR — Architecture Decision Records (stell-llm)

Все архитектурные решения фиксируются здесь. Изменение контрактов
ADR-08…ADR-12 — только через ADR и согласование с worker
(worker_roles.md, «Общие правила ролей»).

| # | Заголовок | Статус |
|---|---|---|
| [ADR-01](ADR-01-core-language.md) | Ядро — C++20 + CUDA 12.x (SM70) | Принято |
| [ADR-02](ADR-02-comms-nccl.md) | Коммуникации — NCCL поверх NVLink; P2P DMA | Принято |
| [ADR-03](ADR-03-weight-formats.md) | Форматы весов — safetensors/GGUF/stell-tier | Принято |
| [ADR-04](ADR-04-kv-cache.md) | KV-кеш — FP16 базово; E4M3 опционально | Принято |
| [ADR-05](ADR-05-cuda-graphs.md) | CUDA Graph для decode; фиксированные пулы | Принято |
| [ADR-06](ADR-06-api-port.md) | API — один процесс на порту 8080 | Принято |
| [ADR-07](ADR-07-build.md) | Сборка — CMake; CI на CPU-путях | Принято |
| [ADR-08](ADR-08-engine-tiers.md) | Контракт engine ↔ tiers — тензорные группы | Принято |
| [ADR-09](ADR-09-engine-models.md) | Контракт engine ↔ models — декларативные описания | Принято |
| [ADR-10](ADR-10-engine-parallel.md) | Контракт engine ↔ parallel — TP/EP/PP/KV-стрим | Принято |
| [ADR-11](ADR-11-engine-serve.md) | Контракт engine ↔ serve — C ABI, события, метрики | Принято |
| [ADR-12](ADR-12-tiers-models.md) | Контракт tiers ↔ models — stell-tier layout | Принято |

Шаблон ADR: Заголовок, Статус, Контекст, Решение, Альтернативы, Последствия.

Статусы: `Proposed` → `Принято` → (иногда) `Заменён ADR-XX`.
