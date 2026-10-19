# tiers — дизайн-документ (заготовка)

> Статус: заготовка. Модуль не реализован. Контракты: ADR-08 (интерфейс), ADR-12 (layout).

## Уточнения контракта (round 1, T-20261017200354)

- stell-tier layout v1 как реализован (`src/tiers/src/layout.cpp`): заголовок —
  нулевой паддинг до 512 байт; имя группы в записи индекса — 16 байт;
  CRC группы — по слоту целиком (payload + padding).
- Открытое исправление кода (round 2, роль tiers): в `layout.cpp` граница
  `header_crc32` — `crc32(hdr, 0x28)`, канон ADR-12 (уточнение round 1) —
  байты [0x00..0x2C) (как у конвертера models). До исправления конвертированные
  файлы не скармливать NVMe-тиру.
- Слой→группы: `TierManager::set_layer_map` (расширение на реализации);
  `layout_version` — величина engine (`ExecutionPlan`, детерминированный хэш
  плана) — см. ADR-08 «Уточнение».

## Уточнения контракта (round 3, T-20261017200354)

- CRC-политика (ADR-12 §2.3 «Уточнение round 3»): reader всегда проверяет
  `header_crc32`; payload-CRC при pin/read — только при
  `STELL_TIER_VERIFY_CRC=1` (env/конфиг, дефолт off в hot path);
  публичный verify-проход слотов (`ITierManager`/harness-команда) —
  обязательный механизм периодической целостности вне пути декода.
  Точка гарантии — writer (CRC при записи).

## Объём (из PLAN.md §4, поток B)

- Tier Manager: VRAM-slots / pinned RAM / NVMe (io_uring).
- stell-tier layout на NVMe (ADR-12); конвертер — со стороны models (ADR-03).
- Политика LRU/LCU со статистикой маршрутизатора; router-lookahead prefetch.
- Метрики: hit-rate prefetch, паузы слоёв p50/p95/p99, пропускная способность тиров (ADR-08).

## Готовность

- MoE > VRAM (FP16) декодируется через tier-контур; p95-пауза ≤ 5 мс;
  prefetch ≥ 60% попаданий (фиксируется в bench/results).
