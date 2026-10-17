# tiers — дизайн-документ (заготовка)

> Статус: заготовка. Модуль не реализован. Контракты: ADR-08 (интерфейс), ADR-12 (layout).

## Объём (из PLAN.md §4, поток B)

- Tier Manager: VRAM-slots / pinned RAM / NVMe (io_uring).
- stell-tier layout на NVMe (ADR-12); конвертер — со стороны models (ADR-03).
- Политика LRU/LCU со статистикой маршрутизатора; router-lookahead prefetch.
- Метрики: hit-rate prefetch, паузы слоёв p50/p95/p99, пропускная способность тиров (ADR-08).

## Готовность

- MoE > VRAM (FP16) декодируется через tier-контур; p95-пауза ≤ 5 мс;
  prefetch ≥ 60% попаданий (фиксируется в bench/results).
