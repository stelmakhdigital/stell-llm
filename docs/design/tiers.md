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
  публичный verify-проход слотов — обязательный механизм периодической
  целостности вне пути декода. Точка гарантии — writer (CRC при записи).
- Verify-проход — метод `TierManager::verify_slots()` (не harness-команда;
  интерфейс `ITierManager` ADR-08 не расширяется — прецедент
  `read_ram_slot`/`set_layer_map`); CRC-once (успешный проход помечает
  группы; hot path с verify=on не перепроверяет); метрика `crc_checks()`;
  OR-семантика флага (config || env); сырые reader'ы — verify=true
  по умолчанию (см. ADR-12 «Уточнение round 3»).
- Конфигурация тиров под профиль co-tenancy («свободно 2–11 ГБ на V100»,
  RAM-тир основной, VRAM-тир мал): `tier_config_from_env(base)` — env
  поверх базового конфига (`STELL_TIER_VRAM_SLOTS`,
  `STELL_TIER_VRAM_SLOT_BYTES`, `STELL_TIER_RAM_MB`, `STELL_TIER_LOOKAHEAD`,
  `STELL_TIER_HISTORY_CAP`, `STELL_TIER_VERIFY_CRC`, `STELL_TIER_POLICY`
  (lru|lcu), `STELL_TIER_NVME_LAYOUT`; нечисловые значения → базовое);
  `tier_config_co_tenancy(vram_budget_bytes, plan_bytes, slot_bytes)` —
  именованный профиль: VRAM-слоты = бюджет/slot (мин. 1, больше плана не
  выделяем), RAM = план целиком, NVMe = весь план; `slot_bytes` не кратен
  4096 → `TierError(PlanMismatch)`.

## Решения round 4 (T-20261017200354, planner [Q6]/[Q2] → [R2])

- **O_DIRECT-бэкенд `TierNvmeFile`** ([Q6]): IORING_OP_READ на fd, открытом
  с `O_DIRECT`; буферы/офсеты уже кратны и выровнены 4096 по layout v1;
  включается флагом конфигурации (`STELL_TIER_ODIRECT` / `TierConfig`),
  buffered-путь остаётся дефолтом; риски O_DIRECT (выравнивание, отсутствие
  readahead) локализованы в одном бэкенде с fallback-прецедентом
  (io_uring → pread). Обоснование: fadvise не гарантирует eviction под
  давлением page cache co-tenancy; O_DIRECT даёт детерминированный cold
  и убирает двойное буферирование. Тест: побитовое равенство O_DIRECT- и
  buffered-чтения одного layout-файла.
- **Deep-prefetch** ([Q2]): параметр глубины k групп в полёте (k=1/2/4)
  в tier-профиле (router-lookahead уже даёт прогноз; пауза между слоями
  перекрывается чтением k слоёв вперёд).
- **Матрица серверного прогона** [R2]: slot size {16 МиБ, 4 МиБ} × I/O
  {buffered, O_DIRECT} × k {1, 4}, профиль остатка в `bench/results/`,
  co-tenancy-пометка обязательна. Замер — **не вердикт по цели §4**
  (p95 ≤ 5 мс): вердикт — только по чистому окну; цель не отменяется.
  Целевой профиль: «слоты 4 МиБ + prefetch k≥4 + O_DIRECT».

## Объём (из PLAN.md §4, поток B)

- Tier Manager: VRAM-slots / pinned RAM / NVMe (io_uring).
- stell-tier layout на NVMe (ADR-12); конвертер — со стороны models (ADR-03).
- Политика LRU/LCU со статистикой маршрутизатора; router-lookahead prefetch.
- Метрики: hit-rate prefetch, паузы слоёв p50/p95/p99, пропускная способность тиров (ADR-08).

## Готовность

- MoE > VRAM (FP16) декодируется через tier-контур; p95-пауза ≤ 5 мс;
  prefetch ≥ 60% попаданий (фиксируется в bench/results).
