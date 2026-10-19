# models — дизайн-документ (заготовка)

> Статус: заготовка. Модуль не реализован. Контракты: ADR-09 (схема/реестр), ADR-03, ADR-12.

## Уточнения контракта (round 1, T-20261017200354)

- GGUF-нейминг (ADR-09 «Уточнение»): `source_pattern` — в конвенции имён входного
  формата, лоадер сравнивает буквально; GGUF-описания генерируются конвертером
  с нативными именами тензоров формата GGUF (`blk.*`; канон ADR-09),
  без неявного HF↔GGUF маппинга.
- Golden: режим фикстуры — `greedy` (argmax); bench-«T=1» = `temperature=1.0`
  (не greedy) — см. ADR-09 «Уточнение».
- Известное упрощение: конвертер пишет `n_tensors = n_groups` (см. ADR-12
  «Уточнение»); `header_crc32` конвертера — байты [0x00..0x2C) (канон ADR-12).

## Уточнения контракта (round 2, T-20261017200354)

- Канон NVFP4/W4A16 — **block32** (ADR-09 «Уточнение round 2»): квантованные
  группы — только линейные веса; F32/F16-группы — норм/router/layernorm;
  dtype — атрибут `tensor_group`, лоадер читает dtype из группы без
  fallback-догадок. Group-16-фикстуры kernels — legacy (не-контракт).
- Прод-токенизатор: byte-level BPE реализуется здесь (`ITokenizer`,
  `tokenizer.json` в каталоге модели, описан в реестре); инжект в engine —
  `Scheduler::set_tokenizer`; byte-fallback engine остаётся контрактной точкой.

## Уточнения контракта (round 3, T-20261017200354)

- Потоковая конвертация в stell-tier: `StellTierConverter::convert_streaming()`
  — память O(максимальной группы) (одна группа + source-буфер, не O(размера
  модели)); flush по группам (порог `flush_group_bytes`, дефолт 8 МиБ);
  атомарная финализация `tmp → rename` + self-check собственным reader'ом;
  fixup индекса (CRC-placeholder до данных, перезапись 64-байтовой записи
  после каждого слота — правило ADR-12 «Уточнение round 3»).
- `StellTierWriterConfig` — параметры тиров принимаются извне (согласованы
  с `TierConfig` tiers; границы тиров задаёт tiers-конфиг); writer только
  валидирует план: группа не влезает в `vram_slot_size` → `SchemaMismatch`
  (fail-fast); `ram_capacity` в layout семантики не имеет; `slot_align`/
  `index_align` фиксированы layout v1 (4096/512), кастомные значения →
  `SchemaMismatch` (новая версия формата — только через ADR-12).
- Публичный CLI `stell_convert_streaming <desc.json> <source> <out_dir>
  [--vram-slot-size B] [--ram-capacity B] [--flush-mb N]`; старый
  материализующий `convert()` не тронут (сравнение/совместимость).
- Прогон на реальных весах — через env `STELL_R7_REAL_WEIGHTS_DIR`
  (каталог safetensors-шардов HF) + `STELL_R7_REAL_DESC_JSON` (описание
  ADR-09 схемы 1.0), без правки кода; веса в git не попадают.

## Объём (из PLAN.md §4, поток D)

- Реестр архитектур: dense llama/qwen-класс; MoE glm/qwen-moe/deepseek-класс
  (минимум 3 архитектуры: 1 dense + 2 MoE).
- Лоадеры: safetensors (HF), GGUF, NVFP4-W4A16, stell-tier (ADR-03); конвертеры.
- Golden-тесты логитов против HF-референсов (CPU — локально, CUDA — на сервере).

## Готовность

- 3 архитектуры в полном контуре с tiering; golden-тесты проходят;
  все 4 формата загружаются и конвертируются.
