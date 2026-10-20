# Models — раунд 7 (итерация 8, поток D): qk-norm + router_renorm

ADR-09 amendment round 7 («Уточнение (round 7…)»): сняты отказы round 6 —
per-head RMSNorm q/k и `norm_topk_prob=false`; перевод перекрёстной сверки
`moe_intermediate_size` на фактические формы экспертных тензоров.
Сторона конвертера (`hf_to_description`); ops_runner-семантика — поток core
(стыковые имена зафиксированы ниже, отступать нельзя).

## [R1] per-head RMSNorm q/k — генерация групп

- Детекция по факту шардов, для ВСЕХ архитектур (не по имени): слой с
  `self_attn.q_norm.weight` И `self_attn.k_norm.weight` → группы
  `l{layer}.attn.q_norm` (форма `[d_head]`) и `l{layer}.attn.k_norm`
  (форма `[d_kv]`, == `[d_head]` в поддержанных arch); роль `attn`;
  dtype — по канону норм: dtype тензора источника (F32/F16, BF16 → F16
  clamp-сатурация и счётчик — `source_group_dtype`, [R5] round 5).
- Ровно все слои с qk-нормами → один шаблон с cardinality `{layer: L}`
  (компактно, как у attn-норм); смешанный набор (часть слоёв без —
  слой без ОБЕИХ групп легитимен, round 7) → literal per-layer группы
  (pad3-конвенция, как в per-layer плане [R3] round 5).
- Частичный набор в слое (q есть, k нет или наоборот) → честный
  `SchemaMismatch` (односторонне нормализованный attention не имеет
  покрытой семантики).
- **eps**: из `config.json` `rms_norm_eps` → параметр op attention
  **`qk_eps`** (float, JSON-число). Пишется ТОЛЬКО при наличии qk-групп;
  qk-тензоры есть, а `rms_norm_eps` в config отсутствует → честный
  `SchemaMismatch` (не дефолт-догадка). Без qk-групп параметр не пишется —
  прежние описания побитово прежние.
- Стыковые имена (для core): группы `l{layer}.attn.q_norm` /
  `l{layer}.attn.k_norm`, параметр `qk_eps` у op attention.

## [R2] router_renorm + перекрёстная сверка

- Отказ `norm_topk_prob=false` снят: явный `false` в config (только для
  MoE-архитектур) → top-level поле описания `"router_renorm": false`;
  при true/отсутствии ключа поле НЕ пишется (дефолт true — семантика
  round 1, канонизированная golden'ом; старые описания и побитовые golden
  не меняются).
- Контракт: `MoeSpec::router_renorm` (bool, default true) в
  `description.hpp`; `parse_model_description` читает top-level
  `"router_renorm"` (bool; при class=dense и false → честный
  `DescriptionError`; отсутствие поля → true). Чтение
  `MoeSpec::router_renorm` в model_spec_bridge (src/engine) — поток core.
- **Перекрёстная сверка `moe_intermediate_size`** переведена с d_ff
  шардов (expert 0 первого MoE-слоя) на фактическую форму ВСЕХ
  экспертных тензоров первого MoE-слоя — `e{expert}.gate/up/down` для
  каждого эксперта (gate/up `[d_ff, d_model]`, down `[d_model, d_ff]`);
  несогласованность config ↔ тензоры или экспертов между собой → честный
  `SchemaMismatch` с именем тензора. Тест на Qwen3Moe-конфиге: эксперт 2
  с `down [8, 16]` при d_ff=8 → отказ.

## Версионирование схемы — решение

Поле `router_renorm` опционально с дефолтом true: старые описания (без
поля) парсятся без изменений, побитовые golden не меняются, поведение по
умолчанию прежнее — обратная совместимость полная. **schema остаётся
"1.0"** (minor-бамп не требуется): правило round 7 — «minor-бамп, если
правило требует»; правило §1 («неизвестные поля — ошибка парсинга») на
практике — игнорирование неизвестных top-level ключей парсером
(зафиксировано в REPORT-ROUND6.md), и прецедент round 6 (`dtype_orig`,
top-level поле без бампа) согласован. Бамп сломал бы жёсткое ограничение
«существующие тесты и golden-файлы НЕИЗМЕННЫ» (тесты фиксируют
`schema=="1.0"` у генерата). Новое поле читается явно обновлённым
парсером — «неизвестным» он его не считает; старые парсеры игнорируют
ключ (документированное поведение).

## Тесты (test_hf_description.cpp, ctest 53/53)

Фикстуры: `qwen3_moe_qknorm_hf_tensors` (теперь happy-path round 7),
`qwen3_moe_qknorm_partial_hf_tensors` (только q_norm),
`write_config_qwen3(..., rms_eps=false)`; прежняя Qwen3Moe-фикстура без
qk-норм и с norm_topk=true — побитово прежнее описание (контроль
обратной совместимости).

- (а) qk-norm + norm_topk=true → генерация без SchemaMismatch: 47 групп,
  `l000.attn.q_norm` ← `model.layers.0.self_attn.q_norm.weight` ([4],
  F16), параметр `qk_eps == 1e-06` у op attention; roundtrip побитово.
- (а-2) частичный набор (только q_norm) → SchemaMismatch (сообщение
  называет k_norm).
- (а-3) qk-тензоры + config без `rms_norm_eps` → SchemaMismatch.
- (б) norm_topk_prob=false → генерация без SchemaMismatch, top-level
  `"router_renorm": false`, парсер → `MoeSpec::router_renorm == false`;
  roundtrip побитово.
- (б-2) дефолт: поле НЕ пишется (побитовые golden не меняются), парсер →
  true.
- (б-4) парсер: class=dense + router_renorm=false → DescriptionError.
- (в) `moe_intermediate_size=16` ≠ факту шардов (d_ff=8) →
  SchemaMismatch (сообщение называет ключ).
- (в-2) форма другого эксперта (`down [8, 16]`) ≠ факту первого MoE-слоя
  → SchemaMismatch (сообщение называет `experts.2.down`).

## НЕ сделано (честно)

- ops_runner-семантика qk-norm и router-пути без renorm — поток core
  (roundtrip-тест фиксирует побитовое совпадение layout ↔ прямому пути
  того же плана; логит-референс с per-head RMSNorm — judge I3 core).
- Смешанный qk-набор (часть слоёв с нормами) покрыт реализацией (literal
  группы) и энергичной валидацией §6, отдельного теста нет.
- Реальный чекпоинт Qwen3-30B-A3B (61 ГБ, вне git) — прогон не выполнялся;
  проверка на синтетической Qwen3Moe-фикстуре.
