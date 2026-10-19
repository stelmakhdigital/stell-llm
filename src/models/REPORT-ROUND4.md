# models — отчёт раунда 4 ([R4] hf_to_description, [R3-подготовка], [R6])

Ветка `agent/models/5` (от master `c51da0e`). Worktree изолирован; чужие каталоги не тронуты.

## Тесты до/после

| Срез | Результат |
|---|---|
| до (master `c51da0e`, CPU) | ctest **45/45** |
| после | ctest **47/47** (+ `models_hf_description`, + `models_weights_dir_mixed`; все старые не тронуты и проходят) |

## Файлы + суть

**[R4] `hf_to_description`** (по прецеденту `gguf_to_description`):
- `src/models/src/hf_description.cpp` — конвертер HF-чекпоинта (каталог
  `config.json` + safetensors-шарды; либо одиночный `.safetensors`) →
  `ModelDescription` (схема ADR-09 «1.0»). Имена тензоров — **нативные HF**;
  `source_pattern` — те же имена, лоадер/конвертер сравнивают буквально.
  Диспетчер по `architectures[0]`:
  - `LlamaForCausalLM` / `Qwen2ForCausalLM` — dense;
  - `Qwen2MoeForCausalLM` / `Qwen1.5MoeForCausalLM` / `OlmoeForCausalLM` —
    MoE, поэертельные тензоры экспертов; имя роутера (`mlp.router.weight` /
    `mlp.gate.weight`) и суффикс экспертных тензоров (`_proj` / пустой)
    выбираются **по факту в шардах** (не по догадкам);
  - `MixtralForCausalLM` — MoE, стековные `block_sparse_moe.w1/w3/w2`,
    `source_slice = "{expert}"` (канон R9).
  config.json: `hidden_size`, `intermediate_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `vocab_size` (умолчание —
  dim0 тензора embed), `rms_norm_eps`, `head_dim`, rope-параметры
  (`rope_theta` → параметр ops attention).
- `src/models/src/description_json.hpp` — общие JSON-писатели групп
  (единственный источник сериализации; `gguf_description.cpp` переключён на
  него — дублирование снято).
- `src/models/include/stell/models/convert.hpp` — объявление + контракт.
- `src/models/tools/hf_description.cpp` + CLI `stell_hf_description
  <hf_dir> [F16|NVFP4|W4A16]` — генерация `STELL_R7_REAL_DESC_JSON`.

**[R6] смешанный шардовый источник** ([Q7]):
- `src/models/src/file_weights_source.hpp` — `FileWeightsSource` принимает
  каталог шардов: `merge_safetensors_shards()` — одна объединённая карта
  тензоров (`TensorInfo::shard_index`, чтение payload из своего шарда);
  один `IWeightsSource` обслуживает весь смешанный план. Шард-каркас
  конвертера (`ShardWeightsSource`) переведён на ту же функцию — один
  контракт для лоадеров и конвертера.
- `tests/models/test_weights_dir_mixed.cpp` — см. ниже.

**[R3-подготовка] BF16 + честные границы путей**:
- BF16-тензора HF-чекпоинтов (Qwen/Mixtral/OLMoE) маппятся в F16 с флагом
  (`TensorInfo::bf16`, `bf16_bits_to_f32` в `quant.hpp`); BF16→F16 —
  допустимая конвертация лоадера наравне с F32→F16; BF16→F32 — честный
  `UnsupportedQuant`.
- `plain_fill_slot` — общий путь `fill_plain` (лоадеры) и `make_slot`
  (конвертер): устранено копирование байтов без конверсии в plain-ветке
  конвертера (раньше F32/BF16-источник F16-группы дал бы расхождение путей).
- `convert()` (материализующий) на каталоге шардов — честный
  `SchemaMismatch` «используйте convert_streaming».
- Профиль co-tenancy в real-weights ветке `test_convert_streaming`:
  VRAM-слот **2 ГиБ** (низ профиля «свободно 2–11 ГБ на V100»; прежние
  64 МиБ несовместимы с реальным планом — embed [32000,4096] ≈ 0.5 ГиБ),
  RAM-тир 11 ГиБ (верх профиля).

## Тесты (новые)

1. `models_hf_description` ([R4]):
   - синтетические каталоги HF (config.json + 2 шарда, fixture от
     синтетического писателя): dense LlamaForCausalLM (F16), dense NVFP4
     (смешанный план: матрицы NVFP4, нормы F32, embed/head F16), MoE
     Qwen2MoeForCausalLM (NVFP4: роутер F32, эксперты NVFP4), MoE
     MixtralForCausalLM (стековные w1/w3/w2, `source_slice "{expert}"`);
   - **roundtrip**: конвертация каталога шардов → stell-tier layout →
     логиты интерпретатора (engine, мост ADR-09 §4) == прямому пути того же
     плана — **побитно** (все слоты и все логиты);
   - BF16→F16 конверсия — побайтово == референсу, хвост слота — нули;
   - error-кейсы: нет `config.json` → `LoaderError(SchemaMismatch)`;
     шард не соответствует плану (тензора нет у группы) → `TensorNotFound`;
     форма не согласована → `SchemaMismatch`; неизвестная архитектура /
     профиль → `SchemaMismatch`; каталог без шардов → `IoFailure`.
2. `models_weights_dir_mixed` ([R6]):
   - один `IWeightsSource` на каталоге шардов, где в **каждом** шарде
     соседствуют группы разного dtype (NVFP4 block32 + W4A16 + F32/F16;
     разбиение по шардам сознательно НЕ по dtype) — чтение одним
     источником, побитово == по-групповому чтению (объединённый файл);
   - **отсутствие «профильных» ветвлений**: два описания, различающиеся
     только `dtype_weights` (NVFP4 vs W4A16, групповые dtype идентичны), —
     байты чтения побитово идентичны (источник диспетчирует по dtype
     **группы**, ADR-09 round 2);
   - plain-лоадер (Safetensors) на том же каталоге: побитно == одиночному
     файлу; ошибки каталога: нет шардов → `IoFailure`, тензор в двух
     шардах → `SchemaMismatch`.

## [R3] — подготовка реальных весов: статус и команды worker'а

**Механизм env-проверен E2E** (на синтетическом HF-каталоге, имитирующем
реальный чекпоинт): без env — честный `PENDING` (не молчаливый успех);
с `STELL_R7_REAL_WEIGHTS_DIR` + `STELL_R7_REAL_DESC_JSON` — конвейер
проходит: описание → streaming-конвертация (профиль co-tenancy) → layout
читается reader'ами models **и** tiers. Код не менялся для включения
(переключение только env).

**Точные команды worker'а** (сервер, после merge; веса — в `~/models`,
вне репозитория):

```bash
export PATH=~/.local/stell-conda/envs/hf/bin:$PATH
mkdir -p ~/models

# Вариант A (основной): Qwen1.5-MoE-A2.7B-Chat — Qwen1.5MoeForCausalLM
# (поддержан hf_to_description; ~28 ГБ BF16 — ВЫШЕ гайда ≤~10 ГБ, но NVMe
# ~95 ГБ свободно; отклонение честно фиксировать в отчёте worker'а)
hf download Qwen/Qwen1.5-MoE-A2.7B-Chat --local-dir ~/models/qwen15-moe-a2.7b-chat

# Вариант B (экономнее, ~13 ГБ BF16): allenai/OLMoE-1B-7B-0924 —
# OlmoeForCausalLM (поэтельный MoE; роутер mlp.gate). ВНИМАНИЕ: фактические
# имена тензоров OLMoE конвертер проверяет честно — при расхождении будет
# честный TensorNotFound → pending с точным именем отсутствующего тензора.
hf download allenai/OLMoE-1B-7B-0924 --local-dir ~/models/olmoe-1b-7b

# (опционально) дешёвый dense-смоук ~1 ГБ: Qwen/Qwen2-0.5B-Instruct (tie_word_embeddings:
# lm_head отсутствует → группа head не генерируется, логиты через embedding)

# 1) генерация описания (CLI round 4; профиль W4A16 — co-tenancy-канон ADR-09 block32)
./build/cpu/src/models/stell_hf_description ~/models/qwen15-moe-a2.7b-chat W4A16 \
    > ~/models/qwen15-moe.desc.json

# 2) прогон конвейера [R7] (env, без правки кода): streaming-конвертация
#    → tier-профиль co-tenancy (VRAM-слот 2 ГиБ / RAM 11 ГиБ — в тесте)
export STELL_R7_REAL_WEIGHTS_DIR=~/models/qwen15-moe-a2.7b-chat
export STELL_R7_REAL_DESC_JSON=~/models/qwen15-moe.desc.json
ctest --test-dir build/cpu -R models_convert_streaming --output-on-failure

#    метрики p95-пауз/префетч/hit-rate — bench-срез tiers (co-tenancy-пометка
#    обязательна; вердикт по цели §4 — только чистое окно, planner [Q2])

# 3) CLI-путь с явным tiers-профилем (тот же writer):
./build/cpu/src/models/stell_convert_streaming ~/models/qwen15-moe.desc.json \
    ~/models/qwen15-moe-a2.7b-chat /tmp/stell_tier_out \
    --vram-slot-size 2147483648 --ram-capacity 11874254848
```

**Проверка приёмки «веса НЕ попадают в git»**: скачивание — в `~/models`
(вне worktree/репозитория, поэтому в git не попадает по конструкции;
`.gitignore` правок не требует — подтверждено чтением корневого
`.gitignore`: там только служебные/сборочные пути). Контроль:
`git status --porcelain` после прогона обязан быть пустым (кроме ожидаемых
коммитов), `git count-objects -v` — без роста крупных blob'ов.

**Честные ограничения пайплайна реальных весов** (пометки для worker'а):
- BF16→F16 конверсия сатурит значения вне диапазона F16 (|v| ≥ 65504 → inf,
  |v| < 2^-14 subnormal → 0) — квант-профили (NVFP4/W4A16, scale E4M3/F16)
  имеют те же границы; точные логиты на реальных весах — задача quality-gates
  ([R8]), не конвертера.
- Shared-expert тензоры (Qwen1.5-MoE/OLMoE: `mlp.shared_expert.*`,
  `shared_expert_gate`) планом не группируются: движковый MoE-контур
  shared-expert не содержит; конвертер игнорирует их честно (в описании их
  нет — интерпретатор такие группы не требует). Если worker'у нужен
  shared-expert в полном контуре — отдельная задача через ADR (новая op).
- `vocab_size` из config'а валидируется по форме тензора embed; `head`
  (lm_head) опционален — при `tie_word_embeddings` логиты через embedding
  (честная пометка, конвенция fixtures).

## НЕ сделано / почему

- **Прогон [R3]/[R8] на реальных весах** — worker после merge (сервер,
  сеть/диск; по правилам раунда скачивание весов — worker). Механизм
  проверен E2E на синтетическом каталоге (см. выше).
- **engine-мост `load_plain_weights` не несёт `source_slice`** (PlainGroup
  без поля среза) — тест [R4] обошёл это открытием лоадера models по
  исходному описанию (см. «открытые вопросы»); правка engine запрещена
  границами раунда.
- **BF16→F32 / BF16-цель** — не заводился: канон ADR-09 (F16/F32-цели норм,
  F16 embed/head) покрывает поддержанные архитектуры; новый путь — через
  ADR при появлении архитектуры с F32-нормами на BF16-чекпоинте.
- **Дополнительные архитектуры** (DeepSeek/Qwen3-Next/etc.) — вне subset
  [R4]; добавление — та же схема (диспетчер + нативные имена), по мере
  требований потоков.

## Открытые вопросы (следующим раундам/потокам)

1. **`source_slice` в engine-мосте** (`PlainGroup`/`PlainSpec`): при
   переходе engine на прямые лоадеры со стековными тензорами (GGUF-mixtral,
   HF-Mixtral) срезы обязаны доехать до `compat_description` — иначе
   `load_plain_weights` не сможет открыть такие планы (сейчас — только путь
   через layout/stell-tier, который срезы уже материализовал). Точка
   правки — `src/engine/src/model_spec_bridge.cpp` (поток C/engine).
2. **Shared-expert**: нужен ли в полном контуре MoE (движковый op) — за
   planner'ом (ADR-09: новая op = новое ядро).
3. **OlmoeForCausalLM**: фактические имена тензоров (нормы OLMo2-стиля)
   не проверялись на реальном чекпоинте — конвертер честно откажется
   (TensorNotFound с именем); при подтверждении имён — правка паттернов.
4. **HF BF16→F16 сатурация**: допустима ли для quality-gates [R8] (точки
   регрессии) или требуется W4A16/NVFP4-профиль со scale-F16 — тот же
   диапазон; вопрос замеряется на [R8].

## Коммиты

| Хеш | Суть |
|---|---|
| `cb231d9` | [R6]/R3-подготовка: каталог шардов — единый IWeightsSource + BF16 + общий plain_fill_slot |
| `d185efa` | [R4]: hf_to_description + CLI stell_hf_description + тесты roundtrip/error-кейсов |
| `67c9b15` | [R6]: mixed-шардовый тест (один источник, без профильных ветвлений) |
| (следующий) | R3-подготовка: профиль co-tenancy в real-weights ветке + отчёт |
