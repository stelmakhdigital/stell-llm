# models — отчёт раунда 5 ([R3] per-layer план, [R4] shared-expert фикстура, [R5] BF16 clamp-сатурация)

Ветка `agent/models/6` (от master `7a6b770`, docs-6 влита). Worktree изолирован;
чужие каталоги (src/engine, src/kernels, src/tiers, …, docs) не тронуты.
Приоритет planner round 4: R3 → R4 → R5. Источник истины —
`.pi/pipeline/T-20261017200354/round-4/planner-decision.md` ([Q4]/[Q5]/[Q6]).

## Тесты до/после

| Срез | Результат |
|---|---|
| до (master `7a6b770`, CPU) | ctest **47/47** |
| после | ctest **47/47** (все старые проходят; новые проверки — внутрь `test_models_hf_description`, число бинарников тестов не менялось) |

Новые проверки в `test_models_hf_description` (секции 6–9):

- **[R3] смешанный dense/MoE каталог** (слои 0,1 — dense `mlp.*_proj`; слои
  2,3 — MoE `mlp.router` + поэертельные эксперты): per-layer описание
  (62 группы), членство групп по слоям, `ops` с ОБЕИМИ ветвями
  (`ffn` + `moe_gather_scatter`); streaming-layout **побитово** == прямому
  лоадеру; +6 честных SchemaMismatch-кейсов (ж–м, см. ниже).
- **[R4] shared-expert фикстура** (Qwen2Moe-класс, `mlp.shared_expert.*`,
  d_ff_shared=12 ≠ routed d_ff=8): группы `ffn.shared` с формами по факту
  тензоров, routed-группы не задеты, план валиден; roundtrip layout ==
  прямому лоадеру (побитно).
- **[R5] clamp-сатурация**: unit `plain_fill_slot` (общий путь лоадера и
  конвертера) — BF16/F32-источники: \|v\| > 65504 → ±65504 (0x7BFF/0xFBFF,
  не inf), 65520 (e=16, раньше inf) → +65504; \|v\| ≤ 65504 — побитово ==
  прежней конверсии (RTNE `f32_to_f16_bits` не менялся); счётчик сатураций;
  `Stats` конвертера (`f16_saturations=4`, `dtype_orig bf16=4/f32=1`).

## Файлы + суть

**[R3] per-layer план в `hf_to_description`** (`src/models/src/hf_description.cpp`,
ветвь Qwen2Moe-класса):

- Классификация **каждого** слоя по факту в шардах: слой с
  `mlp.router.weight`/`mlp.gate.weight` (и полным набором экспертов) → MoE-контур;
  слой с dense-ffn тензорами (`mlp.*_proj`, llama-нейминг) и без роутера →
  dense ffn группы. Суффиксы имён (роутер/эксперты/dense-ffn) — по факту,
  на каждый слой независимо.
- **Однородный чекпоинт — план как раньше (дефолт round 4 побитово):**
  шаблонные группы с `cardinality {layer: L, expert: E}`; dense-архитектуры
  и Mixtral (стековные) не менялись.
- **Смешанное членство — literal-группы** (без cardinality, схема 1.0
  не меняется — ADR-09 round 5: «per-layer — обобщение, не новая схема»):
  шаблон с `cardinality layer==n_layers` размечает все слои сразу и для
  смешанного членства неприменим; dense-слои получают `l%03d.ffn.{gate,up,down}`
  (pad3-конвенция имён, совместима с engine `layer_map`/`ops_runner`),
  MoE-слои — `l%03d.mlp.router` + `l%03d.e%03d.{gate,up,down}`;
  attn/нормы/kv остаются шаблонными на все слои (они есть в каждом слое).
  `ops` смешанного плана — обе ветви (`rmsnorm, attention, ffn,
  moe_gather_scatter`): маршрутизацию per-layer делает интерпретатор плана
  (ADR-09 round 5: слой с `mlp.router` → MoE, без → dense ffn).
- **Честные отказы (SchemaMismatch, fail-fast §3 не ослаблен):** роутер без
  экспертов; эксперты без роутера; роутер И dense-ffn в одном слое;
  слой без ffn-контура; MoE-архитектура без MoE-слоёв; dense-слои без
  `intermediate_size`. Разные d_ff у MoE-слоёв — SchemaMismatch
  (`moe.expert_d_ff` — одно поле схемы). Dense-семейства с MoE-тензорами —
  непокрытая конфигурация (честный отказ существующей валидацией плана).
- Тест-кейсы (ж)–(м) закрывают каждый отказ.

**[R4] ffn.shared** (`hf_description.cpp` + `convert.hpp`):

- Shared-expert тензоры (`mlp.shared_expert.{gate,up,down}_proj.weight` |
  `.*.weight` — по факту) группируются в **dense-FFN группы
  `l%03d.ffn.shared.{gate,up,down}`** (role=attn, dtype профиля; формы — по
  факту тензоров слоя, d_ff_shared может отличаться от routed). Частичный
  набор — SchemaMismatch. Отсутствие тензоров — групп нет (обратная
  совместимость планов: поведение без изменений). Новая операция ядра не
  вводится (ADR-09 round 5); существующие homogeneous-планы без shared —
  побитово прежние.

**[R5] BF16→F16 clamp-сатурация** (ADR-03 round 5):

- `src/models/src/quant.hpp` — `saturate_f16()` (clamp к ±65504 = максимум
  F16), глобальный счётчик `f16_saturation_count()/f16_saturation_reset()`
  (CPU-пути однопоточные); `kF16Max`.
- `src/models/src/file_weights_source.hpp` — `plain_fill_slot` (общий путь
  лоадеров и конвертера) применяет сатурацию в ветвях F32→F16 и BF16→F16;
  NaN/обычные значения — без изменений (биты == прежней конверсии).
- `src/models/src/converter.cpp` + `convert.hpp` — `StellTierConverter::Stats`
  (`f16_saturations`, провенанс `groups_dtype_orig_bf16/f32` —
  `dtype_orig: bf16` по ADR-03 round 5) как опциональный out-параметр
  `convert_streaming` (обратно совместимо).
- `src/models/tools/convert_streaming.cpp` — счётчик сатураций и провенанс
  dtype_orig в логе конвертера: «сатураций F32/BF16→F16: N (clamp к ±65504);
  dtype_orig: bf16=M, f32=K групп».
- `f32_to_f16_bits` (RTNE) НЕ менялся — обычные значения побитово прежние;
  квантованные профили (NVFP4/W4A16) не затронуты (R5 — про F16-цель).

## Тесты (детали)

`tests/models/test_hf_description.cpp`:

- хелпер `roundtrip_all(..., pending_ok)` — слоты layout == прямому лоадеру
  (побитно) всегда; логиты интерпретатора по layout == референсу того же
  плана (побитно). `roundtrip_logits` (round 4) — обёртка без pending,
  все 4 старых roundtrip-секции проходят как раньше.
- фикстуры: `mixed_hf_tensors()` (4 слоя: 2 dense + 2 MoE),
  `moe_shared_hf_tensors()` (homogeneous MoE + shared), `write_config_layers`
  (число слоёв / без intermediate_size).

## Ключевые решения

1. **Literal-группы для смешанного членства.** Схема 1.0 запрещает
   cardinality layer ≠ n_layers (`validate_template`, `description.cpp`) и
   неизвестные поля — компактного синтаксиса поддиапазонов слоёв в схеме нет;
   per-layer план выражается literal-группами (pad3-имена совпадают с
   expansion шаблонов, engine-мост/`layer_map` их исполняет). Однородные
   планы остаются шаблонными — дефолт побитово не меняется.
2. **Честный pending для логитов per-layer/shared.** Интерпретатор плана
   (`ops_runner`, src/engine) — зона потока core-6 ([R3] маршрутизация,
   [R4] суммирование ffn.shared): в этой ветке он ещё не маршрутизирует
   per-layer. Тест печатает `[PENDING]` (plan/layout проверены полностью;
   логиты-сравнение оживёт после смержа core-6 — механизм сам в него
   «перетечёт» без правок). Для ffn.shared дополнительно зонд: логиты плана
   со shared vs план без shared (те же routed-веса) — фиксирует, суммирует
   ли уже интерпретатор.
3. **R5 — сатурация на обеих конверсиях (F32→F16 и BF16→F16)** в общем
   `plain_fill_slot`: «по прецеденту F32→F16» (ADR-03) — асимметрия
   «BF16 clamp, F32 inf» оставила бы inf-заражение через F32-нормы; обычные
   значения (|v| ≤ 65504) побитово прежние (требование теста [R5]).
4. **Провенанс dtype_orig — вне схемы JSON** (строгий парс: неизвестные
   поля — ошибка, новая версия schema — через ADR): считается/печатается
   конвертером (`Stats` + лог CLI) — формулировка ADR-03 round 5
   («конвертер/описание фиксируют») допускает сторону конвертера.

## НЕ сделано / почему

- **Серверный прогон на реальном SmolLM2-360M-MoE (смешанный, 3 плотных из
  31)** — worker после merge: сервер недоступен из ветки (read-only веса в
  `~/models`, env-механизм `STELL_R7_REAL_*` указывает на них); локально
  прикрыто синтетическим смешанным каталогом (те же факты в шардах:
  часть слоёв dense, часть MoE).
- **Логиты == референсу per-layer плана (побитно)** и **суммирование
  ffn.shared в интерпретаторе** — ops_runner (src/engine) — поток core-6
  [R3]/[R4]; в этой ветке честный pending (см. выше). После merge обе
  проверки включаются автоматически.
- Схема 1.0 не менялась (без cardinality-поддиапазонов/новых полей) —
  по ADR-09 round 5 («не новая схема, не новая версия schema»).

## Открытые вопросы

1. Порядок суммирования `ffn.shared` с routed-экспертами в ops_runner
   (hidden += shared, затем += routed — или наоборот): ADR-09 round 5
   порядок не фиксирует; для побитового референса после merge core-6
   порядок стоит зафиксировать в ADR (текущее допущение теста-зонда —
   сравнение «со shared vs без», порядок не важен).
2. `tiny-qwen2moe`/`smollm2-moe` на сервере: содержат ли shared-expert
   тензоры (`mlp.shared_expert.*`) — после merge перегенерировать описания
   ([R7]/[R8] прогон) и проверить, что план включает ffn.shared.
3. Dense-семейство с MoE-слоями (архитектура dense-диспетчера, смешанные
   слои) — оставлено непокрытым (честный SchemaMismatch); реальные
   смешанные чекпоинты объявляют MoE-архитектуру — при появлении такого
   чекпоинта расширить классификацию на dense-ветвь.

## Коммиты (ветка `agent/models/6`)

- `57d210d` models(R5/R3/R4): clamp-сатурация BF16/F32→F16 (±65504, счётчик,
  провенанс dtype_orig) + per-layer план и ffn.shared в hf_to_description
  (ADR-09 round 5)
- `c01b99a` models(tests): [R3] per-layer смешанный dense/MoE каталог (+6
  честных SchemaMismatch-кейсов), [R4] ffn.shared-фикстура, [R5] clamp-
  сатурация ±65504 (unit + счётчик + Stats конвертера); roundtrip-хелпер
  с честным pending на per-layer интерпретатор
- `<см. git log>` — отчёт раунда (этот файл)
