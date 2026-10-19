# serve — раунд 4, [R8] (Поток E): quality-gates на реальных весах — подготовка

> **Fixup** (серверный прогон worker'а на tiny-qwen2moe выявил зависание
> PPL-пути) — см. раздел «Fixup: PPL на больших словарях» ниже; коммит-фиксап
> в той же ветке.

Ветка: `agent/serve/5` (от master `c51da0e`). Зона: python/stell_serve (без
изменений — не требовался), src/bench/stell_bench, tests/serve, bench/results.
Чужие каталоги (src/engine, src/kernels, src/tiers, src/models, docs) — не
тронуты; корневые CMakeLists.txt/.gitignore — не редактировались.

## Суть [R8]

Режим «реальная модель» в quality-gates runner'е (`--real`): каталог модели
с описанием ADR-09 (вывод `hf_to_description` [R4], поток models) + весами
+ опциональным tokenizer.json прогоняется живьём через ABI (`stell_logits`):
PPL + GSM8K-32 + HumanEval-32; точки регрессии — в bench/results/ с
provenance (модель, размер весов, дата, co-tenancy).

## Файлы

- `src/bench/stell_bench/quality_gates_runner.py` — режим `--real`:
  - `resolve_real_model()`: приоритет явных файлов (--model-desc +
    --model-weights, пара обязательна целиком) → каталог (--real-model-dir /
    env `STELL_R8_REAL_MODEL_DIR` — по прецеденту R7, без правки кода).
    Контракт каталога: `description.json` (ADR-09) + `model.safetensors`
    (ровно один файл) + опционально `tokenizer.json`. Отсутствие источника →
    None (вызывающий — честный pending); явно битый источник → громкая
    ошибка (FileNotFoundError с подсказкой: HF-шарды сначала склеиваются —
    plain-лоадер safetensors открывает один файл, догадки запрещены).
  - `run_gates_real()`: загрузка модели СРЕДСТВАМИ движка (cfg
    runner:"ops", weights_format safetensors), источник логитов — ТОЛЬКО ABI:
    `stell_logits`. golden-ref-референс умеет интерпретировать только
    mini-фикстуры — молчаливая подмена источника запрещена (RuntimeError).
  - `run_gates_real_pending()`: честный pending «веса не предоставлены»
    (значения не измерены, причина в notes + provenance).
  - `real_provenance()`: модель, класс, пути, weights_bytes, vocab/layers/
    experts, tokenizer_json, tokenization, date_utc, co_tenancy, engine_lib,
    logits_source.
  - Гейты (PPL/GSM8K/HumanEval) обобщены: kwarg `vocab`/`tokenizer`/
    `notes_extra` при полном сохранении прежнего поведения mini-фикстур
    (дефолты не изменены).
- `src/bench/stell_bench/bpe_tokenizer.py` — НОВЫЙ: Python-порт контракта
  ByteLevelBpeTokenizer (src/models, R10 round 2): subset «чистый byte-level
  BPE» (use_regex=false, без normalizer/post-processor/added/unk/byte_fallback/
  dropout — TokenizerError, тишина запрещена); жадное слияние по минимальному
  рангу, все вхождения слева направо; byte↔unicode таблица GPT-2-конвенции.
  **Сверен с эталоном** fixtures/tokenizer/tokenizer_ref.json (ид
  сгенерированы РЕАЛЬНОЙ реализацией HF tokenizers 0.23.3, offline) —
  расхождение 0 на всех 25 строках; roundtrip OK. Это РЕАЛЬНАЯ токенизация
  для каталогов с tokenizer.json (без сети).
- `src/bench/stell_bench/synthetic_model.py` — НОВЫЙ: сборка синтетического
  каталога «реалистичного профиля» (MoE 4 слоя × 8 экспертов против mini
  2 × 4; vocab=350; NVFP4-профиль линейных групп; HF-style source_pattern;
  эталонный tokenizer.json HF копируется из fixtures — без сети). Веса —
  независимым генератором golden-фикстур (tests/models/golden/
  generate_golden.py: make_fixture_weights + write_safetensors, LCG seed
  0x5E8E) — детерминированно, воспроизводимо. **Честная пометка**: это НЕ
  языковая модель — значения гейтов = точки регрессии конвейера, не оценка
  качества LLM (пометка автоматом в notes гейтов и run.log).
- `tests/serve/test_bpe_tokenizer.py` — НОВЫЙ (7): эталонные ид HF = 0
  расхождений, roundtrip, vocab/merges, subset-отказы (6 шт.), пустой текст,
  неизвестный id, отсутствие файла.
- `tests/serve/test_quality_gates_real.py` — НОВЫЙ (20): протокол-константы
  (имена файлов каталога, env, лимиты протокола); resolve через каталог /
  env / аргументы / приоритеты; честные ошибки (нет каталога/описания/весов,
  HF-шарды без склейки); пара --model-desc/--model-weights; честный pending
  (API + CLI `--real` без источника → exit 0, отчёт real-model-pending);
  громкая ошибка CLI на битом источнике (exit 1); живой прогон через ABI на
  синтетическом каталоге (gates не pending, provenance полный: модель/
  размер/дата/co-tenancy=false, tokenization «byte-level BPE»); fallback
  без tokenizer.json — честная проекция байта (механический референс);
  отказ без stell_logits (golden-ref подмена запрещена); CLI e2e live-real.
- `bench/results/20261019-quality-gates-real/` — НОВАЯ точка регрессии:
  live-real прогон на синтетическом реалистичном профиле через ABI
  (PPL 606.6646 / GSM8K 0.0 / HumanEval 0.0 — механические точки регрессии;
  provenance: weights 130704 B, date_utc, co_tenancy=false, tokenizer.json).
- `bench/results/20261019-quality-gates-real-pending/` — честный pending
  «веса не предоставлены» (образец R7): причина — прогон worker'ом после
  скачивания весов [R3] и hf_to_description [R4].

## Протокол (spec)

- Тесты до/после: ctest **45/45** (без изменений), pytest: до — **143 passed,
  2 skipped**, после — **170 passed, 2 skipped** (+27 новых, старые не сломаны).
- Веса/датасеты в git не попадают: синтетические веса генерируются в
  tmp-каталоге, в bench/results/ попадают только report.json + run.log.
- Серверный прогон на настоящих весах — честный pending (см. выше): веса
  ещё не скачаны ([R3] — worker), `hf_to_description` [R4] потоком models
  не слит в master; конвейер полностью отработан на синтетическом каталоге
  реалистичного профиля, env-переключатель `STELL_R8_REAL_MODEL_DIR`
  готов к прогону worker'ом без правки кода.

## Ключевые решения

1. **Один файл весов в контракте каталога** (`model.safetensors`): plain-лоадер
   safetensors (src/models) открывает один файл; HF-шардовый каталог — честная
   ошибка с подсказкой (склейка/конвертация), не догадки. stell-tier — путь
   [R7]-конвейера, в ABI cfg подставляется weights_format="safetensors".
2. **Только ABI-источник в реальном режиме**: golden-ref не умеет ничего,
   кроме mini-фикстур; fallback = скрытая подмена источника — запрещено.
3. **Реальный tokenizer.json** вместо проекции байта, где он есть: Python-порт
   R10-контракта, сверён с эталоном HF tokenizers (0 расхождений). Без
   tokenizer.json — прежняя честная проекция с пометкой «механический
   референс». Для настоящих весов с BPE-токенизатором значения гейтов —
   уже осмысленные измерения конвейера (жадное декодирование, песочница).
4. **Синтетический каталог — не «фейк реальной модели»**: честные пометки
   «синтетический реалистичный профиль, не языковая модель» в notes гейтов,
   run.log и provenance-схеме; воспроизводимость — LCG-генератор golden.
5. **pending vs громкая ошибка**: отсутствие источника (env/аргументы не
   заданы) → честный pending; ЯВНО заданный битый источник → exit 1
   (нельзя маскировать ошибку конфигурации pending'ом).

## Не сделано / почему

- Прогон на настоящих весах — ждал [R3] (скачивание) и [R4]
  (hf_to_description): обе задачи других потоков в master `c51da0e` не
  слиты; конвейер готов, прогон — worker'ом (см. pending-отчёт).
- CFG-файл как альтернативная форма параметров (--model-desc/--model-weights
  или каталог) — не заводился: каталог + явная пара файлов покрывают сценарий
  R8; третья форма — дублирование контракта (прецедент [Q7] о mixed-профиле).
- docs/design/serve.md (раздел round 4) — не правил: docs вне зоны раунда;
  материал для вклейки — этот отчёт.

## Открытые вопросы

- Если `hf_to_description` [R4] выдаст описание с иным именем файла
  (не description.json) — точка стыковки: REAL_DESCRIPTION_NAME; константа
  меняется в одном месте.
- Для настоящих весов с tokenizer.json, где pretokenizer — ByteLevel с
  use_regex=true (часть Qwen-класса): subset R10 честно отклонит такой
  tokenizer.json (TokenizerError); расширение subset — согласовать с потоком
  models (тот же контракт).

## Fixup: PPL на больших словарях (серверный прогон tiny-qwen2moe)

**Факт worker'а**: tiny-qwen2moe (Qwen2Moe, hidden=32, vocab=151646, 2 слоя ×
8 экспертов; description.json от hf_to_description, веса nvfp4w4a16) — движок
создаётся за 0.58с, positions:"last" работает (n_vocab=151646); но PPL-путь
зовёт positions:"all" → 128 × 151646 ≈ 19.4 млн логитов ≈ 300+ МиБ JSON за
ОДИН вызов → прогон зависал (rc=124, вывод пуст).

**Решение — режим 'incremental-last'** (константа-порог протокола
`PPL_ALL_MAX_ELEMENTS = 2_000_000` элементов логитов за вызов;
n_vocab × n_positions > порога → инкрементальный путь):
- 'all' — teacher-forcing одним вызовом (малые словари; путь и значения
  прежние — mini-фикстуры и синтет. профиль дают побитно те же числа);
- 'incremental-last' — teacher-forcing ПОЗИЦИЯМ: повторные positions:"last"
  по растущему префиксу, по ОДНОЙ строке логитов за вызов (~n_vocab
  элементов ≈ 1–2 МиБ ответ — не зависнет);
- **почему не чанки "all"**: каузальная семантика требует для строки i
  полный префикс tokens[:i+1], а ABI не имеет параметра смещения контекста —
  усечённое окно давало бы логиты без левого контекста (не PPL). Единственная
  честная альтернатива — инкрементальный last;
- **эквивалентность математик**: rows[i] = logits_last(tokens[:i+1]) —
  ровно семантика строк "all" (документирована в runner'е и сверена с
  golden-ref); тест на детерминированном стабе: значения путей побитно
  равны.

Честные пометки: notes гейта PPL фиксируют режим/vocab/позиции/порог;
provenance точки регрессии — поле `ppl_mode` ('all'/'incremental-last')
+ протокол-константы в report.json (`ppl_all_max_elements`,
`ppl_mode_rule`).

Заодно (стыковка с worker'ом): weights_format в ABI-cfg теперь выводится из
ADR-09 описания — квантованный профиль (NVFP4/W4A16) → лоадер `nvfp4w4a16`
(квантование на read, ADR-03; плоские группы F16/F32 он же обслуживает —
фикс round 2), иначе plain `safetensors`. Значения синтетической точки
регрессии не изменились (606.6646 / 0.0 / 0.0).

Тесты fixup (+5): порог (граница включительно), эквивалентность путей
(побитово), большой vocab=120000 на стабе с полным протоколом (127 вызовов
по строке, PPL = vocab, "all" не вызывался), provenance ppl_mode, живой
ABI-прогон big-vocab каталога (позиции ограничены для теста: 24 — выше
порога; константа протокола не меняется). Полный прогон: pytest 175+2skip,
ctest 45/45.

**Честная пометка по вычислительной стоимости**: ABI-вызов stell_logits
пересчитывает весь префикс на каждый вызов (и positions:"last" материализует
логиты всех позиций внутри движка — src/engine/ops_runner.cpp:
run_prompt_logits = run_prompt_logits_all(...).back()). Инкрементальный PPL —
это O(N²/2) токен-форвардов; полный R8-прогон на реальной модели — порядка
десятков минут на CPU-контуре сервера (см. команду worker'а — с generous
таймаутом). Ускорение (не материализовать projection над словарём для
промежуточных позиций при positions:"last") — точка расширения потока core;
контракт runner'а от этого не меняется.
