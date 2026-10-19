# quality-gates (round 2, PLAN.md §5)

Состав:

- `schema.json` — схема результатов гейтов + описание датасетов;
- `data/gsm8k_32.jsonl` — **РЕАЛЬНЫЙ** поднабор GSM8K (openai/grade-school-math,
  test.jsonl, 1319 задач): фиксированная выборка 32 задач seed=42,
  детерминирована скриптом `stell_bench.fetch_quality_gates`;
- `data/humaneval_32.jsonl` — **РЕАЛЬНЫЙ** поднабор HumanEval (openai/human-eval,
  HumanEval.jsonl.gz, 164 задач): выборка 32 задач seed=42, детерминирована тем
  же скриптом;
- `data/ppl_text.txt` — фиксированный текст для PPL (менять запрещено между
  версиями: сравнение версий по §5); round 1 — не меняется.

Загрузка датасетов (только stdlib: urllib/gzip):

```bash
PYTHONPATH=src/bench ~/.local/stell-conda/envs/serve/bin/python \
  -m stell_bench.fetch_quality_gates   # seed=42, 32 задачи из каждого
```

## Прогон гейтов (round 2, поток E — R1/R2-интеграция)

Два режима runner'а (`stell_bench.quality_gates_runner`):

**Pending-снимок (по умолчанию)** — датасеты реальны, прогона нет:
`pending=True` без выдуманных значений. Причина: `stell_logits` ещё не
вклеен в собранную .so (pre-merge core).

**Живой прогон (`--live`)** — гейты считаются по логитам модели:

- источник логитов — `stell_logits` ([R1], планировщик Q1: request
  `{"prompt_tokens":[…], "positions":"last"|"all"}`, возврат 0/-1,
  malloc/free — конвенция ADR-11 §1) через биндинг
  `python/stell_serve/engine.py`; модель загружается СРЕДСТВАМИ ДВИЖКА
  (cfg `runner:"ops"`, реальные веса `tests/models/fixtures`):
  mini_dense / mini_moe;
- PPL — по logprob позиций (`"positions":"all"`, teacher-forcing):
  `PPL = exp(mean(-log P(tok[i+1])))` по позициям предсказания
  (`ppl_from_logits`, чистый Python softmax/logprob);
- GSM8K-32 — жадное декодирование через логиты (argmax, ничья → меньший
  индекс — конвенция golden); корректность — последнее число в
  декодированном тексте против ответа задачи (БЕЗ исполнения — вычислимо
  честно, реализовано);
- HumanEval-32 — честный pending: проверка корректности требует
  ИСПОЛНЕНИЯ сгенерированного кода (unit-тесты HumanEval), а исполнения
  кода в runner нет (песочница/изоляция не входят в раунд).

Пока `stell_logits` не вклеен в .so, `--logits-source auto` (по умолчанию)
честно падает на fallback: НЕЗАВИСИМЫЙ Python-референс логитов
(`tests/models/golden/generate_golden.py` — тот, что сгенерировал
golden-фикстуры; сверен с HF Llama/Mixtral ~1.5e-7); числа помечаются
provenance «golden-ref (fallback)». После merge core автоматически
используется ABI.

## БАЗОВЫЕ РЕФЕРЕНСЫ (первый живой прогон, 2026-10-19)

`bench/results/20261019-quality-gates-live/report.json`
(источник логитов — golden-ref fallback; параметры: PPL — первые 128
токенов фиксированного текста; GSM8K — вопрос ≤ 64 токенов, ≤ 16 шагов
жадного декода):

| гейт | mini_dense | mini_moe |
|---|---|---|
| ppl_fixed_text (127 позиций предсказания) | **18.3928** (mean_nll 2.911957) | **30.0858** (mean_nll 3.404053) |
| gsm8k_32 (accuracy) | **0.0625** (2/32) | **0.0** (0/32) |
| humaneval_32 | pending (нужно исполнение кода) | pending (нужно исполнение кода) |

Честная интерпретация: mini-фикстуры (vocab=16, byte-токенизатор движка)
не являются языковыми моделями — текст отображается в токены
детерминированной проекцией байта → токен (b % vocab), обратная карта
id → текст (цифры < 10 → цифра, иначе разделитель). Полученные числа —
МЕХАНИЧЕСКИЕ базовые референсы пайплайна (точки регрессии), а не оценка
качества LLM.

## Запуск

```bash
PYTHONPATH=src/bench ~/.local/stell-conda/envs/serve/bin/python \
  -m stell_bench.quality_gates_runner                # pending-снимок → bench/results/local/ (gitignored)
PYTHONPATH=src/bench ~/.local/stell-conda/envs/serve/bin/python \
  -m stell_bench.quality_gates_runner --live          # живой прогон → bench/results/local/
PYTHONPATH=src/bench ~/.local/stell-conda/envs/serve/bin/python \
  -m stell_bench.quality_gates_runner --live --committed  # живой прогон → bench/results/<date>-…/ (коммитится)
```
