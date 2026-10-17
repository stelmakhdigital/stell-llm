# quality-gates (round 2, PLAN.md §5)

Состав:

- `schema.json` — схема результатов гейтов + описание датасетов;
- `data/gsm8k_32.jsonl` — **РЕАЛЬНЫЙ** поднабор GSM8K (openai/grade-school-math,
  test.jsonl, 1319 задач): фиксированная выборка 32 задач seed=42,
  детерминирована скриптом `stell_bench.fetch_quality_gates`;
- `data/humaneval_32.jsonl` — **РЕАЛЬНЫЙ** поднабор HumanEval
  (openai/human-eval, HumanEval.jsonl.gz, 164 задач): выборка 32 задач
  seed=42, детерминирована тем же скриптом;
- `data/ppl_text.txt` — фиксированный текст для PPL (менять запрещено между
  версиями: сравнение версий по §5); round 1 — не меняется.

Загрузка датасетов (только stdlib: urllib/gzip):

```bash
PYTHONPATH=src/bench ~/.local/stell-conda/envs/serve/bin/python \
  -m stell_bench.fetch_quality_gates   # seed=42, 32 задачи из каждого
```

Прогон гейтов (round 2): датасеты реальны, но прогона нет — честный
`pending=True`:

- нет модели/весов (реестр и лоадеры — поток D, runner поверх реальных
  весов — поток A);
- ABI ADR-11 round 1 не отдаёт логитов (нужны для PPL) и полных генераций
  для проверки HumanEval/GSM8K — расхождение контракта, зафиксировано в
  отчёте round 2; гейты вычислимы после расширения ABI (новый символ —
  ABI-совместимо) и появления весов/модели.

```bash
PYTHONPATH=src/bench ~/.local/stell-conda/envs/serve/bin/python \
  -m stell_bench.quality_gates_runner   # → bench/results/local/ (gitignored)
```
