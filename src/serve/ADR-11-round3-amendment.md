# Материал для ADR-11 §4 — «Уточнение (round 3)»: строгий формат logits-снапшота

**Для потока docs** (каталог docs/ вне зоны раунда serve): готовый текст
вклейки в `docs/adr/ADR-11-engine-serve.md` после раздела «Уточнение
(round 2 …)». Источник решения: планировщик [Q5] (round 2 → требования
round 3 [R5]); реализация — `tests/serve/test_logits_metrics.py`,
`python/stell_serve/metrics.py` (LOGITS_SNAPSHOT_FIELDS),
`src/ui/app.js` (renderLogits — строгие 4 KV).

---

## Уточнение (round 3, задача T-20261017200354)

Контракт не ломается; фиксация по открытому вопросу round 2 (метрика
`snap.logits` — формат не был зафиксирован; планировщик [Q5]).

- **Строгий формат `logits`-снапшота в /metrics ([R5]).** Поле `logits`
  снапшота `stell_metrics` — **объект статистики вызовов `stell_logits`
  без payload**, ровно четыре поля:

  ```json
  {"requests_total": N, "errors_total": N, "last_n_vocab": V, "last_positions": P}
  ```

  - `requests_total` — всего вызовов `stell_logits`;
  - `errors_total` — из них ошибочных (`out_json` с `{"error":…}`, rc = -1);
  - `last_n_vocab` / `last_positions` — `n_vocab` / `n_positions`
    последнего вызова.

  Payload логитов в /metrics **не передаётся никогда** (размер: массивы
  словаря логитов не проходят через границу метрик — для payload есть
  `stell_logits`). Формат проверяем: служебная валидация
  `is_valid_logits_snapshot` (python/stell_serve/metrics.py) — ровно 4
  поля, значения — числа (массивы/объекты — нарушение контракта); тест
  serve против реального движка (tests/serve/test_logits_metrics.py)
  включает проверку /metrics автоматически после merge core-4 (до того —
  честный SKIP по наличию поля).

- **Дашборд рендерит строго эти 4 KV.** Generic-рендер «каково придёт»
  заменён фиксированной схемой (LOGITS_FIELDS в app.js): ключи движка вне
  четырёх фиксированных игнорируются; отсутствующее поле — «—»;
  отсутствующий блок — честная пометка «нет данных» (UI не падает).
  Обоснование: поле теперь контракт ADR, а не «скрытый договор» между
  serve и ui; счётчики достаточны дашборду и не зависят от vocab-размера
  модели.
