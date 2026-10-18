# Golden-референсы логитов — раунд 2 (ADR-09 §4, ADR-12, режим greedy)

**Статус: референсы готовы (2 генератора на fixture, оба сходятся), прогон
engine (core-2) — в процессе (см. отчёт раунда).**

## Что в этом каталоге

- `generate_golden.py` — НЕЗАВИСИМЫЙ генератор референсных логитов: Python-
  интерпретатор ops-графа описания модели (numpy float32, без torch/transformers).
  Он же генерирует САМИ синтетические fixture-веса
  (`tests/models/fixtures/<fixture>.safetensors`, детерминированный LCG,
  seed 0x00D1/0x00D2 — конвенция раунда, README fixtures).
- `hf_crosscheck.py` — второй референс «best effort»: веса fixture переносятся
  в HF-модель (LlamaForCausalLM / MixtralForCausalLM), forward greedy, логиты
  сравниваются с независимым референсом.
- `logits_<fixture>.json` — референс №1 (независимый интерпретатор), схема
  `logits_ref.schema.json` (schema "1.0": model/profile/path/prompt_tokens/
  seed/greedy/tolerance_rel/logits_ref [+ logits_tested — прогоны engine]).
- `logits_<fixture>_hf.json` — референс №2 (HF-кроссчек), та же схема;
  `meta` содержит implementation/config/patches и max_rel_err_vs_python_ref.
- `logits_ref.schema.json` — описание схемы (round 1, без изменений).
- `golden_logits.cpp` — тест сравнения: валидирует схему фикстур, сравнивает
  `logits_ref` с `logits_tested` (погоны engine, core-2 дописывают поле);
  без `logits_tested` — честный SKIP.

## Как генерировались референсы (проверяемая цепочка)

1. **Fixture-веса**: `generate_golden.py` создаёт safetensors по описанию
   `tests/models/fixtures/<fixture>.json` — веса из детерминированного LCG
   (см. README fixtures); файл committed в репозитории (тесты не обязаны
   зависеть от Python).
2. **Независимый референс (№1)**: тот же `generate_golden.py` читает
   safetensors и исполняет ops-граф описания ПОСЛЕМЕНТНО в float32:
   embedding → rmsnorm (mean/var по последней оси) → attention (causal,
   softmax FP32, GQA: k/v-головы = n_kv_heads, БЕЗ позиционной операции —
   fixture ops не содержат rope) → MLP SwiGLU (`ffn.gate/up/down`) /
   MoE (роутер = матрица [n_experts, d_model], softmax → top-k (ничья →
   меньший индекс) → renorm → взвешенная сумма экспертов) → final norm →
   head (у mini_dense; у mini_moe head отсутствует — конвенция tied
   embeddings, т.е. логиты = y @ embed^T). Режим GREEDY (argmax, ничья →
   меньший индекс), T=1 не используется (см. ADR-12 кларификацию).
3. **HF-кроссчек (№2)**: `hf_crosscheck.py` строит LlamaConfig/MixtralConfig
   из fixture-описания (точные размеры), переносит веса LCG, патчит
   RoPE→identity и лишние нормы→Identity (см. `meta.patches` в файлах),
   делает forward greedy. Для NVFP4-профиля (mini_moe) веса подменяются
   на dequant(quantize(F16→f32)) по РЕФЕРЕНСУ `src/models quant.hpp`
   (блок 32, E2M1+E4M3) — Python-порт quant.cpp сверен с C++ побитно
   (см. test_golden_fixtures.cpp: hex-эталоны + quantize(file) == payload).
4. **Сходимость генераторов**: max rel-ошибка |№1 − №2| на всех логитах
   ≈ 2.4e-07 (mini_dense) и ≈ 1.5e-07 (mini_moe) — ЗАФИКСИРОВАНА в
   `meta.max_rel_err_vs_python_ref` обоих `_hf.json`. Референс №1 и №2
   согласованы ⇒ golden-логиты не зависят от одного генератора.

## Пороги (tolerance_rel)

- mini_dense (F16-профиль, вычисление FP32): **1e-3** (референс №1);
  №2 (HF) — 1e-4 (тот же порядок, но сравнение с №1).
- mini_moe (NVFP4-профиль: квантование входит в эталон): **1e-3**
  (когда-то в README раунда 1 стояло «~1e-2, уточняется» — калибровано
  против фактического rel-ошибки деqуанта: 1e-3 достаточно и надёжно;
  зафиксировано в обоих файлах).

## Семантика ops (контракт между golden и engine — см. README fixtures)

- rmsnorm: mean/var по ПОСЛЕДНЕЙ оси; eps из описания;
- attention: causal, softmax FP32, GQA (kv-головы повторяются), rope
  ОТСУТСТВУЕТ (fixture ops не содержат позиционной операции);
- SwiGLU: silu(gate) ⊙ up → down;
- MoE: softmax → top-k → renorm (деление на сумму выбранных), ничья при
  равных вероятностях → МЕНЬШИЙ индекс;
- NVFP4-веса = dequant(quantize(F16→f32)) по референсу quant.hpp;
- greedy argmax: ничья → меньший индекс.

## Как включить сравнение (когда core-2 допишет logits_tested)

1. Прогон engine по fixture (см. отчёт раунда; `loader` путь, профиль из
   fixture-описания, greedy, seed 0, prompt_tokens из golden-файла);
2. дописать `logits_tested` в `logits_<fixture>.json` (или отдельный файл
   по схеме);
3. `ctest --test-dir build/cpu -R golden_logits` — сравнение начнётся
   автоматически (без поля — SKIP, не молчаливый успех).
