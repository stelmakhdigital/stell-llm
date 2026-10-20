# [R3] round 8 — конвейер на реальных весах Qwen3-30B-A3B (поток B, tiers)

**НЕ ВЕРДИКТ ПО ЦЕЛИ §4** — co-tenancy, CPU-контур; точка регрессии (прецедент round 5/6).

- описание: description.json (NVFP4, dtype_orig=bf16, qk_eps, GQA dqq=4096 != d_model=2048)
- layout: 17153396736 байт, 18867 групп (конвертация 653.5 с, peak RSS 1.2 ГиБ, 0 сатураций)
- per-group bench матрица {O_DIRECT, buffered} x {k=1, k=4} + cold-контраст — report.json (matrix_pergroup)
- побитовая сверка слотов при eviction-чередовании: 43 контрольных групп (честная пометка: подмножество, полный проход слишком долог) — slot_verify.log, 172 сверок, 0 несовпадений
- фиксы, потребованные реальным чекпоинтом: ffa9254 (k_norm [head_dim] GQA), f032199 (o_proj {dm, dqq}), ce38b48 (ленивые CPU VRAM-слоты)
- failed_9slots_*.log — честный отказ профиля 9x594МиБ (TierError SlotExhausted: pin всего слоя)
- co-tenancy: nvidia-smi-at-measurement.txt + строки co-tenancy в каждом логе; strata НЕ тронут; bench — CPU/NVMe
- сравнение ячеек только внутри категории I/O
