# kernels — дизайн-документ (заготовка)

> Статус: заготовка. Модуль не реализован. Контракты: ADR-01, ADR-04, ADR-09.

## Объём (из PLAN.md §4, поток A)

- SM70-attention ядра: FP32 QK/PV, layout-реплей, цель ≥70 TFLOP/s операторно.
- MoE gather/scatter; dequant NVFP4/W4A16; KV-квант E4M3 (ADR-04); драфтер-ядра.

## Готовность

- Все ядра проходят операторные replay-тесты (CPU и CUDA SM70).
