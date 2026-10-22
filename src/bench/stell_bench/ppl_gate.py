"""PPL-delta gate спекуляции (ADR-13 [R6]; round 10, planner T-20261022183913 R2 [R4]).

Контракт bench-отчёта (PLAN.md §5): отдельное поле PPL-дельты цели между
plain- и спекулятивным путём на одном семплинг-конфиге. Спекуляция не должна
систематически сдвигать распределение цели. PPL — по каноническому
определению: exp(mean(-log p_i)) на фиксированной последовательности токенов.

Честность:
  * для путей с побитовым равенством последовательностей (greedy-спекуляция
    без потерь качества, ADR-09 §4) дельта РОВНО 0.0;
  * если измерение недоступно (mock-движок, нет лог-вероятностей) — поле
    отчёта null (отсутствие данных ≠ ноль, прецедент dtype_orig round 6).

Поле отчёта (report.json): ``spec_ppl_delta`` — dict с ключами
``ppl_plain`` / ``ppl_spec`` / ``delta_abs`` / ``delta_rel`` /
``bitwise_equal``; имя зафиксировано тестом (tests/serve/test_ppl_gate.py).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

# Имя поля в report.json (зафиксировано тестом, ADR-13 [R6]).
SPEC_PPL_DELTA_FIELD = "spec_ppl_delta"


def ppl_from_logprobs(logprobs: Sequence[float]) -> float:
    """PPL = exp(mean(-log p)) по лог-вероятностям на фиксированных токенах.

    Пустая последовательность — ValueError (нечего измерять — не ноль).
    """
    if not logprobs:
        raise ValueError("ppl: пустая последовательность лог-вероятностей")
    neg = [-lp for lp in logprobs]
    return math.exp(sum(neg) / len(neg))


def ppl_delta(plain_logprobs: Sequence[float],
              spec_logprobs: Sequence[float]) -> Dict[str, Any]:
    """Дельта PPL цели между plain- и спекулятивным путём.

    Требует одинаковые длины (один семплинг-конфиг, одни токены — §5).
    bitwise_equal — строгое побитовое равенство списков (для greedy-путей
    спекуляция обязана давать ровно его → delta_abs == 0.0).
    """
    if len(plain_logprobs) != len(spec_logprobs):
        raise ValueError("ppl_delta: длины путей не совпадают — это не один "
                         "семплинг-конфиг (запрещено складывать контракты)")
    if not plain_logprobs:
        raise ValueError("ppl_delta: пустые пути")
    bitwise_equal = list(plain_logprobs) == list(spec_logprobs)
    ppl_plain = ppl_from_logprobs(plain_logprobs)
    ppl_spec = ppl_from_logprobs(spec_logprobs)
    delta_abs = ppl_spec - ppl_plain
    return {
        "ppl_plain": ppl_plain,
        "ppl_spec": ppl_spec,
        "delta_abs": delta_abs,
        "delta_rel": (delta_abs / ppl_plain) if ppl_plain != 0 else 0.0,
        "bitwise_equal": bitwise_equal,
    }
