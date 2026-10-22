"""[R4] round 10 (ADR-13 [R6]): PPL-delta gate в контракте bench-отчёта.

Судья проверяет: поле схемы отчёта с зафиксированным именем
(``SPEC_PPL_DELTA_FIELD``); известная дельта на CPU-фикстуре; для путей с
побитовым равенством (greedy) дельта == 0.0; отсутствие измерения — null в
report.json (не ноль), как в прецеденте dtype_orig round 6.
"""

from __future__ import annotations

import math

import pytest

from stell_bench import (BENCH_SESSION, SPEC_PPL_DELTA_FIELD, BenchRunner,
                         BenchmarkProtocol, ppl_delta, ppl_from_logprobs)
from stell_bench.aggregate import RunMetrics
from stell_bench.harness import BenchResult


class _FakeBackend:
    """CPU-фикстура бэкенда: один token + finish за прогон (§5: usage фиксирован)."""

    def __init__(self) -> None:
        self._done = False

    def submit(self, request) -> int:
        self._done = False
        return 1

    def poll(self, sid):
        if self._done:
            return []
        self._done = True
        return [{"seq": 1, "type": "token", "text": "x"},
                {"seq": 2, "type": "finish",
                 "usage": {"prompt_tokens": 2, "completion_tokens": 2}}]

    def metrics(self):
        return {}


def test_field_name_is_fixed():
    """Имя поля схемы отчёта зафиксировано контрактом (ADR-13 [R6])."""
    assert SPEC_PPL_DELTA_FIELD == "spec_ppl_delta"


def test_ppl_known_value_on_cpu_fixture():
    """Известное значение PPL на фикстуре: [0.5, 0.5] → PPL = 2.0."""
    lp = [math.log(0.5), math.log(0.5)]
    assert ppl_from_logprobs(lp) == 2.0
    # Равномерная последовательность из 4 токенов p=0.25 → PPL = 4.0.
    lp4 = [math.log(0.25)] * 4
    assert ppl_from_logprobs(lp4) == pytest.approx(4.0)


def test_ppl_delta_bitwise_equal_is_exactly_zero():
    """Greedy-путь (побитовое равенство лог-вероятностей) → дельта ровно 0.0."""
    plain = [math.log(0.5), math.log(0.25), math.log(0.125)]
    d = ppl_delta(plain, list(plain))
    assert d["bitwise_equal"] is True
    assert d["delta_abs"] == 0.0
    assert d["delta_rel"] == 0.0
    assert d["ppl_plain"] == d["ppl_spec"]


def test_ppl_delta_known_shift_on_cpu_fixture():
    """Известная дельта на фикстуре: спек-путь хуже ровно на один токен p=0.25."""
    # plain: два токена p=0.5 → PPL=2.0; spec: 0.5, 0.25 → PPL=sqrt(0.5*0.25)*... :
    # mean(-log) = (ln2 + ln4)/2 = 1.5*ln2 → PPL = 2^1.5 ≈ 2.8284.
    plain = [math.log(0.5), math.log(0.5)]
    spec = [math.log(0.5), math.log(0.25)]
    d = ppl_delta(plain, spec)
    assert d["ppl_plain"] == 2.0
    assert d["ppl_spec"] == pytest.approx(2.0 ** 1.5)
    assert d["delta_abs"] == pytest.approx(2.0 ** 1.5 - 2.0)
    assert d["bitwise_equal"] is False


def test_ppl_delta_length_mismatch_rejected():
    """Разные длины = разные семплинг-конфиги — запрещено складывать (§5)."""
    with pytest.raises(ValueError):
        ppl_delta([0.1, 0.2], [0.1])


def test_report_schema_contains_field_default_none():
    """Схема отчёта содержит spec_ppl_delta; без измерения — null (не ноль)."""
    warm = [RunMetrics(ttft_ms=1.0, prefill_tok_s=10.0, decode_tok_s=20.0,
                       spec_decode_tok_s=0.0, tpot_ms=0.5,
                       prompt_tokens=10, completion_tokens=20)]
    result = BenchResult(medians={}, runs=warm, env={}, protocol={},
                         pending=True, notes=[])
    payload = result.to_dict()
    assert SPEC_PPL_DELTA_FIELD in payload
    assert payload[SPEC_PPL_DELTA_FIELD] is None


def test_runner_computes_delta_when_fn_available():
    """Харнесс заполняет spec_ppl_delta, если лог-вероятности доступны."""
    plain = [math.log(0.5)] * 2

    def fn():
        return plain, list(plain)  # побитовый greedy → дельта 0.0

    runner = BenchRunner(_FakeBackend(), BenchmarkProtocol(),
                         pending=True, notes=["test"], ppl_logprobs_fn=fn)
    result = runner.run_benchmark("prompt")
    assert result.spec_ppl_delta is not None
    assert result.spec_ppl_delta["delta_abs"] == 0.0
    assert result.spec_ppl_delta["ppl_plain"] == result.spec_ppl_delta["ppl_spec"]


def test_runner_honest_null_when_fn_unavailable():
    """Без callable поле остаётся null — отсутствие данных ≠ ноль (честно)."""
    runner = BenchRunner(_FakeBackend(), BenchmarkProtocol(), pending=True,
                         notes=["test"])
    result = runner.run_benchmark("prompt")
    assert result.spec_ppl_delta is None
    payload = result.to_dict()
    assert payload[SPEC_PPL_DELTA_FIELD] is None
