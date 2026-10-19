"""Живой runner quality-gates (round 2, поток E): PPL по logprob позиций
("positions":"all"), GSM8K-32 через жадное декодирование (корректность —
сопоставление чисел, без исполнения), HumanEval-32 — честный pending
(проверка корректности требует исполнения кода).

Математика PPL/декодирования проверяется на синтетических логитах с
известными значениями; источники логитов — stub-.so с `stell_logits`
(чистый ABI-путь) и НЕЗАВИСИМЫЙ Python-референс (fallback до merge core
[R1]; против golden-фикстур).
"""

from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("numpy", reason="golden-ref источник логитов требует numpy")

from stell_bench.quality_gates_runner import (  # noqa: E402
    GSM8K_MAX_STEPS, GSM8K_PROMPT_TOKENS, PPL_MAX_TOKENS,
    AbiLogitsSource, GoldenRefLogitsSource, _GatesSnapshot, _vocab_size,
    detokenize_ids, extract_answer_number, greedy_token, logprob_of,
    open_logits_source, ppl_from_logits, project_to_vocab, run_gates_live,
    run_gsm8k_gate, run_humaneval_gate, run_ppl_gate,
)

try:
    from .conftest import ROOT as _ROOT  # noqa: F401
except ImportError:  # pragma: no cover
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[2]


# --------------------------- математика: PPL/logprob -------------------------

def test_logprob_uniform_logits():
    # равномерные логиты → logprob = -ln(V); для V=4: -ln(4) ≈ -1.3863
    lp = logprob_of([1.0, 1.0, 1.0, 1.0], 2)
    assert abs(lp - (-math.log(4))) < 1e-12


def test_logprob_confident_prediction():
    # токен с наибольшим логитом → logprob ≈ 0; маловероятный → ≈ ln(eps/1)
    lp_hi = logprob_of([10.0, 0.0, 0.0, 0.0], 0)
    assert abs(lp_hi) < 1e-3                          # ≈ 0 (зазор 10 по логиту)
    lp_lo = logprob_of([10.0, 0.0, 0.0, 0.0], 1)
    assert -11.0 < lp_lo < -9.0                       # ≈ −10: ln(e^-10/1)


def test_ppl_from_logits_uniform_equals_vocab():
    # равномерные логиты на всех позициях → PPL = vocab (здесь 4)
    tokens = [0, 1, 2, 0]
    rows = [[1.0] * 4 for _ in range(len(tokens))]
    stats = ppl_from_logits(rows, tokens)
    assert abs(stats["ppl"] - 4.0) < 1e-9
    assert stats["n_pred_positions"] == 3             # P-1 позиций предсказания
    assert abs(stats["mean_nll"] - math.log(4)) < 1e-9


def test_ppl_from_logits_perfect_prediction():
    # one-hot логиты, «горячий» индекс = фактический следующий токен → PPL ≈ 1
    tokens = [3, 1, 2]
    rows = [[0, 10, 0, 0], [0, 0, 10, 0]]
    stats = ppl_from_logits(rows, tokens)
    assert abs(stats["ppl"] - 1.0) < 1e-3             # зазор 10 по логиту


def test_greedy_token_tie_smaller_index():
    # ничья → меньший индекс (конвенция golden, ADR-09 §4)
    assert greedy_token([1.0, 2.0, 2.0, 2.0]) == 1
    assert greedy_token([0.5, 0.5, 0.5]) == 0


# ------------------------- проекция байта → токен ----------------------------

def test_project_to_vocab_is_deterministic_and_in_range():
    vocab = _vocab_size("mini_dense")
    assert vocab == 16
    toks = project_to_vocab("The warehouse district wakes before dawn.", vocab)
    assert len(toks) > 0
    assert all(0 <= t < vocab for t in toks)
    assert toks == project_to_vocab("The warehouse district wakes before dawn.", vocab)


def test_detokenize_ids_digits_and_separator():
    assert detokenize_ids([3, 12, 7, 15]) == "3 7 "
    assert detokenize_ids([]) == ""


def test_extract_answer_number_last_number_wins():
    assert extract_answer_number("4 4 2 10 7") == 7
    assert extract_answer_number("1 2 3") == 3
    assert extract_answer_number("  ") is None


# --------------------- источник логитов: golden-ref (fallback) ---------------

@pytest.fixture(scope="module")
def golden_ref_dense():
    return GoldenRefLogitsSource("mini_dense")


def test_golden_ref_matches_golden_fixture_last_position(golden_ref_dense):
    """Независимый Python-референс воспроизводит golden-логиты (та же
    реализация, что сгенерировала фикстуры; сама сверена с HF ~1.5e-7)."""
    golden = json.loads(
        (_ROOT / "tests" / "models" / "golden" / "logits_mini_dense.json")
        .read_text())
    prompt = golden["prompt_tokens"]
    got = golden_ref_dense.logits_last(prompt)
    ref = golden["logits_ref"]
    worst = max(abs(a - b) / max(abs(a), abs(b), 1.0) for a, b in zip(ref, got))
    assert worst <= 1e-6


def test_golden_ref_logits_all_semantics(golden_ref_dense):
    """positions="all" по каузальной семантике: строка i — логиты последней
    позиции префикса tokens[:i+1]; строки совпадают с logits_last префиксов."""
    tokens = [1, 2, 3, 4]
    rows = golden_ref_dense.logits_all(tokens)
    assert len(rows) == len(tokens)
    assert all(len(r) == 16 for r in rows)
    for i in range(len(tokens)):
        assert rows[i] == golden_ref_dense.logits_last(tokens[:i + 1])


# ------------------------------- живые гейты ---------------------------------

def test_run_ppl_gate_live(golden_ref_dense):
    gate = run_ppl_gate(golden_ref_dense, "mini_dense")
    assert gate.pending is False
    assert gate.unit == "ppl"
    assert gate.value is not None and gate.value > 1.0
    assert gate.items == PPL_MAX_TOKENS - 1           # 127 позиций предсказания


def test_run_gsm8k_gate_live(golden_ref_dense):
    gate = run_gsm8k_gate(golden_ref_dense, "mini_dense")
    assert gate.pending is False
    assert gate.items == 32
    # мини-фикстура — не языковая модель: значение — механическая точка
    # регрессии пайплайна (0..1, детерминировано на этом прогоне)
    assert gate.value is not None and 0.0 <= gate.value <= 1.0


def test_run_humaneval_gate_stays_pending(golden_ref_dense):
    gate = run_humaneval_gate(golden_ref_dense, "mini_dense")
    assert gate.pending is True
    assert gate.value is None
    assert gate.items == 32
    assert any("исполнени" in n for n in gate.notes)   # честная причина


def test_run_gates_live_full_set():
    gates = run_gates_live()
    datasets = [g.dataset for g in gates]
    assert datasets == ["ppl_fixed_text[mini_dense]", "gsm8k_32[mini_dense]",
                        "humaneval_32[mini_dense]",
                        "ppl_fixed_text[mini_moe]", "gsm8k_32[mini_moe]",
                        "humaneval_32[mini_moe]"]
    assert all(g.pending is False for g in gates if g.dataset.startswith("ppl"))
    assert all(g.pending is False for g in gates if g.dataset.startswith("gsm8k"))
    assert all(g.pending is True for g in gates if g.dataset.startswith("humaneval"))
    snap = _GatesSnapshot(gates).to_dict()
    assert snap["pending"] is False                    # живые гейты: не все pending


# --------------------- ABI-путь: stell_logits через stub ---------------------

def test_abi_logits_source_over_stub(stub_backend):
    """Канонический ABI-путь ([R1]): тот же интерфейс LogitsSource поверх
    символа stell_logits (stub-.so с известным ответом)."""
    src = AbiLogitsSource(stub_backend)
    assert src.name == "abi:stell_logits"
    row = src.logits_last([1, 2, 3])
    assert row == [30, 31, 32, 33]
    rows = src.logits_all([1, 2, 3])
    assert len(rows) == 3
    src.close()  # no-op для стаба (close бэкенда не вызываем — фиксирует тест)


def test_abi_source_requires_symbol(mock_engine_lib):
    """ABI-источник честно отказывается работать без символа ([R1])."""
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(mock_engine_lib)
    try:
        with pytest.raises(RuntimeError, match="stell_logits"):
            AbiLogitsSource(backend)
    finally:
        backend.close()


def test_open_logits_source_abi_prefer_raises_without_symbol(real_engine_lib):
    """prefer='abi' без символа → явная ошибка (не молчаливый fallback)."""
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(real_engine_lib)
    try:
        if backend.supports_logits:
            pytest.skip("символ уже в .so (merge core) — ветка fallback не актуальна")
    finally:
        backend.close()
    with pytest.raises(RuntimeError, match="stell_logits"):
        open_logits_source("mini_dense", str(real_engine_lib), prefer="abi")


def test_open_logits_source_auto_falls_back_to_golden_ref(real_engine_lib):
    """auto: без символа в .so — честный fallback на независимый референс
    (числа помечаются provenance 'golden-ref (fallback)')."""
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(real_engine_lib)
    try:
        if backend.supports_logits:
            pytest.skip("символ уже в .so (merge core) — ABI-путь каноничен")
    finally:
        backend.close()
    src = open_logits_source("mini_dense", str(real_engine_lib), prefer="auto")
    assert isinstance(src, GoldenRefLogitsSource)
    assert "fallback" in src.name
