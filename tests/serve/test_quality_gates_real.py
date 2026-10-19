"""Режим «реальная модель» runner'а quality-gates (R8, round 4).

Проверяется: resolve_real_model читает каталог модели через аргументы и env
(STELL_R8_REAL_MODEL_DIR — по прецеденту R7), протокол-константы, честный
pending «веса не предоставлены» без весов, громкая ошибка на битом источнике;
живой прогон — через ABI на синтетическом каталоге «реалистичного профиля»
(synthetic_model.py: MoE 4×8, vocab=350, РЕАЛЬНЫЙ tokenizer.json) с
provenance (модель, размер, дата, co-tenancy).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="генератор синтетических весов требует numpy")

from stell_bench import quality_gates_runner as qgr                   # noqa: E402
from stell_bench.bpe_tokenizer import BpeTokenizer                       # noqa: E402
from stell_bench.quality_gates_runner import (                            # noqa: E402
    ENV_REAL_MODEL_DIR, GSM8K_MAX_STEPS, GSM8K_PROMPT_TOKENS,
    HUMANEVAL_MAX_STEPS, HUMANEVAL_PROMPT_TOKENS, PPL_ALL_MAX_ELEMENTS,
    PPL_MAX_TOKENS, REAL_DESCRIPTION_NAME, REAL_TOKENIZER_NAME,
    REAL_WEIGHTS_NAME, RealModelSpec, _GatesSnapshot, logprob_of, main,
    ppl_mode_for, real_provenance, resolve_real_model, run_gates_real,
    run_gates_real_pending, run_ppl_gate,
)
from stell_bench.synthetic_model import (                                 # noqa: E402
    DEFAULT_SEED, build_realistic_profile,
)

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def synthetic_model_dir(tmp_path_factory):
    """Синтетический каталог «реалистичного профиля»: description.json (ADR-09)
    + model.safetensors + РЕАЛЬНЫЙ tokenizer.json (эталон HF tokenizers)."""
    d = tmp_path_factory.mktemp("realistic_model")
    build_realistic_profile(d)
    return d


@pytest.fixture(scope="module")
def synthetic_spec(synthetic_model_dir):
    spec = resolve_real_model(model_dir=str(synthetic_model_dir))
    assert spec is not None
    return spec


# ------------------------- resolve_real_model --------------------------------

def test_protocol_constants():
    """Протокол-константы R8 зафиксированы (детерминизм между версиями)."""
    assert REAL_DESCRIPTION_NAME == "description.json"
    assert REAL_WEIGHTS_NAME == "model.safetensors"
    assert REAL_TOKENIZER_NAME == "tokenizer.json"
    assert ENV_REAL_MODEL_DIR == "STELL_R8_REAL_MODEL_DIR"
    # протокол декодирования (тот же контракт, что у mini-фикстур):
    assert PPL_MAX_TOKENS == 128
    assert GSM8K_PROMPT_TOKENS == 64 and GSM8K_MAX_STEPS == 16
    assert HUMANEVAL_PROMPT_TOKENS == 64 and HUMANEVAL_MAX_STEPS == 32
    # R8 fixup: порог режима PPL (элементов логитов за один вызов) —
    # серверный прогон tiny-qwen2moe (vocab=151646) завис на "all"
    # (128 × 151646 ≈ 19.4 млн ≈ 300+ МиБ JSON):
    assert PPL_ALL_MAX_ELEMENTS == 2_000_000


def test_resolve_from_directory(synthetic_spec, synthetic_model_dir):
    """Каталог модели читается: описание ADR-09 + веса + tokenizer.json."""
    assert isinstance(synthetic_spec, RealModelSpec)
    assert synthetic_spec.root == synthetic_model_dir
    assert synthetic_spec.desc_path == synthetic_model_dir / REAL_DESCRIPTION_NAME
    assert synthetic_spec.weights_path == synthetic_model_dir / REAL_WEIGHTS_NAME
    assert synthetic_spec.tokenizer_path == synthetic_model_dir / REAL_TOKENIZER_NAME
    assert synthetic_spec.name == "synthetic-realistic-moe"
    assert synthetic_spec.vocab_size == 350
    assert synthetic_spec.desc["class"] == "moe"
    assert synthetic_spec.desc["n_layers"] == 4          # больше слоёв, чем mini (2)
    assert synthetic_spec.desc["moe"]["n_experts"] == 8  # больше экспертов, чем mini (4)


def test_resolve_reads_env(monkeypatch, synthetic_model_dir):
    monkeypatch.setenv(ENV_REAL_MODEL_DIR, str(synthetic_model_dir))
    spec = resolve_real_model()
    assert spec is not None and spec.root == synthetic_model_dir


def test_resolve_env_argument_priority(monkeypatch, synthetic_model_dir, tmp_path):
    """Явный аргумент приоритетнее env (честный выбор источника)."""
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(ENV_REAL_MODEL_DIR, str(other))
    spec = resolve_real_model(model_dir=str(synthetic_model_dir))
    assert spec is not None and spec.root == synthetic_model_dir


def test_resolve_no_source_returns_none(monkeypatch):
    monkeypatch.delenv(ENV_REAL_MODEL_DIR, raising=False)
    assert resolve_real_model() is None


def test_resolve_missing_dir_is_loud(monkeypatch):
    monkeypatch.delenv(ENV_REAL_MODEL_DIR, raising=False)
    with pytest.raises(FileNotFoundError, match="каталог модели не найден"):
        resolve_real_model(model_dir="/no/such/dir-42")


def test_resolve_env_missing_dir_is_loud(monkeypatch):
    monkeypatch.setenv(ENV_REAL_MODEL_DIR, "/no/such/dir-42")
    with pytest.raises(FileNotFoundError, match="каталог модели не найден"):
        resolve_real_model()


def test_resolve_missing_description_is_loud(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / REAL_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="описание ADR-09"):
        resolve_real_model(model_dir=str(d))


def test_resolve_missing_weights_is_loud_with_hint(tmp_path):
    """Веса отсутствуют — громкая ошибка с подсказкой (не молчаливый pending)."""
    d = tmp_path / "model"
    d.mkdir()
    (d / REAL_DESCRIPTION_NAME).write_text(
        json.dumps({"schema": "1.0", "name": "x", "vocab_size": 4}))
    with pytest.raises(FileNotFoundError, match="веса не найдены"):
        resolve_real_model(model_dir=str(d))


def test_resolve_hf_shard_dir_is_honest_error(tmp_path):
    """Каталог с HF-шардами (model-00001-of-…, без model.safetensors) — честная
    ошибка: plain-лоадер safetensors открывает ровно один файл (шарды сначала
    склеиваются/конвертируются; догадки запрещены)."""
    d = tmp_path / "shards"
    d.mkdir()
    (d / REAL_DESCRIPTION_NAME).write_text(
        json.dumps({"schema": "1.0", "name": "x", "vocab_size": 4}))
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="один файл|скле"):
        resolve_real_model(model_dir=str(d))


def test_resolve_explicit_desc_and_weights_pair(synthetic_model_dir, tmp_path):
    d = tmp_path / "pair"
    d.mkdir()
    desc = synthetic_model_dir / REAL_DESCRIPTION_NAME
    weights = synthetic_model_dir / REAL_WEIGHTS_NAME
    # только один файл — ошибка (пара обязательна целиком)
    with pytest.raises(RuntimeError, match="вместе"):
        resolve_real_model(desc_json=str(desc))
    with pytest.raises(RuntimeError, match="вместе"):
        resolve_real_model(weights=str(weights))
    spec = resolve_real_model(desc_json=str(desc), weights=str(weights))
    assert spec is not None
    assert spec.desc_path == desc and spec.weights_path == weights
    # tokenizer.json ищется рядом с описанием
    assert spec.tokenizer_path == desc.parent / REAL_TOKENIZER_NAME


# ----------------------------- честный pending --------------------------------

def test_pending_without_weights():
    """Honest pending «веса не предоставлены» (по образцу R7): значения
    качества не измерены, причина — в notes и provenance."""
    reason = "веса не предоставлены (тест)"
    gates, provenance = run_gates_real_pending(reason)
    assert all(g.pending for g in gates)
    assert [g.value for g in gates] == [None, None, None]
    datasets = [g.dataset for g in gates]
    assert datasets == ["gsm8k_32", "humaneval_32", "ppl_fixed_text"]
    for g in gates:
        assert any("веса не предоставлены" in n and reason in n for n in g.notes)
    assert provenance["weights_provided"] is False
    assert provenance["reason"] == reason
    assert provenance["co_tenancy"] is None
    snap = _GatesSnapshot(gates, extra={"mode": "real-model-pending",
                                        "provenance": provenance})
    assert snap.to_dict()["pending"] is True


def test_cli_pending_writes_report(monkeypatch, tmp_path):
    """CLI `--real` без источника — честный pending-отчёт (exit 0) в
    переданный --results-dir; env в этом прогоне не задан."""
    monkeypatch.delenv(ENV_REAL_MODEL_DIR, raising=False)
    out = tmp_path / "results"
    rc = main(["--real", "--results-dir", str(out)])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["mode"] == "real-model-pending"
    assert report["pending"] is True
    assert report["provenance"]["weights_provided"] is False
    assert "STELL_R8_REAL_MODEL_DIR" in report["pending_reason"]
    assert (out / "run.log").is_file()


def test_cli_bad_source_is_loud_failure(monkeypatch, tmp_path):
    """Явно заданный битый источник — громкая ошибка (exit 1, не pending)."""
    monkeypatch.delenv(ENV_REAL_MODEL_DIR, raising=False)
    rc = main(["--real", "--real-model-dir", "/no/such/dir-42",
               "--results-dir", str(tmp_path)])
    assert rc == 1


# -------------------------- живой прогон через ABI ----------------------------

def test_real_gates_live_via_abi(synthetic_spec, real_engine_lib):
    """Живой прогон гейтов через ABI (stell_logits) на каталоге модели:
    PPL + GSM8K-32 + HumanEval-32, токенизация — РЕАЛЬНЫЙ byte-level BPE
    (tokenizer.json каталога), значения — точки регрессии конвейера."""
    gates, provenance = run_gates_real(synthetic_spec,
                                       engine_lib=str(real_engine_lib))
    datasets = [g.dataset for g in gates]
    assert datasets == ["ppl_fixed_text[synthetic-realistic-moe]",
                        "gsm8k_32[synthetic-realistic-moe]",
                        "humaneval_32[synthetic-realistic-moe]"]
    assert all(g.pending is False for g in gates)
    ppl, gsm, he = gates
    assert ppl.value is not None and ppl.value > 1.0
    assert ppl.items == PPL_MAX_TOKENS - 1
    assert gsm.items == 32 and he.items == 32
    assert 0.0 <= gsm.value <= 1.0 and 0.0 <= he.value <= 1.0
    # provenance (модель, размер, дата, co-tenancy):
    assert provenance["mode"] == "real-model"
    assert provenance["model"] == "synthetic-realistic-moe"
    assert provenance["weights_bytes"] > 0
    assert provenance["vocab_size"] == 350
    assert provenance["n_layers"] == 4 and provenance["n_experts"] == 8
    assert provenance["tokenizer_json"].endswith(REAL_TOKENIZER_NAME)
    assert provenance["tokenization"] == "byte-level BPE (tokenizer.json)"
    assert provenance["logits_source"] == "abi:stell_logits"
    assert provenance["date_utc"]           # точка регрессии датирована
    assert provenance["co_tenancy"] is False
    # честная пометка синтетического каталога в notes гейтов
    for g in gates:
        assert any("синтетический реалистичный профиль" in n for n in g.notes)


def test_real_provenance_co_tenancy_flag(synthetic_spec):
    """Provenance точки регрессии (модель, размер, дата, co-tenancy) —
    чистая функция: флаг co_tenancy честно попадает в provenance."""
    prov = real_provenance(synthetic_spec, "libstell_core.so", True, None)
    assert prov["co_tenancy"] is True
    assert prov["mode"] == "real-model"
    assert prov["model"] == "synthetic-realistic-moe"
    assert prov["weights_bytes"] == (synthetic_spec.weights_path.stat().st_size)
    assert prov["date_utc"].endswith("Z") or "+" in prov["date_utc"]
    prov2 = real_provenance(synthetic_spec, "libstell_core.so", False, None)
    assert prov2["co_tenancy"] is False


def test_real_gates_projection_fallback_without_tokenizer(real_engine_lib,
                                                          tmp_path):
    """Каталог без tokenizer.json: честная проекция байта → токен (механи-
    ческий базовый референс), а токенизация помечена в provenance."""
    d = tmp_path / "notok"
    build_realistic_profile(d, with_tokenizer=False)
    spec = resolve_real_model(model_dir=str(d))
    assert spec is not None and spec.tokenizer_path is None
    gates, provenance = run_gates_real(spec, engine_lib=str(real_engine_lib))
    assert provenance["tokenizer_json"] is None
    assert "проекция байта" in provenance["tokenization"]
    for g in gates:
        assert any("проекция байта" in n for n in g.notes)  # механический референс


def test_real_gates_abi_only_no_silent_fallback(synthetic_spec, mock_engine_lib):
    """Без stell_logits реальный режим честно отказывается работать:
    golden-ref умеет только mini-фикстуры — подмена источника запрещена."""
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(mock_engine_lib)
    try:
        if backend.supports_logits:
            pytest.skip("символ уже в .so (merge core) — ветка не актуальна")
    finally:
        backend.close()
    with pytest.raises(RuntimeError, match="stell_logits"):
        run_gates_real(synthetic_spec, engine_lib=str(mock_engine_lib))


def test_cli_real_live_end_to_end(synthetic_model_dir, tmp_path, monkeypatch):
    """CLI-контур целиком: --real --real-model-dir … → live-real отчёт с
    provenance в bench/results-структуре (--results-dir для теста)."""
    monkeypatch.delenv(ENV_REAL_MODEL_DIR, raising=False)
    out = tmp_path / "results"
    rc = main(["--real", "--real-model-dir", str(synthetic_model_dir),
               "--results-dir", str(out)])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["mode"] == "live-real"
    assert report["pending"] is False
    assert len(report["gates"]) == 3
    assert report["provenance"]["model"] == "synthetic-realistic-moe"


def test_bpe_tokenizer_matches_reference_impl():
    """Санити: BPE-токенизатор runner'а сверён с эталоном HF tokenizers
    (детально — tests/serve/test_bpe_tokenizer.py)."""
    tok = BpeTokenizer(str(_ROOT / "tests" / "models" / "fixtures" / "tokenizer"
                           / "tokenizer.json"))
    ref = json.loads((_ROOT / "tests" / "models" / "fixtures" / "tokenizer"
                      / "tokenizer_ref.json").read_text())
    for s, ids in zip(ref["strings"], ref["ids"]):
        assert tok.encode(s) == ids


# ---------------------------------------------------------------------------
# R8 fixup: PPL на больших словарях (серверный прогон tiny-qwen2moe,
# vocab=151646, завис на positions:"all" — 128 × 151646 ≈ 19.4 млн логитов
# ≈ 300+ МиБ JSON за один вызов). Порог PPL_ALL_MAX_ELEMENTS → режим
# 'incremental-last': teacher-forcing по позициям, повторные positions:"last"
# по растущему префиксу (по строке логитов за вызов). Чанки "all" с окном
# контекста честно невозможны: каузальная семантика требует полный префикс
# для каждой строки (ABI не имеет смещения контекста).
# ---------------------------------------------------------------------------

class _UniformStub:
    """Равномерный стаб источника логитов: PPL = vocab (nll = log V)."""

    def __init__(self, vocab: int) -> None:
        self._vocab = vocab
        self.n_calls_last = 0
        self.n_calls_all = 0
        self.name = f"stub:uniform(vocab={vocab})"

    def logits_last(self, tokens):
        self.n_calls_last += 1
        return [0.0] * self._vocab

    def logits_all(self, tokens):
        self.n_calls_all += 1
        return [[0.0] * self._vocab for _ in tokens]

    def close(self) -> None:
        pass


class _PrefixDependentStub:
    """Детерминированный стаб: строка логитов зависит от длины префикса —
    для сверки путей 'all' и 'incremental-last' (та же каузальная семантика:
    rows[i] = logits_last(tokens[:i+1]))."""

    def __init__(self, vocab: int) -> None:
        self._vocab = vocab
        self.name = f"stub:prefix-dependent(vocab={vocab})"

    def _row(self, prefix_len: int):
        # детерминированная волна, значения в (-0.5, 0.5)
        return [((prefix_len * 31 + j * 7) % 97) / 97.0 - 0.5
                for j in range(self._vocab)]

    def logits_last(self, tokens):
        return self._row(len(tokens))

    def logits_all(self, tokens):
        return [self._row(i + 1) for i in range(len(tokens))]

    def close(self) -> None:
        pass


def test_ppl_mode_threshold():
    """Порог переключения: n_vocab × n_positions > PPL_ALL_MAX_ELEMENTS →
    incremental-last, иначе all (граница включительно — '>')."""
    assert ppl_mode_for(350, 128) == "all"                  # синтет. профиль R8
    assert ppl_mode_for(16, 128) == "all"                   # mini-фикстуры
    assert ppl_mode_for(151646, 128) == "incremental-last"  # tiny-qwen2moe
    boundary = PPL_ALL_MAX_ELEMENTS
    assert ppl_mode_for(boundary // 128, 128) == "all"        # ровно порог → all
    assert ppl_mode_for(boundary // 128 + 1, 128) == "incremental-last"


def test_ppl_incremental_math_matches_all_path(monkeypatch):
    """Эквивалентность путей: инкрементальный last даёт ТО ЖЕ значение PPL,
    что и один вызов positions:"all" (та же каузальная семантика строк).
    Переключение режима — monkeypatch порога (математика не меняется)."""
    src = _PrefixDependentStub(350)
    try:
        gate_all = run_ppl_gate(src, "stub-eq", vocab=350)   # порог дефолт
        monkeypatch.setattr(qgr, "PPL_ALL_MAX_ELEMENTS", 10)  # порог занижен
        gate_inc = run_ppl_gate(src, "stub-eq", vocab=350)
    finally:
        src.close()
    assert gate_all.value == gate_inc.value          # побитово
    assert gate_all.items == gate_inc.items == PPL_MAX_TOKENS - 1
    assert any("режим PPL: all" in n for n in gate_all.notes)
    assert any("режим PPL: incremental-last" in n for n in gate_inc.notes)


def test_ppl_big_vocab_incremental_stub_no_hang():
    """Большой vocab (120000 ≥ 100к), полный протокол (128 позиций):
    PPL считается через инкрементальный last — 127 вызовов по ОДНОЙ строке
    (~1.2 МиБ), не 300+ МиБ одним вызовом; PPL = vocab (равномерный стаб)."""
    vocab = 120_000
    src = _UniformStub(vocab)
    try:
        gate = qgr.run_ppl_gate(src, "stub-big-vocab", vocab=vocab)
    finally:
        src.close()
    assert gate.pending is False
    assert gate.items == PPL_MAX_TOKENS - 1
    assert src.n_calls_last == gate.items            # вызовы по строке за раз
    assert src.n_calls_all == 0                      # "all" НЕ вызывался
    assert abs(gate.value - vocab) < 1e-6            # PPL = exp(log V) = V
    note = [n for n in gate.notes if "режим PPL" in n][0]
    assert "incremental-last" in note and "120000" in note


def test_real_provenance_ppl_mode(synthetic_spec):
    """PPL-режим честно фиксируется в provenance ('all' / 'incremental-last'):
    точный — из фактического прогона; fallback — по протоколу (vocab × 128)."""
    prov_all = real_provenance(synthetic_spec, "lib", False, None,
                               ppl_mode="all")
    assert prov_all["ppl_mode"] == "all"
    # fallback (без явного режима) — по протоколу: vocab=350 × 128 ≤ порога
    assert real_provenance(synthetic_spec, "lib", False, None)["ppl_mode"] == "all"
    big = RealModelSpec(synthetic_spec.desc_path, synthetic_spec.weights_path,
                        {**synthetic_spec.desc, "vocab_size": 151646},
                        root=synthetic_spec.root)
    assert real_provenance(big, "lib", False, None)["ppl_mode"] == \
        "incremental-last"


def test_ppl_big_vocab_live_engine_incremental(real_engine_lib, tmp_path,
                                               monkeypatch):
    """Живой ABI-прогон PPL на каталоге с vocab=120000: инкрементальный путь,
    НЕ виснет (позиции ограничены для теста: PPL_MAX_TOKENS→24 — выше порога
    120000×23 > 2M, протокол прогона остаётся 128 — константа не меняется)."""
    d = tmp_path / "bigvocab"
    build_realistic_profile(d, vocab_size=120_000, with_tokenizer=False)
    spec = resolve_real_model(model_dir=str(d))
    assert spec is not None and spec.vocab_size == 120_000

    from stell_serve.engine import CtypesEngineBackend

    monkeypatch.setattr(qgr, "PPL_MAX_TOKENS", 24)   # лимит позиций ДЛЯ ТЕСТА
    backend = CtypesEngineBackend(real_engine_lib)
    source = None
    try:
        assert backend.supports_logits
        backend.create({"runner": "ops",
                        "engine": {"max_slots": 16, "batch_step_max": 16},
                        "model": {"description": str(spec.desc_path),
                                  "weights": str(spec.weights_path),
                                  "weights_format": "nvfp4w4a16",
                                  "tokenizer": "byte"}})
        source = qgr.AbiLogitsSource(backend)
        gate = run_ppl_gate(source, spec.name, vocab=spec.vocab_size,
                            notes_extra=["big-vocab R8 fixup: live incremental"])
    finally:
        if source is not None:
            source.close()
        else:
            backend.close()
    assert gate.pending is False
    assert gate.items == 24 - 1                      # 23 позиции предсказания
    assert gate.value is not None and gate.value > 0.0
    note = [n for n in gate.notes if "режим PPL" in n][0]
    assert "incremental-last" in note
    assert any("big-vocab R8 fixup" in n for n in gate.notes)
