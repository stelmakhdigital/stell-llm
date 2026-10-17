"""Тесты runner'а калибровщика на CPU-замерах (round 2, calibrate↔bench)."""

from __future__ import annotations

import json

import pytest

from stell_calibrate.runner import decode_run, measure_tok_s, run_calibration


@pytest.fixture()
def cal_backend(mock_engine_lib):
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(mock_engine_lib)
    backend.create({"model": "mock-model",
                    "mock": {"ttft_ms": 1.0, "tpot_ms": 14.0}})
    yield backend
    backend.close()


def test_decode_run_drives_mock_abi(cal_backend):
    # прогон реально проходит через C ABI (submit/poll) и возвращает n токенов
    assert decode_run(cal_backend, 0.0, n_tokens=8) == 8


def test_measure_tok_s_matches_synthetic_rate(cal_backend):
    # синтетическая модель декода: tpot 14 мс → ~71.4 tok/s (допуск для CI)
    tok_s = measure_tok_s(cal_backend, 14.0, n_tokens=8, repeats=3)
    assert 50.0 <= tok_s <= 110.0


def test_calibration_decision_on_mock(mock_engine_lib):
    run = run_calibration(str(mock_engine_lib))
    d = run.decision
    # спек-драфт: 100/71.4 ≈ 1.4 ≥ 1.05, accept 0.7 ≥ 0.5 → вкл
    assert d["spec_draft"] is True
    # KV-квант: 83.3/71.4 ≈ 1.17 ≥ 1.03, penalty 0.5% ≤ 1% → вкл, E4M3
    assert d["kv_quant"] is True and d["kv_dtype"] == "E4M3"
    # NVMe-тир: 80/200 = 0.4 ≥ 0.25 → вкл; граница ≥ 3 GiB
    assert d["nvme_tier"] is True
    assert d["tier_boundary_bytes"] >= 3 * (1 << 30)
    # честность: замеры есть, результат pending (нет GPU/модели)
    assert run.measurements["decode_tok_s"] > 0
    assert run.pending is True
    assert any("синтетическ" in n for n in run.notes)
    assert any("не отдаёт логитов" in n for n in run.notes)


def test_runner_writes_report_to_results_local(mock_engine_lib, tmp_path):
    from stell_calibrate.runner import main

    rc = main(["--engine-lib", str(mock_engine_lib), "--results-dir",
               str(tmp_path), "--n-tokens", "8"])
    assert rc == 0
    report = tmp_path / "report.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["pending"] is True
    assert data["decision"]["spec_draft"] is True
    assert "decode_tok_s" in data["measurements"]
    # bench/results/local/ — gitignored (конвенция отчётов round 1)
    from stell_bench.report import results_local_dir_for

    d = results_local_dir_for("calibrate-local")
    assert "results" in d.parts and "local" in d.parts
