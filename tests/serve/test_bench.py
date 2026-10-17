"""Юнит-тесты bench: медиана, холодный+3 тёплых, протокол §5, отчёты,
quality-gates заготовки."""

from __future__ import annotations

import json

import pytest

from stell_bench import (BenchRunner, BenchmarkProtocol, aggregate_runs,
                         median, write_report)
from stell_bench.aggregate import RunMetrics
from stell_bench.quality_gates import (gsm8k_32, humaneval_32, load_schema,
                                       ppl_text, run_gates_pending)


def run(ttft, prefill, decode, spec, tpot, prompt=10, completion=20):
    return RunMetrics(ttft_ms=ttft, prefill_tok_s=prefill,
                      decode_tok_s=decode, spec_decode_tok_s=spec,
                      tpot_ms=tpot, prompt_tokens=prompt,
                      completion_tokens=completion)


def test_median_odd_and_even():
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    with pytest.raises(ValueError):
        median([])


def test_aggregate_runs_medians():
    runs = [run(10, 100, 50, 0, 14), run(20, 200, 60, 0, 15), run(30, 300, 70, 0, 16)]
    agg = aggregate_runs(runs)
    assert agg["ttft_ms"] == 20.0
    assert agg["prefill_tok_s"] == 200.0
    assert agg["decode_tok_s"] == 60.0
    assert agg["tpot_ms"] == 15.0
    assert agg["completion_tokens"] == 20.0


def test_protocol_fixed_contract():
    p = BenchmarkProtocol()
    p.validate()  # контракт §5: T=1, top-p=0.95, top-k=20, seed=0, ≤256
    assert p.session == {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                         "seed": 0, "max_tokens": 256, "stop": []}
    assert p.cold_runs == 1 and p.warm_runs == 3
    bad = BenchmarkProtocol(session={**p.session, "top_p": 0.9})
    with pytest.raises(ValueError):
        bad.validate()
    bad2 = BenchmarkProtocol(session={**p.session, "max_tokens": 300})
    with pytest.raises(ValueError):
        bad2.validate()


def test_runner_cold_plus_warm_median(mock_engine_lib):
    from stell_serve.app import default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(mock_engine_lib)
    backend.create(default_engine_cfg())
    runner = BenchRunner(backend, BenchmarkProtocol())
    result = runner.run_benchmark("Hello, bench!")
    backend.close()
    assert len(result.runs) == 3          # 3 тёплых прогона
    assert result.medians["ttft_ms"] >= 0
    assert result.pending is True         # CPU/mock — честно pending
    assert result.env["co_tenancy"] is False
    assert "kv_dtype" in result.env and "cuda_version" in result.env


def test_write_report(tmp_path):
    p = BenchmarkProtocol()
    runner = BenchRunner(None, p)  # backend не используется без run_benchmark
    class FakeResult:
        def to_dict(self):
            return {"medians": {"ttft_ms": 1.0}, "pending": True}
    out = write_report(FakeResult(), tmp_path, log_lines=["line1", "line2"])
    assert (out / "report.json").exists()
    assert (out / "run.log").exists()
    data = json.loads((out / "report.json").read_text())
    assert data["medians"]["ttft_ms"] == 1.0
    assert data["pending"] is True


def test_quality_gates_data_and_schema():
    assert len(gsm8k_32()) == 32
    assert len(humaneval_32()) == 32
    assert len(ppl_text().split()) > 100
    schema = load_schema()
    assert schema["datasets"]["gsm8k_32"]["file"] == "data/gsm8k_32.jsonl"


def test_quality_gates_pending_without_model():
    gates = run_gates_pending()
    assert len(gates) == 3
    assert all(g.pending for g in gates)
    assert all(g.value is None for g in gates)
