"""Обёртка R8-прогона на реальных весах + wall-протокол (round 5, [R8]).

Проверяется ([R8] planner round 5, [Q1]):
- run_real_gates.sh: корректные env (PYTHONPATH), dry-run, честный pending
  e2e, wall-замер (date-дифф) — в wall.json И в provenance report.json
  (обе цифры до/после [R1] — в артефакте), provenance переживает отказ
  (громкая ошибка runner'а → wall.json с exit_code=1);
- wall_provenance(): чистая функция wall-полей (validation, отсутствие
  непереданных полей);
- dtype-происхождение в provenance ([R5] models: dtype_orig из описания).

Обёртка НЕ запускает тяжёлых прогонов: тесты идут в pending-режиме
(без каталога модели) и на битом источнике — это единственные быстрые
пути; живой ABI-прогон покрыт test_quality_gates_real.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="синтетический каталог требует numpy")

from stell_bench.quality_gates_runner import (                         # noqa: E402
    RealModelSpec, main, real_provenance, resolve_real_model, wall_provenance,
)
from stell_bench.synthetic_model import build_realistic_profile        # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_WRAPPER = _ROOT / "src" / "bench" / "stell_bench" / "run_real_gates.sh"

_STELL_ENV_VARS = ("STELL_R8_MODEL_DIR", "STELL_R8_RESULTS_DIR",
                   "STELL_R8_TIMEOUT", "STELL_R8_CO_TENANCY",
                   "STELL_R8_ENGINE_LIB", "STELL_R8_WALL_PHASE",
                   "STELL_R8_WALL_BEFORE", "STELL_R8_DRY_RUN",
                   "STELL_R8_REAL_MODEL_DIR")


@pytest.fixture(scope="module")
def synthetic_spec(tmp_path_factory):
    """Синтетический каталог «реалистичного профиля» (MoE 4×8, vocab=350,
    РЕАЛЬНЫЙ tokenizer.json) — как в test_quality_gates_real."""
    d = tmp_path_factory.mktemp("realistic_model")
    build_realistic_profile(d)
    spec = resolve_real_model(model_dir=str(d))
    assert spec is not None
    return spec


def _run_wrapper(env_over: dict, timeout: int = 300) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("STELL_")}
    env.update({k: str(v) for k, v in env_over.items()})
    return subprocess.run(["bash", str(_WRAPPER)], env=env, timeout=timeout,
                          capture_output=True, text=True)


# ------------------------------ обёртка ---------------------------------------

def test_wrapper_exists_and_is_shell():
    """Обёртка на месте, shell-синтаксис валиден, исполняема."""
    assert _WRAPPER.is_file()
    rc = subprocess.run(["bash", "-n", str(_WRAPPER)])
    assert rc.returncode == 0
    assert os.access(_WRAPPER, os.X_OK)


def test_wrapper_dry_run_prints_command(tmp_path):
    """Dry-run: команда runner'а с PYTHONPATH (python + src/bench) и
    --results-dir; запуск не происходит (артефактов нет)."""
    results = tmp_path / "results"
    proc = _run_wrapper({
        "STELL_R8_MODEL_DIR": "",
        "STELL_R8_RESULTS_DIR": results,
        "STELL_R8_TIMEOUT": "120",
        "STELL_R8_DRY_RUN": "1",
    })
    assert proc.returncode == 0
    lines = proc.stdout.strip().splitlines()
    assert lines[0].startswith("PYTHONPATH=")
    pythonpath = lines[0].split("=", 1)[1]
    parts = pythonpath.split(":")
    # PYTHONPATH = <корень>/python:<корень>/src/bench (без хвостов чужих env):
    assert parts == [f"{_ROOT}/python", f"{_ROOT}/src/bench"]
    cmd = " ".join(lines[1:])
    assert cmd.startswith("timeout 120")
    assert "-m stell_bench.quality_gates_runner --real --live" in cmd
    assert f"--results-dir {results}" in cmd
    assert "--real-model-dir" not in cmd    # источник пуст → pending-режим
    assert not results.exists()             # dry-run ничего не создаёт


def test_wrapper_pending_end_to_end(tmp_path):
    """Полный контур обёртки в pending-режиме (без каталога модели):
    report.json real-model-pending + wall.json provenance обёртки
    (wall_seconds, exit_code, git-коммит, co-tenancy, timeout)."""
    results = tmp_path / "results"
    proc = _run_wrapper({
        "STELL_R8_MODEL_DIR": "",
        "STELL_R8_RESULTS_DIR": str(results),
        "STELL_R8_TIMEOUT": "120",
        "STELL_R8_CO_TENANCY": "1",
    })
    assert proc.returncode == 0
    report = json.loads((results / "report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "real-model-pending"
    wall = json.loads((results / "wall.json").read_text(encoding="utf-8"))
    assert wall["wrapper"] == "src/bench/stell_bench/run_real_gates.sh"
    assert wall["exit_code"] == 0 and wall["killed_by_timeout"] is False
    assert wall["wall_seconds"] >= 0
    assert wall["timeout_s"] == 120
    assert wall["co_tenancy"] is True
    assert wall["git_commit"]          # фиксируется из репо обёртки
    assert wall["date_utc"].endswith("+00:00")
    assert wall["model_dir"] is None   # источник не задан → честный pending


def test_wrapper_patches_report_provenance(tmp_path):
    """Wall-поля (wall_seconds/wall_phase/wall_before_seconds) вплетаются
    в provenance report.json — обе цифры до/после [R1] в самой точке
    регрессии (требование planner [R1])."""
    results = tmp_path / "results"
    proc = _run_wrapper({
        "STELL_R8_MODEL_DIR": "",
        "STELL_R8_RESULTS_DIR": str(results),
        "STELL_R8_TIMEOUT": "120",
        "STELL_R8_WALL_PHASE": "after",
        "STELL_R8_WALL_BEFORE": "4797.5",
    })
    assert proc.returncode == 0
    report = json.loads((results / "report.json").read_text(encoding="utf-8"))
    prov = report["provenance"]
    assert prov["wall_phase"] == "after"
    assert prov["wall_before_seconds"] == 4797.5
    assert prov["wall_seconds"] >= 0
    # wall_before_seconds в report.json == цифре из wall.json (один источник):
    wall = json.loads((results / "wall.json").read_text(encoding="utf-8"))
    assert wall["wall_before_seconds"] == prov["wall_before_seconds"]


def test_wrapper_loud_failure_still_records_wall(tmp_path):
    """Прогон не удался (битый источник → громкая ошибка runner'а, rc=1):
    wall.json ВСЁГДА пишется (wall-факт переживает отказ — прогон до [R1]
    был убит timeout'ом без report.json)."""
    results = tmp_path / "results"
    proc = _run_wrapper({
        "STELL_R8_MODEL_DIR": str(tmp_path / "no-such-model-dir"),
        "STELL_R8_RESULTS_DIR": str(results),
        "STELL_R8_TIMEOUT": "120",
    })
    assert proc.returncode == 1
    wall = json.loads((results / "wall.json").read_text(encoding="utf-8"))
    assert wall["exit_code"] == 1 and wall["killed_by_timeout"] is False
    assert wall["model_dir"] == str(tmp_path / "no-such-model-dir")
    assert not (results / "report.json").exists()


def test_wrapper_default_results_dir_uses_model_slug(tmp_path):
    """Results-dir по умолчанию: bench/results/<дата>-quality-gates-real-<имя
    каталога модели> — слаг из каталога модели (как на сервере)."""
    model_dir = tmp_path / "tiny-qwen2moe"
    model_dir.mkdir()  # каталог есть (пустой — прогон упадёт; нужен только dry-run)
    proc = _run_wrapper({
        "STELL_R8_MODEL_DIR": str(model_dir),
        "STELL_R8_DRY_RUN": "1",
    })
    assert proc.returncode == 0
    assert f"-quality-gates-real-{model_dir.name}" in proc.stdout
    assert f"--real-model-dir {model_dir}" in proc.stdout


# --------------------- wall-поля provenance (runner) ---------------------------

def test_wall_provenance_fields_and_validation():
    """wall_provenance(): только переданные поля (не измерено — не записано),
    rounding, validation wall_phase."""
    assert wall_provenance(12.3456) == {"wall_seconds": 12.346}
    assert wall_provenance(1.0, "before") == {
        "wall_seconds": 1.0, "wall_phase": "before"}
    assert wall_provenance(None, "after", 4797.0) == {
        "wall_phase": "after", "wall_before_seconds": 4797.0}
    assert wall_provenance(None) == {}          # ничего не передано — пусто
    with pytest.raises(ValueError, match="before.*after"):
        wall_provenance(None, "mid")            # чужая фаза — громкая ошибка


def test_cli_wall_fields_reach_provenance(tmp_path, monkeypatch):
    """CLI --wall-seconds/--wall-phase/--wall-before-seconds → provenance
    отчёта (и pending-режима тоже)."""
    monkeypatch.delenv("STELL_R8_REAL_MODEL_DIR", raising=False)
    out = tmp_path / "results"
    rc = main(["--real", "--results-dir", str(out),
               "--wall-seconds", "0.1", "--wall-phase", "after",
               "--wall-before-seconds", "4797.5"])
    assert rc == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    prov = report["provenance"]
    assert prov["wall_seconds"] == 0.1
    assert prov["wall_phase"] == "after"
    assert prov["wall_before_seconds"] == 4797.5


# --------------------- dtype-происхождение ([R5] models) -----------------------

def test_real_provenance_dtype_orig_from_description(synthetic_spec):
    """provenance несёт dtype-происхождение: dtype_weights из описания
    (определяет weights_format), dtype_orig — исходный dtype до конверсии
    ([R5] models: BF16-источник → 'bf16'); поле отсутствует в описании —
    честный None (не догадки)."""
    prov = real_provenance(synthetic_spec, "lib.so", False, None,
                           weights_format="nvfp4w4a16")
    assert prov["dtype_weights"] == "NVFP4"
    assert prov["weights_format"] == "nvfp4w4a16"   # маппинг — в run_gates_real
    assert prov["dtype_orig"] is None        # синтетическое описание без поля
    # дефолт weights_format не изменён (plain-сценарий):
    assert real_provenance(synthetic_spec, "lib.so", False, None)["weights_format"]\
        == "safetensors"
    # описание с dtype_orig (как будет от [R5] models — BF16-источник):
    desc = dict(synthetic_spec.desc)
    desc["dtype_orig"] = "bf16"
    spec = RealModelSpec(synthetic_spec.desc_path, synthetic_spec.weights_path,
                         desc, root=synthetic_spec.root,
                         tokenizer_path=synthetic_spec.tokenizer_path)
    prov2 = real_provenance(spec, "lib.so", False, None)
    assert prov2["dtype_orig"] == "bf16"
