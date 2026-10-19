"""Песочница кода ([R4] round 3): раннер ловит таймаут/exit-code/падение
кода и НЕ ВЕШАЕТ прогон; изоляция — пустой env, изолированный temp-dir,
rlimits. Честная пометка модуля: process-sandbox, не security-граница
(докстринг code_sandbox.py).
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest

from stell_bench.code_sandbox import (  # noqa: E402
    CPU_LIMIT_S, MEM_LIMIT_BYTES, NOFILE_LIMIT, WALL_TIMEOUT_S, run_code,
)
from stell_bench.quality_gates_runner import extract_code  # noqa: E402

try:
    from .conftest import ROOT as _ROOT  # noqa: F401
except ImportError:  # pragma: no cover
    from pathlib import Path as _P

    _ROOT = _P(__file__).resolve().parents[2]


# ------------------------- классификация исходов -----------------------------

def test_correct_answer_passes():
    res = run_code("a = 2 + 2\nassert a == 4\n")
    assert res.outcome == "ok"
    assert res.exit_code == 0


def test_exception_is_caught_not_hanging():
    res = run_code("raise ValueError('бум')\n")
    assert res.outcome == "error"
    assert res.exit_code == 1                       # неперехваченное исключение
    assert "ValueError" in res.stderr_tail          # traceback доступен
    assert res.elapsed_s < WALL_TIMEOUT_S           # без ожидания таймаута


def test_sys_exit_code_is_reported():
    res = run_code("import sys; sys.exit(3)\n")
    assert res.outcome == "error"
    assert res.exit_code == 3
    assert res.elapsed_s < WALL_TIMEOUT_S


def test_infinite_loop_does_not_hang_runner():
    """Зацикленный код: раннер возвращается за wall+запас, кейс не "ok"."""
    started = time.monotonic()
    res = run_code("while True:\n    pass\n")       # жрёт CPU → RLIMIT_CPU
    elapsed = time.monotonic() - started
    assert res.outcome in ("timeout", "error")      # убит (CPU-лимит/стена)
    assert res.outcome != "ok"
    # не вешает прогон: запас — wall-таймаут + 5с на kill/reap
    assert elapsed < WALL_TIMEOUT_S + 5.0


def test_sleeping_loop_hits_wall_timeout():
    """Висим без CPU (sleep) → wall-таймаут, процесс убит."""
    started = time.monotonic()
    res = run_code("import time\nwhile True:\n    time.sleep(0.1)\n")
    elapsed = time.monotonic() - started
    assert res.outcome == "timeout"
    assert res.exit_code is None
    assert elapsed >= WALL_TIMEOUT_S * 0.9
    assert elapsed < WALL_TIMEOUT_S + 5.0


def test_cpu_limit_documented_value():
    """Протокол [R4] зафиксирован: CPU=5с, wall=10с (значения модуля)."""
    assert CPU_LIMIT_S == 5
    assert WALL_TIMEOUT_S == 10.0
    assert MEM_LIMIT_BYTES == 512 * 1024 * 1024
    assert NOFILE_LIMIT == 64


# ----------------------------- изоляция --------------------------------------

def test_parent_env_does_not_leak(monkeypatch):
    """Пустой env: переменные родителя НЕ видны в ребёнке."""
    monkeypatch.setenv("STELL_SANDBOX_CANARY", "must-not-leak")
    res = run_code(
        "import os, sys\n"
        "sys.exit(0 if 'STELL_SANDBOX_CANARY' not in os.environ else 9)\n")
    assert res.outcome == "ok", res.stderr_tail


def test_isolated_temp_dir_as_cwd():
    """cwd ребёнка — свежий temp-dir вне репозитория и тестов."""
    res = run_code(
        "import os, sys, tempfile\n"
        "cwd = os.getcwd()\n"
        "sys.exit(0 if cwd.startswith(tempfile.gettempdir()) else 8)\n")
    assert res.outcome == "ok", res.stderr_tail


def test_rlimit_as_enforced():
    """RLIMIT_AS: запрос памяти сверх лимита падает (не OOM-холост).
    512 МиБ лимита против 2 ГиБ запроса — аллокатор должен отказать."""
    res = run_code("x = bytearray(2 * 1024 * 1024 * 1024)\n")
    assert res.outcome == "error"
    assert res.exit_code != 0
    assert res.elapsed_s < WALL_TIMEOUT_S


def test_no_file_limit_is_small():
    """RLIMIT_NOFILE=64: открыть 200 дескрипторов подряд нельзя."""
    res = run_code(
        "files = []\n"
        "for i in range(200):\n"
        "    files.append(open(f'/tmp/probe-{i}', 'w'))\n")
    assert res.outcome == "error"


def test_result_metadata_records_limits():
    res = run_code("pass")
    assert res.metadata["cpu_limit_s"] == CPU_LIMIT_S
    assert res.metadata["wall_timeout_s"] == WALL_TIMEOUT_S
    assert res.metadata["env"] == "{}"


def test_temp_dir_cleaned_after_run():
    """Изолированный temp-dir удаляется после кейса (мусора не остаётся)."""
    before = set(os.listdir(tempfile.gettempdir()))
    run_code("pass")
    after = set(os.listdir(tempfile.gettempdir()))
    # не оставляет НИ ОДНОГО нового каталога stell-sandbox-*
    assert not any(d.startswith("stell-sandbox-") for d in after - before)


# --------------------------- извлечение кода ---------------------------------

def test_extract_code_markdown_fence_first_block():
    answer = "Пролог.\n```python\ndef f():\n    return 1\n```\nХвост."
    assert extract_code(answer) == "def f():\n    return 1\n"


def test_extract_code_fence_without_language_and_crlf():
    answer = "```\r\nprint('hi')\r\n```"
    assert extract_code(answer) == "print('hi')\r\n"


def test_extract_code_without_fence_whole_answer():
    assert extract_code("def g():\n    return 2") == "def g():\n    return 2"


def test_extract_code_empty_answer_is_none():
    assert extract_code("   \n  ") is None


# ------------- интеграция с HumanEval-контрактом (мини-сценарий) -------------

def test_sandbox_evaluates_humaneval_style_case():
    """Сквозной мини-сценарий: candidate + test + check() — как в гейте."""
    program = (
        "def add(a, b):\n"
        "    return a + b\n")
    test = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    res = run_code(program + "\n\n\n" + test + "\ncheck(add)\n")
    assert res.outcome == "ok"


def test_sandbox_rejects_wrong_humaneval_candidate():
    program = (
        "def add(a, b):\n"
        "    return a - b\n")                     # неверная реализация
    test = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    res = run_code(program + "\n\n\n" + test + "\ncheck(add)\n")
    assert res.outcome == "error"
    assert "AssertionError" in res.stderr_tail


def test_missing_candidate_is_error_not_crash():
    """Пустой completion → NameError в check, гейт считает кейс неверным."""
    test = "def check(candidate):\n    assert candidate(1) == 1\n"
    res = run_code(test + "\ncheck(missing_fn)\n")
    assert res.outcome == "error"


def test_module_docstring_marks_not_security_boundary():
    """Честная пометка [Q4]: process-sandbox, НЕ security-граница —
    зафиксирована в докстринге модуля (не молчать)."""
    import stell_bench.code_sandbox as cs

    doc = cs.__doc__
    assert "process-sandbox" in doc
    assert "НЕ security-" in doc
