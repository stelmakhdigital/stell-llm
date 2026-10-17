"""Фикстуры для контрактных тестов serve.

- сборка mock-движка (C ABI по ADR-11) через cmake в build/cpu;
- свежий FastAPI-приложение на каждый тест (TestClient).

Путь к .so можно переопределить переменной STELL_MOCK_ENGINE_LIB.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

for p in (ROOT / "python", ROOT / "src" / "calibrate", ROOT / "src" / "bench"):
    sys.path.insert(0, str(p))

MOCK_SO = ROOT / "build" / "cpu" / "src" / "serve" / "libstell_mock_core.so"
# ABI-совместимая сборка src/engine (поток A) для проверки РЕАЛЬНОГО движка:
# таргет stell_core_serve_check из src/serve/CMakeLists.txt (round 2). Когда
# поток core вклеит свой SHARED `stell_core` в build/cpu/src/engine/, биндинг
# prefers его (engine.resolve_engine_lib); эта сборка остаётся для локальных
# тестов ABI (включая stell_step).
REAL_SO = ROOT / "build" / "cpu" / "src" / "serve" / "libstell_core_serve.so"


def _toolchain_env() -> dict:
    env = dict(os.environ)
    stell_bin = Path.home() / ".local" / "stell-conda" / "envs" / "stell" / "bin"
    if stell_bin.exists():
        env["PATH"] = str(stell_bin) + os.pathsep + env.get("PATH", "")
    env.setdefault("CC", "x86_64-conda-linux-gnu-gcc")
    env.setdefault("CXX", "x86_64-conda-linux-gnu-g++")
    return env


@pytest.fixture(scope="session")
def mock_engine_lib() -> Path:
    """Собирает CPU-fallback mock-движок (ADR-11 ABI) при необходимости."""
    custom = os.environ.get("STELL_MOCK_ENGINE_LIB")
    if custom:
        return Path(custom)
    if not MOCK_SO.exists():
        if not shutil.which("cmake", path=_toolchain_env()["PATH"]):
            pytest.skip("cmake недоступен для сборки mock-движка")
        subprocess.run(
            ["cmake", "-S", str(ROOT), "-B", str(ROOT / "build" / "cpu"),
             "-G", "Ninja", "-DSTELL_CUDA=OFF"],
            cwd=ROOT, env=_toolchain_env(), check=True,
            capture_output=True)
    # всегда доводим сборку до актуального состояния (ninja инкрементальна)
    subprocess.run(
        ["cmake", "--build", str(ROOT / "build" / "cpu"),
         "--target", "stell_mock_core"],
        cwd=ROOT, env=_toolchain_env(), check=True, capture_output=True)
    return MOCK_SO


@pytest.fixture(scope="session")
def real_engine_lib(mock_engine_lib) -> Path:
    """Реальный движок (src/engine, C ABI по ADR-11, включая stell_step).

    Честный SKIP, если ABI-копия src/engine не собралась (нет тулчейна);
    при наличии сборки отсутствие .so — ошибка (нельзя молча подменить путь).
    """
    if REAL_SO.exists():
        return REAL_SO
    if not shutil.which("cmake", path=_toolchain_env()["PATH"]):
        pytest.skip("cmake недоступен: ABI-копия реального движка не собрать")
    subprocess.run(
        ["cmake", "--build", str(ROOT / "build" / "cpu"),
         "--target", "stell_core_serve_check"],
        cwd=ROOT, env=_toolchain_env(), check=True, capture_output=True)
    if not REAL_SO.exists():
        pytest.skip("stell_core_serve_check не собрался: ABI реального движка недоступен")
    return REAL_SO


@pytest.fixture()
def real_app(real_engine_lib):
    """Приложение поверх РЕАЛЬНОГО движка (src/engine, demo-раннер)."""
    from stell_serve.app import create_app, default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(real_engine_lib)
    application = create_app(backend, engine_cfg=default_engine_cfg("engine-demo"))
    application.state._test_lib = real_engine_lib
    yield application
    backend.close()


@pytest.fixture()
def real_client(real_app):
    from fastapi.testclient import TestClient

    return TestClient(real_app)


@pytest.fixture()
def app(mock_engine_lib):
    from stell_serve.app import create_app, default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(mock_engine_lib)
    application = create_app(backend, engine_cfg=default_engine_cfg("mock-model"))
    application.state._test_lib = mock_engine_lib
    yield application
    backend.close()


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    return TestClient(app)
