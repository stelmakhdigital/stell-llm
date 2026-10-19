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
# Интеграционный фикс (worker, раунд 2): предпочитать РЕАЛЬНЫЙ libstell_core.so
# (SHARED-таргет потока core в build/cpu/src/engine/), ABI-копия — fallback.
REAL_SO_PREF = ROOT / "build" / "cpu" / "src" / "engine" / "libstell_core.so"
REAL_SO = (REAL_SO_PREF if REAL_SO_PREF.exists()
           else ROOT / "build" / "cpu" / "src" / "serve" / "libstell_core_serve.so")


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


# ---------------------------------------------------------------------------
# Stub-.so c минимальным ABI + stell_logits ([R1-интеграция], round 2):
# тесты чистого биндинга логитов (tests/serve/test_logits_binding.py) и
# ABI-путь живого runner quality-gates (test_quality_gates_live.py).
# Именование НЕ начинается с "libstell_mock" → биндинг классифицирует
# kind="real"; stell_step/stell_free у стаба нет — free() вызывающего
# идёт в libc того же процесса (malloc-конвенция ADR-11 §1).
# ---------------------------------------------------------------------------
# --------------------------- stub .so (C-исходник) ---------------------------
# Минимальный ABI ADR-11 + stell_logits. n_vocab=4; значение элемента
# logits[i][j] = n_tokens*10 + j + i (проверка pass-through prompt_tokens и
# режима positions). Именование НЕ начинается с "libstell_mock" → биндинг
# классифицирует kind="real"; stell_step/stell_free у стаба нет — free()
# вызывающего идёт в libc того же процесса (та же malloc-конвенция).
_STUB_C = r"""
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct stell_engine stell_engine_t;

int stell_engine_create(const char* cfg, stell_engine_t** out) {
    (void)cfg; if (!out) return -1; *out = (stell_engine_t*)1; return 0;
}
void stell_engine_destroy(stell_engine_t* e) { (void)e; }
uint64_t stell_submit(stell_engine_t* e, const char* r, char** msg) {
    (void)e; (void)r; if (msg) *msg = NULL; return 1;
}
int stell_poll(stell_engine_t* e, uint64_t sid, char** out) {
    (void)e; (void)sid; if (out) *out = NULL; return 0;
}
/* ADR-11 round 2 (планировщик Q2): 0 = успех, -1 = не найдена/закрыта. */
int stell_cancel(stell_engine_t* e, uint64_t sid) { (void)e; (void)sid; return 0; }
int stell_metrics(stell_engine_t* e, char** out) {
    (void)e; const char* s = "{\"engine\":{\"state\":\"serving\"}}";
    *out = (char*)malloc(strlen(s) + 1); strcpy(*out, s); return 0;
}

static int count_tokens(const char* req) {
    const char* p = strstr(req, "\"prompt_tokens\"");
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    int n = 0, prev_digit = 0;
    for (p++; *p && *p != ']'; ++p) {
        if (isdigit((unsigned char)*p)) { if (!prev_digit) n++; prev_digit = 1; }
        else prev_digit = 0;
    }
    return n;
}

static int has_positions_all(const char* req) {
    const char* p = strstr(req, "\"positions\"");
    if (!p) return 0;
    for (p += strlen("\"positions\""); *p && *p != '}' && *p != '\n'; ++p)
        if (*p == 'a' && strncmp(p, "all", 3) == 0) return 1;
    return 0;
}

int stell_logits(stell_engine_t* e, const char* req, char** out) {
    (void)e;
    if (!req || !out) return -1;
    const int n = count_tokens(req);
    if (n <= 0) {
        const char* err = "{\"error\":\"empty prompt_tokens\"}";
        *out = (char*)malloc(strlen(err) + 1); strcpy(*out, err);
        return -1;                                   /* 0/-1: конвенция ADR-11 */
    }
    const int rows = has_positions_all(req) ? n : 1;
    const int VOCAB = 4;
    /* {"logits":[[…],…],"n_positions":rows,"n_vocab":4}; malloc/free конвенция */
    char* buf = (char*)malloc(64 + (size_t)rows * VOCAB * 24);
    if (!buf) return -1;
    char* w = buf;
    w += sprintf(w, "{\"logits\":[");
    for (int i = 0; i < rows; ++i) {
        w += sprintf(w, "%s[", i ? "," : "");
        for (int j = 0; j < VOCAB; ++j)
            w += sprintf(w, "%s%d", j ? "," : "", n * 10 + j + i);
        w += sprintf(w, "]");
    }
    sprintf(w, "],\"n_positions\":%d,\"n_vocab\":%d}", rows, VOCAB);
    *out = buf;
    return 0;
}
"""


@pytest.fixture(scope="module")
def stub_logits_lib(tmp_path_factory):
    """Собирает stub-.so с ABI + stell_logits (тест чистого биндинга)."""
    out_dir = tmp_path_factory.mktemp("stub_engine")
    src = out_dir / "stub_engine.c"
    so = out_dir / "libstell_stub_logits.so"
    src.write_text(_STUB_C)
    cc = None
    for cand in ("x86_64-conda-linux-gnu-gcc", "gcc", "cc"):
        cc = shutil.which(cand, path=_toolchain_env()["PATH"])
        if cc:
            break
    assert cc, "не найден C-компилятор для stub-.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O0", "-o", str(so), str(src)],
                   check=True, capture_output=True, env=_toolchain_env())
    return so


@pytest.fixture()
def stub_backend(stub_logits_lib):
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(stub_logits_lib)
    backend.create({})
    yield backend
    backend.close()
