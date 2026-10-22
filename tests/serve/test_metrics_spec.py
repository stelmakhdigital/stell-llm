"""Контракт /metrics по spec-полям (round 9, ADR-11 §4).

Состав полей зафиксирован планом round 9 (именование от core-9):
  features.spec_draft (bool), latency.tok_s.spec_decode (float),
  spec.accept_rate (float), spec.n_draft (uint).

Два направления:
- снапшот БЕЗ spec-полей (текущий реальный движок до merge core-9) →
  честная пометка spec_note в ответе (прецедент dtype_orig round 6:
  отсутствие поля ≠ молчаливый None);
- снапшот С spec-полями → passthrough 1:1 без искажений (serve ничего
  не фабрикует и не подменяет).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

try:
    from .conftest import _toolchain_env  # noqa: F401  (путь/тулчейн как у всех)
except ImportError:  # pragma: no cover — запуск вне пакета тестов
    from tests.serve.conftest import _toolchain_env  # noqa: F401

# ---------------- stub .so с расширенным spec-снапшотом ---------------------
# Минимальный ABI ADR-11 (serve-контур submit/poll не используется — только
# /metrics), stell_metrics отдаёт расширенный снапшот СО ВСЕМИ spec-полями
# (mock ABI/fixture: как будет после merge core-9). Именование НЕ
# libstell_mock* → биндинг классифицирует kind="real"; stell_step/stell_logits/
# stell_free у стаба нет — free() вызывающего идёт в libc того же процесса
# (malloc-конвенция ADR-11 §1).
_STUB_SPEC_C = r"""
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
int stell_cancel(stell_engine_t* e, uint64_t sid) { (void)e; (void)sid; return 0; }

int stell_metrics(stell_engine_t* e, char** out_json) {
    (void)e;
    const char* s =
        "{\"engine\":{\"state\":\"serving\",\"sessions_active\":0,"
        "\"slots_busy\":0,\"slots_total\":16,\"batch_size_max\":16},"
        "\"latency\":{\"ttft_ms\":{\"p50\":1.0,\"p95\":1.3,\"p99\":1.6},"
        "\"tpot_ms\":{\"p50\":14.0,\"p95\":18.2,\"p99\":22.4},"
        "\"tok_s\":{\"decode\":71.4,\"prefill\":900.0,"
        "\"spec_decode\":123.456}},"
        "\"features\":{\"spec_draft\":true,\"kv_quant\":\"F16\"},"
        "\"spec\":{\"accept_rate\":0.72,\"n_draft\":8},"
        "\"build\":{\"version\":\"stub-spec\"},\"model\":\"stub-model\"}";
    *out_json = (char*)malloc(strlen(s) + 1);
    strcpy(*out_json, s);
    return 0;
}
"""


@pytest.fixture(scope="module")
def stub_spec_lib(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stub_spec_engine")
    src = out_dir / "stub_spec_engine.c"
    so = out_dir / "libstell_stub_spec.so"
    src.write_text(_STUB_SPEC_C)
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
def spec_client(stub_spec_lib):
    from fastapi.testclient import TestClient
    from stell_serve.app import create_app, default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(stub_spec_lib)
    application = create_app(backend, engine_cfg=default_engine_cfg("stub-model"))
    yield TestClient(application)
    backend.close()


# ---------------- stub .so БЕЗ spec-полей (сценарий «движок старого ABI») ---
# Реальный движок после core round 8 поля отдаёт; сценарий «полей нет»
# остаётся валидным (старый .so / другой ABI) — покрыт отдельным стабом.
_STUB_PLAIN_C = r"""
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
int stell_cancel(stell_engine_t* e, uint64_t sid) { (void)e; (void)sid; return 0; }
int stell_metrics(stell_engine_t* e, char** out_json) {
    (void)e;
    const char* s =
        "{\"engine\":{\"state\":\"serving\",\"sessions_active\":0,"
        "\"slots_busy\":0,\"slots_total\":16,\"batch_size_max\":16},"
        "\"logits\":{\"requests_total\":0,\"errors_total\":0,"
        "\"last_n_vocab\":null,\"last_positions\":null}}";
    *out_json = (char*)malloc(strlen(s) + 1);
    strcpy(*out_json, s);
    return 0;
}
"""


@pytest.fixture(scope="module")
def stub_plain_lib(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stub_plain_engine")
    src = out_dir / "stub_plain_engine.c"
    so = out_dir / "libstell_stub_plain.so"
    src.write_text(_STUB_PLAIN_C)
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
def plain_client(stub_plain_lib):
    from fastapi.testclient import TestClient
    from stell_serve.app import create_app, default_engine_cfg
    from stell_serve.engine import CtypesEngineBackend

    backend = CtypesEngineBackend(stub_plain_lib)
    application = create_app(backend, engine_cfg=default_engine_cfg("stub-model"))
    yield TestClient(application)
    backend.close()


# --------------------------- (а) снапшот БЕЗ полей ---------------------------

def test_metrics_without_spec_fields_honest_note(real_client):
    """Реальный движок после merge core-9 (round 8, срез 4) публикует spec-поля
    (features.spec_draft, latency.tok_s.spec_decode, spec.*) — serve делает
    passthrough 1:1 и НЕ ставит spec_note (пометка — только для честного
    «у движка полей нет»; сценарий «полей нет» остаётся покрытым логикой
    spec_note на стабах, движок ABI больше не даёт такой снапшот)."""
    snap = real_client.get("/metrics").json()
    assert "spec_note" not in snap, (
        "при наличии spec-полей у движка пометка не нужна")
    assert snap["features"]["spec_draft"] in (True, False)
    assert "spec_decode" in snap["latency"]["tok_s"]
    assert "accept_rate" in snap["spec"] and "n_draft" in snap["spec"]


def test_metrics_without_spec_fields_honest_note_stub(plain_client):
    """Сценарий «у движка spec-полей нет» (старый ABI/старый .so): служит
    честной пометке spec_note — НЕ молчаливое отсутствие (прецедент
    dtype_orig round 6); при этом serve НЕ фабрикует пустые spec-поля."""
    snap = plain_client.get("/metrics").json()
    assert "spec_note" in snap, "молчаливое отсутствие spec-полей запрещено"
    note = snap["spec_note"]
    assert "spec-полей" in note
    for name in ("features.spec_draft", "latency.tok_s.spec_decode", "spec"):
        assert name in note
    assert "spec" not in snap
    assert not isinstance(snap.get("features"), dict) or \
        "spec_draft" not in snap["features"]


def test_metrics_without_spec_fields_serve_section_intact(real_client):
    """Пометка не вытесняет остальные секции /metrics (serve-метрика на месте)."""
    snap = real_client.get("/metrics").json()
    assert snap["serve"]["requests_total"] == 0
    assert "engine" in snap and "logits" in snap


def test_metrics_partial_spec_fields_no_note(client):
    """Частичный состав (mock: features.spec_draft=false есть, spec-блока нет)
    — уже НЕ «поле отсутствует»: пометка не ставится, значение false проходит
    как честное «фича выключена» (0/false — значения, а не отсутствие)."""
    snap = client.get("/metrics").json()
    assert "spec_note" not in snap
    assert snap["features"]["spec_draft"] is False


# ------------------------- (б) снапшот С полями -------------------------

def test_metrics_spec_fields_passthrough_untouched(spec_client):
    """Снапшот со всеми spec-полями (mock ABI stub как от core-9) доходит до
    /metrics БЕЗ искажений: имена/типы/значения 1:1, пометка не ставится."""
    snap = spec_client.get("/metrics").json()
    assert "spec_note" not in snap
    # features.spec_draft — bool без подмены
    assert snap["features"]["spec_draft"] is True
    # latency.tok_s.spec_decode — float без искажений (округления нет)
    assert snap["latency"]["tok_s"]["spec_decode"] == 123.456
    assert snap["latency"]["tok_s"]["decode"] == 71.4
    # spec-блок арбитра: accept_rate/n_draft 1:1
    assert snap["spec"] == {"accept_rate": 0.72, "n_draft": 8}
    # rest снапшота тоже не перекошен
    assert snap["engine"]["slots_total"] == 16
    assert snap["build"]["version"] == "stub-spec"


def test_metrics_spec_fields_reach_serve_latency_section(spec_client):
    """bench-контракт (src/bench/stell_bench/harness.py): пути
    snap['features']['spec_draft'] и snap['latency']['tok_s']['spec_decode']
    читаются из того же /metrics — проверяем чтение ровно по этим путям."""
    snap = spec_client.get("/metrics").json()
    enabled = snap.get("features", {}).get("spec_draft")
    spec_tok_s = snap.get("latency", {}).get("tok_s", {}).get("spec_decode", 0.0)
    assert enabled is True
    assert spec_tok_s == 123.456
