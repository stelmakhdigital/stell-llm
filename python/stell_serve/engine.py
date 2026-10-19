"""Биндинг движка через C ABI (ADR-11 §1).

Служебный слой поверх ctypes/dlopen: все запросы/события/метрики — JSON-строки
через границу ABI; строки, возвращённые движком (char**), выделяются malloc и
освобождаются здесь же (уточнение ADR-11 round 1). Контракт менять нельзя.

Два пути (round 2, вклейка реального движка):

- **Реальный движок** `libstell_core.so` (поток core, SHARED-таргет `stell_core`):
  путь из STELL_ENGINE_LIB, либо build/cpu/src/engine/libstell_core.so. Отличается
  от mock тем, что экспортирует `stell_step` (однопоточный CPU-контур: serve сам
  ведёт шаги декода, фоновой нити в ABI раунда 1 нет — ADR-11 «Уточнение») и НЕ
  экспортирует `stell_free`; классификация kind="real" по наличию `stell_step`.
  События token несут только `token_id` (engine не владеет детокенизатором,
  scheduler.hpp) — serve подставляет текст детокенизатором (см. ниже);
  request кодируется в формат engine: {"sampling":…, "prompt_tokens":[…],
  "eos_token":…} (ADR-11 §2).
- **CPU-fallback mock** `libstell_mock_core.so` (src/serve/mock): события несут
  готовый текст; шаги ведёт сам движок. kind="mock".

Расширения ABI round 2 (планировщик T-20261017200354, Q1/Q2; ADR-11 «поля
только добавляются» — новые символы ABI-совместимы):

- `stell_logits(e, request_json, out_json)` — логиты финального слоя
  (до softmax) по готовым prompt_tokens; request
  `{"prompt_tokens":[…], "positions":"last"|"all"}`, ответ
  `{"logits":[[…]],"n_positions":P,"n_vocab":V}`, возврат 0/-1
  (ошибка — `{"error":…}` в out_json); строки malloc/free — конвенция
  ADR-11 §1. Символ добавляет поток core ([R1]); до его merge биндинг
  честно сообщает о недоступности (`supports_logits == False`), а
  quality-gates runner использует fallback-источник логитов.
- `stell_cancel` — конвенция ADR-11 §1 выровнена на **0 = успех,
  -1 = сессия не найдена/уже закрыта** (планировщик Q2; mock уже 0/-1).
  Реальный движок до merge core ([R2]) возвращает 1/0
  (scheduler.cancel → bool): биндинг распознаёт legacy-сборку по
  отсутствию `stell_logits` (оба изменения входят в один merge core) и
  применяет старую интерпретацию; после merge — только 0/-1.

Пока core не вклеил свой SHARED-таргет, реальный путь покрыт локальной
ABI-совместимой сборкой src/engine из src/serve/CMakeLists.txt
(`stell_core_serve_check` → build/cpu/src/serve/libstell_core_serve.so);
эта сборка экспортирует ровно ABI из src/engine/include/stell/engine/capi.h.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "EngineBackend", "CtypesEngineBackend", "EngineError",
    "default_engine_lib", "resolve_engine_lib",
    "byte_tokenizer", "decimal_detokenizer",
]

# Значения "positions" в stell_logits (планировщик Q1): логиты последней
# позиции или всех позиций промпта (teacher-forcing для PPL одним вызовом).
_LOGITS_POSITIONS = ("last", "all")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Пути реального движка (round 2): таргет SHARED `stell_core` потока core
# появится в build/cpu/src/engine/; локальная проверка ABI — build/cpu/src/serve/.
_REAL_CANDIDATES = (
    _REPO_ROOT / "build" / "cpu" / "src" / "engine" / "libstell_core.so",
    _REPO_ROOT / "build" / "cpu" / "src" / "serve" / "libstell_core_serve.so",
)
_DEFAULT_MOCK = _REPO_ROOT / "build" / "cpu" / "src" / "serve" / "libstell_mock_core.so"


class EngineError(Exception):
    """Ошибка постановки запроса/событий со стороны движка (ADR-11)."""


def resolve_engine_lib(explicit: Optional[str] = None) -> tuple:
    """Разрешение пути к .so: явный путь → STELL_ENGINE_LIB → реальный движок
    (build/cpu/src/engine/libstell_core.so → локальная ABI-проверка) → mock.

    Возвращает (path, kind), kind ∈ {"real", "mock"}; путь mock возвращается
    даже если его нет (биндинг сам выдаст внятную EngineError при открытии).
    """
    path: Optional[Path] = None
    if explicit:
        path = Path(explicit)
    else:
        env = os.environ.get("STELL_ENGINE_LIB")
        if env:
            path = Path(env)
    if path is not None:
        return path, "mock" if path.name.startswith("libstell_mock") else "real"
    for cand in _REAL_CANDIDATES:
        if cand.exists():
            return cand, "real"
    return _DEFAULT_MOCK, "mock"


def default_engine_lib() -> Path:
    """Путь к .so (совместимость round 1): реальный движок, если собран,
    иначе mock. См. resolve_engine_lib()."""
    return resolve_engine_lib()[0]


def byte_tokenizer(text: str) -> List[int]:
    """Токенизатор-заглушка serve (ADR-11 §2: serve передаёт готовые токены):
    UTF-8-байты строки. До появления токенизатора реестра моделей (ADR-09);
    контракт с engine (`prompt_tokens`/`eos_token`) не меняется."""
    return list(text.encode("utf-8"))


def decimal_detokenizer(token_id: int) -> str:
    """Детокенизатор-заглушка для реального движка (engine не владеет
    детокенизатором — scheduler.hpp): demo-раннер эмитит ординальные id
    0,1,2,…, рендерим их десятичными цифрами. Контракт событий (token_id)
    не меняется; подмена на детокенизатор модели — точка расширения."""
    return str(token_id)


class EngineBackend:
    """Интерфейс бэкенда движка (ADR-11 §1..§4). Все payload — dict/JSON."""

    kind = "abstract"

    def create(self, cfg: Dict[str, Any]) -> None:
        raise NotImplementedError

    def submit(self, request: Dict[str, Any]) -> int:
        """Возвращает session_id (>0); EngineError при 0 (ADR-11)."""
        raise NotImplementedError

    def poll(self, session_id: int) -> List[Dict[str, Any]]:
        """Список событий [{"seq":..,"type":"token|tool_call|finish|error"}];
        пустой список = событий нет."""
        raise NotImplementedError

    def cancel(self, session_id: int) -> None:
        """Отмена сессии (ADR-11 §3): 0 = успех, -1 = не найдена/закрыта
        (конвенция ADR-11 §1, планировщик Q2); EngineError при -1."""
        raise NotImplementedError

    def logits(self, prompt_tokens: List[int],
               positions: str = "last") -> Dict[str, Any]:
        """Логиты финального слоя (до softmax) по готовым токенам —
        stell_logits (расширение ABI round 2, планировщик Q1). Только
        реальный движок после merge core [R1]."""
        raise NotImplementedError

    @property
    def supports_logits(self) -> bool:
        """True, если .so экспортирует stell_logits ([R1], merge core)."""
        return False

    def step(self) -> None:
        """Один шаг декода батча (только реальный движок: stell_step, ABI-
        совместимое расширение round 1; mock шаги ведёт сам — no-op)."""
        raise NotImplementedError

    @property
    def supports_step(self) -> bool:
        return False

    def metrics(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class CtypesEngineBackend(EngineBackend):
    """Реализация поверх libstell_core.so / libstell_mock_core.so (ADR-11 §1)."""

    def __init__(self, lib_path: Optional[Path] = None,
                 detokenize: Optional[Callable[[int], str]] = None,
                 tokenize: Optional[Callable[[str], List[int]]] = None) -> None:
        if lib_path is None:
            path, kind = resolve_engine_lib()
        else:
            path = Path(lib_path)
            kind = "mock" if path.name.startswith("libstell_mock") else "real"
        if not path.exists():
            raise EngineError(
                f"engine library not found: {path}. "
                "Собери движок: cmake -S . -B build/cpu -DSTELL_CUDA=OFF && "
                "cmake --build build/cpu --target stell_core / stell_mock_core"
            )
        self._lib_path = path
        self.kind = kind
        self._detokenize = detokenize or decimal_detokenizer
        self._tokenize = tokenize or byte_tokenizer
        self.lib = ctypes.CDLL(str(path))
        self._handle = None
        self._free = None
        self._setup_signatures()

    # ------------------------------ ABI binding ------------------------------

    def _setup_signatures(self) -> None:
        lib = self.lib
        lib.stell_engine_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.stell_engine_create.restype = ctypes.c_int
        lib.stell_engine_destroy.argtypes = [ctypes.c_void_p]
        lib.stell_engine_destroy.restype = None
        lib.stell_submit.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
        lib.stell_submit.restype = ctypes.c_uint64
        lib.stell_poll.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
        lib.stell_poll.restype = ctypes.c_int
        lib.stell_cancel.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.stell_cancel.restype = ctypes.c_int
        lib.stell_metrics.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
        lib.stell_metrics.restype = ctypes.c_int
        # stell_step — ABI-совместимое расширение round 1 (ADR-11 «Уточнение»):
        # есть у реального движка, нет у mock (getattr — совместимая проверка).
        try:
            step_fn = lib.stell_step
            step_fn.argtypes = [ctypes.c_void_p]
            step_fn.restype = None
            self._step = step_fn
        except AttributeError:
            self._step = None
        # stell_logits — ABI-совместимое расширение round 2 (планировщик Q1;
        # [R1], символ добавляет поток core): getattr — та же совместимая
        # проверка, что у stell_step. Доступность классифицируется строго по
        # наличию символа в .so (никаких догадок по имени файла).
        try:
            logits_fn = lib.stell_logits
            logits_fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                  ctypes.POINTER(ctypes.c_char_p)]
            logits_fn.restype = ctypes.c_int
            self._logits = logits_fn
        except AttributeError:
            self._logits = None
        # освобождение строк, возвращённых движком (уточнение ADR-11 round 1:
        # malloc-строки, освобождает вызывающий): расширение stell_free (mock),
        # иначе free() из libc того же процесса (in-process .so — тот же heap).
        try:
            free_fn = lib.stell_free
            free_fn.argtypes = [ctypes.c_void_p]
            free_fn.restype = None
            self._free = free_fn
        except AttributeError:
            libc = ctypes.CDLL(None)
            libc.free.argtypes = [ctypes.c_void_p]
            libc.free.restype = None
            self._free = libc.free

    def _release_ptr(self, ptr) -> None:
        self._free(ctypes.c_void_p(ctypes.cast(ptr, ctypes.c_void_p).value or 0))

    # ------------------------------ lifecycle ------------------------------

    def create(self, cfg: Dict[str, Any]) -> None:
        raw = json.dumps(cfg).encode()
        handle = ctypes.c_void_p()
        rc = self.lib.stell_engine_create(raw, ctypes.byref(handle))
        if rc != 0:
            raise EngineError(f"stell_engine_create failed ({rc})")
        self._handle = handle

    def _require_handle(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise EngineError("engine not created (call create() first)")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self.lib.stell_engine_destroy(self._handle)
            self._handle = None

    # ------------------------------ request/events ------------------------------

    def _encode_request(self, request: Dict[str, Any]) -> bytes:
        """Кодирование запроса под kind (ADR-11 §2): serve применяет
        chat-шаблон и передаёт готовые токены; engine не знает диалектов."""
        if self.kind == "mock":
            return json.dumps(request).encode()
        # реальный движок: {"sampling":…, "prompt_tokens":[…], "eos_token":…}
        sampling = request.get("session") or request.get("sampling") or {}
        prompt = request.get("prompt")
        if not prompt:
            msgs = request.get("messages") or []
            prompt = "".join(str(m.get("content") or "") for m in msgs)
        prompt_tokens = self._tokenize(prompt) if prompt else []
        return json.dumps({
            "sampling": {
                "max_tokens": sampling.get("max_tokens", 256),
                "temperature": sampling.get("temperature", 1.0),
                "top_p": sampling.get("top_p", 0.95),
                "top_k": sampling.get("top_k", 20),
                "seed": sampling.get("seed", 0),
            },
            "prompt_tokens": prompt_tokens,
            # eos реального двигателя — точка интеграции реестра моделей
            # (ADR-09); demo-раннер эмитит ординальные id — eos отключён (-1).
            "eos_token": -1,
        }).encode()

    def submit(self, request: Dict[str, Any]) -> int:
        raw = self._encode_request(request)
        msg = ctypes.c_char_p()
        sid = self.lib.stell_submit(self._require_handle(), raw, ctypes.byref(msg))
        text = ctypes.string_at(msg).decode() if msg else ""
        self._release_ptr(msg)
        if sid == 0:
            raise EngineError(text or "submit failed")
        return int(sid)

    def poll(self, session_id: int) -> List[Dict[str, Any]]:
        out = ctypes.c_char_p()
        n = self.lib.stell_poll(self._require_handle(), ctypes.c_uint64(session_id),
                                ctypes.byref(out))
        if n < 0:
            return []  # сессия закрыта: событий больше нет (ADR-11 §3)
        events: List[Dict[str, Any]] = []
        if out:
            raw = ctypes.string_at(out)
            self._release_ptr(out)
            events = list(json.loads(raw.decode()))
        if self.kind == "real":
            # engine не владеет детокенизатором: token события несут только
            # token_id — serve подставляет текст (ADR-11 §3; подмена
            # детокенизатора модели — точка расширения, контракт не меняется).
            for ev in events:
                if ev.get("type") == "token" and not ev.get("text"):
                    # core-2: движок несёт text:"" при runner без прод-токенизатора —
                    # пустой текст тоже замещаем детокенизатором serve.
                    ev["text"] = self._detokenize(int(ev.get("token_id", -1)))
        return events

    def step(self) -> None:
        """Один шаг декода батча (stell_step, однопоточный CPU-контур; ADR-11
        «Уточнение»: старт-стоп фоновой нити в ABI round 1 не вводится)."""
        if self._step is None:
            return  # mock: шаги ведёт сам движок; no-op совместим
        self._step(self._require_handle())

    @property
    def supports_step(self) -> bool:
        return self._step is not None

    def cancel(self, session_id: int) -> None:
        rc = self.lib.stell_cancel(self._require_handle(), ctypes.c_uint64(session_id))

        # Конвенция ADR-11 §1 (планировщик Q2, round 2): 0 = успех,
        # -1 = сессия не найдена/уже закрыта. Mock — уже 0/-1. Реальный
        # движок до merge core ([R2]) возвращает legacy 1/0
        # (scheduler.cancel → bool); legacy-сборка распознаётся по
        # отсутствию stell_logits (оба изменения — один merge core):
        # там 1 = успех, 0 = неизвестная сессия. После merge core путь
        # единый: 0 ок, всё остальное — ошибка.
        if self.kind == "real" and self._logits is None:
            ok = rc != 0              # legacy 1/0 (pre-merge ABI-копия/движок)
        else:
            ok = rc == 0              # ADR-11: 0 = успех, -1 = ошибка
        if not ok:
            raise EngineError(f"cancel failed for session {session_id} (rc={rc})")

    @property
    def supports_logits(self) -> bool:
        return self._logits is not None

    def logits(self, prompt_tokens: List[int],
               positions: str = "last") -> Dict[str, Any]:
        """Логиты финального слоя (до softmax) по готовым токенам —
        stell_logits (планировщик Q1, [R1]).

        request: {"prompt_tokens": […], "positions": "last"|"all"}
        ответ:   {"logits": [[f32,…],…], "n_positions": P, "n_vocab": V}
        возврат: 0/-1; при ошибке out_json = {"error": …}; строки malloc —
        free() вызывающим (конвенция ADR-11 §1).

        EngineError, если символ не экспортирован (до merge core [R1]) —
        quality-гейты в этом случае используют fallback-источник логитов
        (см. stell_bench.quality_gates_runner)."""
        if self._logits is None:
            raise EngineError(
                "stell_logits недоступен: .so не экспортирует символ "
                "(ABI-расширение round 2, включится после merge core [R1])")
        if positions not in _LOGITS_POSITIONS:
            raise EngineError(f"positions must be one of {_LOGITS_POSITIONS}")
        raw = json.dumps({"prompt_tokens": [int(t) for t in prompt_tokens],
                          "positions": positions}).encode()
        out = ctypes.c_char_p()
        rc = self._logits(self._require_handle(), raw, ctypes.byref(out))
        payload: Dict[str, Any] = {}
        if out:
            payload = json.loads(ctypes.string_at(out).decode())
            self._release_ptr(out)
        if rc != 0 or "error" in payload:
            raise EngineError(payload.get("error", f"stell_logits failed (rc={rc})"))
        return payload

    def metrics(self) -> Dict[str, Any]:
        out = ctypes.c_char_p()
        rc = self.lib.stell_metrics(self._require_handle(), ctypes.byref(out))
        if rc != 0 or not out:
            raise EngineError("metrics failed")
        raw = ctypes.string_at(out)
        self._release_ptr(out)
        return json.loads(raw.decode())
