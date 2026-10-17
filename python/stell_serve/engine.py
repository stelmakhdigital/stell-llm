"""Биндинг движка через C ABI (ADR-11 §1).

Служебный слой поверх ctypes/dlopen: все запросы/события/метрики — JSON-строки
через границу ABI; строки, возвращённые движком (char**), выделяются malloc и
освобождаются здесь же. Контракт менять нельзя (ADR-11).

Путь к библиотеке: параметр конструктора, STELL_ENGINE_LIB, либо
build/serve-cpu/src/serve/libstell_mock_core.so (CPU-fallback mock).
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["EngineBackend", "CtypesEngineBackend", "EngineError", "default_engine_lib"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MOCK = _REPO_ROOT / "build" / "cpu" / "src" / "serve" / "libstell_mock_core.so"


class EngineError(Exception):
    """Ошибка постановки запроса/событий со стороны движка (ADR-11)."""


def default_engine_lib() -> Path:
    """Разрешение пути к .so: STELL_ENGINE_LIB → mock в build/serve-cpu."""
    env = os.environ.get("STELL_ENGINE_LIB")
    if env:
        return Path(env)
    if _DEFAULT_MOCK.exists():
        return _DEFAULT_MOCK
    return _DEFAULT_MOCK


class EngineBackend:
    """Интерфейс бэкенда движка (ADR-11 §1..§4). Все payload — dict/JSON."""

    def create(self, cfg: Dict[str, Any]) -> None:
        raise NotImplementedError

    def submit(self, request: Dict[str, Any]) -> int:
        """Возвращает session_id (>0); EngineError при 0 (ADR-11)."""
        raise NotImplementedError

    def poll(self, session_id: int) -> List[Dict[str, Any]]:
        """Список событий [{"seq":..,"type":"token|tool_call|finish|error"}]; пустой
        список = событий нет; сессия закрыта → StopIteration через EngineError."""
        raise NotImplementedError

    def cancel(self, session_id: int) -> None:
        raise NotImplementedError

    def metrics(self) -> Dict[str, Any]:
        raise NotImplementedError


class CtypesEngineBackend(EngineBackend):
    """Реализация поверх libstell_core.so (ADR-11 §1)."""

    def __init__(self, lib_path: Optional[Path] = None) -> None:
        path = Path(lib_path) if lib_path else default_engine_lib()
        if not path.exists():
            raise EngineError(
                f"engine library not found: {path}. "
                "Собери mock: cmake -S . -B build/cpu -DSTELL_CUDA=OFF && "
                "cmake --build build/cpu --target stell_mock_core"
            )
        self._lib_path = path
        self.lib = ctypes.CDLL(str(path))
        self._handle = None
        self._free = None
        self._setup_signatures()

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
        # освобождение строк, возвращённых движком: расширение stell_free (mock),
        # иначе free() из libc того же процесса (in-process .so — тот же heap)
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

    def submit(self, request: Dict[str, Any]) -> int:
        raw = json.dumps(request).encode()
        msg = ctypes.c_char_p()
        sid = self.lib.stell_submit(self._require_handle(), raw, ctypes.byref(msg))
        text = ctypes.string_at(msg).decode() if msg else ""
        self._release_ptr(msg)
        if sid == 0:
            raise EngineError(text or "submit failed")
        return int(sid)

    def poll(self, session_id: int) -> List[Dict[str, Any]]:
        out = ctypes.c_char_p()
        n = self.lib.stell_poll(self._require_handle(), ctypes.c_uint64(session_id), ctypes.byref(out))
        if n < 0:
            return []  # сессия закрыта: событий больше нет (ADR-11 §3)
        events: List[Dict[str, Any]] = []
        if out:
            raw = ctypes.string_at(out)
            self._release_ptr(out)
            arr = json.loads(raw.decode())
            events = list(arr)
        return events

    def cancel(self, session_id: int) -> None:
        rc = self.lib.stell_cancel(self._require_handle(), ctypes.c_uint64(session_id))
        if rc != 0:
            raise EngineError(f"cancel failed for session {session_id}")

    def metrics(self) -> Dict[str, Any]:
        out = ctypes.c_char_p()
        rc = self.lib.stell_metrics(self._require_handle(), ctypes.byref(out))
        if rc != 0 or not out:
            raise EngineError("metrics failed")
        raw = ctypes.string_at(out)
        self._release_ptr(out)
        return json.loads(raw.decode())

    def close(self) -> None:
        if self._handle is not None:
            self.lib.stell_engine_destroy(self._handle)
            self._handle = None


