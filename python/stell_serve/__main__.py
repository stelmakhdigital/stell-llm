"""Запуск control-plane: python -m stell_serve [--port 8080] [--engine-lib PATH]

Порт по умолчанию 8080 (ADR-06). Движок — libstell_core.so (ADR-11); без
собранного движка запускается mock из build/cpu (см. README).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app, default_engine_cfg
from .engine import CtypesEngineBackend, default_engine_lib


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m stell_serve")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--engine-lib", default=None,
                        help="путь к libstell_core.so / libstell_mock_core.so")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--log-metrics", action="store_true",
                        help="печатать /metrics снапшот в конце")
    args = parser.parse_args(argv)

    lib = Path(args.engine_lib) if args.engine_lib else default_engine_lib()
    backend = CtypesEngineBackend(lib)
    app = create_app(backend, engine_cfg=default_engine_cfg(args.model))
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
