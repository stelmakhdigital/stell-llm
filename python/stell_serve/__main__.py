"""Запуск control-plane: python -m stell_serve [--port 8080] [--engine-lib PATH]

Порт по умолчанию 8080 (ADR-06). Конфигурация: python/stell_serve/config.py —
файл config.json / STELL_SERVE_CONFIG, аргументы CLI переопределяют (round 2).
Движок — libstell_core.so (ADR-11); без собранного движка запускается mock из
build/cpu (см. README).
"""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .config import load_config, ServeConfig
from .engine import CtypesEngineBackend


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m stell_serve")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--engine-lib", default=None,
                        help="путь к libstell_core.so / libstell_mock_core.so")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config", default=None,
                        help="путь к JSON-конфигу (иначе STELL_SERVE_CONFIG / config.json)")
    parser.add_argument("--cors-origins", default=None,
                        help="CORS-origins через запятую (переопределяет конфиг)")
    parser.add_argument("--log-metrics", action="store_true",
                        help="печатать /metrics снапшот в конце")
    args = parser.parse_args(argv)

    cfg: ServeConfig = load_config(args.config)
    if args.port is not None:
        cfg.port = args.port
    if args.host is not None:
        cfg.host = args.host
    if args.engine_lib is not None:
        cfg.engine_lib = args.engine_lib
    if args.model is not None:
        cfg.model = args.model
    if args.cors_origins:
        cfg.cors_origins = [o.strip() for o in args.cors_origins.split(",") if o.strip()]
    if args.log_metrics:
        cfg.log_metrics = True
    cfg.validate()

    from stell_serve.app import default_engine_cfg

    backend = CtypesEngineBackend(cfg.engine_lib)
    app = create_app(backend, engine_cfg=default_engine_cfg(cfg.model),
                     cors_origins=cfg.cors_origins)
    try:
        uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
