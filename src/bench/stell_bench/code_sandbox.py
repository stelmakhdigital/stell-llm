"""Песочница исполнения сгенерированного кода (quality-gate HumanEval-32,
[R4] round 3 — решение планировщика [Q4]).

Честная пометка о классе изоляции: это **process-sandbox, НЕ security-
граница**. Раннер ограничивает недоверенный код процессными механизмами
(RLIMIT_CPU/RLIMIT_AS/RLIMIT_NOFILE, wall-таймаут, изолированный temp-dir
как cwd, пустой env, `python -I` без пользовательского окружения) — этого
достаточно, чтобы гейт точности не вешал прогон и не расходовал ресурсы.
Но код исполняется с правами текущего пользователя, сеть на уровне
процесса не отключается (unprivileged seccomp/namespaces недоступны без
внешних инструментов), а файловая система ограничена только правами
пользователя. Граница пригодна для bench-этапа (продукт в прод-контуре
чужой код не исполняет — планировщик [Q4]); для произвольного недоверенного
кода нужна контейнерная изоляция (вне рамок этого раунда).

Механика: один subprocess на кейс — `sys.executable -I <script>` с env={}
(наследуется НИЧЕГО из родительского окружения; интерпретатор сам ставит
LC_CTYPE для stdio — единственная переменная в ребёнке), cwd — свежий
temp-dir (удаляется после кейса), preexec_fn ставит rlimits ДО exec.

Классификация исхода (SandboxResult.outcome):
  - "ok"      — exit 0 (кейс исполнен, включая тест-ассерты);
  - "timeout" — wall-таймаут истёк (процесс убит, раннер не висит);
  - "error"   — ненулевой exit / падение от сигнала (исключение,
    sys.exit(n), SIGXCPU при превышении RLIMIT_CPU и т.п.).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

# Протокол песочницы ([R4], фиксация для детерминизма прогонов):
CPU_LIMIT_S = 5                 # RLIMIT_CPU: 5 секунд процессорного времени
WALL_TIMEOUT_S = 10.0           # общий wall-таймаут кейса, сек
MEM_LIMIT_BYTES = 512 * 1024 * 1024   # RLIMIT_AS: 512 МиБ адресного пространства
NOFILE_LIMIT = 64               # RLIMIT_NOFILE: 64 открытых файла
STDERR_TAIL_CHARS = 2000        # хвост stderr в отчёте (без раздувания)


@dataclass
class SandboxResult:
    """Исход одного кейса в песочнице (см. классификацию в докстринге)."""
    outcome: str                       # "ok" | "timeout" | "error"
    exit_code: Optional[int] = None    # None для timeout
    stderr_tail: str = ""
    elapsed_s: float = 0.0
    detail: str = ""                   # человекочитаемая причина ("killed by signal 24" …)
    metadata: dict = field(default_factory=dict)  # лимиты/каталог кейса (provenance)


def _limit_resources(cpu_limit_s: int, mem_limit_bytes: int, nofile: int):
    """preexec_fn: rlimits ставятся в ребёнке ДО exec (POSIX, stdlib)."""
    import resource

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_s, cpu_limit_s))
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))

    return _apply


def run_code(code: str, *,
             timeout_s: float = WALL_TIMEOUT_S,
             cpu_limit_s: int = CPU_LIMIT_S,
             mem_limit_bytes: int = MEM_LIMIT_BYTES,
             nofile: int = NOFILE_LIMIT) -> SandboxResult:
    """Исполняет `code` как python-скрипт в песочнице (см. докстринг модуля).

    Никогда не вешает вызывающий: wall-таймаут + kill + doжд reap. Любая
    ошибка запуска (нет интерпретатора и т.п.) → outcome="error" — гейт
    считает кейс некорректным, прогон продолжается.
    """
    started = time.monotonic()
    limits = {"cpu_limit_s": cpu_limit_s, "mem_limit_bytes": mem_limit_bytes,
              "nofile": nofile, "wall_timeout_s": timeout_s, "env": "{}",
              "interpreter_flags": "-I (isolated)"}
    try:
        with tempfile.TemporaryDirectory(prefix="stell-sandbox-") as tmp:
            fd, script = tempfile.mkstemp(suffix=".py", dir=tmp)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(code)
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-I", script],
                    cwd=tmp,                     # изолированный temp-dir
                    env={},                      # пустой env: ничего из родителя
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    preexec_fn=_limit_resources(cpu_limit_s, mem_limit_bytes, nofile),
                    start_new_session=True,
                )
            except OSError as exc:
                return SandboxResult("error", None, "", time.monotonic() - started,
                                     f"запуск песочницы не удался: {exc}", limits)
            try:
                _, err = proc.communicate(timeout=timeout_s)
                elapsed = time.monotonic() - started
                err_tail = (err or b"")[-STDERR_TAIL_CHARS:].decode(
                    "utf-8", errors="replace")
                if proc.returncode == 0:
                    return SandboxResult("ok", 0, err_tail, elapsed, "", limits)
                detail = (f"exit={proc.returncode}" if proc.returncode > 0
                          else f"killed by signal {abs(proc.returncode)}"
                          " (RLIMIT_CPU/память/падение)")
                return SandboxResult("error", proc.returncode, err_tail, elapsed,
                                     detail, limits)
            except subprocess.TimeoutExpired:
                # wall-таймаут: убиваем (RLIMIT_CPU мог не сработать — кейс
                # может «висеть» без потребления CPU, напр. sleep/sleep-цикл)
                proc.kill()
                proc.communicate()
                return SandboxResult(
                    "timeout", None, "", time.monotonic() - started,
                    f"wall-таймаут {timeout_s}s (процесс убит; лимиты: "
                    f"cpu={cpu_limit_s}s, mem={mem_limit_bytes}Б, nofile={nofile})",
                    limits)
    except Exception as exc:  # песочница не должна ломать прогон гейта
        return SandboxResult("error", None, "", time.monotonic() - started,
                             f"ошибка песочницы: {exc!r}", limits)
