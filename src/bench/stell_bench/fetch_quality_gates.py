"""Скачивание РЕАЛЬНЫХ quality-gates датасетов (round 2, задача T-…).

Фиксированная выборка 32 задач с seed=42 (воспроизводимо между версиями):
  - GSM8K: openai/grade-school-math, test.jsonl (1319 задач);
  - HumanEval: openai/human-eval, HumanEval.jsonl.gz (164 задачи).
Зависимости — только urllib/gzip (std); файлы пишутся в
src/bench/quality_gates/data/ и коммитятся (объём малый, ~100 КиБ).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "quality_gates" / "data"

GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             "master/grade_school_math/data/test.jsonl")
HUMANEVAL_URL = ("https://raw.githubusercontent.com/openai/human-eval/"
                 "master/data/HumanEval.jsonl.gz")
SAMPLE_SEED = 42
SAMPLE_SIZE = 32
_UA = "stell-llm-quality-gates-fetcher (round 2)"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _sample(items, key, seed=SAMPLE_SEED, size=SAMPLE_SIZE):
    """Детерминированная выборка size элементов из items, отсортированных по key."""
    ordered = sorted(items, key=key)
    rng = random.Random(seed)
    return rng.sample(ordered, size)


def fetch_gsm8k(out_path: Path = DATA_DIR / "gsm8k_32.jsonl",
                seed: int = SAMPLE_SEED, size: int = SAMPLE_SIZE) -> int:
    raw = _fetch(GSM8K_URL).decode("utf-8")
    items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    picked = _sample(items, key=lambda x: x["question"], seed=seed, size=size)
    header = (f"// РЕАЛЬНЫЙ поднабор GSM8K (openai/grade-school-math, test.jsonl, "
              f"1319 задач): выборка {size} задач seed={seed} (детерминирована), "
              f"скрипт stell_bench.fetch_quality_gates; поля: id/question/answer\n")
    lines = []
    for i, item in enumerate(picked):
        lines.append(json.dumps({
            "id": f"gsm8k_test_{i:03d}",
            "question": item["question"],
            "answer": item["answer"].split("####")[-1].strip(),
        }, ensure_ascii=False))
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def fetch_humaneval(out_path: Path = DATA_DIR / "humaneval_32.jsonl",
                    seed: int = SAMPLE_SEED, size: int = SAMPLE_SIZE) -> int:
    raw = gzip.decompress(_fetch(HUMANEVAL_URL)).decode("utf-8")
    items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    picked = _sample(items, key=lambda x: x["task_id"], seed=seed, size=size)
    header = (f"// РЕАЛЬНЫЙ поднабор HumanEval (openai/human-eval, "
              f"HumanEval.jsonl.gz, 164 задачи): выборка {size} задач "
              f"seed={seed} (детерминирована), скрипт "
              f"stell_bench.fetch_quality_gates; поля: task_id/prompt/"
              f"canonical_solution/test/entry_point\n")
    lines = []
    for item in picked:
        lines.append(json.dumps({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item.get("canonical_solution", ""),
            "test": item.get("test", ""),
            "entry_point": item.get("entry_point", ""),
        }, ensure_ascii=False))
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stell_bench.fetch_quality_gates")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args(argv)
    n1 = fetch_gsm8k(seed=args.seed, size=args.size)
    n2 = fetch_humaneval(seed=args.seed, size=args.size)
    for f in (DATA_DIR / "gsm8k_32.jsonl", DATA_DIR / "humaneval_32.jsonl"):
        print(f"{f.name}: {f.stat().st_size} bytes")
    print(f"gsm8k: {n1} задач, humaneval: {n2} задач (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
