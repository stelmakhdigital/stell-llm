"""Quality-gates каркас (PLAN.md §5): GSM8K-32, HumanEval-32, PPL.

Датасеты и схема — src/bench/quality_gates/data/. Прогон без модели
честно помечается pending=True; значения (accuracy, PPL) появляются только
после интеграции реального движка (поток A) и весов (поток D).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_GATES_DIR = Path(__file__).resolve().parents[1] / "quality_gates"
DATA_DIR = _GATES_DIR / "data"
SCHEMA_PATH = _GATES_DIR / "schema.json"


@dataclass
class GateResult:
    gate: str
    dataset: str
    pending: bool
    value: Optional[float] = None  # accuracy / PPL
    unit: str = "score"
    items: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate, "dataset": self.dataset, "pending": self.pending,
            "value": self.value, "unit": self.unit, "items": self.items,
            "notes": self.notes,
        }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            items.append(json.loads(line))
    return items


def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def gsm8k_32() -> List[Dict[str, Any]]:
    return load_jsonl(DATA_DIR / "gsm8k_32.jsonl")


def humaneval_32() -> List[Dict[str, Any]]:
    return load_jsonl(DATA_DIR / "humaneval_32.jsonl")


def ppl_text() -> str:
    return (DATA_DIR / "ppl_text.txt").read_text(encoding="utf-8")


def run_gates_pending(runner_notes: Optional[List[str]] = None) -> List[GateResult]:
    """Заготовка прогона: без модели все гейты помечены pending=True."""
    note = runner_notes or ["нет модели/весов: значения качества не измерены"]
    return [
        GateResult("quality-gate", "gsm8k_32", pending=True, items=len(gsm8k_32()),
                   notes=note),
        GateResult("quality-gate", "humaneval_32", pending=True,
                   items=len(humaneval_32()), notes=note),
        GateResult("quality-gate", "ppl_fixed_text", pending=True,
                   items=len(ppl_text().split()), unit="ppl", notes=note),
    ]
