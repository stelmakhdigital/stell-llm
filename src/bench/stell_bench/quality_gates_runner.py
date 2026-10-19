"""Runner quality-gates (PLAN.md §5): GSM8K-32, HumanEval-32, PPL (round 2).

Round 2 (интеграция, поток E): датасеты РЕАЛЬНЫЕ (см. stell_bench.fetch_quality_gates
— фиксированная выборка 32 задач seed=42 из открытых датасетов OpenAI) и добавлен
ЖИВОЙ прогон (`--live`) поверх логитов движка:

- источник логитов — `stell_logits` ([R1], планировщик Q1: request
  {"prompt_tokens":[…], "positions":"last"|"all"}, возврат 0/-1, malloc/free
  конвенция ADR-11 §1) через биндинг python/stell_serve/engine.py;
  модель загружается СРЕДСТВАМИ движка (cfg runner:"ops", реальные веса
  tests/models/fixtures: mini_dense / mini_moe, weights_format safetensors);
- PPL считается по logprob позиций ("positions":"all", teacher-forcing за
  один вызов): softmax/logprob по логитам, PPL = exp(mean(-log P(tok[i+1])))
  по позициям предсказания (см. ppl_from_logits);
- GSM8K-32: генерация жадным декодированием (argmax, ничья → меньший индекс —
  конвенция golden), проверка корректности — сопоставление чисел (без
  исполнения чего-либо — выполнимо честно, реализовано);
- HumanEval-32: проверка корректности ТРЕБУЕТ ИСПОЛНЕНИЯ сгенерированного
  кода (unit-тесты HumanEval) — исполнения кода в runner нет (песочница/
  изоляция не входят в раунд) → честный pending с причиной.

Честные пометки:
- у mini-фикстур vocab=16 и byte-токенизатор движка не может отобразить
  произвольный байт: runner использует ДЕТЕРМИНИРОВАННУЮ проекцию
  байта → токен (b % vocab) и обратную карту id → текст (цифры < 10 →
  цифра, иначе разделитель). Это МЕХАНИЧЕСКИЙ базовый референс гейтов
  (регрессия пайплайна), а не оценка качества LLM — mini-фикстуры не
  являются языковыми моделями. Числа фиксируются в bench/results.
- пока `stell_logits` не вклеен в .so (pre-merge core), ABI-источник
  недоступен; `--live --logits-source golden-ref` использует НЕЗАВИСИМЫЙ
  Python-референс логитов (tests/models/golden/generate_golden.py, тот
  самый, что сгенерировал golden-фикстуры; сверен с HF 1.5e-7) — числа
  помечены provenance "golden-ref (fallback)".

Запуск: python -m stell_bench.quality_gates_runner [--live] [--engine-lib PATH]
[--results-dir DIR]; отчёты — bench/results/<date>-<slug>/; локальные — в
bench/results/local/ (gitignored).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .quality_gates import GateResult, gsm8k_32, humaneval_32, ppl_text
from .report import results_local_dir_for, results_dir_for, write_report

_ABI_DIVERGENCE = ("ABI ADR-11 round 1 не отдаёт логитов (PPL) и полных "
                   "генераций для проверки — расхождение контракта, "
                   "зафиксировано в отчёте round 2; гейты вычислимы только "
                   "после расширения ABI (новый символ — ABI-совместимо) и "
                   "появления весов/модели")

_NO_MODEL = "нет реальной модели/весов (реестр/лоадеры — поток D, runner — поток A)"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "tests" / "models" / "fixtures"
_GOLDEN_REF = _REPO_ROOT / "tests" / "models" / "golden" / "generate_golden.py"

# Фиксированные параметры прогона (детерминизм между версиями §5):
PPL_MAX_TOKENS = 128          # первые 128 токенов фиксированного текста
GSM8K_PROMPT_TOKENS = 64      # проекция вопроса → не более 64 токенов
GSM8K_MAX_STEPS = 16          # жадное декодирование: до 16 токенов ответа
DEFAULT_MODELS = ("mini_dense", "mini_moe")


# ----------------------------------------------------------------------------
# Расчёт по логитам (чистый Python: softmax/logprob; vocab mini-фикстур = 16)
# ----------------------------------------------------------------------------

def logprob_of(logits: List[float], token: int) -> float:
    """log softmax(logits)[token] (стабильная формула через max)."""
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    return math.log(exps[token] / sum(exps))


def ppl_from_logits(rows: List[List[float]], tokens: List[int]) -> Dict[str, float]:
    """PPL по logprob позиций (teacher-forcing, "positions":"all").

    rows[i] — логиты финального слоя на позиции i (предсказывают токен i+1);
    PPL = exp(mean(-log P(tok[i+1] | rows[i]))) по позициям предсказания
    (последняя позиция промпта не предсказывает ничего известного — не входит).
    """
    nlls = [-logprob_of(rows[i], tokens[i + 1])
            for i in range(min(len(rows), len(tokens) - 1))]
    if not nlls:
        raise ValueError("нет позиций предсказания: rows/tokens пусты")
    mean_nll = sum(nlls) / len(nlls)
    return {"ppl": math.exp(mean_nll), "mean_nll": mean_nll,
            "n_pred_positions": len(nlls)}


def greedy_token(logits: List[float]) -> int:
    """argmax, ничья → меньший индекс (конвенция golden, ADR-09 §4)."""
    best = max(logits)
    return min(i for i, v in enumerate(logits) if v == best)


# ----------------------------------------------------------------------------
# Токены: детерминированная проекция байта → vocab (см. честную пометку выше)
# ----------------------------------------------------------------------------

def project_to_vocab(text: str, vocab: int) -> List[int]:
    return [b % vocab for b in text.encode("utf-8")]


def detokenize_ids(ids: List[int]) -> str:
    """Обратная карта для проекции: id < 10 → цифра, иначе разделитель."""
    return "".join(str(t) if t < 10 else " " for t in ids)


def extract_answer_number(text: str) -> Optional[int]:
    """GSM8K-конвенция оценки: последнее число в завершении модели."""
    nums = re.findall(r"\d+", text)
    return int(nums[-1]) if nums else None


# ----------------------------------------------------------------------------
# Источники логитов (LogitsSource): ABI `stell_logits` — канон; golden-ref —
# честный fallback до merge core [R1].
# ----------------------------------------------------------------------------

class LogitsSource(Protocol):
    name: str

    def logits_last(self, tokens: List[int]) -> List[float]: ...
    def logits_all(self, tokens: List[int]) -> List[List[float]]: ...
    def close(self) -> None: ...


class AbiLogitsSource:
    """Канонический источник: символ `stell_logits` реального движка ([R1]);
    модель загружается средствами движка (runner:"ops", реальные веса)."""

    def __init__(self, backend) -> None:
        if not backend.supports_logits:
            raise RuntimeError("движок не экспортирует stell_logits ([R1])")
        self._backend = backend
        self.name = "abi:stell_logits"

    def logits_last(self, tokens: List[int]) -> List[float]:
        payload = self._backend.logits(tokens, positions="last")
        rows = payload["logits"]
        return list(rows[-1])

    def logits_all(self, tokens: List[int]) -> List[List[float]]:
        payload = self._backend.logits(tokens, positions="all")
        return [list(r) for r in payload["logits"]]

    def close(self) -> None:
        self._backend.close()


class GoldenRefLogitsSource:
    """Fallback до merge core [R1]: НЕЗАВИСИМЫЙ Python-референс логитов
    (tests/models/golden/generate_golden.py — тот, что сгенерировал golden-
    фикстуры; сверен против HF Llama/Mixtral, max_rel ~1.5e-7). Реальные веса
    читаются из tests/models/fixtures (loader-путь референса)."""

    def __init__(self, model: str) -> None:
        spec = importlib.util.spec_from_file_location(
            "stell_golden_generate", _GOLDEN_REF)
        gg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gg)
        self._gg = gg
        self._desc = json.loads((_FIXTURES / f"{model}.json").read_text())
        self._weights = gg.read_safetensors(
            str(_FIXTURES / f"{model}.safetensors"))
        gg.apply_nvfp4_reference(self._desc, self._weights)
        self.name = (f"golden-ref:generate_golden.interpret ({model}, fallback "
                     "до merge core [R1]; референс сверен с HF ~1.5e-7)")

    def logits_last(self, tokens: List[int]) -> List[float]:
        return [float(v) for v in self._gg.interpret(
            self._desc, self._weights, tokens)]

    def logits_all(self, tokens: List[int]) -> List[List[float]]:
        # "positions":"all" по каузальной семантике: логиты позиции i — это
        # логиты последней позиции префикса tokens[:i+1] (позиционного
        # кодирования в ops-схеме нет — конвенция golden README).
        return [self.logits_last(tokens[:i + 1]) for i in range(len(tokens))]

    def close(self) -> None:
        pass


def open_logits_source(model: str, engine_lib: Optional[str],
                       prefer: str = "auto") -> LogitsSource:
    """Выбор источника: 'abi' — только stell_logits; 'golden-ref' — только
    референс; 'auto' — ABI, при его отсутствии честный fallback."""
    from stell_serve.engine import CtypesEngineBackend, resolve_engine_lib

    if prefer == "golden-ref":
        return GoldenRefLogitsSource(model)

    path, _ = resolve_engine_lib(engine_lib)
    try:
        backend = CtypesEngineBackend(path)
    except Exception:
        if prefer == "abi":
            raise
        return GoldenRefLogitsSource(model)
    try:
        if backend.supports_logits:
            cfg_model = dict(description=str(_FIXTURES / f"{model}.json"),
                             weights=str(_FIXTURES / f"{model}.safetensors"),
                             weights_format="safetensors", tokenizer="byte")
            backend.create({"runner": "ops", "engine":
                            {"max_slots": 16, "batch_step_max": 16},
                            "model": cfg_model})
            return AbiLogitsSource(backend)
    except Exception:
        backend.close()
        raise
    backend.close()
    if prefer == "abi":
        raise RuntimeError(
            f"{path}: .so не экспортирует stell_logits ([R1]; включится "
            "после merge core)")
    return GoldenRefLogitsSource(model)


# ----------------------------------------------------------------------------
# Гейты
# ----------------------------------------------------------------------------

def run_ppl_gate(source: LogitsSource, model: str) -> GateResult:
    """PPL фиксированного текста по logprob позиций ("positions":"all")."""
    vocab = _vocab_size(model)
    tokens = project_to_vocab(ppl_text(), vocab)[:PPL_MAX_TOKENS]
    rows = source.logits_all(tokens)
    stats = ppl_from_logits(rows, tokens)
    return GateResult(
        "quality-gate", "ppl_fixed_text", pending=False,
        value=round(stats["ppl"], 4), unit="ppl",
        items=stats["n_pred_positions"],
        notes=[f"модель={model}", f"источник логитов={source.name}",
               f"токенов={len(tokens)} (первые {PPL_MAX_TOKENS} фиксированного "
               "текста, проекция байта → токен b % vocab — vocab mini-"
               "фикстуры=16; это механический базовый референс, а не "
               "языковая PPL)",
               f"mean_nll={stats['mean_nll']:.6f}"])


def run_gsm8k_gate(source: LogitsSource, model: str) -> GateResult:
    """GSM8K-32: жадное декодирование через логиты + сопоставление чисел
    (корректность проверяется БЕЗ исполнения — вычислимо честно)."""
    items = gsm8k_32()
    vocab = _vocab_size(model)
    correct = 0
    for item in items:
        prompt = project_to_vocab(item["question"], vocab)[:GSM8K_PROMPT_TOKENS]
        seq = list(prompt)
        for _ in range(GSM8K_MAX_STEPS):
            row = source.logits_last(seq)
            seq.append(greedy_token(row))
        text = detokenize_ids(seq[len(prompt):])
        predicted = extract_answer_number(text)
        try:
            gold = int(item["answer"])
        except ValueError:
            gold = None
        if predicted is not None and gold is not None and predicted == gold:
            correct += 1
    accuracy = correct / len(items)
    return GateResult(
        "quality-gate", "gsm8k_32", pending=False, value=round(accuracy, 4),
        unit="score", items=len(items),
        notes=[f"модель={model}", f"источник логитов={source.name}",
               f"верно={correct}/{len(items)}",
               ("жадное декодирование (argmax, ≤16 шагов); проверка "
                "корректности — последнее число в декодированном тексте "
                "(без исполнения; вычислимо честно)"),
               ("механический базовый референс пайплайна: mini-фикстуры не "
                "являются языковыми моделями — значение фиксируется как "
                "точка регрессии, а не оценка качества LLM")])


def run_humaneval_gate(source: LogitsSource, model: str) -> GateResult:
    """HumanEval-32: ЧЕСТНЫЙ pending — корректность требует исполнения
    сгенерированного кода (unit-тесты HumanEval), а исполнения кода в
    runner нет (песочница/изоляция — вне раунда)."""
    items = humaneval_32()
    return GateResult(
        "quality-gate", "humaneval_32", pending=True, value=None,
        unit="score", items=len(items),
        notes=[f"модель={model}", f"источник логитов={source.name}",
               ("генерация через логиты возможна (жадное декодирование), но "
                "проверка корректности HumanEval = ИСПОЛНЕНИЕ сгенерированного "
                "кода против unit-тестов — исполнения кода в runner нет "
                "(песочница/изоляция не входят в раунд); честный pending")])


def _vocab_size(model: str) -> int:
    desc = json.loads((_FIXTURES / f"{model}.json").read_text())
    return int(desc["vocab_size"])


class _GatesSnapshot:
    """Обёртка отчёта write_report: снапшот гейтов (pending-статус + значения)."""

    def __init__(self, gates: List[GateResult], extra: Optional[Dict[str, Any]] = None) -> None:
        self.gates = gates
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        d = {"pending": all(g.pending for g in self.gates),
             "gates": [g.to_dict() for g in self.gates]}
        d.update(self.extra)
        return d


def run_gates() -> List[GateResult]:
    """Снимок гейтов: датасеты реальны, значения — честный pending (round 2)."""
    return [
        GateResult("quality-gate", "gsm8k_32", pending=True, value=None,
                   items=len(gsm8k_32()), unit="score",
                   notes=[_NO_MODEL, _ABI_DIVERGENCE]),
        GateResult("quality-gate", "humaneval_32", pending=True, value=None,
                   items=len(humaneval_32()), unit="score",
                   notes=[_NO_MODEL, _ABI_DIVERGENCE]),
        GateResult("quality-gate", "ppl_fixed_text", pending=True, value=None,
                   items=len(ppl_text().split()), unit="ppl",
                   notes=[_NO_MODEL, _ABI_DIVERGENCE]),
    ]


def run_gates_live(models: List[str] = list(DEFAULT_MODELS),
                   engine_lib: Optional[str] = None,
                   prefer: str = "auto") -> List[GateResult]:
    """Живой прогон гейтов против mini-фикстур (реальные веса): PPL +
    GSM8K через логиты; HumanEval — честный pending (нужно исполнение кода).
    По каждой модели — свой комплект гейтов (dataset суффиксируется)."""
    gates: List[GateResult] = []
    for model in models:
        source = open_logits_source(model, engine_lib, prefer=prefer)
        try:
            ppl = run_ppl_gate(source, model)
            gsm = run_gsm8k_gate(source, model)
            he = run_humaneval_gate(source, model)
        finally:
            source.close()
        ppl.dataset = f"ppl_fixed_text[{model}]"
        gsm.dataset = f"gsm8k_32[{model}]"
        he.dataset = f"humaneval_32[{model}]"
        gates.extend([ppl, gsm, he])
    return gates


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stell_bench.quality_gates_runner")
    parser.add_argument("--engine-lib", default=None,
                        help="libstell_core.so (реальный движок или mock)")
    parser.add_argument("--results-dir", default=None,
                        help="переопределить каталог отчёта (для тестов)")
    parser.add_argument("--live", action="store_true",
                        help="живой прогон через логиты (stell_logits / "
                             "golden-ref fallback); без флага — pending-снимок")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="mini-фикстуры для прогона (mini_dense,mini_moe)")
    parser.add_argument("--logits-source", default="auto",
                        choices=["auto", "abi", "golden-ref"],
                        help="источник логитов (auto: ABI, при отсутствии "
                             "символа — честный fallback на референс)")
    parser.add_argument("--committed", action="store_true",
                        help="писать в bench/results/<date>-<slug>/ (коммитится), "
                             "а не в bench/results/local/ (gitignored)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "bench"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.live:
        gates = run_gates_live(models=models, engine_lib=args.engine_lib,
                               prefer=args.logits_source)
        snapshot = _GatesSnapshot(gates, extra={
            "mode": "live",
            "models": models,
            "logits_source": sorted({g.to_dict()["notes"][1]
                                     for g in gates}),
            "protocol": {"ppl_max_tokens": PPL_MAX_TOKENS,
                         "gsm8k_prompt_tokens": GSM8K_PROMPT_TOKENS,
                         "gsm8k_max_steps": GSM8K_MAX_STEPS,
                         "projection": "byte % vocab, detok: id<10 → цифра"},
            "env": {"co_tenancy": False, "gpu": "none (CPU-local)"},
        })
        log_lines = [
            f"quality-gates LIVE: модели={models}, "
            f"источник логитов={snapshot.extra['logits_source']}",
            "HumanEval — честный pending (проверка корректности требует "
            "исполнения сгенерированного кода; песочницы в раунде нет)",
        ]
        if args.results_dir:
            out_dir = Path(args.results_dir)
        elif args.committed:
            out_dir = results_dir_for("quality-gates-live")
        else:
            out_dir = results_local_dir_for("quality-gates-live")
    else:
        gates = run_gates()
        snapshot = _GatesSnapshot(gates)
        out_dir = (Path(args.results_dir) if args.results_dir
                   else results_local_dir_for("quality-gates-local"))
        log_lines = [
            "quality-gates: датасеты реальные (32 задачи, seed=42); прогон — "
            "pending=True (нет модели/весов; ABI без логитов — см. notes)",
        ]

    write_report(snapshot, out_dir, log_lines=log_lines)
    print(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False))
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
