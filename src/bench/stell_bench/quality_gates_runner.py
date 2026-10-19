"""Runner quality-gates (PLAN.md §5): GSM8K-32, HumanEval-32, PPL (round 3).

Round 2 (интеграция, поток E): датасеты РЕАЛЬНЫЕ (см. stell_bench.fetch_quality_gates
— фиксированная выборка 32 задач seed=42 из открытых датасетов OpenAI) и добавлен
ЖИВОЙ прогон (`--live`) поверх логитов движка.

Round 3 ([R4], решение планировщика [Q4]): HumanEval-32 больше НЕ pending —
корректность проверяется ИСПОЛНЕНИЕМ сгенерированного кода в песочнице
(code_sandbox.run_code: process-sandbox с RLIMIT_CPU/AS/NOFILE, wall-таймаут,
изолированный temp-dir, пустой env; **не security-граница** — честная пометка
в докстринге песочницы). Конвейер кейса: жадное декодирование движка →
детокенизация ответа → извлечение кода (первый markdown-fence, иначе весь
ответ) → candidate = prompt + completion → исполнение против test-кейса
задачи в песочнице → accuracy по 32 кейсам.

- источник логитов — `stell_logits` ([R1], планировщик Q1: request
  {"prompt_tokens":[…], "positions":"last"|"all"}, возврат 0/-1, malloc/free
  конвенция ADR-11 §1) через биндинг python/stell_serve/engine.py;
  модель загружается СРЕДСТВАМИ движка (cfg runner:"ops", реальные веса
  tests/models/fixtures: mini_dense / mini_moe, weights_format safetensors);
- PPL считается по logprob позиций (teacher-forcing): режим — по порогу
  PPL_ALL_MAX_ELEMENTS элементов логитов за вызов. Малые словари — "all"
  одним вызовом (teacher-forcing за один раз); большие (n_vocab × n_positions
  > порога; напр. tiny-qwen2moe vocab=151646: 128 × 151646 ≈ 19.4 млн ≈ 300+
  МиБ JSON в одном ответе — прогон зависал) — 'incremental-last': повторные
  positions:"last" по растущему префиксу, по строке за вызов. Чанки "all"
  с окном контекста некорректны каузально (ABI не имеет смещения контекста);
  математика путей идентична: rows[i] = logits_last(tokens[:i+1]) —
  значения совпадают (сверено тестом). softmax/logprob по логитам,
  PPL = exp(mean(-log P(tok[i+1]))) по позициям предсказания (см. ppl_from_logits);
- GSM8K-32: генерация жадным декодированием (argmax, ничья → меньший индекс —
  конвенция golden), проверка корректности — сопоставление чисел (без
  исполнения чего-либо — выполнимо честно, реализовано);
- HumanEval-32 ([R4] round 3): корректность через песочницу кода
  (code_sandbox.py, process-sandbox — не security-граница); у mini-фикстур
  генерация — механическая проекция байта → токен, поэтому accuracy —
  точка регрессии пайплайна (как у GSM8K), а не оценка качества LLM.

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

Round 4 ([R8], режим «реальная модель»): `--real` принимает каталог модели —
description.json (ADR-09, после hf_to_description [R4]) + model.safetensors
(+ опциональный tokenizer.json — РЕАЛЬНЫЙ byte-level BPE, порт R10:
bpe_tokenizer.py, сверено с fixtures/tokenizer/tokenizer_ref.json, расхожде-
ние 0) — или явные файлы (--model-desc/--model-weights); env-переключатель
STELL_R8_REAL_MODEL_DIR — по прецеденту R7 (без правки кода). Токенизация —
BPE каталога; без tokenizer.json — честная проекция байта (значения —
механические точки регрессии). Источник логитов — ТОЛЬКО ABI (golden-ref
умеет только mini-фикстуры; подмена источника запрещена). Provenance точки
регрессии — модель, размер весов, дата, co-tenancy (bench/results/). Без
источника — честный pending «веса не предоставлены»; явно битый источник —
громкая ошибка. До появления настоящих весов ([R3]/[R4]) конвейер отработан
на синтетическом каталоге «реалистичного профиля» (synthetic_model.py:
MoE 4×8, vocab=350, LCG-веса — воспроизводимо).

Запуск: python -m stell_bench.quality_gates_runner [--live] [--engine-lib PATH]
[--results-dir DIR]; отчёты — bench/results/<date>-<slug>/; локальные — в
bench/results/local/ (gitignored).
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .bpe_tokenizer import BpeTokenizer
from .code_sandbox import run_code
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
# R8 fixup (серверный прогон tiny-qwen2moe, vocab=151646): один вызов
# positions:"all" на большом словаре — это n_positions × n_vocab логитов в
# ОДНОМ JSON-ответе (128 × 151646 ≈ 19.4 млн ≈ 300+ МиБ) — прогон зависает.
# Порог переключения режима PPL (элементов логитов за один вызов):
PPL_ALL_MAX_ELEMENTS = 2_000_000
GSM8K_PROMPT_TOKENS = 64      # проекция вопроса → не более 64 токенов
GSM8K_MAX_STEPS = 16          # жадное декодирование: до 16 токенов ответа
HUMANEVAL_PROMPT_TOKENS = 64  # проекция сигнатуры+docstring → ≤ 64 токенов
HUMANEVAL_MAX_STEPS = 32      # жадное декодирование: до 32 токенов ответа
DEFAULT_MODELS = ("mini_dense", "mini_moe")

# Режим «реальная модель» (R8, round 4): каталог с описанием ADR-09 (после
# hf_to_description [R4]) и весами (+ опциональный tokenizer.json — РЕАЛЬНЫЙ
# byte-level BPE); контракт одного файла весов (см. resolve_real_model).
ENV_REAL_MODEL_DIR = "STELL_R8_REAL_MODEL_DIR"   # по прецеденту R7 (env без правки кода)
REAL_DESCRIPTION_NAME = "description.json"       # вывод hf_to_description [R4]
REAL_WEIGHTS_NAME = "model.safetensors"          # один файл (см. honest error при шардах)
REAL_TOKENIZER_NAME = "tokenizer.json"           # опционально; иначе проекция байта


class RealModelSpec:
    """Разрешённый каталог реальной модели (R8): описание ADR-09 + веса
    (+ tokenizer.json). Строится resolve_real_model(); используется
    run_gates_real (живой прогон гейтов через ABI)."""

    def __init__(self, desc_path: Path, weights_path: Path,
                 desc: Dict[str, Any], root: Optional[Path] = None,
                 tokenizer_path: Optional[Path] = None) -> None:
        self.desc_path = Path(desc_path)
        self.weights_path = Path(weights_path)
        self.desc = desc
        self.root = Path(root) if root is not None else None
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else None

    @property
    def name(self) -> str:
        """Метка модели в отчёте: name описания, иначе имя каталога."""
        return str(self.desc.get("name") or (self.root or self.desc_path.parent).name)

    @property
    def vocab_size(self) -> int:
        vocab = self.desc.get("vocab_size")
        if not isinstance(vocab, int) or vocab <= 0:
            raise ValueError(f"description.json: vocab_size должен быть int > 0"
                             f" ({self.desc_path})")
        return vocab


def resolve_real_model(model_dir: Optional[str] = None,
                       desc_json: Optional[str] = None,
                       weights: Optional[str] = None) -> Optional[RealModelSpec]:
    """Разрешение каталога реальной модели (R8).

    Приоритет: явные файлы (--model-desc + --model-weights, оба обязательны
    вместе) → каталог (--real-model-dir / env STELL_R8_REAL_MODEL_DIR):
    description.json обязателен, model.safetensors обязателен (ровно один
    файл — HF-шарды сначала склеиваются/конвертируются: plain-лоадер
    safetensors открывает один файл, честная ошибка вместо догадок);
    tokenizer.json — опционален.

    Возвращает None, если источник не задан ни аргументами, ни env —
    вызывающий честно помечает прогон pending («веса не предоставлены»).
    Явно заданный, но битый источник — честная ошибка (не pending).
    """
    if (desc_json is None) != (weights is None):
        raise RuntimeError("--model-desc и --model-weights задаются вместе")
    if desc_json is not None:
        d_path, w_path = Path(desc_json), Path(weights)
        root = None
    else:
        src = model_dir or os.environ.get(ENV_REAL_MODEL_DIR)
        if not src:
            return None
        root = Path(src)
        if not root.is_dir():
            raise FileNotFoundError(
                f"каталог модели не найден ({ENV_REAL_MODEL_DIR}/{model_dir}): {root}")
        d_path = root / REAL_DESCRIPTION_NAME
        w_path = root / REAL_WEIGHTS_NAME
    if not d_path.is_file():
        raise FileNotFoundError(
            f"описание ADR-09 не найдено: {d_path} (ожидался {REAL_DESCRIPTION_NAME} "
            "— вывод hf_to_description [R4])")
    if not w_path.is_file():
        raise FileNotFoundError(
            f"веса не найдены: {w_path} (ожидался {REAL_WEIGHTS_NAME}; HF-шарды "
            "сначала склеиваются в один safetensors или конвертируются в "
            "stell-tier — plain-лоадер safetensors открывает ровно один файл)")
    try:
        desc = json.loads(d_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{d_path}: битый JSON ({e})") from e
    if not isinstance(desc, dict) or not desc.get("name"):
        raise ValueError(f"{d_path}: описание без name (не ADR-09?)")
    tok_path = root / REAL_TOKENIZER_NAME if root is not None\
        else d_path.parent / REAL_TOKENIZER_NAME
    return RealModelSpec(d_path, w_path, desc, root=root,
                         tokenizer_path=tok_path if tok_path.is_file() else None)


def real_provenance(spec: RealModelSpec, engine_lib: str, co_tenancy: bool,
                    tokenizer: Optional[BpeTokenizer],
                    ppl_mode: Optional[str] = None,
                    weights_format: str = "safetensors") -> Dict[str, Any]:
    """Provenance точки регрессии (модель, размер, дата, co-tenancy) — R8.

    ppl_mode — честная пометка режима PPL ('all' / 'incremental-last');
    None → оценка по протоколу (vocab × PPL_MAX_TOKENS против порога).
    """
    moe = spec.desc.get("moe") or {}
    return {
        "mode": "real-model",
        "model": spec.name,
        "model_class": spec.desc.get("class"),
        # dtype-происхождение весов (честный профиль): dtype_weights — профиль
        # описания (определяет weights_format), dtype_orig — исходный dtype
        # весов до конверсии ([R5] models: BF16-источник → "bf16", конверсия
        # с clamp-сатурацией ±65504, ADR-03 round 5); пока описание не несёт
        # поле — None (честно: поле не предоставлено, не догадки).
        "dtype_weights": spec.desc.get("dtype_weights"),
        "dtype_orig": spec.desc.get("dtype_orig"),
        "description": str(spec.desc_path),
        "weights": str(spec.weights_path),
        "weights_bytes": spec.weights_path.stat().st_size,
        "weights_format": weights_format,
        "vocab_size": spec.vocab_size,
        "n_layers": spec.desc.get("n_layers"),
        "n_experts": moe.get("n_experts"),
        "tokenizer_json": (str(spec.tokenizer_path)
                           if spec.tokenizer_path else None),
        "tokenization": ("byte-level BPE (tokenizer.json)" if tokenizer
                         else "проекция байта → токен b % vocab (tokenizer.json "
                              "в каталоге отсутствует)"),
        "ppl_mode": (ppl_mode if ppl_mode is not None else
                     ppl_mode_for(spec.vocab_size, PPL_MAX_TOKENS)),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "co_tenancy": bool(co_tenancy),
        "engine_lib": str(engine_lib),
        "logits_source": "abi:stell_logits",
    }


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

def _tokens_for_text(text: str, vocab: int, tokenizer: Optional[BpeTokenizer],
                     limit: int) -> List[int]:
    """Токенизация текста: РЕАЛЬНЫЙ byte-level BPE (tokenizer.json каталога,
    R8) или детерминированная проекция байта → токен b % vocab (mini-фикстуры
    и каталоги без tokenizer.json — честная пометка в notes)."""
    if tokenizer is not None:
        return list(tokenizer.encode(text))[:limit]
    return project_to_vocab(text, vocab)[:limit]


def _decode_ids(ids: List[int], tokenizer: Optional[BpeTokenizer]) -> str:
    """Обратная карта: BPE-детокенизатор или проекция (id<10 → цифра)."""
    if tokenizer is not None:
        return tokenizer.decode_ids(ids)
    return detokenize_ids(ids)


def ppl_mode_for(vocab: int, n_positions: int) -> str:
    """Режим PPL (R8 fixup): 'all' или 'incremental-last'.

    - 'all' — teacher-forcing ОДНИМ вызовом positions:"all" (малые словари;
      тот же путь, что у mini-фикстур — значения прежние);
    - 'incremental-last' — teacher-forcing ПОЗИЦИЯМ: повторные вызовы
      positions:"last" с растущим префиксом (по ОДНОЙ строке логитов за
      вызов — ответ ~n_vocab элементов, а не n_positions × n_vocab).

    Чанковые вызовы "all" с окном контекста невозможны честно: каузальная
    семантика требует для строки i полный префикс tokens[:i+1], а ABI не
    имеет параметра смещения контекста — усечённое окно дало бы логиты без
    левого контекста. Поэтому единственная честная альтернатива "all" на
    больших словарях — инкрементальный last, с ТОЙ ЖЕ математикой:
    rows[i] = logits_last(tokens[:i+1]) — значения путей совпадают
    (сверено тестом на детерминированном стабе).

    Порог: n_vocab × n_positions > PPL_ALL_MAX_ELEMENTS элементов логитов
    за один вызов → инкрементальный путь.
    """
    if vocab * n_positions > PPL_ALL_MAX_ELEMENTS:
        return "incremental-last"
    return "all"


def run_ppl_gate(source: LogitsSource, model: str, *, vocab: Optional[int] = None,
                 tokenizer: Optional[BpeTokenizer] = None,
                 notes_extra: Optional[List[str]] = None) -> GateResult:
    """PPL фиксированного текста по logprob позиций (teacher-forcing).

    Режим — ppl_mode_for(vocab, len(tokens)): 'all' одним вызовом (малые
    словари) или 'incremental-last' — повторные positions:"last" по растущему
    префиксу (R8 fixup: на vocab ≥ 100к один вызов "all" — это сотни МиБ
    JSON; математика путей идентична: rows[i] = logits_last(tokens[:i+1]))."""
    vocab = _vocab_size(model) if vocab is None else vocab
    tokens = _tokens_for_text(ppl_text(), vocab, tokenizer, PPL_MAX_TOKENS)
    mode = ppl_mode_for(vocab, len(tokens))
    if mode == "all":
        rows = source.logits_all(tokens)
        stats = ppl_from_logits(rows, tokens)
    else:
        if len(tokens) < 2:
            raise ValueError("нет позиций предсказания: tokens пуст/одиночен")
        nlls = []
        for i in range(len(tokens) - 1):
            row = source.logits_last(tokens[:i + 1])
            nlls.append(-logprob_of(row, tokens[i + 1]))
        mean_nll = sum(nlls) / len(nlls)
        stats = {"ppl": math.exp(mean_nll), "mean_nll": mean_nll,
                 "n_pred_positions": len(nlls)}
    mode_note = (
        f"режим PPL: all (teacher-forcing одним вызовом; vocab={vocab} × "
        f"позиций={len(tokens)} ≤ порога {PPL_ALL_MAX_ELEMENTS} элементов)"
        if mode == "all" else
        f"режим PPL: incremental-last (teacher-forcing по позициям, повторные "
        "positions:last по растущему префиксу, по строке за вызов; "
        f"vocab={vocab} × позиций={len(tokens)} > порога "
        f"{PPL_ALL_MAX_ELEMENTS} элементов; чанки all с окном контекста "
        "некорректны каузально — ABI не имеет смещения контекста; математика "
        "та же: rows[i]=logits_last(tokens[:i+1]), значения путей совпадают)")
    tok_note = (f"токенизация: byte-level BPE tokenizer.json, токенов={len(tokens)} "
                f"(первые {PPL_MAX_TOKENS} фиксированного текста)" if tokenizer else
                f"токенов={len(tokens)} (первые {PPL_MAX_TOKENS} фиксированного "
                "текста, проекция байта → токен b % vocab, vocab=" + str(vocab) +
                "; механический базовый референс, а не языковая PPL)")
    return GateResult(
        "quality-gate", "ppl_fixed_text", pending=False,
        value=round(stats["ppl"], 4), unit="ppl",
        items=stats["n_pred_positions"],
        notes=[f"модель={model}", f"источник логитов={source.name}", mode_note,
               tok_note,
               f"mean_nll={stats['mean_nll']:.6f}"] + (notes_extra or []))


def run_gsm8k_gate(source: LogitsSource, model: str, *,
                   vocab: Optional[int] = None,
                   tokenizer: Optional[BpeTokenizer] = None,
                   notes_extra: Optional[List[str]] = None) -> GateResult:
    """GSM8K-32: жадное декодирование через логиты + сопоставление чисел
    (корректность проверяется БЕЗ исполнения — вычислимо честно)."""
    items = gsm8k_32()
    vocab = _vocab_size(model) if vocab is None else vocab
    correct = 0
    for item in items:
        prompt = _tokens_for_text(item["question"], vocab, tokenizer,
                                  GSM8K_PROMPT_TOKENS)
        seq = list(prompt)
        for _ in range(GSM8K_MAX_STEPS):
            row = source.logits_last(seq)
            seq.append(greedy_token(row))
        text = _decode_ids(seq[len(prompt):], tokenizer)
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
               "жадное декодирование (argmax, ≤16 шагов); проверка "
               "корректности — последнее число в декодированном тексте "
               "(без исполнения; вычислимо честно)"]
               + ([] if tokenizer else [
                   "механический базовый референс пайплайна: проекция байта "
                   "→ токен — mini-фикстуры/каталоги без tokenizer.json — не "
                   "языковые модели; значение — точка регрессии, а не "
                   "оценка качества LLM"])
               + (notes_extra or []))


def extract_code(answer: str) -> Optional[str]:
    """Извлечение кода из ответа модели ([R4]): первый markdown-fence
    ```...``` (метка языка после маркера вырезается); БЕЗ fence — весь
    ответ трактуется как код (модель может отвечать без разметки).
    None — если непустого кода в ответе нет."""
    m = re.search(r"```[\w+-]*[ \t]*\r?\n(.*?)```", answer, re.S)
    code = m.group(1) if m else answer
    return code if code.strip() else None


def humaneval_program(item: Dict[str, Any], answer: str) -> str:
    """candidate-программа кейса: prompt задачи + извлечённый completion
    (completion-контракт HumanEval)."""
    code = extract_code(answer)
    return item["prompt"] + (code or "")


def humaneval_exec_script(item: Dict[str, Any], program: str) -> str:
    """Скрипт для песочницы: candidate + test-кейс задачи + вызов check().
    exit 0 ⇔ все ассерты тест-кейса прошли на сгенерированном коде."""
    return (program + "\n\n\n" + item["test"]
            + f"\n\ncheck({item['entry_point']})\n")


def run_humaneval_gate(source: LogitsSource, model: str, *,
                       vocab: Optional[int] = None,
                       tokenizer: Optional[BpeTokenizer] = None,
                       notes_extra: Optional[List[str]] = None) -> GateResult:
    """HumanEval-32 [R4]: жадное декодирование → извлечение кода →
    исполнение в песочнице (process-sandbox, НЕ security-граница — см.
    code_sandbox) против test-кейса задачи. accuracy = доля кейсов, где
    песочница завершилась с exit 0."""
    items = humaneval_32()
    vocab = _vocab_size(model) if vocab is None else vocab
    correct = 0
    outcomes: Dict[str, int] = {}
    for item in items:
        prompt = _tokens_for_text(item["prompt"], vocab, tokenizer,
                                  HUMANEVAL_PROMPT_TOKENS)
        seq = list(prompt)
        for _ in range(HUMANEVAL_MAX_STEPS):
            row = source.logits_last(seq)
            seq.append(greedy_token(row))
        answer = _decode_ids(seq[len(prompt):], tokenizer)
        script = humaneval_exec_script(item, humaneval_program(item, answer))
        res = run_code(script)
        outcomes[res.outcome] = outcomes.get(res.outcome, 0) + 1
        if res.outcome == "ok":
            correct += 1
    accuracy = correct / len(items)
    return GateResult(
        "quality-gate", "humaneval_32", pending=False, value=round(accuracy, 4),
        unit="score", items=len(items),
        notes=[f"модель={model}", f"источник логитов={source.name}",
               f"верно={correct}/{len(items)}",
               "исходы песочницы: " + ", ".join(
                   f"{k}={v}" for k, v in sorted(outcomes.items())),
               ("конвейер [R4]: жадное декодирование (argmax, ≤32 токенов) → "
                "извлечение кода (первый markdown-fence, иначе весь ответ) → "
                "исполнение в песочнице против test-кейса (process-sandbox: "
                "RLIMIT_CPU=5с/RLIMIT_AS/RLIMIT_NOFILE, wall 10с, temp-dir, "
                "пустой env; НЕ security-граница — честная пометка)")]
               + ([] if tokenizer else [
                   "механический базовый референс пайплайна: проекция байта "
                   "→ токен — mini-фикстуры/каталоги без tokenizer.json — не "
                   "языковые модели; значение — точка регрессии, а не "
                   "оценка качества LLM"])
               + (notes_extra or []))


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
    GSM8K через логиты; HumanEval-32 — исполнение сгенерированного кода
    в песочнице ([R4] round 3). По каждой модели — свой комплект гейтов
    (dataset суффиксируется)."""
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


# ---------------------------------------------------------------------------
# Режим «реальная модель» (R8, round 4): PPL + GSM8K-32 + HumanEval-32 через
# ABI на каталоге с описанием ADR-09 (после hf_to_description [R4]) и весами.
# Источник логитов — ТОЛЬКО ABI (stell_logits): golden-ref-референс умеет
# интерпретировать только mini-фикстуры — молчаливая подмена источника
# запрещена. Прогон — worker ПОСЛЕ скачивания весов ([R3]) и конвертера
# описания; до того — честный pending «веса не предоставлены» (по образцу
# R7: env-переключатель без правки кода).
# ---------------------------------------------------------------------------

_REAL_HONEST_NOTE_SYNTHETIC = (
    "каталог модели — синтетический реалистичный профиль (не языковая "
    "модель): значения — точки регрессии конвейера R8, не оценка качества LLM")


def run_gates_real(spec: RealModelSpec, engine_lib: Optional[str] = None,
                   co_tenancy: bool = False) -> Tuple[List[GateResult], Dict[str, Any]]:
    """Живой прогон гейтов через ABI на каталоге реальной модели.

    Возвращает (гейты, provenance): модель, размер весов, дата, co-tenancy,
    ppl_mode — точки регрессии R8. Токенизация: tokenizer.json каталога
    (РЕАЛЬНЫЙ byte-level BPE, порт R10) или, при его отсутствии, честная
    проекция байта → токен (значения — механические точки регрессии).
    weights_format — из ADR-09 описания: квантованный профиль (NVFP4/W4A16)
    → лоадер nvfp4w4a16 (квантование на read, ADR-03; он же обслуживает
    плоские группы F16/F32 — фикс round 2), иначе — plain safetensors.
    """
    from stell_serve.engine import CtypesEngineBackend, resolve_engine_lib

    profile = str(spec.desc.get("dtype_weights") or "")
    weights_format = "nvfp4w4a16" if profile in ("NVFP4", "W4A16")\
        else "safetensors"
    path, _ = resolve_engine_lib(engine_lib)
    backend = CtypesEngineBackend(path)
    source = None
    try:
        if not backend.supports_logits:
            raise RuntimeError(
                "реальный режим (R8) требует ABI-источник логитов: "
                f"{path} не экспортирует stell_logits ([R1]; golden-ref "
                "умеет только mini-фикстуры — подмена источника запрещена)")
        cfg_model = dict(description=str(spec.desc_path),
                         weights=str(spec.weights_path),
                         weights_format=weights_format, tokenizer="byte")
        backend.create({"runner": "ops",
                        "engine": {"max_slots": 16, "batch_step_max": 16},
                        "model": cfg_model})
        source = AbiLogitsSource(backend)
        tokenizer = None
        if spec.tokenizer_path is not None:
            tokenizer = BpeTokenizer(str(spec.tokenizer_path))
        vocab = spec.vocab_size
        label = spec.name
        # Честная пометка для синтетических каталогов (регрессия конвейера,
        # не оценка качества LLM); для настоящих весов — без пометки.
        notes_extra = ([_REAL_HONEST_NOTE_SYNTHETIC]
                       if spec.name.startswith("synthetic") else None)
        ppl = run_ppl_gate(source, label, vocab=vocab, tokenizer=tokenizer,
                           notes_extra=notes_extra)
        gsm = run_gsm8k_gate(source, label, vocab=vocab, tokenizer=tokenizer,
                             notes_extra=notes_extra)
        he = run_humaneval_gate(source, label, vocab=vocab, tokenizer=tokenizer,
                                notes_extra=notes_extra)
        ppl.dataset = f"ppl_fixed_text[{label}]"
        gsm.dataset = f"gsm8k_32[{label}]"
        he.dataset = f"humaneval_32[{label}]"
        # Честная пометка режима PPL — ТОТ ЖЕ расчёт, что внутри run_ppl_gate
        # (те же входы: vocab, длина токенизированного PPL-текста).
        ppl_tokens = _tokens_for_text(ppl_text(), vocab, tokenizer, PPL_MAX_TOKENS)
        ppl_mode = ppl_mode_for(vocab, len(ppl_tokens))
        provenance = real_provenance(spec, str(path), co_tenancy, tokenizer,
                                     ppl_mode=ppl_mode,
                                     weights_format=weights_format)
        return [ppl, gsm, he], provenance
    finally:
        if source is not None:
            source.close()
        else:
            backend.close()


def run_gates_real_pending(reason: str) -> Tuple[List[GateResult], Dict[str, Any]]:
    """Честный pending «веса не предоставлены» (по образцу R7): значения
    качества НЕ измерены, причина зафиксирована в notes и provenance."""
    gates = [
        GateResult("quality-gate", "gsm8k_32", pending=True, value=None,
                   items=len(gsm8k_32()), unit="score",
                   notes=[f"веса не предоставлены (R8): {reason}"]),
        GateResult("quality-gate", "humaneval_32", pending=True, value=None,
                   items=len(humaneval_32()), unit="score",
                   notes=[f"веса не предоставлены (R8): {reason}"]),
        GateResult("quality-gate", "ppl_fixed_text", pending=True, value=None,
                   items=len(ppl_text().split()), unit="ppl",
                   notes=[f"веса не предоставлены (R8): {reason}"]),
    ]
    provenance = {
        "mode": "real-model",
        "weights_provided": False,
        "reason": reason,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "co_tenancy": None,
    }
    return gates, provenance


def wall_provenance(wall_seconds: Optional[float],
                    wall_phase: Optional[str] = None,
                    wall_before_seconds: Optional[float] = None
                    ) -> Dict[str, Any]:
    """Wall-поля provenance для сравнения до/после [R1] (round 5, [R8]).

    wall_seconds — wall-время прогона этой фазы (замер обёртки);
    wall_phase — фаза относительно движковой оптимизации [R1]: "before"
    (неоптимизированный движок, O(n²) на positions:"last") / "after"
    (после [R1] — проекция только последней позиции);
    wall_before_seconds — цифра фазы "before" в артефакте фазы "after"
    (обе цифры в одном provenance — требование planner [R1]/[R8]).
    Только переданные поля попадают в provenance (честно: не измерено —
    не записано).
    """
    fields: Dict[str, Any] = {}
    if wall_seconds is not None:
        fields["wall_seconds"] = round(float(wall_seconds), 3)
    if wall_phase is not None:
        if wall_phase not in ("before", "after"):
            raise ValueError("wall_phase: ожидалось 'before' или 'after': "
                             f"{wall_phase}")
        fields["wall_phase"] = wall_phase
    if wall_before_seconds is not None:
        fields["wall_before_seconds"] = round(float(wall_before_seconds), 3)
    return fields


def _protocol_snapshot() -> dict:
    """Протокол-константы прогона (детерминизм между версиями §5)."""
    return {
        "ppl_max_tokens": PPL_MAX_TOKENS,
        "gsm8k_prompt_tokens": GSM8K_PROMPT_TOKENS,
        "gsm8k_max_steps": GSM8K_MAX_STEPS,
        "humaneval_prompt_tokens": HUMANEVAL_PROMPT_TOKENS,
        "humaneval_max_steps": HUMANEVAL_MAX_STEPS,
        "humaneval_sandbox": {
            "class": "process-sandbox (НЕ security-граница)",
            "rlimit_cpu_s": 5, "wall_timeout_s": 10,
            "rlimit_as_bytes": 512 * 1024 * 1024,
            "rlimit_nofile": 64,
            "env": "пустой", "cwd": "изолированный temp-dir"},
        "projection": "byte % vocab, detok: id<10 → цифра (без tokenizer.json)",
        "ppl_all_max_elements": PPL_ALL_MAX_ELEMENTS,
        "ppl_mode_rule": ("n_vocab × n_positions > порога → incremental-last "
                          "(повторные positions:last по растущему префиксу, "
                          "по строке логитов за вызов); иначе all одним "
                          "вызовом; математика путей идентична"),
        "wall_measurement": ("date +%s.%N diff (обёртка run_real_gates.sh); "
                            "wall-поля provenance: wall_seconds/wall_phase/"
                            "wall_before_seconds — сравнение до/после [R1]"),
    }


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
    parser.add_argument("--real", action="store_true",
                        help="режим «реальная модель» (R8): PPL+GSM8K+HumanEval "
                             "через ABI на каталоге с описанием ADR-09 (после "
                             "hf_to_description [R4]) и весами; без источника — "
                             "честный pending «веса не предоставлены»")
    parser.add_argument("--real-model-dir", default=None,
                        help="каталог реальной модели (description.json + "
                             "model.safetensors [+ tokenizer.json]); "
                             "по умолчанию — env " + ENV_REAL_MODEL_DIR)
    parser.add_argument("--model-desc", default=None,
                        help="явный путь к описанию ADR-09 (с --model-weights; "
                             "приоритет над каталогом)")
    parser.add_argument("--model-weights", default=None,
                        help="явный путь к весам safetensors (один файл)")
    parser.add_argument("--co-tenancy", action="store_true",
                        help="честная пометка co-tenancy в provenance точки "
                             "регрессии")
    parser.add_argument("--wall-seconds", type=float, default=None,
                        help="wall-время прогона (замер обёртки "
                             "run_real_gates.sh, date-дифф) — в provenance")
    parser.add_argument("--wall-phase", default=None,
                        choices=["before", "after"],
                        help="фаза относительно движковой оптимизации [R1]: "
                             "before — неоптимизированный движок (O(n²) на "
                             "positions:last), after — после [R1]")
    parser.add_argument("--wall-before-seconds", type=float, default=None,
                        help="wall-время прогона фазы before (для артефакта "
                             "after: обе цифры в provenance — planner [R1])")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "bench"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.real:
        # Режим «реальная модель» (R8). Без источника — честный pending
        # («веса не предоставлены», по образцу R7: env-переключатель без
        # правки кода); явно заданный, но битый источник — громкая ошибка.
        slug = "quality-gates-real"
        try:
            spec = resolve_real_model(model_dir=args.real_model_dir,
                                      desc_json=args.model_desc,
                                      weights=args.model_weights)
        except Exception as e:
            print(f"quality-gates real: ошибка источника: {e}", file=sys.stderr)
            return 1
        if spec is None:
            reason = ("веса не предоставлены: задайте --real-model-dir или env "
                      + ENV_REAL_MODEL_DIR + " (каталог с description.json "
                      "[R4] и model.safetensors); прогон — worker после "
                      "скачивания весов [R3] и hf_to_description [R4]")
            gates, provenance = run_gates_real_pending(reason)
            provenance.update(wall_provenance(
                args.wall_seconds, args.wall_phase, args.wall_before_seconds))
            snapshot = _GatesSnapshot(gates, extra={
                "mode": "real-model-pending",
                "provenance": provenance,
                "pending_reason": reason,
                "protocol": _protocol_snapshot()})
            log_lines = ["quality-gates REAL: pending — " + reason]
        else:
            try:
                gates, provenance = run_gates_real(
                    spec, engine_lib=args.engine_lib,
                    co_tenancy=args.co_tenancy)
            except Exception as e:
                print(f"quality-gates real: прогон не удался: {e}",
                      file=sys.stderr)
                return 1
            provenance.update(wall_provenance(
                args.wall_seconds, args.wall_phase, args.wall_before_seconds))
            snapshot = _GatesSnapshot(gates, extra={
                "mode": "live-real",
                "provenance": provenance,
                "protocol": _protocol_snapshot()})
            log_lines = [
                f"quality-gates REAL: модель={spec.name}, "
                f"веса={spec.weights_path} ({provenance['weights_bytes']} B), "
                f"co-tenancy={provenance['co_tenancy']}",
                "точка регрессии R8: PPL + GSM8K-32 + HumanEval-32 через ABI "
                "(stell_logits); provenance в report.json",
            ]
            if spec.name.startswith("synthetic"):
                log_lines.append(
                    "каталог — синтетический «реалистичный профиль»: "
                    "stell_bench.synthetic_model.build_realistic_profile(dir, "
                    "seed=0x5E8E) — воспроизводим (LCG-веса, эталонный "
                    "tokenizer.json HF); значения — точки регрессии конвейера")
        out_dir = (Path(args.results_dir) if args.results_dir
                   else results_dir_for(slug) if args.committed
                   else results_local_dir_for(slug))
    elif args.live:
        gates = run_gates_live(models=models, engine_lib=args.engine_lib,
                               prefer=args.logits_source)
        snapshot = _GatesSnapshot(gates, extra={
            "mode": "live",
            "models": models,
            "logits_source": sorted({g.to_dict()["notes"][1]
                                     for g in gates}),
            "protocol": _protocol_snapshot(),
            "env": {"co_tenancy": False, "gpu": "none (CPU-local)"},
        })
        log_lines = [
            f"quality-gates LIVE: модели={models}, "
            f"источник логитов={snapshot.extra['logits_source']}",
            "HumanEval-32 — accuracy через песочницу кода ([R4] round 3: "
            "process-sandbox с RLIMIT_CPU/AS/NOFILE, wall 10с, temp-dir, "
            "пустой env; НЕ security-граница — честная пометка в докстринге)",
        ]
        out_dir = (Path(args.results_dir) if args.results_dir
                   else results_dir_for("quality-gates-live") if args.committed
                   else results_local_dir_for("quality-gates-live"))
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
