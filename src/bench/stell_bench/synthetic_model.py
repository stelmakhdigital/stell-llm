"""Синтетический каталог модели «реалистичного профиля» (R8, round 4).

Назначение — честная отработка режима «реальная модель» quality-gates runner'а
ДО появления настоящих весов (скачивание — worker, [R3]; hf_to_description —
[R4], поток models): каталог с описанием ADR-09 (description.json), весами
(model.safetensors, HF-style имена по source_pattern) и РЕАЛЬНЫМ
tokenizer.json (эталонный файл HF tokenizers — fixtures/tokenizer/, без сети).

«Реалистичный профиль» = MoE с бОльшим числом слоёв/экспертов, чем mini-фикстуры
(4 слоя × 8 экспертов против 2 × 4), byte-level BPE-токенизатор vocab=350.
Честная пометка: это НЕ языковая модель — синтетические LCG-веса, значения
гейтов = точки регрессии конвейера (как у mini-фикстур), не оценка качества LLM.

Веса генерируются независимым генератором golden-фикстур
(tests/models/golden/generate_golden.py: make_fixture_weights + write_safetensors,
LCG — детерминированно при фиксированном seed), поэтому каталог воспроизводим
и точка регрессии стабильна.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GOLDEN_GEN = _REPO_ROOT / "tests" / "models" / "golden" / "generate_golden.py"
_TOKENIZER_FIXTURE = (_REPO_ROOT / "tests" / "models" / "fixtures" / "tokenizer"
                      / "tokenizer.json")

DEFAULT_SEED = 0x5E8E  # LCG-конвенция golden-генератора (детерминизм)

# Реалистичный профиль: MoE 4 слоя × 8 экспертов, byte-level BPE vocab=350
# (размерности умеренные — прогон гейтов живьём в CPU-тестах; структура —
# слои/эксперты/BPE — важнее абсолютного размера).
REALISTIC_DESC: Dict[str, Any] = {
    "schema": "1.0",
    "name": "synthetic-realistic-moe",
    "class": "moe",
    "dtype_weights": "NVFP4",
    "vocab_size": 350,
    "d_model": 16,
    "n_layers": 4,
    "n_heads": 2,
    "d_head": 8,
    "kv": {"dtype": "F16", "n_kv_heads": 1, "d_kv": 8, "quant": None},
    "moe": {"n_experts": 8, "n_experts_active": 2, "expert_d_ff": 64,
            "router": "softmax-topk"},
    "ops": [
        {"op": "rmsnorm", "io": ["hidden"], "params": {"eps": 1e-5}},
        {"op": "attention", "io": ["hidden", "kv"],
         "params": {"kernel": "sm70_fp32_qk_pv"}},
        {"op": "moe_gather_scatter", "io": ["hidden", "experts"],
         "params": {"kernel": "sm70_moe_gs"}},
    ],
    "tensor_groups": [
        {"name": "embed", "role": "attn", "shape": [350, 16], "dtype": "F16",
         "source_pattern": "embedding.weight"},
        {"name": "norm_final", "role": "attn", "shape": [16], "dtype": "F32",
         "source_pattern": "model.norm.weight"},
        {"name": "l{layer}.attn.norm", "role": "attn", "shape": [16],
         "dtype": "F32", "source_pattern": "layers.{layer}.input_layernorm.weight",
         "cardinality": {"layer": 4}},
        {"name": "l{layer}.attn.wq", "role": "attn", "shape": [16, 16],
         "dtype": "NVFP4", "source_pattern": "layers.{layer}.self_attn.q_proj.weight",
         "cardinality": {"layer": 4}},
        {"name": "l{layer}.attn.wk", "role": "attn", "shape": [8, 16],
         "dtype": "NVFP4", "source_pattern": "layers.{layer}.self_attn.k_proj.weight",
         "cardinality": {"layer": 4}},
        {"name": "l{layer}.attn.wv", "role": "attn", "shape": [8, 16],
         "dtype": "NVFP4", "source_pattern": "layers.{layer}.self_attn.v_proj.weight",
         "cardinality": {"layer": 4}},
        {"name": "l{layer}.attn.wo", "role": "attn", "shape": [16, 16],
         "dtype": "NVFP4", "source_pattern": "layers.{layer}.self_attn.o_proj.weight",
         "cardinality": {"layer": 4}},
        {"name": "l{layer}.kv", "role": "kv", "shape": [8, 16], "dtype": "F16",
         "source_pattern": "layers.{layer}.kv_cache", "kv_block": True,
         "cardinality": {"layer": 4}},
        {"name": "l{layer}.mlp.router", "role": "expert", "shape": [8, 16],
         "dtype": "F32", "source_pattern": "layers.{layer}.mlp.router.weight",
         "cardinality": {"layer": 4}},
        {"name": "l{layer}.e{expert}.gate", "role": "expert", "shape": [32, 16],
         "dtype": "NVFP4",
         "source_pattern": "layers.{layer}.mlp.experts.{expert}.gate.weight",
         "cardinality": {"layer": 4, "expert": 8}},
        {"name": "l{layer}.e{expert}.up", "role": "expert", "shape": [32, 16],
         "dtype": "NVFP4",
         "source_pattern": "layers.{layer}.mlp.experts.{expert}.up.weight",
         "cardinality": {"layer": 4, "expert": 8}},
        {"name": "l{layer}.e{expert}.down", "role": "expert", "shape": [16, 32],
         "dtype": "NVFP4",
         "source_pattern": "layers.{layer}.mlp.experts.{expert}.down.weight",
         "cardinality": {"layer": 4, "expert": 8}},
    ],
}

DESCRIPTION_NAME = "description.json"
WEIGHTS_NAME = "model.safetensors"
TOKENIZER_NAME = "tokenizer.json"


def build_realistic_profile(dest_dir: Path, *, seed: int = DEFAULT_SEED,
                            with_tokenizer: bool = True) -> Dict[str, Any]:
    """Создаёт каталог синтетической модели реалистичного профиля.

    Возвращает сводку {description, weights, tokenizer, weights_bytes}.
    Бросает FileNotFoundError, если эталонный tokenizer.json недоступен
    (тишина запрещена: with_tokenizer=True требует файл).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("stell_golden_generate",
                                                  _GOLDEN_GEN)
    gg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gg)

    desc = dict(REALISTIC_DESC)
    desc["tensor_groups"] = [dict(g) for g in REALISTIC_DESC["tensor_groups"]]
    desc_path = dest_dir / DESCRIPTION_NAME
    desc_path.write_text(json.dumps(desc, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    # Независимый генератор весов golden-фикстур (LCG, детерминизм по seed):
    # тензоры F16/F32 по нативным HF-именам source_pattern (NVFP4-группы —
    # F16-тензора; квантование на read() — лоадер NVFP4W4A16, ADR-03).
    tensors = gg.make_fixture_weights(desc, seed)
    weights_path = dest_dir / WEIGHTS_NAME
    gg.write_safetensors(str(weights_path), tensors)

    tokenizer_path = None
    if with_tokenizer:
        if not _TOKENIZER_FIXTURE.exists():
            raise FileNotFoundError(
                f"эталонный tokenizer.json не найден: {_TOKENIZER_FIXTURE}")
        tokenizer_path = dest_dir / TOKENIZER_NAME
        shutil.copyfile(_TOKENIZER_FIXTURE, tokenizer_path)

    return {
        "description": str(desc_path),
        "weights": str(weights_path),
        "tokenizer": str(tokenizer_path) if tokenizer_path else None,
        "weights_bytes": weights_path.stat().st_size,
        "seed": seed,
    }
