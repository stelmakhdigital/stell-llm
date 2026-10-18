#!/usr/bin/env python3
# ============================================================================
# Golden против HF (best effort, ADR-09 §4 «эталон против HF-референса»).
#
# Собирает HF LlamaForCausalLM (transformers 5.x, CPU float32) с ТОЧНОЙ
# конфигурацией fixture mini_dense (LlamaConfig из JSON: hidden 8, 2 слоя,
# 2 головы, head_dim 4, GQA kv_heads=1, intermediate 16, tie=False, eps 1e-5),
# загружает веса fixture (tests/models/fixtures/mini_dense.safetensors) и
# прогоняет forward greedy. Результат — второй референс
# tests/models/golden/logits_mini_dense_hf.json (сравнивается с независимым
# Python-референсом logits_mini_dense.json).
#
# ЗАФИКСИРОВАННЫЕ ПАТЧИ (честно, расхождение со «стоковым» HF задокументировано):
#   1. RoPE → identity (fixture ops-схема НЕ содержит позиционной операции —
#      конвенция round 1, см. README golden); реализовано заменой
#      LlamaRotaryEmbedding на IdentityRotary (cos=1, sin=0).
#   2. post_attention_layernorm → nn.Identity (в fixture нет pre-MLP нормы —
#      группы l{layer}.ffn.norm нет; Llama всегда применяет norm2).
#   3. MLP: SwiGLU как в llama-class — gate_proj := fixture ffn.gate,
#      up_proj := fixture ffn.up, down_proj := fixture ffn.down (round 2:
#      в mini_dense добавлена ffn.gate — round-1 «silu_mul без gate» не
#      был определён и точно не представим в HF-MLP).
#   4. head — отдельная группа (tie_word_embeddings=False).
# Всё остальное (embedding, rmsnorm, attention/GQA/scaling 1/sqrt(d_head),
# residual-порядок, финальный norm, lm_head) — стоковый HF Llama.
#
# mini_moe (glm-class): HF MixtralForCausalLM (transformers 5.x) — семантика
# fixture (router softmax-topk + renorm, эксперты SwiGLU gate/up/down,
# mlp.norm перед MoE) совпадает со стоковым Mixtral при тех же патчах:
#   1. RoPE → identity; 2. head — TIED embeddings (в fixture нет группы head).
# Веса NVFP4 — референсные dequant(quantize(F16→f32)) (см. generate_golden.py).
# (Инструкция раунда предполагала «HF для MoE недоступна — pending»; Mixtral
#  в transformers 5.x покрывает семантику fixture, поэтому сделано честно с
#  зафиксированными патчами.)
# ============================================================================
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.normpath(os.path.join(HERE, "..", "fixtures"))
GOLDEN = HERE

sys.path.insert(0, HERE)
from transformers import (LlamaConfig, LlamaForCausalLM,  # noqa: E402
                          MixtralConfig, MixtralForCausalLM)  # noqa: E402
import generate_golden as gen  # noqa: E402  (общие хелперы fixture/interpreter)


class IdentityRotary(torch.nn.Module):
    """RoPE → identity: cos=1, sin=0 (fixture ops не содержат rope — round 1)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, hidden_states, position_ids=None):
        bsz, seq = hidden_states.shape[0], hidden_states.shape[1]
        cos = torch.ones(bsz, seq, self.head_dim, dtype=hidden_states.dtype)
        sin = torch.zeros(bsz, seq, self.head_dim, dtype=hidden_states.dtype)
        return cos, sin


def build_hf_dense(desc, weights, prompt_tokens):
    cfg = LlamaConfig(
        vocab_size=desc["vocab_size"],
        hidden_size=desc["d_model"],
        num_hidden_layers=desc["n_layers"],
        num_attention_heads=desc["n_heads"],
        num_key_value_heads=desc["kv"]["n_kv_heads"],
        intermediate_size=desc["d_model"] * 2,  # fixture ffn: up/down [16, 8]
        tie_word_embeddings=False,
        rms_norm_eps=desc["ops"][0]["params"]["eps"],
        max_position_embeddings=64,
        pad_token_id=None,
    )
    cfg._attn_implementation = "eager"
    model = LlamaForCausalLM(config=cfg)
    model.eval()

    # Патчи: RoPE → identity; post_attention_layernorm → nn.Identity.
    head_dim = desc["d_head"]
    model.model.rotary_emb = IdentityRotary(head_dim)
    for layer in model.model.layers:
        layer.post_attention_layernorm = torch.nn.Identity()

    sd = {}
    t_embed = gen.find_template(desc, "embed")
    sd["model.embed_tokens.weight"] = torch.from_numpy(
        gen.weight_for(desc, weights, t_embed, []))
    for layer in range(desc["n_layers"]):
        lv = [("layer", layer)]
        for hf_name, probe in (("q_proj", "attn.wq"), ("k_proj", "attn.wk"),
                               ("v_proj", "attn.wv"), ("o_proj", "attn.wo")):
            t = gen.find_template(desc, probe)
            sd[f"model.layers.{layer}.self_attn.{hf_name}.weight"] = \
                torch.from_numpy(gen.weight_for(desc, weights, t, lv))
        t = gen.find_template(desc, "attn.norm")
        sd[f"model.layers.{layer}.input_layernorm.weight"] = \
            torch.from_numpy(gen.weight_for(desc, weights, t, lv))
        # MLP (SwiGLU): gate := fixture ffn.gate, up := fixture ffn.up,
        # down := fixture ffn.down (см. шапку).
        t_gate = gen.find_template(desc, "ffn.gate")
        sd[f"model.layers.{layer}.mlp.gate_proj.weight"] = torch.from_numpy(
            gen.weight_for(desc, weights, t_gate, lv))
        t_up = gen.find_template(desc, "ffn.up")
        sd[f"model.layers.{layer}.mlp.up_proj.weight"] = torch.from_numpy(
            gen.weight_for(desc, weights, t_up, lv))
        t_down = gen.find_template(desc, "ffn.down")
        sd[f"model.layers.{layer}.mlp.down_proj.weight"] = \
            torch.from_numpy(gen.weight_for(desc, weights, t_down, lv))
    t_final = gen.find_template(desc, "norm_final")
    sd["model.norm.weight"] = torch.from_numpy(
        gen.weight_for(desc, weights, t_final, []))
    t_head = gen.find_template(desc, "head")
    sd["lm_head.weight"] = torch.from_numpy(
        gen.weight_for(desc, weights, t_head, []))
    model.load_state_dict(sd, strict=True)

    ids = torch.tensor([prompt_tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    return logits[0, -1].to(torch.float32).numpy()


def build_hf_moe(desc, weights, prompt_tokens):
    """HF MixtralForCausalLM с конфигурацией fixture mini_moe; патчи:
    RoPE → identity, head = tied embeddings. Смотреть докстринг модуля."""
    d_ff = desc["moe"]["expert_d_ff"]
    cfg = MixtralConfig(
        vocab_size=desc["vocab_size"],
        hidden_size=desc["d_model"],
        num_hidden_layers=desc["n_layers"],
        num_attention_heads=desc["n_heads"],
        num_key_value_heads=desc["kv"]["n_kv_heads"],
        intermediate_size=d_ff,
        num_local_experts=desc["moe"]["n_experts"],
        num_experts_per_tok=desc["moe"]["n_experts_active"],
        tie_word_embeddings=True,       # в fixture нет группы head
        rms_norm_eps=desc["ops"][0]["params"]["eps"],
        max_position_embeddings=64,
        pad_token_id=None,
    )
    cfg._attn_implementation = "eager"
    model = MixtralForCausalLM(config=cfg)
    model.eval()
    model.model.rotary_emb = IdentityRotary(desc["d_head"])
    if gen.find_template(desc, "mlp.norm") is None:
        # fixture mini_moe не имеет pre-MLP нормы (glm-class builtin имеет
        # l{layer}.mlp.norm — интерпретатор поддерживает оба варианта)
        for layer in model.model.layers:
            layer.post_attention_layernorm = torch.nn.Identity()

    def t(probe):
        return gen.find_template(desc, probe)

    def w(tpl, layer, expert=None):
        vals = []
        if layer is not None:
            vals.append(("layer", layer))
        if expert is not None:
            vals.append(("expert", expert))
        return gen.weight_for(desc, weights, tpl, vals)

    sd = {}
    sd["model.embed_tokens.weight"] = torch.from_numpy(w(t("embed"), None))
    for layer in range(desc["n_layers"]):
        for hf_name, probe in (("q_proj", "attn.wq"), ("k_proj", "attn.wk"),
                               ("v_proj", "attn.wv"), ("o_proj", "attn.wo")):
            sd[f"model.layers.{layer}.self_attn.{hf_name}.weight"] = \
                torch.from_numpy(w(t(probe), layer))
        sd[f"model.layers.{layer}.input_layernorm.weight"] = \
            torch.from_numpy(w(t("attn.norm"), layer))
        t_mlp_norm = t("mlp.norm")
        if t_mlp_norm is not None:
            # mlp.norm → post_attention_layernorm (стоковая семантика Mixtral)
            sd[f"model.layers.{layer}.post_attention_layernorm.weight"] = \
                torch.from_numpy(w(t_mlp_norm, layer))
        sd[f"model.layers.{layer}.mlp.gate.weight"] = \
            torch.from_numpy(w(t("mlp.router"), layer))
        gate_up = torch.empty(desc["moe"]["n_experts"], 2 * d_ff, desc["d_model"])
        down = torch.empty(desc["moe"]["n_experts"], desc["d_model"], d_ff)
        for e in range(desc["moe"]["n_experts"]):
            gate_up[e, :d_ff] = torch.from_numpy(w(t("e{expert}.gate"), layer, e))
            gate_up[e, d_ff:] = torch.from_numpy(w(t("e{expert}.up"), layer, e))
            down[e] = torch.from_numpy(w(t("e{expert}.down"), layer, e))
        sd[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = gate_up
        sd[f"model.layers.{layer}.mlp.experts.down_proj"] = down
    sd["model.norm.weight"] = torch.from_numpy(w(t("norm_final"), None))
    model.load_state_dict(sd, strict=False)  # lm_head — tied

    ids = torch.tensor([prompt_tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    return logits[0, -1].to(torch.float32).numpy()


def write_hf(out, path, extra):
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("OK %s  max_rel=%s" % (path, extra.get("max_rel_err_vs_python_ref")))


def compare_and_write(fx, desc, weights, golden, hf, meta):
    ref = np.array(golden["logits_ref"], dtype=np.float32)
    err = np.abs(ref.astype(np.float64) - hf.astype(np.float64))
    denom = np.maximum(np.maximum(np.abs(ref), np.abs(hf)), 1.0)
    rel = err / denom
    max_rel = float(rel.max())
    print("%s: HF vs Python max_rel_err = %.3e (tolerance %g)" %
          (fx, max_rel, golden["tolerance_rel"]))
    out = {
        "schema": "1.0",
        "model": desc["name"],
        "profile": golden["profile"],
        "path": "loader",
        "prompt_tokens": golden["prompt_tokens"],
        "seed": 0,
        "greedy": True,
        "tolerance_rel": max(1e-4, max_rel * 1.05),
        "logits_ref": [float(v) for v in hf],
        "meta": dict(meta, max_rel_err_vs_python_ref=max_rel,
                     verdict=("сходится с независимым Python-референсом в пороге "
                              if max_rel <= golden["tolerance_rel"]
                              else "РАСХОЖДЕНИЕ выше порога — см. отчёт раунда")),
    }
    write_hf(out, os.path.join(GOLDEN, "logits_%s_hf.json" % fx), out["meta"])
    return 0 if max_rel <= golden["tolerance_rel"] else 1


def main():
    rc = 0
    # --- mini_dense: HF LlamaForCausalLM -------------------------------------
    desc = json.load(open(os.path.join(FIXTURES, "mini_dense.json")))
    weights = gen.read_safetensors(os.path.join(FIXTURES, "mini_dense.safetensors"))
    with open(os.path.join(GOLDEN, "logits_mini_dense.json")) as f:
        golden = json.load(f)
    hf = build_hf_dense(desc, weights, golden["prompt_tokens"])
    rc |= compare_and_write(
        "mini_dense", desc, weights, golden, hf,
        {"implementation": "HF LlamaForCausalLM (transformers 5.x, CPU float32) — "
                           "второй референс (best effort, ADR-09 §4)",
         "config": "LlamaConfig из fixture mini_dense.json (hidden 8, 2 слоя, "
                   "2 головы, head_dim 4, GQA kv=1, intermediate 16, tie=False, "
                   "eps 1e-5)",
         "patches": [
             "RoPE → identity (fixture ops НЕ содержат позиционной операции; "
             "round 1 — конвенция, README golden)",
             "post_attention_layernorm → nn.Identity (в fixture нет pre-MLP нормы)",
             "MLP SwiGLU: gate := fixture ffn.gate, up := fixture ffn.up, "
             "down := fixture ffn.down"],
         "weights": "tests/models/fixtures/mini_dense.safetensors"})

    # --- mini_moe: HF MixtralForCausalLM -------------------------------------
    desc = json.load(open(os.path.join(FIXTURES, "mini_moe.json")))
    weights = gen.read_safetensors(os.path.join(FIXTURES, "mini_moe.safetensors"))
    gen.apply_nvfp4_reference(desc, weights)
    with open(os.path.join(GOLDEN, "logits_mini_moe.json")) as f:
        golden = json.load(f)
    hf = build_hf_moe(desc, weights, golden["prompt_tokens"])
    rc |= compare_and_write(
        "mini_moe", desc, weights, golden, hf,
        {"implementation": "HF MixtralForCausalLM (transformers 5.x, CPU float32) — "
                           "второй референс (best effort, ADR-09 §4)",
         "config": "MixtralConfig из fixture mini_moe.json (hidden 8, 2 слоя, "
                   "2 головы, head_dim 4, kv=1, intermediate 8, 4 эксперта, "
                   "top-2, tie=True — head отсутствует в fixture, eps 1e-5)",
         "patches": [
             "RoPE → identity (fixture ops НЕ содержат позиционной операции)",
             "head = tied embeddings (в fixture нет группы head)",
             "веса NVFP4 — референсные dequant(quantize(F16→f32)) "
             "(см. generate_golden.py)"],
         "weights": "tests/models/fixtures/mini_moe.safetensors"})
    return rc


if __name__ == "__main__":
    sys.exit(main())
