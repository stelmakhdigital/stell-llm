#!/usr/bin/env python3
# ============================================================================
# Golden-референсы логитов (ADR-09 §4) — НЕЗАВИСИМАЯ реализация (round 2).
#
# Что делает скрипт:
#   1. Генерирует синтетические safetensors-фикстуры с фиксированными весами:
#        tests/models/fixtures/mini_dense.safetensors
#        tests/models/fixtures/mini_moe.safetensors
#      (формулы весов — детерминированный LCG, см. `make_fixture_weights`;
#       файлы коммитятся в git, чтобы прогоны engine/core-2 читали РОВНО те же
#       веса, что использовал этот референс).
#   2. Исполняет ops-граф из JSON-описания модели (llama-class / glm-class
#      конвенции, fixture-и mini_dense/mini_moe): embedding → rmsnorm →
#      attention (FP32) → ffn/MoE → logits, режим greedy (argmax). Веса
#      NVFP4-групп — quantize→dequant по РЕФЕРЕНСУ quant.hpp (блок 32,
#      E2M1/E4M3) — точный Python-порт src/models quant.cpp.
#   3. Пишет golden-фикстуры по схеме tests/models/golden/logits_ref.schema.json:
#        tests/models/golden/logits_mini_dense.json
#        tests/models/golden/logits_mini_moe.json
#
# СЕМАНТИКА OPS-ГРАФА (контракт раунда, см. golden/README.md):
#   - rmsnorm: y = x * rsqrt(mean(x^2) + eps) * w (HF-Llama RMSNorm);
#     применяется: pre-attention (группа l{layer}.attn.norm), pre-MLP
#     (l{layer}.ffn.norm | l{layer}.mlp.norm — ТОЛЬКО если группа есть в
#     описании), финальный norm_final перед head.
#   - attention: каузальный, масштаб 1/sqrt(d_head); GQA: q-голова h смотрит
#     kv-голову h // (n_heads/n_kv_heads); ПОЗИЦИОННОГО КОДИРОВАНИЯ НЕТ
#     (ops-схема round 1 не содержит rope-операции — конвенция, зафиксированная
#     в README/отчёте раунда); out = Wo @ concat(heads); x += out.
#   - ffn (dense, activation "silu_mul" = SwiGLU, как в llama-class):
#       m = silu(gate@h2) * (up@h2); x += down@m
#       (round 2: в mini_dense ДОБАВЛЕНА группа ffn.gate — round-1 вариант
#        «silu_mul без gate» не представим точно в HF-MLP и не был
#        определён; generic fallback «нет gate → silu(up@h2)» сохранён для
#        общности схемы, fixture-и его не используют).
#   - moe_gather_scatter ("softmax-topk"):
#       h2 = rmsnorm(x, mlp.norm) (если группа есть); r = router@h2;
#       p = softmax(r); top-k = n_experts_active по убыванию p (ничья →
#       меньший индекс); веса ренормализуются p_i / Σ_sel;
#       expert_i (up,down): silu(up_i@h2) → down_i; для glm-class (gate_up
#       [2*d_ff, d_model]) — строки [0:d_ff] = gate, [d_ff:2*d_ff] = up;
#       x += Σ_sel p_i * expert_i(h2).
#   - logits_ref = логиты ПОСЛЕДНЕЙ позиции промпта (prefill, next-token);
#     greedy: токен = argmax(logits) (ничья → меньший индекс).
#
# ТОЧНОСТЬ: всё вычисление в float32 (контракт "attention FP32", ADR-09 §4);
# веса — точные float32 значения (F16 → f32 exact, NVFP4 dequant → f32).
# ============================================================================
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.normpath(os.path.join(HERE, "..", "fixtures"))
GOLDEN = HERE


# ----------------------------------------------------------------------------
# LCG (детерминированные веса; конвенция задокументирована в README).
# ----------------------------------------------------------------------------
class LCG:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def u32(self) -> int:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * (self.u32() / 2**32)


def make_fixture_weights(desc: dict, lcg_seed: int):
    """Тензоры-веса fixture по её JSON-описанию (нативные формулы раунда).

    Возвращает список (name, dtype_str, shape, data_bytes LE) — порядок
    совпадает с порядком tensor_groups описания (KV-группы пропускаются).
    """
    templates = desc["tensor_groups"]

    def expand(text: str, values) -> str:
        out = text
        for key, v in values:
            out = out.replace("{" + key + "}", str(v))
        return out

    # Смешанная система счисления: последний ключ cardinality меняется быстрее
    # всего (как expand_groups в description.cpp).
    def instances(card):
        keys = list(card.items())  # порядок документа
        total = 1
        for _, n in keys:
            total *= n
        out = []
        for it in range(total):
            rem, acc = it, []
            for k in range(len(keys) - 1, -1, -1):
                kk, n = keys[k]
                acc.append((kk, rem % n))
                rem //= n
            acc.reverse()
            out.append(acc)
        return out

    rng = LCG(lcg_seed)
    tensors = []
    for t in templates:
        card = t.get("cardinality") or {}
        for values in instances(card):
            name = expand(t["source_pattern"], values)
            shape = [int(d) for d in t["shape"]]
            n = 1
            for d in shape:
                n *= d
            role = t["role"]
            if role == "kv":
                continue  # KV-кеш весов не имеет (ADR-09 §1)
            dtype_str = "F32" if t["dtype"] == "F32" else "F16"
            if role == "expert" and "{expert}" in t["source_pattern"]:
                # маршрутизатор/эксперты: более широкий диапазон, чтобы
                # топ-k роутинга был содержательным
                lo, hi = (-1.0, 1.0) if "router" in name else (-0.4, 0.4)
            elif "norm" in name:
                lo, hi = 0.5, 1.5  # RMS-нормы — положительные
            else:
                lo, hi = -0.45, 0.45
            w = np.array([rng.uniform(lo, hi) for _ in range(n)], dtype=np.float64)
            if dtype_str == "F16":
                payload = w.astype(np.float16)
                data = payload.tobytes()  # little-endian x86/ARM64
            else:
                data = w.astype(np.float32).tobytes()
            tensors.append((name, dtype_str, shape, data))
    return tensors


# ----------------------------------------------------------------------------
# safetensors writer/reader (совместим с tests/models/synth_writers.hpp и
# с parse_safetensors_header: dtype "F16"/"F32", data_offsets от базы данных).
# ----------------------------------------------------------------------------
def write_safetensors(path: str, tensors) -> None:
    items = []
    begin = 0
    for name, dtype_str, shape, data in tensors:
        end = begin + len(data)
        items.append((name, dtype_str, shape, begin, end, data))
        begin = end
    header = "{" + ",".join(
        json.dumps(name) + ":" + json.dumps({"dtype": dtype_str, "shape": shape,
                                             "data_offsets": [b, e]})
        for (name, dtype_str, shape, b, e, _) in items)
    header += ",\"__metadata__\":{\"stell_golden\":\"round2\"}}"
    pad = header.ljust(((len(header) + 8 + 7) // 8) * 8 - 8, " ")
    while (8 + len(pad)) % 8 != 0:
        pad += " "
    with open(path, "wb") as f:
        f.write(len(pad).to_bytes(8, "little"))
        f.write(pad.encode("ascii"))
        for (_, _, _, _, _, data) in items:
            f.write(data)


def read_safetensors(path: str):
    with open(path, "rb") as f:
        blob = f.read()
    header_len = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8:8 + header_len].decode("ascii"))
    data_base = 8 + header_len
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        raw = blob[data_base + b:data_base + e]
        arr = np.frombuffer(raw, dtype=np.float16 if info["dtype"] == "F16" else np.float32)
        out[name] = arr.astype(np.float32)  # F16 → f32: точно (exact)
    return out


# ----------------------------------------------------------------------------
# Python-порт src/models quant.cpp (блок 32, E2M1 + E4M3) — побитово.
# ----------------------------------------------------------------------------
E2M1_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def round_half_away(v: float) -> int:
    """std::lround: к ближайшему, половинки — прочь от нуля."""
    import decimal
    d = decimal.Decimal(v)
    q = d.quantize(decimal.Decimal(1), rounding=decimal.ROUND_HALF_UP if v >= 0
                   else decimal.ROUND_HALF_UP)
    # Decimal ROUND_HALF_UP == "away from zero" для положительных; для
    # отрицательных тоже away from zero (ROUND_HALF_UP в Decimal = away from zero).
    return int(q)


def f32_to_e4m3(v: float) -> int:
    """f32_to_e4m3: 1 sign, 4 exp (bias 7), 3 mant; max 448; NaN → 0x7F."""
    v32 = np.float32(v)
    if np.isnan(v32):
        return 0x7F
    a = abs(v32)
    sign = bool(np.signbit(v32)) and v32 != np.float32(0.0)
    if a >= np.float32(448.0):
        return (0x80 | 0x7E) if sign else 0x7E
    if a == 0.0:
        return 0
    m32, e = np.frexp(np.float32(a))  # a = m * 2**e, m ∈ [0.5, 1)
    if e + 6 <= 0:
        mant = round_half_away(float(a) * 512.0)  # C++: lround(a * 512.0) — double
        if mant <= 0:
            return 0
        if mant >= 8:
            return (0x80 | 0x08) if sign else 0x08  # min normal 2^-6
        return (0x80 | mant) if sign else mant
    exp_field = e + 6
    if exp_field > 15:
        return (0x80 | 0x7E) if sign else 0x7E
    prod = float(m32) * 8.0  # точно (умножение на 2^3)
    mant = round_half_away(prod)
    if mant >= 8:
        exp_field += 1
        mant = 0
        if exp_field > 15:
            return (0x80 | 0x7E) if sign else 0x7E
    return (0x80 if sign else 0) | (exp_field << 3) | mant


def e4m3_to_f32(b: int) -> float:
    sign = -1.0 if (b & 0x80) else 1.0
    exp_field = (b >> 3) & 0xF
    mant = b & 7
    if exp_field == 15 and mant == 7:
        return float("nan")
    if exp_field == 0:
        return sign * mant / 8.0 * 2.0**-6
    return sign * 2.0**(exp_field - 7) * (1.0 + mant / 8.0)


def e2m1_quantize(scaled: np.float32) -> int:
    idx = 0
    while idx < 7 and scaled >= np.float32(E2M1_BOUNDS[idx]):
        idx += 1
    return idx


def quantize_nvfp4(src: np.ndarray, n: int):
    """Точный порт quantize_nvfp4 (quant.cpp). Возвращает (nibbles, scales)."""
    src = np.ascontiguousarray(src, dtype=np.float32)
    n_blocks = (n + 31) // 32
    nibbles = np.zeros((n + 1) // 2, dtype=np.uint8)
    scales = np.zeros(n_blocks, dtype=np.uint8)
    for blk in range(n_blocks):
        begin = blk * 32
        cnt = min(32, n - begin)
        seg = src[begin:begin + cnt]
        maxabs = np.float32(0.0)
        for x in seg:
            maxabs = max(maxabs, np.float32(abs(x)))
        sb = f32_to_e4m3(maxabs / np.float32(6.0))
        scale = np.float32(e4m3_to_f32(sb))
        scales[blk] = sb
        denom = np.float32(1.0) if scale == np.float32(0.0) else scale
        for i in range(cnt):
            a = np.float32(abs(src[begin + i])) / denom
            a = min(a, np.float32(6.0))
            nib = e2m1_quantize(a)
            if np.signbit(src[begin + i]):
                nib |= 0x8
            byte_i = (begin + i) // 2
            if (begin + i) % 2 == 0:
                nibbles[byte_i] = nib
            else:
                nibbles[byte_i] = (nibbles[byte_i] & 0x0F) | (nib << 4)
    return nibbles, scales


def dequant_nvfp4(nibbles, scales, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    n_blocks = (n + 31) // 32
    for blk in range(n_blocks):
        begin = blk * 32
        cnt = min(32, n - begin)
        scale = np.float32(e4m3_to_f32(int(scales[blk])))
        for i in range(cnt):
            byte = int(nibbles[(begin + i) // 2])
            nib = (byte & 0x0F) if (begin + i) % 2 == 0 else (byte >> 4)
            mag = np.float32(E2M1_TABLE[nib & 7])
            s = -1.0 if (nib & 0x8) else 1.0
            out[begin + i] = np.float32(s) * mag * scale
    return out


def nvfp4_reference_weights(w_f16: np.ndarray) -> np.ndarray:
    """Веса NVFP4-группы = dequant(quantize(F16 → f32)) — референс quant.hpp."""
    src = w_f16.astype(np.float32)
    n = src.size
    nibbles, scales = quantize_nvfp4(src, n)
    return dequant_nvfp4(nibbles, scales, n)



def apply_nvfp4_reference(desc, weights):
    """NVFP4/W4A16 группы: заменяет веса на dequant(quantize(F16→f32)) —
    референс quant.hpp (см. nvfp4_reference_weights)."""
    for t in desc["tensor_groups"]:
        if t["dtype"] not in ("NVFP4", "W4A16"):
            continue
        card = t.get("cardinality") or {}
        keys = list(card.items())
        total = 1
        for _, nn in keys:
            total *= nn
        for it in range(total):
            rem, acc = it, []
            for k in range(len(keys) - 1, -1, -1):
                kk, nn = keys[k]
                acc.append((kk, rem % nn))
                rem //= nn
            acc.reverse()
            pattern = t["source_pattern"]
            for kk, v in acc:
                pattern = pattern.replace("{" + kk + "}", str(v))
            weights[pattern] = nvfp4_reference_weights(weights[pattern])


# ----------------------------------------------------------------------------
# Интерпретатор ops-графа (генерик по JSON-описанию).
# ----------------------------------------------------------------------------
def expand_name(t, values):
    out = t["name"]
    for key, v in values:
        out = out.replace("{" + key + "}", "%03d" % v)
    return out


def find_template(desc, probe):
    for t in desc["tensor_groups"]:
        if probe in t["name"] and not t.get("kv_block"):
            return t
    return None


def weight_for(desc, weights, template, values):
    """values — список пар (key, value); pattern без выравнивания.
    Возвращает тензор, перепрофилированный по shape шаблона (row-major,
    HF-конвенция [out, in]). Round 2 (R9): стековные тензорные группы
    (GGUF-mixtral, source_slice) — срез по СТАРШЕЙ размерности: непрерывный
    блок prod(shape) элементов с номером slice (row-major).
    """
    pattern = template["source_pattern"]
    for key, v in values:
        pattern = pattern.replace("{" + key + "}", str(v))
    arr = weights[pattern]
    shape = tuple(int(d) for d in template["shape"])
    slice_tpl = template.get("source_slice")
    if slice_tpl:
        slice_val = slice_tpl
        for key, v in values:
            slice_val = slice_val.replace("{" + key + "}", str(v))
        idx = int(slice_val)
        n = 1
        for d in shape:
            n *= d
        # Срез непрерывен в ПОСЛЕДОВАТЕЛЬНОЙ памяти (row-major): плоский блок
        # №idx из prod(shape) элементов (старшая размерность — внешняя).
        arr = arr.reshape(-1)[idx * n:(idx + 1) * n]
    return arr.reshape(shape)


def layer_values(t, layer, expert=None):
    card = t.get("cardinality") or {}
    values = []
    if "layer" in card:
        values.append(("layer", layer))
    if expert is not None and "expert" in card:
        values.append(("expert", expert))
    return values


def rmsnorm(x, w, eps):
    var = np.mean(x * x, axis=-1, dtype=np.float32)  # по последней оси
    inv = np.float32(1.0) / np.sqrt(var + eps, dtype=np.float32)
    return x * inv[..., None] * w


def silu(v):
    # silu(z) = z * sigmoid(z) — float32 (согласовано с HF LlamaMLP act_fn)
    return v * (np.float32(1.0) / (np.float32(1.0) + np.exp(-v, dtype=np.float32)))


def attention(x, y, wq, wk, wv, wo, n_heads, n_kv_heads, d_head):
    """x — residual (нормированный ВХОД хранится отдельно), y — вход после
    pre-attention rmsnorm. Возвращает x + Wo @ concat(heads(attn(y)))."""
    T = x.shape[0]
    q = (y @ wq.T).reshape(T, n_heads, d_head)
    k = (y @ wk.T).reshape(T, n_kv_heads, d_head)
    v = (y @ wv.T).reshape(T, n_kv_heads, d_head)
    n_rep = n_heads // n_kv_heads  # GQA: q-голова h → kv-голова h // n_rep
    k_exp = np.repeat(k, n_rep, axis=1)  # interleave: голова h ← kv h//n_rep
    v_exp = np.repeat(v, n_rep, axis=1)
    scale = np.float32(1.0) / np.float32(math.sqrt(d_head))
    scores = np.einsum("thd,shd->ths", q, k_exp, optimize=True) * scale
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # s > t → запрещено
    scores = np.where(mask[:, None, :], np.float32(-np.inf), scores)
    p = softmax_rows(scores)  # softmax по позициям s (в каждой строке (h,t))
    ctx = np.einsum("ths,shd->thd", p, v_exp, optimize=True)
    attn = ctx.reshape(T, n_heads * d_head) @ wo.T
    return x + attn


def ffn_dense(x, w_up, w_gate, w_down, w_norm, eps):
    h2 = rmsnorm(x, w_norm, eps) if w_norm is not None else x
    if w_gate is not None:
        m = silu(h2 @ w_gate.T) * (h2 @ w_up.T)
    else:
        m = silu(h2 @ w_up.T)
    return x + (m @ w_down.T)


def softmax_rows(m):
    m = np.asarray(m, dtype=np.float32)
    e = np.exp(m - m.max(axis=-1, keepdims=True), dtype=np.float32)
    return e / np.sum(e, axis=-1, keepdims=True, dtype=np.float32)


def expert_ffn(h2_vec, e, d_ff):
    """FFN одного эксперта для одного токена: silu(gate·h2)·(up·h2) → down.
    gate_up (glm-class, combined [2*d_ff, d_model]): строки [0:d_ff] = gate,
    [d_ff:2*d_ff] = up; иначе отдельные gate/up; generic fallback: без gate —
    silu(up·h2) → down (mini_moe больше не использует — round 2 добавлен gate)."""
    if e["gate_up"] is not None:
        g = h2_vec @ e["gate_up"][:d_ff].T
        u = h2_vec @ e["gate_up"][d_ff:].T
        return (silu(g) * u) @ e["down"].T
    if e["gate"] is not None:
        g = h2_vec @ e["gate"].T
        u = h2_vec @ e["up"].T
        return (silu(g) * u) @ e["down"].T
    return silu(h2_vec @ e["up"].T) @ e["down"].T  # generic fallback


def moe_block(x, w_norm, w_router, experts, eps, n_active, d_ff, w_bias=None):
    h2 = rmsnorm(x, w_norm, eps) if w_norm is not None else x
    r = h2 @ w_router.T                      # (T, n_experts)
    if w_bias is not None:
        # Round 2 (R9): смещение роутера (нативный GGUF-тензор exp_probs_b).
        r = r + w_bias
    p = softmax_rows(r)                      # softmax-topk (ADR-09: moe.router)
    E = p.shape[1]
    k = min(n_active, E)
    # top-k по убыванию, ничья → меньший индекс (stable argsort по -p)
    topk = np.argsort(-p, axis=1, kind="stable")[:, :k]
    pk = p[np.arange(p.shape[0])[:, None], topk]  # (T, k)
    probs = pk / np.sum(pk, axis=1, keepdims=True, dtype=np.float32)
    out = np.zeros_like(h2)
    for t in range(p.shape[0]):
        for j in range(k):
            idx = int(topk[t, j])
            out[t] = out[t] + probs[t, j] * expert_ffn(h2[t], experts[idx], d_ff)
    return x + out


def interpret(desc, weights, prompt_tokens):
    ops = desc["ops"]
    eps = None
    for op in ops:
        if op["op"] == "rmsnorm":
            eps = np.float32(op["params"]["eps"])
            break
    assert eps is not None, "rmsnorm op обязателен (ADR-09 §1)"

    vocab = desc["vocab_size"]
    d_model = desc["d_model"]
    n_layers = desc["n_layers"]
    n_heads = desc["n_heads"]
    d_head = desc["d_head"]
    n_kv_heads = desc["kv"]["n_kv_heads"]

    embed2d = weight_for(desc, weights, find_template(desc, "embed"), [])
    x = np.stack([embed2d[t] for t in prompt_tokens])
    for layer in range(n_layers):
        t_norm = find_template(desc, "attn.norm")
        if t_norm is None:
            raise RuntimeError("pre-attention rmsnorm группа обязательна")
        w_norm = weight_for(desc, weights, t_norm, layer_values(t_norm, layer))
        y = rmsnorm(x, w_norm, eps)
        wq = weight_for(desc, weights, find_template(desc, "attn.wq"), [("layer", layer)])
        wk = weight_for(desc, weights, find_template(desc, "attn.wk"), [("layer", layer)])
        wv = weight_for(desc, weights, find_template(desc, "attn.wv"), [("layer", layer)])
        wo = weight_for(desc, weights, find_template(desc, "attn.wo"), [("layer", layer)])
        x = attention(x, y, wq, wk, wv, wo, n_heads, n_kv_heads, d_head)

        if desc["class"] == "moe":
            moe = desc["moe"]
            # Pre-MoE норма: mlp.norm (glm-class) ИЛИ ffn.norm (нативная
            # раскладка GGUF-mixtral, R9); отсутствие обеих — без нормы
            # (mini_moe). Смещение роутера — router_bias (exp_probs_b, R9).
            t_mlp_norm = find_template(desc, "mlp.norm") or find_template(desc, "ffn.norm")
            w_mlp_norm = (weight_for(desc, weights, t_mlp_norm, [("layer", layer)])
                          if t_mlp_norm else None)
            t_router = find_template(desc, "mlp.router")
            w_router = weight_for(desc, weights, t_router, [("layer", layer)])
            t_bias = find_template(desc, "router_bias")
            w_bias = (weight_for(desc, weights, t_bias, [("layer", layer)])
                      if t_bias else None)
            experts = []
            t_exp = None
            for t in desc["tensor_groups"]:
                if "e{expert}" in t["name"]:
                    if t_exp is None:
                        t_exp = t
                    break
            card = t_exp.get("cardinality") or {}
            n_experts = moe["n_experts"]
            for e in range(n_experts):
                gate_up = None
                gate = None
                up = None
                down = None
                for t in desc["tensor_groups"]:
                    if "e{expert}" not in t["name"]:
                        continue
                    w = weight_for(desc, weights, t, [("layer", layer), ("expert", e)])
                    if ".gate_up" in t["name"]:
                        gate_up = w
                    elif ".up" in t["name"]:
                        up = w
                    elif ".gate" in t["name"]:
                        gate = w
                    elif ".down" in t["name"]:
                        down = w
                experts.append({"gate_up": gate_up, "gate": gate,
                                "up": up, "down": down})
            x = moe_block(x, w_mlp_norm, w_router, experts, eps,
                          moe["n_experts_active"], moe["expert_d_ff"],
                          w_bias=w_bias)
        else:
            t_ffn_norm = find_template(desc, "ffn.norm")
            w_ffn_norm = (weight_for(desc, weights, t_ffn_norm, [("layer", layer)])
                          if t_ffn_norm else None)
            t_up = find_template(desc, "ffn.up")
            w_up = weight_for(desc, weights, t_up, [("layer", layer)])
            t_gate = find_template(desc, "ffn.gate")
            w_gate = (weight_for(desc, weights, t_gate, [("layer", layer)])
                      if t_gate else None)
            t_down = find_template(desc, "ffn.down")
            w_down = weight_for(desc, weights, t_down, [("layer", layer)])
            x = ffn_dense(x, w_up, w_gate, w_down, w_ffn_norm, eps)

    t_final = find_template(desc, "norm_final")
    w_final = weight_for(desc, weights, t_final, [])
    t_head = find_template(desc, "head")
    if t_head is None:
        # Группы head нет (mini_moe) → конвенция tied embeddings (HF: tie_word
        # _embeddings): head = embed. Зафиксировано в README golden.
        w_head = weight_for(desc, weights, find_template(desc, "embed"), [])
    else:
        w_head = weight_for(desc, weights, t_head, [])
    y = rmsnorm(x[-1], w_final, eps)
    return y @ w_head.T


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Round 2 (R9): mixtral-класс GGUF — синтетический fixture + golden-референс.
#
# Раскладка (нативные имена тензоров формата GGUF, см. ADR-09 «Уточнение»):
#   token_embd.weight, output_norm.weight, output.weight,
#   blk.{l}.attn_norm/ffn_norm.weight, blk.{l}.attn_q/k/v/output.weight,
#   blk.{l}.ffn_gate_inp.weight (роутер), blk.{l}.exp_probs_b.weight
#   (смещение роутера), стековные blk.{l}.ffn_gate_exps/ffn_up_exps/
#   ffn_down_exps.weight [n_experts, d_ff, d_model] (срезы экспертов).
# Метаданные: general.architecture = "mixtral", llama.expert_count (n_experts)
# и llama.expert_used_count (n_experts_active) — из GGUF-метаданных.
# ----------------------------------------------------------------------------
GGML_F32, GGML_F16 = 0, 1
GGUF_T_UINT32, GGUF_T_FLOAT32, GGUF_T_STRING = 4, 8, 11


def _gguf_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return len(b).to_bytes(8, "little") + b


def write_gguf(path: str, tensors, metadata, alignment: int = 32) -> None:
    """GGUF v3 subset (F32/F16), согласован с parse_gguf_header/loader:
    tensors — список (name, ggml_code, dims_GGUF-порядка, payload);
    metadata — список (key, type, value)."""
    meta_blob = b""
    for key, mtype, value in metadata:
        meta_blob += _gguf_string(key) + mtype.to_bytes(4, "little")
        if mtype == GGUF_T_STRING:
            meta_blob += _gguf_string(value)
        elif mtype in (GGUF_T_UINT32, GGUF_T_FLOAT32):
            import struct
            meta_blob += (struct.pack("<I", value) if mtype == GGUF_T_UINT32
                          else struct.pack("<f", value))
        else:
            raise RuntimeError("unsupported meta type")
    infos = b""
    cursor = 0
    for name, code, dims, payload in tensors:
        infos += _gguf_string(name)
        infos += len(dims).to_bytes(4, "little")
        for d in dims:
            infos += int(d).to_bytes(8, "little")
        infos += code.to_bytes(4, "little")
        infos += cursor.to_bytes(8, "little")
        cursor += len(payload)
    base_raw = 4 + 4 + 8 + 8 + len(meta_blob) + len(infos)
    base = (base_raw + alignment - 1) // alignment * alignment
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write((3).to_bytes(4, "little"))          # version
        f.write(len(tensors).to_bytes(8, "little"))  # n_tensors
        f.write(len(metadata).to_bytes(8, "little"))  # n_kv
        f.write(meta_blob)
        f.write(infos)
        for _ in range(base - base_raw):
            f.write(b"\x00")
        for (_, _, _, payload) in tensors:
            f.write(payload)


def read_gguf_f32(path: str):
    """Обратный разбор GGUF v3 (subset F32/F16) → {name: np-f32 (logical shape)}."""
    with open(path, "rb") as f:
        blob = f.read()
    assert blob[:4] == b"GGUF"
    pos = 4
    def u32():
        nonlocal pos
        v = int.from_bytes(blob[pos:pos + 4], "little"); pos += 4; return v
    def u64():
        nonlocal pos
        v = int.from_bytes(blob[pos:pos + 8], "little"); pos += 8; return v
    def gstr():
        nonlocal pos
        n = u64(); s = blob[pos:pos + n].decode("utf-8"); pos += n; return s
    version = u32()
    assert version == 3, version
    n_tensors, n_kv = u64(), u64()
    for _ in range(n_kv):
        key = gstr(); mtype = u32()
        if mtype == GGUF_T_STRING:
            gstr()
        elif mtype in (GGUF_T_UINT32, GGUF_T_FLOAT32):
            pos += 4
        else:
            raise RuntimeError("unsupported meta type %s" % mtype)
    pending = []
    for _ in range(n_tensors):
        name = gstr(); nd = u32(); dims = [u64() for _ in range(nd)]
        code = u32(); off = u64()
        pending.append((name, code, dims, off))
    base = (pos + 31) // 32 * 32
    out = {}
    for name, code, dims, off in pending:
        logical = list(dims)[::-1]  # GGUF-порядок {in, out} → logical [out, in]
        n = 1
        for d in logical:
            n *= d
        raw = blob[base + off:base + off + (n * (2 if code == GGML_F16 else 4))]
        arr = (np.frombuffer(raw, dtype=np.float16) if code == GGML_F16
               else np.frombuffer(raw, dtype=np.float32))
        out[name] = arr.astype(np.float32).reshape(logical)
    return out


# Описание mini_mixtral (1:1 с выводом конвертера gguf_to_description;
# committed fixture fixtures/mini_mixtral.json — побайтово этот же JSON).
MIXTRAL_DESC = {
    "schema": "1.0", "name": "mini-mixtral-gguf", "class": "moe",
    "dtype_weights": "F16", "vocab_size": 16, "d_model": 8, "n_layers": 2,
    "n_heads": 2, "d_head": 4,
    "kv": {"dtype": "F16", "n_kv_heads": 1, "d_kv": 4, "quant": None},
    "moe": {"n_experts": 4, "n_experts_active": 2, "expert_d_ff": 16,
            "router": "softmax-topk"},
    "ops": [
        {"op": "rmsnorm", "io": ["hidden"], "params": {"eps": 1e-05}},
        {"op": "attention", "io": ["hidden", "kv"],
         "params": {"kernel": "sm70_fp32_qk_pv"}},
        {"op": "moe_gather_scatter", "io": ["hidden", "experts"],
         "params": {"kernel": "sm70_moe_gs"}},
    ],
    "tensor_groups": [],  # заполняется ниже (make_mixtral_fixture)
}


def mixtral_group_templates():
    """Шаблоны групп mixtral (нативная GGUF-раскладка; срезы экспертов)."""
    return [
        {"name": "embed", "role": "attn", "shape": [16, 8], "dtype": "F32",
         "source_pattern": "token_embd.weight"},
        {"name": "norm_final", "role": "attn", "shape": [8], "dtype": "F32",
         "source_pattern": "output_norm.weight"},
        {"name": "head", "role": "attn", "shape": [16, 8], "dtype": "F16",
         "source_pattern": "output.weight"},
        {"name": "l{layer}.attn.norm", "role": "attn", "shape": [8], "dtype": "F32",
         "source_pattern": "blk.{layer}.attn_norm.weight",
         "cardinality": {"layer": 2}},
        {"name": "l{layer}.ffn.norm", "role": "attn", "shape": [8], "dtype": "F32",
         "source_pattern": "blk.{layer}.ffn_norm.weight",
         "cardinality": {"layer": 2}},
        {"name": "l{layer}.attn.wq", "role": "attn", "shape": [8, 8], "dtype": "F16",
         "source_pattern": "blk.{layer}.attn_q.weight", "cardinality": {"layer": 2}},
        {"name": "l{layer}.attn.wk", "role": "attn", "shape": [4, 8], "dtype": "F16",
         "source_pattern": "blk.{layer}.attn_k.weight", "cardinality": {"layer": 2}},
        {"name": "l{layer}.attn.wv", "role": "attn", "shape": [4, 8], "dtype": "F16",
         "source_pattern": "blk.{layer}.attn_v.weight", "cardinality": {"layer": 2}},
        {"name": "l{layer}.attn.wo", "role": "attn", "shape": [8, 8], "dtype": "F16",
         "source_pattern": "blk.{layer}.attn_output.weight", "cardinality": {"layer": 2}},
        {"name": "l{layer}.kv", "role": "kv", "shape": [8, 4], "dtype": "F16",
         "source_pattern": "blk.{layer}.kv_cache", "kv_block": True,
         "cardinality": {"layer": 2}},
        {"name": "l{layer}.mlp.router", "role": "expert", "shape": [4, 8],
         "dtype": "F32", "source_pattern": "blk.{layer}.ffn_gate_inp.weight",
         "cardinality": {"layer": 2}},
        {"name": "l{layer}.mlp.router_bias", "role": "expert", "shape": [4],
         "dtype": "F32", "source_pattern": "blk.{layer}.exp_probs_b.weight",
         "cardinality": {"layer": 2}},
        {"name": "l{layer}.e{expert}.gate", "role": "expert", "shape": [16, 8],
         "dtype": "F16", "source_pattern": "blk.{layer}.ffn_gate_exps.weight",
         "source_slice": "{expert}", "cardinality": {"layer": 2, "expert": 4}},
        {"name": "l{layer}.e{expert}.up", "role": "expert", "shape": [16, 8],
         "dtype": "F16", "source_pattern": "blk.{layer}.ffn_up_exps.weight",
         "source_slice": "{expert}", "cardinality": {"layer": 2, "expert": 4}},
        {"name": "l{layer}.e{expert}.down", "role": "expert", "shape": [8, 16],
         "dtype": "F16", "source_pattern": "blk.{layer}.ffn_down_exps.weight",
         "source_slice": "{expert}", "cardinality": {"layer": 2, "expert": 4}},
    ]


def make_mixtral_fixture(seed: int):
    """Синтетический GGUF mixtral-профиля: F32/F16 тензора по нативным именам.
    Возвращает (список GGUF-тензоров, веса-массивы f32 по именам).
    Порядок LCG-отрисовки: слои → перечисленные тензора → эксперты.
    Стековные тензоры: блоки экспертов подряд (slice = leading dim).
    """
    rng = LCG(seed)
    dims2 = lambda a, b: (b, a)  # GGUF-порядок {in, out} → logical [out, in]
    tensors = []
    weights = {}

    def emit(name, ggml_code, logical_shape, lo, hi):
        n = 1
        for d in logical_shape:
            n *= d
        w = np.array([rng.uniform(lo, hi) for _ in range(n)], dtype=np.float64)
        if ggml_code == GGML_F16:
            w = w.astype(np.float16)          # F16-округление — ЭТАЛОН весов
            payload = w.tobytes()
            weights[name] = w.astype(np.float32).reshape(logical_shape)
        else:
            payload = w.astype(np.float32).tobytes()
            weights[name] = w.astype(np.float32).reshape(logical_shape)
        dims = list(logical_shape)[::-1]      # GGUF-порядок {in, out, ...}
        tensors.append((name, ggml_code, dims, payload))

    emit("token_embd.weight", GGML_F32, (16, 8), -0.45, 0.45)
    emit("output_norm.weight", GGML_F32, (8,), 0.5, 1.5)
    emit("output.weight", GGML_F16, (16, 8), -0.45, 0.45)
    for layer in range(2):
        p = "blk.%d." % layer
        emit(p + "attn_norm.weight", GGML_F32, (8,), 0.5, 1.5)
        emit(p + "ffn_norm.weight", GGML_F32, (8,), 0.5, 1.5)
        emit(p + "attn_q.weight", GGML_F16, (8, 8), -0.45, 0.45)
        emit(p + "attn_k.weight", GGML_F16, (4, 8), -0.45, 0.45)
        emit(p + "attn_v.weight", GGML_F16, (4, 8), -0.45, 0.45)
        emit(p + "attn_output.weight", GGML_F16, (8, 8), -0.45, 0.45)
        emit(p + "ffn_gate_inp.weight", GGML_F32, (4, 8), -1.0, 1.0)   # роутер
        emit(p + "exp_probs_b.weight", GGML_F32, (4,), -0.5, 0.5)      # смещение
        # Стековные экспертные тензоры: [n_experts, d_ff, d_model] — блоки
        # экспертов подряд по старшей размерности (slice = expert).
        for name, shape, code in [("ffn_gate_exps.weight", (4, 16, 8), GGML_F16),
                                  ("ffn_up_exps.weight", (4, 16, 8), GGML_F16),
                                  ("ffn_down_exps.weight", (4, 8, 16), GGML_F16)]:
            emit(p + name, code, shape, -0.4, 0.4)
    return tensors, weights


def interpret_mixtral(weights, prompt_tokens):
    """Референс mixtral: интерпретатор ops-графа по MIXTRAL_DESC (срезы
    экспертов — weight_for, смещение роутера — moe_block)."""
    desc = dict(MIXTRAL_DESC)
    desc["tensor_groups"] = mixtral_group_templates()
    return interpret(desc, weights, prompt_tokens)


def main():
    fixtures = ["mini_dense", "mini_moe"]
    prompts = [1, 2, 3]
    seeds = {"mini_dense": 0x00D1, "mini_moe": 0x00D2}

    for fx in fixtures:
        desc = json.load(open(os.path.join(FIXTURES, fx + ".json")))
        tensors = make_fixture_weights(desc, seeds[fx])
        st_path = os.path.join(FIXTURES, fx + ".safetensors")
        write_safetensors(st_path, tensors)

        weights = read_safetensors(st_path)
        # NVFP4/W4A16 группы: референсные веса = dequant(quantize(F16 → f32)).
        apply_nvfp4_reference(desc, weights)

        logits = interpret(desc, weights, prompts)
        tok = int(np.argmax(logits))  # greedy: ничья → меньший индекс

        profile = desc["dtype_weights"]
        # Порог (ADR-09 §4): F16 → 1e-3; NVFP4 → тоже 1e-3, т.к. квантование
        # побитно совпадает с референсом quant.cpp (сверено в раунде) и
        # HF-кроссчек даёт max_rel ~1.5e-7; калибровка зафиксирована в README.
        tolerance_rel = 0.001
        golden = {
            "schema": "1.0",
            "model": desc["name"],
            "profile": profile,
            "path": "loader",
            "prompt_tokens": prompts,
            "seed": 0,
            "greedy": True,
            "tolerance_rel": tolerance_rel,
            "logits_ref": [float(v) for v in logits],
            "meta": {
                "generator": "tests/models/golden/generate_golden.py "
                             "(независимый Python/torch-free интерпретатор ops-графа, "
                             "numpy float32; ADR-09 §4, режим greedy)",
                "weights": fx + ".safetensors (синтетические, LCG-конвенция — README)",
                "quant_note": ("NVFP4-веса = dequant(quantize(F16→f32)) по референсу "
                               "src/models quant.hpp (блок 32, E2M1+E4M3), порт "
                               "quant.cpp в generate_golden.py")
                if profile == "NVFP4"
                else "веса F16 → float32 exact, вычисление FP32",
                "ops_semantics": "см. tests/models/golden/README.md (контракт раунда)",
                "greedy_token": tok,
                "logits_min": float(np.min(logits)),
                "logits_max": float(np.max(logits)),
            },
        }
        out = os.path.join(GOLDEN, "logits_%s.json" % fx)
        with open(out, "w") as f:
            json.dump(golden, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("OK %s: logits_ref=%d значений, greedy_token=%d" %
              (out, len(golden["logits_ref"]), tok))

    # --- mixtral-класс GGUF (R9): синтетический GGUF + golden-референс ------
    gguf_path = os.path.join(FIXTURES, "mini_mixtral.gguf")
    tensors, weights = make_mixtral_fixture(0x00D3)
    metadata = [
        ("general.architecture", GGUF_T_STRING, "mixtral"),
        ("general.name", GGUF_T_STRING, "mini-mixtral-gguf"),
        ("llama.block_count", GGUF_T_UINT32, 2),
        ("llama.attention.head_count", GGUF_T_UINT32, 2),
        ("llama.attention.head_count_kv", GGUF_T_UINT32, 1),
        ("llama.attention.key_length", GGUF_T_UINT32, 4),
        ("llama.embedding_length", GGUF_T_UINT32, 8),
        ("llama.attention.layer_norm_rms_epsilon", GGUF_T_FLOAT32, 1e-5),
        ("llama.vocab_size", GGUF_T_UINT32, 16),
        ("llama.expert_count", GGUF_T_UINT32, 4),
        ("llama.expert_used_count", GGUF_T_UINT32, 2),
    ]
    write_gguf(gguf_path, tensors, metadata)

    # Интерпретатор — по ВЕСАМ, ПРОЧИТАННЫМ ИЗ ФАЙЛА (писатель проверяется).
    weights_back = read_gguf_f32(gguf_path)
    assert set(weights_back) == set(weights), "GGUF readback mismatch"
    for name in weights_back:
        assert np.array_equal(weights_back[name], weights[name]),\
            "GGUF readback values mismatch: %s" % name

    logits = interpret_mixtral(weights_back, prompts)
    tok = int(np.argmax(logits))
    golden = {
        "schema": "1.0",
        "model": "mini-mixtral-gguf",
        "profile": "F16",
        "path": "loader",
        "prompt_tokens": prompts,
        "seed": 0,
        "greedy": True,
        "tolerance_rel": 0.001,
        "logits_ref": [float(v) for v in logits],
        "meta": {
            "generator": "tests/models/golden/generate_golden.py (mixtral-класс "
                         "GGUF, R9: стековные экспертные тензоры + смещение "
                         "роутера exp_probs_b; numpy float32, greedy)",
            "weights": "mini_mixtral.gguf (синтетический, LCG seed 0x00D3 — "
                       "README; веса интерпретатора прочитаны ИЗ файла)",
            "quant_note": "веса F16 → float32 exact, вычисление FP32",
            "ops_semantics": "см. tests/models/golden/README.md; mixtral: "
                             "pre-MoE норма = blk.ffn_norm, роутер = ffn_gate_inp "
                             "+ exp_probs_b, softmax → top-2 → renorm",
            "greedy_token": tok,
            "logits_min": float(np.min(logits)),
            "logits_max": float(np.max(logits)),
        },
    }
    out = os.path.join(GOLDEN, "logits_mini_mixtral.json")
    with open(out, "w") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("OK %s: logits_ref=%d значений, greedy_token=%d" %
          (out, len(golden["logits_ref"]), tok))


if __name__ == "__main__":
    sys.exit(main())
