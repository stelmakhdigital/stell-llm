#!/usr/bin/env python3
# ============================================================================
# Генератор fixture-эталона byte-level BPE токенизатора (round 2, задача R10).
#
# Требует HF-env (tokenizers + transformers, CPU, БЕЗ сети):
#   ~/.local/stell-conda/envs/hf/bin/python generate_tokenizer_fixture.py
#
# Что делает:
#   1. Строит byte-level BPE токенизатор библиотеки tokenizers:
#        pretokenizer ByteLevel(add_prefix_space=False, use_regex=False)
#        (вся строка — один «чанк», без GPT2-регэкспа — конвенция fixture),
#        decoder ByteLevel, полный алфавит 256 байт-символов
#        (pre_tokenizers.ByteLevel.alphabet()) + merges по фиксированному
#        корпусу (BpeTrainer, детерминирован по корпусу).
#   2. Сохраняет tokenizer.json → tests/models/fixtures/tokenizer/.
#   3. Генерирует эталонные id фиксированного набора ASCII-строк через
#      transformers.PreTrainedTokenizerFast (та же библиотека, что и у
#      прод-контура) → tokenizer_ref.json (schema "1.0").
#
# ЧЕСТНАЯ ПОМЕТКА (отчёт раунда): эталонные id сгенерированы ЭТИМ ЖЕ
# tokenizer.json (committed рядом) через tokenizers 0.23.3 /
# transformers PreTrainedTokenizerFast — т.е. C++-реализация (ByteLevelBpe)
# сверяется с HF tokenizers на полном алфавите 256 + 94 merge'ов.
# Сеть не использовалась; файлы коммитятся, тесты от Python не зависят.
# ============================================================================
import json
import os
import sys

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "fixtures", "tokenizer"))

CORPUS = ("the quick brown fox jumps over the lazy dog. "
          "stell is a serving engine for large language models. "
          "the model runs on a server and serves tokens quickly. "
          "hello world, testing the byte level bpe tokenizer! "
          "attention heads and experts route tokens through layers.")

# Фиксированный набор ASCII-строк (идемпотентность encode→decode + HF-сверка).
STRINGS = [
    "",
    "hello world",
    "The Quick!",
    "  spaces  ",
    "tabs\tand\nnewlines",
    "digits 12345 and symbols #$%&*()",
    "stell-llm serving engine",
    "MiXeD CaSe WoRdS",
    "one",
    "aaaaaaaaaa",
    "the the the the",
    "Attention, Experts; Router: softmax-topk! (yes/no)",
    "x",
    "ALLCAPS",
    "mixtral ffn_gate_exps.weight",
    "\r\n",
    "\\backslash\"quote'",
    "~tilde~ ^caret^",
    "0.5 1.5 2.0 3.14159",
    "ends with space ",
    "the quick brown fox",
    "expert routing across 4 experts",
    "vocab merges encode decode",
    "!!!???...,;:",
    "a",
]


def main():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                 use_regex=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=350, special_tokens=[], show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())  # 256 байт-символов
    tok.train_from_iterator([CORPUS], trainer)

    # Идемпотентность encode→decode на наборе — обязательна ДО коммита.
    for s in STRINGS:
        ids = tok.encode(s).ids
        dec = tok.decode(ids, skip_special_tokens=False)
        assert dec == s, "non-idempotent: %r → %r" % (s, dec)

    os.makedirs(OUT, exist_ok=True)
    tok_path = os.path.join(OUT, "tokenizer.json")
    with open(tok_path, "w") as f:
        json.dump(json.loads(tok.to_str()), f, indent=1, ensure_ascii=False)
        f.write("\n")

    hf = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    ref_ids = [hf.encode(s) for s in STRINGS]
    # Сверка tokenizers ↔ transformers на том же файле (консистентность слоёв).
    for s, ids in zip(STRINGS, ref_ids):
        assert hf.decode(ids) == s, "transformers decode mismatch: %r" % s

    ref = {
        "schema": "1.0",
        "kind": "tokenizer_ref",
        "tokenizer_json": "tests/models/fixtures/tokenizer/tokenizer.json",
        "meta": {
            "generator": "tests/models/golden/generate_tokenizer_fixture.py",
            "tokenizers_version": __import__("tokenizers").__version__,
            "transformers_version": __import__("transformers").__version__,
            "reference_impl": "transformers.PreTrainedTokenizerFast "
                              "(tokenizers BPE, byte-level pretokenizer "
                              "add_prefix_space=false use_regex=false, "
                              "decoder ByteLevel)",
            "vocab_size": tok.get_vocab_size(),
            "n_merges": len(json.loads(tok.to_str())["model"]["merges"]),
            "n_strings": len(STRINGS),
            "note": "Эталонные id сгенерированы ЗАРАНЕЕ в HF-env (offline, "
                    "без сети) и закоммичены; C++-токенизатор сверяется "
                    "с этим файлом. Полный алфавит 256 байт-символов.",
        },
        "strings": STRINGS,
        "ids": ref_ids,
    }
    ref_path = os.path.join(OUT, "tokenizer_ref.json")
    with open(ref_path, "w") as f:
        json.dump(ref, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("OK: %s (vocab=%d, merges=%d)" %
          (tok_path, tok.get_vocab_size(),
           len(json.loads(tok.to_str())["model"]["merges"])))
    print("OK: %s (%d строк)" % (ref_path, len(STRINGS)))


if __name__ == "__main__":
    sys.exit(main())
