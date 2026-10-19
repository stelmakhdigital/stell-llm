"""Byte-level BPE токенизатор runner'а quality-gates (R8, round 4).

Порт контракта ByteLevelBpeTokenizer (src/models, R10 round 2) на Python:
bpe_tokenizer.py. Эталон — fixtures/tokenizer/tokenizer_ref.json (ид
сгенерированы РЕАЛЬНОЙ эталонной реализацией HF tokenizers 0.23.3 заранее,
offline; C++-токенизатор сверяется с тем же файлом — тест test_models_tokenizer).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stell_bench.bpe_tokenizer import BpeTokenizer, TokenizerError

_ROOT = Path(__file__).resolve().parents[2]
_TOK_DIR = _ROOT / "tests" / "models" / "fixtures" / "tokenizer"


@pytest.fixture(scope="module")
def tok():
    return BpeTokenizer(str(_TOK_DIR / "tokenizer.json"))


@pytest.fixture(scope="module")
def ref():
    return json.loads((_TOK_DIR / "tokenizer_ref.json").read_text())


def test_matches_hf_reference_ids(tok, ref):
    """Расхождение с эталонной реализацией HF tokenizers = 0 на всех 25
    строках фикстуры (тот же эталон, что и для C++-реализации R10)."""
    assert len(ref["strings"]) == 25
    for s, ids in zip(ref["strings"], ref["ids"]):
        assert tok.encode(s) == ids, repr(s)


def test_roundtrip(tok, ref):
    for s in ref["strings"]:
        assert tok.decode_ids(tok.encode(s)) == s


def test_vocab_and_merges(tok, ref):
    assert tok.vocab_size() == ref["meta"]["vocab_size"]
    assert tok.n_merges == ref["meta"]["n_merges"]


def test_empty_text(tok):
    assert tok.encode("") == []
    assert tok.decode_ids([]) == ""


def test_unknown_id_decodes_to_empty(tok):
    assert tok.decode(-1) == ""
    assert tok.decode(10**6) == ""


def test_subset_rejections(tmp_path):
    """Тишина запрещена: unsupported-фичи tokenizer.json — честная ошибка
    (тот же subset, что у C++-реализации R10)."""
    base = json.loads((_TOK_DIR / "tokenizer.json").read_text())

    def write(mutate):
        d = json.loads(json.dumps(base))          # глубокая копия: мутации изолированы
        mutate(d)
        p = tmp_path / "tokenizer.json"
        p.write_text(json.dumps(d))
        return str(p)

    with pytest.raises(TokenizerError, match="ByteLevel"):
        BpeTokenizer(write(
            lambda d: d.__setitem__("pre_tokenizer", {"type": "Regex"})))
    with pytest.raises(TokenizerError, match="use_regex"):
        BpeTokenizer(write(
            lambda d: d["pre_tokenizer"].__setitem__("use_regex", True)))
    with pytest.raises(TokenizerError, match="normalizer"):
        BpeTokenizer(write(
            lambda d: d.__setitem__("normalizer", {"type": "NFC"})))
    with pytest.raises(TokenizerError, match="added"):
        BpeTokenizer(write(
            lambda d: d.__setitem__("added_tokens", [{"id": 999}])))
    with pytest.raises(TokenizerError, match="unk_token"):
        BpeTokenizer(write(
            lambda d: d["model"].__setitem__("unk_token", "[UNK]")))
    with pytest.raises(TokenizerError, match="byte_fallback"):
        BpeTokenizer(write(
            lambda d: d["model"].__setitem__("byte_fallback", True)))
    with pytest.raises(TokenizerError, match="dropout"):
        BpeTokenizer(write(
            lambda d: d["model"].__setitem__("dropout", 0.1)))


def test_missing_file():
    with pytest.raises(OSError):
        BpeTokenizer(str(_TOK_DIR / "no-such-tokenizer.json"))
