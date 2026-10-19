"""Byte-level BPE токенизатор для quality-gates runner'а (R8, round 4).

Порт контракта ByteLevelBpeTokenizer (src/models, задача R10 round 2) на
Python — bench-сторонe нужен encode/decode, когда в каталоге реальной модели
есть tokenizer.json (формат эталонных токенизаторов; subset — ЧИСТЫЙ
byte-level BPE, тот же, что у C++-реализации):

  - byte-level pretokenizer БЕЗ регэкспа (use_regex=false): вся строка —
    один чанк; байты UTF-8 маппятся в алфавитные символы по таблице
    byte→unicode (конвенция GPT-2-семейства: печатные диапазоны '!'..'~',
    '¡'..'¬', '®'..'ÿ' — как есть; остальные 68 байтов — кодовые точки
    256+n по возрастанию байта);
  - merges: жадное слияние по рангу — на каждом шаге выбирается пара
    с МИНИМАЛЬНЫМ рангом среди смежных, сливаются ВСЕ её вхождения
    слева направо (семантика классического BPE, сверено с эталонной
    реализацией в tests/models/test_tokenizer.cpp);
  - decode: id → token-строка → обратная маппинга символов в байты →
    UTF-8 строка; неизвестный id → пустая строка (конвенция byte-fallback
    ByteTokenizer в engine).

Не поддержано (честный TokenizerError, тишина запрещена — как в C++):
нормализаторы, post-processor'ы, added/special tokens, unk/byte_fallback,
dropout, regex-претокенизация.

Эталон: tests/models/fixtures/tokenizer/ (tokenizer.json + tokenizer_ref.json
с ид reference-реализации HF tokenizers 0.23.3) — тест
tests/serve/test_bpe_tokenizer.py сверяет расхождение = 0.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple


class TokenizerError(Exception):
    """Ошибка загрузки/использования tokenizer.json (subset не соблюдён)."""


def _is_printable_range(b: int) -> bool:
    """Печатные диапазоны byte-level алфавита (GPT-2-конвенция)."""
    return (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255)


# Ленивая таблица byte (0..255) → юникод-символ алфавита (кодовая точка ≤ 255
# для печатных, иначе 256+n; n — число непечатных байтов меньше данного).
_BYTE_TO_CHAR: Optional[List[str]] = None


def _byte_to_char_table() -> List[str]:
    global _BYTE_TO_CHAR
    if _BYTE_TO_CHAR is None:
        table: List[str] = []
        for b in range(256):
            if _is_printable_range(b):
                table.append(chr(b))
            else:
                n = sum(1 for prev in range(b) if not _is_printable_range(prev))
                table.append(chr(256 + n))
        _BYTE_TO_CHAR = table
    return _BYTE_TO_CHAR


def _char_to_byte(cp: int) -> int:
    """Обратная маппинга кодовой точки → байт (−1 = символ вне алфавита)."""
    if cp <= 255 and _is_printable_range(cp):
        return cp
    if 256 <= cp < 256 + 68:
        n = cp - 256
        for b in range(256):
            if not _is_printable_range(b):
                if n == 0:
                    return b
                n -= 1
    return -1


class BpeTokenizer:
    """Чистый byte-level BPE (subset R10) — encode/decode по tokenizer.json."""

    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            try:
                root = json.load(f)
            except json.JSONDecodeError as e:
                raise TokenizerError(f"tokenizer.json: parse error: {e}") from e
        if not isinstance(root, dict):
            raise TokenizerError("tokenizer.json: root must be an object")

        # Subset-проверки (тишина запрещена — как в C++-реализации).
        pt = root.get("pre_tokenizer")
        if not isinstance(pt, dict) or pt.get("type") != "ByteLevel":
            raise TokenizerError(
                "tokenizer.json: поддержан только ByteLevel pretokenizer "
                "(use_regex=false — конвенция fixture, R10)")
        if pt.get("use_regex"):
            raise TokenizerError(
                "tokenizer.json: ByteLevel use_regex=true не поддержан "
                "(regex-сплиты — вне subset R10)")
        dec = root.get("decoder")
        if not isinstance(dec, dict) or dec.get("type") != "ByteLevel":
            raise TokenizerError("tokenizer.json: поддержан только ByteLevel decoder")
        if root.get("normalizer") is not None:
            raise TokenizerError("tokenizer.json: normalizer не поддержан")
        added = root.get("added_tokens")
        if added:  # не None и не пустой список
            raise TokenizerError(
                "tokenizer.json: added/special tokens не поддержаны "
                "(subset R10: чистый byte-level BPE)")

        model = root.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise TokenizerError("tokenizer.json: model.type != BPE")
        if model.get("unk_token") is not None:
            raise TokenizerError("tokenizer.json: unk_token не поддержан")
        if model.get("byte_fallback"):
            raise TokenizerError("tokenizer.json: byte_fallback не поддержан")
        if model.get("dropout"):
            raise TokenizerError("tokenizer.json: BPE dropout не поддержан")

        vocab_raw = model.get("vocab")
        if not isinstance(vocab_raw, dict) or not vocab_raw:
            raise TokenizerError("tokenizer.json: пустой/отсутствующий vocab")
        max_id = -1
        for token, v in vocab_raw.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise TokenizerError("tokenizer.json: vocab id must be int >= 0")
            max_id = max(max_id, v)
        self._vocab: Dict[str, int] = {}
        self._id2token: List[str] = [""] * (max_id + 1)
        for token, v in vocab_raw.items():
            if token in self._vocab:
                raise TokenizerError("tokenizer.json: дубликат токена " + repr(token))
            self._vocab[token] = v
            if self._id2token[v]:
                raise TokenizerError(f"tokenizer.json: дубликат vocab id {v}")
            self._id2token[v] = token

        self._ranks: Dict[Tuple[str, str], int] = {}
        merges_raw = model.get("merges")
        if not isinstance(merges_raw, list):
            raise TokenizerError("tokenizer.json: отсутствует/битый merges")
        for i, m in enumerate(merges_raw):
            if (isinstance_tuple(m) and len(m) == 2
                    and isinstance(m[0], str) and isinstance(m[1], str)):
                pair = (m[0], m[1])
            elif isinstance(m, str):
                # legacy-формат "a b" (пробел в byte-level алфавите невозможен:
                # байт 0x20 маппится в символ).
                sp = m.find(" ")
                if sp < 0:
                    raise TokenizerError(f"tokenizer.json: bad merge entry #{i}")
                pair = (m[:sp], m[sp + 1:])
            else:
                raise TokenizerError(f"tokenizer.json: bad merge entry type #{i}")
            if pair in self._ranks:
                raise TokenizerError(f"tokenizer.json: дубликат merge #{i}")
            self._ranks[pair] = len(self._ranks)

    # ------------------------------------------------------------------ encode

    def encode(self, text: str) -> List[int]:
        """Текст → id (byte-level BPE, один чанк — use_regex=false)."""
        if not text:
            return []
        table = _byte_to_char_table()
        raw = text.encode("utf-8")
        symbols = [table[b] for b in raw]
        self._merge_symbols(symbols)
        ids: List[int] = []
        for sym in symbols:
            tid = self._vocab.get(sym)
            if tid is None:
                raise TokenizerError(
                    "encode: символ " + repr(sym) +
                    " отсутствует в vocab (битый tokenizer.json?)")
            ids.append(tid)
        return ids

    def _merge_symbols(self, symbols: List[str]) -> None:
        n_ranks = len(self._ranks)
        while len(symbols) >= 2:
            # Пара с минимальным рангом среди смежных (ранги уникальны).
            best = n_ranks
            best_pos = 0
            for i in range(len(symbols) - 1):
                r = self._ranks.get((symbols[i], symbols[i + 1]))
                if r is not None and r < best:
                    best = r
                    best_pos = i
            if best == n_ranks:
                break
            first, second = symbols[best_pos], symbols[best_pos + 1]
            # Слияние ВСЕХ вхождений выбранной пары слева направо.
            out: List[str] = []
            i = 0
            while i < len(symbols):
                if (i + 1 < len(symbols)
                        and symbols[i] == first and symbols[i + 1] == second):
                    out.append(first + second)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols[:] = out

    # ------------------------------------------------------------------ decode

    def decode(self, token_id: int) -> str:
        """Текст ОДНОГО токена; неизвестный id → пустая строка (byte-fallback)."""
        if not 0 <= token_id < len(self._id2token):
            return ""
        token = self._id2token[token_id]
        out_bytes = bytearray()
        for ch in token:
            b = _char_to_byte(ord(ch))
            if b < 0:
                return ""            # символ вне алфавита byte-level
            out_bytes.append(b)
        try:
            return out_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return ""                # некорректная UTF-8 последовательность

    def decode_ids(self, ids: List[int]) -> str:
        """Конкатенация текстов токенов (детокенизация сессии, ADR-11 §3)."""
        return "".join(self.decode(t) for t in ids)

    def vocab_size(self) -> int:
        return len(self._id2token)

    @property
    def n_merges(self) -> int:
        return len(self._ranks)


def isinstance_tuple(m) -> bool:  # pragma: no cover - утилита читаемости
    return isinstance(m, (tuple, list))
