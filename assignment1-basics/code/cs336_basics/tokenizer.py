"""BPE tokenizer: encode text to ids and decode ids back to text."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import PAT, load_tokenizer_files

_PAT_RE = re.compile(PAT)
# A position that is preceded by a non-whitespace character and followed by a
# whitespace character is always a pre-token boundary, so a stream may be cut there.
_SAFE_CUT_RE = re.compile(r"(?<=\S)(?=\s)")
_STREAM_BUFFER_CHARS = 1 << 15
_MAX_CACHE_ENTRIES = 1 << 20


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])

        self.bytes_to_id = {tok: i for i, tok in self.vocab.items()}
        for tok in self.special_tokens:
            encoded = tok.encode("utf-8")
            if encoded not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = encoded
                self.bytes_to_id[encoded] = new_id

        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        if self.special_tokens:
            # Longest first, so overlapping special tokens resolve to the longest match.
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(t) for t in ordered) + ")")
            self._max_special_len = len(ordered[0])
        else:
            self._special_re = None
            self._max_special_len = 0
        self._special_ids = {t: self.bytes_to_id[t.encode("utf-8")] for t in self.special_tokens}
        self._cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab, merges = load_tokenizer_files(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached
        parts = [bytes([b]) for b in pretoken.encode("utf-8")]
        rank = self.merge_rank
        while len(parts) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(parts, parts[1:]):
                r = rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            left, right = best_pair
            merged: list[bytes] = []
            i, n = 0, len(parts)
            while i < n:
                if i + 1 < n and parts[i] == left and parts[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = tuple(self.bytes_to_id[p] for p in parts)
        if len(self._cache) < _MAX_CACHE_ENTRIES:
            self._cache[pretoken] = ids
        return ids

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in _PAT_RE.finditer(text):
            yield from self._encode_pretoken(match.group(0))

    def _iter_encode(self, text: str) -> Iterator[int]:
        if self._special_re is None:
            yield from self._encode_ordinary(text)
            return
        for part in self._special_re.split(text):
            if not part:
                continue
            special_id = self._special_ids.get(part)
            if special_id is not None:
                yield special_id
            else:
                yield from self._encode_ordinary(part)

    def encode(self, text: str) -> list[int]:
        return list(self._iter_encode(text))

    def _safe_cut(self, buffer: str) -> int:
        """Largest index at which ``buffer`` can be split without changing its tokenization (0 if none)."""
        limit = len(buffer) - self._max_special_len
        if limit <= 0:
            return 0
        cuts = [m.start() for m in _SAFE_CUT_RE.finditer(buffer, 0, limit)]
        if not cuts:
            return 0
        if self._special_re is None:
            return cuts[-1]
        special_spans = [m.span() for m in self._special_re.finditer(buffer)]
        for cut in reversed(cuts):
            if not any(start < cut < end for start, end in special_spans):
                return cut
        return 0

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle) with bounded memory.

        The result is identical to ``encode("".join(iterable))``: text is buffered and
        only cut at positions that are guaranteed to be token boundaries.
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            if len(buffer) >= _STREAM_BUFFER_CHARS:
                cut = self._safe_cut(buffer)
                if cut:
                    yield from self._iter_encode(buffer[:cut])
                    buffer = buffer[cut:]
        if buffer:
            yield from self._iter_encode(buffer)

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
