"""Byte-pair-encoding tokenizer training.

The trainer works in three stages:

1. Pre-tokenization.  The corpus is split on special tokens (so that no merge can
   ever cross a document boundary) and then with the GPT-2 regex.  For large files
   this is done in parallel over chunks whose boundaries fall on a special token.
2. The pre-token frequency table is turned into "words" (lists of token ids) plus
   an index from every adjacent pair to the set of words that contain it.
3. Merging.  The most frequent pair (ties broken by the lexicographically greater
   pair of byte strings) is popped from a lazily-invalidated max-heap and only the
   words that contain it are rewritten, so each merge costs time proportional to
   the number of affected words instead of the size of the corpus.
"""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(PAT)

# Files smaller than this are pre-tokenized in-process: starting a worker pool
# costs more than it saves.
_PARALLEL_THRESHOLD_BYTES = 32 * 1024 * 1024
_TARGET_CHUNK_BYTES = 64 * 1024 * 1024


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Split a file into chunks that start at ``split_special_token`` (course-provided helper)."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = max(1, file_size // desired_num_chunks)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size + len(split_special_token) - 1)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
    return re.split(pattern, text)


def pretokenize_text(text: str, special_tokens: list[str]) -> Counter[str]:
    """Count the pre-tokens of ``text``; special tokens are removed and act as hard boundaries."""
    counts: Counter[str] = Counter()
    for part in _split_on_special_tokens(text, special_tokens):
        counts.update(m.group(0) for m in _PAT_RE.finditer(part))
    return counts


def _pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> Counter[str]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_text(text, special_tokens)


def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[str]:
    file_size = os.path.getsize(input_path)
    if file_size < _PARALLEL_THRESHOLD_BYTES or num_processes == 1 or not special_tokens:
        with open(input_path, encoding="utf-8", errors="ignore") as f:
            return pretokenize_text(f.read(), special_tokens)

    num_processes = num_processes or min(48, os.cpu_count() or 1)
    num_chunks = max(num_processes, file_size // _TARGET_CHUNK_BYTES)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
    jobs = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    total: Counter[str] = Counter()
    with Pool(num_processes) as pool:
        for counts in pool.imap_unordered(_pretokenize_chunk, jobs):
            total.update(counts)
    return total


class _HeapItem:
    """Max-heap entry ordered by (count, left bytes, right bytes)."""

    __slots__ = ("count", "key", "pair")

    def __init__(self, count: int, key: tuple[bytes, bytes], pair: tuple[int, int]):
        self.count = count
        self.key = key
        self.pair = pair

    def __lt__(self, other: _HeapItem) -> bool:  # reversed: heapq is a min-heap
        if self.count != other.count:
            return self.count > other.count
        return self.key > other.key


def merge_pretoken_counts(
    pretoken_counts: dict[str, int],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Run the BPE merge loop on a pre-token frequency table."""
    vocab: dict[int, bytes] = {}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    byte_offset = len(vocab)
    for b in range(256):
        vocab[byte_offset + b] = bytes([b])

    words: list[list[int]] = []
    freqs: list[int] = []
    for token, count in pretoken_counts.items():
        encoded = token.encode("utf-8")
        if len(encoded) < 2:
            continue  # a single byte can never take part in a merge
        words.append([byte_offset + b for b in encoded])
        freqs.append(count)

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, word in enumerate(words):
        f = freqs[wi]
        for pair in zip(word, word[1:]):
            pair_counts[pair] += f
            pair_to_words[pair].add(wi)

    heap = [_HeapItem(c, (vocab[p[0]], vocab[p[1]]), p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and heap:
        item = heapq.heappop(heap)
        pair = item.pair
        if pair_counts.get(pair, 0) != item.count:
            continue  # stale entry; a fresh one was pushed when the count changed
        if item.count <= 0:
            break

        a, b = pair
        new_id = len(vocab)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        changed: set[tuple[int, int]] = set()
        for wi in pair_to_words.pop(pair, ()):
            word = words[wi]
            f = freqs[wi]
            # Remove this word's contribution, rewrite it, then add it back.
            for old in zip(word, word[1:]):
                pair_counts[old] -= f
                changed.add(old)
            new_word: list[int] = []
            i, n = 0, len(word)
            while i < n:
                if i + 1 < n and word[i] == a and word[i + 1] == b:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            old_pairs = set(zip(word, word[1:]))
            new_pairs = set(zip(new_word, new_word[1:]))
            for gone in old_pairs - new_pairs:
                if gone != pair:
                    pair_to_words[gone].discard(wi)
            for new in zip(new_word, new_word[1:]):
                pair_counts[new] += f
                pair_to_words[new].add(wi)
                changed.add(new)
            words[wi] = new_word

        pair_counts.pop(pair, None)
        changed.discard(pair)
        for p in changed:
            c = pair_counts[p]
            if c > 0:
                heapq.heappush(heap, _HeapItem(c, (vocab[p[0]], vocab[p[1]]), p))
            else:
                del pair_counts[p]
                pair_to_words.pop(p, None)

    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Returns the vocabulary (token id -> bytes) and the merges in creation order.
    ``vocab_size`` counts the 256 byte tokens and the special tokens.
    """
    pretoken_counts = pretokenize_file(input_path, special_tokens, num_processes)
    return merge_pretoken_counts(pretoken_counts, vocab_size, special_tokens)


def save_tokenizer(
    vocab: dict[int, bytes],
    merges: Iterable[tuple[bytes, bytes]],
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
) -> None:
    """Serialize with hex-encoded bytes so that arbitrary byte strings survive the round trip."""
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({str(i): tok.hex() for i, tok in vocab.items()}, f)
    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")


def load_tokenizer_files(
    vocab_path: str | os.PathLike, merges_path: str | os.PathLike
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_path, encoding="utf-8") as f:
        vocab = {int(i): bytes.fromhex(tok) for i, tok in json.load(f).items()}
    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            left, right = line.split()
            merges.append((bytes.fromhex(left), bytes.fromhex(right)))
    return vocab, merges
