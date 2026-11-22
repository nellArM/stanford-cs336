"""Exact line deduplication and MinHash + LSH fuzzy document deduplication."""

from __future__ import annotations

import hashlib
import os
import random
import string
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from functools import lru_cache
from pathlib import Path

import mmh3
import numpy as np
import regex as re


def _line_hash(line: str) -> bytes:
    # A fixed 8-byte digest per distinct line keeps the counter small regardless of line length.
    return hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest()


def exact_line_deduplication(input_files: Sequence[os.PathLike], output_directory: os.PathLike) -> None:
    """Rewrite every file keeping only the lines that occur exactly once in the whole corpus."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    counts: Counter[bytes] = Counter()
    for path in input_files:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                counts[_line_hash(line.rstrip("\n"))] += 1

    for path in input_files:
        with (
            open(path, encoding="utf-8", errors="replace") as f,
            open(output_directory / Path(path).name, "w", encoding="utf-8") as out,
        ):
            for line in f:
                if counts[_line_hash(line.rstrip("\n"))] == 1:
                    out.write(line)


_PUNCTUATION_RE = re.compile(r"\p{P}+")
_WHITESPACE_RE = re.compile(r"\s+")
_PRIME = np.uint64((1 << 31) - 1)


def normalize_text(text: str) -> str:
    """Lowercase, NFD-normalize, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _PUNCTUATION_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return _WHITESPACE_RE.sub(" ", text).strip()


def word_ngrams(text: str, n: int) -> set[str]:
    words = normalize_text(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


@lru_cache(maxsize=8)
def _hash_family(num_hashes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=num_hashes, dtype=np.uint64)
    b = rng.integers(0, _PRIME, size=num_hashes, dtype=np.uint64)
    return a, b


def minhash_signature(ngrams: Iterable[str], num_hashes: int, seed: int = 0) -> np.ndarray:
    """MinHash signature of a set of n-grams.

    Each n-gram is hashed once with MurmurHash3 (to 31 bits) and the ``num_hashes`` hash
    functions are simulated with universal hashing ``(a * x + b) mod p`` with p = 2^31 - 1,
    which is far cheaper than calling MurmurHash ``num_hashes`` times per n-gram.  All
    products fit in uint64.
    """
    a, b = _hash_family(num_hashes, seed)
    x = np.fromiter((mmh3.hash(g, signed=False) % int(_PRIME) for g in ngrams), dtype=np.uint64)
    signature = np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(x), 4096):  # bound the temporary (num_hashes x chunk) matrix
        chunk = x[start : start + 4096]
        hashed = (a[:, None] * chunk[None, :] + b[:, None]) % _PRIME
        np.minimum(signature, hashed.min(axis=1), out=signature)
    return signature


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def find_duplicate_clusters(
    ngram_sets: Sequence[set[str]],
    num_hashes: int,
    num_bands: int,
    jaccard_threshold: float,
    signatures: Sequence[np.ndarray] | None = None,
) -> list[list[int]]:
    """Cluster document indices whose true n-gram Jaccard similarity exceeds the threshold.

    Candidates come from LSH banding of MinHash signatures; clusters are the connected
    components of the confirmed-duplicate graph (so duplication is transitive).
    """
    assert num_hashes % num_bands == 0
    rows = num_hashes // num_bands
    if signatures is None:
        signatures = [minhash_signature(s, num_hashes) for s in ngram_sets]

    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for doc_idx, signature in enumerate(signatures):
        for band in range(num_bands):
            buckets[(band, signature[band * rows : (band + 1) * rows].tobytes())].append(doc_idx)

    candidate_pairs: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidate_pairs.add((members[i], members[j]))

    union_find = _UnionFind(len(ngram_sets))
    for i, j in candidate_pairs:
        if union_find.find(i) != union_find.find(j) and jaccard(ngram_sets[i], ngram_sets[j]) >= jaccard_threshold:
            union_find.union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for doc_idx in range(len(ngram_sets)):
        clusters[union_find.find(doc_idx)].append(doc_idx)
    return list(clusters.values())


def minhash_deduplication(
    input_files: Sequence[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
    seed: int = 0,
) -> None:
    """Fuzzy document deduplication: one file is one document; one random document per cluster is kept."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    texts = []
    for path in input_files:
        with open(path, encoding="utf-8", errors="replace") as f:
            texts.append(f.read())
    ngram_sets = [word_ngrams(text, ngrams) for text in texts]
    clusters = find_duplicate_clusters(ngram_sets, num_hashes, num_bands, jaccard_threshold)

    rng = random.Random(seed)
    for cluster in clusters:
        keep = rng.choice(cluster)
        with open(output_directory / Path(input_files[keep]).name, "w", encoding="utf-8") as out:
            out.write(texts[keep])
