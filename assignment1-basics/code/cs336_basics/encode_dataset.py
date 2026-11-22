"""Encode a text file into a flat uint16 array of token ids (saved as .npy), in parallel.

Example:
    uv run python -m cs336_basics.encode_dataset --input data/TinyStoriesV2-GPT4-valid.txt \
        --tokenizer artifacts/tokenizers/tinystories --output data/tinystories_valid.npy
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from cs336_basics.bpe import find_chunk_boundaries
from cs336_basics.tokenizer import Tokenizer

_TOKENIZER: Tokenizer | None = None


def _init(prefix: str, special_tokens: list[str]) -> None:
    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_files(f"{prefix}_vocab.json", f"{prefix}_merges.txt", special_tokens)


def _encode_chunk(args: tuple[str, int, int]) -> np.ndarray:
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    assert _TOKENIZER is not None
    return np.fromiter(_TOKENIZER._iter_encode(text), dtype=np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True, help="prefix of the *_vocab.json / *_merges.txt files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-processes", type=int, default=min(48, os.cpu_count() or 1))
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()
    special_tokens = ["<|endoftext|>"]

    file_size = os.path.getsize(args.input)
    num_chunks = max(1, file_size // (args.chunk_mb * 2**20))
    with open(args.input, "rb") as f:
        # Chunks start exactly at a document separator, so encoding them independently
        # gives the same ids as encoding the whole file.
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode())
    jobs = [(args.input, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

    t0 = time.perf_counter()
    pieces: list[np.ndarray] = []
    with Pool(args.num_processes, initializer=_init, initargs=(args.tokenizer, special_tokens)) as pool:
        for piece in pool.imap(_encode_chunk, jobs):
            pieces.append(piece)
    ids = np.concatenate(pieces)
    np.save(args.output, ids)
    elapsed = time.perf_counter() - t0
    print(
        f"{args.input}: {file_size:,} bytes -> {len(ids):,} tokens "
        f"({file_size / len(ids):.3f} bytes/token) in {elapsed:.1f}s with {args.num_processes} processes"
    )


if __name__ == "__main__":
    main()
