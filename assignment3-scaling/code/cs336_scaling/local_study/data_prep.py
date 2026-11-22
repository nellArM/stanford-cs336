"""Download and tokenize DCLM-baseline shards for the local scaling study.

Mirrors cs336_scaling/tokenized_data.py: parquet shards of mlfoundations/dclm-baseline-1.0-parquet
are shuffled with seed 67, each document is tokenized with the Llama-2 tokenizer
(add_special_tokens=False) and followed by an EOS token, and ids are stored as uint16.

Output (under data/dclm/):
  train_0000.npy, train_0001.npy, ...  one array per source shard, consumed in this order
  val.npy                               2**21 held-out tokens from a shard never used for training
                                        (the first 2**18 are the "API" validation set)
  manifest.json                         shard names and token counts

Usage:
  uv run --extra study python -m cs336_scaling.local_study.data_prep --min-train-tokens 2.5e9
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = "mlfoundations/dclm-baseline-1.0-parquet"
TOKENIZER = "NousResearch/Llama-2-7b-hf"
SHUFFLE_SEED = 67
VAL_TOKENS = 2**21
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "dclm"


def shuffled_shards() -> list[str]:
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    return random.Random(SHUFFLE_SEED).sample(files, len(files))


def tokenize_shard(shard_path: str, out_path: str, max_tokens: int | None = None) -> int:
    """Tokenize one parquet shard to a uint16 .npy; returns the token count."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    eos = tokenizer.eos_token_id
    assert eos is not None and len(tokenizer) < 2**16

    local = hf_hub_download(REPO, shard_path, repo_type="dataset", local_dir=str(DATA_DIR / "_parquet"))
    pieces: list[np.ndarray] = []
    total = 0
    for batch in pq.ParquetFile(local).iter_batches(batch_size=2000, columns=["text"]):
        texts = batch.column("text").to_pylist()
        for ids in tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]:
            arr = np.empty(len(ids) + 1, dtype=np.uint16)
            arr[:-1] = ids
            arr[-1] = eos
            pieces.append(arr)
            total += len(arr)
        if max_tokens is not None and total >= max_tokens:
            break
    tokens = np.concatenate(pieces)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    tmp = out_path + ".tmp.npy"
    np.save(tmp, tokens)
    os.replace(tmp, out_path)
    os.remove(local)  # keep disk use low: the parquet file is no longer needed
    return int(len(tokens))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-tokens", type=float, default=2.5e9)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"train": [], "val": None}
    shards = shuffled_shards()

    if manifest["val"] is None:
        # The last shard of the shuffled order is reserved for validation.
        n = tokenize_shard(shards[-1], str(DATA_DIR / "val.npy"), max_tokens=VAL_TOKENS)
        manifest["val"] = {"shard": shards[-1], "file": "val.npy", "tokens": n}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"validation: {n:,} tokens from {shards[-1]}")

    def have() -> int:
        return sum(entry["tokens"] for entry in manifest["train"])

    next_index = len(manifest["train"])
    while have() < args.min_train_tokens and next_index < args.max_shards:
        batch = list(range(next_index, min(next_index + args.workers, args.max_shards)))
        results: dict[int, int] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(tokenize_shard, shards[i], str(DATA_DIR / f"train_{i:04d}.npy")): i for i in batch
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        # Append in shard order so the token stream is deterministic.
        for i in batch:
            manifest["train"].append({"shard": shards[i], "file": f"train_{i:04d}.npy", "tokens": results[i]})
        next_index = batch[-1] + 1
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"{len(manifest['train'])} train shards, {have():,} tokens")

    print(f"done: {have():,} train tokens in {len(manifest['train'])} shards")


if __name__ == "__main__":
    main()
