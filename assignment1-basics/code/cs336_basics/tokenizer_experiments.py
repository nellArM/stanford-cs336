"""Tokenizer experiments: compression ratio, cross-domain tokenization and throughput."""

from __future__ import annotations

import json
import random
import time

from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def sample_documents(path: str, n: int, seed: int = 0, read_bytes: int = 50_000_000) -> list[str]:
    with open(path, "rb") as f:
        text = f.read(read_bytes).decode("utf-8", errors="ignore")
    docs = [d for d in text.split(EOT)[:-1] if d.strip()]
    return random.Random(seed).sample(docs, n)


def bytes_per_token(tokenizer: Tokenizer, docs: list[str]) -> float:
    num_bytes = sum(len(d.encode("utf-8")) for d in docs)
    return num_bytes / sum(len(tokenizer.encode(d)) for d in docs)


def main() -> None:
    prefix = "artifacts/tokenizers"
    ts_tok = Tokenizer.from_files(f"{prefix}/tinystories_vocab.json", f"{prefix}/tinystories_merges.txt", [EOT])
    owt_tok = Tokenizer.from_files(f"{prefix}/owt_vocab.json", f"{prefix}/owt_merges.txt", [EOT])
    ts_docs = sample_documents("data/TinyStoriesV2-GPT4-train.txt", 10)
    owt_docs = sample_documents("data/owt_train.txt", 10)

    results = {
        "ts_docs_ts_tokenizer": bytes_per_token(ts_tok, ts_docs),
        "owt_docs_owt_tokenizer": bytes_per_token(owt_tok, owt_docs),
        "owt_docs_ts_tokenizer": bytes_per_token(ts_tok, owt_docs),
        "ts_docs_owt_tokenizer": bytes_per_token(owt_tok, ts_docs),
    }

    # Single-process throughput on ~20 MB of raw text, with a cold pre-token cache.
    for name, path, vocab in [("ts", "data/TinyStoriesV2-GPT4-valid.txt", "tinystories"), ("owt", "data/owt_valid.txt", "owt")]:
        with open(path, "rb") as f:
            text = f.read(20_000_000).decode("utf-8", errors="ignore")
        tok = Tokenizer.from_files(f"{prefix}/{vocab}_vocab.json", f"{prefix}/{vocab}_merges.txt", [EOT])
        start = time.perf_counter()
        n_tokens = len(tok.encode(text))
        elapsed = time.perf_counter() - start
        rate = len(text.encode("utf-8")) / elapsed
        results[f"{name}_throughput_bytes_per_s"] = rate
        results[f"{name}_throughput_tokens_per_s"] = n_tokens / elapsed
        results[f"{name}_pile_825GB_hours_single_process"] = 825e9 / rate / 3600

    print(json.dumps(results, indent=2))
    with open(f"{prefix}/tokenizer_experiments.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
