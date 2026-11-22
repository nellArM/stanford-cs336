"""Train a BPE tokenizer on a text file and report time, memory and the longest token.

Example:
    uv run python -m cs336_basics.train_tokenizer --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --name tinystories
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import resource
import time
from pathlib import Path

from cs336_basics.bpe import merge_pretoken_counts, pretokenize_file, save_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default="artifacts/tokenizers")
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--profile", action="store_true", help="cProfile the main process")
    args = parser.parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]

    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()

    t0 = time.perf_counter()
    counts = pretokenize_file(args.input, special_tokens, args.num_processes)
    t1 = time.perf_counter()
    vocab, merges = merge_pretoken_counts(counts, args.vocab_size, special_tokens)
    t2 = time.perf_counter()

    if profiler:
        profiler.disable()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer(vocab, merges, out_dir / f"{args.name}_vocab.json", out_dir / f"{args.name}_merges.txt")

    longest = max(vocab.values(), key=len)
    # ru_maxrss is in KiB on Linux.  Children = the pre-tokenization workers (max over workers).
    peak_self_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    peak_child_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    stats = {
        "input": args.input,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "unique_pretokens": len(counts),
        "pretokenize_seconds": round(t1 - t0, 2),
        "merge_seconds": round(t2 - t1, 2),
        "total_seconds": round(t2 - t0, 2),
        "peak_rss_main_gb": round(peak_self_gb, 2),
        "peak_rss_largest_worker_gb": round(peak_child_gb, 2),
        "longest_token": longest.decode("utf-8", errors="replace"),
        "longest_token_num_bytes": len(longest),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    with open(out_dir / f"{args.name}_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if profiler:
        with open(out_dir / f"{args.name}_profile.txt", "w") as f:
            pstats.Stats(profiler, stream=f).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
