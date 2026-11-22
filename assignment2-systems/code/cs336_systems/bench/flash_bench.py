"""FlashAttention-2 vs. plain PyTorch attention (problem flash_benchmarking).

Batch size 1, causal masking, timed with ``triton.testing.do_bench``.

    uv run python -m cs336_systems.bench.flash_bench --out experiments/results/flash.csv
    uv run python -m cs336_systems.bench.flash_bench --tune --out experiments/results/flash_tiles.csv
"""

from __future__ import annotations

import argparse
import itertools

import torch
import triton

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.bench.common import append_row
from cs336_systems.flash_attention import FlashAttentionTriton, FlashAttentionTritonFull

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODELS = [16, 32, 64, 128]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def pytorch_attention(q, k, v, is_causal=True):
    n = q.shape[-2]
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril() if is_causal else None
    return scaled_dot_product_attention(q, k, v, mask)


IMPLS = {
    "pytorch": pytorch_attention,
    "flash_triton": FlashAttentionTriton.apply,  # Triton forward + torch.compile'd PyTorch backward
    "flash_triton_full": FlashAttentionTritonFull.apply,  # Triton forward + Triton backward
}


def bench_impl(fn, n: int, d: int, dtype, warmup: int, rep: int) -> dict:
    torch.cuda.empty_cache()
    q, k, v = (torch.randn(1, n, d, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
    d_out = torch.randn(1, n, d, device="cuda", dtype=dtype)

    def forward():
        with torch.no_grad():
            fn(q, k, v, True)

    def end_to_end():
        fn(q, k, v, True).backward(d_out)
        q.grad = k.grad = v.grad = None

    out = fn(q, k, v, True)

    def backward():
        out.backward(d_out, retain_graph=True)
        q.grad = k.grad = v.grad = None

    return dict(
        forward_ms=round(triton.testing.do_bench(forward, warmup=warmup, rep=rep), 4),
        backward_ms=round(triton.testing.do_bench(backward, warmup=warmup, rep=rep), 4),
        end_to_end_ms=round(triton.testing.do_bench(end_to_end, warmup=warmup, rep=rep), 4),
    )


def run_sweep(args) -> None:
    for dtype_name, n, d in itertools.product(DTYPES, SEQ_LENS, D_MODELS):
        if n > args.max_seq:
            continue
        for impl, fn in IMPLS.items():
            row = dict(impl=impl, dtype=dtype_name, seq_len=n, d_model=d)
            try:
                row.update(status="ok", **bench_impl(fn, n, d, DTYPES[dtype_name], args.warmup, args.rep))
            except torch.OutOfMemoryError:
                row.update(status="OOM", forward_ms="", backward_ms="", end_to_end_ms="")
            print(row, flush=True)
            append_row(args.out, {k: row[k] for k in ["impl", "dtype", "seq_len", "d_model", "status", "forward_ms", "backward_ms", "end_to_end_ms"]})


def run_tune(args) -> None:
    """Sweep tile sizes for the fully-Triton implementation on a few representative shapes."""
    tiles = [16, 32, 64, 128]
    for dtype_name, n, d in itertools.product(DTYPES, [1024, 8192], D_MODELS):
        for q_tile, k_tile in itertools.product(tiles, tiles):
            FlashAttentionTritonFull.tiles = (q_tile, k_tile)
            row = dict(dtype=dtype_name, seq_len=n, d_model=d, q_tile=q_tile, k_tile=k_tile)
            try:
                row.update(status="ok", **bench_impl(FlashAttentionTritonFull.apply, n, d, DTYPES[dtype_name], args.warmup, args.rep))
            except Exception as exc:  # compile failures for tiles that do not fit are also informative
                row.update(status=type(exc).__name__, forward_ms="", backward_ms="", end_to_end_ms="")
            print(row, flush=True)
            append_row(args.out, row)
    FlashAttentionTritonFull.tiles = None


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--max-seq", type=int, default=65536)
    p.add_argument("--warmup", type=int, default=25, help="do_bench warm-up time in ms")
    p.add_argument("--rep", type=int, default=100, help="do_bench measurement time in ms")
    args = p.parse_args()
    run_tune(args) if args.tune else run_sweep(args)
