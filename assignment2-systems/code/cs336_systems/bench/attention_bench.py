"""Single-head attention benchmark, eager vs. torch.compile (problems pytorch_attention, torch_compile (a)).

    uv run python -m cs336_systems.bench.attention_bench --out experiments/results/attention.csv
"""

from __future__ import annotations

import argparse
import timeit

import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.bench.common import append_row, gib

BATCH = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]


def bench_config(attn, d: int, n: int, max_reps: int, budget_s: float) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    q, k, v = (torch.randn(BATCH, n, d, device="cuda", requires_grad=True) for _ in range(3))

    def forward():
        out = attn(q, k, v)
        torch.cuda.synchronize()
        return out

    # Warm-up (also triggers compilation), and a probe that sets the number of repetitions:
    # the handout asks for 100, which we use whenever 100 passes fit in the time budget.
    for _ in range(3):
        forward()
    t0 = timeit.default_timer()
    out = forward()
    probe = timeit.default_timer() - t0
    reps = int(max(5, min(max_reps, budget_s / max(probe, 1e-6))))

    t0 = timeit.default_timer()
    for _ in range(reps):
        forward()
    fwd_ms = 1e3 * (timeit.default_timer() - t0) / reps

    # Backward: a fresh graph per repetition (compiled functions cannot be back-propagated twice);
    # only the backward call itself is timed.
    def fresh_loss():
        loss = attn(q, k, v).sum()
        torch.cuda.synchronize()
        return loss

    fresh_loss().backward()  # warm-up
    torch.cuda.synchronize()
    bwd_total, mem_before_backward = 0.0, 0
    for _ in range(reps):
        loss = fresh_loss()
        mem_before_backward = torch.cuda.memory_allocated()
        t0 = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        bwd_total += timeit.default_timer() - t0
        del loss
    bwd_ms = 1e3 * bwd_total / reps
    return dict(status="ok", reps=reps, forward_ms=round(fwd_ms, 3), backward_ms=round(bwd_ms, 3),
                mem_before_backward_gib=round(gib(mem_before_backward), 3), peak_gib=round(gib(torch.cuda.max_memory_allocated()), 3))  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/results/attention.csv")
    p.add_argument("--max-reps", type=int, default=100)
    p.add_argument("--budget-s", type=float, default=20.0, help="time budget per timed loop")
    args = p.parse_args()

    variants = {"eager": scaled_dot_product_attention, "compiled": torch.compile(scaled_dot_product_attention)}
    for n in SEQ_LENS:
        for d in D_MODELS:
            for name, attn in variants.items():
                free_gib = round(gib(torch.cuda.mem_get_info()[0]), 1)
                row = dict(variant=name, d_model=d, seq_len=n, free_gib_at_start=free_gib)
                try:
                    row.update(bench_config(attn, d, n, args.max_reps, args.budget_s))
                except torch.OutOfMemoryError:
                    row.update(status="OOM", reps="", forward_ms="", backward_ms="", mem_before_backward_gib="",
                               peak_gib=round(gib(torch.cuda.max_memory_allocated()), 3))  # fmt: skip
                print(row, flush=True)
                append_row(args.out, row)


if __name__ == "__main__":
    main()
