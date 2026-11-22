"""End-to-end benchmark of the Transformer LM (problems benchmarking_script,
benchmarking_mixed_precision, torch_compile (b), nsys_profile, memory_profiling).

Each invocation benchmarks one configuration and appends one row to a CSV file, so an
out-of-memory failure in one configuration cannot affect the others.

    uv run python -m cs336_systems.bench.benchmark --size small --mode train --out experiments/results/e2e.csv
"""

from __future__ import annotations

import argparse
import statistics
import timeit

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.bench.common import BATCH_SIZE, MODEL_SIZES, append_row, autocast_context, gib, make_model, random_batch


def annotate_attention() -> None:
    """Swap in an attention function with NVTX ranges around its three stages."""
    import math

    from einops import einsum

    from cs336_basics.nn_utils import softmax

    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        with nvtx.range("scaled dot product attention"):
            d_k = K.shape[-1]
            with nvtx.range("computing attention scores"):
                scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
                if mask is not None:
                    scores = torch.where(mask, scores, float("-inf"))
            with nvtx.range("computing softmax"):
                weights = softmax(scores, dim=-1)
            with nvtx.range("final matmul"):
                return einsum(weights, V, "... query key, ... key d_v ->  ... query d_v")

    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(MODEL_SIZES), default="small")
    p.add_argument("--num-layers", type=int, default=None, help="override the number of layers")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="train")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--bf16", action="store_true", help="run under bf16 autocast")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--nvtx", action="store_true", help="emit NVTX ranges (for nsys)")
    p.add_argument("--memory-snapshot", default=None, help="dump a torch memory snapshot of one step to this path")
    p.add_argument("--out", default=None, help="CSV file to append the result to")
    p.add_argument("--tag", default="")
    p.add_argument("--retry-on-oom", action="store_true",
                   help="exit with status 3 on OOM instead of recording it (the GPU is shared; the caller retries later)")
    args = p.parse_args()

    row = dict(tag=args.tag, size=args.size, num_layers=args.num_layers or MODEL_SIZES[args.size]["num_layers"],
               context_length=args.context_length, batch_size=args.batch_size, mode=args.mode, warmup=args.warmup,
               steps=args.steps, bf16=int(args.bf16), compile=int(args.compile))  # fmt: skip
    try:
        row.update(run(args))
        row["status"] = "ok"
    except torch.OutOfMemoryError:
        if args.retry_on_oom:
            raise SystemExit(3)
        row["status"] = "OOM"
        row["peak_gib"] = round(gib(torch.cuda.max_memory_allocated()), 2)
    print(row)
    if args.out:
        keys = ["tag", "size", "num_layers", "context_length", "batch_size", "mode", "warmup", "steps", "bf16", "compile", "status",
                "params_m", "forward_ms", "forward_std_ms", "backward_ms", "backward_std_ms", "optimizer_ms", "optimizer_std_ms",
                "total_ms", "total_std_ms", "peak_gib", "baseline_gib"]  # fmt: skip
        append_row(args.out, {k: row.get(k, "") for k in keys})


def run(args) -> dict:
    if args.nvtx:
        annotate_attention()
    torch.cuda.reset_peak_memory_stats()
    model = make_model(args.size, args.context_length, args.num_layers)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = AdamW(model.parameters(), lr=1e-4) if args.mode == "train" else None
    x, y = random_batch(args.batch_size, args.context_length)
    run_model = torch.compile(model) if args.compile else model
    if args.mode == "forward":
        model.eval()

    times = {"forward": [], "backward": [], "optimizer": [], "total": []}

    def step(record: bool, label: str) -> None:
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        with nvtx.range(label) if args.nvtx else torch.autograd.profiler.record_function(label):
            if args.mode == "forward":
                with torch.no_grad(), autocast_context(args.bf16), nvtx.range("forward"):
                    run_model(x)
                torch.cuda.synchronize()
                t1 = t2 = t3 = timeit.default_timer()
            else:
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                else:
                    model.zero_grad(set_to_none=True)
                with autocast_context(args.bf16), nvtx.range("forward"):
                    loss = cross_entropy(run_model(x), y)
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                with nvtx.range("backward"):
                    loss.backward()
                torch.cuda.synchronize()
                t2 = timeit.default_timer()
                if optimizer is not None:
                    with nvtx.range("optimizer"):
                        optimizer.step()
                    torch.cuda.synchronize()
                t3 = timeit.default_timer()
        if record:
            times["forward"].append(t1 - t0)
            times["backward"].append(t2 - t1)
            times["optimizer"].append(t3 - t2)
            times["total"].append(t3 - t0)

    for _ in range(args.warmup):
        step(False, "warmup step")
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        step(False, "profiled step")
        torch.cuda.memory._dump_snapshot(args.memory_snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)

    for _ in range(args.steps):
        step(True, "measured step")

    out = {"params_m": round(params_m, 1), "peak_gib": round(gib(torch.cuda.max_memory_allocated()), 3), "baseline_gib": round(gib(baseline), 3)}
    for name, values in times.items():
        if values:
            out[f"{name}_ms"] = round(1e3 * statistics.mean(values), 3)
            out[f"{name}_std_ms"] = round(1e3 * (statistics.stdev(values) if len(values) > 1 else 0.0), 3)
    return out


if __name__ == "__main__":
    main()
