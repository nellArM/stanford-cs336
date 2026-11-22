"""FSDP training-step benchmark and all-gather timing (problem fsdp_accounting (b)).

    CUDA_VISIBLE_DEVICES=1,0 uv run python -m cs336_systems.bench.fsdp_bench --size xl --bf16
"""

from __future__ import annotations

import argparse
import statistics
import timeit

import torch
import torch.multiprocessing as mp

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.bench.common import MODEL_SIZES, append_row, autocast_context, gib, make_model, random_batch
from cs336_systems.bench.dist_utils import pick_port, cleanup, dump_gpu_trace, gather_list, setup, sync
from cs336_systems.fsdp import FSDP


def worker(rank: int, world_size: int, args) -> None:
    device = setup(rank, world_size, "nccl")
    torch.manual_seed(rank)
    row = dict(size=args.size, num_layers=args.num_layers or MODEL_SIZES[args.size]["num_layers"], context_length=args.context_length,
               global_batch=args.batch_size, bf16=int(args.bf16), prefetch=int(not args.no_prefetch))  # fmt: skip
    model = make_model(args.size, args.context_length, args.num_layers, device=device)
    full_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    fsdp = FSDP(model, compute_dtype=torch.bfloat16 if args.bf16 else None, prefetch=not args.no_prefetch)
    del model
    torch.cuda.empty_cache()
    local_param_bytes = sum(p.numel() * p.element_size() for p in fsdp.parameters())
    optimizer = AdamW(fsdp.parameters(), lr=1e-4)
    x, y = random_batch(args.batch_size // world_size, args.context_length, device=device)
    init_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args.bf16):
            loss = cross_entropy(fsdp(x).float(), y)
        loss.backward()
        fsdp.finish_gradient_synchronization()
        optimizer.step()
        sync(device)

    for _ in range(args.warmup):
        step()
    fsdp.stats.update(gather_hits=0, gather_misses=0, gather_ready=0)
    times = []
    for _ in range(args.steps):
        sync(device)
        t0 = timeit.default_timer()
        step()
        times.append(timeit.default_timer() - t0)
    hits, misses = fsdp.stats["gather_hits"], fsdp.stats["gather_misses"]
    ready = fsdp.stats["gather_ready"]

    # Instrumented steps: synchronise before every weight use and time how long the step is
    # actually blocked waiting for an all-gather that has not finished yet.
    fsdp.measure_waits = True
    fsdp.stats["wait_seconds"] = 0.0
    instrumented = []
    for _ in range(args.instrumented_steps):
        sync(device)
        t0 = timeit.default_timer()
        step()
        instrumented.append(timeit.default_timer() - t0)
    wait_per_step = fsdp.stats["wait_seconds"] / max(1, args.instrumented_steps)
    fsdp.measure_waits = False

    if args.trace:
        if rank == 0:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                step()
            dump_gpu_trace(prof, args.trace)
        else:
            step()

    stats = gather_list((statistics.mean(times), wait_per_step, statistics.mean(instrumented) if instrumented else 0.0,
                         torch.cuda.max_memory_allocated(), init_peak))  # fmt: skip
    if rank == 0:
        n_sites = len(fsdp._order)
        row.update(status="ok", full_params_gib=round(gib(full_param_bytes), 2), local_params_gib=round(gib(local_param_bytes), 2),
                   step_ms=round(1e3 * statistics.mean(s[0] for s in stats), 1), step_std_ms=round(1e3 * statistics.stdev(times), 1),
                   blocked_on_gather_ms=round(1e3 * statistics.mean(s[1] for s in stats), 1),
                   instrumented_step_ms=round(1e3 * statistics.mean(s[2] for s in stats), 1),
                   gather_hits=hits, gather_misses=misses, gather_ready=ready, call_sites=n_sites,
                   peak_gib=round(gib(max(s[3] for s in stats)), 2), init_peak_gib=round(gib(max(s[4] for s in stats)), 2))  # fmt: skip
        print(row, flush=True)
        if args.out:
            append_row(args.out, row)
    cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(MODEL_SIZES), default="xl")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--instrumented-steps", type=int, default=3)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no-prefetch", action="store_true")
    p.add_argument("--trace", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    pick_port()
    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
