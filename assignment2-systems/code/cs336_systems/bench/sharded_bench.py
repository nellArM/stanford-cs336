"""Peak memory and step time with and without optimizer state sharding
(problem optimizer_state_sharding_accounting).

    CUDA_VISIBLE_DEVICES=1,0 uv run python -m cs336_systems.bench.sharded_bench --size large --sharded
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
from cs336_systems.bench.dist_utils import pick_port, cleanup, gather_list, setup, sync
from cs336_systems.ddp import DDP
from cs336_systems.sharded_optimizer import ShardedOptimizer


def worker(rank: int, world_size: int, args) -> None:
    device = setup(rank, world_size, "nccl")
    torch.manual_seed(rank)
    model = make_model(args.size, args.context_length, args.num_layers, device=device)
    ddp = DDP(model, mode="overlap")
    if args.sharded:
        optimizer = ShardedOptimizer(ddp.parameters(), AdamW, lr=1e-4)
    else:
        optimizer = AdamW(ddp.parameters(), lr=1e-4)
    x, y = random_batch(args.batch_size // world_size, args.context_length, device=device)
    sync(device)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    mem = {"after_init": torch.cuda.max_memory_allocated(), "allocated_after_init": torch.cuda.memory_allocated()}

    times, step_times = [], []
    for it in range(args.warmup + args.steps):
        sync(device)
        t0 = timeit.default_timer()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args.bf16):
            loss = cross_entropy(ddp(x), y)
        loss.backward()
        ddp.finish_gradient_synchronization()
        sync(device)
        if it == args.warmup:  # steady state: optimizer state already exists
            mem["before_step"] = torch.cuda.max_memory_allocated()
            mem["allocated_before_step"] = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        t1 = timeit.default_timer()
        optimizer.step()
        sync(device)
        t2 = timeit.default_timer()
        if it == args.warmup:
            mem["after_step"] = torch.cuda.max_memory_allocated()
            mem["allocated_after_step"] = torch.cuda.memory_allocated()
        if it >= args.warmup:
            times.append(t2 - t0)
            step_times.append(t2 - t1)
        if it == args.warmup - 1:
            torch.cuda.reset_peak_memory_stats()

    if args.sharded:
        state_bytes = optimizer.local_state_bytes()
    else:
        state_bytes = sum(v.numel() * v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
    stats = gather_list((mem, state_bytes, statistics.mean(times), statistics.mean(step_times)))
    if rank == 0:
        for r, (m, sb, t, st) in enumerate(stats):
            row = dict(size=args.size, num_layers=args.num_layers or MODEL_SIZES[args.size]["num_layers"], sharded=int(args.sharded), rank=r,
                       bf16=int(args.bf16), params_gib=round(gib(param_bytes), 2), optimizer_state_gib=round(gib(sb), 2),
                       peak_after_init_gib=round(gib(m["after_init"]), 2), peak_before_step_gib=round(gib(m["before_step"]), 2),
                       peak_during_step_gib=round(gib(m["after_step"]), 2), allocated_after_step_gib=round(gib(m["allocated_after_step"]), 2),
                       iter_ms=round(1e3 * t, 1), optimizer_step_ms=round(1e3 * st, 1))  # fmt: skip
            print(row, flush=True)
            if args.out:
                append_row(args.out, row)
    cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(MODEL_SIZES), default="large")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sharded", action="store_true")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    pick_port()
    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
