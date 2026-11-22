"""DDP training-step benchmark: naive vs. flat vs. overlapped gradient communication
(problems naive_ddp_benchmarking, minimal_ddp_flat_benchmarking,
ddp_overlap_individual_parameters_benchmarking).

    CUDA_VISIBLE_DEVICES=1,0 uv run python -m cs336_systems.bench.ddp_bench --size large --mode overlap
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
from cs336_systems.ddp import DDP


def worker(rank: int, world_size: int, args) -> None:
    device = setup(rank, world_size, "nccl")
    torch.manual_seed(rank)
    model = make_model(args.size, args.context_length, args.num_layers, device=device)
    ddp = DDP(model, mode=args.mode)
    optimizer = AdamW(ddp.parameters(), lr=1e-4)
    x, y = random_batch(args.batch_size // world_size, args.context_length, device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args.bf16):
            loss = cross_entropy(ddp(x), y)
        loss.backward()
        ddp.finish_gradient_synchronization()
        optimizer.step()
        sync(device)

    for _ in range(args.warmup):
        step()
    totals, comms = [], []
    for _ in range(args.steps):
        sync(device)
        t0 = timeit.default_timer()
        step()
        totals.append(timeit.default_timer() - t0)
        comms.append(ddp.comm_seconds)

    if args.trace and rank == 0:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
        dump_gpu_trace(prof, args.trace)
    elif args.trace:
        step()  # keep the collectives matched with rank 0's profiled step

    stats = gather_list((statistics.mean(totals), statistics.mean(comms), torch.cuda.max_memory_allocated()))
    if rank == 0:
        total_ms = 1e3 * statistics.mean(s[0] for s in stats)
        comm_ms = 1e3 * statistics.mean(s[1] for s in stats)
        row = dict(size=args.size, num_layers=args.num_layers or MODEL_SIZES[args.size]["num_layers"], context_length=args.context_length,
                   global_batch=args.batch_size, world_size=world_size, bf16=int(args.bf16), mode=args.mode, status="ok",
                   step_ms=round(total_ms, 1), step_std_ms=round(1e3 * statistics.stdev(totals), 1), blocked_on_comm_ms=round(comm_ms, 1),
                   comm_fraction=round(comm_ms / total_ms, 4), peak_gib=round(gib(max(s[2] for s in stats)), 2))  # fmt: skip
        print(row, flush=True)
        if args.out:
            append_row(args.out, row)
    cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(MODEL_SIZES), default="large")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4, help="global batch, split evenly across ranks")
    p.add_argument("--mode", choices=["naive", "flat", "overlap"], default="overlap")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--trace", default=None, help="write a GPU kernel trace of one extra step (rank 0) to this CSV")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    pick_port()
    mp.spawn(worker, args=(args.world_size, args), nprocs=args.world_size, join=True)
