"""All-reduce latency vs. payload size and number of processes
(problem distributed_communication_single_node).

    CUDA_VISIBLE_DEVICES=1,0 uv run python -m cs336_systems.bench.allreduce_bench --backend nccl --world-sizes 2
    uv run python -m cs336_systems.bench.allreduce_bench --backend gloo --world-sizes 2 4 6
"""

from __future__ import annotations

import argparse
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.bench.common import append_row
from cs336_systems.bench.dist_utils import pick_port, cleanup, gather_list, setup, sync

SIZES_MB = [1, 10, 100, 1000]


def worker(rank: int, world_size: int, backend: str, out: str, reps: int) -> None:
    device = setup(rank, world_size, backend)
    for size_mb in SIZES_MB:
        data = torch.rand(size_mb * 1_000_000 // 4, dtype=torch.float32, device=device)
        for _ in range(5):  # warm-up: the first NCCL calls set up connections
            dist.all_reduce(data)
            sync(device)
        times = []
        for _ in range(reps):
            sync(device)
            t0 = timeit.default_timer()
            dist.all_reduce(data)
            sync(device)
            times.append(timeit.default_timer() - t0)
        per_rank = gather_list(sum(times) / len(times))
        if rank == 0:
            mean_ms = 1e3 * sum(per_rank) / len(per_rank)
            row = dict(backend=backend, world_size=world_size, size_mb=size_mb, mean_ms=round(mean_ms, 3),
                       min_rank_ms=round(1e3 * min(per_rank), 3), max_rank_ms=round(1e3 * max(per_rank), 3),
                       throughput_gb_s=round(size_mb / 1000 / (mean_ms / 1e3), 3))  # fmt: skip
            print(row, flush=True)
            append_row(out, row)
        del data
    cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    p.add_argument("--world-sizes", type=int, nargs="+", default=[2])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out", default="experiments/results/allreduce.csv")
    args = p.parse_args()
    for world_size in args.world_sizes:
        pick_port()
        mp.spawn(worker, args=(world_size, args.backend, args.out, args.reps), nprocs=world_size, join=True)
