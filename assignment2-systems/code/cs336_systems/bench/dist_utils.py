"""Helpers shared by the multi-process benchmarks."""

from __future__ import annotations

import csv
import os

import torch
import torch.distributed as dist


def pick_port() -> None:
    """Call in the parent before ``mp.spawn``: reserve a free rendezvous port for this run."""
    import socket

    with socket.socket() as s:
        s.bind(("localhost", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])


def setup(rank: int, world_size: int, backend: str, port: int = 29517) -> str:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(port))
    # Direct GPU-to-GPU (PCIe peer-to-peer) transfers hang on this machine: the NCCL kernels
    # never complete.  Disabling P2P makes NCCL route data through host shared memory instead.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    if backend == "nccl":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup() -> None:
    dist.barrier()
    dist.destroy_process_group()


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def gather_mean(value: float) -> float:
    """Average a Python float over all ranks."""
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return sum(values) / len(values)


def gather_list(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def dump_gpu_trace(prof, path: str) -> None:
    """Write every GPU kernel / memcpy of a torch.profiler run as (name, start_us, dur_us)."""
    rows = []
    for evt in prof.events():
        if str(evt.device_type).endswith("CUDA") and evt.time_range.end > evt.time_range.start:
            rows.append((evt.name, evt.time_range.start, evt.time_range.end - evt.time_range.start))
    rows.sort(key=lambda r: r[1])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "start_us", "dur_us"])
        writer.writerows(rows)
