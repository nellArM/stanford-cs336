"""Shared helpers for the benchmarking scripts."""

from __future__ import annotations

import csv
import os
from contextlib import nullcontext

import torch

from cs336_basics.model import BasicsTransformerLM

# Table 1 of the handout.
MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}
VOCAB_SIZE = 10_000
BATCH_SIZE = 4


def make_model(size: str, context_length: int, num_layers: int | None = None, device: str = "cuda") -> BasicsTransformerLM:
    cfg = dict(MODEL_SIZES[size])
    if num_layers is not None:
        cfg["num_layers"] = num_layers
    return BasicsTransformerLM(vocab_size=VOCAB_SIZE, context_length=context_length, **cfg).to(device)


def random_batch(batch_size: int, context_length: int, device: str = "cuda"):
    x = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    return x, y


def autocast_context(bf16: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else nullcontext()


def append_row(path: str, row: dict) -> None:
    """Append one result row to a CSV file, writing the header on first use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def gib(num_bytes: float) -> float:
    return num_bytes / 2**30
