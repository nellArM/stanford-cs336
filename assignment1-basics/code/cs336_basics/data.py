"""Batch sampling and checkpointing."""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows; targets are the inputs shifted by one token."""
    rng = rng or np.random.default_rng()
    starts = rng.integers(0, len(dataset) - context_length, size=batch_size)
    offsets = starts[:, None] + np.arange(context_length + 1)[None, :]
    # Fancy indexing copies just these windows out of a (possibly memory-mapped) array.
    window = torch.from_numpy(np.asarray(dataset[offsets], dtype=np.int64))
    window = window.to(device, non_blocking=True)
    return window[:, :-1], window[:, 1:]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
    **extra,
) -> None:
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration, **extra},
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
