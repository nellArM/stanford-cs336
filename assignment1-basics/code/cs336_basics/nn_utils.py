"""Stateless numerical helpers: activations, softmax, cross-entropy, gradient clipping."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    # Subtracting the max leaves the result unchanged but keeps exp() from overflowing.
    # The shift is a constant as far as the gradient is concerned (softmax is shift-invariant),
    # so it is detached; this also keeps -inf (masked) entries out of the max's backward pass.
    shifted = x - x.amax(dim=dim, keepdim=True).detach()
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def log_softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True).detach()
    return shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean negative log-likelihood. ``logits``: (..., vocab), ``targets``: (...) of class indices."""
    log_probs = log_softmax(logits, dim=-1)
    picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -picked.mean()


def perplexity(logits: Tensor, targets: Tensor) -> Tensor:
    return torch.exp(cross_entropy(logits, targets))


@torch.no_grad()
def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> float:
    """Scale all gradients in place so their combined L2 norm is at most ``max_l2_norm``.

    Returns the norm before clipping.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
    return float(total_norm)
