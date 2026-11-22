"""AdamW and the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                # Bias-corrected step size, then decoupled weight decay, then the Adam update.
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup to the max LR, cosine decay to the min LR until ``cosine_cycle_iters``, then constant."""
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate
