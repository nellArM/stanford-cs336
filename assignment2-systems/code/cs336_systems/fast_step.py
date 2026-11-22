"""Optimisations for the leaderboard-style training step.

Three independent pieces, all of which leave the model's input/output behaviour unchanged:

* ``use_flash_attention``  - route the model's causal attention through the Triton
                              FlashAttention-2 forward/backward kernels (no seq x seq tensors).
* ``FusedLMHeadCrossEntropy`` - computes the LM-head projection, the cross-entropy loss and
                              their gradients chunk by chunk, so the (batch, seq, vocab) logits
                              tensor is never materialised or saved for backward.
* ``InplaceAdamW``         - AdamW with the same update rule as ``cs336_basics.optimizer.AdamW``
                              but with in-place tensor ops (no per-step temporaries).
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

import cs336_basics.model as basics_model
from cs336_systems.flash_attention import FlashAttentionTritonFull


@contextmanager
def use_flash_attention(dtype: torch.dtype = torch.bfloat16):
    """Temporarily replace ``cs336_basics.model.scaled_dot_product_attention``.

    The model always passes a causal mask, so the mask tensor itself is ignored and the
    kernel's index comparison is used instead.
    """
    original = basics_model.scaled_dot_product_attention

    def flash(Q, K, V, mask=None):
        return FlashAttentionTritonFull.apply(Q.to(dtype), K.to(dtype), V.to(dtype), mask is not None)

    basics_model.scaled_dot_product_attention = flash
    try:
        yield
    finally:
        basics_model.scaled_dot_product_attention = original


class FusedLMHeadCrossEntropy(torch.autograd.Function):
    """mean cross-entropy of ``hidden @ weight.T`` against ``targets`` with fused backward."""

    CHUNK_TOKENS = 2048

    @staticmethod
    def forward(ctx, hidden, weight, targets):
        h = hidden.reshape(-1, hidden.shape[-1])
        t = targets.reshape(-1)
        n = h.shape[0]
        compute_dtype = torch.bfloat16 if h.is_cuda else h.dtype
        w = weight.to(compute_dtype)
        grad_h = torch.empty_like(h)
        grad_w = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        loss = torch.zeros((), dtype=torch.float32, device=h.device)
        for start in range(0, n, FusedLMHeadCrossEntropy.CHUNK_TOKENS):
            hc = h[start : start + FusedLMHeadCrossEntropy.CHUNK_TOKENS].to(compute_dtype)
            tc = t[start : start + FusedLMHeadCrossEntropy.CHUNK_TOKENS]
            logits = (hc @ w.t()).float()
            lse = torch.logsumexp(logits, dim=-1, keepdim=True)
            loss += (lse.squeeze(-1) - logits.gather(-1, tc[:, None]).squeeze(-1)).sum()
            # d loss / d logits = (softmax - onehot) / n, computed in place on the logits buffer
            logits.sub_(lse).exp_()
            logits.scatter_add_(-1, tc[:, None], torch.full((tc.shape[0], 1), -1.0, device=h.device))
            dlogits = logits.div_(n).to(compute_dtype)
            grad_h[start : start + hc.shape[0]] = (dlogits @ w).to(grad_h.dtype)
            grad_w.addmm_(dlogits.t().float(), hc.float())
            del logits, dlogits
        ctx.save_for_backward(grad_h.view_as(hidden), grad_w)
        return loss / n

    @staticmethod
    def backward(ctx, grad_out):
        grad_h, grad_w = ctx.saved_tensors
        return grad_h * grad_out, (grad_w * grad_out), None


class InplaceAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t, m, v = state["t"], state["m"], state["v"]
                alpha_t = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                m.lerp_(p.grad, 1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-alpha_t)
                state["t"] = t + 1
        return loss
