"""Fully-sharded data parallel (FSDP) training.

Each rank stores only its 1/world_size slice (along dim 0) of every ``Linear`` and ``Embedding``
weight; the slice is the ``nn.Parameter`` the optimizer sees, so gradients and optimizer state
are sharded too.  Small parameters (norm gains) stay replicated and are synchronised DDP-style.

Life of a sharded weight during one step:

* forward  - the full weight is all-gathered (cast to ``compute_dtype`` first, if given), used,
             and dropped.  Gathers are scheduled ahead of use: the gather for call-site ``i``
             starts as soon as call-site ``i-2`` has finished its forward, so at most one
             gathered weight is "in flight" while another one is being used.
* backward - linear layers all-gather the weight again (it was not kept), compute the input
             gradient and the full weight gradient, and reduce-scatter the latter so that every
             rank receives the summed gradient of just its slice.  All communication is
             asynchronous; ``finish_gradient_synchronization`` waits for it and writes ``.grad``.

The wrapped modules stay instances of their original classes (only ``forward`` is overridden per
instance and ``weight`` is replaced by the slice), so ``named_parameters()`` keeps its names.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
from torch import nn

from cs336_basics.model import Embedding, Linear


class _ShardedWeight:
    """Bookkeeping for one sharded parameter."""

    def __init__(self, full: torch.Tensor, rank: int, world_size: int):
        self.full_rows, self.cols = full.shape
        self.rows_per_rank = -(-self.full_rows // world_size)  # ceil
        padded_rows = self.rows_per_rank * world_size
        start = rank * self.rows_per_rank
        local = torch.zeros(self.rows_per_rank, self.cols, dtype=full.dtype, device=full.device)
        available = max(0, min(self.full_rows, start + self.rows_per_rank) - start)
        if available:
            local[:available] = full.data[start : start + available]
        self.shard = nn.Parameter(local, requires_grad=full.requires_grad)
        self.padded_rows = padded_rows
        self.pending = None  # (handle, buffer) of an in-flight / completed all-gather
        self.grad_ops: list = []  # in-flight reduce-scatters: (handle, buffer)


class _FSDPLinearFn(torch.autograd.Function):
    """y = x W^T where W is supplied by FSDP and deliberately *not* saved for backward."""

    @staticmethod
    def forward(ctx, x, shard, fsdp, sw, site_index):
        W = fsdp._take(sw)
        in_dtype = x.dtype
        if torch.is_autocast_enabled(x.device.type):
            compute_dtype = torch.get_autocast_dtype(x.device.type)
        else:
            compute_dtype = W.dtype
        xc = x.to(compute_dtype)
        y = xc @ W.to(compute_dtype).t()
        ctx.save_for_backward(xc)
        ctx.fsdp, ctx.sw, ctx.site_index, ctx.in_dtype = fsdp, sw, site_index, in_dtype
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x,) = ctx.saved_tensors
        fsdp, sw = ctx.fsdp, ctx.sw
        grad_x = None
        if ctx.needs_input_grad[0]:
            W = fsdp._take(sw, backward=True)  # all-gather again: the forward copy was freed
            fsdp._prefetch_backward(ctx.site_index - 1)
            grad_x = (grad_y @ W.to(grad_y.dtype)).to(ctx.in_dtype)
            del W
        else:
            fsdp._discard(sw)
        if ctx.needs_input_grad[1]:
            g2 = grad_y.reshape(-1, grad_y.shape[-1])
            x2 = x.reshape(-1, x.shape[-1]).to(g2.dtype)
            fsdp._reduce_scatter(sw, g2.t() @ x2)
        fsdp._prefetch_backward(ctx.site_index - 2)
        return grad_x, None, None, None, None


class _FSDPEmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, shard, fsdp, sw, site_index):
        W = fsdp._take(sw)
        ctx.token_ids, ctx.fsdp, ctx.sw, ctx.site_index = token_ids, fsdp, sw, site_index
        return W[token_ids]

    @staticmethod
    def backward(ctx, grad_out):
        fsdp, sw = ctx.fsdp, ctx.sw
        if ctx.needs_input_grad[1]:
            flat_ids = ctx.token_ids.reshape(-1)
            flat_grad = grad_out.reshape(-1, grad_out.shape[-1])
            full_grad = torch.zeros(sw.full_rows, sw.cols, dtype=flat_grad.dtype, device=flat_grad.device)
            full_grad.index_add_(0, flat_ids, flat_grad)
            fsdp._reduce_scatter(sw, full_grad)
        fsdp._prefetch_backward(ctx.site_index - 2)
        return None, None, None, None, None


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None, prefetch: bool = True):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.prefetch = prefetch
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self._nccl = dist.get_backend() == "nccl"

        # All ranks start from rank 0's weights.
        with torch.no_grad():
            for tensor in self.module.state_dict().values():
                dist.broadcast(tensor, src=0)

        # ---- shard Linear / Embedding weights -------------------------------------------
        self._weights: dict[int, _ShardedWeight] = {}  # id(original parameter) -> record
        self._by_shard: dict[int, _ShardedWeight] = {}  # id(shard parameter) -> record
        for mod in self.module.modules():
            if not isinstance(mod, (Linear, Embedding)):
                continue
            original = mod._parameters["weight"]
            sw = self._weights.get(id(original))
            if sw is None:
                sw = _ShardedWeight(original, self.rank, self.world_size)
                self._weights[id(original)] = sw
                self._by_shard[id(sw.shard)] = sw
            mod._parameters["weight"] = sw.shard  # tied modules share the same shard
            self._patch_forward(mod, sw)
        self._weights = {id(sw.shard): sw for sw in self._by_shard.values()}  # drop refs to full weights

        # ---- replicated parameters: average gradients like DDP ----------------------------
        self._replicated = [p for p in self.module.parameters() if p.requires_grad and id(p) not in self._by_shard]
        self._replicated_handles: list = []
        self._replicated_pending: list[nn.Parameter] = []
        for p in self._replicated:
            p.register_post_accumulate_grad_hook(self._on_replicated_grad)

        # ---- call-site schedule, learnt from the previous forward pass --------------------
        self._order: list[tuple[_ShardedWeight, bool]] = []  # (weight, is_linear) per call-site
        self._current: list[tuple[_ShardedWeight, bool]] = []
        self.stats = {"gather_hits": 0, "gather_misses": 0, "gather_ready": 0, "wait_seconds": 0.0}
        self.measure_waits = False  # when True, _take synchronises CUDA to time residual waits

    # ------------------------------------------------------------------ module patching --
    def _patch_forward(self, mod: nn.Module, sw: _ShardedWeight) -> None:
        is_linear = isinstance(mod, Linear)
        fn = _FSDPLinearFn if is_linear else _FSDPEmbeddingFn

        def forward(x):
            site_index = len(self._current)
            self._current.append((sw, is_linear))
            out = fn.apply(x, sw.shard, self, sw, site_index)
            # This call-site is done: the gather for the site two ahead may start now.
            self._prefetch_forward(site_index + 2)
            return out

        mod.forward = forward

    # -------------------------------------------------------------------- communication --
    def _launch_gather(self, sw: _ShardedWeight) -> None:
        if sw.pending is not None:
            return
        src = sw.shard.detach()
        if self.compute_dtype is not None:
            src = src.to(self.compute_dtype)  # cast *before* communicating: half the bytes
        if self._nccl:
            out = torch.empty(sw.padded_rows, sw.cols, dtype=src.dtype, device=src.device)
            handle = dist.all_gather_into_tensor(out, src, async_op=True)
        else:
            out = [torch.empty_like(src) for _ in range(self.world_size)]
            handle = dist.all_gather(out, src, async_op=True)
        sw.pending = (handle, out)

    def _take(self, sw: _ShardedWeight, backward: bool = False) -> torch.Tensor:
        """Return the full weight, waiting for (or starting) its all-gather."""
        hit = sw.pending is not None
        self.stats["gather_hits" if hit else "gather_misses"] += 1
        if hit and sw.pending[0].is_completed():
            self.stats["gather_ready"] += 1  # the prefetched all-gather had already finished
        if self.measure_waits and torch.cuda.is_available():
            torch.cuda.synchronize()  # finish compute first so only communication is timed
            start = time.perf_counter()
        if not hit:
            self._launch_gather(sw)
        handle, out = sw.pending
        sw.pending = None
        handle.wait()
        if self.measure_waits and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.stats["wait_seconds"] += time.perf_counter() - start
        full = out if self._nccl else torch.cat(out, dim=0)
        return full[: sw.full_rows] if sw.padded_rows != sw.full_rows else full

    def _discard(self, sw: _ShardedWeight) -> None:
        if sw.pending is not None:
            sw.pending[0].wait()
            sw.pending = None

    def _prefetch_forward(self, site_index: int) -> None:
        if self.prefetch and 0 <= site_index < len(self._order):
            self._launch_gather(self._order[site_index][0])

    def _prefetch_backward(self, site_index: int) -> None:
        # Only linear layers need their weight in the backward pass.
        if self.prefetch and 0 <= site_index < len(self._current):
            sw, is_linear = self._current[site_index]
            if is_linear:
                self._launch_gather(sw)

    def _reduce_scatter(self, sw: _ShardedWeight, full_grad: torch.Tensor) -> None:
        if sw.padded_rows != sw.full_rows:
            pad = torch.zeros(sw.padded_rows - sw.full_rows, sw.cols, dtype=full_grad.dtype, device=full_grad.device)
            full_grad = torch.cat([full_grad, pad], dim=0)
        full_grad = full_grad.contiguous()
        if self._nccl:
            out = torch.empty(sw.rows_per_rank, sw.cols, dtype=full_grad.dtype, device=full_grad.device)
            handle = dist.reduce_scatter_tensor(out, full_grad, op=dist.ReduceOp.SUM, async_op=True)
            sw.grad_ops.append((handle, out, full_grad))  # keep the input alive until completion
        else:
            # Gloo has no reduce-scatter: all-reduce and keep only this rank's slice.
            handle = dist.all_reduce(full_grad, op=dist.ReduceOp.SUM, async_op=True)
            start = self.rank * sw.rows_per_rank
            sw.grad_ops.append((handle, full_grad[start : start + sw.rows_per_rank], full_grad))

    def _on_replicated_grad(self, param: nn.Parameter) -> None:
        self._replicated_handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))
        self._replicated_pending.append(param)

    # ----------------------------------------------------------------------- public API --
    def forward(self, *inputs, **kwargs):
        self._current = []
        if self.prefetch:
            self._prefetch_forward(0)
            self._prefetch_forward(1)
        out = self.module(*inputs, **kwargs)
        if self.prefetch and torch.is_grad_enabled() and self._current:
            # Get a head start on the backward pass, which visits the call-sites in reverse.
            self._prefetch_backward(len(self._current) - 1)
        self._order = list(self._current)
        return out

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        for sw in self._by_shard.values():
            if sw.grad_ops:
                total = None
                for handle, piece, _keep_alive in sw.grad_ops:
                    handle.wait()
                    piece = piece.to(sw.shard.dtype)  # master weights / optimizer stay in fp32
                    total = piece if total is None else total + piece
                total = total / self.world_size
                sw.shard.grad = total if sw.shard.grad is None else sw.shard.grad + total
                sw.grad_ops.clear()
            # Any gathered weight still around would be stale after the optimizer step.
            self._discard(sw)
        for handle in self._replicated_handles:
            handle.wait()
        for p in self._replicated_pending:
            p.grad.div_(self.world_size)
        self._replicated_handles.clear()
        self._replicated_pending.clear()

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """All-gather every sharded parameter (in fp32); replicated parameters are returned as-is."""
        full: dict[str, torch.Tensor] = {}
        for name, param in self.module.named_parameters():
            sw = self._by_shard.get(id(param))
            if sw is None:
                full[name] = param.data
                continue
            pieces = [torch.empty_like(param.data) for _ in range(self.world_size)]
            dist.all_gather(pieces, param.data.contiguous())
            full[name] = torch.cat(pieces, dim=0)[: sw.full_rows]
        return full

    def sharded_bytes(self) -> int:
        return sum(sw.shard.numel() * sw.shard.element_size() for sw in self._by_shard.values())
