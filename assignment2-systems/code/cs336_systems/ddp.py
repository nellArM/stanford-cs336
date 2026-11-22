"""Distributed data parallel training.

``DDP`` wraps a module, broadcasts rank 0's weights at construction and averages gradients
across ranks every step.  Three synchronisation modes are provided so they can be benchmarked
against each other:

* ``"naive"``   - after the backward pass, one blocking all-reduce per parameter tensor.
* ``"flat"``    - after the backward pass, a single all-reduce over all gradients flattened
                  into one buffer.
* ``"overlap"`` - (default) an asynchronous all-reduce is issued for each parameter as soon as
                  its gradient has been accumulated, overlapping communication with the rest
                  of the backward pass.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDP(nn.Module):
    def __init__(self, module: nn.Module, mode: str = "overlap"):
        super().__init__()
        assert mode in ("naive", "flat", "overlap")
        self.module = module
        self.mode = mode
        self.world_size = dist.get_world_size()
        self._handles: list = []
        self._pending: list[nn.Parameter] = []
        self.comm_seconds = 0.0  # time spent blocked on gradient communication in the last step

        # Every rank starts from rank 0's parameters and buffers.
        with torch.no_grad():
            for tensor in self.module.state_dict().values():
                dist.broadcast(tensor, src=0)

        # parameters() already de-duplicates tied weights.
        self._params = [p for p in self.module.parameters() if p.requires_grad]
        if mode == "overlap":
            for p in self._params:
                p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def _on_grad_ready(self, param: nn.Parameter) -> None:
        # The gradient of this parameter is final for this backward pass: start averaging it
        # while autograd keeps working on earlier layers.
        self._handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True))
        self._pending.append(param)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _sync(self) -> None:
        if torch.cuda.is_available() and next(self.module.parameters()).is_cuda:
            torch.cuda.synchronize()

    @torch.no_grad()
    def finish_gradient_synchronization(self) -> None:
        self._sync()  # make sure the backward pass itself is not billed as communication
        start = time.perf_counter()
        if self.mode == "overlap":
            for handle in self._handles:
                handle.wait()
            for p in self._pending:
                p.grad.div_(self.world_size)
            self._handles.clear()
            self._pending.clear()
        elif self.mode == "naive":
            for p in self._params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(self.world_size)
        else:  # flat
            grads = [p.grad for p in self._params if p.grad is not None]
            if grads:
                flat = _flatten_dense_tensors(grads)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(self.world_size)
                for grad, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
                    grad.copy_(synced)
        self._sync()
        self.comm_seconds = time.perf_counter() - start
