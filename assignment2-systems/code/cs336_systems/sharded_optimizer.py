"""Optimizer state sharding (a simplified ZeRO stage 1).

Every rank holds the full model, but its local optimizer only owns (and keeps state for) about
1/world_size of the parameters.  After the local update each parameter is broadcast from its
owner so that all replicas stay identical.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any):
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_optimizer: Optimizer | None = None
        self._owners: dict[int, int] = {}  # id(param) -> owning rank
        self._load = [0] * self.world_size  # number of elements assigned to each rank so far
        self._local_group_index: list[int | None] = []  # global group -> local group (if any)
        # The base constructor calls add_param_group for every group.
        super().__init__(params, kwargs)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]

        # Greedy balancing by element count.  Every rank sees the parameters in the same
        # order, so all ranks compute the same assignment without communicating.
        local_params = []
        for p in group["params"]:
            owner = min(range(self.world_size), key=lambda r: (self._load[r], r))
            self._load[owner] += p.numel()
            self._owners[id(p)] = owner
            if owner == self.rank:
                local_params.append(p)

        if not local_params:
            self._local_group_index.append(None)
            return
        local_group = {k: v for k, v in group.items() if k != "params"}
        local_group["params"] = local_params
        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
        else:
            self.local_optimizer.add_param_group(local_group)
        self._local_group_index.append(len(self.local_optimizer.param_groups) - 1)

    def _push_hyperparameters(self) -> None:
        """Mirror hyper-parameter changes (e.g. from an LR schedule) into the local optimizer."""
        if self.local_optimizer is None:
            return
        for group, local_index in zip(self.param_groups, self._local_group_index):
            if local_index is None:
                continue
            local_group = self.local_optimizer.param_groups[local_index]
            for key, value in group.items():
                if key != "params":
                    local_group[key] = value

    def step(self, closure=None, **kwargs):
        loss = None
        self._push_hyperparameters()
        if self.local_optimizer is not None:
            loss = self.local_optimizer.step(closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Synchronise: each parameter is sent by the rank that just updated it.
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    dist.broadcast(p.data, src=self._owners[id(p)])
        return loss

    def local_state_bytes(self) -> int:
        """Bytes of optimizer state held on this rank."""
        if self.local_optimizer is None:
            return 0
        total = 0
        for state in self.local_optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total
