"""Autograd residual accounting for a TransformerBlock (problem memory_profiling (f)).

Measures, for one configuration:
* which operations save how many bytes for the backward pass inside a single block
  (``torch.autograd.graph.saved_tensors_hooks``, grouped by the grad_fn that created the tensor);
* how much allocated memory grows per block in the forward pass and how it changes per block
  in the backward pass (module hooks reading ``torch.cuda.memory_allocated``).

    uv run python -m cs336_systems.bench.residuals --size xl --num-layers 4 --context-length 512
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from cs336_basics.nn_utils import cross_entropy
from cs336_systems.bench.common import MODEL_SIZES, autocast_context, make_model, random_batch

MIB = 2**20


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(MODEL_SIZES), default="xl")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    model = make_model(args.size, args.context_length, args.num_layers)
    x, y = random_batch(args.batch_size, args.context_length)
    d_model = MODEL_SIZES[args.size]["d_model"]
    act_mib = args.batch_size * args.context_length * d_model * 4 / MIB
    print(f"config: {args.size} layers={args.num_layers} ctx={args.context_length} batch={args.batch_size} bf16={args.bf16}")
    print(f"one residual-stream activation (fp32): {act_mib:.1f} MiB")

    # ---- (1) saved tensors of one block, grouped by producing op --------------------------
    by_op: dict[str, int] = defaultdict(int)
    seen: set[int] = set()
    # Views of weights (einsum reshapes them) share storage with the parameters: not activations.
    param_ptrs = {p.data_ptr() for p in model.parameters()}

    def pack(t):
        if isinstance(t, torch.nn.Parameter) or t.data_ptr() in seen or t.data_ptr() in param_ptrs:
            return t  # parameters / weight views are not activations; a storage is only counted once
        seen.add(t.data_ptr())
        name = type(t.grad_fn).__name__ if t.grad_fn is not None else "leaf/input"
        by_op[f"{name} {tuple(t.shape)} {str(t.dtype).replace('torch.', '')}"] += t.numel() * t.element_size()
        return t

    block = model.layers[0]
    h = torch.randn(args.batch_size, args.context_length, d_model, device="cuda", requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), autocast_context(args.bf16):
        out = block(h)
    total = sum(by_op.values())
    print(f"\nsaved for backward by ONE TransformerBlock: {total / MIB:.1f} MiB (unique storages)")
    for name, size in sorted(by_op.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {size / MIB:9.1f} MiB  {100 * size / total:5.1f}%  {name}")
    del out, h
    torch.cuda.empty_cache()

    # ---- (2) allocated memory after each block, forward and backward ---------------------
    fwd, bwd = [], []
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda m, a, o, i=i: fwd.append((i, torch.cuda.memory_allocated())))
        layer.register_full_backward_hook(lambda m, gi, go, i=i: bwd.append((i, torch.cuda.memory_allocated())))

    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    with autocast_context(args.bf16):
        logits = model(x)
        loss = cross_entropy(logits, y)
    del logits
    before_backward = torch.cuda.memory_allocated()
    loss.backward()
    print(f"\nallocated before forward {start / MIB:.0f} MiB, before backward {before_backward / MIB:.0f} MiB")
    prev = start
    for i, mem in fwd:
        print(f"  forward  block {i}: +{(mem - prev) / MIB:8.1f} MiB")
        prev = mem
    prev = None
    for i, mem in bwd:
        delta = "" if prev is None else f"{(mem - prev) / MIB:+8.1f} MiB vs previous hook"
        print(f"  backward block {i}: allocated {mem / MIB:9.1f} MiB {delta}")
        prev = mem
    block_params = sum(p.numel() * p.element_size() for p in model.layers[0].parameters())
    print(f"parameter (= gradient) bytes of one block: {block_params / MIB:.1f} MiB")


if __name__ == "__main__":
    main()
