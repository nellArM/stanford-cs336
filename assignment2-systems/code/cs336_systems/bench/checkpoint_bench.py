"""Peak memory of a forward+backward pass with one level of gradient checkpointing
(problem gradient_checkpointing (b)).

``--segment k`` wraps every k consecutive TransformerBlocks in one
``torch.utils.checkpoint.checkpoint`` call; ``--segment 0`` disables checkpointing.

    uv run python -m cs336_systems.bench.checkpoint_bench --size xl --num-layers 16 --context-length 2048 --segment 1
"""

from __future__ import annotations

import argparse
import timeit

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.nn_utils import cross_entropy
from cs336_systems.bench.common import MODEL_SIZES, append_row, autocast_context, gib, make_model, random_batch


def forward_checkpointed(model, token_ids, segment: int):
    x = model.token_embeddings(token_ids)
    layers = list(model.layers)
    if segment <= 0:
        for layer in layers:
            x = layer(x)
    else:
        for start in range(0, len(layers), segment):
            group = layers[start : start + segment]

            def run_group(h, group=group):
                for layer in group:
                    h = layer(h)
                return h

            x = checkpoint(run_group, x, use_reentrant=False)
    return model.lm_head(model.ln_final(x))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(MODEL_SIZES), default="xl")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--segment", type=int, default=1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--retry-on-oom", action="store_true", help="exit 3 on OOM instead of recording it")
    args = p.parse_args()

    row = dict(size=args.size, num_layers=args.num_layers or MODEL_SIZES[args.size]["num_layers"], context_length=args.context_length,
               batch_size=args.batch_size, segment=args.segment, bf16=int(args.bf16))  # fmt: skip
    try:
        model = make_model(args.size, args.context_length, args.num_layers)
        x, y = random_batch(args.batch_size, args.context_length)
        torch.cuda.synchronize()
        params_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = timeit.default_timer()
        with autocast_context(args.bf16):
            loss = cross_entropy(forward_checkpointed(model, x, args.segment), y)
        torch.cuda.synchronize()
        peak_forward = torch.cuda.max_memory_allocated()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = timeit.default_timer() - t0
        peak = torch.cuda.max_memory_allocated()
        row.update(status="ok", params_gib=round(gib(params_bytes), 2), peak_forward_gib=round(gib(peak_forward), 2), peak_gib=round(gib(peak), 2),
                   peak_minus_params_gib=round(gib(peak - params_bytes), 2), seconds=round(elapsed, 1))  # fmt: skip
    except torch.OutOfMemoryError:
        if args.retry_on_oom:
            raise SystemExit(3)
        row.update(status="OOM", params_gib="", peak_forward_gib="", peak_gib=round(gib(torch.cuda.max_memory_allocated()), 2), peak_minus_params_gib="", seconds="")
    print(row)
    if args.out:
        append_row(args.out, row)


if __name__ == "__main__":
    main()
