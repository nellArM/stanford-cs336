"""Scaled-down version of the leaderboard benchmark: wall-clock time of one full training
step (forward, loss, backward, AdamW) of a long-context, large-vocabulary model in bf16.

The official configuration (8B parameters, context 32,768, two B200s) does not fit on the
hardware used here, so the default configuration keeps its character (large vocabulary,
128-dim heads, long context) at a size that fits on one 48 GB card.

    uv run python -m cs336_systems.bench.leaderboard --variant baseline
    uv run python -m cs336_systems.bench.leaderboard --variant flash+ce+adamw
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
import triton

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.bench.common import append_row, gib
from cs336_systems.fast_step import FusedLMHeadCrossEntropy, InplaceAdamW, use_flash_attention


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="baseline", help="'baseline' or any '+'-joined subset of flash, ce, adamw")
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--d-ff", type=int, default=2752)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup-ms", type=int, default=2000)
    p.add_argument("--rep-ms", type=int, default=8000)
    p.add_argument("--out", default=None)
    p.add_argument("--retry-on-oom", action="store_true", help="exit 3 on OOM instead of recording it")
    args = p.parse_args()
    opts = set(args.variant.split("+")) - {"baseline"}

    row = dict(variant=args.variant, ctx_len=args.ctx_len, vocab_size=args.vocab_size, d_model=args.d_model, d_ff=args.d_ff,
               num_layers=args.num_layers, num_heads=args.num_heads, batch_size=args.batch_size)  # fmt: skip
    try:
        torch.manual_seed(0)
        model = BasicsTransformerLM(vocab_size=args.vocab_size, context_length=args.ctx_len, d_model=args.d_model,
                                    num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff).cuda()  # fmt: skip
        inputs, targets = torch.randint(high=args.vocab_size, size=(2, args.batch_size, args.ctx_len), device="cuda")
        optimizer = (InplaceAdamW if "adamw" in opts else AdamW)(model.parameters())
        attention = use_flash_attention() if "flash" in opts else nullcontext()

        def compute_loss():
            if "ce" in opts:
                x = model.token_embeddings(inputs)
                for layer in model.layers:
                    x = layer(x)
                return FusedLMHeadCrossEntropy.apply(model.ln_final(x), model.lm_head.weight, targets)
            return cross_entropy(model(inputs), targets)

        losses = []

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = compute_loss()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())

        with attention:
            train_step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ms = triton.testing.do_bench(train_step, warmup=args.warmup_ms, rep=args.rep_ms)
        row.update(status="ok", step_ms=round(ms, 1), peak_gib=round(gib(torch.cuda.max_memory_allocated()), 2),
                   first_loss=round(losses[0].item(), 4), second_loss=round(losses[1].item(), 4))  # fmt: skip
    except torch.OutOfMemoryError:
        if args.retry_on_oom:
            raise SystemExit(3)
        row.update(status="OOM", step_ms="", peak_gib=round(gib(torch.cuda.max_memory_allocated()), 2), first_loss="", second_loss="")
    print(row)
    if args.out:
        append_row(args.out, row)


if __name__ == "__main__":
    main()
