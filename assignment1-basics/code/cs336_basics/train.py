"""Train a Transformer LM on a tokenized dataset.

Every hyperparameter is a command-line flag.  Metrics are appended to
``<out-dir>/metrics.jsonl`` with both the gradient step and the wall-clock time,
and the full configuration is written to ``<out-dir>/config.json``.

Example (TinyStories base configuration):
    uv run python -m cs336_basics.train --name ts_base --train-data data/tinystories_train.npy \
        --valid-data data/tinystories_valid.npy --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--out-root", default="experiments/runs")
    p.add_argument("--train-data", required=True)
    p.add_argument("--valid-data", required=True)
    # model
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    p.add_argument("--norm-mode", choices=["pre", "post", "none"], default="pre")
    p.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    p.add_argument("--no-rope", action="store_true")
    p.add_argument("--tie-embeddings", action="store_true")
    # optimization
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--total-tokens", type=int, default=327_680_000, help="batch * steps * context")
    p.add_argument("--max-seconds", type=float, default=None, help="optional wall-clock budget for training")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr-frac", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # runtime
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--final-eval-batches", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=0, help="0 = only at the end")
    p.add_argument("--resume", default=None)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_batches, autocast, seed=1234) -> float:
    """Mean loss over a fixed set of random validation windows (same windows at every call)."""
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device, rng)
        with autocast():
            logits = model(x)
        losses.append(cross_entropy(logits.float(), y).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_float32_matmul_precision("high")

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
        use_rope=not args.no_rope,
        tie_embeddings=args.tie_embeddings,
        device=args.device,
    )
    # No weight decay on gains (1-D parameters); decay matrices only.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    total_steps = args.total_tokens // (args.batch_size * args.context_length)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    min_lr = args.lr * args.min_lr_frac

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)

    config = vars(args) | {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "non_embedding_params": model.num_parameters(),
        "total_params": sum(p.numel() for p in model.parameters()),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2), flush=True)

    train_model = model if args.no_compile else torch.compile(model)
    use_bf16 = args.dtype == "bf16" and args.device.startswith("cuda")

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    metrics_file = open(out_dir / "metrics.jsonl", "a")

    def log(record: dict) -> None:
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    def sync() -> None:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    train_seconds = 0.0  # wall-clock spent in training steps only (evaluation excluded)
    status = "completed"
    step = start_step
    last_tick = time.perf_counter()
    for step in range(start_step, total_steps):
        if args.max_seconds is not None:
            # Time-budgeted run: the schedule follows the fraction of the budget used.
            progress = min(1.0, train_seconds / args.max_seconds)
            lr = get_lr_cosine_schedule(int(progress * total_steps), args.lr, min_lr, warmup_steps, total_steps)
        else:
            lr = get_lr_cosine_schedule(step, args.lr, min_lr, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device, rng)
        with autocast():
            logits = train_model(x)
        loss = cross_entropy(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_gradient(model.parameters(), args.max_grad_norm)
        optimizer.step()

        is_log = (step + 1) % args.log_every == 0
        is_eval = (step + 1) % args.eval_every == 0 or step + 1 == total_steps
        if is_log or is_eval:
            loss_value = loss.item()
            sync()
            now = time.perf_counter()
            train_seconds += now - last_tick
            record = {
                "step": step + 1,
                "tokens": (step + 1) * args.batch_size * args.context_length,
                "train_seconds": round(train_seconds, 2),
                "train_loss": loss_value,
                "grad_norm": grad_norm,
                "lr": lr,
            }
            if not math.isfinite(loss_value):
                status = "diverged"
                log(record | {"status": status})
                print(f"step {step + 1}: loss is not finite, stopping", flush=True)
                break
            if is_eval:
                record["val_loss"] = evaluate(
                    train_model, valid_data, args.batch_size, args.context_length, args.device, args.eval_batches, autocast
                )
            log(record)
            print(record, flush=True)
            last_tick = time.perf_counter()
            if args.max_seconds is not None and train_seconds >= args.max_seconds:
                status = "time_budget_reached"
                break

        if args.checkpoint_every and (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step + 1, out_dir / "checkpoint.pt")
            last_tick = time.perf_counter()

    final = {"status": status, "final_step": step + 1, "train_seconds": round(train_seconds, 2)}
    if status != "diverged":
        final["final_val_loss"] = evaluate(
            train_model, valid_data, args.batch_size, args.context_length, args.device, args.final_eval_batches, autocast
        )
        save_checkpoint(model, optimizer, step + 1, out_dir / "checkpoint.pt", config=config)
    log(final)
    print(final, flush=True)
    with open(out_dir / "final.json", "w") as f:
        json.dump(final | {"name": args.name}, f, indent=2)
    metrics_file.close()


if __name__ == "__main__":
    main()
