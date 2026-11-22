"""Train one model for the local scaling study, mirroring the training API's conventions.

  * seq_len 512, fixed deterministic data order, no repetition (the stream is consumed from
    its start in non-overlapping 512-token sequences; the label of the last position is the
    first token of the next sequence),
  * AdamW (weight decay 0.01 on all parameters, betas (0.9, 0.95), grad clip 1.0) with linear
    warmup (5 %) and cosine decay to 0.1 x peak,
  * training is split into n_evals equal chunks; after each chunk the model is evaluated on the
    2**18-token validation set; the final validation loss is the last entry,
  * bfloat16 autocast and torch.compile.

Every run appends JSON lines to experiments/runs/<name>.jsonl and ends with a "final" record.
A checkpoint is written at every evaluation, so an interrupted run resumes where it stopped.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run --extra study python -m cs336_scaling.local_study.train \
      --name demo --d-model 256 --n-layer 4 --tokens 20e6 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_scaling.local_study import flops as F
from cs336_scaling.local_study.model import CausalLM

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "dclm"
RUN_DIR = ROOT / "experiments" / "runs"
CKPT_DIR = ROOT / "experiments" / "checkpoints"
API_VAL_TOKENS = 2**18


class TokenStream:
    """The concatenation of the tokenized training shards, read through memory maps."""

    def __init__(self, data_dir: Path = DATA_DIR):
        manifest = json.loads((data_dir / "manifest.json").read_text())
        self.arrays = [np.load(data_dir / entry["file"], mmap_mode="r") for entry in manifest["train"]]
        self.offsets = np.cumsum([0] + [len(a) for a in self.arrays])
        self.total = int(self.offsets[-1])

    def read(self, start: int, length: int) -> np.ndarray:
        if start + length > self.total:
            raise ValueError(f"need tokens up to {start + length:,} but only {self.total:,} are prepared")
        out = np.empty(length, dtype=np.int64)
        filled = 0
        shard = int(np.searchsorted(self.offsets, start, side="right") - 1)
        while filled < length:
            local = start + filled - int(self.offsets[shard])
            take = min(length - filled, len(self.arrays[shard]) - local)
            out[filled : filled + take] = self.arrays[shard][local : local + take]
            filled += take
            shard += 1
        return out

    def batch(self, step: int, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        n = batch_size * F.SEQ_LEN
        tokens = torch.from_numpy(self.read(step * n, n + 1)).to(device, non_blocking=True)
        return tokens[:-1].view(batch_size, F.SEQ_LEN), tokens[1:].view(batch_size, F.SEQ_LEN)


def lr_at(step: int, total_steps: int, peak: float, warmup_frac: float, final_frac: float) -> float:
    warmup = int(total_steps * warmup_frac)
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    end = peak * final_frac
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, val_tokens: torch.Tensor, n_tokens: int, batch_size: int) -> float:
    model.eval()
    n_seq = n_tokens // F.SEQ_LEN
    inputs = val_tokens[: n_seq * F.SEQ_LEN].view(n_seq, F.SEQ_LEN)
    labels = val_tokens[1 : n_seq * F.SEQ_LEN + 1].view(n_seq, F.SEQ_LEN)
    total = 0.0
    for i in range(0, n_seq, batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(inputs[i : i + batch_size], labels[i : i + batch_size])
        total += loss.item() * len(inputs[i : i + batch_size])
    model.train()
    return total / n_seq


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--n-layer", type=int, required=True)
    p.add_argument("--tokens", type=float, required=True, help="training tokens (rounded down to whole eval chunks)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--final-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-evals", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tie", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--group", default="misc", help="free-form tag used by the fitting scripts")
    p.add_argument("--budget", type=float, default=None, help="nominal compute budget this run belongs to")
    args = p.parse_args()

    assert torch.cuda.is_available(), "a CUDA device is required"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    batch_size = args.batch_size or F.choose_batch_size(args.tokens)
    total_tokens = F.round_tokens(args.tokens, batch_size, args.n_evals)
    total_steps = total_tokens // (batch_size * F.SEQ_LEN)
    steps_per_eval = total_steps // args.n_evals

    config = F.model_config(args.d_model, args.n_layer)
    if args.tie:
        config = type(config)(**{**config.to_dict(), "tie_word_embeddings": True})
    torch.manual_seed(args.seed)
    model = CausalLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=True,
    )
    run_model = model if args.no_compile else torch.compile(model)

    stream = TokenStream()
    val_tokens = torch.from_numpy(np.load(DATA_DIR / "val.npy").astype(np.int64)).to(device)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / f"{args.name}.jsonl"
    ckpt_path = CKPT_DIR / f"{args.name}.pt"

    header = {
        "type": "config",
        "name": args.name,
        "group": args.group,
        "budget": args.budget,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "n_params": F.non_embedding_params(args.d_model, args.n_layer),
        "n_eff": F.effective_params(args.d_model, args.n_layer),
        "n_params_actual": sum(p.numel() for p in model.parameters()),
        "total_train_tokens": total_tokens,
        "flops": F.flops(args.d_model, args.n_layer, total_tokens),
        "batch_size": batch_size,
        "total_steps": total_steps,
        "n_evals": args.n_evals,
        "lr": args.lr,
        "warmup_frac": args.warmup_frac,
        "final_lr_frac": args.final_lr_frac,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "architecture": config.to_dict(),
        "gpu": torch.cuda.get_device_name(0),
    }

    step = 0
    val_losses: list[float] = []
    train_seconds = 0.0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, val_losses, train_seconds = state["step"], state["val_losses"], state["train_seconds"]
        print(f"resumed {args.name} at step {step}/{total_steps}", flush=True)
    else:
        with open(log_path, "w") as f:
            f.write(json.dumps(header) + "\n")

    def log(record: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(json.dumps(header), flush=True)
    model.train()
    chunk_start = time.perf_counter()
    running: list[torch.Tensor] = []
    while step < total_steps:
        lr = lr_at(step, total_steps, args.lr, args.warmup_frac, args.final_lr_frac)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, labels = stream.batch(step, batch_size, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = run_model.loss(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        running.append(loss.detach())
        step += 1

        if step % steps_per_eval == 0:
            train_loss = torch.stack(running).mean().item()
            running = []
            if not math.isfinite(train_loss):
                log({"type": "final", "status": "diverged", "step": step, "val_losses": val_losses})
                print("diverged", flush=True)
                return
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - chunk_start
            val_loss = evaluate(run_model, val_tokens, API_VAL_TOKENS, args.val_batch_size)
            val_losses.append(val_loss)
            tokens_seen = step * batch_size * F.SEQ_LEN
            record = {
                "type": "eval",
                "step": step,
                "tokens": tokens_seen,
                "flops": F.flops(args.d_model, args.n_layer, tokens_seen),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_seconds": train_seconds,
                "tokens_per_second": tokens_seen / train_seconds,
            }
            log(record)
            print(json.dumps(record), flush=True)
            if step < total_steps:
                tmp = ckpt_path.with_suffix(".tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "val_losses": val_losses,
                        "train_seconds": train_seconds,
                    },
                    tmp,
                )
                os.replace(tmp, ckpt_path)
            chunk_start = time.perf_counter()

    big_val = evaluate(run_model, val_tokens, len(val_tokens) - 1, args.val_batch_size)
    final = {
        "type": "final",
        "status": "completed",
        "final_val_loss": val_losses[-1],
        "final_val_loss_2M_tokens": big_val,
        "val_losses": val_losses,
        "train_seconds": train_seconds,
        "tokens_per_second": total_tokens / train_seconds,
        "flops_per_second": F.flops(args.d_model, args.n_layer, total_tokens) / train_seconds,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    log(final)
    print(json.dumps(final), flush=True)
    ckpt_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
