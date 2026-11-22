"""Run every training experiment of the assignment sequentially on one GPU.

Restart-safe: a run whose ``final.json`` exists is skipped.  The best learning rate
found by the sweep is reused automatically by the later stages.

    CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_all.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
TS = ["--train-data", "data/tinystories_train.npy", "--valid-data", "data/tinystories_valid.npy"]
OWT = ["--train-data", "data/owt_train.npy", "--valid-data", "data/owt_valid.npy", "--vocab-size", "32000"]
LOW = 40_000_000 // (128 * 256) * (128 * 256)  # the handout's low-resource budget (~40M tokens)
FULL = 327_680_000


def run(name: str, *args: str) -> dict:
    final = RUNS / name / "final.json"
    if not final.exists():
        cmd = [sys.executable, "-m", "cs336_basics.train", "--name", name, "--out-root", str(RUNS), *args]
        print(">>>", " ".join(cmd), flush=True)
        with open(ROOT / "logs" / f"{name}.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if not final.exists():  # crashed (e.g. out of memory): record it and move on
        (RUNS / name).mkdir(parents=True, exist_ok=True)
        tail = (ROOT / "logs" / f"{name}.log").read_text()[-600:]
        final.write_text(json.dumps({"name": name, "status": "crashed", "log_tail": tail}, indent=2))
    return json.loads(final.read_text())


def val(result: dict) -> float:
    return result.get("final_val_loss", float("inf"))


def main() -> None:
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    low = ["--total-tokens", str(LOW), "--eval-every", "100", "--log-every", "20"]

    # 1. Learning-rate sweep (low-resource budget), including values that diverge.
    lrs = ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2", "3e-2", "1e-1"]
    sweep = {lr: run(f"lr_{lr}", *TS, *low, "--lr", lr) for lr in lrs}
    best_lr = min(sweep, key=lambda lr: val(sweep[lr]))
    print("best lr:", best_lr, {lr: val(r) for lr, r in sweep.items()}, flush=True)
    best = float(best_lr)

    # 2. Full-budget TinyStories run at the best learning rate.
    run("ts_full", *TS, "--total-tokens", str(FULL), "--lr", best_lr, "--eval-every", "500")

    # 3. Batch-size sweep at a fixed token budget; LR scaled with sqrt(batch/128).
    #    Batch size 1 would need 156k steps, so it gets a 15-minute wall-clock cap instead.
    for bs in [1, 16, 64, 256, 512, 1024]:
        lr = f"{best * (bs / 128) ** 0.5:.2e}"
        extra = ["--max-seconds", "900", "--eval-every", "5000", "--log-every", "500"] if bs == 1 else []
        steps = LOW // (bs * 256)
        every = ["--eval-every", str(max(10, steps // 12)), "--log-every", str(max(5, steps // 60))] if bs != 1 else []
        run(f"bs_{bs}", *TS, "--total-tokens", str(LOW), "--batch-size", str(bs), "--lr", lr, *every, *extra)

    # 4. Ablations (low-resource budget, compared against lr_<best>).
    for frac, tag in [(1, "best"), (1 / 3, "third"), (1 / 10, "tenth"), (1 / 30, "thirtieth")]:
        run(f"abl_nonorm_{tag}", *TS, *low, "--norm-mode", "none", "--lr", f"{best * frac:.2e}")
    run("abl_postnorm", *TS, *low, "--norm-mode", "post", "--lr", best_lr)
    run("abl_nope", *TS, *low, "--no-rope", "--lr", best_lr)
    run("abl_silu", *TS, *low, "--ffn-type", "silu", "--d-ff", "2048", "--lr", best_lr)

    # 5. OpenWebText with the same architecture and number of iterations.
    run("owt_full", *OWT, "--total-tokens", str(FULL), "--lr", best_lr, "--eval-every", "500")

    # 6. Local "leaderboard" run: 45 minutes of training on one A6000, OWT only.
    run(
        "owt_leaderboard", *OWT, "--total-tokens", str(4 * FULL), "--max-seconds", "2700",
        "--num-layers", "6", "--num-heads", "8", "--tie-embeddings", "--lr", best_lr,
        "--eval-every", "500", "--log-every", "100",
    )  # fmt: skip


if __name__ == "__main__":
    main()
