"""Turn the run logs into the figures and the summary table used by the report.

    uv run --with matplotlib python experiments/make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FIGS = ROOT.parent / "writeup" / "figures"


def load(name: str) -> tuple[list[dict], dict, dict]:
    run = RUNS / name
    records = []
    metrics = run / "metrics.jsonl"
    if metrics.exists():
        records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    final = json.loads((run / "final.json").read_text()) if (run / "final.json").exists() else {}
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    return [r for r in records if "step" in r], final, config


def curve(ax, name: str, label: str, key: str = "val_loss", x: str = "tokens", **style) -> None:
    records, final, _ = load(name)
    points = [(r[x], r[key]) for r in records if key in r]
    if not points:
        return
    xs, ys = zip(*points)
    scale = 1e6 if x == "tokens" else (60 if x == "train_seconds" else 1)
    suffix = " (diverged)" if final.get("status") == "diverged" else ""
    ax.plot([v / scale for v in xs], ys, label=label + suffix, **style)


def finish(ax, title: str, xlabel: str, ylabel: str, ylim=None, logy: bool = False) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def finite(x):
    """NaN/inf are not valid JSON; store them as null."""
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in RUNS.iterdir() if p.is_dir())

    summary = {}
    for name in names:
        records, final, config = load(name)
        vals = [r["val_loss"] for r in records if "val_loss" in r]
        summary[name] = {
            "status": final.get("status", "running"),
            "final_val_loss": final.get("final_val_loss"),
            "best_val_loss": min(vals) if vals else None,
            "last_train_loss": finite(records[-1]["train_loss"]) if records else None,
            "steps": final.get("final_step"),
            "train_minutes": round(final.get("train_seconds", 0) / 60, 1),
            "lr": config.get("lr"),
            "batch_size": config.get("batch_size"),
            "non_embedding_params": config.get("non_embedding_params"),
        }
    (FIGS / "summary.json").write_text(json.dumps(summary, indent=1))

    lr_runs = sorted((n for n in names if n.startswith("lr_")), key=lambda n: float(n[3:]))
    # LR sweep: validation curves and training curves (training curve shows divergence).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n in lr_runs:
        curve(axes[0], n, f"lr={n[3:]}")
        curve(axes[1], n, f"lr={n[3:]}", key="train_loss")
    finish(axes[0], "LR sweep: validation loss (40M tokens)", "tokens processed (M)", "val loss", ylim=(1.5, 4))
    finish(axes[1], "LR sweep: training loss", "tokens processed (M)", "train loss", logy=True)
    save(fig, "lr_sweep")

    # Final loss vs learning rate.
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ok = [(float(n[3:]), summary[n]["final_val_loss"]) for n in lr_runs if summary[n]["final_val_loss"]]
    bad = [float(n[3:]) for n in lr_runs if not summary[n]["final_val_loss"]]
    if ok:
        ax.plot(*zip(*ok), "o-", label="final val loss")
    for i, lr in enumerate(bad):
        ax.axvline(lr, color="red", ls="--", label="diverged" if i == 0 else None)
    ax.set_xscale("log")
    finish(ax, "Final validation loss vs. learning rate", "learning rate", "val loss")
    save(fig, "lr_vs_loss")

    # Full-budget runs: against steps and against wall-clock time.
    for name, title in [("ts_full", "TinyStories, 327.68M tokens"), ("owt_full", "OpenWebText, 327.68M tokens")]:
        if name not in names:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, x, xlabel in [(axes[0], "step", "gradient steps"), (axes[1], "train_seconds", "training wall-clock (min)")]:
            curve(ax, name, "train", key="train_loss", x=x, alpha=0.5)
            curve(ax, name, "validation", x=x)
            finish(ax, title, xlabel, "loss")
        save(fig, name)

    # Batch-size sweep.
    bs_runs = sorted((n for n in names if n.startswith("bs_") and (RUNS / n / "metrics.jsonl").exists()), key=lambda n: int(n[3:]))
    best_lr_run = min((n for n in lr_runs if summary[n]["final_val_loss"]), key=lambda n: summary[n]["final_val_loss"], default=None)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, x, xlabel in [(axes[0], "tokens", "tokens processed (M)"), (axes[1], "train_seconds", "training wall-clock (min)")]:
        for n in bs_runs:
            curve(ax, n, f"batch {n[3:]}", x=x)
        if best_lr_run:
            curve(ax, best_lr_run, "batch 128", x=x)
        for n in sorted((n for n in names if n.startswith("bsretune_")), key=lambda n: int(n[9:])):
            curve(ax, n, f"batch {n[9:]}, lr 3e-3", x=x, ls="--")
        finish(ax, "Batch-size sweep: validation loss", xlabel, "val loss", ylim=(1.5, 4))
    save(fig, "batch_size")

    # Ablations, each against the pre-norm / RoPE / SwiGLU baseline at the same budget.
    groups = {
        "abl_norm": [("abl_nonorm_best", "no norm, best LR"), ("abl_nonorm_third", "no norm, LR/3"),
                     ("abl_nonorm_tenth", "no norm, LR/10"), ("abl_nonorm_thirtieth", "no norm, LR/30")],
        "abl_postnorm": [("abl_postnorm", "post-norm")],
        "abl_nope": [("abl_nope", "no position embedding")],
        "abl_silu": [("abl_silu", "SiLU FFN (d_ff = 4 d_model)")],
    }  # fmt: skip
    for fig_name, runs in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, key, ylabel in [(axes[0], "val_loss", "val loss"), (axes[1], "train_loss", "train loss")]:
            if best_lr_run:
                curve(ax, best_lr_run, "baseline (pre-norm, RoPE, SwiGLU)", key=key, color="black")
            for n, label in runs:
                if n in names:
                    curve(ax, n, label, key=key)
            finish(ax, fig_name, "tokens processed (M)", ylabel, ylim=(1.5, 5) if key == "val_loss" else None, logy=key == "train_loss")
        save(fig, fig_name)

    if "owt_leaderboard" in names:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        curve(ax, "owt_leaderboard", "attempt 1: 6 layers, tied embeddings (val)", x="train_seconds")
        if "owt_leaderboard_v2" in names:
            curve(ax, "owt_leaderboard_v2", "attempt 2: train", key="train_loss", x="train_seconds", alpha=0.4)
            curve(ax, "owt_leaderboard_v2", "attempt 2: base model, 45-min schedule (val)", x="train_seconds")
        if "owt_full" in names:
            curve(ax, "owt_full", "owt_full (10k-step schedule, val)", x="train_seconds", ls=":")
        ax.axvline(45, color="gray", ls="--", lw=0.8)
        ax.set_xlim(0, 50)
        finish(ax, "45-minute OWT runs (one shared A6000)", "training wall-clock (min)", "loss", ylim=(3.8, 6))
        save(fig, "owt_leaderboard")

    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
