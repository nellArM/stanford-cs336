"""Regenerate the train_model figure and summary table from the training log.

    uv run --with matplotlib python writeup/plot_training.py [path/to/metrics.jsonl]

Writes writeup/figures/learning_curve.png and writeup/figures/training_summary.summary
(both are read by writeup.typ).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE.parent / "local-shared-data" / "output" / "filtered_cc" / "metrics.jsonl"


def main() -> None:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    train = [(r["step"], r["train_loss"]) for r in rows if "train_loss" in r]
    evals = [(r["step"], r["eval_loss"], r["elapsed_seconds"]) for r in rows if "eval_loss" in r]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    ax.plot(*zip(*train), color="#9aa5b1", linewidth=0.8, label="train loss (filtered Common Crawl)")
    if evals:
        ax.plot([e[0] for e in evals], [e[1] for e in evals], color="#1f5fbf", marker="o", markersize=3.5,
                linewidth=1.5, label="validation loss (Paloma C4-100-domains)")
    ax.set_xlabel("optimizer step (1,024 sequences x 512 tokens each)")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_ylim(top=min(8.0, max(l for _, l in train)))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "learning_curve.png")

    best = min(evals, key=lambda e: e[1]) if evals else None
    summary = {
        "available": bool(evals),
        "steps": max((s for s, _ in train), default=0) + 1,
        "best_eval_loss": round(best[1], 4) if best else None,
        "best_eval_step": best[0] if best else None,
        "best_eval_perplexity": round(math.exp(best[1]), 2) if best else None,
        "final_eval_loss": round(evals[-1][1], 4) if evals else None,
        "wall_hours": round(rows[-1]["elapsed_seconds"] / 3600, 2) if rows else None,
        "evals": [[s, round(l, 4)] for s, l, _ in evals],
    }
    (HERE / "figures" / "training_summary.summary").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
