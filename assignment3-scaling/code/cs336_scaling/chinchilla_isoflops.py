"""Problem chinchilla_isoflops: fit Chinchilla-style scaling laws from IsoFLOP profiles.

For every compute budget C_i the run with the lowest final loss gives N_opt(C_i); the
matching dataset size is D_opt = C_i / (6 N_opt).  Power laws N_opt = k_N C^a and
D_opt = k_D C^b are fitted by least squares in log-log space and extrapolated.

Usage:  uv run --extra study python -m cs336_scaling.chinchilla_isoflops
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "isoflops_curves.json"
FIG_DIR = ROOT / "writeup" / "figures"
OUT_PATH = ROOT / "writeup" / "generated" / "chinchilla_results.json"
TARGET_BUDGETS = (1e23, 1e24)


def load_optima(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return (C_i, N_opt(C_i), D_opt(C_i)) sorted by C, plus the raw profiles."""
    profiles: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for run in json.loads(path.read_text()):
        profiles[run["compute_budget"]].append((run["parameters"], run["final_loss"]))
    budgets = np.array(sorted(profiles))
    n_opt = np.array([min(profiles[c], key=lambda r: r[1])[0] for c in budgets], dtype=float)
    d_opt = budgets / (6 * n_opt)
    return budgets, n_opt, d_opt, profiles


def _log_linear(log_c: np.ndarray, log_k: float, exponent: float) -> np.ndarray:
    return log_k + exponent * log_c


def fit_power_law(c: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = k * C^exponent; returns (k, exponent)."""
    (log_k, exponent), _ = curve_fit(_log_linear, np.log10(c), np.log10(y), p0=(0.0, 0.5))
    return 10**log_k, exponent


def _plot(c, y, k, exponent, ylabel, symbol, path):
    grid = np.logspace(np.log10(c.min()) - 0.3, 24.5, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(grid, k * grid**exponent, color="#1f4e79", label=f"fit: ${symbol} = {k:.3g}\\,C^{{{exponent:.3f}}}$")
    ax.scatter(c, y, color="#c0392b", zorder=3, label=f"lowest-loss run per budget ({len(c)} points)")
    for target in TARGET_BUDGETS:
        value = k * target**exponent
        ax.scatter([target], [value], marker="*", s=140, color="#e69f00", zorder=4)
        ax.annotate(f"{value:.2e}", (target, value), textcoords="offset points", xytext=(-58, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute budget C (FLOPs)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_profiles(profiles, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cmap = plt.get_cmap("viridis")
    for i, c in enumerate(sorted(profiles)):
        runs = sorted(profiles[c])
        n, loss = zip(*runs)
        color = cmap(i / max(1, len(profiles) - 1))
        ax.plot(n, loss, marker="o", ms=3, color=color, label=f"{c:.0e}")
        best = min(runs, key=lambda r: r[1])
        ax.scatter([best[0]], [best[1]], marker="*", s=110, color=color, edgecolor="black", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters N")
    ax.set_ylabel("Final training loss")
    ax.legend(title="C (FLOPs)", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    budgets, n_opt, d_opt, profiles = load_optima()
    k_n, a = fit_power_law(budgets, n_opt)
    k_d, b = fit_power_law(budgets, d_opt)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _plot(budgets, n_opt, k_n, a, "Compute-optimal parameters $N_{opt}$", "N_{opt}", FIG_DIR / "chinchilla_n_opt.png")
    _plot(budgets, d_opt, k_d, b, "Compute-optimal tokens $D_{opt}$", "D_{opt}", FIG_DIR / "chinchilla_d_opt.png")
    _plot_profiles(profiles, FIG_DIR / "chinchilla_profiles.png")

    results = {
        "points": [
            {"compute_budget": float(c), "n_opt": float(n), "d_opt": float(d)}
            for c, n, d in zip(budgets, n_opt, d_opt)
        ],
        "n_fit": {"k": k_n, "exponent": a},
        "d_fit": {"k": k_d, "exponent": b},
        "predictions": {
            f"{t:.0e}": {"n_opt": k_n * t**a, "d_opt": k_d * t**b} for t in TARGET_BUDGETS
        },
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"{'C':>10} {'N_opt':>14} {'D_opt':>14}")
    for c, n, d in zip(budgets, n_opt, d_opt):
        print(f"{c:10.1e} {n:14.4e} {d:14.4e}")
    print(f"N_opt = {k_n:.4g} * C^{a:.4f}")
    print(f"D_opt = {k_d:.4g} * C^{b:.4f}")
    for t in TARGET_BUDGETS:
        print(f"C = {t:.0e}: N_opt = {k_n * t**a:.3e} parameters, D_opt = {k_d * t**b:.3e} tokens")


if __name__ == "__main__":
    main()
