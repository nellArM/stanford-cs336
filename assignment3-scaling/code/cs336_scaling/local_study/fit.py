"""Fit scaling laws to the finished grid runs and choose the target configuration.

Two fits are made from the same runs:

1. IsoFLOP minima (Hoffmann et al., approach 2).  For every budget a parabola in log N is
   fitted through the three points around the lowest-loss run to interpolate N_opt(C_i) on the
   coarse size grid (the raw argmin is reported as well); then N_opt = k * C^a is fitted in
   log-log space and extrapolated to C_TARGET.  This fit chooses the model size.
2. Parametric loss (approach 3):  L(N, D) = E + A / N^alpha + B / D^beta, fitted on all grid
   runs by least squares on log L from many initialisations.  This fit predicts the final loss
   of the chosen configuration, and its own N_opt(C_TARGET) is reported as a cross-check.

A third, independent estimate of the loss extrapolates the lower envelope
L_min(C) = E + A * C^(-gamma) through the per-budget minima.

Outputs: experiments/prediction.json, figures under writeup/figures/, and
writeup/generated/study_fit.typ (included by the write-up).

Usage:  uv run --extra study python -m cs336_scaling.local_study.fit
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from cs336_scaling.local_study import flops as F
from cs336_scaling.local_study import grid as G

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "writeup" / "figures"
GEN_DIR = ROOT / "writeup" / "generated"
PREDICTION_PATH = ROOT / "experiments" / "prediction.json"
TARGET_N_EVALS = 32


def load_runs(groups=("lr", "grid")) -> list[dict]:
    runs = []
    for path in sorted(G.RUN_DIR.glob("*.jsonl")):
        config, final = G.read_log(path.stem)
        if not config or not final or config.get("group") not in groups or final["status"] != "completed":
            continue
        runs.append({**config, **final})
    return runs


def isoflop_minima(runs: list[dict], lr: float) -> list[dict]:
    profiles: dict[float, list[dict]] = defaultdict(list)
    for run in runs:
        if run["lr"] == lr and run["budget"]:
            profiles[run["budget"]].append(run)
    minima = []
    for budget in sorted(profiles):
        profile = sorted(profiles[budget], key=lambda r: r["n_params"])
        losses = np.array([r["final_val_loss"] for r in profile])
        log_n = np.log([r["n_params"] for r in profile])
        best = int(losses.argmin())
        n_raw = float(np.exp(log_n[best]))
        n_interp, loss_interp, bracketed = n_raw, float(losses[best]), 0 < best < len(profile) - 1
        if bracketed:
            a, b, c = np.polyfit(log_n[best - 1 : best + 2], losses[best - 1 : best + 2], 2)
            if a > 0:
                x = -b / (2 * a)
                n_interp, loss_interp = float(np.exp(x)), float(a * x * x + b * x + c)
        minima.append(
            {
                "budget": budget,
                "n_opt_raw": n_raw,
                "n_opt": n_interp,
                "loss_min": loss_interp,
                "loss_min_raw": float(losses[best]),
                "best_model": F.model_name(profile[best]["d_model"], profile[best]["n_layer"]),
                "bracketed": bool(bracketed),
                "profile": [(r["n_params"], r["final_val_loss"], r["total_train_tokens"]) for r in profile],
            }
        )
    return minima


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    exponent, log_k = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_k)), float(exponent)


def fit_parametric(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> dict:
    """L = E + A N^-alpha + B D^-beta, parameters (log A, log B, log E, alpha, beta)."""

    def predict(theta, n, d):
        log_a, log_b, log_e, alpha, beta = theta
        return np.exp(log_e) + np.exp(log_a) * n**-alpha + np.exp(log_b) * d**-beta

    def residual(theta):
        return np.log(predict(theta, n, d)) - np.log(loss)

    best = None
    for log_a in (0.0, 5.0, 10.0):
        for log_b in (0.0, 5.0, 10.0):
            for log_e in (-1.0, 0.0, 1.0):
                for alpha in (0.2, 0.5):
                    for beta in (0.2, 0.5):
                        result = least_squares(
                            residual,
                            x0=[log_a, log_b, log_e, alpha, beta],
                            bounds=([-10, -10, -5, 0.01, 0.01], [40, 40, np.log(loss.min()), 2, 2]),
                            loss="huber",
                            f_scale=1e-2,
                        )
                        if best is None or result.cost < best.cost:
                            best = result
    theta = best.x
    fitted = predict(theta, n, d)
    return {
        "A": float(np.exp(theta[0])),
        "B": float(np.exp(theta[1])),
        "E": float(np.exp(theta[2])),
        "alpha": float(theta[3]),
        "beta": float(theta[4]),
        "rmse": float(np.sqrt(np.mean((fitted - loss) ** 2))),
        "max_abs_error": float(np.max(np.abs(fitted - loss))),
        "theta": [float(t) for t in theta],
    }


def parametric_loss(fit: dict, n: float, d: float) -> float:
    return fit["E"] + fit["A"] * n ** -fit["alpha"] + fit["B"] * d ** -fit["beta"]


def fit_envelope(c: np.ndarray, loss: np.ndarray) -> dict | None:
    """L_min(C) = E + A C^-gamma through the per-budget minima (needs >= 4 points)."""
    if len(c) < 4:
        return None
    x = c / c.min()

    def residual(theta):
        return theta[0] + np.exp(theta[1]) * x ** -theta[2] - loss

    best = None
    for e0 in (0.0, 0.5 * loss.min(), 0.9 * loss.min()):
        for gamma in (0.1, 0.3, 0.6):
            result = least_squares(
                residual, x0=[e0, np.log(max(loss.max() - e0, 1e-3)), gamma],
                bounds=([0.0, -10, 0.01], [loss.min(), 10, 2]),
            )  # fmt: skip
            if best is None or result.cost < best.cost:
                best = result
    e, log_a, gamma = best.x
    return {"E": float(e), "A": float(np.exp(log_a)), "gamma": float(gamma), "c_ref": float(c.min())}


def envelope_loss(fit: dict, c: float) -> float:
    return fit["E"] + fit["A"] * (c / fit["c_ref"]) ** -fit["gamma"]


def nearest_family_member(n_target: float) -> tuple[int, int]:
    return min(F.FAMILY, key=lambda m: abs(np.log(F.non_embedding_params(*m)) - np.log(n_target)))


def parametric_optimum(fit: dict, compute: float) -> tuple[int, int]:
    """Family member with the lowest parametric loss when it spends ``compute`` (N_eff accounting)."""
    return min(F.FAMILY, key=lambda m: parametric_loss(fit, F.non_embedding_params(*m), F.tokens_for_budget(*m, compute)))


def make_figures(minima, k, a, parametric, runs, lr, chosen, prediction):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, m in enumerate(minima):
        color = cmap(i / max(1, len(minima) - 1))
        n, loss, _ = zip(*m["profile"])
        ax.plot(n, loss, marker="o", ms=4, color=color, label=f"C = {m['budget']:.1e}")
        ax.scatter([m["n_opt"]], [m["loss_min"]], marker="*", s=130, color=color, edgecolor="black", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Non-embedding parameters N = 12 L d²")
    ax.set_ylabel("Final validation loss")
    ax.set_title("IsoFLOP profiles (stars: interpolated minima)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_isoflops.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    c = np.array([m["budget"] for m in minima])
    grid = np.logspace(np.log10(c.min()) - 0.2, np.log10(G.C_TARGET) + 0.2, 100)
    ax.plot(grid, k * grid**a, color="#1f4e79", label=f"$N_{{opt}} = {k:.3g}\\,C^{{{a:.3f}}}$")
    ax.scatter(c, [m["n_opt"] for m in minima], color="#c0392b", zorder=3, label="interpolated IsoFLOP minima")
    ax.scatter(c, [m["n_opt_raw"] for m in minima], facecolor="none", edgecolor="#c0392b", label="raw argmin")
    ax.scatter([G.C_TARGET], [F.non_embedding_params(*chosen)], marker="*", s=160, color="#e69f00", zorder=4,
               label=f"target run: {F.model_name(*chosen)}")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute C = 6 N_eff D (FLOPs)")
    ax.set_ylabel("Compute-optimal N")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_n_opt.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    grid_runs = [r for r in runs if r["lr"] == lr]
    actual = np.array([r["final_val_loss"] for r in grid_runs])
    fitted = np.array([parametric_loss(parametric, r["n_params"], r["total_train_tokens"]) for r in grid_runs])
    ax.scatter(actual, fitted, c=np.log10([r["budget"] for r in grid_runs]), cmap="viridis")
    lim = [min(actual.min(), fitted.min()) - 0.05, max(actual.max(), fitted.max()) + 0.05]
    ax.plot(lim, lim, color="gray", lw=1)
    ax.set_xlabel("Measured final validation loss")
    ax.set_ylabel("Parametric fit")
    ax.set_title(f"L = E + A/N^α + B/D^β  (RMSE {parametric['rmse']:.3f})", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_parametric_fit.png", dpi=200)
    plt.close(fig)


def _tm(x: float) -> str:
    """Format a number for Typst math mode (no bare 'e+05' exponents)."""
    text = f"{x:.4g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"{mantissa} times 10^({int(exponent)})"
    return text


def write_typst(prediction: dict, minima: list[dict], runs: list[dict], lr_table: dict | None) -> None:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    p = prediction
    lines = ["// Generated by cs336_scaling/local_study/fit.py -- do not edit by hand.", ""]

    if lr_table:
        lines += ["=== Learning-rate check", ""]
        names = [F.model_name(*F.FAMILY[i]) for i in G.LR_CHECK_MODELS]
        lines.append(f"#table(columns: {len(names) + 1}, align: (left,{' right,' * len(names)}),")
        lines.append("  table.header([*Peak LR*], " + ", ".join(f"[*{n}*]" for n in names) + "),")
        for lr, losses in lr_table.items():
            cells = ", ".join("[diverged]" if not np.isfinite(v) else f"[{v:.4f}]" for v in losses)
            lines.append(f"  [{float(lr):g}], {cells},")
        lines += [")", "", f"Final validation loss at budget $C_0$. The grid uses a peak learning rate of *{p['lr']:g}*.", ""]

    lines += ["=== IsoFLOP runs", ""]
    lines.append("#table(columns: 7, align: (right, left, right, right, right, right, right),")
    lines.append("  table.header([*C (FLOPs)*], [*model*], [*N*], [*tokens D*], [*batch*], [*val loss*], [*A6000 s*]),")
    for r in sorted((r for r in runs if r["lr"] == p["lr"]), key=lambda r: (r["budget"], r["n_params"])):
        lines.append(
            f"  [{r['budget']:.1e}], [{F.model_name(r['d_model'], r['n_layer'])}], [{r['n_params'] / 1e6:.1f}M], "
            f"[{r['total_train_tokens'] / 1e6:.1f}M], [{r['batch_size']}], [{r['final_val_loss']:.4f}], [{r['train_seconds']:.0f}],"
        )
    lines += [")", ""]

    lines += ["=== IsoFLOP minima and the fitted law", ""]
    lines.append("#table(columns: 5, align: (right, left, right, right, right),")
    lines.append("  table.header([*C (FLOPs)*], [*best model*], [*N#sub[opt] raw*], [*N#sub[opt] interpolated*], [*L#sub[min]*]),")
    for m in minima:
        flag = "" if m["bracketed"] else " (edge)"
        lines.append(
            f"  [{m['budget']:.1e}], [{m['best_model']}{flag}], [{m['n_opt_raw'] / 1e6:.1f}M], "
            f"[{m['n_opt'] / 1e6:.1f}M], [{m['loss_min']:.4f}],"
        )
    lines += [")", ""]
    par = p["parametric_fit"]
    lines += [
        f"Power law through the interpolated minima: $N_\"opt\" = {_tm(p['n_opt_fit']['k'])} dot C^({p['n_opt_fit']['exponent']:.3f})$ "
        f"(raw-argmin variant: exponent {p['n_opt_fit_raw']['exponent']:.3f}).",
        "",
        f"Parametric fit over {p['n_runs_parametric']} runs: $L(N, D) = {par['E']:.3f} + {_tm(par['A'])} dot N^(-{par['alpha']:.3f}) + "
        f"{_tm(par['B'])} dot D^(-{par['beta']:.3f})$, RMSE {par['rmse']:.4f}, largest absolute error {par['max_abs_error']:.4f} nats.",
        "",
        f"Scaling budget actually used: {p['scaling_flops_used']:.3e} FLOPs "
        f"({p['scaling_flops_used'] / G.SCALING_BUDGET * 100:.0f}% of the cap), {p['scaling_seconds_used'] / 3600:.2f} A6000-hours of training time.",
        "",
        "=== Prediction for the target budget",
        "",
        f"- Target budget: $C = {_tm(p['c_target'])}$ FLOPs (4 x the scaling budget).",
        f"- IsoFLOP power law: $N_\"opt\" = {p['n_opt_pred'] / 1e6:.1f}$M; the nearest family member is *{p['model']}* "
        f"($N = {p['n_params'] / 1e6:.1f}$M, d_model {p['d_model']}, {p['n_layer']} layers, {p['d_model'] // 64} heads).",
        f"- Parametric-fit optimum within the family (cross-check): {p['parametric_optimum_model']}.",
        f"- Training tokens: $D = {p['tokens'] / 1e6:.0f}$M, batch size {p['batch_size']}, {p['total_steps']} steps, peak LR {p['lr']:g}.",
        f"- *Predicted final validation loss: {p['predicted_final_loss']:.3f}* (parametric fit)"
        + (f"; the envelope extrapolation $L_\"min\"(C)$ gives {p['envelope_predicted_loss']:.3f}." if p.get("envelope_predicted_loss") else "."),
        "",
        '#figure(image("../figures/study_isoflops.png", width: 75%), caption: [IsoFLOP profiles of the local study.])',
        '#figure(image("../figures/study_n_opt.png", width: 75%), caption: [Compute-optimal model size and its extrapolation to the target budget.])',
        '#figure(image("../figures/study_parametric_fit.png", width: 55%), caption: [Parametric fit against measured losses.])',
    ]
    (GEN_DIR / "study_fit.typ").write_text("\n".join(lines) + "\n")


def main() -> None:
    runs = load_runs()
    lr = G.choose_lr()
    lr_table = json.loads(G.CHOSEN_LR_PATH.read_text())["table"] if G.CHOSEN_LR_PATH.exists() else None
    minima = isoflop_minima(runs, lr)
    if len(minima) < 2:
        raise SystemExit("need finished IsoFLOP profiles for at least two budgets")

    c = np.array([m["budget"] for m in minima])
    k, a = fit_power_law(c, np.array([m["n_opt"] for m in minima]))
    k_raw, a_raw = fit_power_law(c, np.array([m["n_opt_raw"] for m in minima]))
    n_opt_pred = k * G.C_TARGET**a

    fit_set = [r for r in runs if r["lr"] == lr]
    parametric = fit_parametric(
        np.array([r["n_params"] for r in fit_set], dtype=float),
        np.array([r["total_train_tokens"] for r in fit_set], dtype=float),
        np.array([r["final_val_loss"] for r in fit_set]),
    )
    envelope = fit_envelope(c, np.array([m["loss_min"] for m in minima]))

    chosen = nearest_family_member(n_opt_pred)
    tokens_raw = F.tokens_for_budget(*chosen, G.C_TARGET)
    batch_size = F.choose_batch_size(tokens_raw)
    tokens = F.round_tokens(tokens_raw, batch_size, TARGET_N_EVALS)

    prediction = {
        "c_target": G.C_TARGET,
        "lr": lr,
        "n_opt_fit": {"k": k, "exponent": a},
        "n_opt_fit_raw": {"k": k_raw, "exponent": a_raw},
        "n_opt_pred": n_opt_pred,
        "model": F.model_name(*chosen),
        "d_model": chosen[0],
        "n_layer": chosen[1],
        "n_params": F.non_embedding_params(*chosen),
        "tokens": tokens,
        "batch_size": batch_size,
        "total_steps": tokens // (batch_size * F.SEQ_LEN),
        "n_evals": TARGET_N_EVALS,
        "parametric_fit": parametric,
        "n_runs_parametric": len(fit_set),
        "parametric_optimum_model": F.model_name(*parametric_optimum(parametric, G.C_TARGET)),
        "predicted_final_loss": parametric_loss(parametric, F.non_embedding_params(*chosen), tokens),
        "envelope_fit": envelope,
        "envelope_predicted_loss": envelope_loss(envelope, G.C_TARGET) if envelope else None,
        "scaling_flops_used": sum(r["flops"] for r in runs),
        "scaling_seconds_used": sum(r["train_seconds"] for r in runs),
        "minima": [{k_: v for k_, v in m.items() if k_ != "profile"} for m in minima],
    }
    PREDICTION_PATH.write_text(json.dumps(prediction, indent=2))
    manifest = ROOT / "data" / "dclm" / "manifest.json"
    if manifest.exists():
        prepared = sum(e["tokens"] for e in json.loads(manifest.read_text())["train"])
        if tokens + 1 > prepared:
            print(f"WARNING: the target run needs {tokens:,} tokens but only {prepared:,} are prepared; "
                  f"run data_prep with --min-train-tokens {tokens * 1.02:.3g} first")
    make_figures(minima, k, a, parametric, runs, lr, chosen, prediction)
    write_typst(prediction, minima, runs, lr_table)

    print(json.dumps({k_: v for k_, v in prediction.items() if k_ not in ("minima", "parametric_fit")}, indent=2))
    throughput = np.median([r["flops_per_second"] for r in runs])
    print(f"median throughput {throughput:.3e} FLOP/s -> target run ~{G.C_TARGET / throughput / 3600:.1f} h on one GPU")


if __name__ == "__main__":
    main()
