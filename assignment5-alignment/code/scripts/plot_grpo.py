"""Aggregate experiment logs into the figures and tables used by writeup/writeup.typ.

Reads experiments/<run>/{config.json,metrics.jsonl,...} and experiments/baselines/, and writes
    writeup/figures/*.png
    writeup/generated/*.typ   (tables / examples that the report #includes)
    writeup/generated/summary.csv

Sections whose runs have not finished yet get a clearly marked "pending" snippet, so the report
always compiles.

    uv run --extra gpu --with matplotlib python scripts/plot_grpo.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cs336_alignment.sweeps import SWEEPS

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
FIG = REPO / "writeup" / "figures"
GEN = REPO / "writeup" / "generated"

METRIC_LABELS = {
    "loss": "loss",
    "grad_norm": "gradient norm (pre-clip)",
    "token_entropy": "token entropy (nats)",
    "train_reward": "train reward",
    "train_format_reward": "train format reward",
    "val_reward": "validation reward",
    "val_format_reward": "validation format reward",
    "val_mean_response_tokens": "validation response length (tokens)",
    "train_mean_response_tokens": "train response length (tokens)",
    "clip_fraction": "clip fraction",
    "frac_zero_advantage": "fraction of zero-advantage rollouts",
    "optimizer_steps": "optimizer steps per rollout batch",
}


# ----------------------------------------------------------------------------- loading
def read_jsonl(path: Path) -> list[dict]:
    # Split on "\n" only: str.splitlines would also split on Unicode line separators that
    # occur inside model generations.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def load_runs(sweep: str) -> dict[str, dict]:
    groups = {r["run_name"]: r["group"] for r in SWEEPS[sweep]}
    groups.update({r["run_name"]: r["group"] for r in SWEEPS.get("offpolicy_shuffled", [])})
    runs = {}
    for run_dir in sorted(EXP.glob("*")):
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists() or ".interrupted-" in run_dir.name or run_dir.name not in groups:
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        train, evals = [], []
        for rec in read_jsonl(metrics_path):
            (train if rec["kind"] == "train" else evals).append(rec)
        runs[run_dir.name] = dict(
            name=run_dir.name,
            dir=run_dir,
            group=groups[run_dir.name],
            cfg=cfg,
            train=train,
            evals=evals,
            finished=(run_dir / "final.json").exists(),
        )
    return runs


def method_key(run: dict) -> str:
    """Label that identifies a run up to its seed."""
    cfg = run["cfg"]
    if run["group"] == "lr":
        return f"lr={cfg['learning_rate']:g}"
    if run["group"] == "prompt":
        return cfg["prompt"]
    if run["group"] == "offpolicy_shuffled":
        return cfg["variant"].replace("offpolicy_", "") + " shuffled"
    return cfg["variant"]


def by_method(runs: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in runs:
        out[method_key(r)].append(r)
    return out


def series(run: dict, metric: str) -> tuple[list[int], list[float]]:
    records = run["evals"] if metric.startswith("val_") else run["train"]
    pts = [(r["step"], r[metric]) for r in records if metric in r and r[metric] == r[metric]]
    return [p[0] for p in pts], [p[1] for p in pts]


def smooth(ys: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(ys)):
        chunk = ys[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def final_eval(run: dict) -> dict | None:
    finals = [e for e in run["evals"] if e.get("final")]
    return finals[-1] if finals else None


# ----------------------------------------------------------------------------- plotting
def plot_curves(methods: dict[str, list[dict]], metrics: list[str], path: Path, title: str, ncols: int = 3) -> None:
    """One panel per metric; per method: thin lines for seeds, thick mean, shaded min-max."""
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    colors = plt.get_cmap("tab10")
    for ax, metric in zip(axes.flat, metrics):
        for ci, (label, runs) in enumerate(methods.items()):
            color = colors(ci % 10)
            per_step = defaultdict(list)
            for run in runs:
                xs, ys = series(run, metric)
                if not xs:
                    continue
                if not metric.startswith("val_"):
                    ys = smooth(ys, 5)
                ax.plot(xs, ys, color=color, alpha=0.35, linewidth=0.8)
                for x, y in zip(xs, ys):
                    per_step[x].append(y)
            if not per_step:
                continue
            steps = sorted(per_step)
            mean = [statistics.fmean(per_step[s]) for s in steps]
            ax.plot(steps, mean, color=color, linewidth=2, label=f"{label} (n={len(runs)})",
                    marker="o" if metric.startswith("val_") else None, markersize=3)
            if len(runs) > 1:
                ax.fill_between(steps, [min(per_step[s]) for s in steps], [max(per_step[s]) for s in steps],
                                color=color, alpha=0.12)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_xlabel("rollout step")
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(metrics):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=9, frameon=False)
    fig.suptitle(title + "  (thick: mean over seeds; thin: individual seeds; band: min-max; train curves smoothed over 5 steps)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_lr(runs: list[dict], path: Path) -> None:
    points = defaultdict(list)
    for run in runs:
        fe = final_eval(run)
        if fe:
            points[run["cfg"]["learning_rate"]].append(fe["val_reward"])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    crashed = [r for r in runs if final_eval(r) is None and r["evals"]]
    for r in crashed:
        last = r["evals"][-1]
        ax.scatter([r["cfg"]["learning_rate"]], [last["val_reward"]], marker="x", color="tab:red", s=60,
                   label=f"crashed at step ~{r['train'][-1]['step']} (last 256-example eval, step {last['step']})")
    lrs = sorted(points)
    for lr in lrs:
        ax.scatter([lr] * len(points[lr]), points[lr], color="tab:blue", alpha=0.6)
    ax.plot(lrs, [statistics.fmean(points[lr]) for lr in lrs], color="tab:blue", marker="D", label="mean over seeds")
    ax.set_xscale("log")
    ticks = sorted({r["cfg"]["learning_rate"] for r in runs})
    ax.set_xticks(ticks, [f"{t:.0e}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final validation reward (1,024 examples)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- typst helpers
def typ_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", " ") + '"'


def typ_raw(text: str, limit: int = 1400) -> str:
    if len(text) > limit:
        text = text[:limit] + " [...truncated]"
    return f"#block(fill: luma(244), inset: 6pt, radius: 3pt, width: 100%, text(size: 8pt, raw({typ_str(text)}, block: true)))"


def results_table(methods: dict[str, list[dict]]) -> tuple[str, list[dict]]:
    rows, csv_rows = [], []
    for label, runs in methods.items():
        finals = [(r, final_eval(r)) for r in runs]
        done = [(r, f) for r, f in finals if f]
        if not done:
            continue
        acc = [f["val_reward"] for _, f in done]
        fmt = [f["val_format_reward"] for _, f in done]
        length = [f["val_mean_response_tokens"] for _, f in done]
        init = [r["evals"][0]["val_reward"] for r, _ in done]
        hours = [r["evals"][-1]["elapsed_seconds"] / 3600 for r, _ in done]
        spread = f"{min(acc):.3f} - {max(acc):.3f}" if len(acc) > 1 else "n/a"
        rows.append(
            f"  [{label.replace('_', ' ')}], [{len(done)}], [{statistics.fmean(init):.3f}], [*{statistics.fmean(acc):.3f}*], "
            f"[{spread}], [{statistics.fmean(fmt):.3f}], [{statistics.fmean(length):.0f}], [{statistics.fmean(hours):.2f}],"
        )
        for r, f in done:
            csv_rows.append(dict(run=r["name"], group=r["group"], method=label, seed=r["cfg"]["seed"],
                                 initial_val_reward=r["evals"][0]["val_reward"], final_val_reward=f["val_reward"],
                                 final_val_format_reward=f["val_format_reward"],
                                 final_val_response_tokens=f["val_mean_response_tokens"],
                                 hours=r["evals"][-1]["elapsed_seconds"] / 3600))
    if not rows:
        return "", []
    table = (
        "#table(\n  columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr), align: (left, center, center, center, center, center, center, center),\n"
        "  table.header([*method*], [*seeds*], [*val. reward before*], [*final val. reward (mean)*], [*min - max*], "
        "[*final format reward*], [*resp. tokens*], [*hours / run*]),\n" + "\n".join(rows) + "\n)\n"
    )
    return table, csv_rows


def pending(what: str) -> str:
    return (
        "#block(fill: rgb(\"#fff3cd\"), inset: 8pt, radius: 3pt, width: 100%)["
        f"*Pending.* The runs for _{what}_ have not finished; this block is regenerated by `scripts/plot_grpo.py`.]\n"
    )


def write_section(name: str, what: str, methods: dict[str, list[dict]], metrics: list[str], title: str,
                  all_csv: list[dict], extra_fig: str | None = None) -> None:
    table, csv_rows = results_table(methods)
    all_csv.extend(csv_rows)
    has_curves = any(r["train"] or r["evals"] for runs in methods.values() for r in runs)
    parts = []
    if has_curves:
        plot_curves(methods, metrics, FIG / f"{name}.png", title)
        unfinished = [r["name"] for runs in methods.values() for r in runs if not r["finished"]]
        parts.append(f'#figure(image("../figures/{name}.png", width: 100%), caption: [{title}.])\n')
        if unfinished:
            parts.append(f"_Runs without a final evaluation (they stopped before step {max(r['cfg']['num_rollout_steps'] for runs in methods.values() for r in runs)}; see the text): {', '.join(unfinished).replace('_', ' ')}._\n")
    if extra_fig:
        parts.append(extra_fig)
    parts.append(table if table else pending(what))
    (GEN / f"{name}.typ").write_text("\n".join(parts))


def baselines_section() -> None:
    path = EXP / "baselines" / "summary.json"
    if not path.exists():
        for name in ("baselines", "baseline_examples", "baseline_misparse"):
            (GEN / f"{name}.typ").write_text(pending("the prompting baselines"))
        return
    summary = json.loads(path.read_text())
    rows = []
    for name, s in summary.items():
        rows.append(
            f"  [`{name}`], [{s['n']}], [{s['correct']} ({100 * s['correct'] / s['n']:.1f}%)], "
            f"[{s['formatted_wrong']} ({100 * s['formatted_wrong'] / s['n']:.1f}%)], "
            f"[{s['unformatted']} ({100 * s['unformatted'] / s['n']:.1f}%)], [{s['mean_response_tokens']:.0f}],"
        )
    (GEN / "baselines.typ").write_text(
        "#table(\n  columns: (auto, auto, 1fr, 1fr, 1fr, auto), align: (left, center, center, center, center, center),\n"
        "  table.header([*prompt*], [*n*], [*(1) format 1, answer 1*], [*(2) format 1, answer 0*], "
        "[*(3) format 0, answer 0*], [*mean resp. tokens*]),\n" + "\n".join(rows) + "\n)\n"
    )

    # A few short examples per prompt: one of each category, preferring responses whose last
    # number equals the ground truth (i.e. candidates for "right but not parsed").
    import re

    number = re.compile(r"-?\d[\d,]*\.?\d*")

    def last_number_matches(row: dict) -> bool:
        found = number.findall(row["response"].split("<answer>")[-1])
        return bool(found) and found[-1].replace(",", "").rstrip(".") == row["ground_truth"].replace(",", "")

    parts = []
    labels = {"correct": "(1) correct", "formatted_wrong": "(2) formatted, graded wrong", "unformatted": "(3) unformatted"}
    for name in summary:
        rows_ = read_jsonl(EXP / "baselines" / f"{name}.jsonl")
        parts.append(f"*Prompt `{name}`* --- the question is inserted into the template `prompts/{name}`.\n")
        for category, label in labels.items():
            candidates = [r for r in rows_ if r["category"] == category and 40 < len(r["response"]) < 420]
            candidates.sort(key=lambda r: (not last_number_matches(r), len(r["response"])))
            if not candidates:
                continue
            r = candidates[0]
            parts.append(f"{label}; ground truth {r['ground_truth']}. Question:\n" + typ_raw(r["question"], 500) + "\n")
            parts.append("Response:\n" + typ_raw(r["response"], 500) + "\n")
    (GEN / "baseline_examples.typ").write_text("\n".join(parts))

    lines = []
    for name in summary:
        rows_ = read_jsonl(EXP / "baselines" / f"{name}.jsonl")
        for category in ("formatted_wrong", "unformatted"):
            sel = [r for r in rows_ if r["category"] == category]
            hits = sum(last_number_matches(r) for r in sel)
            truncated = sum(r["finish_reason"] == "length" for r in sel)
            lines.append(f"  [`{name}`], [{labels[category]}], [{len(sel)}], [{hits}], [{truncated}],")
    (GEN / "baseline_misparse.typ").write_text(
        "#table(\n  columns: (auto, auto, auto, 1fr, 1fr), align: (left, left, center, center, center),\n"
        "  table.header([*prompt*], [*category*], [*n*], [*last number in the response equals the ground truth*], "
        "[*hit the 512-token limit*]),\n" + "\n".join(lines) + "\n)\n"
    )


def rollouts_section(runs: dict[str, dict]) -> None:
    run = runs.get("grpo_s0")
    if not run or not run["finished"]:
        (GEN / "rollouts.typ").write_text(pending("rollouts before and after training"))
        return
    last = run["cfg"]["num_rollout_steps"]
    before = read_jsonl(run["dir"] / "val_generations_step0.jsonl")
    after = read_jsonl(run["dir"] / f"val_generations_step{last}.jsonl")
    parts = []
    for i in range(3):
        parts.append(f"*Validation example {i + 1}.* {typ_raw(before[i]['question'], 600)}\n")
        parts.append(f"Before training (reward {before[i]['reward']:.0f}, ground truth {before[i]['ground_truth']}):\n"
                     + typ_raw(before[i]["response"]) + "\n")
        parts.append(f"After training (reward {after[i]['reward']:.0f}):\n" + typ_raw(after[i]["response"]) + "\n")
    (GEN / "rollouts.typ").write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="scaled")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.sweep)
    group = lambda g: [r for r in runs.values() if r["group"] == g]  # noqa: E731
    all_csv: list[dict] = []
    standard = by_method(group("standard"))

    baselines_section()
    rollouts_section(runs)

    write_section(
        "standard", "standard on-policy GRPO", standard,
        ["loss", "grad_norm", "token_entropy", "train_reward", "train_format_reward", "val_reward",
         "val_format_reward", "val_mean_response_tokens", "frac_zero_advantage"],
        "Standard on-policy GRPO, r1_zero prompt", all_csv,
    )

    lr_runs = group("lr") + group("standard")
    lr_methods = defaultdict(list)
    for r in lr_runs:
        lr_methods[f"lr={r['cfg']['learning_rate']:g}"].append(r)
    lr_methods = dict(sorted(lr_methods.items(), key=lambda kv: float(kv[0].split("=")[1])))
    extra = None
    if any(final_eval(r) for r in group("lr")):
        plot_lr(lr_runs, FIG / "lr_final.png")
        extra = '#figure(image("../figures/lr_final.png", width: 55%), caption: [Final validation reward versus learning rate.])\n'
    write_section("lr", "the learning-rate sweep", lr_methods if group("lr") else {},
                  ["val_reward", "train_reward", "token_entropy", "grad_norm", "val_format_reward", "val_mean_response_tokens"],
                  "Learning-rate sweep (standard GRPO)", [] if not group("lr") else all_csv, extra)

    prompt_methods = {}
    if group("prompt"):
        prompt_methods = {"r1_zero": [r for runs_ in standard.values() for r in runs_], **by_method(group("prompt"))}
    write_section("prompt", "the prompt ablation", prompt_methods,
                  ["val_reward", "val_format_reward", "train_reward", "token_entropy", "val_mean_response_tokens", "grad_norm"],
                  "Prompt ablation (standard GRPO)", all_csv)

    variant_methods = {**standard, **by_method(group("variants"))} if group("variants") else {}
    write_section("variants", "the on-policy variants", variant_methods,
                  ["val_reward", "train_reward", "val_format_reward", "token_entropy", "grad_norm",
                   "val_mean_response_tokens", "frac_zero_advantage", "loss"],
                  "On-policy estimator variants", all_csv)

    off_methods = {**standard, **by_method(group("offpolicy"))} if group("offpolicy") else {}
    if group("offpolicy_shuffled"):
        shuffled = {**standard, **by_method(group("offpolicy_shuffled"))}
        write_section("offpolicy_shuffled", "the shuffled-minibatch off-policy runs", shuffled,
                      ["val_reward", "val_format_reward", "train_mean_response_tokens", "token_entropy",
                       "optimizer_steps", "clip_fraction"],
                      "First off-policy implementation (minibatches shuffled across prompts): collapse", all_csv)
    write_section("offpolicy", "the off-policy variants", off_methods,
                  ["val_reward", "train_reward", "clip_fraction", "token_entropy", "grad_norm",
                   "val_mean_response_tokens", "optimizer_steps", "val_format_reward"],
                  "Off-policy variants (32 minibatch steps per rollout batch) vs. on-policy GRPO", all_csv)

    own_methods = {}
    if group("own"):
        variants = by_method(group("variants"))
        own_methods = {k: variants[k] for k in ("dr_grpo", "maxrl") if k in variants}
        own_methods.update(by_method(group("own")))
    write_section("own", "the sqrt-MaxRL estimator", own_methods,
                  ["val_reward", "train_reward", "token_entropy", "grad_norm", "val_format_reward", "val_mean_response_tokens"],
                  "Own estimator (sqrt-MaxRL) vs. its neighbours Dr. GRPO and MaxRL", all_csv)

    seen, unique = set(), []
    for row in all_csv:
        if row["run"] not in seen:
            seen.add(row["run"])
            unique.append(row)
    if unique:
        with open(GEN / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(unique[0]))
            writer.writeheader()
            writer.writerows(unique)
    print(f"loaded {len(runs)} runs ({sum(r['finished'] for r in runs.values())} finished); wrote {GEN} and {FIG}")


if __name__ == "__main__":
    main()
