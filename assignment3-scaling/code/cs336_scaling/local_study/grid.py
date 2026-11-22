"""Run plan for the local scaling study and a restart-safe sequential queue.

Budget structure (mirrors the handout's 12 h of scaling runs -> 48 h target, i.e. 1 : 4):

  stage "lr"    a brief learning-rate check at the smallest budget, on a small and a larger model;
  stage "grid"  IsoFLOP profiles: 4 compute budgets (C0, 2 C0, 4 C0, 8 C0) x 5 model sizes each.
                If the best model of a profile sits on the edge of its window the queue adds the
                next size on that side (at most MAX_EXTENSIONS per profile) so that every
                minimum is bracketed;
  target        one run with C_TARGET = 4 x SCALING_BUDGET (launched by run_target.py after
                fit.py has chosen the configuration).

SCALING_BUDGET is the hard cap for everything in the "lr" and "grid" stages; the queue
refuses to start a run that would exceed it.

Usage (one GPU, runs everything that is not finished yet; safe to re-launch at any time):
  CUDA_VISIBLE_DEVICES=1 uv run --extra study python -m cs336_scaling.local_study.grid
  ... --stage lr | grid      run a single stage
  ... --dry-run              print the plan and its cost
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from cs336_scaling.local_study import flops as F

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "experiments" / "runs"
CHOSEN_LR_PATH = ROOT / "experiments" / "chosen_lr.json"

# --- budget -----------------------------------------------------------------------------
# C0 was chosen from the smoke-test throughput so that the scaling runs take about three
# A6000-hours in total (see the write-up for the measurement).
C0 = 3.0e15
SCALING_BUDGET = 90 * C0  # hard cap for lr check + IsoFLOP grid (incl. adaptive extensions)
C_TARGET = 4 * SCALING_BUDGET
BUDGETS = [C0, 2 * C0, 4 * C0, 8 * C0]

# Index windows into F.FAMILY for each budget (5 sizes each, shifted up with the budget).
WINDOWS = {
    C0: [0, 1, 2, 3, 5],
    2 * C0: [1, 2, 3, 5, 7],
    4 * C0: [1, 3, 5, 7, 9],
    8 * C0: [3, 5, 7, 9, 11],
}
MAX_EXTENSIONS = 2

LR_CANDIDATES = [5e-4, 1e-3, 2e-3, 4e-3]
LR_CHECK_MODELS = [1, 5]  # FAMILY indices: d256L4 and d512L8
DEFAULT_LR = 1e-3
N_EVALS = 8


@dataclass(frozen=True)
class Run:
    name: str
    group: str
    d_model: int
    n_layer: int
    budget: float
    lr: float

    @property
    def tokens(self) -> float:
        return F.tokens_for_budget(self.d_model, self.n_layer, self.budget)

    def command(self) -> list[str]:
        return [
            sys.executable, "-m", "cs336_scaling.local_study.train",
            "--name", self.name, "--group", self.group,
            "--d-model", str(self.d_model), "--n-layer", str(self.n_layer),
            "--tokens", repr(self.tokens), "--budget", repr(self.budget),
            "--lr", repr(self.lr), "--n-evals", str(N_EVALS),
        ]  # fmt: skip


def _name(budget: float, index: int, lr: float) -> str:
    d_model, n_layer = F.FAMILY[index]
    return f"c{budget / C0:g}_{F.model_name(d_model, n_layer)}_lr{lr:g}"


def make_run(budget: float, index: int, lr: float, group: str) -> Run:
    d_model, n_layer = F.FAMILY[index]
    return Run(_name(budget, index, lr), group, d_model, n_layer, budget, lr)


def read_log(name: str) -> tuple[dict | None, dict | None]:
    """(config record, final record) of a run; either may be None."""
    path = RUN_DIR / f"{name}.jsonl"
    if not path.exists():
        return None, None
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    config = next((r for r in records if r["type"] == "config"), None)
    final = next((r for r in records if r["type"] == "final"), None)
    return config, final


def final_loss(name: str) -> float | None:
    _, final = read_log(name)
    if final is None:
        return None
    return final["final_val_loss"] if final["status"] == "completed" else float("inf")


def spent() -> float:
    """Compute already charged to the scaling budget (finished lr/grid runs)."""
    total = 0.0
    for path in RUN_DIR.glob("*.jsonl"):
        config, final = read_log(path.stem)
        if config and final and config.get("group") in ("lr", "grid"):
            total += config["flops"]
    return total


def lr_runs() -> list[Run]:
    return [make_run(C0, i, lr, "lr") for i in LR_CHECK_MODELS for lr in LR_CANDIDATES]


def choose_lr() -> float:
    """Pick the LR with the lowest mean final loss over the check models (once all are done)."""
    if CHOSEN_LR_PATH.exists():
        return json.loads(CHOSEN_LR_PATH.read_text())["lr"]
    table = {lr: [final_loss(_name(C0, i, lr)) for i in LR_CHECK_MODELS] for lr in LR_CANDIDATES}
    if any(v is None for losses in table.values() for v in losses):
        return DEFAULT_LR
    best = min(table, key=lambda lr: sum(table[lr]) / len(table[lr]))
    CHOSEN_LR_PATH.write_text(json.dumps({"lr": best, "table": {str(k): v for k, v in table.items()}}, indent=2))
    return best


def grid_runs(lr: float) -> list[Run]:
    return [make_run(budget, i, lr, "grid") for budget in BUDGETS for i in WINDOWS[budget]]


def extension(budget: float, lr: float) -> Run | None:
    """If the profile's minimum is on the edge of the sizes run so far, return the next size out."""
    done = {}
    for i in range(len(F.FAMILY)):
        loss = final_loss(_name(budget, i, lr))
        if loss is not None:
            done[i] = loss
    if len(done) < len(WINDOWS[budget]) or len(done) >= len(WINDOWS[budget]) + MAX_EXTENSIONS:
        return None
    best = min(done, key=done.get)
    if best == min(done) and best > 0:
        return make_run(budget, best - 1, lr, "grid")
    if best == max(done) and best < len(F.FAMILY) - 1:
        return make_run(budget, best + 1, lr, "grid")
    return None


def execute(run: Run) -> bool:
    if final_loss(run.name) is not None:
        return True
    cost = F.flops(run.d_model, run.n_layer, run.tokens)
    if spent() + cost > SCALING_BUDGET * 1.001:
        print(f"SKIP {run.name}: would exceed the scaling budget", flush=True)
        return False
    print(f"RUN  {run.name}  tokens={run.tokens:.3e}  cost={cost / C0:.2f} C0", flush=True)
    log_dir = ROOT / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{run.name}.log", "a") as log:
        result = subprocess.run(run.command(), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"FAILED {run.name} (see experiments/logs/{run.name}.log)", flush=True)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["lr", "grid", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        plan = lr_runs() + grid_runs(DEFAULT_LR)
        total = 0.0
        for run in plan:
            total += run.budget
            print(
                f"{run.group:5s} {run.name:28s} N={F.non_embedding_params(run.d_model, run.n_layer) / 1e6:7.1f}M "
                f"tokens={run.tokens / 1e6:8.1f}M  batch={F.choose_batch_size(run.tokens)}"
            )
        # The lr-check runs at the chosen LR are reused by the grid, so they are only paid for once.
        reused = len(LR_CHECK_MODELS) * C0
        print(f"planned: {(total - reused) / C0:.0f} C0 of {SCALING_BUDGET / C0:.0f} C0; target = {C_TARGET:.3e} FLOPs")
        return

    if args.stage in ("lr", "all"):
        for run in lr_runs():
            execute(run)
    if args.stage in ("grid", "all"):
        lr = choose_lr()
        print(f"grid learning rate: {lr:g}", flush=True)
        for run in grid_runs(lr):
            execute(run)
        for budget in BUDGETS:
            while (extra := extension(budget, lr)) is not None:
                if not execute(extra):
                    break
    print(f"scaling budget used: {spent() / C0:.1f} C0 of {SCALING_BUDGET / C0:.0f} C0", flush=True)


if __name__ == "__main__":
    main()
