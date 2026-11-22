"""Generate the tables (Typst) and figures (PNG) of the report from experiments/results/*.csv.

    uv run python writeup/make_tables.py

Everything in writeup/generated/ is produced by this script; nothing in the report's tables is
typed by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
GEN = ROOT / "writeup" / "generated"
GEN.mkdir(parents=True, exist_ok=True)


def read(name: str) -> pd.DataFrame | None:
    path = RES / name
    if not path.exists() or path.stat().st_size == 0:
        print(f"[skip] {name} missing")
        return None
    return pd.read_csv(path)


def fmt(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return "--"
    if isinstance(value, (int,)) or (isinstance(value, float) and float(value).is_integer() and abs(value) >= 1000):
        return f"{int(value):,}"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:,.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.{max(digits, 2)}f}"
    return str(value)


def esc(text: str) -> str:
    for ch in ("\\", "#", "$", "*", "_", "<", ">", "@", "[", "]"):
        text = text.replace(ch, "\\" + ch)
    return text


def write_table(name: str, headers: list[str], rows: list[list], align: str | None = None) -> None:
    n = len(headers)
    lines = [f"#table(columns: {n}, align: ({align or ', '.join(['left'] + ['right'] * (n - 1))}), stroke: none, inset: (x: 5pt, y: 3pt),"]
    lines.append("  table.hline(),")
    lines.append("  table.header(" + ", ".join(f"[*{esc(h)}*]" for h in headers) + "),")
    lines.append("  table.hline(stroke: 0.5pt),")
    for row in rows:
        lines.append("  " + ", ".join(f"[{esc(fmt(c))}]" for c in row) + ",")
    lines.append("  table.hline(),")
    lines.append(")")
    (GEN / f"{name}.typ").write_text("\n".join(lines) + "\n")
    print(f"[table] {name}: {len(rows)} rows")


def pm(mean, std) -> str:
    if pd.isna(mean):
        return "--"
    return f"{fmt(float(mean))} ± {fmt(float(std))}"


# ---------------------------------------------------------------------------------------
def e2e_tables() -> None:
    fp32, bf16 = read("e2e_fp32.csv"), read("e2e_bf16.csv")
    if fp32 is None:
        return
    for name, df in (("e2e_fp32", fp32), ("e2e_bf16", bf16)):
        if df is None:
            continue
        rows = []
        for size in ["small", "medium", "large", "xl", "10B"]:
            f = df[(df["size"] == size) & (df["mode"] == "forward")]
            t = df[(df["size"] == size) & (df["mode"] == "train")]
            f = f.iloc[-1] if len(f) else None
            t = t.iloc[-1] if len(t) else None
            params = next((r["params_m"] for r in (f, t) if r is not None and not pd.isna(r["params_m"])), float("nan"))
            fwd_only = "OOM" if f is not None and f["status"] != "ok" else (pm(f["forward_ms"], f["forward_std_ms"]) if f is not None else "--")
            if t is None:
                tr = ["--"] * 4
            elif t["status"] != "ok":
                tr = ["OOM"] * 4
            else:
                tr = [pm(t["forward_ms"], t["forward_std_ms"]), pm(t["backward_ms"], t["backward_std_ms"]),
                      pm(t["optimizer_ms"], t["optimizer_std_ms"]), fmt(float(t["peak_gib"]))]  # fmt: skip
            rows.append([size, fmt(float(params)) if not pd.isna(params) else "--", fwd_only, *tr])
        write_table(name, ["size", "params (M)", "forward only (ms)", "forward (ms)", "backward (ms)", "optimizer (ms)", "peak (GiB)"], rows)

    warm = read("e2e_warmup.csv")
    if warm is not None:
        rows = []
        for (size, warmup), g in warm.groupby(["size", "warmup"], sort=False):
            r = g.iloc[-1]
            rows.append([size, int(warmup), pm(r["forward_ms"], r["forward_std_ms"]), pm(r["backward_ms"], r["backward_std_ms"]), pm(r["optimizer_ms"], r["optimizer_std_ms"])])
        rows.sort(key=lambda r: (["small", "medium"].index(r[0]), r[1]))
        write_table("e2e_warmup", ["size", "warm-up steps", "forward (ms)", "backward (ms)", "optimizer (ms)"], rows)

    comp = read("e2e_compile.csv")
    if comp is not None:
        rows = []
        for size in ["small", "medium", "large"]:
            row = [size]
            for mode, col in (("forward", "forward_ms"), ("train", "total_ms")):
                for tag in ("eager", "compiled"):
                    g = comp[(comp["size"] == size) & (comp["mode"] == mode) & (comp["tag"] == tag)]
                    if not len(g):
                        row.append("--")
                    elif g.iloc[-1]["status"] != "ok":
                        row.append("OOM")
                    else:
                        row.append(fmt(float(g.iloc[-1][col])))
            rows.append(row)
        write_table("e2e_compile", ["size", "forward, eager (ms)", "forward, compiled (ms)", "train step, eager (ms)", "train step, compiled (ms)"], rows)


def attention_tables() -> None:
    df = read("attention.csv")
    if df is None:
        return
    rows = []
    for (n, d), g in df.groupby(["seq_len", "d_model"], sort=True):
        row = [int(n), int(d)]
        for variant in ("eager", "compiled"):
            r = g[g["variant"] == variant]
            if not len(r):
                row += ["--", "--"]
            elif r.iloc[-1]["status"] != "ok":
                row += ["OOM", "OOM"]
            else:
                row += [fmt(float(r.iloc[-1]["forward_ms"]), 2), fmt(float(r.iloc[-1]["backward_ms"]), 2)]
        r = g[g["variant"] == "eager"]
        ok = len(r) and r.iloc[-1]["status"] == "ok"
        row.append(fmt(float(r.iloc[-1]["mem_before_backward_gib"]), 2) if ok else "OOM")
        row.append(int(r.iloc[-1]["reps"]) if ok else "--")
        rows.append(row)
    write_table("attention", ["seq len", "d", "fwd eager (ms)", "bwd eager (ms)", "fwd compiled (ms)", "bwd compiled (ms)", "mem before bwd (GiB)", "reps"], rows,
                align="right, right, right, right, right, right, right, right")  # fmt: skip


def flash_tables() -> None:
    df = read("flash.csv")
    if df is not None:
        for dtype in ("bf16", "fp32"):
            sub = df[df["dtype"] == dtype]
            rows = []
            for (n, d), g in sub.groupby(["seq_len", "d_model"], sort=True):
                row = [int(n), int(d)]
                for impl in ("pytorch", "flash_triton", "flash_triton_full"):
                    r = g[g["impl"] == impl]
                    if not len(r):
                        row += ["--"] * 3
                    elif r.iloc[-1]["status"] != "ok":
                        row += ["OOM"] * 3
                    else:
                        row += [fmt(float(r.iloc[-1][c]), 2) for c in ("forward_ms", "backward_ms", "end_to_end_ms")]
                rows.append(row)
            write_table(f"flash_{dtype}", ["seq", "d", "PT fwd", "PT bwd", "PT e2e", "FA2 fwd", "FA2 bwd", "FA2 e2e", "FA2-T fwd", "FA2-T bwd", "FA2-T e2e"], rows,
                        align=", ".join(["right"] * 11))  # fmt: skip
        # figure: end-to-end latency vs sequence length at d=64
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), sharey=True)
        for ax, dtype in zip(axes, ("bf16", "fp32")):
            sub = df[(df["dtype"] == dtype) & (df["d_model"] == 64) & (df["status"] == "ok")]
            for impl, label, style in (("pytorch", "PyTorch attention", "o-"), ("flash_triton", "FA2: Triton fwd + compiled PyTorch bwd", "s-"),
                                       ("flash_triton_full", "FA2: Triton fwd + Triton bwd", "^-")):  # fmt: skip
                g = sub[sub["impl"] == impl].sort_values("seq_len")
                ax.plot(g["seq_len"], g["end_to_end_ms"], style, label=label, markersize=4)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{dtype}, d = 64, batch 1, causal", fontsize=9)
            ax.set_xlabel("sequence length")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("forward + backward latency (ms)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(GEN / "flash_e2e.png", dpi=170)
        plt.close(fig)

    tiles = read("flash_tiles.csv")
    if tiles is not None:
        ok = tiles[tiles["status"] == "ok"]
        rows = []
        for (dtype, n, d), g in ok.groupby(["dtype", "seq_len", "d_model"], sort=True):
            best = g.sort_values("end_to_end_ms").iloc[0]
            worst = g.sort_values("end_to_end_ms").iloc[-1]
            default = g[(g["q_tile"] == (64 if d <= 64 else 32)) & (g["k_tile"] == (64 if d <= 64 else 32))]
            rows.append([dtype, int(n), int(d), f"{int(best['q_tile'])} x {int(best['k_tile'])}", fmt(float(best["end_to_end_ms"]), 2),
                         fmt(float(default.iloc[0]["end_to_end_ms"]), 2) if len(default) else "--",
                         f"{int(worst['q_tile'])} x {int(worst['k_tile'])}", fmt(float(worst["end_to_end_ms"]), 2)])  # fmt: skip
        write_table("flash_tiles", ["dtype", "seq", "d", "best tiles (q x k)", "best e2e (ms)", "default tiles e2e (ms)", "worst tiles", "worst e2e (ms)"], rows,
                    align="left, right, right, right, right, right, right, right")  # fmt: skip


def memory_tables() -> None:
    df = read("memory.csv")
    if df is not None:
        rows = []
        full = df[df["tag"] == "xl_full"]
        for ctx in (128, 2048):
            row = [ctx]
            for mode in ("forward", "train"):
                for bf16 in (0, 1):
                    r = full[(full["context_length"] == ctx) & (full["mode"] == mode) & (full["bf16"] == bf16)]
                    if not len(r):
                        row.append("--")
                    elif r.iloc[-1]["status"] != "ok":
                        row.append("OOM")
                    else:
                        row.append(fmt(float(r.iloc[-1]["peak_gib"]), 2))
            rows.append(row)
        write_table("memory_full", ["context", "forward fp32 (GiB)", "forward bf16 (GiB)", "train step fp32 (GiB)", "train step bf16 (GiB)"], rows)

        red = df[(df["tag"] == "xl_reduced") & (df["status"] == "ok")]
        rows = []
        extrap = {}
        for (ctx, bf16), g in red.groupby(["context_length", "bf16"], sort=True):
            g = g.sort_values("num_layers")
            layers, peaks = g["num_layers"].tolist(), g["peak_gib"].tolist()
            if len(layers) >= 2:
                slope = (peaks[-1] - peaks[0]) / (layers[-1] - layers[0])
                extrap[(ctx, bf16)] = peaks[-1] + slope * (32 - layers[-1])
            else:
                slope = float("nan")
            rows.append([int(ctx), "bf16" if bf16 else "fp32", ", ".join(f"L={int(l)}: {p:.2f}" for l, p in zip(layers, peaks)),
                         fmt(slope, 2), fmt(extrap.get((ctx, bf16), float("nan")))])  # fmt: skip
        write_table("memory_reduced", ["context", "precision", "measured train-step peak (GiB)", "GiB per layer", "extrapolated to 32 layers (GiB)"], rows,
                    align="right, left, left, right, right")  # fmt: skip

    ck = read("checkpointing.csv")
    if ck is not None:
        ck = ck.drop_duplicates(subset=["num_layers", "segment", "bf16"], keep="last")  # a later successful retry replaces an OOM row
        ck = ck.sort_values(["num_layers", "bf16", "segment"], key=lambda c: c.replace(0, 99) if c.name == "segment" else c)
        rows = []
        for _, r in ck.iterrows():
            seg = "none" if r["segment"] == 0 else int(r["segment"])
            prec = "bf16" if r["bf16"] else "fp32"
            if r["status"] != "ok":
                rows.append([int(r["num_layers"]), prec, seg, "--", "OOM", "OOM", "--"])
            else:
                rows.append([int(r["num_layers"]), prec, seg, fmt(float(r["params_gib"]), 2), fmt(float(r["peak_gib"]), 2), fmt(float(r["peak_minus_params_gib"]), 2), fmt(float(r["seconds"]))])
        write_table("checkpointing", ["layers", "precision", "blocks per checkpoint", "parameters (GiB)", "peak (GiB)", "peak - parameters (GiB)", "fwd+bwd time (s)"], rows,
                    align="right, left, right, right, right, right, right")  # fmt: skip


def parse_nsys_summary(path: Path) -> dict:
    """Parse the per-stage kernel listing printed by cs336_systems.bench.nsys_analysis."""
    import re

    stages, current = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"\[(.+?)\] python wall .* GPU kernel time ([\d.]+) ms/step", line)
        if m:
            current = stages.setdefault(m.group(1), {"gpu_ms": float(m.group(2)), "kernels": []})
            continue
        m = re.match(r"\s+([\d.]+)%\s+([\d.]+) ms\s+x\s*([\d.]+)/step\s+(.+)", line)
        if m and current is not None:
            current["kernels"].append((float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4).strip()))
    for info in stages.values():
        # "Kernel2" is cutlass::Kernel2<cutlass_80_simt_sgemm_...>, i.e. also a GEMM kernel.
        info["matmul_ms"] = sum(ms for _, ms, _, name in info["kernels"] if "gemm" in name.lower() or name == "Kernel2")
    return stages


def nsys_tables() -> None:
    py = read("nsys_python_timings.csv")
    rows, top_rows = [], []
    for size in ("small", "large"):
        for ctx in (256, 512, 1024):
            for mode in ("forward", "train"):
                label = f"{size}_ctx{ctx}_{mode}"
                path = RES / f"nsys_{label}.summary"
                if not path.exists():
                    continue
                st = parse_nsys_summary(path)
                if "forward" not in st:
                    rows.append([size, ctx, mode] + ["OOM"] * 7)
                    continue
                fwd_py = float("nan")
                if py is not None:
                    p = py[py["tag"] == label]
                    if len(p) and p.iloc[-1]["status"] == "ok":
                        fwd_py = float(p.iloc[-1]["forward_ms"])
                f = st["forward"]
                row = [size, ctx, mode, fmt(fwd_py), fmt(f["gpu_ms"]), f"{100 * f['matmul_ms'] / f['gpu_ms']:.0f}%"]
                if mode == "train":
                    total = sum(st[k]["gpu_ms"] for k in ("forward", "backward", "optimizer"))
                    mm = sum(st[k]["matmul_ms"] for k in ("forward", "backward", "optimizer"))
                    row += [fmt(st["backward"]["gpu_ms"]), fmt(st["optimizer"]["gpu_ms"]), f"{100 * mm / total:.0f}%"]
                else:
                    row += ["--", "--", "--"]
                attn_mm = st["computing attention scores"]["matmul_ms"] + st["final matmul"]["matmul_ms"]
                row.append(f"{fmt(st['computing softmax']['gpu_ms'], 1)} / {fmt(attn_mm, 1)}")
                rows.append(row)
                top = f["kernels"][0]
                top_rows.append([size, ctx, mode, top[3], f"{top[0]:.0f}%", fmt(top[2], 0),
                                 st["backward"]["kernels"][0][3] if mode == "train" else "--"])  # fmt: skip
    if rows:
        write_table("nsys", ["size", "ctx", "run", "fwd, timeit (ms)", "fwd GPU kernels (ms)", "fwd matmul share", "bwd kernels (ms)", "opt kernels (ms)",
                             "step matmul share", "attn: softmax / matmul (ms)"], rows,
                    align="left, right, left, right, right, right, right, right, right, right")  # fmt: skip
        write_table("nsys_top", ["size", "ctx", "run", "top forward kernel", "share of fwd", "calls / fwd", "top backward kernel"], top_rows,
                    align="left, right, left, left, right, right, left")  # fmt: skip


def distributed_tables() -> None:
    ar = read("allreduce.csv")
    if ar is not None:
        rows = [[r["backend"], int(r["world_size"]), int(r["size_mb"]), fmt(float(r["mean_ms"]), 2), fmt(float(r["throughput_gb_s"]), 2)] for _, r in ar.iterrows()]
        write_table("allreduce", ["backend", "processes", "payload (MB)", "all-reduce time (ms)", "payload / time (GB/s)"], rows, align="left, right, right, right, right")
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for (backend, ws), g in ar.groupby(["backend", "world_size"], sort=False):
            label = f"{backend.upper()} ({'GPU' if backend == 'nccl' else 'CPU'}), {ws} procs"
            ax.plot(g["size_mb"], g["mean_ms"], "o-" if backend == "nccl" else "s--", label=label, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("payload (MB, float32)")
        ax.set_ylabel("all-reduce time (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(GEN / "allreduce.png", dpi=170)
        plt.close(fig)

    ddp = read("ddp.csv")
    if ddp is not None:
        rows = []
        for _, r in ddp.iterrows():
            rows.append([r["size"], "bf16" if r["bf16"] else "fp32", r["mode"], pm(r["step_ms"], r["step_std_ms"]), fmt(float(r["blocked_on_comm_ms"])),
                         f"{100 * float(r['comm_fraction']):.1f}%", fmt(float(r["peak_gib"]), 2)])  # fmt: skip
        write_table("ddp", ["model", "precision", "gradient sync", "time per step (ms)", "blocked on communication (ms)", "share of step", "peak / GPU (GiB)"], rows,
                    align="left, left, left, right, right, right, right")  # fmt: skip

    sh = read("sharded.csv")
    if sh is not None:
        rows = []
        for _, r in sh.iterrows():
            rows.append([r["size"], "yes" if r["sharded"] else "no", int(r["rank"]), fmt(float(r["params_gib"]), 2), fmt(float(r["optimizer_state_gib"]), 2),
                         fmt(float(r["peak_after_init_gib"]), 2), fmt(float(r["peak_before_step_gib"]), 2), fmt(float(r["peak_during_step_gib"]), 2),
                         fmt(float(r["allocated_after_step_gib"]), 2), fmt(float(r["iter_ms"])), fmt(float(r["optimizer_step_ms"]))])  # fmt: skip
        write_table("sharded", ["model", "sharded", "rank", "params", "opt. state", "peak after init", "peak before step", "peak in step", "held after step",
                                "iter. (ms)", "opt. step (ms)"], rows, align="left, left, right, right, right, right, right, right, right, right, right")  # fmt: skip

    fs = read("fsdp.csv")
    if fs is not None:
        rows = []
        for _, r in fs.iterrows():
            if r["status"] != "ok":
                rows.append([r["size"], int(r["num_layers"]), "bf16" if r["bf16"] else "fp32", "yes" if r["prefetch"] else "no"] + ["OOM"] * 6)
                continue
            rows.append([r["size"], int(r["num_layers"]), "bf16" if r["bf16"] else "fp32", "yes" if r["prefetch"] else "no", fmt(float(r["full_params_gib"]), 2),
                         fmt(float(r["local_params_gib"]), 2), pm(r["step_ms"], r["step_std_ms"]), fmt(float(r["blocked_on_gather_ms"])),
                         f"{int(r['gather_ready'])} / {int(r['gather_hits']) + int(r['gather_misses'])}", fmt(float(r["peak_gib"]), 2)])  # fmt: skip
        write_table("fsdp", ["model", "layers", "compute dtype", "prefetch", "full params (GiB)", "per-rank params (GiB)", "step (ms)", "blocked on all-gather (ms/step)",
                             "gathers already complete when needed", "peak / GPU (GiB)"], rows, align="left, right, left, left, right, right, right, right, right, right")  # fmt: skip

    lb = read("leaderboard.csv")
    if lb is not None:
        rows = []
        for _, r in lb.iterrows():
            cfg = f"ctx {int(r['ctx_len'])}, {int(r['num_layers'])} layers"
            if r["status"] != "ok":
                rows.append([cfg, r["variant"], "OOM", "--", "--"])
            else:
                rows.append([cfg, r["variant"], fmt(float(r["step_ms"])), fmt(float(r["peak_gib"]), 2), f"{float(r['first_loss']):.3f}"])
        write_table("leaderboard", ["configuration", "variant", "training step (ms)", "peak (GiB)", "loss at step 1"], rows, align="left, left, right, right, right")


def trace_figures() -> None:
    """Timeline of GPU kernels on rank 0: communication (NCCL) vs. compute, from torch.profiler."""
    for name, title in (("trace_ddp_naive", "DDP, naive: all-reduce per parameter after backward"), ("trace_ddp_overlap", "DDP, overlapped: async all-reduce as gradients become ready"),
                        ("trace_fsdp", "FSDP: all-gather / reduce-scatter vs. compute")):  # fmt: skip
        path = RES / f"{name}.csv"
        if not path.exists():
            print(f"[skip] {name}")
            continue
        df = pd.read_csv(path)
        df = df[~df["name"].str.startswith("nccl:")]  # profiler annotations, not kernels
        t0 = df["start_us"].min()
        df["start_ms"] = (df["start_us"] - t0) / 1e3
        df["dur_ms"] = df["dur_us"] / 1e3
        is_comm = df["name"].str.contains("ncclDevKernel")
        fig, ax = plt.subplots(figsize=(8.2, 1.9))
        for mask, y, color, label in ((~is_comm, 1, "#3b6ea5", "compute kernels"), (is_comm, 0, "#d1603d", "NCCL kernels (communication)")):
            sub = df[mask]
            ax.broken_barh(list(zip(sub["start_ms"], sub["dur_ms"].clip(lower=0.05))), (y - 0.35, 0.7), facecolors=color, label=label)
        ax.set_yticks([0, 1], ["communication", "compute"])
        ax.set_xlabel("time within one training step on rank 0 (ms)")
        ax.set_title(title, fontsize=9)
        ax.margins(x=0.01)
        fig.tight_layout()
        fig.savefig(GEN / f"{name}.png", dpi=170)
        plt.close(fig)
        comm_ms = df[is_comm]["dur_ms"].sum()
        span = (df["start_ms"] + df["dur_ms"]).max()
        (GEN / f"{name}_stats.typ").write_text(f"{comm_ms:.0f} ms of NCCL kernel time in a {span:.0f} ms step")
        print(f"[figure] {name}: comm {comm_ms:.0f} ms of {span:.0f} ms, {int(is_comm.sum())} nccl kernels")


def copy_memory_figures() -> None:
    import shutil

    for png in (ROOT / "experiments" / "figures").glob("mem_*.png"):
        shutil.copy(png, GEN / png.name)


if __name__ == "__main__":
    os.chdir(ROOT)
    copy_memory_figures()
    e2e_tables()
    attention_tables()
    flash_tables()
    memory_tables()
    nsys_tables()
    distributed_tables()
    trace_figures()
