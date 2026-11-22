"""Turn a ``torch.cuda.memory._dump_snapshot`` pickle into an active-memory timeline plot and a
list of the largest allocations with their Python stack (problem memory_profiling (a), (e)).

This reproduces offline what pytorch.org/memory_viz shows (the machine used has no browser).

    uv run python -m cs336_systems.bench.memory_analysis snapshot.pickle --title "xl forward" --fig out.png
"""

from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_events(path: str):
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    traces = max(snapshot["device_traces"], key=len)
    return [e for e in traces if e["action"] in ("alloc", "free_completed")]


def timeline(events):
    active, series = 0, []
    for e in events:
        active += e["size"] if e["action"] == "alloc" else -e["size"]
        series.append(active)
    base = -min(0, min(series)) if series else 0  # frees of tensors allocated before recording started
    return [(s + base) / 2**30 for s in series]


def describe_frames(frames, limit: int = 4) -> str:
    keep = [f for f in frames if "cs336" in f["filename"] or "einops" in f["filename"]]
    keep = keep or frames
    return " <- ".join(f"{f['name']}@{f['filename'].split('/')[-1]}:{f['line']}" for f in keep[:limit])


def top_allocations(events, top: int = 8):
    allocs = sorted((e for e in events if e["action"] == "alloc"), key=lambda e: -e["size"])
    out, seen = [], {}
    for e in allocs:
        key = (e["size"], describe_frames(e.get("frames", [])))
        seen[key] = seen.get(key, 0) + 1
    for (size, where), count in sorted(seen.items(), key=lambda kv: -kv[0][0])[:top]:
        out.append((size / 2**20, count, where))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("snapshot")
    p.add_argument("--title", default="")
    p.add_argument("--fig", default=None)
    args = p.parse_args()

    events = load_events(args.snapshot)
    series = timeline(events)
    print(f"{args.title}: {len(events)} events, peak active (relative to start of recording) {max(series):.2f} GiB")
    for size_mib, count, where in top_allocations(events):
        print(f"  {size_mib:9.1f} MiB x{count:<4d} {where}")
    if args.fig:
        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.fill_between(range(len(series)), series, step="post", color="#3b6ea5", alpha=0.85, linewidth=0)
        ax.set_xlabel("allocator event index (time order)")
        ax.set_ylabel("active memory (GiB)")
        ax.set_title(args.title, fontsize=10)
        ax.margins(x=0)
        fig.tight_layout()
        fig.savefig(args.fig, dpi=170)


if __name__ == "__main__":
    main()
