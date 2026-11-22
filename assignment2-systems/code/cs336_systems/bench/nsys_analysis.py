"""Summarise an Nsight Systems capture (exported with ``nsys export --type sqlite``).

Kernels are attributed to NVTX ranges through the CUDA runtime call that launched them
(joined on the correlation id), so the attribution is correct even though kernels execute
asynchronously after the Python range has been left.  This is the offline equivalent of
"CUDA GPU Kernel Summary" filtered by NVTX range in the Nsight GUI.

    uv run python -m cs336_systems.bench.nsys_analysis profile.sqlite --csv out.csv --label small-512
"""

from __future__ import annotations

import argparse
import bisect
import sqlite3
from collections import defaultdict

from cs336_systems.bench.common import append_row

MATMUL_MARKERS = ("gemm", "cutlass", "gemv", "matmul", "sgemm", "xmma", "kernel2")  # Kernel2 = cutlass::Kernel2<...sgemm...>


def is_matmul(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in MATMUL_MARKERS)


def short(name: str, width: int = 70) -> str:
    return name if len(name) <= width else name[: width - 3] + "..."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sqlite")
    p.add_argument("--label", default="")
    p.add_argument("--csv", default=None)
    p.add_argument("--top", type=int, default=6)
    args = p.parse_args()

    con = sqlite3.connect(args.sqlite)
    strings = dict(con.execute("select id, value from StringIds"))
    ranges = con.execute("select text, start, end from NVTX_EVENTS where end is not null and text is not null order by start").fetchall()
    kernels = con.execute(
        "select k.start, k.end, k.shortName, r.start from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId = r.correlationId"
    ).fetchall()
    kernels = [(launch, end - start, strings.get(name_id, str(name_id))) for start, end, name_id, launch in kernels]
    kernels.sort()
    launches = [k[0] for k in kernels]

    def kernels_in(start: int, end: int):
        return kernels[bisect.bisect_left(launches, start) : bisect.bisect_right(launches, end)]

    measured = [(s, e) for text, s, e in ranges if text == "measured step"]
    n_steps = max(1, len(measured))

    def within_measured(s: int, e: int) -> bool:
        return any(ms <= s and e <= me for ms, me in measured)

    report = {}
    for stage in ("forward", "backward", "optimizer", "scaled dot product attention", "computing attention scores", "computing softmax", "final matmul"):
        spans = [(s, e) for text, s, e in ranges if text == stage and within_measured(s, e)]
        if not spans:
            continue
        by_kernel = defaultdict(lambda: [0, 0])
        wall = 0
        for s, e in spans:
            wall += e - s
            for _, dur, name in kernels_in(s, e):
                by_kernel[name][0] += dur
                by_kernel[name][1] += 1
        total = sum(v[0] for v in by_kernel.values())
        matmul = sum(v[0] for name, v in by_kernel.items() if is_matmul(name))
        report[stage] = dict(wall_ms=wall / 1e6 / n_steps, gpu_ms=total / 1e6 / n_steps, matmul_frac=matmul / total if total else 0.0,
                             kernels=sorted(by_kernel.items(), key=lambda kv: -kv[1][0]))  # fmt: skip

    print(f"=== {args.label}: {n_steps} measured steps ===")
    for stage, info in report.items():
        print(f"[{stage}] python wall (range) {info['wall_ms']:.2f} ms/step, GPU kernel time {info['gpu_ms']:.2f} ms/step, matmul share {100 * info['matmul_frac']:.1f}%")
        for name, (dur, count) in info["kernels"][: args.top]:
            share = 100 * dur / (info["gpu_ms"] * 1e6 * n_steps) if info["gpu_ms"] else 0
            print(f"    {share:5.1f}%  {dur / 1e6 / n_steps:9.3f} ms  x{count / n_steps:7.1f}/step  {short(name)}")
        if args.csv:
            top_name, (top_dur, top_count) = info["kernels"][0] if info["kernels"] else ("", (0, 0))
            append_row(args.csv, dict(label=args.label, stage=stage, wall_ms=round(info["wall_ms"], 3), gpu_kernel_ms=round(info["gpu_ms"], 3),
                                      matmul_share=round(info["matmul_frac"], 4), top_kernel=short(top_name, 60),
                                      top_kernel_ms=round(top_dur / 1e6 / n_steps, 3), top_kernel_calls_per_step=round(top_count / n_steps, 1)))  # fmt: skip


if __name__ == "__main__":
    main()
