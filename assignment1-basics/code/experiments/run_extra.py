"""Follow-up runs added after looking at the first results (see the experiment log in the report).

    CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_extra.py
"""

from run_all import FULL, LOW, OWT, TS, run


def main() -> None:
    # Batch 512 at the sqrt-scaled LR (6e-3) is past the stability edge; retune with the base LR.
    # (Batch 1024 does not fit in the 48 GB card at all, so it is not retried.)
    for bs in [512]:
        steps = LOW // (bs * 256)
        every = ["--eval-every", str(max(10, steps // 12)), "--log-every", str(max(5, steps // 60))]
        run(f"bsretune_{bs}", *TS, "--total-tokens", str(LOW), "--batch-size", str(bs), "--lr", "3e-3", *every)

    # Second 45-minute OWT attempt: the baseline architecture (untied, 4 layers) with the cosine
    # schedule fitted to the wall-clock budget, because the 6-layer tied model was behind the
    # baseline at equal time.
    run(
        "owt_leaderboard_v2", *OWT, "--total-tokens", str(4 * FULL), "--max-seconds", "2700",
        "--lr", "3e-3", "--eval-every", "500", "--log-every", "100",
    )  # fmt: skip


if __name__ == "__main__":
    main()
