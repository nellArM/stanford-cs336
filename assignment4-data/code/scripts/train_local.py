"""Local launcher for scripts/train.py (which is left untouched).

The staff run uses 8 B200s x 128 sequences/device = 1,024 sequences per optimizer step.
On two shared 48 GB A6000s the same global batch is obtained with a smaller per-device
batch and gradient accumulation, which train.py already supports; the model, optimizer,
schedule and loss are exactly those of cs336_basics/train_config.py.  Only the number of
steps (compute budget), the evaluation cadence and the logging sink differ:

  * train/eval losses that train.py would send to wandb are appended to
    <output>/metrics.jsonl instead (wandb is never initialised);
  * paths point into the repo's local-shared-data directory.

Single GPU:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_local.py --train-bin X --train-steps 8
Two GPUs:
    CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 scripts/train_local.py --train-bin X
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as staff_train  # noqa: E402  (scripts/train.py)

from cs336_basics.train_config import Config, PathsConfig, TrainingConfig  # noqa: E402
from cs336_data.paths import OUTPUT_DIR, PALOMA_VALID_BIN  # noqa: E402

GLOBAL_BATCH_SEQUENCES = 8 * 128  # staff setup: 8 devices x 128 sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--valid-bin", default=str(PALOMA_VALID_BIN))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "filtered_cc"))
    parser.add_argument("--train-steps", type=int, default=1024, help="staff run: 16,384")
    parser.add_argument("--per-device-batch", type=int, default=32)
    parser.add_argument("--global-batch", type=int, default=GLOBAL_BATCH_SEQUENCES)
    parser.add_argument("--eval-interval", type=int, default=128)
    parser.add_argument("--eval-iterations", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    accumulation, remainder = divmod(args.global_batch, world_size * args.per_device_batch)
    assert remainder == 0, "global batch must be divisible by world_size * per-device batch"

    output = Path(args.output)
    is_master = int(os.environ.get("RANK", 0)) == 0
    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()
    if is_master:
        output.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text("")

    def log_to_file(metrics: dict, step: int | None = None) -> None:
        record = {"step": step, "elapsed_seconds": round(time.perf_counter() - started, 1)}
        record.update({k: float(v) for k, v in metrics.items()})
        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # train.py logs through `wandb.log` whenever a project name is set, and only calls
    # `wandb.init` when an entity is set as well.  Redirect the former to a local file.
    staff_train.wandb.log = log_to_file

    cfg = Config(
        paths=PathsConfig(train_bin=Path(args.train_bin), valid_bin=Path(args.valid_bin), model_output=output),
        training=TrainingConfig(
            train_batch_size=args.per_device_batch,
            gradient_accumulation_steps=accumulation,
            train_steps=args.train_steps,
            eval_interval=args.eval_interval,
            eval_iterations=args.eval_iterations,
            eval_batch_size=args.eval_batch_size,
            compile=not args.no_compile,
            wandb_project="local-file",
            wandb_entity=None,
            log_interval=4,
        ),
    )
    staff_train.train_from_config(cfg)


if __name__ == "__main__":
    main()
