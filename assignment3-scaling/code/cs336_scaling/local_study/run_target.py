"""Launch (or resume) the target run chosen by fit.py (reads experiments/prediction.json).

Usage:  CUDA_VISIBLE_DEVICES=1 uv run --extra study python -m cs336_scaling.local_study.run_target
"""

from __future__ import annotations

import json
import subprocess
import sys

from cs336_scaling.local_study.fit import PREDICTION_PATH, ROOT


def main() -> None:
    p = json.loads(PREDICTION_PATH.read_text())
    command = [
        sys.executable, "-m", "cs336_scaling.local_study.train",
        "--name", "target", "--group", "target",
        "--d-model", str(p["d_model"]), "--n-layer", str(p["n_layer"]),
        "--tokens", str(p["tokens"]), "--batch-size", str(p["batch_size"]),
        "--budget", repr(p["c_target"]), "--lr", repr(p["lr"]), "--n-evals", str(p["n_evals"]),
    ]  # fmt: skip
    print(" ".join(command), flush=True)
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
