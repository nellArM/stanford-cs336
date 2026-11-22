"""Run a list of GRPO experiments one after another.

- Runs that already have experiments/<run_name>/final.json are skipped, so the queue can be
  stopped and restarted at any time; an interrupted run is started again from scratch (its
  partial logs are moved aside).
- Before every attempt the queue waits until both GPUs have enough free memory (they are shared).
- A run is retried up to --max-attempts times and killed after --run-timeout-hours; the training
  script itself aborts if a weight sync hangs. A failed run is recorded and the queue moves on.
- Unknown flags are passed on to every train_grpo.py call (e.g. --num-rollout-steps 200).

The first GPU in CUDA_VISIBLE_DEVICES hosts vLLM and the second the policy (override with
--vllm-gpu / --policy-gpu, physical indices).

Example:
    CUDA_VISIBLE_DEVICES=0,1 nohup .venv/bin/python scripts/run_queue.py --sweep scaled \
        > experiments/logs/queue.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cs336_alignment import gsm8k
from cs336_alignment.sweeps import SWEEPS


def free_gpu_memory_gb(gpu: int) -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip().splitlines()[0]) / 1024


def wait_for_gpu_memory(required_gb: dict[int, float], max_wait_hours: float, name: str) -> bool:
    """Block until every GPU has the required free memory on two checks 30 s apart."""
    deadline = time.time() + max_wait_hours * 3600
    announced = False
    while True:
        ok = True
        for _ in range(2):  # must hold twice, so a job that is just starting up is not missed
            free = {gpu: free_gpu_memory_gb(gpu) for gpu in required_gb}
            if any(free[gpu] < need for gpu, need in required_gb.items()):
                ok = False
                break
            time.sleep(30)
        if ok:
            return True
        if not announced:
            print(f"[queue] {name}: waiting for GPU memory (free {free}, need {required_gb})", flush=True)
            announced = True
        if time.time() > deadline:
            return False
        time.sleep(120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="scaled", choices=sorted(SWEEPS))
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these run names or groups")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--policy-gpu", type=int, default=None, help="physical GPU for the policy (default: 2nd visible)")
    parser.add_argument("--vllm-gpu", type=int, default=None, help="physical GPU for vLLM (default: 1st visible)")
    parser.add_argument("--min-free-policy-gb", type=float, default=26.0, help="wait until the policy GPU has this much free")
    parser.add_argument("--min-free-vllm-gb", type=float, default=19.0, help="wait until the vLLM GPU has this much free")
    parser.add_argument("--max-wait-hours", type=float, default=48.0, help="give up waiting for GPU memory after this long")
    parser.add_argument("--run-timeout-hours", type=float, default=6.0, help="kill a run that takes longer")
    args, passthrough = parser.parse_known_args()  # unknown flags go to every train_grpo.py call

    experiments = gsm8k.REPO_ROOT / "experiments"
    experiments.mkdir(exist_ok=True)
    runs = SWEEPS[args.sweep]
    if args.only:
        runs = [r for r in runs if r["run_name"] in args.only or r["group"] in args.only]

    visible = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")]
    if len(visible) < 2:
        sys.exit("run_queue.py needs two GPUs in CUDA_VISIBLE_DEVICES (one for the policy, one for vLLM)")
    vllm_gpu = args.vllm_gpu if args.vllm_gpu is not None else visible[0]
    policy_gpu = args.policy_gpu if args.policy_gpu is not None else visible[1]
    # train_grpo.py takes the policy device as an index into the visible devices and the vLLM
    # GPU as a physical index (the provided VLLMServer sets CUDA_VISIBLE_DEVICES for the server).
    placement = ["--policy-device", f"cuda:{visible.index(policy_gpu)}", "--vllm-gpu", str(vllm_gpu)]

    env = os.environ.copy()
    # The system CUDA 12.2 libraries must not shadow the ones bundled with the torch wheels.
    env.pop("LD_LIBRARY_PATH", None)

    for i, run in enumerate(runs):
        name = run["run_name"]
        run_dir = experiments / name
        if (run_dir / "final.json").exists():
            print(f"[queue] ({i + 1}/{len(runs)}) {name}: already finished, skipping", flush=True)
            continue
        command = [sys.executable, "-m", "cs336_alignment.train_grpo"]
        for key, value in run.items():
            if key != "group":
                command += ["--" + key.replace("_", "-"), str(value)]
        if args.dry_run:
            print(" ".join(command + placement + passthrough))
            continue
        command += placement + passthrough

        for attempt in range(1, args.max_attempts + 1):
            # The cards are shared: start only when both have enough free memory, and re-check
            # before every attempt (vLLM refuses to start, and the policy runs out of memory, otherwise).
            if not wait_for_gpu_memory(
                {policy_gpu: args.min_free_policy_gb, vllm_gpu: args.min_free_vllm_gb}, args.max_wait_hours, name
            ):
                print(f"[queue] {name}: gave up waiting for GPU memory; stopping the queue", flush=True)
                return
            if run_dir.exists():
                run_dir.rename(experiments / f"{name}.interrupted-{int(time.time())}")
            run_dir.mkdir(parents=True)
            print(f"[queue] ({i + 1}/{len(runs)}) {name}: starting (attempt {attempt})", flush=True)
            start = time.time()
            with open(run_dir / "stdout.log", "w") as log:
                # Own session, so that on a timeout the trainer and its vLLM server die together.
                proc = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=gsm8k.REPO_ROOT, start_new_session=True
                )
                try:
                    code = proc.wait(timeout=args.run_timeout_hours * 3600)
                except subprocess.TimeoutExpired:
                    print(f"[queue] {name}: exceeded {args.run_timeout_hours} h, killing it", flush=True)
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait()
            # The vLLM server runs in its own session; make sure it is gone (only the PID this run recorded).
            pid_file = run_dir / "vllm.pid"
            if pid_file.exists():
                try:
                    os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                except (ProcessLookupError, ValueError, PermissionError):
                    pass
            minutes = (time.time() - start) / 60
            if code == 0 and (run_dir / "final.json").exists():
                print(f"[queue] {name}: finished in {minutes:.1f} min", flush=True)
                break
            print(f"[queue] {name}: FAILED with exit code {code} after {minutes:.1f} min", flush=True)
            time.sleep(30)  # give the GPUs time to release memory before retrying


if __name__ == "__main__":
    main()
