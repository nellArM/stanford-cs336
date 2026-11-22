# CS336 Assignment 2 — Systems (Spring 2026 handout v26.1.3)

Self-study solution. The assignment is about making Transformer training fast and
memory-efficient: profiling and benchmarking a language model, mixed precision, memory
profiling and gradient checkpointing, writing a FlashAttention-2 kernel in Triton, and
building distributed training (DDP, optimizer-state sharding, FSDP) on top of
`torch.distributed`, plus pen-and-paper analysis of parallelism strategies.

* **Report:** `writeup/writeup.pdf` (source `writeup/*.typ`; every table and figure is
  generated from `experiments/results/` by `writeup/make_tables.py`).
* **Tests:** all 14 provided tests pass (`uv run pytest`), the distributed ones 5× in a row
  on GPU and 5× on CPU.

## What was implemented, and where

| Handout problem | Code |
|---|---|
| `benchmarking_script`, `benchmarking_mixed_precision`, `torch_compile` (b) | `cs336_systems/bench/benchmark.py` (timing, bf16 autocast, `torch.compile`, NVTX ranges, memory snapshots) |
| `nsys_profile` | `experiments/run_nsys.sh`, `cs336_systems/bench/nsys_analysis.py` (per-NVTX-range kernel summary from the nsys SQLite export) |
| `memory_profiling` | `cs336_systems/bench/memory_analysis.py` (timeline + largest allocations from a snapshot), `cs336_systems/bench/residuals.py` (saved tensors per block) |
| `gradient_checkpointing` | `cs336_systems/bench/checkpoint_bench.py` |
| `pytorch_attention`, `torch_compile` (a) | `cs336_systems/bench/attention_bench.py` |
| `flash_forward`, `flash_backward` | `cs336_systems/flash_attention.py`: `FlashAttentionPyTorch` (tiled PyTorch), `FlashAttentionTriton` (Triton forward with causal flag + compiled PyTorch backward — the class returned by the adapter), `FlashAttentionTritonFull` (optional Triton two-pass backward) |
| `flash_benchmarking` | `cs336_systems/bench/flash_bench.py` (sweep and tile-size tuning) |
| `distributed_communication_single_node` | `cs336_systems/bench/allreduce_bench.py` |
| `naive_ddp`, `minimal_ddp_flat_benchmarking`, `ddp_overlap_individual_parameters` | `cs336_systems/ddp.py` (`DDP(module, mode="naive" | "flat" | "overlap")`), `cs336_systems/bench/ddp_bench.py` |
| `optimizer_state_sharding` (+ accounting) | `cs336_systems/sharded_optimizer.py`, `cs336_systems/bench/sharded_bench.py` |
| `fsdp` (+ accounting) | `cs336_systems/fsdp.py`, `cs336_systems/bench/fsdp_bench.py` |
| `leaderboard` (scaled down, not submitted) | `cs336_systems/fast_step.py` (FlashAttention hook, fused LM-head + cross-entropy, in-place AdamW), `cs336_systems/bench/leaderboard.py` |
| Written problems (mixed precision, checkpointing strategy, ZeRO comparison, Section 8 parallelism maths) | `writeup/writeup.pdf` |

`tests/adapters.py` only wires these classes to the tests. The vendored staff model in
`cs336-basics/` is used unchanged.

## Headline results

Measured on RTX A6000 cards (48 GB) that were **shared with other jobs**, so absolute
timings are noisy; see the note at the top of the report.

* FP32 forward / backward / AdamW step at batch 4, context 512: 0.15 / 0.33 / 0.09 s (small),
  0.94 / 1.88 / 0.66 s (large). BF16 autocast: forward 2.0–2.9× faster, backward 1.7–1.9×.
* A training step of the xl model needs ≈ 51 GiB (context 128) to ≈ 250 GiB (context 2048)
  and cannot run on one 48 GB card; with one checkpoint per block the full xl model does a
  forward+backward at batch 4, context 2048 with a 37.9 GiB peak.
* FlashAttention-2 (Triton forward + Triton backward) vs. PyTorch attention, batch 1, causal,
  d = 64, BF16: 0.84 vs 11.8 ms at sequence length 8,192 and 9.4 vs 184 ms at 32,768; it runs
  at 65,536 where PyTorch attention is out of memory.
* DDP on 2 GPUs (large model): 3.98 s/step naive (71% communication), 3.84 s flat,
  3.29 s with overlapped per-parameter all-reduces.
* Optimizer-state sharding halves the AdamW state per rank (7.22 → 3.6 GiB, large model) but
  costs 3.35 → 5.30 s per iteration on this machine's slow GPU-to-GPU path.
* FSDP trains the full xl model on 2 GPUs with a 29.0 GiB peak per GPU (unsharded: ≥ 50.8 GiB).
* Scaled leaderboard configuration: 1.19 s → 0.56 s per training step and 37.6 → 12.9 GiB.

## Deviations from the handout (all stated in the report)

* Hardware: shared RTX A6000s instead of dedicated B200s; only two GPUs, so all multi-GPU
  runs use 2 processes (the all-reduce benchmark adds 2/4/6 CPU processes with Gloo).
* `NCCL_P2P_DISABLE=1` is required on this machine (direct GPU peer-to-peer transfers hang);
  it is set automatically by `cs336_systems/bench/dist_utils.py`.
* "xl on 2 GPUs" benchmarks for DDP and optimizer sharding use the large and medium models
  (xl does not fit unsharded); memory profiling of xl training uses 1–8-layer xl models and
  extrapolates; FSDP is run on the full xl model.
* No GUI/browser: Nsight Systems captures are analysed from the exported SQLite database,
  and memory-viz / Nsight screenshots are replaced by plots of the same data.
* The 8B-parameter leaderboard configuration cannot fit; a scaled configuration was
  optimised instead and nothing was submitted to the leaderboard.
* The cells that first failed only because the card was shared were re-measured on a nearly
  free card with `experiments/run_pending.sh`; every OOM left in the tables is a limit of a
  48 GB card (xl/10B training steps, attention at sequence length 16,384, the large model's
  training step at context 1,024 under nsys). The large model's eager and compiled training
  steps were both re-measured on the same idle card.

## How to run

```bash
uv sync                                   # torch is pinned to the CUDA 12.6 wheels (driver 555.x)
uv run pytest                             # 14 tests; Triton tests need a GPU
uv run python experiments/check_flash.py  # extra FlashAttention checks (bf16, odd lengths, heads)

# experiments (each script waits for free GPU memory and retries; see the headers)
CUDA_VISIBLE_DEVICES=1 bash experiments/run_e2e.sh         # timing sweeps
CUDA_VISIBLE_DEVICES=1 bash experiments/run_nsys.sh        # Nsight Systems profiles (needs nsys >= 2024)
bash experiments/run_remaining.sh                          # memory, checkpointing, attention, flash, leaderboard
CUDA_VISIBLE_DEVICES=1,0 bash experiments/run_distributed.sh light   # all-reduce, DDP, sharding, FSDP (16-layer xl)
CUDA_VISIBLE_DEVICES=1,0 bash experiments/run_distributed.sh heavy   # large-model DDP/sharding, full-xl FSDP
bash experiments/run_pending.sh                            # re-measures the cells that need an (almost) free 48 GB card

# report
uv run python writeup/make_tables.py
uv run --no-project --with typst python writeup/compile.py
```

Raw results are in `experiments/results/` (CSV plus a few text outputs); `experiments/nsys/`
and `experiments/logs/` hold profiler captures and logs and are excluded from the
submission zip by `test_and_make_submission.sh`.
