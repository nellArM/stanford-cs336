# CS336 Assignment 5 — Alignment: reasoning RL on GSM8K

Self-study solution to the Spring 2026 version (handout v26.0.0) of Stanford CS336 assignment 5.
Only the **main assignment** is attempted; the optional supplement (instruction tuning, safety
evaluation and DPO on Llama 3.1 8B) is not, so its adapters in `tests/adapters.py` are left
unimplemented and `tests/test_data.py`, `tests/test_dpo.py` and `tests/test_metrics.py` fail by design.

## What the assignment covers

Improving the maths-word-problem accuracy of a small base language model
(`allenai/OLMo-2-0425-1B`) on GSM8K with reinforcement learning:

1. **Prompting baselines** — zero-shot `question_only`, zero-shot `r1_zero` (chain of thought with
   `<think>`/`<answer>` tags) and 3-shot `r1_zero`.
2. **GRPO** — policy gradients from first principles (REINFORCE, baselines, advantage normalisation,
   length normalisation) and a full on-policy training loop with vLLM generating the rollouts.
3. **Estimator variants** — GRPO with constant normalisation, Dr. GRPO, rejection fine-tuning (RFT), MaxRL.
4. **Off-policy training** — 32 optimizer steps per rollout batch with no reweighting, unclipped
   token-level importance weights, PPO/GRPO clipping, and GSPO sequence-level clipping.
5. **An estimator of our own** — "sqrt-MaxRL" (see the report).

## Where things are

| Path | Contents |
|---|---|
| `cs336_alignment/grpo.py` | The tested components: `tokenize_prompt_and_output`, `get_response_log_probs`, `compute_rollout_rewards`, `compute_group_normalized_rewards`, `compute_policy_gradient_loss`, `aggregate_loss_across_microbatch`, `grpo_train_step` |
| `cs336_alignment/train_grpo.py` | GRPO training loop (all on-policy and off-policy variants), local JSONL logging |
| `cs336_alignment/gsm8k.py` | Data loading, prompt formatting, reward-function selection, sampling parameters |
| `cs336_alignment/sweeps.py` | Experiment lists: `scaled` (what was run), `handout` (as specified), `smoke` |
| `scripts/eval_gsm8k_baselines.py` | Prompting baselines with the three-category breakdown |
| `scripts/run_queue.py` | Runs a sweep sequentially; skips finished runs, safe to restart, waits for free GPU memory, per-run timeout |
| `scripts/check_weight_sync.py` | Verifies that a weight sync changes (and then restores) what vLLM generates |
| `scripts/check_sync_drift.py` | Repeated syncs of unchanged weights: vLLM outputs and log-probs vs. the HF policy |
| `scripts/check_offpolicy_ratios.py` | Importance ratios before any update, for both old-log-prob scoring modes |
| `scripts/plot_grpo.py` | Turns the logs into the figures and tables included by the report |
| `tests/adapters.py` | Thin glue from the tests to `cs336_alignment/grpo.py` |
| `writeup/writeup.typ`, `writeup/writeup.pdf` | The report (Typst source and PDF) |

Provided course files (`vllm_utils.py`, `drgrpo_grader.py`, `checkpoint.py`, `modal_utils*.py`, prompts,
tests) are unchanged.

## Deviations from the handout

The handout budgets two B200 GPUs per run on Modal. This work used two shared RTX A6000 cards, so:

- runs are **200 rollout steps** as in the handout, but periodic validation uses 256 examples every 20
  steps (handout: 1,024 every 10), with a final evaluation on 1,024 examples;
- **fewer seeds**: 2 for standard GRPO, the on-policy variants and the own estimator; 1 for the
  learning-rate, prompt and off-policy runs (handout: 4);
- the learning-rate-3e-5 run diverged around step 80 and then ran out of GPU memory in all three attempts;
  it has no final evaluation (its last 256-example evaluation is reported instead);
- off-policy minibatches hold one prompt's whole group (8 rollouts), a choice the handout leaves open; a first
  implementation that shuffled rollouts across prompts collapsed (see "Off-policy collapse" below);
- the policy weights are bfloat16 as the handout prescribes; note that only 3--9% of the entries of a probed weight
  matrix change in an average optimizer step at these learning rates (bf16 rounding);
- the policy uses PyTorch `sdpa` attention instead of `flash_attention_2` (the pinned flash-attn wheel
  needs glibc ≥ 2.32, which the machine lacks) — the same computation;
- vLLM runs with a reduced `gpu_memory_utilization` because the card is shared;
- NCCL uses its shared-memory transport (`NCCL_P2P_DISABLE=1`, set automatically by the scripts): with the
  default direct GPU-to-GPU transport the first collective between the trainer and vLLM hangs on this machine;
- rollouts are requested one prompt at a time with seed `base + index`, because vLLM applies a request's seed to
  every prompt in a batched request, which correlates the samples of different questions;
- logging is local (JSONL + matplotlib) rather than Weights & Biases; Modal is not used.

The report states these again next to the results they affect.

## Results

Final validation reward on 1,024 GSM8K test examples after 200 rollout steps (`r1_zero` prompt unless noted;
learning rate 1e-5 unless noted). The binomial standard error of one evaluation is about 0.015; the
seed-to-seed spread is up to 0.16, so differences between methods below about 0.1 are not significant.

| Experiment | Run(s) | Final val. reward |
|---|---|---|
| Prompting baseline (no training) | question_only / r1_zero / r1_zero_three_shot | 0.001 / 0.001 / 0.193 |
| Standard GRPO (target ≥ 0.25 ✔) | seeds 0, 1 | 0.461, 0.450 (mean 0.456) |
| Learning rate 3e-6 | seed 0 | 0.149 (still rising) |
| Learning rate 3e-5 | seed 0 | diverged at step ~80 (last interim 0.35, best 0.41), then out of memory |
| Prompt: question_only | seed 0 | 0.388 (100-step cold start) |
| Prompt: r1_zero_three_shot | seed 0 | 0.462 (no cold start) |
| GRPO_constant | seeds 0, 1 | 0.440, 0.368 |
| Dr. GRPO | seeds 0, 1 | 0.284, 0.446 |
| RFT | seeds 0, 1 | 0.423, 0.432 |
| MaxRL | seeds 0, 1 | 0.429, 0.409 |
| sqrt-MaxRL (own estimator) | seeds 0, 1 | 0.443, 0.423 |
| Off-policy naive (32 minibatches / rollout batch) | seed 0 | 0.450 (2 of 3 attempts diverged) |
| Off-policy noclip | seed 0 | 0.505 |
| Off-policy clip (ε = 0.2; clip fraction 0.8%) | seed 0 | 0.439 |
| Off-policy GSPO (ε = 3e-4; clip fraction 44%) | seed 0 | 0.478 |

### Off-policy collapse

The first off-policy implementation shuffled the 256 rollouts across prompts before cutting minibatches of 8.
Naive, noclip and clip then collapsed (final reward 0.000: 5-token answers, or long unformatted text) and GSPO
stalled for 60 steps. The cause: GRPO advantages sum to zero within each group, which cancels the gradient on tokens
that all responses share (the answer format). Spread over separate minibatches, each formatted wrong answer gets its
own full-size AdamW step pushing that format down. Ratios, loss sign and clipping, pruning, masking and weight sync
were all checked and are correct (`scripts/check_offpolicy_ratios.py`, `scripts/check_sync_drift.py`). Keeping each
group in one minibatch (`--minibatch-order grouped`, now the default) fixes it; the four off-policy runs above were
re-run that way. Old log-probs are now also scored with exactly the training forward's batch shapes
(`--old-logprob-scoring matched`), which makes the importance ratio exactly 1 before the first update; separate
scoring batches had left 21% of sequences outside GSPO's ±3e-4 band from bf16 noise alone. The collapsed runs are
kept in `experiments/*_shuffled_s0` and analysed in the report.

## Running

```bash
# Environment. flash-attn is skipped because of the glibc requirement; keep using the venv's
# interpreter directly (a bare `uv run` would try to reinstall it).
uv sync --extra gpu --no-install-package flash-attn
unset LD_LIBRARY_PATH        # a system CUDA toolkit on this path shadows the wheels' CUDA libraries

# Tests for the main assignment
.venv/bin/python -m pytest tests/test_grpo.py

# Prompting baselines (one GPU; --vllm-gpu is the physical GPU index)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_gsm8k_baselines.py --vllm-gpu 0

# One GRPO run: policy on cuda:1 of the visible devices, vLLM on physical GPU 0
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m cs336_alignment.train_grpo --run-name grpo_s0 --variant grpo --seed 0

# Check that weight sync works on this machine (about 2 minutes)
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/check_weight_sync.py

# The 20 reported runs, sequentially and restartable. Each run waits until both cards have enough
# free memory, is retried up to 3 times and is killed after 6 hours; extra flags are passed to every
# run. --only selects runs or groups (standard, lr, prompt, variants, own, offpolicy).
CUDA_VISIBLE_DEVICES=0,1 nohup .venv/bin/python scripts/run_queue.py --sweep scaled > experiments/logs/queue.log 2>&1 &

# The collapsed first off-policy implementation (ablation)
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/run_queue.py --sweep offpolicy_shuffled

# Diagnostics: weight sync vs. HF log-probs over repeated syncs; importance ratios before any update
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/check_sync_drift.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/check_offpolicy_ratios.py experiments/offpolicy_clip_shuffled_s0/rollouts_step0.jsonl

# Figures, tables and the report
.venv/bin/python scripts/plot_grpo.py
uv run --no-project --with typst python -c "import typst; typst.compile('writeup/writeup.typ', output='writeup/writeup.pdf')"
```

Each run writes `experiments/<run_name>/{config.json, metrics.jsonl, rollouts_step*.jsonl,
val_generations_step*.jsonl, final.json}`. `experiments/` is not part of the submission archive.

Variants accepted by `--variant`: `grpo`, `grpo_constant`, `dr_grpo`, `rft`, `maxrl`, `sqrt_maxrl`,
`offpolicy_naive`, `offpolicy_noclip`, `offpolicy_clip`, `offpolicy_gspo`. Any field of the `Config`
dataclass in `train_grpo.py` can be overridden on the command line (for example `--learning-rate 3e-5`,
`--prompt question_only`, `--num-rollout-steps 200`).
