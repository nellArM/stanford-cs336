# CS336 Assignment 3 (scaling): Scaling Laws — self-study solution

Handout: `cs336_assignment3_scaling.pdf` (v26.0.5, Spring 2026). The report is `writeup/writeup.pdf`.

## What the assignment covers

1. **`chinchilla_isoflops` (5 pts).** Reproduce the IsoFLOP method of Hoffmann et al. (Chinchilla) on the
   synthetic runs in `data/isoflops_curves.json`: find the compute-optimal model size for each budget,
   fit power laws for the optimal model size and dataset size, and extrapolate to 1e23 and 1e24 FLOPs.
2. **`scaling_laws` (50 pts).** Use a training API with a capped budget (12 B200-hours) to fit a scaling
   law, then predict the best model configuration and its validation loss for a run four times larger
   (48 B200-hours).

## What was done

| Part | Status | Where |
|---|---|---|
| `chinchilla_isoflops` | Done as specified | `cs336_scaling/chinchilla_isoflops.py` |
| `scaling_laws` via the course API | **Not possible** (see below) | — |
| `scaling_laws`, local substitute study | Done: 21 grid runs (2.49e17 FLOPs, 5.8 A6000-hours), fit, prediction, and the target run (8.7 A6000-hours) | `cs336_scaling/local_study/` |

`chinchilla_isoflops` result: N_opt = 1.163 · C^0.469 and D_opt = 0.143 · C^0.531, giving about 7.0e10
parameters / 2.4e11 tokens at 1e23 FLOPs and 2.1e11 parameters / 8.1e11 tokens at 1e24 FLOPs.

### Deviation: the training API is Stanford-only

The API at `hyperturing.stanford.edu:8000` is reachable only from the Stanford network and is keyed by a
registered 8-digit student ID, so the main problem cannot be done as designed. It is replaced by a
small-scale study that keeps the task's structure and trains every run locally on one RTX A6000:

- **Model:** PyTorch port of the staff JAX model (pre-norm RMSNorm, QK-norm, RoPE, SwiGLU, optional
  weight tying) in `local_study/model.py`.
- **Data:** DCLM-baseline shards tokenized with the Llama-2 tokenizer, same shuffle seed and EOS
  convention as the staff pipeline, fixed order, no repetition; a 2^18-token validation set from a
  held-out shard (`local_study/data_prep.py`).
- **Training:** sequence length 512, AdamW with warmup + cosine decay, bf16 autocast, `torch.compile`,
  n evaluations per run, resumable (`local_study/train.py`).
- **Budget:** in FLOPs instead of wall-clock (the GPU is shared), C = 6 · (N + vocab · d_model) · D with
  N = 12 · n_layer · d_model². Scaling runs are capped at 90 · C0 = 2.7e17 FLOPs (about 3–4 A6000-hours);
  the target run gets 4× that, 1.08e18 FLOPs (about 8–12 A6000-hours).
- **Runs:** a brief learning-rate check, then IsoFLOP profiles at 4 budgets × 5 model sizes, extended
  automatically when a minimum falls on the edge of its window (`local_study/grid.py`).
- **Fit:** IsoFLOP minima → N_opt(C) power law to choose the target model; parametric
  L(N, D) = E + A/N^α + B/D^β to predict its loss (`local_study/fit.py`).
- **Check:** the target run is actually trained and compared with the prediction
  (`local_study/run_target.py`, `local_study/compare.py`).

Result: the IsoFLOP fit (N_opt ∝ C^0.69) chose a 14-layer, 896-wide model (135M parameters, 1.1B tokens)
for the 1.08e18-FLOP target; predicted validation loss 3.617, **actual 3.465** (prediction 0.15 nats /
4.4% too pessimistic; the best scaling run had 4.053).

Results at this scale do not transfer to the handout's 48 B200-hour setting, and nothing was submitted to
the course API or leaderboard.

Other local changes: a `study` extra in `pyproject.toml` (numpy, scipy, matplotlib, torch, pyarrow,
transformers) with torch taken from the CUDA 12.6 wheel index because this machine's driver does not
support CUDA 13; extra excludes in `make_submission.sh` for data, checkpoints and virtual environments.
The staff server code and its tests (which need PostgreSQL) are untouched and were not run.

## Reproducing

```bash
uv sync --extra study
uv run --extra study python -m cs336_scaling.chinchilla_isoflops           # figures + numbers
uv run --extra study python -m cs336_scaling.local_study.data_prep --min-train-tokens 5.8e9   # ~15 GB
CUDA_VISIBLE_DEVICES=0 uv run --extra study python -m cs336_scaling.local_study.grid        # LR check + grid, restart-safe
uv run --extra study python -m cs336_scaling.local_study.fit               # prediction.json + report section
CUDA_VISIBLE_DEVICES=0 uv run --extra study python -m cs336_scaling.local_study.run_target  # resumable
uv run --extra study python -m cs336_scaling.local_study.compare           # prediction vs. actual
uv run --no-project --with typst python writeup/compile.py                  # writeup/writeup.pdf
```

Run logs are JSON lines under `experiments/runs/`; generated report fragments are under
`writeup/generated/`.
