# CS336 Assignment 1 (basics): Building a Transformer LM — self-study solution

Handout: `cs336_assignment1_basics.pdf` (Spring 2026). The report with all written answers, figures
and the experiment log is `writeup.pdf`.

## What the assignment covers

Build everything needed to train a small Transformer language model from scratch, without using
`torch.nn` layers or optimizers:

1. **Byte-pair-encoding tokenizer** — training (`train_bpe`) and a `Tokenizer` class with encoding,
   memory-bounded streaming encoding and decoding; tokenizer experiments on TinyStories and OpenWebText.
2. **Transformer language model** — `Linear`, `Embedding`, RMSNorm, SwiGLU, rotary position
   embeddings, scaled dot-product attention, causal multi-head self-attention, pre-norm blocks, the
   full LM; resource accounting (parameters, FLOPs, memory).
3. **Training machinery** — cross-entropy, AdamW, cosine schedule with warm-up, gradient clipping,
   batch sampling, checkpointing, a configurable training script, and a decoder with temperature and
   top-p sampling.
4. **Experiments** — learning-rate and batch-size sweeps, text generation, four architecture
   ablations (no RMSNorm, post-norm, no position embedding, SiLU instead of SwiGLU), a run on
   OpenWebText, and a time-limited "leaderboard" run.

## What is in this folder

| Path | Content |
|---|---|
| `writeup.pdf` | Report: answers to every written problem, learning curves, experiment log |
| `code.zip` | Output of the repository's `make_submission.sh` |
| `code/cs336_basics/bpe.py` | BPE training: parallel pre-tokenization, incremental pair counts, heap with lazy invalidation |
| `code/cs336_basics/tokenizer.py` | `Tokenizer` (`encode`, `encode_iterable`, `decode`, `from_files`) |
| `code/cs336_basics/model.py` | All model components, ablation switches, `generate` |
| `code/cs336_basics/nn_utils.py`, `optimizer.py`, `data.py` | Softmax / cross-entropy / clipping, AdamW and schedule, batching and checkpoints |
| `code/cs336_basics/train.py`, `generate.py` | Training loop with local logging; sampling from a checkpoint |
| `code/cs336_basics/train_tokenizer.py`, `encode_dataset.py`, `tokenizer_experiments.py` | Tokenizer training, dataset encoding to `uint16`, tokenizer statistics |
| `code/tests/adapters.py` | Glue between the tests and the implementation |
| `code/experiments/` | Run queues (`run_all.py`, `run_extra.py`), figure script, and each run's `metrics.jsonl` |
| `code/writeup/` | Typst source of the report, figures, generated samples, accounting script |

Datasets, encoded token arrays, checkpoints and the trained vocabularies' large files are not
included; everything can be regenerated with the commands below.

## Results

- Tests: `uv run pytest` → 47 passed, 1 expected failure (the handout's intentionally xfailed test).
- BPE: TinyStories 10K vocabulary in 19.6 s; OpenWebText 32K vocabulary in 23.6 min (10.2 GB RAM).
- TinyStories, full budget (327.68M tokens): **validation loss 1.329** (target ≤ 1.45).
- TinyStories, 40M tokens: best learning rate 3e-3 (loss 1.664); 1e-2 and above diverge.
- Ablations at 40M tokens: no RMSNorm diverges at 3e-3 (1.79 at 1e-3); post-norm 1.682;
  no position embedding 1.788; SiLU feed-forward 1.674.
- OpenWebText, same model and steps: validation loss 3.932.
- Batch sizes at 40M tokens: 64 is best (1.641); 512 diverges at the scaled rate and reaches 2.070
  at the base rate; 1024 is out of memory.
- 45-minute OpenWebText run (local, not submitted): validation loss 3.920 with the base model and a
  schedule fitted to the time budget; a 6-layer tied-embedding attempt was worse (4.337).

## Deviations from the handout

- Hardware: one RTX A6000 (48 GB) shared with other jobs instead of a B200; wall-clock numbers are
  slower and noisy.
- The learning-rate sweep, batch-size sweep and ablations use the handout's low-resource budget
  (40M tokens) rather than the full budget; one TinyStories and one OpenWebText run use the full budget.
- Batch size 1024 does not fit in 48 GB with this implementation and is reported as out of memory.
- The "leaderboard" run was done locally for 45 minutes on the A6000 and not submitted.
- Logging is local (JSON lines) instead of a hosted tracker.
- `pyproject.toml` pins the PyTorch CUDA 12.6 wheels, because the machine's driver does not support
  the CUDA 13 build in the original lockfile.

## Reproducing

```sh
uv run pytest                                                   # tests
uv run python -m cs336_basics.train_tokenizer --help            # BPE training
uv run python -m cs336_basics.encode_dataset --help             # encode to uint16 .npy
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_all.py     # every training run (restart-safe)
CUDA_VISIBLE_DEVICES=0 uv run python experiments/run_extra.py   # follow-up runs
uv run --with matplotlib python experiments/make_figures.py     # figures + summary
uv run --no-project --with typst python -c \
  "import typst; typst.compile('writeup/writeup.typ', output='writeup/writeup.pdf')"
```
