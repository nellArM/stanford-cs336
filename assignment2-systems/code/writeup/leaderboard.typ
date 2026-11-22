#import "lib.typ": gen

== Problem `leaderboard`

The official benchmark (≈ 8 B parameters, context 32,768, vocabulary 151,936, two B200s) cannot be run here: its weights alone are 16 GiB in BF16 and 32 GiB in FP32, before gradients and optimizer state, against 48 GB cards that were shared. Nothing was submitted to the leaderboard. Instead I applied the suggested optimisations to a *scaled-down configuration with the same character* — vocabulary 151,936, 128-dimensional heads, BF16 autocast, causal attention, batch 2, but $d_"model" = 1024$, $d_"ff" = 2752$, 8 heads, 8 layers, context 4096 (412 M parameters) — on one A6000, timed with `triton.testing.do_bench` exactly as in the official test (`cs336_systems/bench/leaderboard.py`). A second configuration (context 16,384, 12 layers) shows the memory effect.

The optimisations (`cs336_systems/fast_step.py`), none of which changes the model's input/output behaviour:

- *flash* — the model's attention is routed through the Triton FlashAttention-2 forward and two-pass Triton backward kernels (causal early exit, no $N times N$ tensors).
- *ce* — a fused LM-head + cross-entropy `autograd.Function`: logits are produced, turned into the loss and into gradients chunk by chunk (2,048 tokens at a time), so the (batch, seq, vocab) logits tensor — 2.3 GiB in BF16 here, 19 GiB in the official configuration — is never materialised or saved.
- *adamw* — an AdamW with the same update rule as the reference but using in-place operations.

#figure(gen("leaderboard"), caption: [Full training step (forward, loss, backward, AdamW) of the scaled leaderboard configuration, one RTX A6000, BF16. The loss at the first step is identical for all variants, confirming unchanged behaviour (after one update the losses agree to $3 times 10^(-4)$).])

The baseline needs 1.19 s and 37.6 GiB per step. FlashAttention alone halves the time (0.62 s) and saves 12 GiB; the fused LM-head/cross-entropy saves another 12.7 GiB (12.9 GiB peak) at slightly better speed; in-place AdamW brings the step to *0.56 s, 2.1× faster than the baseline at one third of the memory*. At context 16,384 with 12 layers the baseline cannot run at all (its attention matrices alone would be 8 GiB per layer per tensor), while the optimised step takes 3.2 s and 33.6 GiB. Ideas not implemented: Triton autotuning of tile sizes per shape, fusing AdamW into a single kernel, a second GPU, and activation checkpointing (not needed at this size).
