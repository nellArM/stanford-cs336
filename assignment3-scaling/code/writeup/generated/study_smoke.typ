// Written after the smoke test of 2026-11-15 (experiments/smoke.sh, logs in experiments/runs/smoke_*.jsonl).
Before fixing $C_0$, three family members were trained for a few million tokens each on the GPU that
the study uses (RTX A6000, shared with other jobs; bf16 autocast, `torch.compile`). Throughput is the
best of the four evaluation chunks of each run, which excludes compilation.

#table(
  columns: 7,
  align: (left, right, right, right, right, right, right),
  table.header([*model*], [*N*], [*N#sub[eff]*], [*batch*], [*tokens/s*], [*$6 N D$ FLOP/s*], [*$6 N_"eff" D$ FLOP/s*]),
  [d256L4], [3.1M], [11.3M], [32], [174k], [$3.3 times 10^12$], [$1.19 times 10^13$],
  [d512L8], [25.2M], [41.5M], [16], [77k], [$1.17 times 10^13$], [$1.93 times 10^13$],
  [d768L12], [84.9M], [109.5M], [16], [45k], [$2.29 times 10^13$], [$2.95 times 10^13$],
)

- *The pipeline trains.* Validation loss fell monotonically in every run (e.g. d512L8: 6.13 → 5.67 → 5.42 → 5.32 over 5M tokens).
- *FLOP accounting.* Under the plain $6 N D$ rule the three models differ by a factor of 7 in FLOPs per
  second, so a "FLOP" of a small model would cost seven times the wall-clock of a "FLOP" of a large one.
  Counting the unembedding ($N_"eff"$) narrows this to a factor of 2.5. The rest is fixed per-step
  overhead (optimizer, data transfer, small batches) that weighs most on the smallest models; it is a
  property of this implementation and is not charged to the budget. The consequence is that small
  models are somewhat more expensive in wall-clock than their FLOP budget says, which the handout's
  wall-clock budget would have penalised.
- *Choice of $C_0$.* At $1.2$–$3 times 10^13$ FLOP/s, the $90 C_0$ cap with $C_0 = 3 times 10^15$ FLOPs
  corresponds to about 3–4 hours of training on an otherwise idle A6000, and the $360 C_0$ target run to
  about 8–12 hours.
- *Memory.* A fourth test with batch size 64 at d768L12 took 36 GB (the 32000-way logits of $64 times
  512$ positions dominate) and was stopped; batch size is therefore capped at 32, which peaks near
  12–20 GB and leaves room for the GPU's other users.
- *Contention.* An earlier attempt at the d256L4 point, made while another job held the same GPU at
  100 % utilisation, ran at 27k tokens/s, six times slower than the figure above. Wall-clock numbers in
  this report should be read with that in mind.
