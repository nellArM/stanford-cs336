#import "lib.typ": gen

== Problem `pytorch_attention`

`cs336_systems/bench/attention_bench.py` times the course's `scaled_dot_product_attention` on random $bold(Q), bold(K), bold(V)$ of shape $(8, N, d)$ (no head dimension, no mask), with warm-up and a `torch.cuda.synchronize()` after every pass. It uses 100 forward and 100 backward passes whenever 100 passes take less than 6 s, and fewer (at least 5, see the "reps" column) for the largest sizes. Memory "before backward" is `torch.cuda.memory_allocated()` after the forward pass, i.e. inputs plus everything saved for the backward pass.

#figure(gen("attention"), caption: [Attention forward/backward time per pass and memory held before the backward pass, eager vs. `torch.compile`, measured on a card with 27–41 GiB free at sequence length 8192 (see `free_gib_at_start` in `attention.csv`). The OOM entries at sequence length 16384 are inherent to a 48 GB card, as the accounting below shows.]) <tab-attn>

*Timings.* Up to $N = 1024$ a pass costs 0.3–2 ms and is dominated by per-kernel overhead; from there on the cost grows as $N^2$ (≈ 13 ms forward and ≈ 33 ms backward at $N = 4096$, ≈ 50 ms and ≈ 125 ms at $N = 8192$) and is almost independent of $d$, because the $N times N$ element-wise work (masking, softmax) dominates the two matmuls, whose cost is proportional to $d$.

*Memory accounting.* One $(8, N, N)$ FP32 tensor takes $8 dot N^2 dot 4$ bytes: 0.5 GiB at $N = 4096$, 2 GiB at 8192 and 8 GiB at 16384. The measurements show 2.03 GiB held before the backward pass and a peak of 4.03 GiB at $N = 4096$, and 8.05 GiB / 16.05 GiB at $N = 8192$: *four* $N times N$ tensors stay alive for backward (the scaled scores, the shifted scores, their exponential and the normalised probabilities $bold(P)$ — each is an input or output that some backward formula needs) and up to eight exist at the peak; the inputs and the output are negligible ($3 dot 8 dot N dot d dot 4$ bytes, 0.1 GiB at $N = 8192$, $d = 128$). At $N = 16384$ the same accounting gives 32 GiB retained and a 64 GiB peak, more than the card has, so *16384 is the smallest sequence length that runs out of memory, for every $d$, already in the forward pass* — the run with 30.8 GiB free failed after allocating 40 GiB, i.e. on its sixth 8 GiB tensor.

The memory saved for backward grows quadratically with the sequence length (×4 for every doubling) and does not depend on $d$. To eliminate it, do not store the $N times N$ intermediates at all: save only $bold(Q), bold(K), bold(V), bold(O)$ and the per-row log-sum-exp $L$ ($O(N d)$ memory), and recompute the attention probabilities tile by tile in the backward pass, as FlashAttention-2 does; fusing the masking/softmax chain into one kernel also removes the transient copies in the forward pass.

== Problem `torch_compile`

*(a)* The "compiled" columns of @tab-attn are the same function wrapped in `torch.compile`. At $N = 256$ compilation makes no consistent difference (overhead-dominated); at $N = 1024$ it gains 30–45%, and at $N >= 4096$ it more than halves the forward pass (12.8 → 5.3 ms at 4096, 49 → 20 ms at 8192) and nearly halves the backward pass (32 → 17 ms, 124 → 62 ms), while lowering the peak from 16 to 12 GiB at $N = 8192$: the compiler fuses the chain of element-wise operations (divide, mask, subtract max, exp, normalise) into a couple of generated kernels, so the $N times N$ tensors are traversed fewer times and fewer temporaries exist. It cannot remove the tensors that autograd needs, so the memory before backward is unchanged (8.05 GiB at $N = 8192$), the quadratic scaling remains, and $N = 16384$ is out of memory either way.

*(b)* Compiling the whole Transformer (`--compile` in the end-to-end script):

#figure(gen("e2e_compile"), caption: [Whole-model `torch.compile`, FP32, batch 4, context 512 (5 warm-up steps exclude compilation time). The large model's training step was measured last, eager and compiled both on a nearly idle card. The other rows were measured earlier, under load.])

The forward pass gets 13–31% faster (154 → 107 ms for small) and a full training step 16–27% faster (572 → 420 ms for small, 1343 → 1082 ms for large), with a somewhat lower peak memory (5.0 → 4.3 GiB for small, 27.5 → 24.0 GiB for large). The gain is smaller for a training step than for inference because the optimizer step is not compiled, and it shrinks with model size as GEMMs — which compilation cannot speed up — take a larger share. For the large model's training step, re-measured eager and compiled on the same idle card, compilation saves 19%. That is in line with the trend of the other rows.

== Problem `flash_forward`

Implemented in `cs336_systems/flash_attention.py`:

- *(a)* `FlashAttentionPyTorch`: an `autograd.Function` that follows Algorithm 1 in plain PyTorch with 32 × 32 tiles, keeping the running maximum $m$, the running normaliser $l$ and the un-normalised output in FP32, and saving $L, bold(Q), bold(K), bold(V), bold(O)$.
- *(b)* `flash_fwd_kernel`: a Triton kernel with the prescribed signature and launch grid $(T_q, "batch")$; one program instance loads one query tile, loops once over the key tiles (advancing the K/V block pointers at the end of each iteration), accumulates with `tl.dot(..., acc=...)` into FP32 on-chip buffers and writes $bold(O)$ and $L$ in the input dtype. Inputs with extra leading dimensions (batch, heads) are flattened into the batch dimension.
- *(c)* `is_causal: tl.constexpr` adds $-10^6$ to scores whose key index exceeds the query index. In addition, a causal program instance stops its loop after the last key tile that contains any visible key, which skips about half of all tiles.

`FlashAttentionTriton` is the class returned by the adapter. All six attention tests pass (PyTorch and Triton, causal and non-causal); `experiments/check_flash.py` additionally checks outputs and gradients against a reference for BF16 and FP32, several shapes including a sequence length that is not a multiple of the tile size, and inputs with a head dimension. One observation from that check: for FP32 inputs the kernel agrees with the reference to about $2 times 10^(-3)$ rather than $10^(-6)$, because Triton's `tl.dot` uses TF32 tensor-core arithmetic for FP32 operands; this is within the tests' tolerance of $10^(-2)$.

== Problem `flash_backward`

The backward pass follows Equations 13–19: $D = "rowsum"(bold(O) compose bold(d O))$, then $bold(P) = exp(bold(S) - L)$ recomputed from the saved log-sum-exp, $bold(d V) = bold(P)^top bold(d O)$, $bold(d P) = bold(d O) bold(V)^top$, $bold(d S) = bold(P) compose (bold(d P) - D)$, $bold(d Q) = bold(d S) bold(K) \/ sqrt(d)$, $bold(d K) = bold(d S)^top bold(Q) \/ sqrt(d)$. It is an ordinary PyTorch function wrapped in `torch.compile` (with an eager fallback if no compiler is available), shared by the PyTorch and Triton classes; the three backward tests pass.

Beyond what is required, I also implemented the optional Triton backward of Section 4.2.3 (`FlashAttentionTritonFull`): two kernels, one per query tile accumulating $bold(d Q)$ and one per key tile accumulating $bold(d K), bold(d V)$, each recomputing $bold(P)$, so no program writes outside its own tile and no atomics are needed. It never materialises an $N times N$ tensor, which the compiled PyTorch backward still does.

== Problem `flash_benchmarking`

`cs336_systems/bench/flash_bench.py` uses `triton.testing.do_bench` with batch size 1 and causal masking on *one RTX A6000* (not a B200; this sweep happened to run while the card was otherwise nearly idle). "PT" is the course's attention with a boolean causal mask; "FA2" is the required configuration (Triton forward, compiled PyTorch backward); "FA2-T" adds the Triton backward. Times are in milliseconds.

#figure(gen("flash_bf16"), caption: [BF16: forward, backward and end-to-end (forward+backward) latency in ms.])
#figure(gen("flash_fp32"), caption: [FP32: forward, backward and end-to-end latency in ms.])
#figure(image("generated/flash_e2e.png", width: 100%), caption: [End-to-end latency against sequence length for $d = 64$.])

- *Forward.* The Triton kernel is faster than PyTorch attention at every size, and the gap widens with sequence length: at $d = 64$ in BF16 it is 0.23 vs 3.95 ms at $N = 8192$ (17×) and 2.65 vs 62.7 ms at $N = 32768$ (24×). The kernel does the same FLOPs but touches each score once in registers instead of making six passes over an $N times N$ tensor in memory, and the causal early exit halves the work.
- *Backward.* With the compiled PyTorch backward, FA2 is only on par with or modestly faster than PyTorch (51.5 vs 122 ms at $N = 32768$, BF16): that backward still materialises $N times N$ tensors. With the Triton backward the whole step is 0.84 vs 11.8 ms at $N = 8192$ (14×) and 9.4 vs 184 ms at $N = 32768$ (20×) in BF16, and 20.4 vs 415 ms at $N = 32768$ in FP32.
- *Memory.* PyTorch attention runs out of memory at $N = 65536$ (a single $N times N$ FP32 tensor is 16 GiB), and so does FA2 with the PyTorch backward in FP32; the fully-Triton version handles $N = 65536$ in every configuration (37 ms end to end at $d = 64$, BF16).
- *Small sizes.* Below $N approx 1024$ every variant takes a fraction of a millisecond and the numbers are dominated by Python/launch overhead (the end-to-end column includes re-creating the autograd graph), so differences there are not meaningful.

*Tile sizes.* I swept query × key tile sizes in ${16, 32, 64, 128}^2$ for the fully-Triton version at two sequence lengths:

#figure(gen("flash_tiles"), caption: [Tile-size sweep (end-to-end ms): best and worst tiling, and the tiling used by default.])

Larger tiles win for long sequences and small $d$ (fewer program instances, better use of the tensor cores), while for $d = 128$ the best tiles are smaller, since the tile × $d$ operands and the FP32 accumulators have to fit in registers/shared memory — several combinations with a 128-wide tile at $d = 128$ fail to compile with `OutOfResources`. The default rule is therefore 64 × 64 for $d <= 64$ and 32 × 32 for $d = 128$, clipped to the sequence length; at $N = 8192$ it coincides with the best tiling in FP32 for $d <= 64$ and is within 1.5× of the best in BF16 (at $N = 1024$ all tilings are within launch-overhead noise of each other).
