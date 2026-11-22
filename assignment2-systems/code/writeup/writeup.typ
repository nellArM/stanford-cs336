#set document(title: "CS336 Assignment 2 (Systems): Writeup")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.6em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)
#show heading.where(level: 2): it => block(above: 1.1em, below: 0.6em, it)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure: set block(breakable: true)
#show table: set text(size: 8pt)
#show raw: set text(size: 8.5pt)
#import "lib.typ": gen
#let note(body) = block(fill: luma(240), inset: 7pt, radius: 3pt, width: 100%, text(size: 9pt, body))

#align(center)[
  #text(size: 16pt, weight: "bold")[CS336 Assignment 2 (Systems): Writeup] \
  #v(2pt)
  #text(size: 10pt)[Stanford CS336, Spring 2026 handout (v26.1.3) · completed as a self-study exercise]
]

#v(4pt)
#note[
  *How to read the numbers in this report.* The handout assumes a dedicated Nvidia B200 (≈180 GB) node. Everything here was measured on *RTX A6000 cards (48 GB, Ampere) that were shared with other running jobs* — other processes held between 3 and 45 GB of each card and kept it at 100% utilisation for the whole measurement period. Consequences, which apply to every table:

  - *Absolute timings are several times slower than an idle A6000 and are noisy*; ratios measured back-to-back within one table are the meaningful quantity. Occasional large standard deviations are another job starting or stopping, not a property of the code.
  - *Out-of-memory limits are lower than 48 GB.* Where a configuration cannot fit on a 48 GB card _in principle_ it is marked OOM; where a run only failed because of the neighbours, the runner waited and retried (the last such cells were re-measured on a nearly free card), so every OOM in the tables is a limit of a 48 GB card itself; the substitutes that were run instead are stated per section.
  - Only two GPUs (physical GPU 0 and 1) were available, so all multi-GPU experiments use 2 processes; the "xl on 2 GPUs" items for DDP and optimizer-state sharding use the *large* and *medium* models because xl cannot be trained unsharded on these cards, while FSDP was run on the full xl model (stated per section).
  - There is no NVLink and direct PCIe peer-to-peer transfers hang on this machine, so NCCL was run with `NCCL_P2P_DISABLE=1` (data goes GPU → host shared memory → GPU). Communication is therefore much slower relative to compute than on the reference hardware, which exaggerates every communication effect.
  - The machine has no display or browser: Nsight Systems captures were taken with the `nsys` CLI (2025.3.1) and analysed from the exported SQLite database, and the memory-viz / Nsight "screenshots" are replaced by plots rendered from the same underlying data.

  All tables and figures are generated from the CSV files in `experiments/results/` by `writeup/make_tables.py`; the scripts that produced them are in `cs336_systems/bench/` and `experiments/*.sh`.
]

= Profiling and benchmarking

== Problem `benchmarking_script`

*(a)* `cs336_systems/bench/benchmark.py` builds a `BasicsTransformerLM` from a Table-1 size (optionally overriding the number of layers), creates a random batch, runs $w$ warm-up steps and then times $n$ steps with `timeit.default_timer`, calling `torch.cuda.synchronize()` after the forward pass, after the backward pass and after the optimizer step so that each phase is timed separately. `--mode` selects forward-only (under `no_grad`), forward+backward, or the full training step with AdamW; `--bf16`, `--compile`, `--nvtx` and `--memory-snapshot` add mixed precision, `torch.compile`, NVTX ranges and the PyTorch memory profiler.

*(b)* Vocabulary 10,000, batch 4, context 512, FP32, 5 warm-up and 10 measured steps:

#figure(gen("e2e_fp32"), caption: [FP32 timings (mean ± std over 10 steps). "Forward only" is an inference pass under `no_grad`; the other columns are the three phases of a training step.]) <tab-e2e>

A forward pass takes about 0.15 s (small), 0.45 s (medium), 0.94 s (large) and 2.6 s (xl); the backward pass costs almost exactly twice the forward pass (0.33 s, ≈0.9 s, 1.9 s) and the AdamW step adds a further 0.09–0.66 s. When no other job changed state during the run the standard deviation is tiny — below 1% of the mean (e.g. 154.1 ± 0.5 ms) — so 10 steps are plenty; the one row with a large deviation (medium, training) coincided with a neighbouring job starting. The xl model (3.4 B parameters, 12.7 GiB of FP32 weights) can run inference but not a training step: weights + gradients + two AdamW moments alone are 4 × 12.7 = 51 GiB. The 10B configuration (12.7 B parameters = 47 GiB in FP32) does not fit at all.

*(c)* Warm-up:

#figure(gen("e2e_warmup"), caption: [Effect of the number of warm-up steps (training step, FP32).])

Without warm-up the mean forward time rises by 6–18% and, more tellingly, its standard deviation jumps from ≈1 ms to ≈90 ms: the first measured step includes one-off costs — CUDA context and cuDNN/cuBLAS handle creation, the allocator growing its pool for the first time (every later step reuses cached blocks), lazy initialisation of kernels and, for AdamW, the allocation of the optimizer state. One or two warm-up steps remove almost all of it (std back to 1.5–3 ms), but the mean can still differ slightly from the 5-step value because the caching allocator and the GPU's clock/power state take a few iterations to settle, and with so few steps a single disturbed step is not averaged away (the small/2-warm-up row shows exactly such a disturbance from a co-tenant).

== Problem `nsys_profile`

I profiled the *small* and *large* models at context lengths 256, 512 and 1024 (the training step of the large model at context 1024 does not fit on a 48 GB card — it ran out of memory twice with 46 GiB free — and is marked OOM), forward-only and full training step, with `nsys profile --trace=cuda,nvtx`. NVTX ranges mark warm-up steps, measured steps, the three phases, and the three parts of attention; `cs336_systems/bench/nsys_analysis.py` attributes each kernel to the NVTX range containing the CUDA API call that launched it (joined on the correlation id), considering measured steps only — the offline equivalent of the GUI's "CUDA GPU Kernel Summary" with an NVTX filter.

#figure(gen("nsys"), caption: [Nsight Systems summary per measured step. "matmul share" is the fraction of GPU kernel time spent in GEMM kernels (`ampere_sgemm_*` and `cutlass::Kernel2<…sgemm…>`).]) <tab-nsys>

#figure(gen("nsys_top"), caption: [Kernel with the largest cumulative GPU time in the forward pass (and in the backward pass for training runs).])

*(a)* The summed GPU kernel time of a forward pass agrees with the `timeit` measurement to within 2–10% (e.g. 148 vs 159 ms for small/512, 920 vs 944 ms for large/512). The kernel sum is always slightly smaller because the Python-side time also contains kernel-launch overhead and gaps between kernels; the gap is relatively largest for the smallest configuration, where the GPU is not saturated.

*(b)* The forward pass is dominated by a single-precision GEMM kernel, `ampere_sgemm_128x64_tn`, in every configuration. It is the kernel behind the `Linear` layers: 7 per block (Q, K, V, output projection and the three SwiGLU matrices) plus the LM head, i.e. exactly $7 dot 12 + 1 = 85$ invocations per forward pass of the small model and $7 dot 36 + 1 = 253$ for the large model at context 256 (at longer contexts cuBLAS moves part of these calls to the `128x128_tn` tile variant, so the count of the top kernel drops to 73 and 217); the two attention matmuls per block use separate `_nn`/`_tn` variants. It is _not_ the top kernel of the backward pass: there the leader is usually `Kernel2`, which is cuBLAS's `cutlass::Kernel2<cutlass_80_simt_sgemm_…>` — still a matrix multiply, but the variant chosen for the transposed products $bold(x)^top bold(d y)$ that produce the weight gradients — and for the small model at context 1024 it is the element-wise kernel, i.e. not a matmul at all.

*(c)* Besides GEMM, the forward pass spends 30–50% of its GPU time in PyTorch's `elementwise_kernel` / `vectorized_elementwise_kernel` (the adds, multiplies, divisions, `exp`, `where` and dtype casts of RMSNorm, SiLU, RoPE, the residual additions, the causal masking and the softmax) and a few percent in `reduce_kernel` (the `max`/`sum` of softmax and the mean of RMSNorm). Their share grows with context length because the attention-score tensors grow quadratically.

*(d)* In a full training step the matmul share drops from 59–76% (inference) to 41–50%. The backward pass doubles the GEMM work but adds proportionally more element-wise work (every element-wise forward op has an element-wise backward, plus gradient accumulation), and the optimizer step is 100% element-wise kernels — the reference AdamW issues about 5,200 small element-wise kernels per step for the large model and takes as long as two-thirds of a forward pass.

*(e)* Inside attention, the two matmuls ($bold(Q) bold(K)^top$ and $bold(P) bold(V)$) cost about $4 N^2 d$ FLOPs per head against roughly $5 N^2$ for the softmax (max, subtract, exp, sum, divide): with $d = 64$ the matmuls have ≈50× more FLOPs. Yet the softmax takes _longer_ than both matmuls together as soon as the context reaches 512 (21.6 vs 8.8 ms at 512 and 93 vs 34 ms at 1024 for the small model; 418 vs 213 ms for large/1024); only at context 256 are the matmuls slower. FLOPs are a poor predictor here: GEMM is compute-bound and runs at tens of TFLOP/s, while softmax is memory-bound — five separate kernels each read and write an entire $N times N$ tensor per head. This is the inefficiency FlashAttention removes by never writing the $N times N$ tensors to memory.

== Problem `mixed_precision_accumulation`

The four loops print `10.0001` (FP32 accumulator, FP32 addend), `9.9531` (FP16, FP16), `10.0021` (FP32 accumulator, FP16 addend) and `10.0021` (FP32 accumulator, FP16 addend cast to FP32). Accumulating in FP16 is badly wrong (−0.47%): once the running sum exceeds 8, the spacing between representable FP16 numbers is $2^(-7) approx 0.008$, the same size as the addend, so each addition is rounded to a multiple of that spacing and the errors accumulate. With an FP32 accumulator the only error left is the representation error of the addend itself (0.01 in FP16 is 0.010002136, giving 10.0021 after 1000 additions; whether the cast is implicit or explicit makes no difference). This is why autocast keeps reductions and accumulations in FP32 even when the operands are half precision.

== Problem `benchmarking_mixed_precision`

*(a)* Verified by running the model on the GPU under `torch.autocast("cuda", dtype=torch.float16)` (`experiments/results/autocast_dtypes.out`):

#figure(table(columns: 2, align: (left, left), stroke: none, inset: (x: 6pt, y: 3pt),
  table.hline(), table.header([*quantity*], [*dtype*]), table.hline(stroke: 0.5pt),
  [model parameters inside the autocast context], [`float32` (autocast never changes the stored parameters)],
  [output of `fc1`], [`float16` (linear layers are autocast to half precision)],
  [output of the layer norm], [`float32` (layer norm is on autocast's FP32 list)],
  [predicted logits (`fc2` output)], [`float16`],
  [loss (cross-entropy computed inside the context)], [`float32`],
  [gradients], [`float32` (same dtype as the parameters they belong to)],
  table.hline()), caption: [Data types under FP16 autocast.])

*(b)* Layer normalisation computes a mean and a variance (a sum of squares) over the feature dimension and then divides by $sqrt("var" + epsilon)$. In FP16 the squares overflow for activations above ≈255 (max 65,504), the accumulation suffers from exactly the rounding problem of the previous question, and a small $epsilon$ (1e-5) is at the edge of FP16's resolution, so the division becomes inaccurate for small variances. BF16 has the same exponent range as FP32, so overflow and the $epsilon$ problem disappear, but it has only 8 bits of mantissa (fewer than FP16's 11), so the reductions and the subtraction of the mean are even less precise. Autocast therefore still runs layer norm in FP32 with BF16 (the experiment above prints `float32` for the layer-norm output under BF16 as well) — and since normalisation is a tiny fraction of the compute, nothing is lost by doing so. The course's RMSNorm does the same by explicitly up-casting its input.

*(c)* The same benchmark under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`:

#figure(gen("e2e_bf16"), caption: [BF16 mixed-precision timings; compare with @tab-e2e.])

Mixed precision speeds up the forward pass by 2.0–2.9× (154 → 59 ms for small, 469 → 235 ms for medium, 940 → 449 ms for large, 2645 → 916 ms for xl inference) and the backward pass by 1.7–1.9× (326 → 188 ms, 1877 → 984 ms). The speed-up grows with model size because a larger share of the time is spent in GEMMs, which are the operations that move to tensor cores in half precision; the element-wise and normalisation work, and all of the optimizer step (which acts on the FP32 master weights; 87 ms and 659 ms in both tables), do not get faster. For the same reason the end-to-end gain of a training step shrinks to 1.6–1.7×: the optimizer becomes the bottleneck (31% of a large-model step in BF16).

== Problem `memory_profiling`

#include "memory.typ"

= Single-GPU memory

== Problem `gradient_checkpointing`

#include "checkpointing.typ"

= GPU kernels

#include "kernels.typ"

= Distributed data parallel training

#include "distributed.typ"

#include "parallelism.typ"

= Leaderboard

#include "leaderboard.typ"
