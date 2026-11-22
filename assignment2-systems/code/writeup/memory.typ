#import "lib.typ": gen

The xl model has 3.41 B parameters, i.e. 12.7 GiB of FP32 weights. A full training step keeps weights, gradients and two AdamW moments, $4 times 12.7 = 50.8$ GiB before a single activation is stored, so *the full-size training step cannot run on a 48 GB card at either context length* (and less than 48 GB was actually free). I therefore (i) profiled the forward pass of the full 32-layer model, and (ii) profiled complete training steps of xl models truncated to 1–8 layers and extrapolated linearly in the number of layers; the per-layer slope is exactly what the question is after.

*(a)* The "active memory timeline" below is rendered from the `_dump_snapshot` pickle by `cs336_systems/bench/memory_analysis.py` (sum of live allocations after every allocator event — the same quantity pytorch.org/memory_viz plots).

#figure(grid(columns: 2, gutter: 6pt,
  image("generated/mem_xl_ctx128_forward.png"), image("generated/mem_xl_ctx2048_forward.png"),
  image("generated/mem_xl8_ctx128_train.png"), image("generated/mem_xl2_ctx2048_train.png")),
  caption: [Active memory above the pre-existing baseline (weights, and optimizer state for training) during one step. Top: forward pass of the full xl model at context 128 and 2048. Bottom: full training step of an 8-layer xl model at context 128 and of a 2-layer xl model at context 2048.])

The stages are easy to tell apart. Inference is a sawtooth with one tooth per Transformer block (32 teeth): attention temporaries are allocated and freed inside each block and nothing accumulates, so the peak is that of a single block. In a training step the forward pass is a staircase (each block's saved activations stay alive), the backward pass descends the staircase while gradient tensors appear, and the optimizer step shows as a final burst of short-lived allocations of the size of the parameters (the reference AdamW builds new moment tensors out-of-place). At context 2048 the attention-score tensors dwarf everything else; at context 128 the picture is dominated by weight-sized tensors.

*(b)* Peak memory (`torch.cuda.max_memory_allocated`, which includes the 12.8 GiB of weights):

#figure(gen("memory_full"), caption: [Peak allocated memory of the full xl model. Training-step cells are OOM by construction (≥ 50.8 GiB needed).])

#figure(gen("memory_reduced"), caption: [Training-step peaks of xl models with $L$ layers, and the linear extrapolation to the full 32 layers.])

So a forward pass peaks at 12.9 GiB (context 128) and 21.3 GiB (context 2048) in FP32; a full training step would need about *51 GiB at context 128* (essentially just $4 times$ the weights: +1.57 GiB per layer, of which 1.56 GiB are the layer's weights, gradients and moments) and about *250 GiB at context 2048* (+7.5 GiB per layer, almost all of it saved attention matrices).

*(c)* Mixed precision does not reduce peak memory much, and for inference it _increases_ it: 19.1 GiB instead of 12.9 GiB at context 128 and 25.4 GiB instead of 21.3 GiB at context 2048. Under autocast the FP32 master weights stay in memory and every matmul additionally materialises a BF16 copy of its weight; these copies are cached for the duration of the autocast context, adding half of the weight memory (6.3 GiB). For training at context 128 the peak is unchanged (13.80 GiB for 8 layers in both precisions) because it is dominated by FP32 weights, gradients and optimizer state. Only at context 2048, where activations dominate, does BF16 help: the per-layer cost falls from 7.5 to 5.8 GiB (−23%), less than the naive 50%, because weight-sized terms do not shrink and normalisation (and whatever it feeds without an intervening matmul) stays in FP32.

*(d)* A residual-stream activation has shape (batch, context, $d_"model"$) $= (4, 512, 2560)$ at the reference context length, i.e. $4 dot 512 dot 2560 dot 4 "bytes" = 20,971,520 "B" = 20.0 "MiB"$ in single precision (5 MiB at context 128 and 80 MiB at context 2048).

*(e)* At context 2048 the largest allocations in the forward-pass snapshot are all exactly 2048 MiB $= 4 dot 32 dot 2048 dot 2048 dot 4$ bytes, the (batch, heads, seq, seq) attention tensors. Six of them are made per layer, and the stack traces point to `scaled_dot_product_attention` (`model.py:427–432`) and `softmax` (`nn_utils.py:5–7`): the `einsum` computing $bold(Q) bold(K)^top$, its division by $sqrt(d_k)$, the `torch.where` applying the causal mask, and the three steps of the softmax (subtract max, `exp`, normalise). The next size class is 320 MiB, the $(4, 2048, 10240)$ feed-forward activations. At context 128 the largest allocations are only 20 MiB (the feed-forward activations $(4, 128, 10240)$ from the `Linear` einsum and `silu`).

*(f)* Without a GUI I measured the residuals directly instead of reading them off the Nsight timeline: `cs336_systems/bench/residuals.py` wraps one `TransformerBlock` in `torch.autograd.graph.saved_tensors_hooks` and groups the saved bytes by the `grad_fn` that produced them (unique storages only, excluding views of the weights), and uses module forward/backward hooks to read `torch.cuda.memory_allocated()` after every block.

At the reference context length 512 (batch 4, FP32) one block saves *857 MiB* for the backward pass, and the allocated memory indeed grows by 857 MiB per block in the forward pass. The five largest contributors are: the two SwiGLU projections' outputs $(4, 512, 10240)$, saved together as one 160 MiB buffer (18.7%); the exponentiated attention scores from the softmax, $(4, 32, 512, 512)$, 128 MiB (14.9%); the normalised attention probabilities, 128 MiB (14.9%); the `sigmoid` output inside SiLU, 80 MiB (9.3%); and the SiLU product, 80 MiB (9.3%). At context 2048 the same block saves 6.5 GiB (allocated memory grows by 6.5 GiB per block), and the two attention matrices alone are 63% of it (2048 MiB each); there the backward pass frees 6.1 GiB per block, again leaving 400 MiB of gradients.

During the backward pass the allocated memory falls by 457 MiB per block: 857 MiB of residuals are released while the block's gradient tensors are created. The gradients therefore take $857 - 457 = 400$ MiB per block, which is exactly the size of the block's parameters (104.9 M parameters $times$ 4 bytes $= 400$ MiB) — as expected, since each parameter receives a gradient of its own shape and the activation gradients are transient.
