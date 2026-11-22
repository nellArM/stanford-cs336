#import "lib.typ": gen

*(a)* Let $R$ be the residuals of one block and $c$ one checkpoint (a single activation tensor, $c << R$). A checkpointed segment costs nothing but its input during the forward pass; in the backward pass it is recomputed with gradients, so its internals are alive only while autograd is inside it. Peak activation memory is minimised by never having more than one block "open" at a time *and* never storing more than $O(1)$ inputs, which is achieved by nesting checkpoints so that each block is recomputed from the very first input:

```python
def prefix(k, x):                       # blocks[0..k]
    if k == 0:
        return blocks[0](x)
    h = checkpoint(prefix, k - 1, x, use_reentrant=False)   # saves only x
    return blocks[k](h)                 # the only block whose residuals are alive

y = checkpoint(prefix, N - 1, x, use_reentrant=False)
```

During the backward pass the outermost checkpoint re-runs `prefix(N-1)`: the inner checkpoint produces the input of block $N-1$ without saving anything, block $N-1$ is run with gradients, back-propagated and freed, and only then is the inner checkpoint re-entered for block $N-2$, and so on. Peak activation memory is $R + O(1) dot c = O(1)$ in $N$ — one block's residuals plus a constant number of activations — at the price of $1 + 2 + ... + N = O(N^2)$ block evaluations. If compute matters, the classic compromise is recursive bisection (checkpoint the first and the second half of the current range, recursively): at most one block is open and $log_2 N$ checkpoints are alive, giving $R + c log_2 N = O(log N)$ memory for $O(N log N)$ compute. Both beat the $O(N)$ memory of plain back-propagation and the $O(sqrt(N))$ optimum of single-level checkpointing.

*(b)* With one level of checkpointing and segments of $k$ blocks, the peak is roughly $(N\/k) dot c$ for the stored segment inputs plus $k dot R$ for the segment being back-propagated. This is minimised at $k^* = sqrt(N c \/ R)$. For xl at batch 4 and context 2048, $c = 80$ MiB and $R approx 6.5$ GiB (measured above; the handout's 3.65 GiB is for a compiled block with 16 heads), so $k^* = sqrt(32 dot 0.078 \/ 6.5) approx 0.6$: the checkpoints are so cheap compared with a block's residuals that the best strategy is the smallest possible segment, *one block per checkpoint*. There is no smaller block size to compare with, so I compared $k = 1$ with the next larger sizes.

#figure(gen("checkpointing"), caption: [Peak memory of one forward+backward pass of xl (batch 4, context 2048) with one level of checkpointing. The 8-layer model is used for the comparison across block sizes because larger segments of the full model cannot fit; the 32-layer rows are the full xl model.])

The measurements confirm the hypothesis. In BF16 the peak grows from 19.0 GiB ($k = 1$) to 23.3 GiB ($k = 2$) and 31.8 GiB ($k = 4$): every extra block in a segment costs ≈ 4.3 GiB, whereas the checkpoints it saves are worth 0.08 GiB each. In FP32 the peak is 19.1, 25.0 and 37.0 GiB for $k = 1, 2, 4$ (≈ 6 GiB per extra block, matching $R$), and without checkpointing even the 8-layer model does not fit (out of memory after 43.6 GiB). With $k = 1$ the *full 32-layer xl model* completes a forward+backward pass at batch 4 and context 2048 with a peak of *37.9 GiB*, of which 25.6 GiB are weights and gradients — against roughly 220 GiB of saved activations without checkpointing. The peak is identical in FP32 and BF16, i.e. it is no longer set by saved residuals: the remaining ≈ 12 GiB above weights and gradients are the transient tensors of the single open block (six $N times N$ attention tensors of 2 GiB each exist at some point, see `memory_profiling` (e)) and of the LM head — the part that FlashAttention removes. The price is the recomputation, one extra forward pass through every block.
