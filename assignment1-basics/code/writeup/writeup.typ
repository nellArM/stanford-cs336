#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)
#show raw.where(block: false): set text(size: 9pt)
#show figure.caption: set text(size: 9pt)
#set table(inset: 5pt, stroke: 0.4pt + luma(150))
#show table: set text(size: 9pt)

#let s = json("figures/summary.json")
#let acc = json("accounting.json")
#let f3(x) = if x == none { "–" } else { str(calc.round(x, digits: 3)) }
#let sample(path) = block(
  inset: 8pt, fill: luma(245), radius: 3pt, width: 100%,
  text(size: 9pt, read(path).trim()),
)
#let prob(id) = text(weight: "bold")[Problem (#raw(id)).]
#let pct(x) = str(calc.round(100 * x, digits: 1)) + "%"
#let sci(x) = {
  let e = calc.floor(calc.log(x, base: 10))
  [#calc.round(x / calc.pow(10.0, e), digits: 2)$times 10^#e$]
}

#align(center)[
  #text(size: 17pt, weight: "bold")[CS336 Assignment 1 (Basics): Building a Transformer LM] \
  #v(0.2em)
  #text(size: 11pt)[Written answers and experiment report — self-study, Spring 2026 handout]
]

#block(inset: 9pt, stroke: 0.5pt + luma(120), radius: 3pt, width: 100%)[
  *Scope and deviations from the handout.* All code and all written questions are done in full
  (`uv run pytest`: 47 passed, 1 expected failure). Experiments were run on one NVIDIA RTX A6000
  (48 GB) that was *shared with other jobs*, not on a B200, so every wall-clock number is slower
  and noisier than the handout's reference. To fit the budget:
  - the learning-rate sweep, batch-size sweep and all ablations use the handout's own
    _low-resource_ budget of 40M tokens (1,220 steps at batch 128, context 256);
  - one TinyStories run and one OpenWebText run use the full 327.68M-token budget;
  - the "leaderboard" run is 45 minutes on the A6000 and was *not* submitted anywhere;
  - no hosted tracker was used: every run writes `config.json`, `metrics.jsonl` and `final.json`
    locally, and all figures and tables here are regenerated from those files.
  Training uses bf16 autocast and `torch.compile`. Losses are natural-log per-token cross-entropy.
]

= Tokenization

#prob("unicode1")
(a) `chr(0)` returns the NULL character, U+0000.
(b) Its `repr` is the escape sequence `'\x00'`, whereas printing it produces no visible output.
(c) It is an ordinary character of the string: `"this is a test" + chr(0) + "string"` has it in the
middle (shown as `\x00` in the `repr`), and `print` emits it invisibly, so the text looks like
`this is a teststring`; it does not terminate the string as it would in C.

#prob("unicode2")
(a) UTF-8 is a byte-oriented, ASCII-compatible, variable-length code: mostly-English text takes one
byte per character, while UTF-16/UTF-32 spend 2 or 4 bytes on it and fill the sequence with zero
bytes (and need a byte order), so UTF-8 gives shorter sequences and a 256-symbol base vocabulary
whose symbols are all used.
(b) The function decodes each byte on its own, but a character outside ASCII is encoded as several
bytes that are only valid together. Example: `"こ".encode("utf-8") = b'\xe3\x81\x93'` raises
`UnicodeDecodeError`, because `b'\xe3'` alone is not a complete UTF-8 sequence.
(c) `b'\xc3\x28'`: `0xC3` announces a two-byte character and must be followed by a continuation
byte of the form `10xxxxxx`, but `0x28` is `00101000`.

#prob("train_bpe_tinystories")
(a) Training the 10,000-entry vocabulary on the 2.2 GB TinyStories training file took *19.6 s*
(14.1 s parallel pre-tokenization on 48 processes, 5.5 s for the 9,743 merges) with a peak
resident memory of 0.11 GB in the main process and 0.32 GB in the largest worker. The longest
token is `" accomplishment"` (15 bytes), which makes sense: it is a frequent whole word, with its
leading space, of a corpus written in a small children's vocabulary.
(b) Profiling with `cProfile` shows that pre-tokenization (the regex scan and counting, seen in the
main process as time waiting for the worker pool) takes about 72% of the time; the merge loop is
cheap because pair counts are updated incrementally and the best pair is taken from a heap with
lazy invalidation. On OpenWebText the balance reverses (below), because there are 110 times more
distinct pre-tokens to update.

#prob("train_bpe_expts_owt")
(a) Training the 32,000-entry vocabulary on the 11.9 GB OpenWebText training file took 23.6 min
(69 s pre-tokenization, 1,348 s merging over 6.6M distinct pre-tokens) and 10.2 GB of memory. The
longest tokens are 64 bytes long: a run of the two characters `Ã` and `Â` repeated 16 times, and a
run of 64 hyphens. Both make sense as artefacts of web text rather than as language: the first is
mojibake (UTF-8 text that was decoded as Latin-1 and encoded again, several times) and the second
is a horizontal rule; both are long strings that recur verbatim often enough to be merged.
(b) The TinyStories vocabulary consists almost entirely of whole, common English words and word
pieces from a small, clean vocabulary. The OpenWebText vocabulary is three times larger and much
more varied: names, numbers, URL and markup fragments, non-English and non-ASCII sequences and
encoding noise, reflecting a broader and dirtier distribution.

#prob("tokenizer_experiments")
(a) On 10 sampled documents each, the TinyStories tokenizer compresses TinyStories to
*#f3(3.928) bytes/token* and the OpenWebText tokenizer compresses OpenWebText to
*#f3(4.382) bytes/token* (over the full training files: 4.12 and 4.37).
(b) Encoding the OpenWebText sample with the TinyStories tokenizer still works, since every byte is
in the vocabulary, but the ratio drops to 3.26 bytes/token: words, names and symbols that never
occur in children's stories are broken into many short pieces. (In the other direction the
OpenWebText tokenizer gets 3.81 bytes/token on TinyStories, slightly below the matched tokenizer.)
(c) A single process encodes about 5.2 MB/s with the TinyStories tokenizer and 4.0 MB/s with the
OpenWebText one (1.25M and 0.91M tokens/s), so the 825 GB Pile would take about 44–58 hours on one
core, or roughly 1–1.5 hours with the 48 processes used here.
(d) Both vocabularies (10,000 and 32,000 ids) fit in 16 bits ($2^16 = 65,536$), so `uint16`
represents every id exactly while using half the space of `int32` and a quarter of `int64`; the
2.73-billion-token OpenWebText training set is 5.5 GB instead of 21.8 GB. The encoded sets are
541,229,347 / 5,465,883 tokens (TinyStories train / valid) and 2,727,120,758 / 66,401,105 tokens
(OpenWebText train / valid).

= Resource accounting

Notation: vocabulary $V$, context length $T$, layers $L$, width $d$, heads $h$, feed-forward width
$d_"ff"$, batch size $B$. A product of an $m times n$ and an $n times p$ matrix costs $2 m n p$ FLOPs.

#prob("transformer_accounting")
(a) The model has $2 V d + L (4 d^2 + 3 d d_"ff" + 2 d) + d =$ *1,640,452,800* trainable parameters
(token embedding and output embedding $V d$ each; per block four attention projections, three
feed-forward matrices and two RMSNorm gains; one final gain). In single precision that is
*6.56 GB* (6.11 GiB).

(b) Matrix multiplies for one sequence of $T = 1024$ tokens (the embedding lookup and RoPE
rotations are not matrix multiplies):

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, right, right),
  table.header([*Operation*], [*Shapes*], [*FLOPs each*], [*Count*]),
  [Q, K, V projections], [$T times d$ by $d times d$], [$2 T d^2 = 5.24 times 10^9$], [$3 L$],
  [Attention scores $Q K^top$], [$h$ heads: $T times d/h$ by $d/h times T$], [$2 T^2 d = 3.36 times 10^9$], [$L$],
  [Weighted sum of values], [$h$ heads: $T times T$ by $T times d/h$], [$2 T^2 d = 3.36 times 10^9$], [$L$],
  [Attention output projection], [$T times d$ by $d times d$], [$2 T d^2 = 5.24 times 10^9$], [$L$],
  [SwiGLU $W_1$, $W_3$], [$T times d$ by $d times d_"ff"$], [$2 T d d_"ff" = 1.41 times 10^10$], [$2 L$],
  [SwiGLU $W_2$], [$T times d_"ff"$ by $d_"ff" times d$], [$2 T d d_"ff" = 1.41 times 10^10$], [$L$],
  [Output embedding], [$T times d$ by $d times V$], [$2 T d V = 1.65 times 10^11$], [$1$],
)

In total $L (8 T d^2 + 4 T^2 d + 6 T d d_"ff") + 2 T d V =$ *$3.52 times 10^12$ FLOPs* per forward pass.

(c) The feed-forward networks dominate (57.5%), followed by the attention projections (28.6%);
the $T^2$ attention products are only 9.2% and the output embedding 4.7% at this context length.

(d) Shares of the forward-pass FLOPs (context 1,024, $d_"ff"$ the multiple of 64 nearest $8d\/3$):

#table(
  columns: 7,
  align: (left, right, right, right, right, right, right),
  table.header([*Model*], [*Params*], [*Total FLOPs*], [*Attn. proj.*], [*Attn. $T^2$*], [*FFN*], [*Output emb.*]),
  ..("small", "medium", "large", "xl").map(k => (
    [GPT-2 #k], [#calc.round(acc.at(k).params / 1e6, digits: 1)M], sci(acc.at(k).flops.total),
    pct(acc.at(k).share.at("attn_proj(QKVO)")), pct(acc.at(k).share.at("attn_matmul(QK^T, AV)")),
    pct(acc.at(k).share.ffn), pct(acc.at(k).share.lm_head),
  )).flatten(),
)

As the model grows, the terms that scale with $L d^2$ (feed-forward and attention projections) take
a growing share, from 60% to 86%, while the output embedding ($prop d$, independent of depth) falls
from 27% to 5% and the $T^2$ attention products ($prop L d$) fall from 13% to 9%.

(e) With a context of 16,384 the forward pass costs *$1.34 times 10^14$ FLOPs*, 38 times more for
16 times more tokens: the terms linear in $T$ grow 16-fold but the attention products grow 256-fold
and now account for 61.7% of the total (FFN 24.2%, attention projections 12.1%, output embedding 2.0%).

#prob("learning_rate_tuning")
With the toy loss and the $alpha \/ sqrt(t+1)$ step, a learning rate of 1 decays slowly (26.3 to
21.7 after 10 steps), 10 decays faster (to 3.5), 100 decays fastest (the first step exactly
reflects the weights, then the loss collapses to about $10^(-23)$), and 1000 diverges, growing by
orders of magnitude every step (to $2.4 times 10^18$).

#prob("adamw_accounting")
(a) Counting float32 values, with $d_"ff" = 8d\/3$:
- *Parameters:* $P = 2 V d + L (12 d^2 + 2 d) + d$.
- *Activations* (per sequence, the listed tensors only): each block stores two RMSNorm outputs
  ($2 T d$), Q, K, V ($3 T d$), attention scores and their softmax ($2 h T^2$), the weighted values
  and the output projection ($2 T d$), and in the feed-forward network $W_1 x$, its SiLU, $W_3 x$
  and their product ($4 T d_"ff"$) and $W_2$'s output ($T d$). With the final RMSNorm ($T d$), the
  logits ($T V$) and the cross-entropy's log-probabilities ($T V$):
  $A = L (56/3 T d + 2 h T^2) + T d + 2 T V$ per sequence, i.e. $B A$ in total.
- *Gradients:* $P$ (one per parameter; activation gradients are transient and at most the size of
  the activations, so they are not counted separately).
- *Optimizer state:* $2 P$ (first and second moments).
- *Total:* $4 (4 P + B A)$ bytes.

(b) For GPT-2 XL, $P = 1.636 times 10^9$ and $A = 4.089 times 10^9$, giving
*$16.36 dot B + 26.17$ GB*. The largest batch that fits in 80 GB is *3*
($(80 - 26.17) \/ 16.36 = 3.29$).

(c) The optimizer update itself is element-wise: about 14 FLOPs per parameter (3 for the first
moment, 4 for the second, 5 for the bias-corrected step $alpha_t m \/ (sqrt(v) + epsilon)$, 2 for
the decoupled weight decay), i.e. $approx 14 P$, which is negligible. A whole training step is
dominated by the forward and backward passes; taking the backward pass as twice the forward pass,
$ "FLOPs per step" approx 3 B [L (24 T d^2 + 4 T^2 d) + 2 T d V] + 14 P. $

(d) One step at batch 1,024 costs $3 times 1024 times 3.52 times 10^12 = 1.08 times 10^16$ FLOPs,
so 400K steps cost $4.32 times 10^21$ FLOPs. At 50% of 495 TFLOP/s this takes $1.75 times 10^7$ s,
about *4,850 hours (202 days)* on a single H100.

= Training experiments

== Setup and experiment log

#prob("experiment_log")
`cs336_basics/train.py` takes every hyperparameter as a flag and writes three files per run:
`config.json` (all flags, step counts, parameter counts), `metrics.jsonl` (one record per logging
step with the gradient step, tokens processed, *training wall-clock seconds*, training loss,
pre-clipping gradient norm and learning rate, plus the validation loss on evaluation steps) and
`final.json` (status and the final validation loss over 200 batches). Wall-clock time counts
training steps only; evaluation time is excluded. Validation during training uses the same 20
random batches at every evaluation. `experiments/run_all.py` and `experiments/run_extra.py` are
restart-safe queues that define every run in this report, and `experiments/make_figures.py`
regenerates all figures and the table below from the logs.

Base model (handout): vocabulary 10,000, context 256, $d = 512$, $d_"ff" = 1344$, 4 layers,
16 heads, RoPE with $Theta = 10000$; 17.6M non-embedding parameters (22.7M in total). Optimizer:
AdamW with $beta = (0.9, 0.95)$, $epsilon = 10^(-8)$, weight decay 0.1 on matrices only, gradient
clipping at norm 1, linear warm-up over 3% of the steps and cosine decay to 5% of the peak rate,
reached exactly at the last step.

What was tried, in order:
+ *Smoke tests and a numerical bug.* The first compiled bf16 run produced non-finite gradients
  while the eager run was fine. The cause was the max-subtraction in my `softmax`/`log_softmax`:
  the gradient flowing through the `amax` over masked ($-infinity$) attention scores became
  NaN in the compiled graph. Since softmax is shift-invariant, the shift is now detached; compiled
  and eager gradients agree.
+ *Learning-rate sweep* at 40M tokens: seven rates, half a decade apart, from $10^(-4)$ to
  $10^(-1)$ (@lr).
+ *Full-budget TinyStories run* at the best rate (@fig-full).
+ *Batch-size sweep* at 40M tokens (@bs).
+ *Ablations* at 40M tokens against the best run of the sweep (@abl).
+ *OpenWebText* at the full budget with the same architecture (@owt), and two 45-minute runs (@lb).

#figure(
  table(
    columns: 8,
    align: (left, right, right, right, right, right, right, left),
    table.header([*Run*], [*Batch*], [*Peak LR*], [*Steps*], [*Train min*], [*Final val*], [*Best val*], [*Status*]),
    ..s.pairs().filter(p => p.at(1).lr != none).map(p => {
      let (name, r) = p
      (raw(name), [#r.batch_size], [#r.lr], [#r.steps], [#r.train_minutes],
       f3(r.final_val_loss), f3(r.best_val_loss), [#r.status.replace("_", " ")])
    }).flatten(),
  ),
  caption: [Every training run. "Final val" is the mean over 200 validation batches after training;
    "best val" is the lowest 20-batch estimate logged during training. Train minutes exclude evaluation
    and are affected by other jobs on the GPU.],
) <runs>

== Learning rate <lr>

#prob("learning_rate")
(a) *Search strategy.* Because the loss is sensitive to the order of magnitude of the rate, I swept
a logarithmic grid with two points per decade from $10^(-4)$ to $10^(-1)$, deliberately extending
it until runs diverged, at the 40M-token budget with everything else fixed, and then spent the full
budget only on the winner. Final validation losses: $10^(-4)$: #f3(s.at("lr_1e-4").final_val_loss),
$3 times 10^(-4)$: #f3(s.at("lr_3e-4").final_val_loss), $10^(-3)$: #f3(s.at("lr_1e-3").final_val_loss),
$3 times 10^(-3)$: *#f3(s.at("lr_3e-3").final_val_loss)*; $10^(-2)$ diverged at step 220,
$3 times 10^(-2)$ at step 60 and $10^(-1)$ at step 20 (@fig-lr).

#figure(image("figures/lr_sweep.png", width: 100%),
  caption: [Learning-rate sweep on TinyStories, 40M tokens. Runs marked "diverged" stopped at a non-finite loss.]) <fig-lr>

#figure(image("figures/ts_full.png", width: 100%),
  caption: [Full-budget TinyStories run (10,000 steps, batch 128, peak rate $3 times 10^(-3)$) against gradient steps and against training wall-clock time.]) <fig-full>

*Model with validation loss at most 1.45.* The full run (327.68M tokens, 10,000 steps, 43.5
minutes of training on the shared A6000) reaches a final validation loss of
*#f3(s.at("ts_full").final_val_loss)* (perplexity 3.78), below the 1.45 target (@fig-full). The
40M-token run at the same rate reaches #f3(s.at("lr_3e-3").final_val_loss), better than the
handout's low-resource reference of 1.80.

(b) *Edge of stability.* Within the stable range the final loss improves monotonically as the rate
grows, and the best rate ($3 times 10^(-3)$) is the largest one that does not diverge: the next
grid point, only half a decade higher, blows up. The way it fails is informative: at $10^(-2)$ the
loss first falls _faster_ than any stable run (3.84 at step 100, when the best run is still above
that), then the pre-clipping gradient norm jumps from below 1 to $10^4$ and beyond around step 140
and the loss becomes non-finite at step 220, shortly after warm-up ends and the rate is at its
peak. Higher rates fail the same way, only earlier (@fig-lrloss). So convergence speed increases
with the rate right up to the point where the curvature of the loss can no longer support the step
size; the best rate sits just below that boundary (here within a factor of about 3), which is the
"edge of stability" folk wisdom. Warm-up and gradient clipping push the boundary outwards but do
not remove it.

#figure(image("figures/lr_vs_loss.png", width: 52%),
  caption: [Final validation loss against learning rate; dashed lines mark diverged runs.]) <fig-lrloss>

== Batch size <bs>

#prob("batch_size_experiment")
All runs process the same 40M tokens, so the number of updates is inversely proportional to the
batch size; the peak rate is scaled as $3 times 10^(-3) sqrt(B \/ 128)$ (@fig-bs). Batch size 1
would need 156,160 steps, so it was capped at 15 minutes of training instead (50,500 steps, 12.9M
tokens).

#figure(image("figures/batch_size.png", width: 100%),
  caption: [Batch-size sweep on TinyStories at a fixed budget of 40M tokens, against tokens processed (left) and training wall-clock time (right). Batch 1 is capped at 15 minutes.]) <fig-bs>

#table(
  columns: 9,
  align: (left, right, right, right, right, right, right, right, right),
  table.header([*Batch*], [1], [16], [64], [128], [256], [512], [512], [1024]),
  [Peak rate], [$2.65 times 10^(-4)$], [$1.06 times 10^(-3)$], [$2.12 times 10^(-3)$], [$3 times 10^(-3)$], [$4.24 times 10^(-3)$], [$6 times 10^(-3)$], [$3 times 10^(-3)$], [–],
  [Updates], [50,500], [9,760], [2,440], [1,220], [610], [305], [305], [–],
  [Final val loss], [#f3(s.at("bs_1").final_val_loss)], [#f3(s.at("bs_16").final_val_loss)], [*#f3(s.at("bs_64").final_val_loss)*], [#f3(s.at("lr_3e-3").final_val_loss)], [#f3(s.at("bs_256").final_val_loss)], [diverged], [#f3(s.at("bsretune_512").final_val_loss)], [out of memory],
)

Findings:
- *At a fixed token budget, small-to-medium batches win.* Batches 16–128 end within 0.05 of each
  other (64 is best at 1.64), while 256 and 512 are clearly worse (1.82 and 2.07): they simply make
  too few updates (610 and 305), and a larger batch cannot compensate with a proportionally larger
  step because of the stability limit found in @lr. Indeed, batch 512 at the square-root-scaled
  rate $6 times 10^(-3)$ diverged after about 100 steps, and had to be re-run at the base rate.
- *Very small batches waste the hardware.* Batch 1 makes the most updates per token but each one is
  noisy and the GPU is almost idle: in 15 minutes it processed a third of the token budget and
  reached only 2.26, whereas batches 64–128 finish all 40M tokens in 5–6 minutes. Per unit of
  wall-clock time the medium batches are the best choice on this model; larger batches give higher
  throughput but, at this small budget, not enough updates.
- *The memory limit* of the 48 GB card is between 512 and 1,024: batch 1,024 fails with an
  out-of-memory error (the float32 logits alone are $1024 times 256 times 10000 times 4$ bytes
  $= 10.5$ GB, and several tensors of that size are alive in the loss and its gradient).
- With a larger token budget the picture would shift towards larger batches, since the number of
  updates would no longer be the binding constraint; the rates here were scaled by a rule and only
  re-tuned for batch 512, so differences of a few hundredths between neighbouring batch sizes
  should not be over-interpreted.

== Generated text

#prob("generate")
Sample from the full-budget TinyStories model, prompt "Once upon a time", temperature 0.8, top-p 0.9
(the model ended the story itself with `<|endoftext|>` after 156 new tokens):

#sample("samples/ts_full_s0.md")

The text is fluent, grammatical and locally coherent, with a clear beginning, problem and ending
and consistent characters; its weaknesses are mild repetition and thin plot logic (the bug that
"did not come" then "jumped out"). Two factors that determine the quality:
- *Decoding parameters.* Lower temperature and nucleus truncation remove the unreliable tail of the
  distribution. With temperature 1.3 and no top-p truncation the same checkpoint produces word
  salad:
  #sample("samples/ts_full_hot_s0.md")
  Very low temperatures err the other way and give repetitive, generic stories.
- *How well the model is trained.* Validation loss tracks fluency: the 40M-token models (loss
  about 1.7–1.8) make more agreement and coherence errors than the 1.33 model. Model size and the
  256-token context also bound how long a story can stay consistent.
A third factor is the data: TinyStories is narrow and simple, so a small model can cover it well
(compare OpenWebText in @owt).

== Ablations <abl>

All ablations use the 40M-token budget and are compared with the best run of the sweep
(pre-norm, RoPE, SwiGLU, rate $3 times 10^(-3)$, final validation loss
#f3(s.at("lr_3e-3").final_val_loss)). Each is a single seed, so differences of about 0.01–0.02
are within noise.

#prob("layer_norm_ablation")
Without any RMSNorm the model *diverges at the previously optimal rate*: the loss falls to 3.4 by
step 60, the gradient norm then explodes and the loss is non-finite by step 100
(@fig-norm). Lower rates restore stability: $10^(-3)$ gives
#f3(s.at("abl_nonorm_third").final_val_loss), $3 times 10^(-4)$ gives
#f3(s.at("abl_nonorm_tenth").final_val_loss) and $10^(-4)$ gives
#f3(s.at("abl_nonorm_thirtieth").final_val_loss). The best normalisation-free model (1.79) is about
as good as the normalised model at the _same_ rate $10^(-3)$ (#f3(s.at("lr_1e-3").final_val_loss)),
so in this shallow 4-layer network RMSNorm does not help much at a fixed safe rate. Its value is
that it keeps activation scales under control so that a three-times larger rate can be used, and
that larger rate is what buys the better result (1.66) in the same number of steps.

#figure(image("figures/abl_norm.png", width: 100%),
  caption: [Removing RMSNorm. At the best rate of the normalised model training diverges; smaller rates are stable but slower.]) <fig-norm>

#prob("pre_norm_ablation")
The post-norm model trains stably at the same rate and ends at
#f3(s.at("abl_postnorm").final_val_loss) against #f3(s.at("lr_3e-3").final_val_loss) for pre-norm
(@fig-post): slightly worse, with a visibly slower start, but the gap is small. With only four
layers the residual stream is short, so the well-known advantage of pre-norm (an identity path from
input to output that keeps gradients well-scaled in deep stacks) is barely exercised here.

#figure(image("figures/abl_postnorm.png", width: 100%),
  caption: [Post-norm against pre-norm.]) <fig-post>

#prob("no_pos_emb")
Without any position information the model still learns, as a causal mask lets it infer position
implicitly, but it is clearly worse: #f3(s.at("abl_nope").final_val_loss) against
#f3(s.at("lr_3e-3").final_val_loss) with RoPE, and the gap is present throughout training
(@fig-nope). This is the largest effect among the three architecture ablations that keep training
stable.

#figure(image("figures/abl_nope.png", width: 100%),
  caption: [No position embedding (NoPE) against RoPE.]) <fig-nope>

#prob("swiglu_ablation")
The ungated SiLU feed-forward network with $d_"ff" = 4 d = 2048$ (17.71M non-embedding parameters
against 17.58M for SwiGLU with $d_"ff" = 1344$) ends at #f3(s.at("abl_silu").final_val_loss)
against #f3(s.at("lr_3e-3").final_val_loss) for SwiGLU (@fig-silu). SwiGLU is ahead for the whole
run, but by only about 0.01 at the end, which a single seed cannot separate from noise. At this
scale and budget, gating gives at most a small gain at matched parameter count; the result is
consistent with the small but consistent improvements reported for GLU variants, not evidence of a
large effect.

#figure(image("figures/abl_silu.png", width: 100%),
  caption: [SiLU feed-forward network against SwiGLU at matched parameter count.]) <fig-silu>

== OpenWebText <owt>

#prob("main_experiment")
The same architecture (with a 32,000-entry vocabulary, so 28.8M non-embedding and 45.2M total
parameters) trained for the same 10,000 steps at batch 128 reaches a final validation loss of
*#f3(s.at("owt_full").final_val_loss)* (perplexity 51) on OpenWebText (@fig-owt), against
#f3(s.at("ts_full").final_val_loss) (perplexity 3.8) on TinyStories.

#figure(image("figures/owt_full.png", width: 100%),
  caption: [OpenWebText, same architecture and number of iterations as the TinyStories run.]) <fig-owt>

*Interpreting the difference.* The two numbers are not a like-for-like quality comparison. They are
per-token losses under different tokenizers (a uniform guess costs $ln 32000 = 10.4$ against
$ln 10000 = 9.2$), and above all the data differ in entropy: TinyStories is generated text with a
vocabulary of a few thousand words and highly stereotyped story structure, so most next tokens are
nearly determined, whereas OpenWebText covers news, forums, code and every topic and register on
the web, where the next token is intrinsically far less predictable. The higher loss therefore
mostly measures a harder distribution, and partly an under-sized, under-trained model: the curve is
still falling steeply at the end, and 327.68M tokens is 61% of one pass over TinyStories but only
12% of one pass over OpenWebText.

*Generated text* (prompt "The city council voted on Tuesday to", temperature 0.8, top-p 0.9, 256 new tokens):

#sample("samples/owt_full_s0.md")

The output has the surface form of a news article (plausible sentence structure, quotations, named
officials) and stays on topic, but it is not coherent: statements contradict each other, phrases
such as "the ordinance on the ordinance" loop, and some word choices are nonsense. It is clearly
less fluent than the TinyStories sample even though model and compute are the same, because the
same capacity and the same number of updates must now cover a vastly broader distribution: far
more vocabulary, facts, entities and styles, each seen far fewer times. A model of this size can
essentially saturate TinyStories but is far from fitting web text.

== A 45-minute run <lb>

#prob("leaderboard")
This run was done locally, for 45 minutes of training time on one shared RTX A6000 (roughly
several times slower than a B200), using only the provided OpenWebText training set. *It was not
submitted to the course leaderboard.* The training script enforces the limit itself
(`--max-seconds 2700`): the wall-clock time spent in training steps is tracked, the cosine schedule
is driven by the _fraction of the time budget used_ instead of the step count so that the rate
reaches its minimum exactly when time runs out, and the loop stops at the first logging step past
the limit.

*Final validation loss: #f3(s.at("owt_leaderboard_v2").final_val_loss)* (200 batches of 128
sequences) after 9,100 steps in 45.1 minutes, which beats the naive 5.0 baseline (@fig-lb).

#figure(image("figures/owt_leaderboard.png", width: 72%),
  caption: [Validation loss against training wall-clock time for the two 45-minute attempts, with the unconstrained 10,000-step run for reference. The dashed vertical line is the 45-minute limit.]) <fig-lb>

What I did:
+ *Attempt 1: a larger model with tied embeddings* (6 layers, 8 heads, input and output embeddings
  shared, 35.1M parameters). Tying removes 16.4M parameters and was meant to pay for the two extra
  layers. It was a step backwards: only 5,200 steps fit in the budget and the final loss was
  #f3(s.at("owt_leaderboard").final_val_loss), behind the plain base model at every point in time.
  The tied model also started much more slowly (5.90 against 5.19 at step 500); I did not
  investigate whether a different initialisation scale or learning rate for the shared matrix
  would fix this.
+ *Attempt 2 (final): the base model with a time-aware schedule.* Same architecture as the
  OpenWebText run above (4 layers, 16 heads, untied), batch 128, peak rate $3 times 10^(-3)$, bf16
  and `torch.compile`, with the schedule fitted to the 45-minute budget. Annealing the rate fully
  within the budget is what makes the difference: the 10,000-step run is at about 4.17 after 45
  minutes because its rate is still high, while this run reaches 3.92 in the same time, the same
  loss the 10,000-step run needs 81 minutes for.
The honest summary is that my only architectural modification did not help and the gain comes from
the schedule and from throughput; the two attempts also ran while other jobs were using the same
GPU to different degrees, so their step counts are not an exact like-for-like comparison. Sample
from the final model (same prompt and decoding settings as above):

#sample("samples/owt_leaderboard_v2_s0.md")
