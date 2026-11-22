#set document(title: "CS336 Assignment 3 (scaling): Scaling Laws")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show table: set text(size: 9pt)
#show figure: set block(spacing: 1.2em)

#let results = json("generated/chinchilla_results.json")
#let sci(x, digits: 2) = {
  let e = calc.floor(calc.log(x))
  let m = calc.round(x / calc.pow(10.0, e), digits: digits)
  [$#m times 10^#e$]
}

#align(center)[
  #text(size: 16pt, weight: "bold")[CS336 Assignment 3 (scaling): Scaling Laws]
  #v(0.2em)
  Self-study write-up · handout version 26.0.5 (Spring 2026)
]

#block(fill: luma(240), inset: 9pt, radius: 3pt)[
  *Scope of this write-up.* Problem `chinchilla_isoflops` is answered exactly as specified.
  Problem `scaling_laws` depends on a training API that is reachable only from the Stanford
  network and is keyed by a registered Stanford student ID, so it could not be done as designed.
  It is replaced by a self-run, small-scale study that keeps the structure of the task (a capped
  budget of scaling runs, a prediction for a budget four times larger, and a single target run),
  with every training run executed locally on one shared RTX A6000. @sec-substitute lists every
  difference from the handout. No result was submitted to the course API or leaderboard.
]

= Problem `chinchilla_isoflops`

*Method.* The script `cs336_scaling/chinchilla_isoflops.py` groups the #results.points.len()·8 synthetic
runs in `data/isoflops_curves.json` by compute budget $C_i$. As the handout recommends, the run with the
lowest final loss in each group gives $N_"opt" (C_i)$ directly (no quadratic fit), and the matching
dataset size is $D_"opt" (C_i) = C_i \/ (6 N_"opt")$. The power laws $N_"opt" = k_N C^a$ and
$D_"opt" = k_D C^b$ are fitted with `scipy.optimize.curve_fit` as straight lines in
$log_10$–$log_10$ space, so every budget carries equal weight regardless of its magnitude.

#figure(
  table(
    columns: 3,
    align: (right, right, right),
    table.header([*$C_i$ (FLOPs)*], [*$N_"opt"$ (parameters)*], [*$D_"opt"$ (tokens)*]),
    ..results.points.map(p => (sci(p.compute_budget, digits: 1), sci(p.n_opt), sci(p.d_opt))).flatten(),
  ),
  caption: [Lowest-loss run of each IsoFLOP profile.],
)

Fitted laws: $N_"opt" = #calc.round(results.n_fit.k, digits: 3) dot C^(#calc.round(results.n_fit.exponent, digits: 3))$ and
$D_"opt" = #calc.round(results.d_fit.k, digits: 4) dot C^(#calc.round(results.d_fit.exponent, digits: 3))$.
The exponents sum to one, as they must when $D = C \/ (6N)$, and both are close to the 0.5 reported by
Hoffmann et al.: model size and data should grow at nearly the same rate.

== (a) Compute-optimal model size

#figure(image("figures/chinchilla_n_opt.png", width: 72%), caption: [Scaling law for model size. Red points: the $chevron.l C_i, N_"opt" (C_i) chevron.r$ pairs used for the fit; stars: extrapolations to $10^23$ and $10^24$ FLOPs.])

The predicted compute-optimal model size is about #sci(results.predictions.at("1e+23").n_opt) parameters (≈ 70 billion)
for a budget of $10^23$ FLOPs and about #sci(results.predictions.at("1e+24").n_opt) parameters (≈ 206 billion) for $10^24$ FLOPs.

== (b) Compute-optimal dataset size

#figure(image("figures/chinchilla_d_opt.png", width: 72%), caption: [Scaling law for dataset size, same conventions.])

The predicted compute-optimal dataset size is about #sci(results.predictions.at("1e+23").d_opt) tokens (≈ 238 billion)
for $10^23$ FLOPs and about #sci(results.predictions.at("1e+24").d_opt) tokens (≈ 809 billion) for $10^24$ FLOPs.

*Caveats.* Taking the raw argmin makes $N_"opt"$ jump between grid values (the profiles sample model
size in steps of roughly 1.6×, and e.g. the $6 times 10^20$ and $10^21$ budgets select almost the same
size), so the nine points scatter around the line by up to about 30 %. The extrapolation spans 1.5 to
2.5 orders of magnitude beyond the largest budget; an error of ±0.03 in the exponent alone, pivoting at the
largest fitted budget, changes the $10^24$ prediction by about ±20 %.

#figure(image("figures/chinchilla_profiles.png", width: 62%), caption: [The nine IsoFLOP profiles; stars mark the selected minima.])

= Problem `scaling_laws`: local substitute study <sec-substitute>

== Why the handout's procedure could not be followed

The handout asks for a scaling law fitted from at most 12 B200-hours of runs submitted to a training API at
`hyperturing.stanford.edu:8000`, and for a prediction of the best configuration and its loss for a
48 B200-hour run that the course staff then train. That API (i) is reachable only from the Stanford
network or VPN, (ii) authenticates with an 8-digit Stanford student ID that must be in the staff's user
table, and (iii) holds the budget ledger, the final-submission endpoint and the graded run. Without
Stanford credentials none of it is available, and the repository ships no offline surrogate.

== What was done instead

The same question is asked at a scale one shared RTX A6000 can answer, with the runs trained locally:

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  table.header([], [*Handout*], [*This study*]),
  [Who trains], [Staff API on B200 GPUs], [Local PyTorch runs, one shared RTX A6000],
  [Budget unit], [Wall-clock seconds], [FLOPs, $C = 6 N_"eff" D$ (see below); wall-clock on a shared GPU is not reproducible],
  [Scaling budget], [12 B200-hours (hard cap)], [$90 C_0$ = #sci(2.7e17) FLOPs (hard cap enforced by the queue), about 3 A6000-hours],
  [Target budget], [48 B200-hours = 4 × scaling budget], [$360 C_0$ = #sci(1.08e18) FLOPs = 4 × scaling budget, about 12 A6000-hours],
  [Target run], [Trained by staff, result not shown before grading], [Actually trained here, and compared with the prediction],
  [Model], [Staff JAX model: pre-norm RMSNorm, QK-norm, RoPE, SwiGLU], [PyTorch port of that model (`local_study/model.py`), fp32 weights with bf16 autocast instead of bf16 weights],
  [Data], [DCLM-baseline, Llama-2 tokenizer (32K), fixed order, no repetition, 500B tokens], [Same source, tokenizer, shuffle seed (67) and EOS convention; first 7.6B tokens prepared (48 shards)],
  [Validation], [$2^18$ tokens], [$2^18$ tokens from a shard never used for training (plus a $2^21$-token set for the final run)],
  [Optimizer], [AdamW, warmup + cosine], [Same: weight decay 0.01, betas (0.9, 0.95), clip 1.0, 5 % warmup, final LR 0.1 × peak],
)

Consequently the numbers below (optimal sizes of tens of millions of parameters, losses around 4–5 nats)
describe a regime roughly three orders of magnitude below the handout's, and say nothing about what the
best 48 B200-hour configuration would be. What carries over is the method.

== Design decisions

*Model size and compute accounting.* Model size is $N = 12 n_"layer" d_"model"^2$, as the handout
prescribes. For the budget, the plain $C = 6 N D$ rule is badly wrong at this scale: the unembedding
matrix has $32000 d_"model"$ weights and is applied to every token, which is 2.6 × the whole Transformer
stack for the smallest model in the grid and still 0.3 × for a 12-layer, 768-wide model. The handout's
wall-clock budget charges for that work, so the budget here does too: $C = 6 (N + V d_"model") D =
6 N_"eff" D$. Embedding lookups are free and are not counted. The smoke test in @sec-smoke checks
this accounting against measured time.

*Model family.* One shape family, so that size is a single axis: head dimension 64, $n_"layer" =
d_"model" \/ 64$, SwiGLU width $approx 8\/3 d_"model"$, untied embeddings, RoPE $theta = 10^6$,
RMSNorm $epsilon = 10^(-6)$ (the values of the handout's API example). Widths go from 192 to 1024 in
steps of 64, giving $N$ from 1.3M to 201M with a factor 1.2–2.4 between neighbours.

*Which runs.* An IsoFLOP design (Hoffmann et al., approach 2) was chosen over a Kaplan-style
sweep because the deliverable is the compute-optimal size at a given budget, which IsoFLOP
profiles measure directly, while the same runs still support a parametric $L(N, D)$ fit. Four budgets
$C_0, 2C_0, 4C_0, 8C_0$ with $C_0 = 3 times 10^15$ FLOPs, five model sizes each, the size window
shifted upwards with the budget. Since the location of each minimum is not known in advance, the queue
is adaptive: when the best size of a profile is on the edge of its window, the next size on that side is
added (at most two per profile) so that every minimum ends up bracketed. The geometric spacing spends
8/15 of the grid on the largest budget, the one closest to the target. The planned runs cost $81 C_0$;
the remaining $9 C_0$ of the cap is the reserve for extensions.

*Learning rate.* Before the grid, a brief check at $C_0$ on a small (d256L4) and a larger (d512L8) model
with peak learning rates $5 times 10^(-4)$, $10^(-3)$, $2 times 10^(-3)$ and $4 times 10^(-3)$; the rate
with the lowest mean final loss is used for every later run, and the two check runs at that rate are
reused as grid points. Checking two sizes guards against a rate that only suits small models, which
would bias every profile towards small $N$. These eight runs count against the scaling budget.

*Batch size.* Each run uses batch size 32 if that leaves at least 2000 optimizer steps and 16
otherwise. Larger batches do not fit next to the GPU's other users (@sec-smoke), and the rule is applied
identically to the target run.

*Seeds.* One seed per configuration; the budget is better spent on more sizes than on repeats. The
fit residuals give a rough idea of the noise.

== Fitting procedure

+ *IsoFLOP minima.* For each budget, a parabola in $log N$ through the three points around the
  lowest-loss run interpolates $N_"opt" (C_i)$ and $L_"min" (C_i)$. Unlike in the synthetic problem, the
  size grid here is coarse and the budget for it tiny, so the interpolated minimum is used for the fit and
  the raw argmin is reported next to it.
+ *Model-size law.* $N_"opt" = k C^a$ by least squares in log–log space, extrapolated to the target
  budget. The family member whose $N$ is nearest in log space is the target model, and it is given
  $D = C_"target" \/ (6 N_"eff")$ tokens.
+ *Loss prediction.* Hoffmann et al.'s parametric form $L(N, D) = E + A N^(-alpha) + B D^(-beta)$,
  fitted to all grid runs at the chosen learning rate with a Huber loss on $log L$ from 108 initialisations.
  Evaluated at the target configuration it gives the predicted final loss. Its own optimum over the
  family at the target budget is reported as a cross-check of step 2.
+ *Second opinion.* $L_"min" (C) = E + A C^(-gamma)$ through the four per-budget minima, extrapolated
  to the target budget.

The prediction is written to `experiments/prediction.json` before the target run is started, and the
target run reads its configuration from that file.

== Smoke test: throughput and FLOP accounting <sec-smoke>

#include "generated/study_smoke.typ"

== Results of the scaling runs

#include "generated/study_fit.typ"

== Target run: prediction against reality

#include "generated/study_target.typ"

*Reading the result.* The target run finished at a validation loss of 3.465, against a prediction of
3.617 made before it started: the parametric fit was pessimistic by 0.15 nats (4.4%), and the simple
envelope extrapolation by 0.35 nats. The target run improves on the best scaling run (4.053) by 0.59
nats, so the 0.15 error is about a quarter of the improvement being predicted. Three remarks:
- *Direction of the error.* Both predictors under-estimate how much a 45-fold larger budget helps. The
  grid only contains short runs (at most 176M tokens) of small models, where the loss is still in its
  fast early phase and where 8 of 21 runs had to use batch 16 instead of 32; the fitted data exponent
  ($beta = 0.33$) and irreducible term ($E = 3.21$) are therefore weakly constrained, and the run ended
  only 0.26 above the fitted "irreducible" loss, which suggests $E$ is too high.
- *The two ways of choosing the model disagreed.* The IsoFLOP power law selected d896L14 (135M
  parameters), whereas the parametric fit's own optimum at the target budget was the much smaller
  d512L8, for which it predicted about 3.54. I followed the IsoFLOP law, as the handout's method
  does, and only that model was trained, so whether d512L8 with 4.3B tokens would have done better
  or worse than 3.465 is not known. The disagreement itself is a sign that the exponent of
  $N_"opt"(C)$ (0.69, well above Chinchilla's 0.5) is not pinned down by four budgets spanning less
  than one decade, the largest of which has its minimum on the edge of its window (the run that
  would have bracketed it was skipped to stay inside the scaling budget).
- *What I would do with more budget:* spend it on fewer, larger budgets with bracketed minima rather
  than on the learning-rate check, and train the parametric optimum as a second target.

== Limitations

- *Scale.* Budgets of $10^15$–$10^18$ FLOPs and models of 1–200M parameters; at this scale embeddings
  and the vocabulary projection are a large part of the model, so exponents measured here need not match
  those at the handout's scale.
- *One seed, one family, one learning rate.* Aspect ratio, tied embeddings, weight decay and the batch
  size rule were fixed by judgement, not searched.
- *Extrapolation range.* The target budget is 45 × the largest grid budget; the handout's ratio between
  the largest affordable scaling run and the target is of the same order, so this is faithful, but it
  makes the loss prediction sensitive to the fitted exponents.
- *Shared hardware.* The GPU was shared with other users' jobs at 100 % utilisation, so times and
  throughputs are indicative only. Budgets are in FLOPs for that reason.
- *Precision.* bf16 autocast over fp32 weights, where the staff model keeps bf16 weights.

= Reproducing

```
uv sync --extra study
uv run --extra study python -m cs336_scaling.chinchilla_isoflops
uv run --extra study python -m cs336_scaling.local_study.data_prep --min-train-tokens 5.8e9
CUDA_VISIBLE_DEVICES=0 uv run --extra study python -m cs336_scaling.local_study.grid
uv run --extra study python -m cs336_scaling.local_study.fit
CUDA_VISIBLE_DEVICES=0 uv run --extra study python -m cs336_scaling.local_study.run_target
uv run --extra study python -m cs336_scaling.local_study.compare
uv run --no-project --with typst python writeup/compile.py
```
