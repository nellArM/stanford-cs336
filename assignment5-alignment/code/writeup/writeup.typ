#set document(title: "CS336 Assignment 5 (Alignment): Reasoning RL on GSM8K")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.4em, below: 0.8em, it)
#show raw.where(block: false): it => text(size: 9pt, it)
#set table(stroke: 0.5pt + luma(150), inset: 5pt)
#show table: set text(size: 8.5pt)
#show figure.caption: set text(size: 9pt)

#let problem(id, pts) = block(above: 1.2em, below: 0.6em, fill: rgb("#eef3fb"), inset: 6pt, radius: 3pt, width: 100%)[
  *Problem* #raw(id) (#pts)
]
#let E = math.op("E")
#let Var = math.op("Var")

#align(center)[
  #text(size: 16pt, weight: "bold")[CS336 Assignment 5 (Alignment): Reasoning RL on GSM8K]
  #v(0.3em)
  Spring 2026 handout v26.0.0 --- main assignment only (the optional supplement is not attempted) \
  Self-study submission
]

= Setup and deviations from the handout <deviations>

All code lives in `cs336_alignment/` (`grpo.py`: the seven tested components; `train_grpo.py`: the training loop;
`gsm8k.py`: data, prompts, rewards; `sweeps.py`: experiment lists) and `scripts/` (`eval_gsm8k_baselines.py`,
`run_queue.py`, `plot_grpo.py`). `uv run pytest tests/test_grpo.py` passes all GRPO tests.

The handout assumes two B200 GPUs per run on Modal. This work was done on a shared workstation, using two
RTX A6000 (48 GB) cards that were shared with other running jobs. The differences that follow from this are:

#table(
  columns: (1.1fr, 1.4fr, 1.6fr),
  table.header([*Item*], [*Handout*], [*This report*]),
  [Hardware], [2 #sym.times B200 per run, about 15 min per run], [2 #sym.times shared RTX A6000; one run at a time (a run needs both cards)],
  [Rollout steps per run], [200], [200],
  [Periodic validation], [1,024 examples every 10 steps], [256 examples every 20 steps, plus a final evaluation on 1,024 examples at the end of every run],
  [Seeds], [4 per configuration], [2 for standard GRPO, the on-policy variants and our own estimator; 1 for the learning-rate, prompt and off-policy runs],
  [Off-policy minibatches], [not specified], [one minibatch = one prompt's group of 8 rollouts. A first implementation that shuffled rollouts across prompts collapsed; it is kept as an ablation (see `grpo_experiments_off_policy`)],
  [Attention kernel of the policy], [`flash_attention_2`], [`sdpa` (PyTorch scaled-dot-product attention). The pinned flash-attn wheel needs glibc #sym.gt.eq 2.32; the machine has 2.31. Mathematically identical.],
  [NCCL transport for weight sync], [default (direct GPU-to-GPU P2P)], [`NCCL_P2P_DISABLE=1` (shared-memory transport). With the default transport the communicator initialises but the first collective between the trainer and vLLM never completes on this machine; a two-process reproduction outside the assignment code shows the same hang.],
  [vLLM memory], [`gpu_memory_utilization=0.9`], [0.35 (the card is shared)],
  [Sampling seeds], [one seed per batched vLLM request], [one request per prompt with seed `base + index`. vLLM applies a request's seed to every prompt in it, which makes samples for different questions share one random stream (visible as the same rare tokens across unrelated responses).],
  [Logging], [Weights & Biases], [local JSONL files under `experiments/`, plotted with matplotlib],
  [Infrastructure], [Modal], [plain local processes; `modal_utils.py` is unused],
)

Everything else follows the handout: model `allenai/OLMo-2-0425-1B` in bfloat16, GSM8K train/test, sampling
temperature 1.0, top-p 1.0, at most 512 new tokens, `</answer>` stop string for the `r1_zero` prompts, AdamW with
$beta = (0.9, 0.95)$ and no weight decay, learning rate $10^(-5)$, 32 prompts #sym.times 8 rollouts per step,
gradient-norm clipping at 1.0, and the provided `VLLMServer` with NCCL weight sync. The number of rollout
steps matches the handout, but with one or two seeds and 256-example interim evaluations the conclusions are weaker
than the handout's protocol would give; each section says how much weight its comparison can bear. The final
1,024-example evaluation has a binomial standard error of about 0.015 at an accuracy of 0.45, while the
seed-to-seed spread measured below is 0.01--0.16, so seed variance dominates every comparison.

= Prompting

#problem("prompting_baselines", "5 points")

*(a)* `scripts/eval_gsm8k_baselines.py` starts a vLLM server, formats all 1,319 GSM8K test questions with each
prompt, samples one response per question (temperature 1.0, top-p 1.0, 512 tokens; `</answer>` stop string for the
two `r1_zero` prompts only) and grades with `r1_zero_reward_fn` / `question_only_reward_fn`.

#include "generated/baselines.typ"

#include "baseline_commentary.typ"

= Group Relative Policy Optimization

#problem("baseline_calcs", "5 points")

With $p = sigma(theta)$ we have $nabla_theta log pi_theta (A = 1) = 1 - p$ and $nabla_theta log pi_theta (A = 0) = -p$.

*(a)* Let $Z = r(A) nabla_theta log pi_theta (A)$. Then $Z = 1 - p$ with probability $p$ and $Z = 0$ otherwise, so
$E[Z] = p(1-p)$, $E[Z^2] = p(1-p)^2$ and
$ Var(Z) = p(1-p)^2 - p^2(1-p)^2 = p(1-p)^3. $
The samples are i.i.d., so the estimator (a sample mean) has variance
$ Var = (p(1-p)^3) / n. $

*(b)* Now $Z_b = (r(A) - b) nabla_theta log pi_theta (A)$ takes the value $(1-b)(1-p)$ with probability $p$ and
$(-b)(-p) = b p$ with probability $1-p$. Its mean is unchanged,
$E[Z_b] = p(1-p)(1-b) + (1-p) b p = p(1-p)$, and
$ E[Z_b^2] = p(1-p)^2(1-b)^2 + (1-p) p^2 b^2 = p(1-p) [ (1-p)(1-b)^2 + p b^2 ]. $
Subtracting $E[Z_b]^2 = p(1-p) dot p(1-p)$ and expanding the bracket,
$(1-p)(1-b)^2 + p b^2 - p(1-p) = b^2 - 2b(1-p) + (1-p)^2$, gives
$ Var = (p(1-p) (b - (1-p))^2) / n. $
The baseline leaves the expectation alone but changes the variance quadratically in $b$. It is minimised (in fact
zero) at $b = 1 - p$: there both outcomes give the same value $Z_b = p(1-p)$, so a single sample already equals the
true gradient. $b = 0$ recovers (a). A baseline further than $2(1-p)$ from zero, or negative, is worse than no baseline.

*(c)* With $b = p$: $ Var = (p(1-p)(2p-1)^2) / n. $
Relative to (a) the ratio is $(2p-1)^2 \/ (1-p)^2$, which is below 1 exactly when $|2p - 1| < 1 - p$, i.e. when
$p < 2\/3$. So it *depends on $p$*: the mean baseline lowers the variance for $p < 2\/3$ (and removes it entirely at
$p = 1\/2$), ties at $p = 2\/3$, and *raises* it for $p > 2\/3$, where the optimal baseline $1-p$ is closer to 0
than to $p$.

#problem("tokenize_prompt_and_output, get_response_log_probs, compute_rollout_rewards, compute_group_normalized_rewards_grpo, compute_policy_gradient_loss_on_policy, aggregate_loss_across_microbatch_sequence, grpo_train_step_standard_on_policy", "code")

Implemented in `cs336_alignment/grpo.py` and exposed through `tests/adapters.py`; the corresponding tests pass.
Notes on choices that are not dictated by the tests:
- `get_response_log_probs` computes the log-probabilities with `cross_entropy` on float32 logits, so that autograd
  only retains the logits rather than a second (batch #sym.times length #sym.times 100k) log-softmax tensor; the entropy is
  computed without gradient. It also accepts an optional `response_mask`; when given, the output layer is applied only
  to response positions, which gives the same values there and avoids vocabulary-sized logits for prompt and padding
  tokens (most of the memory for the 3-shot prompt). The training loop uses this path.
- `grpo_train_step` tokenizes the whole batch once, then for each microbatch drops the columns that are pure padding.
  The loss of a microbatch is scaled by (microbatch size / full batch size) for sequence normalisation, or divided by
  the constant $Z$ for constant normalisation, so the accumulated gradient equals the full-batch gradient.
- Besides the required metrics (loss, pre-clip gradient norm, response-token entropy, train reward and format
  reward) it logs pass\@$k$ for $k = G$, the fraction of zero-advantage rollouts, the number of sequences and tokens
  actually trained on and, for the clipped estimators, the clip fraction.

#problem("grpo_experiments_standard_on_policy", "10 points")

*(a)* `cs336_alignment/train_grpo.py` is the training script. Each rollout step it (1) pushes the policy weights to
vLLM, (2) every `eval_every` steps samples one response for each validation question and logs reward, format reward,
response length and truncation rate, (3) samples 32 training questions and 8 rollouts each, (4) calls
`grpo_train_step`, and (5) appends all metrics to `metrics.jsonl`; training rollouts are dumped every 40 steps and
validation generations at the first and last step.

*(b), (c)*

#include "generated/standard.typ"

#include "generated/rollouts.typ"

*Is the script correct? (b)* Validation reward rises from 0.004 at step 0 to 0.05 (seed 0) and 0.14 (seed 1) at
step 20 and 0.30/0.38 at step 40, and the rollouts become well-formed, terse solutions (examples above). That is the
evidence that convinced us the loop is right; we also checked separately that a weight sync really changes vLLM's
output and that vLLM's log-probabilities match the trained policy's (mean absolute difference 0.024 nats per token,
stable across repeated syncs; `scripts/check_weight_sync.py`, `scripts/check_sync_drift.py`).

*(c) Metrics over training.* The *25% target is met*: the final 1,024-example validation reward is 0.461 and 0.450
(mean 0.456), and every interim evaluation from step 40 onwards is above 0.30. Training goes through three phases.
- *Steps 0--10: nothing to learn from.* The base model almost never answers correctly with the `r1_zero` format
  (0.08% in the baseline), so every group has identical (zero) rewards, every advantage is zero, and loss and gradient
  norm are exactly 0 (the zero-advantage fraction is 1.0). Learning starts only once a few groups contain a lucky
  correct answer.
- *Steps 10--50: format first, then accuracy.* Training format reward goes from 0.5 to 0.9 by step 50 and validation
  format reward from 0.50 to 0.90 by step 40; reward follows. Entropy falls from about 1.0--1.5 nats to 0.2 and the
  zero-advantage fraction from 1.0 to 0.3. Validation response length first *drops* (108 to 54--85 tokens at step 20:
  the earliest rewarded answers are short) and then grows to 120--160 tokens as the reasoning gets longer.
- *Steps 50--200: slow improvement.* Validation reward creeps from 0.36 to 0.46; entropy settles at 0.10--0.13 nats
  (an almost deterministic policy); the zero-advantage fraction rises back to 0.4--0.5 because more and more prompts
  are solved by all 8 samples. The pre-clip gradient norm stays around 0.8--1.0, so the clip at 1.0 is active on a
  large fraction of steps. The "loss" is not a loss in the usual sense (its minimum at step 25 coincides with the
  fastest change in advantages, not with the best model).

*Variance between runs* is small for this configuration: the two seeds differ mostly in *when* learning starts
(0.05 vs 0.14 at step 20), and by step 40 the curves overlap; the final rewards differ by 0.011. Each run took
1.0--1.1 hours (16--19 s per rollout step: about 10 s of generation, 6 s of training, 2 s of weight sync).

*Rollouts before and after training.* Before training the model often reasons correctly but does not produce the
format: in example 3 the arithmetic is right (80) but the `<think>`/`<answer>` structure is broken, and in example 1
the answer tag contains invented text. After training, answers are a short chain of thought ending in a bare number
(`<answer> 16</answer>`). Example 2 shows a failure that RL did not remove: the model now uses 4.2857 weeks per month
instead of the 4 weeks the GSM8K answer assumes --- the format is learned, but this kind of convention mismatch is
not.

#problem("grpo_learning_rate", "3 points")

#include "generated/lr.typ"

Three learning rates were run: $3 times 10^(-6)$ and $3 times 10^(-5)$ with one seed each, and the default
$10^(-5)$ with two seeds (the standard runs above).
- *$3 times 10^(-6)$: too slow.* Validation reward stays at or below 0.03 until step 160, reaches 0.08 at step 180
  and ends at *0.149*; training reward is still rising at the end (0.13 averaged over the last 20 steps), so this
  learning rate would probably catch up with a much longer run. Part of the slowness is a bf16 effect: the policy
  is stored in bfloat16 as the handout prescribes, and only 2.9% of the entries of a probed weight matrix change in
  an average optimizer step (vs 7--9% at $10^(-5)$), because most updates are smaller than half a bf16 ulp and round
  away.
- *$10^(-5)$: best final reward* (*0.456* mean of two seeds).
- *$3 times 10^(-5)$: fastest start, then divergence.* It learns fastest (validation reward 0.36 at step 20 vs
  0.05--0.14 at $10^(-5)$) but is unstable from early on: interim validation rewards are 0.36, 0.33, 0.41, 0.35 at
  steps 20--80 (best 0.406 at step 60) and training reward oscillates between 0.27 and 0.54. Around step 80 it
  breaks down: training reward 0.49 (step 72) #sym.arrow 0.27 (80) #sym.arrow 0.11 (88), token entropy
  0.14 #sym.arrow 0.30 #sym.arrow 1.47, pre-clip gradient norm 1.3 #sym.arrow 3.2 #sym.arrow 4.2, and responses
  lengthen from about 150 to 240 tokens. The run then ran out of GPU memory at the same point in all three attempts
  (the longer, higher-entropy rollouts need more activation memory on a card shared with another job), so it has no
  final evaluation; the plot shows its last 256-example evaluation (step 80) as a cross. We count it as diverged:
  the entropy blow-up and reward collapse happened before the crash, and were not caused by it.

So the useful range is narrow: $3 times$ below the default is too slow for a 200-step budget and $3 times$ above it is
past the edge of stability. With one seed for the two new learning rates the ordering $10^(-5) > 3 times 10^(-6)$
in final reward is solid (0.46 vs 0.15 is far outside the noise), but where exactly instability starts between
$10^(-5)$ and $3 times 10^(-5)$ is not resolved. We kept $10^(-5)$ for all later experiments.

#problem("grpo_prompt_ablation", "3 points")

#include "generated/prompt.typ"

One seed for each new prompt, compared against the two `r1_zero` seeds.
- *`r1_zero_three_shot`: best throughout training, same final reward.* It starts where the prompting baseline left
  off (0.18 at step 0; 19.3% in the full baseline evaluation) with 95% of responses already well-formed, so there is
  no cold-start phase: 0.34 at step 20, 0.43 at step 60, a plateau of 0.46--0.48 from step 100, and *0.462* at the
  end --- the same as zero-shot `r1_zero` (0.456). Its steps are slower (12 s of training per step instead of 6 s,
  peak memory 35 GB instead of 30 GB) because every prompt carries three worked examples.
- *`question_only`: long cold start, then fast learning.* Validation reward stays at or below 0.02 for 100 steps:
  the base model almost never puts a correct answer in `\boxed{}`, so almost every group has zero advantage. It then
  takes off (0.07 at step 120, 0.23 at 140, 0.35 at 180) and ends at *0.388*, still rising. Without the `</answer>`
  stop string its responses are 2.5 times longer (367 tokens at the end, vs 120--160 for `r1_zero`), which makes
  both generation and training slower.
- *`r1_zero`* sits in between: a 10-step cold start, then the curve described above.

The prompt therefore matters mostly for *how soon* RL gets a reward signal; once it has one, all three prompts climb
to similar accuracy. On the questions asked: three-shot has the best average reward over the run and ties for the
best final reward; question-only has the worst. Variance cannot be compared, since the two ablation prompts have one
seed each. We are confident in the ordering of the cold-start phases (it follows directly from the baseline format
rates and the effect is large) but not in any difference between the final rewards of `r1_zero` and
`r1_zero_three_shot`, and the `question_only` take-off time is itself random (it depends on when the first
correctly boxed answers happen to be sampled).

= RL algorithm variants

#problem("think_about_length_normalization", "1 point")

Per-sequence normalisation gives every rollout the same total weight in the update whatever its length, so the
gradient scale does not depend on how long the model's answers are or on the generation limit, and one very long
(often truncated, degenerate) rollout cannot dominate a batch. The price is bias: the true policy gradient weights
$nabla log pi(y)$, a _sum_ over tokens, by the advantage, whereas dividing by $"len"(y)$ down-weights every token of a
long response. Among correct answers this favours short ones, and among wrong answers it penalises long ones _less_
per token, so a model that is often wrong is pushed towards longer and longer responses --- the length drift reported
in the Dr. GRPO paper.

A constant normaliser keeps the estimator proportional to the true gradient and has no built-in preference about
length. Its drawbacks are practical: the update size now scales with the number of response tokens (with $Z = B G L$
and typical responses far shorter than $L$ the gradient is small and its scale drifts as lengths change, which
interacts with gradient clipping and the learning rate), and long rollouts contribute more variance.

Sequence normalisation is attractive when lengths vary wildly or contain outliers and stability matters more than
fidelity; constant normalisation is preferable when response length is itself something we do not want the
optimiser to bias (reasoning traces, a fixed token budget) or when we want to reason about the objective actually
being optimised.

#problem("compute_group_normalized_rewards_drgrpo, aggregate_loss_across_microbatch_constant", "code")

Implemented (`baseline="none"`, `advantage_normalizer="none"`, `loss_normalization="constant"`); tests pass.

#problem("think_about_rft", "2 points")

For binary rewards the RFT gradient is $1/Z sum_x sum_j r(y^((j))) nabla log pi(y^((j)))$: plain REINFORCE with
no baseline. Its expectation is $(G\/Z) nabla_theta eta(x)$ per prompt, where $eta(x) = E_(y tilde pi) r(y|x)$, so it
_is_ an unbiased policy gradient (despite looking like supervised learning) as long as the data are freshly sampled
from the current policy. The Dr. GRPO estimator has expectation $((G-1)\/Z) nabla_theta eta(x)$ (Eq. 24). The two
therefore point in the same direction in expectation and differ only by the factor $G\/(G-1)$.

They differ in variance. Dr. GRPO subtracts the group mean; by `baseline_calcs` a mean baseline reduces variance
whenever the success rate is below about 2/3 and removes it entirely for groups that are all right or all wrong
(those contribute exactly zero, whereas RFT still takes a full-size likelihood step on every correct sample of an
already-solved prompt). When the success rate is very low the two nearly coincide: $mu approx 0$, the correct
samples get weight $approx 1$ and the wrong ones a small negative weight.

RFT is preferable when simplicity and cost matter (only correct rollouts go through the model; it is ordinary
maximum-likelihood training, so it can never be asked to push a log-probability towards $-infinity$, which makes it
very stable), and early in training when rewards are sparse. Dr. GRPO is preferable once the policy is often right:
RFT then keeps sharpening the distribution on prompts it already solves (lower entropy, larger updates, more
variance), while the baseline makes those prompts fade out and also uses the information in failures.

#problem("derive_difficulty_reweightings", "6 points")

Write $eta(x) = E_(y tilde pi_theta) r(y|x)$ for the current success rate on prompt $x$, so that
$nabla_theta eta(x) = E_y [r(y|x) nabla_theta log pi_theta (y|x)]$ and
$nabla_theta J_(theta, w) = E_x [w(x) nabla_theta eta(x)]$.

*(a) Dr. GRPO: $w(x) = 1$.* As $G -> infinity$ the group mean $mu -> eta(x)$ almost surely (law of large numbers) and
the average over the group becomes an expectation, so with $Z = G$ the inner term tends to
$E_y [(r - eta(x)) nabla log pi] = nabla eta(x) - eta(x) E_y [nabla log pi] = nabla eta(x)$, using
$E_y [nabla log pi] = 0$ (Eq. 17). The estimator is the plain policy gradient: no reweighting. (For finite $G$ the
factor is $(G-1)\/G$, independent of $x$.)

*(b) GRPO: $w(x) = 1 \/ sqrt(eta(x)(1 - eta(x)))$.* For a binary reward the group standard deviation converges to the
Bernoulli standard deviation $sqrt(eta(x)(1-eta(x)))$, a constant given $x$ and the (stop-gradient) policy. It
factors out of the limit in (a), giving $nabla eta(x) \/ sqrt(eta(x)(1-eta(x)))$. Prompts that are nearly always
solved _or_ nearly never solved are up-weighted; prompts with $eta = 1\/2$ get the smallest weight (2).

*(c) MaxRL: $w(x) = 1 \/ eta(x)$.* The group mean converges to $eta(x)$, so the limit is
$nabla eta(x) \/ eta(x) = nabla log eta(x)$. Hard prompts are up-weighted in inverse proportion to their success rate
and easy prompts are not; the surrogate is $E_x [log eta(x)]$, the log-likelihood of producing a correct answer,
which is where the name "maximum likelihood RL" comes from.

#problem("think_about_advantage_normalization", "2 points")

*No normalisation* (Dr. GRPO) optimises the true expected reward, but the signal from a prompt is proportional to
$eta(1-eta)$-sized fluctuations: it vanishes on very hard prompts, so progress on them is slow, and its scale
depends on the reward scale. *Std normalisation* (GRPO) makes every non-degenerate group contribute advantages of
unit scale, so it is invariant to the scale/shift of the reward and gives hard prompts a strong signal
($w = 1\/sqrt(eta(1-eta))$). But it up-weights nearly solved prompts just as much: with $G = 8$ and seven correct
answers the single failure gets advantage $-2.47$, so the optimiser spends effort (and variance) chasing rare
failures, which are disproportionately grader errors or noise. *Mean normalisation* (MaxRL, $w = 1\/eta$) focuses
only on the hard end and has a clean maximum-likelihood interpretation, and easy prompts fade out as with no
normalisation. Its weakness is variance and fragility at small $eta$: a single lucky success in a group of 8 gets
advantage $+7$, so a false positive of the grader or a right-for-the-wrong-reason answer is amplified strongly, and
it is only meaningful for non-negative rewards (a zero or negative group mean breaks it).

Std normalisation suits shaped or arbitrarily scaled rewards; mean normalisation suits sparse binary rewards where
the frontier of barely solvable prompts matters most; no normalisation suits the late phase where most prompts are
solved and we do not want to amplify the remaining noise, or cases where an unbiased objective is required.

#problem("compute_group_normalized_rewards_maxrl, grpo_train_step_variants_on_policy", "code")

Implemented; tests pass. Zero-advantage rollouts are removed before any forward pass and the remaining ones are
processed with the same microbatch size, so the number of forward/backward passes shrinks proportionally. The
normalisers still use the size of the full batch ($B G$ for sequence normalisation, $Z$ for constant normalisation),
so the gradient is identical to the unpruned one --- the pruned terms are exactly zero.

#problem("grpo_experiments_variants_on_policy", "10 points")

$Z = B G L = 256 times 512$ for all constant-normalised variants.

#include "generated/variants.typ"

#table(
  columns: (auto, auto, auto, auto, 1fr),
  align: (left, center, center, center, left),
  table.header([*method*], [*final val. reward (seeds 0, 1)*], [*mean*], [*spread*], [*notes*]),
  [GRPO (sequence norm.)], [0.461, 0.450], [*0.456*], [0.011], [pre-clip grad. norm about 1.0],
  [GRPO_constant], [0.440, 0.368], [0.404], [0.072], [seed 1 dips to 0.34--0.36 at steps 160--180],
  [Dr_GRPO], [0.284, 0.446], [0.365], [0.162], [seed 0 plateaued at 0.37, then fell to 0.28 (format 0.72) at the end],
  [RFT], [0.423, 0.432], [0.427], [0.009], [slowest start; highest format reward (0.95--0.96)],
  [MaxRL], [0.429, 0.409], [0.419], [0.020], [seed 1 starts late (0.10 at step 40)],
)

*Average performance.* Standard GRPO has the highest mean final reward (0.456), followed by RFT (0.427), MaxRL
(0.419), GRPO_constant (0.404) and Dr_GRPO (0.365). The spread of the means (0.09) is about the same size as the
spread between two seeds of a single method, so *with two seeds we cannot rank the variants*; the only robust
statement is that all of them reach 0.37--0.46 in 200 steps and none is clearly better than standard GRPO.

*Variance.* RFT (0.009) and standard GRPO (0.011) are the most consistent across seeds; Dr_GRPO is the least
(0.162). Its seed 0 reached 0.37 by step 140 and then degraded: the final evaluation has a format reward of only 0.72
and 184-token responses, so this is a late instability of that run rather than evaluation noise. GRPO_constant also
has one weaker seed. Two seeds are too few to tell whether these are properties of the methods or bad luck.

*Other systematic differences.* (i) The pre-clip gradient norm of the constant-normalised variants is 3--8 times
smaller (0.1--0.4 vs about 1.0) because the loss is divided by $Z = B G L = 131{,}072$ instead of by each response's
length. With AdamW the overall scale of the gradient largely cancels, so the effective step size is similar; but
gradient clipping at 1.0 never activates for them, while it does for standard GRPO. (ii) RFT trains on fewer
sequences (all incorrect rollouts are pruned, so the zero-advantage fraction stays at 0.5--0.65), which makes it the
cheapest variant (10.5 s per step vs 16--19 s for standard GRPO) and the slowest to start, but it ends with the
highest format reward. That fits `think_about_rft`: it is plain maximum-likelihood training on its own correct
samples. (iii) Entropy and response length behave alike across variants (0.10--0.17 nats and 120--160 tokens at the
end).

*Tuning.* All variants used the learning rate tuned for standard GRPO, so a variant that does worse is not shown to
be worse. Dr_GRPO and GRPO_constant, whose weaker seeds degraded late in training, might prefer a lower learning
rate or a decay schedule; RFT, which is stable but slow to start, might tolerate a higher one.

= Off-policy RL

#problem("derive_surrogate_objectives", "2 points")

Assume even length $L$. For each pair index $t in {1, ..., L\/2}$ define the surrogate policy that samples every
position from the old policy $pi_0$ _except_ positions $2t-1$ and $2t$, which are sampled from the current policy
$pi_theta$ (the second of them conditioned on the first, as usual):
$ tilde(pi)_t^"pair" (y|x) = (product_(s < 2t-1) pi_0 (y_s | x, y_(<s))) pi_theta (y_(2t-1) | x, y_(<2t-1)) pi_theta (y_(2t) | x, y_(<2t)) (product_(s > 2t) pi_0 (y_s | x, y_(<s))). $
The estimator optimises the expected reward under each of these, summed over pairs:
$ J_theta^"pair" = E_x [ sum_(t=1)^(L\/2) E_(y tilde tilde(pi)_t^"pair" (y|x)) [ r(y|x) ] ]. $
_Derivation._ Write $rho_t (y) = (pi_theta (y_(2t-1) | dot) pi_theta (y_(2t) | dot)) \/ (pi_0 (y_(2t-1) | dot) pi_0 (y_(2t) | dot))$. Since
$tilde(pi)_t^"pair" (y|x) = pi_0 (y|x) rho_t (y)$, importance sampling gives
$E_(y tilde tilde(pi)_t^"pair") [r] = E_(y tilde pi_0) [rho_t (y) r(y|x)]$. Only $rho_t$ depends on $theta$, and by the
log-derivative trick $nabla_theta rho_t = rho_t nabla_theta log (pi_theta (y_(2t-1) | dot) pi_theta (y_(2t) | dot))$. Hence
$ nabla_theta J_theta^"pair" = E_x E_(y tilde pi_0) [ sum_(t=1)^(L\/2) rho_t (y) r(y|x) nabla_theta log (pi_theta (y_(2t-1) | x, y_(<2t-1)) pi_theta (y_(2t) | x, y_(<2t))) ], $
which is exactly the expectation of the given estimator. The pairwise scheme sits between token-level reweighting
(blocks of size 1) and full sequence-level reweighting (one block of size $L$, which is unbiased): the larger the
block, the more of the prefix/suffix mismatch is corrected and the higher the variance of the weight.

#problem("compute_policy_gradient_loss_off_policy, compute_policy_gradient_loss_off_policy_gspo, grpo_train_step_off_policy", "code")

Implemented; tests pass. The GSPO weight is computed as $exp$ of the masked mean of the per-token log-ratios (never as
a product of ratios), which is numerically stable. The clip fraction is the fraction of response tokens for which the
clipped branch of the $min$ is active, i.e. tokens that contribute no gradient.

In the training script the advantages need group statistics, so they are computed once on the whole rollout batch of
256. The batch is then split into 32 minibatches of 8 sequences, each giving one optimizer step; a minibatch whose
sequences all have zero advantage is skipped, so the number of optimizer steps per rollout batch can be below 32 (it
is logged). Two options matter for the results below:
- *Minibatch order.* `--minibatch-order grouped` (the default, used for the reported off-policy runs) shuffles
  prompts but keeps each prompt's group of 8 together, i.e. one minibatch is one group. `--minibatch-order shuffled`
  shuffles all 256 rollouts before cutting minibatches; our first implementation did this, and it makes off-policy
  training collapse (analysed under `grpo_experiments_off_policy`).
- *Old log-probabilities.* They must come from the sampling policy, so they are computed for all minibatches before
  the first update, under `no_grad`. The first implementation computed them in separate scoring batches of 16 sequences.
  Because bf16 results depend slightly on batch shape, the ratio before any update was then not exactly 1: on the
  first rollout batch of `offpolicy_clip_s0` the per-token $|log w|$ averages 0.003 (only 0.04% of tokens are outside
  $1 plus.minus 0.2$), but 21% of sequences have a geometric-mean ratio outside GSPO's $1 plus.minus 3 times 10^(-4)$
  (`experiments/logs/offpolicy_ratio_check.json`). The reported runs use `--old-logprob-scoring matched`, which
  recomputes the old log-probs with exactly the tensors of the training forward; the ratio is then exactly 1 before
  the first update (maximum $|log w| = 0$, measured with the same script).

#problem("think_about_importance_reweighting", "2 points")

*(a) No reweighting* adds no variance at all but is biased, and the bias grows with every step taken on the same
batch because the samples become less representative of the current policy. *(b) Clipped token-level reweighting* is
in the middle: each weight involves one token and is effectively confined to $[1-epsilon, 1+epsilon]$, so the variance
is small and bounded, and the clipping doubles as a trust region that stops the update once a token's probability has
moved by a factor $1 plus.minus epsilon$. It is still biased --- it ignores how the prefix and suffix distributions
have shifted --- and the per-token weights within one response can disagree with each other even though the reward
is a property of the whole response. *(c) GSPO* uses one weight per response, the geometric mean of the token ratios.
Averaging about $L$ log-ratios makes that weight much less noisy than any single token's ratio and consistent with
the sequence-level reward, but it is not the true sequence weight (which is the product), so it is biased too, and a
clipped response loses its whole gradient at once. With the very tight $epsilon = 3 times 10^(-4)$ it behaves like a
strict trust region on the average per-token change.

No reweighting is the right choice when the policy barely moves between rollouts (few steps, small learning rate):
correction would only add noise. Token-level clipping is a robust default for many reuse steps. GSPO should help when
per-token ratios are noisy or occasionally extreme --- long responses, or models whose token probabilities jump
(the GSPO paper motivates it with mixture-of-experts instabilities) --- where a few outlier tokens would otherwise
dominate or be clipped arbitrarily.

#problem("grpo_experiments_off_policy", "10 points")

The off-policy runs use standard GRPO advantages and sequence normalisation, `rollout_batch_size = 256`,
`train_batch_size = 8`, `gradient_accumulation_steps = 1`, and $epsilon = 0.2$ (clip) or $3 times 10^(-4)$ (GSPO).

#include "generated/offpolicy.typ"

The runs reported in the table use group-aligned minibatches (one minibatch = one prompt's 8 rollouts) and matched
old-log-prob scoring. Our first implementation shuffled rollouts across prompts and collapsed; it is analysed at the
end of this section.

*Performance.* With one seed each, all four off-policy variants end at about the same accuracy as on-policy GRPO:
naive 0.450, noclip 0.505, clip 0.439, GSPO 0.478, against 0.450--0.461 for the two on-policy seeds. The spread among
them (0.07) is within the seed-to-seed spread seen elsewhere in this report, so we do not rank them by final reward.

*Speed.* Taking several optimizer steps per rollout batch costs almost nothing extra in wall-clock time here (15.6--18.5
s per rollout step vs 16--19 s on-policy), because the same 256 sequences go through forward and backward once either
way; only the number of optimizer steps changes. About half of the 32 minibatches are skipped because the whole group
has zero advantage, so the runs take about 16 optimizer steps per rollout batch instead of 1. Early learning is
much faster for the variants whose updates are not restricted: validation reward at step 20 is 0.32 (naive) and 0.20
(clip), against 0.05/0.14 on-policy. Noclip (0.02) and GSPO (0.00) are slower to start, and GSPO only takes off after
step 40, because its tight clip holds back about half of the sequences (below). After about step 60 all curves are
on the same plateau.

*Stability.* The naive variant is not stable. The same configuration was attempted three times (the queue retries a
failed run). In the first two attempts the policy diverged --- at step 113 and at step 193 the training reward fell
to 0.29 and 0.05, token entropy jumped from about 0.1 to 1.7 and 2.5 nats and the pre-clip gradient norm rose to 14
and 40 --- after which the runs ran out of GPU memory. Only the third attempt stayed stable to step 200, and the 0.450
reported above is that survivor, so it overstates how reliable naive off-policy training is. The two divergent
attempts were at or above on-policy accuracy until shortly before they broke (0.41 at step 100; 0.39 at step 180).
The other variants ran once and finished, but noclip shows a gradient-norm spike to about 18 around step 170, and
all off-policy variants run with pre-clip gradient norms of 5--10 (vs about 1 on-policy, because each minibatch
averages over only 8 sequences), so the clip at 1.0 is active on almost every step. GSPO has the smoothest gradient
norm (3--5) and keeps the highest entropy.

*Clip fractions.* The two clipping schemes behave completely differently at their default $epsilon$. Token-level
clipping with $epsilon = 0.2$ almost never binds: 0.8% of response tokens are clipped, and the mean importance ratio
stays at 1.0002 even after 16 updates on the same batch. In this regime the clip variant is practically the noclip
estimator, which fits their similar curves. GSPO with $epsilon = 3 times 10^(-4)$ clips 44--46% of sequences
throughout training, so it acts as a strong brake on how far one rollout batch can move the policy. That explains both
its slow start and its smoother gradients.

*Tuning.* Token clipping would need a much smaller $epsilon$ (or more steps per batch) before its trust region does
anything at all. GSPO's $epsilon$ could probably be loosened to learn faster. Naive off-policy training would need a
lower learning rate, fewer steps per batch or a decay schedule to be reliable.

*Confidence.* One seed per variant. The solid findings are: off-policy training with group-aligned minibatches
reaches on-policy accuracy at the same compute; the naive variant diverges in 2 of 3 attempts; and the token-clip
fraction is below 1% while GSPO's is about 45%. Any ranking of final rewards is not supported.

#include "generated/offpolicy_shuffled.typ"

*The shuffled-minibatch implementation collapses.* In our first implementation the 256 rollouts were shuffled
across prompts before being cut into minibatches of 8, and old log-probs were scored in separate batches of 16. Three of
the four variants collapsed completely (final validation reward 0.000 for naive, noclip and clip); GSPO reached
0.499, but only after a 60-step stall. The naive run collapsed to 5-token responses by step 4; noclip and clip lost the answer format within
20--40 steps (format reward 0.03 by step 30 for clip) and drifted to 80- and 300-token unformatted responses; GSPO
showed no improvement until step 60.

We checked the possible implementation errors directly:
- The importance ratio on the first minibatch after each rollout is close to 1 (0.97--1.01 on update steps; per-token
  $|log w|$ averages 0.003 when re-measured on a saved batch), so the old log-probs come from the same policy state and
  mode as the training forward.
- Sign and clipping of the loss are pinned by the snapshot tests.
- The zero-advantage pruning uses the full-batch normalisation.
- The masks are shared between the old and new scoring.
- Weight sync is not to blame either: repeatedly syncing unchanged weights leaves vLLM's greedy outputs and log-probs
  unchanged and matching the trainer's to 0.024 nats per token (`experiments/logs/sync_drift.json`).

The cause is how the minibatches were built.
- Within a group the advantages sum to zero: with one correct answer out of 8, GRPO gives $+2.47$ to it and $-0.35$
  to each of the other seven. In the on-policy update these terms are added up, so the gradient on tokens shared by
  all eight responses --- `</think> <answer>`, the number formatting, the opening words --- largely cancels, and what
  remains distinguishes the correct answer from the wrong ones.
- After shuffling, those 8 sequences land in about 7 different minibatches. Each minibatch then contains a single
  formatted *wrong* answer with a negative advantage and nothing to cancel it.
- AdamW normalises the update size, so each of those minibatches produces a full-size step pushing down the
  probability of the format itself.

Early in training, when a rollout batch typically contains exactly one group with a correct answer, a single batch
produced 5--8 such steps (logged as 5--8 optimizer steps per update in the shuffled runs), and the format
disappeared after a few such batches. That matches the logs: the collapse happens exactly on the rollout steps with
non-zero updates, while the clip fraction stays at 1--5% and the ratio near 1 --- clipping cannot help, because each
offending token is only updated in its own minibatch, once. GSPO survived only because its tight clip held back many
of those steps.

Keeping each group in one minibatch restores the zero-sum structure inside every optimizer step and removes the
collapse. In an 80-step diagnostic with the same seed, naive and clip with grouped minibatches reached 0.38 and 0.44
at step 80 (format 0.93 and 0.95), against 0.00 for both with shuffling; the 200-step runs in the table above use the
fix. The off-policy loop now defaults to `--minibatch-order grouped` and `--old-logprob-scoring matched`; the shuffled
runs are kept (`experiments/*_shuffled_s0`, sweep `offpolicy_shuffled`) as an ablation. The general lesson: when
advantages are normalised within groups, an off-policy minibatch must contain whole groups, or the baseline no longer
does its job inside each step.

= Own policy gradient estimator

#problem("try_your_own", "10 points")

*Proposal: sqrt-MaxRL.* Take MaxRL (mean baseline, constant length normalisation) and change exactly one thing: divide
the centred reward by the _square root_ of the group mean instead of the group mean,
$ hat(g) <- 1/Z sum_(i=1)^B sum_(j=1)^G sum_t (r(y^((i,j)) | x^((i))) - mu_i) / (sqrt(mu_i) + epsilon) nabla_theta log pi_theta (y_t^((i,j)) | x^((i)), y_(<t)^((i,j))). $
In the code this is `advantage_normalizer="sqrt_mean"` (variant `sqrt_maxrl` in `train_grpo.py`).

*Justification.* By the argument of `derive_difficulty_reweightings`, in the large-group limit this optimises
$J_(theta, w)$ with $w(x) = eta(x)^(-1\/2)$, i.e. it does gradient ascent on $E_x [2 sqrt(eta(x))]$. The three
normalisers studied so far that do not up-weight easy prompts form a family $w = eta^(-alpha)$ with surrogate
objectives
#align(center, table(
  columns: 4, align: center,
  table.header([*estimator*], [$alpha$], [*weight* $w(x)$], [*surrogate objective*]),
  [Dr. GRPO], [0], [1], [$E_x [eta(x)]$ (accuracy)],
  [*sqrt-MaxRL (ours)*], [1/2], [$eta^(-1\/2)$], [$E_x [2 sqrt(eta(x))]$],
  [MaxRL], [1], [$eta^(-1)$], [$E_x [log eta(x)]$ (likelihood of a correct answer)],
))
and GRPO's $1\/sqrt(eta(1-eta))$ behaves like $alpha = 1\/2$ at the hard end but additionally blows up at the easy end.
The motivation for the intermediate exponent is a bias--variance argument at finite group size. MaxRL's weight is
estimated by $1\/mu_i$ from only $G = 8$ samples. The smallest non-zero group mean is $1\/8$, where the single
correct rollout receives advantage $(1 - 1\/8) \/ (1\/8) = 7$, and that group mean is compatible with true success
rates anywhere from a few percent to 40%: the weight is both large and very noisy, and it lands on exactly the
samples most likely to be lucky guesses or grader false positives. With the square root the same rollout gets
$(7\/8) \/ sqrt(1\/8) approx 2.5$ --- still 2.8 times more than under Dr. GRPO, so hard prompts remain emphasised, but
the amplification of the estimation noise is tempered (the multiplier ranges over $[1, 2.83]$ instead of $[1, 8]$).
Compared with GRPO it shares the hard-end behaviour ($eta^(-1\/2)$) but, like MaxRL, does not chase the last
failures on nearly solved prompts: with seven of eight correct the lone failure gets advantage $-0.94$ instead of
GRPO's $-2.47$. The expected outcome is learning that is faster than Dr. GRPO early (sparse rewards) with lower
seed-to-seed variance than MaxRL.

#include "generated/own.typ"

*Results (two seeds each, same hyperparameters as the other variants).*
#align(center, table(
  columns: 4, align: (left, center, center, center),
  table.header([*estimator*], [*final val. reward (seeds 0, 1)*], [*mean*], [*val. reward at step 40 (seeds 0, 1)*]),
  [Dr. GRPO ($alpha = 0$)], [0.284, 0.446], [0.365], [0.06, 0.36],
  [*sqrt-MaxRL ($alpha = 1\/2$)*], [0.443, 0.423], [*0.433*], [0.29, 0.29],
  [MaxRL ($alpha = 1$)], [0.429, 0.409], [0.419], [0.36, 0.10],
  [standard GRPO (sequence norm.)], [0.461, 0.450], [0.456], [0.30, 0.38],
))

sqrt-MaxRL has a higher mean final reward than both of the estimators it interpolates between (by 0.07 over Dr. GRPO
and 0.014 over MaxRL), and its two seeds are close together, both at the end (0.02 apart) and during the take-off
(both 0.29 at step 40, whereas Dr. GRPO and MaxRL each had one fast and one slow seed). That is the direction the
variance argument predicts. But the evidence is weak: with two seeds per method, a 0.014 difference from MaxRL is
well inside the seed-to-seed spread, the larger gap to Dr. GRPO comes entirely from one degraded Dr. GRPO seed, and
sqrt-MaxRL does *not* beat standard GRPO (0.433 vs 0.456). Our conclusion is that the intermediate exponent is a
reasonable, stable choice that is at least as good as its neighbours here, not that it is better; establishing that
would need at least the handout's four seeds and a learning rate tuned per method.

= Reproduction

```
uv sync --extra gpu --no-install-package flash-attn       # flash-attn wheel needs glibc >= 2.32
.venv/bin/python -m pytest tests/test_grpo.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_gsm8k_baselines.py --vllm-gpu 0
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/check_weight_sync.py              # sync really changes vLLM's output
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/run_queue.py --sweep scaled              # the 20 reported runs
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/run_queue.py --sweep offpolicy_shuffled  # collapse ablation
.venv/bin/python scripts/plot_grpo.py
```
