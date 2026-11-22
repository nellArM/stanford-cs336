"""Experiment lists for the run queue (scripts/run_queue.py).

``scaled``  the reduced-budget sweep actually run for the report (2 shared RTX A6000s).
``handout`` the sweep as specified in the handout (200 rollout steps, 4 seeds, 1,024 validation examples).

Each entry is a dict of train_grpo.py flags; ``group`` is only used by the plotting script.
"""

from __future__ import annotations

# The sweep was run with 200 rollout steps (the handout's number) after an initial 80-step plan.
SCALED_COMMON = dict(num_rollout_steps=200, n_val_examples=256, eval_every=20, n_final_val_examples=1024)
HANDOUT_COMMON = dict(num_rollout_steps=200, n_val_examples=1024, eval_every=10, n_final_val_examples=1024)

ON_POLICY_VARIANTS = ["grpo_constant", "dr_grpo", "rft", "maxrl"]
OFF_POLICY_VARIANTS = ["offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo"]


def _run(name: str, group: str, variant: str = "grpo", seed: int = 0, **kwargs) -> dict:
    return dict(run_name=name, group=group, variant=variant, seed=seed, **kwargs)


def make_sweep(
    common: dict,
    standard_seeds: int,
    lr_values: list[float],
    lr_seeds: int,
    prompt_seeds: int,
    variant_seeds: int,
    off_policy_seeds: int,
    own_seeds: int,
) -> list[dict]:
    runs: list[dict] = []
    # grpo_experiments_standard_on_policy
    for s in range(standard_seeds):
        runs.append(_run(f"grpo_s{s}", "standard", seed=s))
    # grpo_learning_rate (the default 1e-5 is covered by the standard runs)
    for lr in lr_values:
        for s in range(lr_seeds):
            runs.append(_run(f"grpo_lr{lr:g}_s{s}", "lr", seed=s, learning_rate=lr))
    # grpo_experiments_variants_on_policy
    for variant in ON_POLICY_VARIANTS:
        for s in range(variant_seeds):
            runs.append(_run(f"{variant}_s{s}", "variants", variant=variant, seed=s))
    # try_your_own
    for s in range(own_seeds):
        runs.append(_run(f"sqrt_maxrl_s{s}", "own", variant="sqrt_maxrl", seed=s))
    # grpo_prompt_ablation
    for prompt in ["question_only", "r1_zero_three_shot"]:
        for s in range(prompt_seeds):
            runs.append(_run(f"grpo_{prompt}_s{s}", "prompt", seed=s, prompt=prompt))
    # grpo_experiments_off_policy
    for variant in OFF_POLICY_VARIANTS:
        for s in range(off_policy_seeds):
            runs.append(_run(f"{variant}_s{s}", "offpolicy", variant=variant, seed=s))
    return [{**common, **r} for r in runs]


SWEEPS = {
    # 2 + 2 + 8 + 2 + 2 + 4 = 20 runs
    "scaled": make_sweep(
        SCALED_COMMON,
        standard_seeds=2,
        lr_values=[3e-6, 3e-5],
        lr_seeds=1,
        prompt_seeds=1,
        variant_seeds=2,
        off_policy_seeds=1,
        own_seeds=2,
    ),
    # 4 + 8 + 16 + 4 + 4 + 16 = 52 runs
    "handout": make_sweep(
        HANDOUT_COMMON,
        standard_seeds=4,
        lr_values=[3e-6, 3e-5],
        lr_seeds=4,
        prompt_seeds=2,
        variant_seeds=4,
        off_policy_seeds=4,
        own_seeds=4,
    ),
    # The first version of the off-policy runs: minibatches shuffled across prompts and old log-probs
    # scored in separate batches of 16.  These runs collapse (see the report); their logs were moved to
    # experiments/<variant>_shuffled_s0 and are plotted as an ablation.  Rerun with this sweep.
    "offpolicy_shuffled": [
        dict(run_name=f"{v}_shuffled_s0", group="offpolicy_shuffled", variant=v, seed=0, minibatch_order="shuffled",
             old_logprob_scoring="separate", **SCALED_COMMON)
        for v in OFF_POLICY_VARIANTS
    ],
    # 80-step diagnostic that first showed grouped minibatches fix the collapse (superseded by the
    # 200-step grouped runs of the "scaled" sweep, not plotted).
    "offpolicy_grouped_diagnostic": [
        dict(run_name=f"{v}_grouped_s0", group="offpolicy_grouped_diagnostic", variant=v, seed=0,
             minibatch_order="grouped", num_rollout_steps=80, n_val_examples=256, eval_every=20, n_final_val_examples=1024)
        for v in ("offpolicy_naive", "offpolicy_clip")
    ],
    # A tiny end-to-end check of the whole loop (weight sync, rollouts, update, eval).
    "smoke": [
        # The 3-shot prompt gives non-zero rewards from the first step, so these two exercise (and
        # time) a full-size policy update; zero-shot r1_zero rollouts are almost all reward 0 at first.
        dict(
            run_name="smoke_three_shot", group="smoke", variant="grpo", seed=0, prompt="r1_zero_three_shot",
            num_rollout_steps=3, n_val_examples=64, eval_every=3, n_final_val_examples=64,
        ),
        dict(
            run_name="smoke_three_shot_offpolicy_clip", group="smoke", variant="offpolicy_clip", seed=0,
            prompt="r1_zero_three_shot", num_rollout_steps=2, n_val_examples=64, eval_every=2, n_final_val_examples=64,
        ),
        dict(
            run_name="smoke_grpo", group="smoke", variant="grpo", seed=0,
            num_rollout_steps=3, n_val_examples=64, eval_every=3, n_final_val_examples=64,
        ),
        dict(
            run_name="smoke_offpolicy_gspo", group="smoke", variant="offpolicy_gspo", seed=0,
            num_rollout_steps=2, n_val_examples=64, eval_every=2, n_final_val_examples=64,
        ),
    ],
}
