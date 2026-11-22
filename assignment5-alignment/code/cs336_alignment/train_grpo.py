"""GRPO training loop for OLMo-2-0425-1B on GSM8K (on-policy and off-policy variants).

The Hugging Face policy lives on one GPU and a vLLM server on another; before each batch of
rollouts the policy weights are pushed into vLLM over NCCL.

Example:
    CUDA_VISIBLE_DEVICES=0,1 uv run --extra gpu python -m cs336_alignment.train_grpo \
        --run-name grpo_s0 --variant grpo --seed 0

Everything is logged locally under experiments/<run-name>/:
    config.json      the resolved configuration
    metrics.jsonl    one record per rollout step ("kind": "train") and per evaluation ("kind": "eval")
    rollouts_*.jsonl sampled training rollouts / validation generations
    final.json       written last; its presence marks the run as finished
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_alignment import gsm8k
from cs336_alignment.grpo import (
    compute_group_normalized_rewards,
    compute_rollout_rewards,
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.vllm_utils import VLLMServer

# name -> (baseline, advantage_normalizer, loss_normalization, importance method, off_policy)
VARIANTS: dict[str, dict] = {
    "grpo": dict(baseline="mean", advantage_normalizer="std", loss_normalization="sequence"),
    "grpo_constant": dict(baseline="mean", advantage_normalizer="std", loss_normalization="constant"),
    "dr_grpo": dict(baseline="mean", advantage_normalizer="none", loss_normalization="constant"),
    "rft": dict(baseline="none", advantage_normalizer="none", loss_normalization="constant"),
    "maxrl": dict(baseline="mean", advantage_normalizer="mean", loss_normalization="constant"),
    # try_your_own: MaxRL with the group mean replaced by its square root.
    "sqrt_maxrl": dict(baseline="mean", advantage_normalizer="sqrt_mean", loss_normalization="constant"),
    "offpolicy_naive": dict(off_policy=True, importance_reweighting_method="none"),
    "offpolicy_noclip": dict(off_policy=True, importance_reweighting_method="noclip"),
    "offpolicy_clip": dict(off_policy=True, importance_reweighting_method="grpo", cliprange=0.2),
    "offpolicy_gspo": dict(off_policy=True, importance_reweighting_method="gspo", cliprange=3e-4),
}


@dataclass
class Config:
    run_name: str
    variant: str = "grpo"
    model: str = gsm8k.MODEL_ID
    prompt: str = "r1_zero"
    train_path: str = str(gsm8k.REPO_ROOT / "data/gsm8k/train.jsonl")
    val_path: str = str(gsm8k.REPO_ROOT / "data/gsm8k/test.jsonl")
    out_root: str = str(gsm8k.REPO_ROOT / "experiments")
    seed: int = 0
    # data / schedule
    n_train_examples: int = 6400
    n_val_examples: int = 1024
    n_final_val_examples: int = 1024
    num_rollout_steps: int = 200
    eval_every: int = 10
    log_rollouts_every: int = 40
    # optimisation
    learning_rate: float = 1e-5
    rollout_batch_size: int = 256
    train_batch_size: int = 256
    group_size: int = 8
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    # sampling
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 512
    # estimator
    baseline: str = "mean"
    advantage_normalizer: str = "std"
    loss_normalization: str = "sequence"
    importance_reweighting_method: str = "none"
    cliprange: float | None = None
    off_policy: bool = False
    advantage_eps: float = 1e-6
    # infrastructure (indices are positions within CUDA_VISIBLE_DEVICES for the policy,
    # and a physical index for the vLLM subprocess, as in the provided VLLMServer)
    policy_device: str = "cuda:1"
    vllm_gpu: int = 0
    vllm_gpu_memory_utilization: float = 0.35
    vllm_port: int = 8336
    attn_implementation: str = "sdpa"
    scoring_batch_size: int = 16
    # Off-policy minibatches: "grouped" keeps each prompt's group together in one minibatch, so the
    # advantages of every optimizer step sum to zero within each group.  "shuffled" mixes sequences of
    # different prompts; it makes off-policy training collapse (see the report) and is kept for that ablation.
    minibatch_order: str = "grouped"
    # "matched": old log-probs are computed with exactly the minibatches/padding the training forward
    # uses, so the ratio is exactly 1 before the first update.  "separate": scored in batches of
    # scoring_batch_size (used by the shuffled-minibatch ablation runs; bf16 noise of ~1e-3 per sequence).
    old_logprob_scoring: str = "matched"
    sync_timeout_seconds: float = 300.0
    save_model: bool = False


def parse_config() -> Config:
    parser = argparse.ArgumentParser()
    defaults = Config(run_name="unset")
    for name, value in asdict(defaults).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif name == "cliprange":
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=type(value), default=None, required=(name == "run_name"))
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}

    variant = args.get("variant", "grpo")
    cfg = asdict(defaults)
    cfg.update(VARIANTS[variant])
    if cfg["off_policy"] and "train_batch_size" not in args:
        # 32x off-policy: 32 optimizer steps of 8 sequences per rollout batch.
        cfg["train_batch_size"] = 8
        cfg["gradient_accumulation_steps"] = 1
    cfg.update(args)  # explicit flags win over the variant preset
    return Config(**cfg)


def load_policy(cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Same as cs336_alignment.checkpoint.get_model_and_tokenizer, except that the attention
    # implementation is configurable: the flash-attn wheel pinned by the course needs
    # glibc >= 2.32, which the machine used here (glibc 2.31) does not have.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        device_map=cfg.policy_device,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    return model, tokenizer


def call_with_timeout(fn, seconds: float, what: str, on_timeout=None):
    """Run ``fn()`` and abort the whole process if it does not return within ``seconds``.

    NCCL calls block forever when the two ends fail to rendezvous, and a blocked native call
    cannot be interrupted from Python, so on timeout we clean up and hard-exit.  The run queue
    then records the run as failed instead of waiting on it indefinitely.
    """
    result: dict = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        print(f"[fatal] {what} did not finish within {seconds:.0f}s; aborting the run", flush=True)
        import faulthandler
        import sys

        faulthandler.dump_traceback(file=sys.stdout, all_threads=True)  # shows where it is stuck
        sys.stdout.flush()
        if on_timeout is not None:
            on_timeout()
        os._exit(3)
    if "error" in result:
        raise result["error"]
    return result.get("value")


class JsonlLogger:
    def __init__(self, path: Path):
        self.file = open(path, "a", encoding="utf-8")

    def log(self, record: dict) -> None:
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()


def evaluate(server: VLLMServer, cfg: Config, template: str, reward_fn, examples: list[dict], seed: int):
    prompts = [gsm8k.format_prompt(template, ex["question"]) for ex in examples]
    params = gsm8k.sampling_params(cfg.prompt, seed, 1, cfg.sampling_temperature, cfg.sampling_max_tokens)
    completions = gsm8k.generate_independent(server, prompts, params)
    rewards = [reward_fn(c.text, ex["answer"]) for c, ex in zip(completions, examples)]
    n = len(examples)
    metrics = {
        "val_reward": sum(r["reward"] for r in rewards) / n,
        "val_format_reward": sum(r["format_reward"] for r in rewards) / n,
        "val_mean_response_tokens": sum(len(c.token_ids) for c in completions) / n,
        "val_truncated_frac": sum(c.finish_reason == "length" for c in completions) / n,
        "val_n": n,
    }
    generations = [
        {"question": ex["question"], "ground_truth": ex["answer"], "response": c.text, **r}
        for ex, c, r in zip(examples, completions, rewards)
    ]
    return metrics, generations


@torch.no_grad()
def compute_old_log_probs(model, tokenizer, prompts: list[str], responses: list[str], batch_size: int) -> torch.Tensor:
    """Per-token log-probs of the rollouts under the sampling policy, padded like the full batch."""
    tokens = tokenize_prompt_and_output(prompts, responses, tokenizer)
    device = next(model.parameters()).device
    out = torch.zeros(tokens["input_ids"].shape, dtype=torch.float32)
    lengths = tokens["response_mask"].shape[1] - tokens["response_mask"].flip(1).int().argmax(dim=1)
    for start in range(0, len(prompts), batch_size):
        rows = slice(start, start + batch_size)
        seq_len = max(int(lengths[rows].max()), 1)
        scored = get_response_log_probs(
            model,
            tokens["input_ids"][rows, :seq_len].to(device),
            tokens["labels"][rows, :seq_len].to(device),
            response_mask=tokens["response_mask"][rows, :seq_len].to(device),
        )
        out[rows, :seq_len] = scored["log_probs"].float().cpu()
    return out


def main() -> None:
    cfg = parse_config()
    gsm8k.prepare_environment()
    out_dir = Path(cfg.out_root) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    logger = JsonlLogger(out_dir / "metrics.jsonl")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    # The subsets are the same for every run (fixed seed 0), only the sampling order differs.
    train_examples = gsm8k.subsample(gsm8k.load_gsm8k(cfg.train_path), cfg.n_train_examples, seed=0)
    val_all = gsm8k.subsample(gsm8k.load_gsm8k(cfg.val_path), None, seed=0)
    val_examples = val_all[: cfg.n_val_examples]
    final_val_examples = val_all[: cfg.n_final_val_examples]
    template = gsm8k.load_prompt_template(cfg.prompt)
    reward_fn = gsm8k.get_reward_fn(cfg.prompt)

    server = VLLMServer(
        model_id=cfg.model,
        gpu=cfg.vllm_gpu,
        port=cfg.vllm_port,
        seed=cfg.seed,
        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
    )
    server.start()
    if server.process is not None:  # lets the run queue clean up the server if this process dies
        (out_dir / "vllm.pid").write_text(str(server.process.pid))
    policy, tokenizer = load_policy(cfg)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
    call_with_timeout(
        lambda: server.init_weight_sync(cfg.policy_device), cfg.sync_timeout_seconds, "NCCL weight-sync init", server.stop
    )
    print("[init] weight-sync group established", flush=True)

    probe = max((p for p in policy.parameters() if p.ndim == 2 and p.numel() < 2**23), key=lambda p: p.numel())
    n_prompts = cfg.rollout_batch_size // cfg.group_size
    normalization_constant = cfg.train_batch_size * cfg.sampling_max_tokens  # Z = B*G*L
    train_kwargs = dict(
        model=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        max_grad_norm=cfg.max_grad_norm,
        reward_fn=reward_fn,
        group_size=cfg.group_size,
        baseline=cfg.baseline,
        advantage_eps=cfg.advantage_eps,
        advantage_normalizer=cfg.advantage_normalizer,
        importance_reweighting_method=cfg.importance_reweighting_method,
        cliprange=cfg.cliprange,
        loss_normalization=cfg.loss_normalization,
        normalization_constant=normalization_constant,
    )

    run_start = time.perf_counter()
    try:
        for step in range(cfg.num_rollout_steps + 1):
            step_start = time.perf_counter()
            call_with_timeout(
                lambda: server.sync_policy_weights(policy), cfg.sync_timeout_seconds, "weight sync", server.stop
            )
            sync_seconds = time.perf_counter() - step_start

            is_last = step == cfg.num_rollout_steps
            if step % cfg.eval_every == 0 or is_last:
                examples = final_val_examples if is_last else val_examples
                t0 = time.perf_counter()
                metrics, generations = evaluate(server, cfg, template, reward_fn, examples, seed=500_000_000 + step * 10_000)
                logger.log(
                    {
                        "kind": "eval",
                        "step": step,
                        "final": is_last,
                        "elapsed_seconds": time.perf_counter() - run_start,
                        "eval_seconds": time.perf_counter() - t0,
                        **metrics,
                    }
                )
                print(f"[eval] step {step}: {metrics}", flush=True)
                if step == 0 or is_last:
                    with open(out_dir / f"val_generations_step{step}.jsonl", "w", encoding="utf-8") as f:
                        for g in generations[:200]:
                            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            if is_last:
                break

            # ---- rollouts -------------------------------------------------------------
            t0 = time.perf_counter()
            batch = rng.sample(train_examples, n_prompts)
            prompts = [gsm8k.format_prompt(template, ex["question"]) for ex in batch]
            params = gsm8k.sampling_params(
                cfg.prompt,
                seed=cfg.seed * 1_000_003 + step * 1_000,
                n=cfg.group_size,
                temperature=cfg.sampling_temperature,
                max_tokens=cfg.sampling_max_tokens,
            )
            completions = gsm8k.generate_independent(server, prompts, params)
            responses = [c.text for c in completions]
            repeated_prompts = [p for p in prompts for _ in range(cfg.group_size)]
            repeated_truths = [ex["answer"] for ex in batch for _ in range(cfg.group_size)]
            rollout_seconds = time.perf_counter() - t0

            # ---- policy update --------------------------------------------------------
            t0 = time.perf_counter()
            probe_before = probe.detach().clone()
            if not cfg.off_policy:
                _, meta = grpo_train_step(
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    repeated_prompts=repeated_prompts,
                    rollout_responses=responses,
                    repeated_ground_truths=repeated_truths,
                    **train_kwargs,
                )
                meta["optimizer_steps"] = 1.0
            else:
                meta = off_policy_update(cfg, train_kwargs, repeated_prompts, responses, repeated_truths, rng)
            train_seconds = time.perf_counter() - t0
            # Fraction of entries of one weight matrix that the update actually changed (bfloat16
            # weights cannot represent an update much smaller than their own magnitude).
            meta["frac_probe_weights_changed"] = float((probe.detach() != probe_before).float().mean())

            record = {
                "kind": "train",
                "step": step,
                "elapsed_seconds": time.perf_counter() - run_start,
                "sync_seconds": sync_seconds,
                "rollout_seconds": rollout_seconds,
                "train_seconds": train_seconds,
                "train_mean_response_tokens": sum(len(c.token_ids) for c in completions) / len(completions),
                "policy_gpu_peak_gb": torch.cuda.max_memory_allocated(cfg.policy_device) / 2**30,
                **{k: float(v) for k, v in meta.items()},
            }
            logger.log(record)
            print(
                f"[train] step {step}: reward={record['train_reward']:.3f} format={record['train_format_reward']:.3f} "
                f"loss={record['loss']:.4g} gnorm={record['grad_norm']:.3g} ent={record['token_entropy']:.3f} "
                f"rollout={rollout_seconds:.0f}s train={train_seconds:.0f}s",
                flush=True,
            )

            if step % cfg.log_rollouts_every == 0 or step == cfg.num_rollout_steps - 1:
                rewards, _ = compute_rollout_rewards(reward_fn, responses, repeated_truths)
                with open(out_dir / f"rollouts_step{step}.jsonl", "w", encoding="utf-8") as f:
                    for p, r, gt, rew in zip(repeated_prompts, responses, repeated_truths, rewards.tolist()):
                        f.write(
                            json.dumps({"prompt": p, "response": r, "ground_truth": gt, "reward": rew}, ensure_ascii=False)
                            + "\n"
                        )

        if cfg.save_model:
            policy.save_pretrained(out_dir / "model")
            tokenizer.save_pretrained(out_dir / "model")
    finally:
        server.stop()

    (out_dir / "final.json").write_text(
        json.dumps({"finished": True, "total_seconds": time.perf_counter() - run_start}, indent=2)
    )


@torch.no_grad()
def score_like_train_step(
    model, tokenizer, prompts: list[str], responses: list[str], advantages: torch.Tensor, gradient_accumulation_steps: int
) -> torch.Tensor:
    """Log-probs computed with exactly the tensors grpo_train_step will feed the model.

    grpo_train_step tokenizes the minibatch, drops zero-advantage sequences and runs microbatches
    truncated to their longest response.  Reproducing those shapes here makes the old and new
    forward passes identical before any update, so the importance ratio starts at exactly 1.
    """
    tokens = tokenize_prompt_and_output(prompts, responses, tokenizer)
    out = torch.zeros(tokens["input_ids"].shape, dtype=torch.float32)
    lengths = tokens["response_mask"].shape[1] - tokens["response_mask"].flip(1).int().argmax(dim=1)
    lengths = torch.where(tokens["response_mask"].any(dim=1), lengths, torch.ones_like(lengths))
    keep = torch.nonzero(advantages.reshape(-1) != 0).flatten().tolist()
    microbatch_size = max(1, len(prompts) // gradient_accumulation_steps)
    device = next(model.parameters()).device
    for start in range(0, len(keep), microbatch_size):
        idx = torch.tensor(keep[start : start + microbatch_size])
        seq_len = int(lengths[idx].max())
        scored = get_response_log_probs(
            model,
            tokens["input_ids"][idx, :seq_len].to(device),
            tokens["labels"][idx, :seq_len].to(device),
            return_token_entropy=True,
            response_mask=tokens["response_mask"][idx, :seq_len].to(device),
        )
        out[idx, :seq_len] = scored["log_probs"].float().cpu()
    return out


def off_policy_update(cfg: Config, train_kwargs: dict, prompts: list[str], responses: list[str], truths: list[str], rng):
    """Several optimizer steps on one rollout batch.

    Advantages use group statistics, so they are computed once on the whole rollout batch; the
    batch is then shuffled and split into minibatches of ``train_batch_size`` sequences.
    """
    model, tokenizer = train_kwargs["model"], train_kwargs["tokenizer"]
    raw_rewards, _ = compute_rollout_rewards(train_kwargs["reward_fn"], responses, truths)
    advantages, _ = compute_group_normalized_rewards(
        raw_rewards, cfg.group_size, cfg.baseline, cfg.advantage_eps, cfg.advantage_normalizer
    )
    if cfg.minibatch_order == "grouped":
        groups = list(range(len(responses) // cfg.group_size))
        rng.shuffle(groups)
        order = [g * cfg.group_size + j for g in groups for j in range(cfg.group_size)]
    elif cfg.minibatch_order == "shuffled":
        order = list(range(len(responses)))
        rng.shuffle(order)
    else:
        raise ValueError(f"unknown minibatch_order {cfg.minibatch_order!r}")
    minibatches = [order[i : i + cfg.train_batch_size] for i in range(0, len(order), cfg.train_batch_size)]

    old_log_probs = None
    matched_old: list[torch.Tensor | None] = [None] * len(minibatches)
    if cfg.importance_reweighting_method != "none":
        if cfg.old_logprob_scoring == "matched":
            for k, idx in enumerate(minibatches):
                matched_old[k] = score_like_train_step(
                    model, tokenizer, [prompts[i] for i in idx], [responses[i] for i in idx],
                    advantages[idx], cfg.gradient_accumulation_steps,
                )
        else:
            old_log_probs = compute_old_log_probs(model, tokenizer, prompts, responses, cfg.scoring_batch_size)
    sums: dict[str, float] = {}
    weights: dict[str, float] = {}
    steps_taken = 0
    for k, idx in enumerate(minibatches):
        old = matched_old[k] if matched_old[k] is not None else (old_log_probs[idx] if old_log_probs is not None else None)
        _, meta = grpo_train_step(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            repeated_prompts=[prompts[i] for i in idx],
            rollout_responses=[responses[i] for i in idx],
            repeated_ground_truths=[truths[i] for i in idx],
            old_log_probs=old,
            advantages=advantages[idx],
            **{**train_kwargs, "group_size": 1},
        )
        trained = meta["num_sequences_trained"] > 0
        steps_taken += int(trained)
        for key, value in meta.items():
            value = float(value)
            if value != value:  # NaN (e.g. entropy of an all-pruned minibatch)
                continue
            # Token-level statistics only exist for minibatches that were actually trained on.
            if key in ("token_entropy", "clip_fraction", "importance_ratio_mean", "grad_norm", "loss") and not trained:
                continue
            sums[key] = sums.get(key, 0.0) + value
            weights[key] = weights.get(key, 0.0) + 1.0
    out = {k: sums[k] / weights[k] for k in sums}
    for key in ("num_sequences_trained", "num_response_tokens_trained"):
        out[key] = sums.get(key, 0.0)
    out["train_reward"] = float(raw_rewards.mean())
    out["pass_at_k"] = float((raw_rewards.reshape(-1, cfg.group_size).max(-1).values > 0).float().mean())
    out["frac_zero_advantage"] = float((advantages == 0).float().mean())
    out["optimizer_steps"] = float(steps_taken)
    out.setdefault("token_entropy", float("nan"))
    out.setdefault("loss", 0.0)
    out.setdefault("grad_norm", 0.0)
    return out


if __name__ == "__main__":
    main()
