"""GRPO and its variants: tokenization, scoring, advantages, losses and the train step.

Every on-policy variant used in the assignment is a combination of three switches

    baseline              "mean" | "none"
    advantage_normalizer  "std" | "none" | "mean"   (+ extra estimators, see below)
    loss_normalization    "sequence" | "constant"

and every off-policy variant adds an importance-reweighting method
("none" | "noclip" | "grpo" | "gspo").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

Baseline = Literal["mean", "none"]
# "sqrt_mean" is our own estimator for the try_your_own problem (see the writeup).
AdvantageNormalizer = Literal["std", "none", "mean", "sqrt_mean"]
ImportanceMethod = Literal["none", "noclip", "grpo", "gspo"]
LossNormalization = Literal["sequence", "constant"]


def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer) -> dict[str, torch.Tensor]:
    """Tokenize prompts and outputs separately, concatenate, and build shifted inputs/labels.

    ``response_mask`` is aligned with ``labels`` and is True exactly where the label is a response token.
    """
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs]
    output_ids = [tokenizer.encode(o, add_special_tokens=False) for o in output_strs]
    lengths = [len(p) + len(o) for p, o in zip(prompt_ids, output_ids)]
    max_len = max(lengths)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    full = torch.full((len(prompt_strs), max_len), pad_id, dtype=torch.long)
    is_response = torch.zeros((len(prompt_strs), max_len), dtype=torch.bool)
    for i, (p, o) in enumerate(zip(prompt_ids, output_ids)):
        full[i, : len(p) + len(o)] = torch.tensor(p + o, dtype=torch.long)
        is_response[i, len(p) : len(p) + len(o)] = True

    return {"input_ids": full[:, :-1], "labels": full[:, 1:], "response_mask": is_response[:, 1:]}


def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    response_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-token log p(label_t | prefix) and, optionally, the entropy of each next-token distribution.

    If ``response_mask`` is given, the output layer is only applied at the masked positions and
    the other entries of the returned tensors are zero.  The values at the masked positions are
    the same as without the mask; skipping the rest avoids materialising vocabulary-sized logits
    for prompt and padding tokens, which is most of the memory when the prompt is long.
    """
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if response_mask is not None and head is not None and getattr(model, "base_model", model) is not model:
        mask = response_mask.bool()
        hidden = model.base_model(input_ids)[0]
        logits = head(hidden[mask]).float()  # (num_response_tokens, vocab)
        selected = -F.cross_entropy(logits, labels[mask], reduction="none")
        log_probs = torch.zeros(labels.shape, dtype=selected.dtype, device=selected.device).masked_scatter(mask, selected)
        out = {"log_probs": log_probs}
        if return_token_entropy:
            with torch.no_grad():
                logp = torch.log_softmax(logits.detach(), dim=-1)
                entropy = -(logp.exp() * logp).sum(dim=-1)
                out["token_entropy"] = torch.zeros(labels.shape, dtype=entropy.dtype, device=entropy.device).masked_scatter(
                    mask, entropy
                )
        return out

    logits = model(input_ids).logits.float()
    # cross_entropy only keeps the logits for the backward pass, which is much lighter
    # than materialising (and retaining) a full log-softmax over a ~100k vocabulary.
    log_probs = -F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    out = {"log_probs": log_probs}
    if return_token_entropy:
        with torch.no_grad():
            logp = torch.log_softmax(logits.detach(), dim=-1)
            out["token_entropy"] = -(logp.exp() * logp).sum(dim=-1)
    return out


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    results = [reward_fn(resp, gt) for resp, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([r["reward"] for r in results], dtype=torch.float32)
    n = max(len(results), 1)
    metadata = {
        "mean_reward": float(raw_rewards.mean()) if results else 0.0,
        "mean_format_reward": sum(r.get("format_reward", 0.0) for r in results) / n,
        "mean_answer_reward": sum(r.get("answer_reward", 0.0) for r in results) / n,
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Baseline = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: AdvantageNormalizer = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Turn flat rewards (n_prompts * group_size,) into advantages using per-group statistics."""
    grouped = raw_rewards.reshape(-1, group_size).float()
    mean = grouped.mean(dim=-1, keepdim=True)

    if baseline == "mean":
        advantages = grouped - mean
    elif baseline == "none":
        advantages = grouped.clone()
    else:
        raise ValueError(f"unknown baseline {baseline!r}")

    if advantage_normalizer == "std":
        advantages = advantages / (grouped.std(dim=-1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (mean + advantage_eps)
    elif advantage_normalizer == "sqrt_mean":
        advantages = advantages / (mean.clamp_min(0).sqrt() + advantage_eps)
    elif advantage_normalizer != "none":
        raise ValueError(f"unknown advantage_normalizer {advantage_normalizer!r}")

    advantages = advantages.reshape(-1)
    metadata = {
        "reward_mean": float(raw_rewards.float().mean()),
        "reward_std": float(raw_rewards.float().std()) if raw_rewards.numel() > 1 else 0.0,
        "advantage_abs_mean": float(advantages.abs().mean()),
        "advantage_max": float(advantages.max()),
        "advantage_min": float(advantages.min()),
        "frac_zero_advantage": float((advantages == 0).float().mean()),
        "frac_groups_all_same": float((grouped.max(-1).values == grouped.min(-1).values).float().mean()),
    }
    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: ImportanceMethod = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token loss (the negative objective), shape (batch, seq)."""
    adv = raw_rewards_or_advantages.reshape(-1, 1).to(policy_log_probs.device, policy_log_probs.dtype)
    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        return -adv * policy_log_probs, metadata

    if old_log_probs is None:
        raise ValueError("old_log_probs is required for importance reweighting")
    old_log_probs = old_log_probs.to(policy_log_probs.device, policy_log_probs.dtype)
    log_ratio = policy_log_probs - old_log_probs

    if importance_reweighting_method == "gspo":
        # Geometric mean of the token ratios = exp(mean log-ratio over response tokens).
        if response_mask is None:
            mean_log_ratio = log_ratio.mean(dim=-1, keepdim=True)
        else:
            mask = response_mask.to(log_ratio.dtype)
            mean_log_ratio = (log_ratio * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1)
        ratio = torch.exp(mean_log_ratio).expand_as(policy_log_probs)
    else:
        ratio = torch.exp(log_ratio)

    if importance_reweighting_method == "noclip":
        metadata["ratio_mean"] = ratio.detach().mean()
        return -adv * ratio, metadata

    if importance_reweighting_method not in ("grpo", "gspo"):
        raise ValueError(f"unknown importance_reweighting_method {importance_reweighting_method!r}")
    if cliprange is None:
        raise ValueError("cliprange is required for clipped importance reweighting")

    unclipped = adv * ratio
    clipped = adv * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    loss = -torch.minimum(unclipped, clipped)
    # A position is "clipped" when the clipped branch is active, i.e. it contributes no gradient.
    is_clipped = (clipped < unclipped).detach()
    metadata["is_clipped"] = is_clipped
    metadata["ratio_mean"] = ratio.detach().mean()
    return loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: LossNormalization = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    mask = mask.to(per_token_policy_gradient_loss.dtype)
    masked = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        per_sequence = masked.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
        return per_sequence.mean()
    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError("normalization_constant is required for constant normalization")
        return masked.sum() / normalization_constant
    raise ValueError(f"unknown loss_normalization {loss_normalization!r}")


def grpo_train_step(
    model,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Baseline = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: AdvantageNormalizer = "std",
    importance_reweighting_method: ImportanceMethod = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: LossNormalization = "sequence",
    normalization_constant: int | None = None,
    advantages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """One optimizer step on a batch of rollouts, accumulating gradients over microbatches.

    Sequences whose advantage is exactly zero contribute nothing to the gradient, so they are
    never passed through the model.  Normalisation still uses the size of the *full* batch, so
    the result is identical to the unpruned computation.

    ``advantages`` may be given to override the group computation: the off-policy loop computes
    advantages on the whole rollout batch and then trains on small slices of it.
    """
    batch_size = len(rollout_responses)
    raw_rewards, reward_meta = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    if advantages is None:
        advantages, adv_meta = compute_group_normalized_rewards(
            raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
        )
    else:
        advantages = advantages.detach().float().cpu().reshape(-1)
        adv_meta = {"frac_zero_advantage": float((advantages == 0).float().mean())}

    microbatch_size = max(1, batch_size // gradient_accumulation_steps)
    keep = torch.nonzero(advantages != 0).flatten().tolist()
    device = next(model.parameters()).device

    tokens = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    lengths = tokens["response_mask"].shape[1] - tokens["response_mask"].flip(1).int().argmax(dim=1)
    lengths = torch.where(tokens["response_mask"].any(dim=1), lengths, torch.ones_like(lengths))

    total_loss = torch.zeros((), device=device)
    entropy_sum, token_count = 0.0, 0.0
    clipped_tokens = 0.0
    ratio_sum, ratio_count = 0.0, 0

    for start in range(0, len(keep), microbatch_size):
        idx = torch.tensor(keep[start : start + microbatch_size])
        seq_len = int(lengths[idx].max())  # drop columns that are padding for this microbatch
        input_ids = tokens["input_ids"][idx, :seq_len].to(device)
        labels = tokens["labels"][idx, :seq_len].to(device)
        mask = tokens["response_mask"][idx, :seq_len].to(device)

        scored = get_response_log_probs(model, input_ids, labels, return_token_entropy=True, response_mask=mask)
        old = old_log_probs[idx, :seq_len] if old_log_probs is not None else None
        per_token_loss, loss_meta = compute_policy_gradient_loss(
            advantages[idx],
            scored["log_probs"],
            importance_reweighting_method,
            old_log_probs=old,
            cliprange=cliprange,
            response_mask=mask,
        )
        if loss_normalization == "sequence":
            # Mean over this microbatch's sequences, reweighted to a mean over the full batch.
            loss = aggregate_loss_across_microbatch(per_token_loss, mask, "sequence") * (len(idx) / batch_size)
        else:
            loss = aggregate_loss_across_microbatch(per_token_loss, mask, "constant", normalization_constant)
        loss.backward()

        total_loss = total_loss + loss.detach()
        mask_f = mask.float()
        entropy_sum += float((scored["token_entropy"] * mask_f).sum())
        token_count += float(mask_f.sum())
        if "is_clipped" in loss_meta:
            clipped_tokens += float((loss_meta["is_clipped"].float() * mask_f).sum())
        if "ratio_mean" in loss_meta:
            ratio_sum += float(loss_meta["ratio_mean"])
            ratio_count += 1

    params = [p for p in model.parameters() if p.grad is not None]
    if params:
        grad_norm = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in params))
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
    else:  # every sequence had zero advantage: nothing to learn from this batch
        grad_norm = torch.zeros(())
    optimizer.zero_grad(set_to_none=True)

    grouped = raw_rewards.reshape(-1, group_size)
    metadata: dict[str, torch.Tensor | float] = {
        "loss": float(total_loss),
        "grad_norm": float(grad_norm),
        "token_entropy": entropy_sum / token_count if token_count else float("nan"),
        "train_reward": reward_meta["mean_reward"],
        "train_format_reward": reward_meta["mean_format_reward"],
        "frac_zero_advantage": adv_meta["frac_zero_advantage"],
        "num_sequences_trained": float(len(keep)),
        "num_response_tokens_trained": token_count,
        "pass_at_k": float((grouped.max(dim=-1).values > 0).float().mean()),
    }
    if importance_reweighting_method in ("grpo", "gspo"):
        metadata["clip_fraction"] = clipped_tokens / token_count if token_count else 0.0
    if ratio_count:
        metadata["importance_ratio_mean"] = ratio_sum / ratio_count
    return total_loss, metadata
