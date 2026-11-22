"""Measure importance ratios before any update, with the training code's own scoring paths.

Takes the first saved rollout batch of an off-policy run, computes old log-probs exactly as the
training loop does (compute_old_log_probs, scoring batches of 16) and current log-probs exactly as
grpo_train_step does (minibatches of 8 with per-minibatch padding), with the same, unchanged
weights.  Any deviation of the ratio from 1 is then purely numerical (bf16, different batch shapes).

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/check_offpolicy_ratios.py experiments/offpolicy_clip_s0/rollouts_step0.jsonl
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import gsm8k
from cs336_alignment.grpo import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.train_grpo import compute_old_log_probs

gsm8k.prepare_environment()
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8").read().split("\n") if line.strip()]
prompts = [r["prompt"] for r in rows]
responses = [r["response"] for r in rows]
model = AutoModelForCausalLM.from_pretrained(gsm8k.MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
model.train()
tokenizer = AutoTokenizer.from_pretrained(gsm8k.MODEL_ID)

old = compute_old_log_probs(model, tokenizer, prompts, responses, batch_size=16)
tokens = tokenize_prompt_and_output(prompts, responses, tokenizer)
token_log_ratios, seq_log_ratios = [], []
with torch.no_grad():
    for start in range(0, len(rows), 8):
        idx = list(range(start, start + 8))
        mask_full = tokens["response_mask"][idx]
        lengths = mask_full.shape[1] - mask_full.flip(1).int().argmax(dim=1)
        seq_len = int(lengths.max())
        new = get_response_log_probs(
            model,
            tokens["input_ids"][idx, :seq_len].cuda(),
            tokens["labels"][idx, :seq_len].cuda(),
            response_mask=mask_full[:, :seq_len].cuda(),
        )["log_probs"].float().cpu()
        mask = mask_full[:, :seq_len]
        lr = (new - old[idx, :seq_len]) * mask
        token_log_ratios.append(lr[mask])
        seq_log_ratios.append(lr.sum(-1) / mask.sum(-1).clamp_min(1))
t = torch.cat(token_log_ratios).abs()
s = torch.cat(seq_log_ratios).abs()
print(json.dumps({
    "n_sequences": len(rows),
    "n_tokens": int(t.numel()),
    "token |log ratio| mean": float(t.mean()), "token |log ratio| p99": float(t.quantile(0.99)), "token |log ratio| max": float(t.max()),
    "frac tokens with |ratio-1| > 0.2": float((t > 0.182).float().mean()),
    "sequence |mean log ratio| mean": float(s.mean()), "sequence |mean log ratio| max": float(s.max()),
    "frac sequences outside GSPO band 1 +- 3e-4": float((s > 3e-4).float().mean()),
}, indent=1))

# ---- matched scoring (the default now): old log-probs from the exact training shapes ----------
from cs336_alignment.grpo import compute_group_normalized_rewards, compute_rollout_rewards
from cs336_alignment.train_grpo import score_like_train_step

reward_fn = gsm8k.get_reward_fn("r1_zero")
raw, _ = compute_rollout_rewards(reward_fn, responses, [r["ground_truth"] for r in rows])
adv, _ = compute_group_normalized_rewards(raw, 8)
adv = torch.where(adv == 0, torch.ones_like(adv), adv)  # score every sequence for this check
seq_dev, tok_dev = [], []
for start in range(0, len(rows), 8):
    idx = list(range(start, start + 8))
    old_m = score_like_train_step(model, tokenizer, [prompts[i] for i in idx], [responses[i] for i in idx], adv[idx], 1)
    t = tokenize_prompt_and_output([prompts[i] for i in idx], [responses[i] for i in idx], tokenizer)
    mask = t["response_mask"]
    lengths = mask.shape[1] - mask.flip(1).int().argmax(dim=1)
    seq_len = int(lengths.max())
    with torch.enable_grad():  # the training forward runs with autograd on
        new = get_response_log_probs(
            model, t["input_ids"][:, :seq_len].cuda(), t["labels"][:, :seq_len].cuda(),
            return_token_entropy=True, response_mask=mask[:, :seq_len].cuda(),
        )["log_probs"].detach().float().cpu()
    m = mask[:, :seq_len]
    d = (new - old_m[:, :seq_len]) * m
    tok_dev.append(d[m].abs())
    seq_dev.append((d.sum(-1) / m.sum(-1).clamp_min(1)).abs())
tok_dev, seq_dev = torch.cat(tok_dev), torch.cat(seq_dev)
print(json.dumps({
    "matched: token |log ratio| max": float(tok_dev.max()),
    "matched: sequence |mean log ratio| max": float(seq_dev.max()),
    "matched: frac sequences outside GSPO band": float((seq_dev > 3e-4).float().mean()),
}, indent=1))
