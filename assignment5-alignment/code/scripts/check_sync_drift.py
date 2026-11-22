"""Diagnose whether repeated weight syncs change the vLLM model when the policy does not change.

Starts vLLM and the Hugging Face policy, then repeatedly syncs the (unchanged) policy weights
into vLLM.  After each round it greedily decodes a fixed set of GSM8K prompts with token
log-probabilities and reports
  - how many greedy continuations differ from the ones before any sync,
  - the mean |logprob(vLLM) - logprob(HF policy)| on the vLLM-generated tokens,
  - the max |w_vllm - w_policy| is not observable over HTTP, so the two above are the signal.

    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/check_sync_drift.py --rounds 10
"""

from __future__ import annotations

import argparse
import json

import torch

from cs336_alignment import gsm8k
from cs336_alignment.grpo import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.train_grpo import call_with_timeout
from cs336_alignment.vllm_utils import VLLMServer, _http_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=gsm8k.MODEL_ID)
    parser.add_argument("--policy-device", default="cuda:1")
    parser.add_argument("--vllm-gpu", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--port", type=int, default=8337)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--n-prompts", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out", default="experiments/logs/sync_drift.json")
    args = parser.parse_args()
    gsm8k.prepare_environment()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = gsm8k.subsample(gsm8k.load_gsm8k(gsm8k.REPO_ROOT / "data/gsm8k/test.jsonl"), args.n_prompts, seed=1)
    template = gsm8k.load_prompt_template("r1_zero")
    prompts = [gsm8k.format_prompt(template, ex["question"]) for ex in examples]

    server = VLLMServer(
        model_id=args.model, gpu=args.vllm_gpu, port=args.port, gpu_memory_utilization=args.gpu_memory_utilization
    )
    server.start()
    results = []
    try:
        policy = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.policy_device, dtype=torch.bfloat16)
        policy.train()  # same mode as in training
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        call_with_timeout(lambda: server.init_weight_sync(args.policy_device), args.timeout, "weight-sync init", server.stop)

        def decode():
            payload = {
                "model": args.model, "prompt": prompts, "temperature": 0.0, "max_tokens": args.max_tokens,
                "n": 1, "seed": 0, "logprobs": 0, "return_token_ids": True,
            }
            response = _http_json("POST", f"{server.base_url}/v1/completions", payload, timeout=600)
            choices = sorted(response["choices"], key=lambda c: c["index"])
            return [(c["text"], c["logprobs"]["token_logprobs"]) for c in choices]

        @torch.no_grad()
        def hf_logprobs(texts):
            t = tokenize_prompt_and_output(prompts, texts, tokenizer)
            out = get_response_log_probs(
                policy, t["input_ids"].to(args.policy_device), t["labels"].to(args.policy_device),
                response_mask=t["response_mask"].to(args.policy_device),
            )["log_probs"].float().cpu()
            mask = t["response_mask"]
            return [out[i][mask[i]].tolist() for i in range(len(texts))]

        def record(label: str, reference):
            decoded = decode()
            texts = [d[0] for d in decoded]
            hf = hf_logprobs(texts)
            diffs = []
            for (_, vllm_lp), hf_lp in zip(decoded, hf):
                n = min(len(vllm_lp), len(hf_lp))
                diffs += [abs(a - b) for a, b in zip(vllm_lp[:n], hf_lp[:n])]
            changed = sum(a != b for a, b in zip(texts, reference)) if reference is not None else 0
            row = {
                "label": label,
                "greedy_texts_changed_vs_initial": changed,
                "mean_abs_logprob_diff_vllm_vs_hf": sum(diffs) / len(diffs),
                "max_abs_logprob_diff_vllm_vs_hf": max(diffs),
                "sample": texts[0][:120],
            }
            print(json.dumps(row), flush=True)
            results.append(row)
            return texts

        initial = record("before any sync", None)
        for r in range(1, args.rounds + 1):
            call_with_timeout(lambda: server.sync_policy_weights(policy), args.timeout, "weight sync", server.stop)
            torch.cuda.synchronize(args.policy_device)
            record(f"after {r} sync(s) of unchanged weights", initial)
    finally:
        server.stop()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
