"""Evaluate the base model on GSM8K with the three prompting strategies (problem prompting_baselines).

For every prompt this writes all generations with their rewards and a summary with the three
categories asked for in the handout:
  (1) format reward 1, answer reward 1
  (2) format reward 1, answer reward 0
  (3) format reward 0, answer reward 0

Example (vLLM on the first visible GPU):
    CUDA_VISIBLE_DEVICES=1 uv run --extra gpu python scripts/eval_gsm8k_baselines.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cs336_alignment import gsm8k
from cs336_alignment.vllm_utils import VLLMServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=gsm8k.MODEL_ID)
    parser.add_argument("--data", default=str(gsm8k.REPO_ROOT / "data/gsm8k/test.jsonl"))
    parser.add_argument("--out-dir", default=str(gsm8k.REPO_ROOT / "experiments/baselines"))
    parser.add_argument("--prompts", nargs="+", default=list(gsm8k.PROMPT_FILES))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--vllm-gpu", type=int, default=0, help="physical GPU index for the vLLM server")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--port", type=int, default=8336)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    gsm8k.prepare_environment()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = gsm8k.load_gsm8k(args.data)
    if args.max_examples:
        examples = examples[: args.max_examples]

    server = VLLMServer(
        model_id=args.model,
        gpu=args.vllm_gpu,
        port=args.port,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    server.start()
    summary = {}
    try:
        for prompt_name in args.prompts:
            template = gsm8k.load_prompt_template(prompt_name)
            reward_fn = gsm8k.get_reward_fn(prompt_name)
            prompts = [gsm8k.format_prompt(template, ex["question"]) for ex in examples]

            start = time.perf_counter()
            completions = gsm8k.generate_independent(
                server, prompts, gsm8k.sampling_params(prompt_name, seed=args.seed * 1_000_003)
            )
            elapsed = time.perf_counter() - start

            counts = {"correct": 0, "formatted_wrong": 0, "unformatted": 0}
            total_tokens = 0
            with open(out_dir / f"{prompt_name}.jsonl", "w", encoding="utf-8") as f:
                for ex, prompt, completion in zip(examples, prompts, completions):
                    reward = reward_fn(completion.text, ex["answer"])
                    if reward["format_reward"] == 1.0 and reward["answer_reward"] == 1.0:
                        category = "correct"
                    elif reward["format_reward"] == 1.0:
                        category = "formatted_wrong"
                    else:
                        category = "unformatted"
                    counts[category] += 1
                    total_tokens += len(completion.token_ids)
                    f.write(
                        json.dumps(
                            {
                                "question": ex["question"],
                                "ground_truth": ex["answer"],
                                "prompt": prompt,
                                "response": completion.text,
                                "finish_reason": completion.finish_reason,
                                "num_tokens": len(completion.token_ids),
                                "category": category,
                                **reward,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            n = len(examples)
            summary[prompt_name] = {
                "n": n,
                **counts,
                "accuracy": counts["correct"] / n,
                "format_rate": (counts["correct"] + counts["formatted_wrong"]) / n,
                "mean_response_tokens": total_tokens / n,
                "generation_seconds": round(elapsed, 1),
            }
            print(prompt_name, json.dumps(summary[prompt_name]))
    finally:
        server.stop()

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
