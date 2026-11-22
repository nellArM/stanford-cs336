"""GSM8K data, prompt templates and reward functions shared by the evaluation and RL scripts."""

from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MODEL_ID = "allenai/OLMo-2-0425-1B"

PROMPT_FILES = {
    "question_only": "question_only.prompt",
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}


def load_gsm8k(path: str | Path) -> list[dict[str, str]]:
    """Return a list of {"question", "answer"} where answer is just the final number."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append({"question": row["question"], "answer": row["answer"].split("####")[-1].strip()})
    return examples


def subsample(examples: list[dict], n: int | None, seed: int = 0) -> list[dict]:
    """Deterministic subset: the first ``n`` items of a seeded shuffle (prefixes are nested)."""
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    if n is not None:
        order = order[:n]
    return [examples[i] for i in order]


def load_prompt_template(prompt_name: str) -> str:
    return (PROMPT_DIR / PROMPT_FILES[prompt_name]).read_text(encoding="utf-8")


def format_prompt(template: str, question: str) -> str:
    # The templates use str.format syntax ({question}, and {{}} for literal braces).
    return template.format(question=question)


def get_reward_fn(prompt_name: str) -> Callable[[str, str], dict[str, float]]:
    return question_only_reward_fn if prompt_name == "question_only" else r1_zero_reward_fn


def sampling_params(prompt_name: str, seed: int, n: int = 1, temperature: float = 1.0, max_tokens: int = 512) -> dict:
    params = {"temperature": temperature, "max_tokens": max_tokens, "n": n, "seed": seed}
    if prompt_name != "question_only":
        # Stop as soon as the model closes its answer (as in Dr. GRPO).
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def prepare_environment() -> None:
    """Make the `vllm` executable of the current interpreter's environment visible to subprocesses.

    Also drops LD_LIBRARY_PATH for child processes: a system-wide CUDA toolkit on that path would
    shadow the CUDA libraries bundled with the torch wheels.
    """
    bin_dir = os.path.dirname(sys.executable)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ.pop("LD_LIBRARY_PATH", None)
    # On the machine used here NCCL's direct GPU-to-GPU (P2P) transport between two processes
    # hangs in the very first collective (the communicator initialises, then the warm-up
    # all-reduce never completes).  The shared-memory transport works, so use it unless the
    # caller has decided otherwise.  This must be set before either process creates its communicator.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")


def generate_independent(server, prompts: list[str], params: dict, max_workers: int = 64) -> list:
    """Sample completions with a different seed for every prompt.

    vLLM applies the request seed to each prompt of a batched request, so prompts sent together
    share one random stream and their samples are visibly correlated (the same unlikely tokens
    show up across unrelated questions).  Sending one request per prompt with seed ``seed + i``
    keeps runs reproducible while making the samples independent across prompts.  With ``n > 1``
    the result is flat, prompt by prompt, like ``VLLMServer.generate_completions``.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        index, prompt = item
        return server.generate_completions([prompt], {**params, "seed": params["seed"] + index})

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(one, enumerate(prompts)))
    return [completion for group in results for completion in group]
