"""Check that syncing policy weights really changes what the vLLM server generates.

1. Greedy-decode a prompt with the freshly started server            -> text A
2. Corrupt the policy's output layer, sync, decode again            -> text B, must differ from A
3. Restore the original weights, sync, decode again                 -> text C, must equal A

Needs little memory (no gradients): about 4 GB for the policy and whatever vLLM is given.

    CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python scripts/check_weight_sync.py --vllm-gpu 0 --policy-device cuda:1
"""

from __future__ import annotations

import argparse
import sys

import torch

from cs336_alignment import gsm8k
from cs336_alignment.train_grpo import call_with_timeout
from cs336_alignment.vllm_utils import VLLMServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=gsm8k.MODEL_ID)
    parser.add_argument("--policy-device", default="cuda:1")
    parser.add_argument("--vllm-gpu", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--port", type=int, default=8336)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    gsm8k.prepare_environment()

    from transformers import AutoModelForCausalLM

    server = VLLMServer(
        model_id=args.model, gpu=args.vllm_gpu, port=args.port, gpu_memory_utilization=args.gpu_memory_utilization
    )
    server.start()
    try:
        policy = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.policy_device, dtype=torch.bfloat16)
        call_with_timeout(lambda: server.init_weight_sync(args.policy_device), args.timeout, "weight-sync init", server.stop)

        template = gsm8k.load_prompt_template("r1_zero_three_shot")
        prompt = gsm8k.format_prompt(template, "Tom has 3 boxes with 12 pencils each. He gives away 7 pencils. How many are left?")
        params = {"temperature": 0.0, "max_tokens": 48, "n": 1, "seed": 0}

        def decode() -> str:
            return server.generate_completions([prompt], params)[0].text

        def sync() -> None:
            call_with_timeout(lambda: server.sync_policy_weights(policy), args.timeout, "weight sync", server.stop)

        text_a = decode()
        head = policy.get_output_embeddings().weight
        original = head.detach().clone()
        with torch.no_grad():
            head.copy_(original.flip(0))  # permute the vocabulary: output must become garbage
        sync()
        text_b = decode()
        with torch.no_grad():
            head.copy_(original)
        sync()
        text_c = decode()
    finally:
        server.stop()

    print("A (initial)  :", repr(text_a))
    print("B (corrupted):", repr(text_b))
    print("C (restored) :", repr(text_c))
    ok = text_a != text_b and text_a == text_c
    print("WEIGHT SYNC CHECK:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
