"""Sample text from a trained checkpoint.

Example:
    uv run python -m cs336_basics.generate --run experiments/runs/ts_full \
        --tokenizer artifacts/tokenizers/tinystories --prompt "Once upon a time" --temperature 0.8 --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory containing config.json and checkpoint.pt")
    p.add_argument("--tokenizer", required=True, help="prefix of the *_vocab.json / *_merges.txt files")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    tokenizer = Tokenizer.from_files(
        f"{args.tokenizer}_vocab.json", f"{args.tokenizer}_merges.txt", special_tokens=["<|endoftext|>"]
    )
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        norm_mode=config["norm_mode"],
        ffn_type=config["ffn_type"],
        use_rope=not config["no_rope"],
        tie_embeddings=config["tie_embeddings"],
        device=args.device,
    )
    state = torch.load(run / "checkpoint.pt", map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()

    eos_id = tokenizer.encode("<|endoftext|>")[0]
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=args.device)
    for i in range(args.num_samples):
        out = model.generate(prompt_ids, args.max_new_tokens, args.temperature, args.top_p, eos_id, generator)
        new_tokens = len(out) - len(prompt_ids)
        print(f"--- sample {i} ({new_tokens} new tokens) ---")
        print(tokenizer.decode(out.tolist()))


if __name__ == "__main__":
    main()
