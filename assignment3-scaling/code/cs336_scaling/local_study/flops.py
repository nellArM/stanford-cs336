"""Compute accounting and the model family used by the local scaling study.

Model size follows the handout: N = 12 * n_layer * d_model^2 (non-embedding parameters).

The compute budget is denominated in FLOPs with the usual C = 6 * N * D rule, with one
correction that matters at this small scale: the unembedding matmul (vocab_size * d_model
weights applied to every token) is real per-token work that is comparable to, or larger than,
the whole Transformer stack for the smallest models.  The handout's API budget is wall-clock
time, which charges for that work too, so the budget here uses

    C = 6 * (N + vocab_size * d_model) * D = 6 * N_eff * D.

The scaling law itself is still expressed in the handout's N.
"""

from __future__ import annotations

from cs336_scaling.local_study.model import ModelConfig

VOCAB_SIZE = 32_000
SEQ_LEN = 512
HEAD_DIM = 64

# (d_model, n_layer): depth grows with width (n_layer = d_model / 64), head_dim fixed at 64.
# Each step multiplies N by roughly 1.6-2.
FAMILY: list[tuple[int, int]] = [
    (192, 3),
    (256, 4),
    (320, 5),
    (384, 6),
    (448, 7),
    (512, 8),
    (576, 9),
    (640, 10),
    (704, 11),
    (768, 12),
    (832, 13),
    (896, 14),
    (960, 15),
    (1024, 16),
]


def non_embedding_params(d_model: int, n_layer: int) -> int:
    return 12 * n_layer * d_model**2


def effective_params(d_model: int, n_layer: int, vocab_size: int = VOCAB_SIZE) -> int:
    return non_embedding_params(d_model, n_layer) + vocab_size * d_model


def flops(d_model: int, n_layer: int, tokens: int) -> float:
    return 6.0 * effective_params(d_model, n_layer) * tokens


def tokens_for_budget(d_model: int, n_layer: int, compute: float) -> float:
    return compute / (6.0 * effective_params(d_model, n_layer))


def model_config(d_model: int, n_layer: int) -> ModelConfig:
    return ModelConfig(hidden_size=d_model, num_hidden_layers=n_layer, num_attention_heads=d_model // HEAD_DIM)


def model_name(d_model: int, n_layer: int) -> str:
    return f"d{d_model}L{n_layer}"


def choose_batch_size(tokens: float, min_steps: int = 2000, choices=(32, 16)) -> int:
    """Largest batch size that still leaves at least ``min_steps`` optimizer steps."""
    for batch_size in choices:
        if tokens / (batch_size * SEQ_LEN) >= min_steps:
            return batch_size
    return choices[-1]


def round_tokens(tokens: float, batch_size: int, n_evals: int) -> int:
    """Round down so that the step count is a multiple of n_evals (an API validation rule)."""
    steps = int(tokens // (batch_size * SEQ_LEN))
    steps = max(n_evals, steps - steps % n_evals)
    return steps * batch_size * SEQ_LEN
