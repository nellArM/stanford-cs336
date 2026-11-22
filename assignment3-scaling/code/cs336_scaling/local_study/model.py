"""PyTorch port of the staff JAX/Equinox model (cs336_scaling/training/model/basic_model.py).

Architecture, matching the reference:
  * token embedding (N(0,1) init, as equinox.nn.Embedding),
  * pre-norm decoder layers: x + Attn(RMSNorm(x)), then x + MLP(RMSNorm(x)),
  * attention with bias-free q/k/v/o projections, QK-norm (an RMSNorm over head_dim on the
    queries and on the keys, weights shared across heads), then RoPE on q and k, causal mask,
  * SwiGLU MLP: down(silu(gate(x)) * up(x)),
  * final RMSNorm and an (optionally weight-tied) LM head.
Linear layers use equinox's default init U(-1/sqrt(fan_in), 1/sqrt(fan_in)).

Differences from the reference: parameters are kept in float32 and bfloat16 is applied with
autocast (the JAX model stores bfloat16 parameters), and grouped-query attention is not
supported (the API rejects it as well).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int = 64
    intermediate_size: int | None = None  # default: the multiple of 64 nearest to 8/3 * hidden
    vocab_size: int = 32_000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False

    def __post_init__(self):
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.intermediate_size is None:
            object.__setattr__(self, "intermediate_size", default_intermediate_size(self.hidden_size))
        if self.intermediate_size < self.hidden_size:
            raise ValueError("intermediate_size must be >= hidden_size")

    def to_dict(self) -> dict:
        return asdict(self)


def default_intermediate_size(hidden_size: int) -> int:
    return max(64, round(8 * hidden_size / 3 / 64) * 64)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)


def _linear(in_features: int, out_features: int, bias: bool = False) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    bound = 1.0 / math.sqrt(in_features)
    nn.init.uniform_(layer.weight, -bound, bound)
    if bias:
        nn.init.uniform_(layer.bias, -bound, bound)
    return layer


def rope_tables(head_dim: int, seq_len: int, theta: float, device=None) -> tuple[Tensor, Tensor]:
    freqs = theta ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float64) / head_dim)
    angles = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float64), freqs)
    return angles.cos().float(), angles.sin().float()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (batch, heads, seq, head_dim) with (even, odd) pairs interleaved, as in the reference."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.stack((cos * x1 - sin * x2, sin * x1 + cos * x2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        inner = self.num_heads * self.head_dim
        self.q_proj = _linear(config.hidden_size, inner, config.attention_bias)
        self.k_proj = _linear(config.hidden_size, inner, config.attention_bias)
        self.v_proj = _linear(config.hidden_size, inner, config.attention_bias)
        self.o_proj = _linear(inner, config.hidden_size, config.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        shape = (batch, seq, self.num_heads, self.head_dim)
        q = self.q_norm(self.q_proj(x).view(shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(shape)).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = _linear(config.hidden_size, config.intermediate_size)
        self.up_proj = _linear(config.hidden_size, config.intermediate_size)
        self.down_proj = _linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 512):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # N(0, 1) init
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = None if config.tie_word_embeddings else _linear(config.hidden_size, config.vocab_size)
        cos, sin = rope_tables(config.head_dim, max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        seq = input_ids.shape[1]
        cos, sin = self.rope_cos[:seq], self.rope_sin[:seq]
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        if self.lm_head is None:
            return F.linear(x, self.embed_tokens.weight)
        return self.lm_head(x)

    def loss(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        """Mean next-token cross-entropy (every sequence has the same length, so this equals
        the reference's mean of per-sequence means)."""
        logits = self(input_ids)
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.reshape(-1))
