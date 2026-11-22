"""Transformer language model built from scratch (no torch.nn layers besides containers)."""

from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import Tensor, nn

from cs336_basics.nn_utils import silu, softmax


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # squaring can overflow in half precision
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(in_dtype)


class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class SwiGLU(nn.Module):
    """W2(SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = max(64, round(8 * d_model / 3 / 64) * 64)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Ungated ablation: W2 SiLU(W1 x)."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE needs an even dimension"
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(positions, inv_freq, "pos, freq -> pos freq")
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """x: (..., seq, d_k); token_positions: (..., seq), broadcastable against x's leading dims."""
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
        return rotated.flatten(-2)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """Q: (..., queries, d_k), K: (..., keys, d_k), V: (..., keys, d_v); mask True = may attend."""
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return einsum(weights, V, "... q k, ... k d -> ... q d")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        seq_len = x.shape[-2]
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            else:
                token_positions = token_positions.unsqueeze(-2)  # broadcast over the head dim
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        out = scaled_dot_product_attention(q, k, v, causal_mask)
        return self.output_proj(rearrange(out, "... h s d -> ... s (h d)"))


class TransformerBlock(nn.Module):
    """Transformer block.

    ``norm_mode``: "pre" (default), "post" or "none".  ``ffn_type``: "swiglu" or "silu".
    The non-default settings exist for the ablation experiments.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert norm_mode in ("pre", "post", "none")
        assert ffn_type in ("swiglu", "silu")
        self.norm_mode = norm_mode
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope, device=device, dtype=dtype)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        if norm_mode == "none":
            self.ln1, self.ln2 = Identity(), Identity()
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_mode == "post":
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu",
        use_rope: bool = True,
        tie_embeddings: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = (
            RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
            if use_rope
            else None
        )
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, rope, norm_mode, ffn_type, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        # A post-norm stack already ends in a norm; the no-norm ablation has none anywhere.
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm_mode == "pre" else Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        if tie_embeddings:
            # Use the (small) linear init for the shared matrix: unit-variance embeddings
            # would make the output logits enormous at initialization.
            self.token_embeddings.weight = self.lm_head.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))

    def num_parameters(self, non_embedding: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and self.token_embeddings.weight is not self.lm_head.weight:
            total -= self.token_embeddings.weight.numel()
        return total

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressively sample from a 1-D prompt with temperature scaling and nucleus (top-p) sampling."""
        ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self(ids[-self.context_length :].unsqueeze(0))[0, -1].float()
            if temperature <= 0:
                next_id = logits.argmax().unsqueeze(0)
            else:
                probs = softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    # Keep the smallest prefix whose mass reaches top_p (always at least one token).
                    drop = cumulative - sorted_probs >= top_p
                    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
                    probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                    probs = probs / probs.sum()
                next_id = torch.multinomial(probs, 1, generator=generator)
            ids = torch.cat([ids, next_id])
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
        return ids
