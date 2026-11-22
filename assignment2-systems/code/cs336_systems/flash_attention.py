# ruff: noqa: E741  (O, L, ... follow the notation of the FlashAttention-2 paper)
"""FlashAttention-2.

Three ``torch.autograd.Function`` classes share one interface ``apply(Q, K, V, is_causal=False)``
with Q of shape (..., n_queries, d) and K, V of shape (..., n_keys, d):

* ``FlashAttentionPyTorch``  - tiled forward in plain PyTorch (reference / debugging aid).
* ``FlashAttentionTriton``   - Triton forward kernel, backward is a ``torch.compile``d PyTorch
                               function that recomputes P from the saved logsumexp L.
* ``FlashAttentionTritonFull`` - Triton forward *and* Triton backward (the optional two-pass
                               algorithm), so no (n_queries x n_keys) tensor is ever materialised.
"""

from __future__ import annotations

import math

import torch

try:  # Triton only exists on Linux + CUDA builds; the PyTorch class must work without it.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

MASK_VALUE = -1e6


# --------------------------------------------------------------------------------------
# Backward pass in PyTorch (Equations 13-19 of the handout), shared by two of the classes
# --------------------------------------------------------------------------------------
def _flash_backward_impl(Q, K, V, O, dO, L, is_causal: bool):
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    D = (O * dO).sum(dim=-1, keepdim=True)  # (b, q, 1)
    S = torch.matmul(Q, K.transpose(-1, -2)) * scale
    if is_causal:
        n_q, n_k = Q.shape[-2], K.shape[-2]
        future = torch.arange(n_k, device=Q.device)[None, :] > torch.arange(n_q, device=Q.device)[:, None]
        S = S + future.to(S.dtype) * MASK_VALUE
    P = torch.exp(S - L.unsqueeze(-1))
    dV = torch.matmul(P.transpose(-1, -2), dO)
    dP = torch.matmul(dO, V.transpose(-1, -2))
    dS = P * (dP - D)
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-1, -2), Q) * scale
    return dQ, dK, dV


_compiled_backward = None
_compile_failed = False


def flash_backward(Q, K, V, O, dO, L, is_causal: bool):
    """Compiled backward; falls back to eager execution if no compiler toolchain is available."""
    global _compiled_backward, _compile_failed
    if not _compile_failed:
        if _compiled_backward is None:
            _compiled_backward = torch.compile(_flash_backward_impl)
        try:
            return _compiled_backward(Q, K, V, O, dO, L, is_causal)
        except Exception:  # e.g. missing C++ compiler for the CPU backend
            _compile_failed = True
    return _flash_backward_impl(Q, K, V, O, dO, L, is_causal)


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.shape[-2], x.shape[-1]).contiguous()


# --------------------------------------------------------------------------------------
# (a) Tiled forward in pure PyTorch
# --------------------------------------------------------------------------------------
class FlashAttentionPyTorch(torch.autograd.Function):
    Q_TILE = 32
    K_TILE = 32

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        n_q, n_k, d = Q.shape[-2], K.shape[-2], Q.shape[-1]
        scale = 1.0 / math.sqrt(d)
        bq, bk = FlashAttentionPyTorch.Q_TILE, FlashAttentionPyTorch.K_TILE
        O = torch.empty(*Q.shape[:-1], V.shape[-1], dtype=Q.dtype, device=Q.device)
        L = torch.empty(Q.shape[:-1], dtype=Q.dtype, device=Q.device)

        for qs in range(0, n_q, bq):
            q_tile = Q[..., qs : qs + bq, :]
            tq = q_tile.shape[-2]
            o_acc = torch.zeros(*Q.shape[:-2], tq, V.shape[-1], dtype=torch.float32, device=Q.device)
            l_acc = torch.zeros(*Q.shape[:-2], tq, dtype=torch.float32, device=Q.device)
            m_acc = torch.full_like(l_acc, float("-inf"))
            q_idx = torch.arange(qs, qs + tq, device=Q.device)
            for ks in range(0, n_k, bk):
                if is_causal and ks > qs + tq - 1:
                    break  # every key in this and later tiles is in the future
                k_tile = K[..., ks : ks + bk, :]
                v_tile = V[..., ks : ks + bk, :]
                S = torch.matmul(q_tile, k_tile.transpose(-1, -2)).float() * scale
                if is_causal:
                    k_idx = torch.arange(ks, ks + k_tile.shape[-2], device=Q.device)
                    S = S + (k_idx[None, :] > q_idx[:, None]).float() * MASK_VALUE
                m_new = torch.maximum(m_acc, S.amax(dim=-1))
                P = torch.exp(S - m_new.unsqueeze(-1))
                alpha = torch.exp(m_acc - m_new)
                l_acc = alpha * l_acc + P.sum(dim=-1)
                o_acc = alpha.unsqueeze(-1) * o_acc + torch.matmul(P.to(v_tile.dtype), v_tile).float()
                m_acc = m_new
            O[..., qs : qs + tq, :] = (o_acc / l_acc.unsqueeze(-1)).to(Q.dtype)
            L[..., qs : qs + tq] = (m_acc + torch.log(l_acc)).to(Q.dtype)

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


# --------------------------------------------------------------------------------------
# (b, c) Triton kernels
# --------------------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):  # fmt: skip
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_acc = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_base = tl.arange(0, K_TILE_SIZE)

        # With a causal mask, key tiles that start after the last query of this tile are
        # entirely masked, so the loop can stop early.
        if is_causal:
            last_query = tl.minimum((query_tile_index + 1) * Q_TILE_SIZE, N_QUERIES)
            n_key_tiles = tl.cdiv(tl.minimum(last_query, N_KEYS), K_TILE_SIZE)
        else:
            n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        for j in range(n_key_tiles):
            K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale
            k_idx = j * K_TILE_SIZE + k_base
            # Keys past the end of the sequence (only possible when the tile does not divide N_KEYS).
            S = tl.where(k_idx[None, :] < N_KEYS, S, float("-inf"))
            if is_causal:
                S = S + tl.where(k_idx[None, :] > q_idx[:, None], -1e6, 0.0)

            m_new = tl.maximum(m_acc, tl.max(S, axis=1))
            P = tl.exp(S - m_new[:, None])
            alpha = tl.exp(m_acc - m_new)
            l_acc = alpha * l_acc + tl.sum(P, axis=1)
            O_acc = O_acc * alpha[:, None]
            O_acc = tl.dot(P.to(V_tile.dtype), V_tile, acc=O_acc)
            m_acc = m_new

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        O_out = O_acc / l_acc[:, None]
        L_out = m_acc + tl.log(l_acc)
        tl.store(O_block_ptr, O_out.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, L_out.to(L_block_ptr.type.element_ty), boundary_check=(0,))

    @triton.jit
    def flash_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Dvec_ptr, dQ_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        stride_dqb, stride_dqq, stride_dqd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):  # fmt: skip
        """Pass 1 of the backward: one program per query tile accumulates dQ over all key tiles."""
        qt = tl.program_id(0)
        b = tl.program_id(1)

        Q_bp = tl.make_block_ptr(Q_ptr + b * stride_qb, (N_QUERIES, D), (stride_qq, stride_qd), (qt * Q_TILE_SIZE, 0), (Q_TILE_SIZE, D), (1, 0))
        dO_bp = tl.make_block_ptr(dO_ptr + b * stride_dob, (N_QUERIES, D), (stride_doq, stride_dod), (qt * Q_TILE_SIZE, 0), (Q_TILE_SIZE, D), (1, 0))
        dQ_bp = tl.make_block_ptr(dQ_ptr + b * stride_dqb, (N_QUERIES, D), (stride_dqq, stride_dqd), (qt * Q_TILE_SIZE, 0), (Q_TILE_SIZE, D), (1, 0))
        L_bp = tl.make_block_ptr(L_ptr + b * stride_lb, (N_QUERIES,), (stride_lq,), (qt * Q_TILE_SIZE,), (Q_TILE_SIZE,), (0,))
        Dv_bp = tl.make_block_ptr(Dvec_ptr + b * stride_db, (N_QUERIES,), (stride_dq,), (qt * Q_TILE_SIZE,), (Q_TILE_SIZE,), (0,))
        K_bp = tl.make_block_ptr(K_ptr + b * stride_kb, (N_KEYS, D), (stride_kk, stride_kd), (0, 0), (K_TILE_SIZE, D), (1, 0))
        V_bp = tl.make_block_ptr(V_ptr + b * stride_vb, (N_KEYS, D), (stride_vk, stride_vd), (0, 0), (K_TILE_SIZE, D), (1, 0))

        Q_tile = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")
        dO_tile = tl.load(dO_bp, boundary_check=(0, 1), padding_option="zero")
        L_tile = tl.load(L_bp, boundary_check=(0,), padding_option="zero").to(tl.float32)
        D_tile = tl.load(Dv_bp, boundary_check=(0,), padding_option="zero").to(tl.float32)

        dQ_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        q_idx = qt * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        k_base = tl.arange(0, K_TILE_SIZE)

        if is_causal:
            last_query = tl.minimum((qt + 1) * Q_TILE_SIZE, N_QUERIES)
            n_key_tiles = tl.cdiv(tl.minimum(last_query, N_KEYS), K_TILE_SIZE)
        else:
            n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        for j in range(n_key_tiles):
            K_tile = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")
            V_tile = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")
            S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale
            k_idx = j * K_TILE_SIZE + k_base
            S = tl.where(k_idx[None, :] < N_KEYS, S, float("-inf"))
            if is_causal:
                S = S + tl.where(k_idx[None, :] > q_idx[:, None], -1e6, 0.0)
            P = tl.exp(S - L_tile[:, None])
            dP = tl.dot(dO_tile, tl.trans(V_tile)).to(tl.float32)
            dS = P * (dP - D_tile[:, None]) * scale
            dQ_acc = tl.dot(dS.to(K_tile.dtype), K_tile, acc=dQ_acc)
            K_bp = K_bp.advance((K_TILE_SIZE, 0))
            V_bp = V_bp.advance((K_TILE_SIZE, 0))

        tl.store(dQ_bp, dQ_acc.to(dQ_bp.type.element_ty), boundary_check=(0, 1))

    @triton.jit
    def flash_bwd_dkv_kernel(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Dvec_ptr, dK_ptr, dV_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq,
        stride_dkb, stride_dkk, stride_dkd,
        stride_dvb, stride_dvk, stride_dvd,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):  # fmt: skip
        """Pass 2 of the backward: one program per key tile accumulates dK and dV over query tiles.

        Recomputing P here (instead of scattering from the dQ pass) means every program writes
        only to its own key tile, so no atomics are needed.
        """
        kt = tl.program_id(0)
        b = tl.program_id(1)

        K_bp = tl.make_block_ptr(K_ptr + b * stride_kb, (N_KEYS, D), (stride_kk, stride_kd), (kt * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0))
        V_bp = tl.make_block_ptr(V_ptr + b * stride_vb, (N_KEYS, D), (stride_vk, stride_vd), (kt * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0))
        dK_bp = tl.make_block_ptr(dK_ptr + b * stride_dkb, (N_KEYS, D), (stride_dkk, stride_dkd), (kt * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0))
        dV_bp = tl.make_block_ptr(dV_ptr + b * stride_dvb, (N_KEYS, D), (stride_dvk, stride_dvd), (kt * K_TILE_SIZE, 0), (K_TILE_SIZE, D), (1, 0))

        K_tile = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")
        k_idx = kt * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        q_base = tl.arange(0, Q_TILE_SIZE)

        dK_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dV_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

        # With a causal mask only queries at or after the first key of this tile contribute.
        if is_causal:
            first_q_tile = (kt * K_TILE_SIZE) // Q_TILE_SIZE
        else:
            first_q_tile = 0
        n_query_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

        Q_bp = tl.make_block_ptr(Q_ptr + b * stride_qb, (N_QUERIES, D), (stride_qq, stride_qd), (first_q_tile * Q_TILE_SIZE, 0), (Q_TILE_SIZE, D), (1, 0))
        dO_bp = tl.make_block_ptr(dO_ptr + b * stride_dob, (N_QUERIES, D), (stride_doq, stride_dod), (first_q_tile * Q_TILE_SIZE, 0), (Q_TILE_SIZE, D), (1, 0))
        L_bp = tl.make_block_ptr(L_ptr + b * stride_lb, (N_QUERIES,), (stride_lq,), (first_q_tile * Q_TILE_SIZE,), (Q_TILE_SIZE,), (0,))
        Dv_bp = tl.make_block_ptr(Dvec_ptr + b * stride_db, (N_QUERIES,), (stride_dq,), (first_q_tile * Q_TILE_SIZE,), (Q_TILE_SIZE,), (0,))

        for i in range(first_q_tile, n_query_tiles):
            Q_tile = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")
            dO_tile = tl.load(dO_bp, boundary_check=(0, 1), padding_option="zero")
            L_tile = tl.load(L_bp, boundary_check=(0,), padding_option="zero").to(tl.float32)
            D_tile = tl.load(Dv_bp, boundary_check=(0,), padding_option="zero").to(tl.float32)

            q_idx = i * Q_TILE_SIZE + q_base
            S = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale
            if is_causal:
                S = S + tl.where(k_idx[None, :] > q_idx[:, None], -1e6, 0.0)
            P = tl.exp(S - L_tile[:, None])
            # Rows/columns beyond the sequence ends must contribute nothing.
            P = tl.where((q_idx[:, None] < N_QUERIES) & (k_idx[None, :] < N_KEYS), P, 0.0)

            dV_acc = tl.dot(tl.trans(P).to(dO_tile.dtype), dO_tile, acc=dV_acc)
            dP = tl.dot(dO_tile, tl.trans(V_tile)).to(tl.float32)
            dS = P * (dP - D_tile[:, None]) * scale
            dK_acc = tl.dot(tl.trans(dS).to(Q_tile.dtype), Q_tile, acc=dK_acc)

            Q_bp = Q_bp.advance((Q_TILE_SIZE, 0))
            dO_bp = dO_bp.advance((Q_TILE_SIZE, 0))
            L_bp = L_bp.advance((Q_TILE_SIZE,))
            Dv_bp = Dv_bp.advance((Q_TILE_SIZE,))

        tl.store(dK_bp, dK_acc.to(dK_bp.type.element_ty), boundary_check=(0, 1))
        tl.store(dV_bp, dV_acc.to(dV_bp.type.element_ty), boundary_check=(0, 1))


def default_tile_sizes(n_queries: int, n_keys: int, d: int) -> tuple[int, int]:
    """Tile sizes picked from the tuning sweep in ``benchmarks/flash_benchmark.py``.

    Larger heads need smaller tiles so the (tile x d) operands and the (tile x tile) score
    block still fit in registers / shared memory of one program instance.
    """
    q_tile = 64 if d <= 64 else 32
    k_tile = 64 if d <= 64 else 32
    return max(16, min(q_tile, n_queries)), max(16, min(k_tile, n_keys))


def _triton_forward(Q, K, V, is_causal: bool, tiles: tuple[int, int] | None = None):
    lead_shape = Q.shape[:-2]
    Qf, Kf, Vf = _flatten_batch(Q), _flatten_batch(K), _flatten_batch(V)
    batch, n_q, d = Qf.shape
    n_k = Kf.shape[1]
    q_tile, k_tile = tiles or default_tile_sizes(n_q, n_k, d)
    O = torch.empty_like(Qf)
    L = torch.empty((batch, n_q), dtype=Qf.dtype, device=Qf.device)
    grid = (triton.cdiv(n_q, q_tile), batch)
    flash_fwd_kernel[grid](
        Qf, Kf, Vf, O, L,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Kf.stride(0), Kf.stride(1), Kf.stride(2),
        Vf.stride(0), Vf.stride(1), Vf.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        n_q, n_k, 1.0 / math.sqrt(d),
        D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=bool(is_causal),
    )  # fmt: skip
    return O.view(*lead_shape, n_q, d), L.view(*lead_shape, n_q)


def _triton_backward(Q, K, V, O, dO, L, is_causal: bool, tiles: tuple[int, int] | None = None):
    lead_shape = Q.shape[:-2]
    Qf, Kf, Vf, dOf = _flatten_batch(Q), _flatten_batch(K), _flatten_batch(V), _flatten_batch(dO)
    Of = _flatten_batch(O)
    batch, n_q, d = Qf.shape
    n_k = Kf.shape[1]
    q_tile, k_tile = tiles or default_tile_sizes(n_q, n_k, d)
    Lf = L.reshape(batch, n_q).contiguous()
    Dvec = (Of.float() * dOf.float()).sum(dim=-1)  # fp32 (batch, n_q)
    dQ, dK, dV = torch.empty_like(Qf), torch.empty_like(Kf), torch.empty_like(Vf)
    scale = 1.0 / math.sqrt(d)

    flash_bwd_dq_kernel[(triton.cdiv(n_q, q_tile), batch)](
        Qf, Kf, Vf, dOf, Lf, Dvec, dQ,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Kf.stride(0), Kf.stride(1), Kf.stride(2),
        Vf.stride(0), Vf.stride(1), Vf.stride(2),
        dOf.stride(0), dOf.stride(1), dOf.stride(2),
        Lf.stride(0), Lf.stride(1),
        Dvec.stride(0), Dvec.stride(1),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        n_q, n_k, scale,
        D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=bool(is_causal),
    )  # fmt: skip
    flash_bwd_dkv_kernel[(triton.cdiv(n_k, k_tile), batch)](
        Qf, Kf, Vf, dOf, Lf, Dvec, dK, dV,
        Qf.stride(0), Qf.stride(1), Qf.stride(2),
        Kf.stride(0), Kf.stride(1), Kf.stride(2),
        Vf.stride(0), Vf.stride(1), Vf.stride(2),
        dOf.stride(0), dOf.stride(1), dOf.stride(2),
        Lf.stride(0), Lf.stride(1),
        Dvec.stride(0), Dvec.stride(1),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        n_q, n_k, scale,
        D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=bool(is_causal),
    )  # fmt: skip
    return dQ.view(*lead_shape, n_q, d), dK.view(*lead_shape, n_k, d), dV.view(*lead_shape, n_k, d)


class FlashAttentionTriton(torch.autograd.Function):
    """Triton forward, ``torch.compile``d PyTorch backward (the configuration the handout asks for)."""

    tiles: tuple[int, int] | None = None  # override for tuning; None -> default_tile_sizes

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = _triton_forward(Q, K, V, is_causal, FlashAttentionTriton.tiles)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


class FlashAttentionTritonFull(torch.autograd.Function):
    """Triton forward and Triton two-pass backward: O(n) memory in both directions."""

    tiles: tuple[int, int] | None = None

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        O, L = _triton_forward(Q, K, V, is_causal, FlashAttentionTritonFull.tiles)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _triton_backward(Q, K, V, O, dO.contiguous(), L, ctx.is_causal, FlashAttentionTritonFull.tiles)
        return dQ, dK, dV, None
