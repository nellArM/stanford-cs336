"""Extra correctness checks for the Triton kernels beyond the provided tests."""
import torch

from cs336_systems.flash_attention import FlashAttentionTriton, FlashAttentionTritonFull


def reference(q, k, v, causal):
    s = q.float() @ k.float().transpose(-1, -2) / q.shape[-1] ** 0.5
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.triu(torch.ones(n, k.shape[-2], dtype=torch.bool, device=q.device), 1), -1e6)
    return torch.softmax(s, -1) @ v.float()


torch.manual_seed(0)
for dtype in (torch.float32, torch.bfloat16):
    for n, d in ((128, 16), (512, 64), (2048, 128), (384, 32), (200, 32)):
        for causal in (False, True):
            q, k, v = (torch.randn(2, 3, n, d, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
            do = torch.randn(2, 3, n, d, device="cuda", dtype=dtype)
            ref = reference(q, k, v, causal)
            gq, gk, gv = torch.autograd.grad(ref, (q, k, v), do.float())
            for cls in (FlashAttentionTriton, FlashAttentionTritonFull):
                out = cls.apply(q, k, v, causal)
                dq, dk, dv = torch.autograd.grad(out, (q, k, v), do)
                tol = 1e-2 if dtype == torch.float32 else 8e-2  # tl.dot uses TF32 for fp32 inputs
                errs = [(out.float() - ref).abs().max().item()] + [(a.float() - b).abs().max().item() / (b.abs().max().item() + 1e-6) for a, b in ((dq, gq), (dk, gk), (dv, gv))]
                status = "ok" if max(errs) < tol else "FAIL"
                print(f"{status} {cls.__name__:26s} {str(dtype):15s} n={n:5d} d={d:3d} causal={causal!s:5s} errs={[f'{e:.1e}' for e in errs]}")
