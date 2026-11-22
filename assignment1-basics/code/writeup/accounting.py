"""Numbers for the written resource-accounting questions (transformer_accounting, adamw_accounting,
learning_rate_tuning). Prints a JSON blob used when writing the report."""
import json
import math

import torch


def params(V, T, L, d, h, dff):
    per_layer = 4 * d * d + 3 * d * dff + 2 * d
    return V * d + L * per_layer + d + V * d


def flops(V, T, L, d, h, dff):
    """Forward-pass matmul FLOPs for one sequence of T tokens (2mnp per (m x n)(n x p) product)."""
    qkv = L * 3 * 2 * T * d * d
    scores = L * 2 * T * T * d  # h heads, each 2*T*T*(d/h)
    weighted = L * 2 * T * T * d
    out = L * 2 * T * d * d
    ffn = L * 3 * 2 * T * d * dff
    head = 2 * T * d * V
    parts = {"attn_proj(QKVO)": qkv + out, "attn_matmul(QK^T, AV)": scores + weighted, "ffn": ffn, "lm_head": head}
    parts["total"] = sum(parts.values())
    return parts


res = {}
configs = {
    "small": (50257, 1024, 12, 768, 12, 2048),
    "medium": (50257, 1024, 24, 1024, 16, 2752),
    "large": (50257, 1024, 36, 1280, 20, 3392),
    "xl": (50257, 1024, 48, 1600, 25, 4288),
    "xl_ctx16384": (50257, 16384, 48, 1600, 25, 4288),
}
for name, cfg in configs.items():
    p = params(*cfg)
    f = flops(*cfg)
    res[name] = {
        "d_ff": cfg[5],
        "params": p,
        "fp32_GiB": p * 4 / 2**30,
        "fp32_GB": p * 4 / 1e9,
        "flops": f,
        "share": {k: v / f["total"] for k, v in f.items() if k != "total"},
    }

# adamw_accounting (b): memory = 4 bytes * (4 * P + B * A), A = activation floats per sequence.
V, T, L, d, h = 50257, 1024, 48, 1600, 25
dff = 8 * d / 3
P = 2 * V * d + L * (4 * d * d + 3 * d * dff + 2 * d) + d
# per block: 2 RMSNorm outputs (2Td) + QKV (3Td) + scores (hT^2) + softmax (hT^2) + weighted sum (Td)
#            + out proj (Td) + W1 (Tdff) + SiLU (Tdff) + W3 (Tdff) + product (Tdff) + W2 (Td)
block = 8 * T * d + 2 * h * T * T + 4 * T * dff
A = L * block + T * d + T * V + T * V  # final norm, logits, cross-entropy (log-softmax)
a_bytes = 4 * A
b_bytes = 4 * 4 * P
res["adamw"] = {
    "P": P,
    "A_floats_per_example": A,
    "a_GB": a_bytes / 1e9,
    "b_GB": b_bytes / 1e9,
    "max_batch_80GB": math.floor((80e9 - b_bytes) / a_bytes),
    "a_GiB": a_bytes / 2**30,
    "b_GiB": b_bytes / 2**30,
    "max_batch_80GiB": math.floor((80 * 2**30 - b_bytes) / a_bytes),
}
fwd = res["xl"]["flops"]["total"]
steps, batch = 400_000, 1024
total = 3 * fwd * batch * steps
res["adamw"]["train_flops"] = total
res["adamw"]["hours_at_50pct_MFU"] = total / (0.5 * 495e12) / 3600

# learning_rate_tuning: the toy SGD example with lr / sqrt(t+1) decay.
toy = {}
for lr in (1.0, 1e1, 1e2, 1e3):
    torch.manual_seed(0)
    w = torch.nn.Parameter(5 * torch.randn((10, 10)))
    losses = []
    for t in range(10):
        loss = (w**2).mean()
        losses.append(loss.item())
        (g,) = torch.autograd.grad(loss, w)
        with torch.no_grad():
            w -= lr / math.sqrt(t + 1) * g
    toy[str(lr)] = losses
res["toy_sgd"] = toy
print(json.dumps(res, indent=1))
