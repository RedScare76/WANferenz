import os
import sys
import types

import torch


V4_KERNELS = os.environ.get("V4_KERNELS", "auto")

FP8_MAX = 448.0
FP4_MAX = 6.0

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _round_scale(amax, max_inv):

    x = amax * max_inv
    mant, exp = torch.frexp(x)
    return torch.ldexp(torch.ones_like(x), torch.where(mant == 0.5, exp - 1, exp))


def _blocks(x, block_size):

    n = x.size(-1)
    assert n % block_size == 0, (
        f"v4_kernels_cpu: last dim {n} not a multiple of {block_size}"
    )
    return x.contiguous().float().unflatten(-1, (n // block_size, block_size))


def act_quant(
    x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False
):

    z = _blocks(x, block_size)
    amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = (
        _round_scale(amax, 1.0 / FP8_MAX)
        if scale_fmt is not None
        else amax * (1.0 / FP8_MAX)
    )
    q = (z / s).clamp(-FP8_MAX, FP8_MAX)
    if inplace:
        y = (q.to(torch.float8_e4m3fn).float() * s).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    return q.to(torch.float8_e4m3fn).flatten(-2), s.squeeze(-1).to(scale_dtype)


def _e2m1(a):

    mid = torch.tensor(E2M1_MID, dtype=torch.float32, device=a.device)
    lo = torch.bucketize(a, mid, right=False)
    hi = torch.bucketize(a, mid, right=True)
    code = lo + ((hi != lo) & (lo % 2 == 1)).to(lo.dtype)
    return torch.tensor(E2M1, dtype=torch.float32, device=a.device)[code], code


def fp4_act_quant(x, block_size=32, inplace=False):

    z = _blocks(x, block_size)
    amax = z.abs().amax(-1, keepdim=True).clamp_min(FP4_MAX * 2.0**-126)
    s = _round_scale(amax, 1.0 / FP4_MAX)
    q = (z / s).clamp(-FP4_MAX, FP4_MAX)
    val, code = _e2m1(q.abs())
    if inplace:
        y = (torch.where(q < 0, -val, val) * s).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    code = (code | ((q < 0).to(code.dtype) * 8)).flatten(-2).to(torch.uint8)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
    return packed.view(torch.float4_e2m1fn_x2), s.squeeze(-1).to(torch.float8_e8m0fnu)


def _dequant(v, s, block):

    return v.float() * s.float().repeat_interleave(block, -1)[..., : v.size(-1)]


def unpack_fp4(b):

    u = b.view(torch.uint8)
    table = torch.tensor(
        E2M1 + tuple(-v for v in E2M1), dtype=torch.float32, device=b.device
    )
    return torch.stack(
        [table[(u & 0x0F).long()], table[(u >> 4).long()]], dim=-1
    ).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):

    k, n = a.size(-1), b.size(0)
    da = _dequant(a.reshape(-1, k), a_s.reshape(-1, a_s.size(-1)), 128)
    db = _dequant(b, b_s.repeat_interleave(128, 0)[:n], 128)
    c = da @ db.t()
    return c.view(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):

    k, n = a.size(-1), b.size(0)
    da = _dequant(a.reshape(-1, k), a_s.reshape(-1, a_s.size(-1)), 128)
    db = unpack_fp4(b).float() * b_s.float().repeat_interleave(32, -1)[:, :k]
    c = da @ db.t()
    return c.view(*a.shape[:-1], n).to(torch.get_default_dtype())


SPARSE_ATTN_BLOCK = 64


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):

    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)
    block = SPARSE_ATTN_BLOCK
    ar_b = torch.arange(b, device=kv.device)[:, None, None]
    acc_o = torch.zeros(b, s, h, d, dtype=torch.float32, device=q.device)
    sum_exp = torch.zeros(b, s, h, dtype=torch.float32, device=q.device)
    m = torch.full((b, s, h), float("-inf"), dtype=torch.float32, device=q.device)
    for t in range(0, max(topk, 1), block):
        n = min(block, topk - t)
        idx = topk_idxs.new_full((b, s, block), -1)
        if n > 0:
            idx[..., :n] = topk_idxs[..., t : t + n]
        idx = idx.long()
        valid = idx >= 0
        gathered = kv[ar_b, idx.clamp_min(0)].float()
        gathered = torch.where(
            valid.unsqueeze(-1), gathered, torch.zeros_like(gathered)
        )
        scores = torch.einsum("bshd,bstd->bsht", q.float(), gathered) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        m_prev = m
        m = torch.maximum(m, scores.amax(-1))
        rescale = torch.exp(m_prev - m)
        rescale = torch.where(torch.isnan(rescale), torch.ones_like(rescale), rescale)
        p = torch.exp(scores - m.unsqueeze(-1))
        sum_exp = sum_exp * rescale + p.sum(-1)
        acc_o = acc_o * rescale.unsqueeze(-1) + torch.einsum(
            "bsht,bstd->bshd", p, gathered
        )
    sum_exp = sum_exp + torch.exp(attn_sink.float().view(1, 1, h) - m)
    return (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):

    hc = hc_mult
    mixes = mixes.float()
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = mixes[..., 2 * hc :].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[
        2 * hc :
    ].view(hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hadamard_transform(x, scale=1.0):

    d = x.size(-1)
    assert d & (d - 1) == 0, (
        f"v4_kernels_cpu: hadamard needs a power-of-two last dim, got {d}"
    )
    y = x.float()
    h = 1
    while h < d:
        y = y.unflatten(-1, (d // (2 * h), 2, h))
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], dim=-2).flatten(-3)
        h *= 2
    return (y * scale).to(x.dtype)


_MODULES = {
    "wanferenz.kernels.reference": {
        "act_quant": act_quant,
        "fp4_act_quant": fp4_act_quant,
        "fp8_gemm": fp8_gemm,
        "fp4_gemm": fp4_gemm,
        "sparse_attn": sparse_attn,
        "hc_split_sinkhorn": hc_split_sinkhorn,
    },
    "fast_hadamard_transform": {"hadamard_transform": hadamard_transform},
}


def backend():

    if V4_KERNELS in ("cpu", "tilelang"):
        return V4_KERNELS
    if V4_KERNELS != "auto":
        raise RuntimeError(
            f"V4_KERNELS={V4_KERNELS!r} — expected auto, cpu or tilelang"
        )
    return "cpu" if not torch.cuda.is_available() else "tilelang"


def _foreign(name):

    mod = sys.modules.get(name)
    return mod is not None and not getattr(mod, "_v4_cpu_backend", False)


def install():

    be = backend()
    if be == "tilelang":
        try:
            __import__("fast_hadamard_transform")
        except Exception:
            mod = types.ModuleType("fast_hadamard_transform")
            mod._v4_cpu_backend = True
            mod.hadamard_transform = hadamard_transform
            sys.modules["fast_hadamard_transform"] = mod
        return be
    clash = [n for n in _MODULES if _foreign(n)]
    if clash and V4_KERNELS != "cpu":
        raise RuntimeError(
            f"v4_kernels_cpu: {clash} already imported by something else; "
            f"set V4_KERNELS=cpu to override deliberately"
        )
    for name, attrs in _MODULES.items():
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "_v4_cpu_backend", False):
            mod = types.ModuleType(name)
            mod._v4_cpu_backend = True
            sys.modules[name] = mod
        for attr, value in attrs.items():
            setattr(mod, attr, value)
    return be


__all__ = [
    "backend",
    "install",
    "act_quant",
    "fp4_act_quant",
    "fp8_gemm",
    "fp4_gemm",
    "unpack_fp4",
    "sparse_attn",
    "hc_split_sinkhorn",
    "hadamard_transform",
]
