import os

import torch


from wanferenz.kernels.gb10 import pipeline_stages

_SHIPPED_TILE = (128, pipeline_stages(), 128)
_BLOCK_M = 32
_BLOCK_K = 128
_TARGET_BLOCKS = 192


def _parse_gemv(s):

    if not s or s == "0":
        return None, None
    if s in ("1", "auto"):
        return "auto", None
    if s == "gb10":
        return "gb10", None
    try:
        vals = [int(p) for p in s.split(",")]
    except ValueError:
        return None, f"not integers: {s!r}"
    if not 1 <= len(vals) <= 3:
        return None, f"want BN[,STAGES[,THREADS]], got {s!r}"
    bn = vals[0]
    stages = vals[1] if len(vals) > 1 else _SHIPPED_TILE[1]
    threads = vals[2] if len(vals) > 2 else _SHIPPED_TILE[2]
    if bn <= 0 or 128 % bn:
        return None, f"block_N {bn} must divide 128 (the fp8 weight-scale group)"
    if not 1 <= stages <= 6:
        return None, f"num_stages {stages} outside 1..6"
    if threads % 32 or not 32 <= threads <= 512:
        return None, f"threads {threads} must be a multiple of 32 in 32..512"
    return (bn, stages, threads), None


V4_FP8_GEMV = os.environ.get("V4_FP8_GEMV", "")
_GEMV_MODE, _GEMV_INVALID = _parse_gemv(V4_FP8_GEMV)
if _GEMV_INVALID:
    print(f"[v4] V4_FP8_GEMV={V4_FP8_GEMV!r} IGNORED — {_GEMV_INVALID}", flush=True)

V4_FP8_SHARED = os.environ.get("V4_FP8_SHARED", "0") not in ("", "0")


_REF_FP8_GEMM = None
_REF_EXPERT_FORWARD = None
_MOD = None

_KERNELS = {}
_GEMV_VERDICTS = {}
_SHARED_VERDICTS = {}


def _tl_dtype(scale_dtype):
    return "float8_e8m0fnu" if scale_dtype == torch.float8_e8m0fnu else "float32"


def fp8_gemm_tiled_kernel(N, K, scale_dtype="float32", tile=None):

    key = (N, K, scale_dtype, tile)
    if key in _KERNELS:
        return _KERNELS[key]
    import tilelang
    import tilelang.language as T

    tilelang.set_log_level("WARNING")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    FP8, FP32, BF16 = "float8_e4m3", "float32", "bfloat16"
    out_dtype, accum_dtype = BF16, FP32
    group_size = 128
    block_M = _BLOCK_M
    block_N, num_stages, threads = tile if tile is not None else _SHIPPED_TILE
    block_K = _BLOCK_K

    assert 128 % block_N == 0, (
        f"block_N={block_N} must divide 128 (the fp8 weight-scale group)"
    )

    M = T.symbolic("M")

    @tilelang.jit(pass_configs=pass_configs)
    def _build():

        @T.prim_func
        def fp8_gemm_tiled_kernel_(
            A: T.Tensor[(M, K), FP8],
            B: T.Tensor[(N, K), FP8],
            C: T.Tensor[(M, N), out_dtype],
            scales_a: T.Tensor[(M, T.ceildiv(K, group_size)), scale_dtype],
            scales_b: T.Tensor[
                (T.ceildiv(N, group_size), T.ceildiv(K, group_size)), scale_dtype
            ],
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
            ) as (
                bx,
                by,
            ):
                A_shared = T.alloc_shared((block_M, block_K), FP8)
                B_shared = T.alloc_shared((block_N, block_K), FP8)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                Scale_C_shared = T.alloc_shared((block_M), FP32)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)

                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                T.clear(C_local_accum)

                K_iters = T.ceildiv(K, block_K)
                for k in T.Pipelined(K_iters, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)

                    Scale_B = T.Cast(FP32, scales_b[bx * block_N // group_size, k])
                    for i in T.Parallel(block_M):
                        Scale_C_shared[i] = (
                            T.Cast(FP32, scales_a[by * block_M + i, k]) * Scale_B
                        )

                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                    T.clear(C_local)
                T.copy(C_local_accum, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return fp8_gemm_tiled_kernel_

    _KERNELS[key] = _build()
    return _KERNELS[key]


def _run_tiled(a, a_s, b, b_s, scale_dtype, tile):

    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), (
        "Scaling factor tensors must be contiguous"
    )
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp8_gemm_tiled_kernel(N, K, _tl_dtype(scale_dtype), tile)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c


def _auto_tile(N):

    if N // 128 >= _TARGET_BLOCKS:
        return None
    for bn in (64, 32, 16):
        if N // bn >= _TARGET_BLOCKS:
            return (bn, _SHIPPED_TILE[1], _SHIPPED_TILE[2])
    return (16, _SHIPPED_TILE[1], _SHIPPED_TILE[2])


_GB10_TILES = {
    (8192, 1024): (16, 1, 128),
    (2048, 4096): (32, 1, 128),
    (4096, 2048): (128, 1, 128),
}


def _selected_tile(N, K, tl_dtype):
    if _GEMV_MODE == "gb10":
        if (
            tl_dtype == "float8_e8m0fnu"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 1)
        ):
            return _GB10_TILES.get((N, K), _auto_tile(N))
        return _auto_tile(N)
    return _auto_tile(N) if _GEMV_MODE == "auto" else _GEMV_MODE


def _probe_seeded(N, K, tl_dtype, M, seed=0x84F8):

    dev = "cuda"
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed + M)
    a = (
        torch.randn(M, K, device=dev, dtype=torch.float32, generator=gen)
        .clamp_(-3, 3)
        .to(torch.float8_e4m3fn)
    )
    wb = torch.randint(0, 256, (N, K), device=dev, dtype=torch.uint8, generator=gen)
    wb[(wb & 0x7F) == 0x7F] = 0
    w = wb.view(torch.float8_e4m3fn)
    if tl_dtype == "float8_e8m0fnu":
        a_s = torch.randint(
            120,
            135,
            (M, (K + 127) // 128),
            device=dev,
            dtype=torch.uint8,
            generator=gen,
        ).view(torch.float8_e8m0fnu)
        w_s = torch.randint(
            120,
            135,
            ((N + 127) // 128, (K + 127) // 128),
            device=dev,
            dtype=torch.uint8,
            generator=gen,
        ).view(torch.float8_e8m0fnu)
    else:
        a_s = (
            torch.rand(
                M, (K + 127) // 128, device=dev, dtype=torch.float32, generator=gen
            )
            + 0.5
        )
        w_s = (
            torch.rand(
                (N + 127) // 128,
                (K + 127) // 128,
                device=dev,
                dtype=torch.float32,
                generator=gen,
            )
            + 0.5
        )
    return a, a_s, w, w_s


def _probe_gemv(N, K, tl_dtype, tile):

    sd = torch.float8_e8m0fnu if tl_dtype == "float8_e8m0fnu" else torch.float32
    try:
        for M in range(1, _BLOCK_M + 1):
            a, a_s, w, w_s = _probe_seeded(N, K, tl_dtype, M)
            c_ref = _REF_FP8_GEMM(a, a_s, w, w_s, sd)
            c_tuned = _run_tiled(a, a_s, w, w_s, sd, tile)
            torch.cuda.synchronize()
            if not torch.equal(c_ref, c_tuned):
                d = (c_ref.float() - c_tuned.float()).abs().max().item()
                return False, (
                    f"tuned != vendored at M={M} (max|d| {d:.3e}) — the tile "
                    f"reassociates on this box"
                )
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _resolve_gemv(N, K, tl_dtype):

    if _GEMV_MODE is None:
        return None
    key = (N, K, tl_dtype)
    if key in _GEMV_VERDICTS:
        v = _GEMV_VERDICTS[key]
        return v if isinstance(v, tuple) else None

    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        return None
    tile = _selected_tile(N, K, tl_dtype)
    if tile is None:
        _GEMV_VERDICTS[key] = "shipped"
        print(
            f"[v4] V4_FP8_GEMV at N={N} K={K}: shipped grid already >= {_TARGET_BLOCKS} blocks — "
            f"vendored kernel serves",
            flush=True,
        )
        return None
    ok, why = _probe_gemv(N, K, tl_dtype, tile)
    _GEMV_VERDICTS[key] = tile if ok else False
    print(
        f"[v4] V4_FP8_GEMV={V4_FP8_GEMV} at N={N} K={K}: "
        + (
            f"PROVEN tile={tile} — torch.equal vs the vendored kernel at M=1/8/{_BLOCK_M}"
            if ok
            else f"DECLINED — {why}"
        ),
        flush=True,
    )
    return tile if ok else None


def _fp8_gemm_fast(a, a_s, b, b_s, scale_dtype=torch.float32):

    if a.is_cuda:
        K = a.size(-1)
        if a.numel() // K <= _BLOCK_M:
            tile = _resolve_gemv(b.size(0), K, _tl_dtype(scale_dtype))
            if tile is not None:
                return _run_tiled(a, a_s, b, b_s, scale_dtype, tile)
    return _REF_FP8_GEMM(a, a_s, b, b_s, scale_dtype)


_fp8_gemm_fast._v4_fp8_gemv = True


def gemv_status():

    if not V4_FP8_GEMV or V4_FP8_GEMV == "0":
        return "off"
    if _GEMV_MODE is None:
        return f"invalid({_GEMV_INVALID})"
    if not _GEMV_VERDICTS:
        return "armed"
    ok = sum(1 for v in _GEMV_VERDICTS.values() if v is not False)
    n = len(_GEMV_VERDICTS)
    return f"on/{ok}-of-{n}" if ok else f"declined/{n}"


def _lay_shared(e):

    if getattr(e, "_v4_w13", None) is not None:
        return False
    w1, w3 = getattr(e, "w1", None), getattr(e, "w3", None)
    if (
        w1 is None
        or w3 is None
        or w1.weight.dtype != torch.float8_e4m3fn
        or w3.weight.dtype != torch.float8_e4m3fn
    ):
        return False
    inter, dim = w1.weight.shape
    if tuple(w3.weight.shape) != (inter, dim) or inter % 128 or dim % 128:
        return False
    dev = w1.weight.device
    with torch.no_grad():
        bank = torch.empty(2 * inter, dim, dtype=w1.weight.dtype, device=dev)
        sbank = torch.empty(
            2 * (inter // 128), dim // 128, dtype=w1.scale.dtype, device=dev
        )

        bank[:inter].view(torch.uint8).copy_(w1.weight.detach().view(torch.uint8))
        bank[inter:].view(torch.uint8).copy_(w3.weight.detach().view(torch.uint8))
        sbank[: inter // 128].view(torch.uint8).copy_(
            w1.scale.detach().view(torch.uint8)
        )
        sbank[inter // 128 :].view(torch.uint8).copy_(
            w3.scale.detach().view(torch.uint8)
        )
        w1.weight.data = bank[:inter]
        w3.weight.data = bank[inter:]
        w1.scale.data = sbank[: inter // 128]
        w3.scale.data = sbank[inter // 128 :]
    e._v4_w13 = (bank, sbank)
    return True


def shared_bank_layout(module):

    if not V4_FP8_SHARED:
        return 0
    mods = module.modules() if hasattr(module, "modules") else [module]
    return sum(
        1
        for m in mods
        if getattr(m, "shared_experts", None) is not None
        and _lay_shared(m.shared_experts)
    )


def _probe_shared(N2, K, tl_dtype):

    sd = torch.float8_e8m0fnu if tl_dtype == "float8_e8m0fnu" else torch.float32
    inter = N2 // 2
    try:
        for M in range(1, _BLOCK_M + 1):
            a, a_s, w13, s13 = _probe_seeded(N2, K, tl_dtype, M, seed=0x5A13)
            fused = _fp8_gemm_fast(a, a_s, w13, s13, sd)
            lo = _REF_FP8_GEMM(a, a_s, w13[:inter], s13[: inter // 128], sd)
            hi = _REF_FP8_GEMM(a, a_s, w13[inter:], s13[inter // 128 :], sd)
            torch.cuda.synchronize()
            if not (
                torch.equal(fused[..., :inter], lo)
                and torch.equal(fused[..., inter:], hi)
            ):
                d = (fused.float() - torch.cat([lo, hi], -1).float()).abs().max().item()
                return False, f"fused != per-matrix at M={M} (max|d| {d:.3e})"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _resolve_shared(N2, K, tl_dtype):

    key = (N2, K, tl_dtype)
    v = _SHARED_VERDICTS.get(key)
    if v is not None:
        return v
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        return False
    ok, why = _probe_shared(N2, K, tl_dtype)
    _SHARED_VERDICTS[key] = ok
    print(
        f"[v4] V4_FP8_SHARED at [{N2},{K}]: "
        + (
            "PROVEN — one launch, torch.equal per half vs the vendored per-matrix calls"
            if ok
            else f"DECLINED — {why}"
        ),
        flush=True,
    )
    return ok


def _w13_gemm(x, w13, s13):

    xq, xs = _MOD.act_quant(x, _MOD.block_size, _MOD.scale_fmt, _MOD.scale_dtype)
    sd = _MOD.scale_dtype
    inter = w13.size(0) // 2
    if not x.is_cuda:
        lo = _REF_FP8_GEMM(xq, xs, w13[:inter], s13[: inter // 128], sd)
        hi = _REF_FP8_GEMM(xq, xs, w13[inter:], s13[inter // 128 :], sd)
        return torch.cat([lo, hi], dim=-1)
    if not _resolve_shared(w13.size(0), x.size(-1), _tl_dtype(sd)):
        return None
    return _fp8_gemm_fast(xq, xs, w13, s13, sd)


def _decline_shared(e, why, x, weights):

    tally = getattr(e, "_shared_declined", None)
    if tally is None:
        tally = e._shared_declined = {}
    tally[why] = tally.get(why, 0) + 1
    return _REF_EXPERT_FORWARD(e, x, weights)


def _shared_forward(self, x, weights=None):

    bank = getattr(self, "_v4_w13", None)
    if bank is None:
        return _REF_EXPERT_FORWARD(self, x, weights)
    if x.numel() // x.size(-1) > _BLOCK_M:
        return _decline_shared(self, "m>32", x, weights)
    both = _w13_gemm(x, *bank)
    if both is None:
        return _decline_shared(self, "gate-declined", x, weights)
    dtype = x.dtype
    inter = bank[0].size(0) // 2
    gate = both[..., :inter].float()
    up = both[..., inter:].float()
    if self.swiglu_limit > 0:
        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        gate = torch.clamp(gate, max=self.swiglu_limit)
    h = torch.nn.functional.silu(gate) * up
    if weights is not None:
        h = weights * h
    self._shared_steps = getattr(self, "_shared_steps", 0) + 1
    return self.w2(h.to(dtype))


_shared_forward._v4_fp8_shared = True


def shared_status():

    if not V4_FP8_SHARED:
        return "off"
    if not _SHARED_VERDICTS:
        return "armed"
    ok = sum(1 for v in _SHARED_VERDICTS.values() if v)
    n = len(_SHARED_VERDICTS)
    return f"on/{ok}-of-{n}" if ok else f"declined/{n}"


def install(mod):

    global _REF_FP8_GEMM, _REF_EXPERT_FORWARD, _MOD
    _MOD = mod
    if _REF_FP8_GEMM is None and not getattr(mod.fp8_gemm, "_v4_fp8_gemv", False):
        _REF_FP8_GEMM = mod.fp8_gemm
    took_gemv = False
    if (
        _GEMV_MODE is not None
        and torch.cuda.is_available()
        and not getattr(mod.fp8_gemm, "_v4_fp8_gemv", False)
    ):
        mod.fp8_gemm = _fp8_gemm_fast
        took_gemv = True
    took_shared = False
    if V4_FP8_SHARED and not getattr(mod.Expert.forward, "_v4_fp8_shared", False):
        _REF_EXPERT_FORWARD = mod.Expert.forward
        mod.Expert.forward = _shared_forward
        took_shared = True
    return took_gemv, took_shared


def _smoke():

    assert torch.cuda.is_available(), "v4 fp8 gemv smoke needs a CUDA device"
    import wanferenz.kernels.fp4_grouped as fp4_grouped

    global _GEMV_MODE, V4_FP8_GEMV
    mod = fp4_grouped._load_model_module()
    if _GEMV_MODE is None:
        V4_FP8_GEMV, _GEMV_MODE = "auto", "auto"
    globals()["V4_FP8_SHARED"] = True
    install(mod)
    shapes = (
        ("attn.wq_a", 1024, 4096),
        ("attn.wkv", 512, 4096),
        ("attn.wq_b", 32768, 1024),
        ("attn.wo_b", 4096, 8192),
        ("idx.wq_b", 8192, 1024),
        ("shared.w13", 4096, 4096),
        ("shared.w2", 4096, 2048),
    )
    for name, N, K in shapes:
        tile = _resolve_gemv(N, K, "float8_e8m0fnu")
        v = _GEMV_VERDICTS[(N, K, "float8_e8m0fnu")]
        print(f"{name:12s} N={N:6d} K={K:5d}  -> {'tile ' + str(tile) if tile else v}")
    print(f"gemv_status  {gemv_status()}")

    args = fp4_grouped.real_dims_args(mod)
    torch.manual_seed(3)
    with mod.set_dtype(torch.bfloat16), torch.device("cuda"):
        ref_e = mod.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        fus_e = mod.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
    with torch.no_grad():
        for lin, (out_f, in_f) in (
            (ref_e.w1, (args.moe_inter_dim, args.dim)),
            (ref_e.w3, (args.moe_inter_dim, args.dim)),
            (ref_e.w2, (args.dim, args.moe_inter_dim)),
        ):
            w, s = fp4_grouped._fp8_block_quant(
                torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda") * 0.02
            )
            lin.weight.data.copy_(w)
            lin.scale.data.copy_(s)
        for k in ("w1", "w2", "w3"):
            getattr(fus_e, k).weight.data.copy_(getattr(ref_e, k).weight.detach())
            getattr(fus_e, k).scale.data.copy_(getattr(ref_e, k).scale.detach())
    assert _lay_shared(fus_e), (
        "the w13 bank layout must take on a shipped-dims fp8 expert"
    )
    for t in range(8):
        x = torch.randn(1, args.dim, dtype=torch.bfloat16, device="cuda")
        with torch.no_grad():
            ref = _REF_EXPERT_FORWARD(ref_e, x)
            got = _shared_forward(fus_e, x)
        eq = torch.equal(ref, got)
        print(
            f"shared draw {t}  equal={eq}  "
            f"max|d|={(ref.float() - got.float()).abs().max().item():.3e}"
        )
        assert eq, "the fused shared expert is not bit-exact to the reference"
    print(f"shared_status  {shared_status()}   steps={fus_e._shared_steps}")
    print("OK")


if __name__ == "__main__":
    _smoke()
