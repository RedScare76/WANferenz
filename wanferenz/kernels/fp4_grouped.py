import os

import torch

V4_MOE_GROUPED = os.environ.get("V4_MOE_GROUPED", "0") not in ("", "0")


_REF_FORWARD = None
_MOD = None
_WORLD_SIZE = 1

_KERNELS = {}


_BLOCK_M, _BLOCK_N = 32, 128


def grouped_fp4_gemm_kernel(N, K, scale_dtype="float32"):

    key = (N, K, scale_dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    import tilelang
    import tilelang.language as T

    tilelang.set_log_level("WARNING")
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
    FP8, FP4, FP32, BF16 = "float8_e4m3", "float4_e2m1fn", "float32", "bfloat16"
    out_dtype, accum_dtype = BF16, FP32
    act_group_size = 128
    weight_group_size = 32
    block_M = _BLOCK_M
    block_N = _BLOCK_N
    block_K = 32
    n_sub = act_group_size // block_K

    G = T.symbolic("G")

    @tilelang.jit(pass_configs=pass_configs)
    def _build():

        @T.prim_func
        def grouped_fp4_gemm_kernel_(
            A: T.Tensor[(G, K), FP8],
            W: T.Tensor[(G, N, K), FP4],
            C: T.Tensor[(G, N), out_dtype],
            scales_a: T.Tensor[(G, T.ceildiv(K, act_group_size)), scale_dtype],
            scales_w: T.Tensor[(G, N, T.ceildiv(K, weight_group_size)), scale_dtype],
        ):
            with T.Kernel(T.ceildiv(N, block_N), G, threads=128) as (bx, g):
                A_shared = T.alloc_shared((block_M, block_K), FP8)
                B_fp4_shared = T.alloc_shared((block_N, block_K), FP4)
                B_shared = T.alloc_shared((block_N, block_K), FP8)
                C_shared = T.alloc_shared((block_M, block_N), out_dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_local_accum = T.alloc_fragment((block_M, block_N), accum_dtype)
                scale_a_frag = T.alloc_fragment((block_M,), FP32)
                scale_b_frag = T.alloc_fragment((block_N,), FP32)

                T.use_swizzle(panel_size=10)
                T.clear(C_local)
                T.clear(C_local_accum)

                K_iters = T.ceildiv(K, block_K)
                for k in T.Pipelined(K_iters, num_stages=2):
                    T.copy(A[0, k * block_K], A_shared)
                    T.copy(W[g, bx * block_N, k * block_K], B_fp4_shared)

                    for i, j in T.Parallel(block_N, block_K):
                        B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))
                    for i in T.Parallel(block_N):
                        scale_b_frag[i] = T.Cast(FP32, scales_w[g, bx * block_N + i, k])
                    for i in T.Parallel(block_M):
                        scale_a_frag[i] = T.Cast(FP32, scales_a[i, k // n_sub])
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                    for i, j in T.Parallel(block_M, block_N):
                        C_local_accum[i, j] += (
                            C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]
                        )
                    T.clear(C_local)

                T.copy(C_local_accum, C_shared)

                for j in T.Parallel(block_N):
                    C[g, bx * block_N + j] = C_shared[g, j]

        return grouped_fp4_gemm_kernel_

    _KERNELS[key] = _build()
    return _KERNELS[key]


def grouped_fp4_gemm(a, a_s, w, w_s, scale_dtype=torch.float32):

    assert a.is_contiguous() and w.is_contiguous(), (
        "grouped fp4: a and w must be contiguous"
    )
    assert a_s.is_contiguous() and w_s.is_contiguous(), (
        "grouped fp4: scales must be contiguous"
    )
    assert w.size(1) % _BLOCK_N == 0, (
        f"grouped fp4: N={w.size(1)} is not a whole number of {_BLOCK_N}-wide tiles — the kernel's "
        f"store loop is unpredicated and the last block would write past the row"
    )
    assert a.size(0) <= _BLOCK_M, (
        f"grouped fp4: {a.size(0)} expert slots exceeds the block_M={_BLOCK_M} A tile the kernel "
        f"copies; only rows below it can be stored"
    )
    if not a.is_cuda:
        import importlib

        kernel = importlib.import_module("wanferenz.kernels.reference")
        return torch.cat(
            [
                kernel.fp4_gemm(a[g : g + 1], a_s[g : g + 1], w[g], w_s[g], scale_dtype)
                for g in range(a.size(0))
            ]
        )
    tl_dtype = "float8_e8m0fnu" if scale_dtype == torch.float8_e8m0fnu else "float32"
    G, K = a.shape
    N = w.size(1)
    c = a.new_empty(G, N, dtype=torch.get_default_dtype())
    kernel = grouped_fp4_gemm_kernel(N, K, tl_dtype)
    kernel(a, w, c, a_s, w_s)
    return c


def _gather_fp(t, ids):

    return t.view(torch.uint8)[ids.long()].view(t.dtype)


_BANK_GROUPS = (("w13", ("w1", "w3")), ("w2", ("w2",)))


_EXPERT_KINDS = tuple(k for _key, kinds in _BANK_GROUPS for k in kinds)


def _relayout_moe(moe, preserve=True):

    if getattr(moe, "_grouped_bank", None):
        return False
    lo, hi = moe.experts_start_idx, moe.experts_end_idx
    experts = [moe.experts[i] for i in range(lo, hi)]
    if not experts or experts[0].w1.weight.dtype != torch.float4_e2m1fn_x2:
        return False

    capacity = []
    for key, kinds in _BANK_GROUPS:
        for attr, suffix in (("weight", ""), ("scale", "_s")):
            first = [getattr(getattr(experts[0], k), attr) for k in kinds]
            rows = [t.shape[0] for t in first]
            t0 = first[0]
            params = [[getattr(getattr(e, k), attr) for k in kinds] for e in experts]
            shape = (len(experts), sum(rows)) + tuple(t0.shape[1:])
            capacity.append((key + suffix, params, rows, shape, t0.dtype, t0.device))
    cuda = experts[0].w1.weight.is_cuda
    if not preserve:
        void = {}
        for _key, params, _rows, _shape, dtype, device in capacity:
            v = void.setdefault(
                (dtype, device), torch.empty(0, dtype=dtype, device=device)
            )
            for group in params:
                for p in group:
                    p.data = v
        if cuda:
            torch.cuda.empty_cache()
    bank = {}
    with torch.no_grad():
        for key, params, rows, shape, dtype, device in capacity:
            b = torch.empty(shape, dtype=dtype, device=device)
            bu = b.view(torch.uint8)
            for j, group in enumerate(params):
                off = 0
                for n, p in zip(rows, group):
                    if preserve:
                        bu[j, off : off + n].copy_(p.detach().view(torch.uint8))
                    p.data = b[j, off : off + n]
                    off += n
            bank[key] = b
            if preserve and cuda:
                torch.cuda.empty_cache()
    moe._grouped_bank = bank
    return True


def bank_layout(module, preserve=True):

    if not V4_MOE_GROUPED:
        return 0
    return sum(
        1
        for m in module.modules()
        if hasattr(m, "experts")
        and hasattr(m, "experts_start_idx")
        and _relayout_moe(m, preserve)
    )


_BANK_HEADROOM_BYTES = 2 << 30


def _bank_fits(experts):

    if not torch.cuda.is_available():
        return True
    need = sum(
        t.numel() * t.element_size()
        for e in experts
        for k in _EXPERT_KINDS
        for t in (getattr(e, k).weight, getattr(e, k).scale)
    )
    free, _total = torch.cuda.mem_get_info()
    return free - need >= _BANK_HEADROOM_BYTES


def _expert_bank(moe):

    bank = getattr(moe, "_grouped_bank", None)
    if bank is not None:
        return bank if bank is not False else None
    lo, hi = moe.experts_start_idx, moe.experts_end_idx
    experts = [moe.experts[i] for i in range(lo, hi)]
    if not _bank_fits(experts):
        moe._grouped_bank = False
        print(
            f"[v4] grouped MoE declined on layer {getattr(moe, 'layer_id', '?')} — stacking the "
            f"expert bank would not fit beside the per-expert weights; this layer stays on the "
            f"decode path. A stage gets the bank from bank_layout() at load and never reaches "
            f"this; an MoE built outside LayerPartition does.",
            flush=True,
        )
        return None

    bank = {}
    for key, kinds in _BANK_GROUPS:
        for attr, suffix in (("weight", ""), ("scale", "_s")):
            dtype = getattr(getattr(experts[0], kinds[0]), attr).dtype
            stacked = torch.stack(
                [
                    torch.cat(
                        [getattr(getattr(e, k), attr).view(torch.uint8) for k in kinds]
                    )
                    for e in experts
                ]
            )
            bank[key + suffix] = stacked.contiguous().view(dtype)
    moe._grouped_bank = bank
    return bank


def _decline(moe, why, x, input_ids):

    tally = getattr(moe, "_grouped_declined", None)
    if tally is None:
        tally = moe._grouped_declined = {}
    tally[why] = tally.get(why, 0) + 1
    return _REF_FORWARD(moe, x, input_ids)


def _keep_last_of_each(ids):

    same = ids[:, None] == ids[None, :]
    return ~same.triu(1).any(dim=1)


def grouped_forward(self, x, input_ids):

    shape = x.size()
    xv = x.view(-1, self.dim)
    if xv.size(0) != 1:
        return _decline(self, "s>1", x, input_ids)
    if _WORLD_SIZE > 1:
        return _decline(self, "world_size>1", x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())
    ids = indices[0].to(torch.int32)
    bank = _expert_bank(self)
    if bank is None:
        return _decline(self, "bank-would-not-fit", x, input_ids)

    w13, w13_s = _gather_fp(bank["w13"], ids), _gather_fp(bank["w13_s"], ids)
    w2, w2_s = _gather_fp(bank["w2"], ids), _gather_fp(bank["w2_s"], ids)

    scale_fmt, scale_dtype = _MOD.scale_fmt, _MOD.scale_dtype
    block = _MOD.block_size
    act_quant = _MOD.act_quant

    G = ids.numel()
    xq1, xs1 = act_quant(xv, block, scale_fmt, scale_dtype)
    xq = xq1.expand(G, -1).contiguous()
    xs = xs1.expand(G, -1).contiguous()
    both = grouped_fp4_gemm(xq, xs, w13, w13_s, scale_dtype)
    inter = both.size(1) // 2
    gate6, up6 = both[:, :inter], both[:, inter:]

    g = gate6.float()
    u = up6.float()
    if self.experts[self.experts_start_idx].swiglu_limit > 0:
        lim = self.experts[self.experts_start_idx].swiglu_limit
        u = torch.clamp(u, min=-lim, max=lim)
        g = torch.clamp(g, max=lim)
    h = torch.nn.functional.silu(g) * u
    h = weights[0, :, None] * h
    h = h.to(xv.dtype)

    hq, hs = act_quant(h, block, scale_fmt, scale_dtype)
    out6 = grouped_fp4_gemm(hq, hs, w2, w2_s, scale_dtype)

    if self.gate.hash:
        keep = _keep_last_of_each(ids)[:, None]
        out6 = torch.where(keep, out6, torch.zeros_like(out6))

    out_sorted = out6[torch.argsort(ids, stable=True)]
    y = torch.zeros_like(xv, dtype=torch.float32)
    for slot in range(out_sorted.size(0)):
        y += out_sorted[slot : slot + 1]
    y += self.shared_experts(xv)
    self._grouped_steps = getattr(self, "_grouped_steps", 0) + 1
    return y.type_as(xv).view(shape)


def coverage(module):

    out = {}
    for m in module.modules() if hasattr(module, "modules") else [module]:
        if hasattr(m, "experts") and hasattr(m, "experts_start_idx"):
            out[getattr(m, "layer_id", len(out))] = (
                getattr(m, "_grouped_steps", 0),
                dict(getattr(m, "_grouped_declined", {})),
            )
    return out


def install(mod):

    global _REF_FORWARD, _MOD, _WORLD_SIZE
    if not V4_MOE_GROUPED or getattr(mod.MoE.forward, "_v4_grouped", False):
        return False
    if not torch.cuda.is_available():
        return False
    _REF_FORWARD = mod.MoE.forward
    _MOD = mod
    _WORLD_SIZE = int(getattr(mod, "world_size", 1) or 1)
    grouped_forward._v4_grouped = True
    mod.MoE.forward = grouped_forward
    return True


def _fp8_block_quant(w, block=128):

    out, inn = w.shape
    b = (
        w.float()
        .unflatten(0, (out // block, block))
        .unflatten(-1, (inn // block, block))
    )
    amax = b.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (b / scale[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(out, inn).contiguous(), scale.to(torch.float8_e8m0fnu).contiguous()


def build_real_dims_moe(mod, args, seed=0, layer_id=7, bank=False):

    from wanferenz.kernels.reference import fp4_act_quant

    torch.manual_seed(seed)
    with mod.set_dtype(torch.bfloat16), torch.device("cuda"):
        moe = mod.MoE(layer_id, args)
    blk = mod.block_size

    def rand(*sh):
        return torch.randn(*sh, dtype=torch.bfloat16, device="cuda") * 0.02

    with torch.no_grad():
        for i in range(moe.experts_start_idx, moe.experts_end_idx):
            e = moe.experts[i]
            for lin, out_f, in_f in (
                (e.w1, args.moe_inter_dim, args.dim),
                (e.w3, args.moe_inter_dim, args.dim),
                (e.w2, args.dim, args.moe_inter_dim),
            ):
                w, s = fp4_act_quant(rand(out_f, in_f), mod.fp4_block_size)
                lin.weight.data.copy_(w)
                lin.scale.data.copy_(s)
        for lin, out_f, in_f in (
            (moe.shared_experts.w1, args.moe_inter_dim, args.dim),
            (moe.shared_experts.w3, args.moe_inter_dim, args.dim),
            (moe.shared_experts.w2, args.dim, args.moe_inter_dim),
        ):
            w, s = _fp8_block_quant(rand(out_f, in_f), blk)
            lin.weight.data.copy_(w)
            lin.scale.data.copy_(s)
        moe.gate.weight.data.copy_(
            rand(args.n_routed_experts, args.dim).to(moe.gate.weight.dtype)
        )
        if moe.gate.hash:
            moe.gate.tid2eid.data.random_(0, args.n_routed_experts)
        else:
            moe.gate.bias.data.normal_(0, 0.02)
    if bank:
        assert _relayout_moe(moe), "the bank layout must take on a shipped-dims fp4 MoE"
    return moe.eval()


def real_dims_args(mod):
    return mod.ModelArgs(
        dim=4096,
        moe_inter_dim=2048,
        n_routed_experts=256,
        n_activated_experts=6,
        n_shared_experts=1,
        n_hash_layers=3,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        dtype="fp8",
        scale_dtype="fp8",
        expert_dtype="fp4",
    )


from wanferenz.model.assets import reference_directory


def _load_model_module():

    import importlib.util
    import sys
    import wanferenz.kernels.torch_ops as torch_ops

    torch_ops.install()
    inf = reference_directory()
    if inf not in sys.path:
        sys.path.insert(0, inf)
    if torch_ops.backend() == "tilelang":
        import wanferenz.kernels.gb10 as gb10

        gb10.install()
    spec = importlib.util.spec_from_file_location(
        "dsv4_model", os.path.join(inf, "architecture.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv4_model"] = mod
    spec.loader.exec_module(mod)
    torch.set_default_dtype(torch.bfloat16)
    mod.world_size, mod.rank = 1, 0
    mod.default_dtype = torch.float8_e4m3fn
    mod.scale_fmt, mod.scale_dtype = "ue8m0", torch.float8_e8m0fnu
    return mod


def _smoke():

    assert torch.cuda.is_available(), "v4 moe grouped smoke needs a CUDA device"
    mod = _load_model_module()
    args = real_dims_args(mod)
    ref_moe = build_real_dims_moe(mod, args)
    bank_moe = build_real_dims_moe(mod, args, bank=True)
    assert bank_moe._grouped_bank, "the bank layout did not take"
    ref_forward = mod.MoE.forward
    install(mod)
    hash_ref = build_real_dims_moe(mod, args, seed=1, layer_id=0)
    hash_bank = build_real_dims_moe(mod, args, seed=1, layer_id=0, bank=True)

    pattern = torch.tensor([3, 3, 17, 3, 200, 17], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        for m in (hash_ref, hash_bank):
            m.gate.tid2eid.data.copy_(pattern.expand_as(m.gate.tid2eid))
    for tag, (a, b) in (
        ("decode", (ref_moe, bank_moe)),
        ("hash", (hash_ref, hash_bank)),
    ):
        for t in range(8):
            x = torch.randn(1, 1, args.dim, dtype=torch.bfloat16, device="cuda")
            ids = torch.randint(0, args.vocab_size, (1, 1), device="cuda")
            with torch.no_grad():
                ref = ref_forward(a, x, ids)
                got = grouped_forward(b, x, ids)
            eq = torch.equal(ref, got)
            d = (ref.float() - got.float()).abs().max().item()
            print(f"{tag} draw {t}  equal={eq}  max|d|={d:.3e}")
            assert eq, (
                f"grouped MoE ({tag}) on the bank layout is not bit-exact to the reference"
            )
    for t, s in enumerate((2, 5, 17)):
        x = torch.randn(1, s, args.dim, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, args.vocab_size, (1, s), device="cuda")
        with torch.no_grad():
            ref = ref_forward(ref_moe, x, ids)
            got = grouped_forward(bank_moe, x, ids)
        eq = torch.equal(ref, got)
        print(
            f"prefill s={s}  equal={eq}  max|d|={(ref.float() - got.float()).abs().max().item():.3e}"
        )
        assert eq, (
            "the s > 1 fallback over banked experts is not bit-exact to the reference"
        )
    print(
        "bit-exact over 8 score-routed + 8 hash-routed decode draws + 3 prefill shapes, "
        "on the bank layout"
    )


if __name__ == "__main__":
    _smoke()
