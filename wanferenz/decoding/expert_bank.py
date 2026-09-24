import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

V4_DSPARK_MOE = os.environ.get("V4_DSPARK_MOE", "0") not in ("", "0")

_MOD = None
_WORLD_SIZE = 1
_ROWS = {}


def _grouped():

    import wanferenz.kernels.fp4_grouped as fp4_grouped

    return fp4_grouped


def _row_of_pair(T, k, device):

    key = (T, k, str(device))
    r = _ROWS.get(key)
    if r is None:
        r = _ROWS[key] = torch.arange(T, device=device).repeat_interleave(k)
    return r


def _take_rows(t, idx):

    u = t.view(torch.uint8)
    return u[idx].view(t.dtype).view(len(idx), *t.shape[1:])


def _pair_gemms_cuda(xq, xs, flat, bank, scale_dtype):

    G = _grouped()
    rows = _row_of_pair(xq.size(0), flat.numel() // xq.size(0), xq.device)
    a, s = _take_rows(xq, rows), _take_rows(xs, rows)
    both = G.grouped_fp4_gemm(
        a,
        s,
        G._gather_fp(bank["w13"], flat),
        G._gather_fp(bank["w13_s"], flat),
        scale_dtype,
    )

    def w2(hq, hs):
        return G.grouped_fp4_gemm(
            hq,
            hs,
            G._gather_fp(bank["w2"], flat),
            G._gather_fp(bank["w2_s"], flat),
            scale_dtype,
        )

    return both, w2


def _pair_gemms_cpu(xq, xs, flat, bank, scale_dtype, inter):

    import importlib

    kernel = importlib.import_module("wanferenz.kernels.reference")
    T = xq.size(0)
    k = flat.numel() // T
    by_expert = {}
    for p, e in enumerate(flat.tolist()):
        by_expert.setdefault(e, []).append(p)

    def run(w_key, lo, hi, aq, as_, row_of):

        out = None
        for e in sorted(by_expert):
            ps = by_expert[e]
            rows = torch.tensor([row_of(p) for p in ps], dtype=torch.long)
            a, s = _take_rows(aq, rows), _take_rows(as_, rows)
            w = bank[w_key][e, lo:hi].contiguous()
            ws = bank[w_key + "_s"][e, lo:hi].contiguous()
            c = kernel.fp4_gemm(a, s, w, ws, scale_dtype)
            if out is None:
                out = c.new_empty(flat.numel(), c.size(-1))
            out[torch.tensor(ps, dtype=torch.long)] = c
        return out

    g1 = run("w13", 0, inter, xq, xs, lambda p: p // k)
    u3 = run("w13", inter, 2 * inter, xq, xs, lambda p: p // k)
    both = torch.cat([g1, u3], dim=-1)

    def w2(hq, hs):
        return run("w2", 0, bank["w2"].size(1), hq, hs, lambda p: p)

    return both, w2


def _decline(self, why, x, input_ids):

    tally = getattr(self, "_draft_declined", None)
    if tally is None:
        tally = self._draft_declined = {}
    tally[why] = tally.get(why, 0) + 1
    return type(self).forward(self, x, input_ids)


def draft_forward(self, x, input_ids):

    shape = x.size()
    xv = x.view(-1, self.dim)
    T = xv.size(0)
    k = self.n_activated_experts
    if T <= 1:
        return _decline(self, "s<=1", x, input_ids)
    if _WORLD_SIZE > 1:
        return _decline(self, "world_size>1", x, input_ids)
    if self.gate.hash:
        return _decline(self, "hash-routed", x, input_ids)
    if T * k > _grouped()._BLOCK_M:
        return _decline(self, "pairs>block_M", x, input_ids)
    bank = getattr(self, "_grouped_bank", None)
    if not bank:
        return _decline(self, "no-bank", x, input_ids)
    inter = bank["w13"].size(1) // 2
    if xv.is_cuda and (
        bank["w13"].size(1) % _grouped()._BLOCK_N
        or bank["w2"].size(1) % _grouped()._BLOCK_N
    ):
        return _decline(self, "N%block_N", x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())

    order = torch.argsort(indices, dim=1)
    flat = indices.gather(1, order).reshape(-1)
    wts = weights.gather(1, order).reshape(-1, 1)

    scale_fmt, scale_dtype = _MOD.scale_fmt, _MOD.scale_dtype
    act_quant = _MOD.act_quant

    xq, xs = act_quant(xv, _MOD.block_size, scale_fmt, scale_dtype)
    pair = _pair_gemms_cuda if xv.is_cuda else _pair_gemms_cpu
    args = (xq, xs, flat, bank, scale_dtype) + (() if xv.is_cuda else (inter,))
    both, w2 = pair(*args)

    g = both[:, :inter].float()
    u = both[:, inter:].float()
    lim = self.experts[self.experts_start_idx].swiglu_limit
    if lim > 0:
        u = torch.clamp(u, min=-lim, max=lim)
        g = torch.clamp(g, max=lim)
    h = torch.nn.functional.silu(g) * u
    h = wts * h
    h = h.to(xv.dtype)
    hq, hs = act_quant(h, _MOD.block_size, scale_fmt, scale_dtype)
    out = w2(hq, hs)

    y = torch.zeros_like(xv, dtype=torch.float32)
    outT = out.view(T, k, -1)
    for j in range(k):
        y += outT[:, j]
    y += self.shared_experts(xv)
    self._draft_steps = getattr(self, "_draft_steps", 0) + 1
    return y.type_as(xv).view(shape)


def install_drafter(dstail):

    if not V4_DSPARK_MOE:
        return 0
    global _MOD, _WORLD_SIZE
    import wanferenz.serving.partition as partition

    _MOD = partition.ref()
    _WORLD_SIZE = int(getattr(_MOD, "world_size", 1) or 1)
    G = _grouped()
    took = 0
    for blk in dstail.mtp:
        moe = getattr(blk, "ffn", None)
        if moe is None or moe.gate.hash:
            continue
        e0 = moe.experts[moe.experts_start_idx]
        if e0.w1.weight.dtype != torch.float4_e2m1fn_x2:
            continue
        if not getattr(moe, "_grouped_bank", None):
            G._relayout_moe(moe, preserve=False)
        if not getattr(moe, "_grouped_bank", None):
            continue
        moe.forward = types.MethodType(draft_forward, moe)
        took += 1
    return took


def coverage(dstail):

    out = {}
    for blk in dstail.mtp:
        moe = blk.ffn
        out[moe.layer_id] = (
            getattr(moe, "_draft_steps", 0),
            dict(getattr(moe, "_draft_declined", {})),
        )
    return out


def swap_in_fp4_moes(dstail, moe_inter_dim=128, seed=0):

    from wanferenz.kernels.reference import fp4_act_quant
    import wanferenz.serving.partition as partition

    mod = partition.ref()
    a0 = dstail.args
    a = mod.ModelArgs(
        dim=a0.dim,
        moe_inter_dim=moe_inter_dim,
        n_routed_experts=a0.n_routed_experts,
        n_activated_experts=a0.n_activated_experts,
        n_shared_experts=1,
        n_hash_layers=0,
        score_func=a0.score_func,
        route_scale=a0.route_scale,
        swiglu_limit=a0.swiglu_limit,
        expert_dtype="fp4",
        dtype="bf16",
        scale_fmt=None,
        scale_dtype="fp32",
        vocab_size=a0.vocab_size,
    )
    g = torch.Generator().manual_seed(seed)
    for bi, blk in enumerate(dstail.mtp):
        with mod.set_dtype(torch.bfloat16):
            moe = mod.MoE(blk.layer_id, a)
        with torch.no_grad():
            for i in range(a.n_routed_experts):
                e = moe.experts[i]
                for lin, out_f, in_f in (
                    (e.w1, a.moe_inter_dim, a.dim),
                    (e.w3, a.moe_inter_dim, a.dim),
                    (e.w2, a.dim, a.moe_inter_dim),
                ):
                    w, s = fp4_act_quant(
                        torch.randn(out_f, in_f, generator=g, dtype=torch.bfloat16),
                        mod.fp4_block_size,
                    )
                    lin.weight.data.copy_(w)
                    lin.scale.data.copy_(s)
            for lin in (
                moe.shared_experts.w1,
                moe.shared_experts.w2,
                moe.shared_experts.w3,
            ):
                lin.weight.data.normal_(0, 0.02, generator=g)
            moe.gate.weight.data.normal_(0, 0.02, generator=g)
            moe.gate.bias.data.normal_(0, 0.02, generator=g)

        assert _grouped()._relayout_moe(moe, preserve=True), "fp4 toy bank must take"
        blk.ffn = moe.eval()
    return dstail


def verify_locally():

    import wanferenz.model.oracle as oracle
    import wanferenz.serving.partition as partition
    import wanferenz.decoding.dspark as D
    import wanferenz.decoding.expert_bank as DM

    args = oracle.miniature_parameters(
        n_routed_experts=64,
        n_activated_experts=6,
        dspark_block_size=5,
        n_mtp_layers=3,
        compress_ratios=(0, 0, 4, 8, 4, 8, 4, 0, 0, 0, 0),
    )
    oracle = oracle.create_oracle(args, 0)

    def build():
        st = partition.LayerPartition(
            0, args.n_layers, args, head=True, tail=True, dspark=True, device="cpu"
        )
        for li in range(args.n_layers):
            st.layers[li].load_state_dict(oracle.layers[li].state_dict(), strict=True)
        st.embed_tokens.load_state_dict(oracle.embed.state_dict(), strict=True)
        st.norm.load_state_dict(oracle.norm.state_dict(), strict=True)
        st.lm_head.load_state_dict(oracle.head.state_dict(), strict=True)
        with torch.no_grad():
            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                getattr(st, n).data.copy_(getattr(oracle, n).data)
        st._dspark = True
        dr = D.DSparkPartition(st)
        for k, blk in enumerate(dr.mtp):
            sd = {
                n: v
                for n, v in oracle.mtp[k].state_dict().items()
                if n not in D.ALIAS_KEYS
            }
            blk.load_state_dict(sd, strict=False)
        DM.swap_in_fp4_moes(dr, moe_inter_dim=128, seed=7)
        return st, dr

    def run(dr_mutate=None):
        st, dr = build()
        if dr_mutate:
            dr_mutate(dr)
        torch.manual_seed(0)
        ids = torch.randint(0, args.vocab_size, (1, 13))
        tok = st.logits_all(
            st.forward(st.embed(ids), ids, 0), full_logits=False
        ).argmax(-1)
        dr.prefill(tok, st.tail_main_hidden())
        out = []
        for i in range(13, 18):
            h = st.forward(st.embed(tok.unsqueeze(1)), tok.unsqueeze(1), i)
            tok = st.logits_all(h, full_logits=False).argmax(-1)
            dr.advance_and_draft(tok.unsqueeze(1), st.tail_main_hidden(), start_pos=i)
            out.append(tuple(t.clone() for t in dr.last_spec))
        return out, [b.attn.kv_cache.clone() for b in dr.mtp], dr

    DM.V4_DSPARK_MOE = False
    ref_out, ref_kv, _ = run()

    DM.V4_DSPARK_MOE = True
    banked = []

    def arm(dr):
        banked.append(DM.install_drafter(dr))

    got_out, got_kv, dr2 = run(arm)
    assert banked == [3], f"install must claim all 3 drafter MoEs, took {banked}"
    for i, (a, b) in enumerate(zip(ref_out, got_out)):
        for x, y, what in zip(a, b, ("output_ids", "logits", "confidence")):
            assert torch.equal(x, y), f"drafter {what} diverged at round {i}"
    for i, (rk, gk) in enumerate(zip(ref_kv, got_kv)):
        assert torch.equal(rk, gk), f"mtp {i} kv_cache diverged"
    cov = coverage(dr2)
    assert all(steps == len(ref_out) and not dec for steps, dec in cov.values()), (
        f"the pair path must serve EVERY round of EVERY block: {cov}"
    )
    print(
        f"[v4 dspark moe] pair path bit-exact over {len(ref_out)} drafted rounds "
        f"(T={args.dspark_block_size} x k={args.n_activated_experts} pairs, "
        f"{args.n_mtp_layers} fp4 mtp blocks), coverage {cov}",
        flush=True,
    )


if __name__ == "__main__":
    verify_locally()
