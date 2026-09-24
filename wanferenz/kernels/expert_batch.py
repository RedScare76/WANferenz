import os

import torch

V4_MOE_MULTI = os.environ.get("V4_MOE_MULTI", "0") not in ("", "0")


V4_MOE_MULTI_MAX = int(os.environ.get("V4_MOE_MULTI_MAX", "32"))

_REF_FORWARD = None
_WORLD_SIZE = 1


def multi_forward(self, x, input_ids):

    shape = x.size()
    xv = x.view(-1, self.dim)
    T = xv.size(0)
    if T == 1 or T > V4_MOE_MULTI_MAX or _WORLD_SIZE > 1:
        return _REF_FORWARD(self, x, input_ids)

    weights, indices = self.gate(xv, input_ids.flatten())

    sel = indices.tolist()
    buckets = {}
    for t, row in enumerate(sel):
        for c, e in enumerate(row):
            b = buckets.get(e)
            if b is None:
                buckets[e] = b = ([], [])
            b[0].append(t)
            b[1].append(c)

    order = [
        i for i in sorted(buckets) if self.experts_start_idx <= i < self.experts_end_idx
    ]

    flat = [t for i in order for t in buckets[i][0]] + [
        c for i in order for c in buckets[i][1]
    ]
    pairs = len(flat) // 2
    both = torch.tensor(flat, dtype=torch.long, device=xv.device)
    idx_all, top_all = both[:pairs], both[pairs:]

    y = torch.zeros_like(xv, dtype=torch.float32)
    off = 0
    for i in order:
        n = len(buckets[i][0])
        idx, top = idx_all[off : off + n], top_all[off : off + n]
        off += n
        y[idx] += self.experts[i](xv[idx], weights[idx, top, None])
    y += self.shared_experts(xv)
    return y.type_as(xv).view(shape)


def install(mod):

    global _REF_FORWARD, _WORLD_SIZE
    if not V4_MOE_MULTI or getattr(mod.MoE.forward, "_v4_multi", False):
        return False
    _REF_FORWARD = mod.MoE.forward
    _WORLD_SIZE = int(getattr(mod, "world_size", 1) or 1)
    multi_forward._v4_multi = True
    mod.MoE.forward = multi_forward
    return True


def uninstall(mod):

    global _REF_FORWARD
    if _REF_FORWARD is None:
        return False
    mod.MoE.forward = _REF_FORWARD
    _REF_FORWARD = None
    return True


def verify_locally():

    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wanferenz.model.oracle as oracle
    import wanferenz.serving.partition as partition
    import wanferenz.decoding.dspark as D

    import wanferenz.kernels.expert_batch as MM

    args = oracle.miniature_parameters()
    M = partition.ref()
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
        return st, dr

    def run():
        st, dr = build()
        torch.manual_seed(0)
        ids = torch.randint(0, args.vocab_size, (1, 13))
        tok = st.logits_all(
            st.forward(st.embed(ids), ids, 0), full_logits=False
        ).argmax(-1)
        dr.prefill(tok, st.tail_main_hidden())
        out = []
        for i in range(13, 17):
            h = st.forward(st.embed(tok.unsqueeze(1)), tok.unsqueeze(1), i)
            tok = st.logits_all(h, full_logits=False).argmax(-1)
            blk, conf = dr.advance_and_draft(
                tok.unsqueeze(1), st.tail_main_hidden(), start_pos=i
            )
            out.append((blk.clone(), conf.clone(), dr.last_spec[1].clone()))
        return out, [b.attn.kv_cache.clone() for b in dr.mtp]

    MM.uninstall(M)
    MM.V4_MOE_MULTI = False
    ref_out, ref_kv = run()
    assert not getattr(M.MoE.forward, "_v4_multi", False), (
        "the baseline leg must not be the lever"
    )
    MM.V4_MOE_MULTI = True
    assert MM.install(M), "install must take when V4_MOE_MULTI is on"
    try:
        got_out, got_kv = run()
    finally:
        MM.uninstall(M)
    for i, ((rb, rc, rl), (gb, gc, gl)) in enumerate(zip(ref_out, got_out)):
        assert torch.equal(rb, gb), f"draft ids diverged round {i}"
        assert torch.equal(rc, gc), f"confidence diverged round {i}"
        assert torch.equal(rl, gl), f"draft logits diverged round {i}"
    for i, (rk, gk) in enumerate(zip(ref_kv, got_kv)):
        assert torch.equal(rk, gk), f"mtp {i} kv_cache diverged"
    print(
        f"[v4] drafter MoE: bit-exact over {len(ref_out)} drafted rounds "
        f"(s={args.dspark_block_size}, {args.n_mtp_layers} mtp blocks)",
        flush=True,
    )


if __name__ == "__main__":
    verify_locally()
