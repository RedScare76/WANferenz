import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


INDEXER_BUCKETS = (16, 64, 256, 1024, 4096, 16384, 65536)


V4_GRAPH_MAX = int(os.environ.get("V4_GRAPH_MAX", "192"))
_GRAPH_COUNT = 0
_GRAPH_SKIPPED = 0


V4_MOE_IN_GRAPH = os.environ.get("V4_MOE_IN_GRAPH", "0") not in ("", "0")
_MOE_GRAPHED = 0
_MOE_REFUSED = 0


def bucket_width(need, maxw, floor=0):

    need = max(need, floor)
    for b in INDEXER_BUCKETS:
        if b >= need:
            return min(b, maxw)
    return maxw


class _Ref:
    def __init__(self, M):
        self.apply_rotary_emb = M.apply_rotary_emb
        self.rotate_activation = M.rotate_activation
        self.act_quant = M.act_quant
        self.fp4_act_quant = M.fp4_act_quant
        self.sparse_attn = M.sparse_attn
        self.M = M

    @property
    def scale_fmt(self):
        return self.M.scale_fmt

    @property
    def scale_dtype(self):
        return self.M.scale_dtype

    @property
    def fp4_block_size(self):
        return self.M.fp4_block_size


def _compressor_decode_cs(R, C, x, pos, compress):

    ratio, overlap, d, rd = C.compress_ratio, C.overlap, C.head_dim, C.rope_head_dim
    slot_r = pos % ratio
    comp_slot = pos // ratio
    dtype = x.dtype
    x = x.float()
    kv = C.wkv(x)
    score = C.wgate(x)

    score = score + C.ape.index_select(0, slot_r)
    kv1 = kv.squeeze(1)
    score1 = score.squeeze(1)
    ks = C.kv_state.narrow(0, 0, 1)
    ss = C.score_state.narrow(0, 0, 1)
    if overlap:
        ks.index_copy_(1, ratio + slot_r, kv1.unsqueeze(1))
        ss.index_copy_(1, ratio + slot_r, score1.unsqueeze(1))
        if compress:
            kv_state = torch.cat([ks[:, :ratio, :d], ks[:, ratio:, d:]], dim=1)
            score_state = torch.cat([ss[:, :ratio, :d], ss[:, ratio:, d:]], dim=1)
            kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
            ks[:, :ratio].copy_(ks[:, ratio:])
            ss[:, :ratio].copy_(ss[:, ratio:])
    else:
        ks.index_copy_(1, slot_r, kv1.unsqueeze(1))
        ss.index_copy_(1, slot_r, score1.unsqueeze(1))
        if compress:
            kv = (ks * ss.softmax(dim=1)).sum(dim=1, keepdim=True)
    if not compress:
        return
    kv = C.norm(kv.to(dtype))
    freqs_row = C.freqs_cis.index_select(0, pos + 1 - ratio)
    R.apply_rotary_emb(kv[..., -rd:], freqs_row)
    if C.rotate:
        kv = R.rotate_activation(kv)
        R.fp4_act_quant(kv, R.fp4_block_size, True)
    else:
        R.act_quant(kv[..., :-rd], 64, R.scale_fmt, R.scale_dtype, True)
    C.kv_cache.narrow(0, 0, 1).index_copy_(1, comp_slot, kv)


def _indexer_decode_cs(
    R, I, x, qr, pos, end_ratio, freqs_row, offset, arange_w, compress, read_w
):

    import wanferenz.kernels.short_context as slim

    if (
        getattr(type(I).forward, "_v4_ref_slim", False)
        and slim._ACTIVE
        and I.kv_cache.size(1) <= I.index_topk
    ):
        _compressor_decode_cs(R, I.compressor, x, pos, compress)
        arange = arange_w.view(1, 1, -1)
        return torch.where(arange < end_ratio, arange + offset, -1).int()
    n_local_heads, head_dim, rd = I.n_local_heads, I.head_dim, I.rope_head_dim
    index_topk = I.index_topk
    q = I.wq_b(qr)
    q = q.unflatten(-1, (n_local_heads, head_dim))
    R.apply_rotary_emb(q[..., -rd:], freqs_row)
    q = R.rotate_activation(q)
    R.fp4_act_quant(q, R.fp4_block_size, True)
    _compressor_decode_cs(R, I.compressor, x, pos, compress)
    weights = I.weights_proj(x) * (I.softmax_scale * I.n_heads**-0.5)

    index_score = torch.einsum(
        "bshd,btd->bsht", q, I.kv_cache.narrow(0, 0, 1).narrow(1, 0, read_w)
    )
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    return select_compress_topk(index_score, end_ratio, index_topk, offset, arange_w)


def _select_topk_width_invariant(score, valid, k, arange_w):

    s = score.masked_fill(~valid, float("-inf"))
    pick = s.sort(dim=-1, descending=True, stable=True).indices.narrow(-1, 0, k)
    return pick, valid.gather(-1, pick)


def select_compress_topk(index_score, end_ratio, index_topk, offset, arange_w):

    arange = arange_w.view(1, 1, -1)
    valid = arange < end_ratio

    k = min(index_topk, index_score.shape[-1])
    pick, kept = _select_topk_width_invariant(index_score, valid, k, arange)
    return torch.where(kept, pick + offset, pick.new_full((), -1)).int()


def attn_decode_cs(R, A, x, pos, win_topk, comp_topk, arange_w, compress, read_w):

    win, ratio, rd = A.window_size, A.compress_ratio, A.rope_head_dim
    freqs_row = A.freqs_cis.index_select(0, pos)

    qr = q = A.q_norm(A.wq_a(x))
    q = A.wq_b(q).unflatten(-1, (A.n_local_heads, A.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + A.eps)
    R.apply_rotary_emb(q[..., -rd:], freqs_row)

    kv = A.wkv(x)
    kv = A.kv_norm(kv)
    R.apply_rotary_emb(kv[..., -rd:], freqs_row)
    R.act_quant(kv[..., :-rd], 64, R.scale_fmt, R.scale_dtype, True)
    topk_idxs = win_topk
    if ratio:
        end_ratio = (pos + 1) // ratio
        if A.indexer is not None:
            compress_topk_idxs = _indexer_decode_cs(
                R,
                A.indexer,
                x,
                qr,
                pos,
                end_ratio,
                freqs_row,
                win,
                arange_w,
                compress,
                read_w,
            )
        else:
            compress_topk_idxs = comp_topk
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

    A.kv_cache.narrow(0, 0, 1).index_copy_(1, pos % win, kv)
    if ratio:
        _compressor_decode_cs(R, A.compressor, x, pos, compress)
    o = R.sparse_attn(
        q, A.kv_cache.narrow(0, 0, 1), A.attn_sink, topk_idxs, A.softmax_scale
    )
    R.apply_rotary_emb(o[..., -rd:], freqs_row, True)

    o = o.view(1, 1, A.n_local_groups, -1)
    wo_a = A.wo_a.weight.view(A.n_local_groups, A.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return A.wo_b(o.flatten(2))


def moe_stub(L, x, ids=None):

    ffn = L.ffn
    shape = x.size()
    xf = x.view(-1, ffn.dim)
    y = ffn.shared_experts(xf)
    k = ffn.n_activated_experts
    picked = 0
    for i in range(ffn.experts_start_idx, ffn.experts_end_idx):
        if ffn.experts[i] is None or picked >= k:
            continue
        y = y + ffn.experts[i](xf)
        picked += 1
    return y.type_as(x).view(shape)


def real_moe(L, x, ids):

    return L.ffn(x, ids)


def grouped_at_one_token(mod):

    import wanferenz.serving.controls as controls

    chain = controls.moe_chain(mod)
    while chain and chain[0] == "multi":
        chain = chain[1:]
    return bool(chain) and chain[0] == "grouped"


def moe_probe(L, dev, dt, ids):

    moe = L.ffn
    steps0 = getattr(moe, "_grouped_steps", 0)
    declined0 = dict(getattr(moe, "_grouped_declined", {}) or {})
    x = torch.zeros(1, 1, moe.dim, dtype=dt, device=dev)
    with torch.no_grad():
        moe(x, ids)
    steps1 = getattr(moe, "_grouped_steps", 0)
    declined1 = dict(getattr(moe, "_grouped_declined", {}) or {})
    moe._grouped_steps = steps0
    if declined0:
        moe._grouped_declined = declined0
    elif getattr(moe, "_grouped_declined", None) is not None:
        moe._grouped_declined.clear()
    new = [k for k in declined1 if declined1[k] != declined0.get(k, 0)]
    if new:
        return f"the probe step DECLINED grouping: {new}"
    if steps1 != steps0 + 1:
        return (
            f"the probe step did not reach the grouped kernel "
            f"(_grouped_steps {steps0} -> {steps1}, expected +1)"
        )
    return None


def _moe_refusal(L, dev, dt, ids, mod):

    if not grouped_at_one_token(mod):
        import wanferenz.serving.controls as controls

        return (
            f"a single-token step does not reach the grouped MoE (live chain: "
            f"{'>'.join(controls.moe_chain(mod))}) — capturing a `.tolist()` dispatch would "
            f"freeze the capture step's expert set into every replay"
        )
    bank = getattr(L.ffn, "_grouped_bank", None)
    if not bank:
        return (
            "this layer has no load-time expert bank (bank_layout declined or never ran), so the "
            "first grouped step would STACK one — an allocation out of the graph's private pool "
            "that a later capture in that pool may hand to something else"
        )
    import wanferenz.kernels.fp4_grouped as fp4_grouped

    if int(getattr(fp4_grouped, "_WORLD_SIZE", 1) or 1) > 1:
        return "world_size > 1: the reference all-reduces the routed sum, and the grouped path cannot"
    if ids is None:
        return "no token ids: the first n_hash_layers route on tid2eid[input_ids]"
    return moe_probe(L, dev, dt, ids)


def moe_graph_coverage(stage):

    bgs = [
        bg
        for bg in (getattr(stage, "_block_graphs", None) or [])
        if getattr(bg, "moe_requested", False)
    ]
    got = [bg.moe_in_graph for bg in bgs]
    return (
        sum(g is True for g in got),
        sum(g is False for g in got),
        sum(g is None for g in got),
    )


def block_pre_cs(R, L, h, pos, win_topk, comp_topk, arange_w, compress, read_w):

    residual = h
    x, post, comb = L.hc_pre(h, L.hc_attn_fn, L.hc_attn_scale, L.hc_attn_base)
    x = L.attn_norm(x)
    x = attn_decode_cs(
        R, L.attn, x, pos, win_topk, comp_topk, arange_w, compress, read_w
    )
    x = L.hc_post(x, residual, post, comb)
    residual = x
    x, post, comb = L.hc_pre(x, L.hc_ffn_fn, L.hc_ffn_scale, L.hc_ffn_base)
    x = L.ffn_norm(x)
    return x, residual, post, comb


def block_post_cs(L, ffn_out, residual, post, comb):

    return L.hc_post(ffn_out, residual, post, comb)


def block_decode_cs(
    R, L, h, ids, pos, win_topk, comp_topk, arange_w, compress, moe, read_w
):

    ffn_in, residual, post, comb = block_pre_cs(
        R, L, h, pos, win_topk, comp_topk, arange_w, compress, read_w
    )
    ffn_out = moe(L, ffn_in, ids)
    return block_post_cs(L, ffn_out, residual, post, comb)


def build_win_topk(M, win, start_pos):

    return M.get_window_topk_idxs(win, 1, 1, start_pos)


def build_comp_topk(M, ratio, start_pos, offset, maxw):

    idx = M.get_compress_topk_idxs(ratio, 1, 1, start_pos, offset)
    k = idx.size(-1)
    out = idx.new_full((1, 1, maxw), -1)
    if k:
        out[..., :k] = idx
    return out.int()


def _layer_state(L):

    bufs = [L.attn.kv_cache]
    A = L.attn
    if A.compress_ratio:
        bufs += [A.compressor.kv_state, A.compressor.score_state]
        if A.indexer is not None:
            bufs += [
                A.indexer.kv_cache,
                A.indexer.compressor.kv_state,
                A.indexer.compressor.score_state,
            ]
    return bufs


class CapturedDecoder:
    def __init__(self, L, stage, moe_mode="eager"):
        self.L = L
        self.st = stage
        self.moe_mode = moe_mode
        self.a = stage.args
        self.dev = stage.device
        self.dt = stage.dtype
        self.R = _Ref(stage._M)
        self.win = L.attn.window_size
        self.ratio = L.attn.compress_ratio
        self.has_indexer = bool(self.ratio) and L.attn.indexer is not None
        self.moe = moe_stub if moe_mode == "stub" else real_moe

        self.moe_requested = moe_mode == "graph"
        self.moe_in_graph = None
        self.eager = False

        if self.has_indexer:
            self.maxw = L.attn.indexer.kv_cache.size(1)
        elif self.ratio:
            self.maxw = self.a.max_seq_len // self.ratio
        else:
            self.maxw = 0

        self.h_buf = torch.zeros(
            1, 1, self.a.hc_mult, self.a.dim, dtype=self.dt, device=self.dev
        )
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=self.dev)
        self.ids_buf = torch.zeros(1, 1, dtype=torch.long, device=self.dev)
        self.win_topk_buf = torch.zeros(
            1, 1, self.win, dtype=torch.int32, device=self.dev
        )
        self._bufs = {}
        self._graphs = {}
        self._pool = None
        self.ho = self.ffn_out_buf = self.g_post = None

    def _plan(self, start_pos):

        if not self.ratio:
            return 0, False
        end_ratio = (start_pos + 1) // self.ratio
        floor = self.L.attn.indexer.index_topk if self.has_indexer else 0
        return bucket_width(end_ratio, self.maxw, floor), (
            start_pos + 1
        ) % self.ratio == 0

    def _bufs_for(self, bucket):

        if bucket not in self._bufs:
            comp = (
                torch.full((1, 1, bucket), -1, dtype=torch.int32, device=self.dev)
                if self.ratio and not self.has_indexer
                else None
            )
            ar = torch.arange(bucket, device=self.dev) if self.has_indexer else None
            self._bufs[bucket] = (comp, ar)
        return self._bufs[bucket]

    def _block(self, compress, bucket):

        comp, ar = self._bufs_for(bucket)
        return block_decode_cs(
            self.R,
            self.L,
            self.h_buf,
            self.ids_buf,
            self.pos_buf,
            self.win_topk_buf,
            comp,
            ar,
            compress,
            self.moe,
            bucket,
        )

    def _pre(self, compress, bucket):
        comp, ar = self._bufs_for(bucket)
        return block_pre_cs(
            self.R,
            self.L,
            self.h_buf,
            self.pos_buf,
            self.win_topk_buf,
            comp,
            ar,
            compress,
            bucket,
        )

    def _capture_pos(self, compress, bucket):

        if not self.ratio:
            return max(1, self.win)
        top = min(bucket, self.maxw)
        p = top * self.ratio - 1
        if not compress:
            p = max(p - 1, 1)
            if (p + 1) % self.ratio == 0:
                p = max(p - 1, 1)
        return p

    def _feed_capture(self, compress, bucket, ids=None):

        p = self._capture_pos(compress, bucket)
        self.pos_buf.fill_(p)
        if ids is not None:
            self.ids_buf.copy_(ids.view(1, 1))
        self.win_topk_buf.copy_(build_win_topk(self.R.M, self.win, p))
        comp, _ = self._bufs_for(bucket)
        if comp is not None:
            comp.copy_(build_comp_topk(self.R.M, self.ratio, p, self.win, bucket))

    def _warm_and_capture(self, fn, restore=True):

        global _GRAPH_COUNT
        snap = [b.clone() for b in _layer_state(self.L)] if restore else None
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        if restore:
            for b, s in zip(_layer_state(self.L), snap):
                b.copy_(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool), torch.no_grad():
            out = fn()
        if self._pool is None:
            self._pool = g.pool()
        if restore:
            for b, s in zip(_layer_state(self.L), snap):
                b.copy_(s)
        _GRAPH_COUNT += 1
        return g, out

    def _resolve_moe_mode(self, ids):

        global _MOE_GRAPHED, _MOE_REFUSED
        why = _moe_refusal(self.L, self.dev, self.dt, ids, self.R.M)
        self.moe_in_graph = why is None
        if why is None:
            _MOE_GRAPHED += 1
            return
        _MOE_REFUSED += 1
        self.moe_mode = "eager"
        print(
            f"[v4] V4_MOE_IN_GRAPH: layer {self.L.layer_id} keeps its routed MoE EAGER — {why}",
            flush=True,
        )

    def _drop_moe_from_graph(self):

        global _MOE_GRAPHED, _MOE_REFUSED
        if self.moe_in_graph:
            self.moe_in_graph, _MOE_GRAPHED, _MOE_REFUSED = (
                False,
                _MOE_GRAPHED - 1,
                _MOE_REFUSED + 1,
            )
        self.moe_mode = "eager" if self.moe_mode == "graph" else self.moe_mode

    def _capture(self, key, ids=None):

        bucket, compress = key

        self._feed_capture(compress, bucket, ids)
        if self.moe_mode != "eager":
            g, out = self._warm_and_capture(lambda: self._block(compress, bucket))
            return {"graph": g, "out": out}
        outer = [b.clone() for b in _layer_state(self.L)]
        if self.ho is None:
            with torch.no_grad():
                ex = self._pre(False, bucket)
            self.ho = [torch.zeros_like(t) for t in ex]
            self.ffn_out_buf = torch.zeros_like(ex[0])

        def pre_fn():
            for buf, t in zip(self.ho, self._pre(compress, bucket)):
                buf.copy_(t)

        g, _ = self._warm_and_capture(pre_fn)
        entry = {"graph": g}
        if self.g_post is None:
            with torch.no_grad():
                self._feed_capture(compress, bucket)
                g.replay()
                self.ffn_out_buf.copy_(real_moe(self.L, self.ho[0], self.ids_buf))
            gp, out = self._warm_and_capture(
                lambda: block_post_cs(
                    self.L, self.ffn_out_buf, self.ho[1], self.ho[2], self.ho[3]
                ),
                restore=False,
            )
            self.g_post = {"graph": gp, "out": out}
        for b, s in zip(_layer_state(self.L), outer):
            b.copy_(s)
        return entry

    def run(self, h, ids, start_pos):

        global _GRAPH_SKIPPED
        if self.eager:
            return self._eager(h, ids, start_pos)
        if self.moe_requested and self.moe_in_graph is None:
            self._resolve_moe_mode(ids)
        if self.moe_mode == "graph" and ids is None:
            raise RuntimeError(
                f"v4 whole-layer graph: layer {self.L.layer_id} captured its routed MoE "
                f"(V4_MOE_IN_GRAPH=1) and was handed no token ids — a hash-routed layer would replay "
                f"the capture step's expert set. Carry the ids with the payload."
            )
        key = self._plan(start_pos)
        entry = self._graphs.get(key)
        if entry is None:
            need = 2 if (self.moe_mode == "eager" and self.g_post is None) else 1
            if _GRAPH_COUNT + need > V4_GRAPH_MAX:
                self.eager, _GRAPH_SKIPPED = True, _GRAPH_SKIPPED + need
                self._drop_moe_from_graph()
                print(
                    f"[v4] graph budget V4_GRAPH_MAX={V4_GRAPH_MAX} spent — layer "
                    f"{self.L.layer_id} stays eager",
                    flush=True,
                )
                return self._eager(h, ids, start_pos)
            try:
                entry = self._graphs[key] = self._capture(key, ids)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager, self._graphs, _GRAPH_SKIPPED = (
                    True,
                    {},
                    _GRAPH_SKIPPED + need,
                )
                self._drop_moe_from_graph()
                print(
                    f"[v4] whole-layer capture failed for layer {self.L.layer_id} at bucket "
                    f"{key[0]}: {type(e).__name__}: {e} — layer stays eager",
                    flush=True,
                )
                return self._eager(h, ids, start_pos)
        self._feed(h, ids, start_pos, key[0])
        entry["graph"].replay()
        if self.moe_mode != "eager":
            return entry["out"].clone()
        self.ffn_out_buf.copy_(real_moe(self.L, self.ho[0], ids))
        self.g_post["graph"].replay()
        return self.g_post["out"].clone()

    def _feed(self, h, ids, start_pos, bucket):

        self.h_buf.copy_(h)
        self.pos_buf.fill_(start_pos)
        if ids is not None:
            self.ids_buf.copy_(ids.view(1, 1))
        self.win_topk_buf.copy_(build_win_topk(self.R.M, self.win, start_pos))
        comp, _ = self._bufs_for(bucket)
        if comp is not None:
            comp.copy_(
                build_comp_topk(self.R.M, self.ratio, start_pos, self.win, bucket)
            )

    def _eager(self, h, ids, start_pos):

        bucket, compress = self._plan(start_pos)
        pos = torch.tensor([start_pos], dtype=torch.long, device=self.dev)
        win_topk = build_win_topk(self.R.M, self.win, start_pos)
        comp_topk = (
            build_comp_topk(self.R.M, self.ratio, start_pos, self.win, bucket)
            if self.ratio and not self.has_indexer
            else None
        )
        ar = torch.arange(bucket, device=self.dev) if self.has_indexer else None
        return block_decode_cs(
            self.R,
            self.L,
            h,
            ids,
            pos,
            win_topk,
            comp_topk,
            ar,
            compress,
            self.moe,
            bucket,
        )
