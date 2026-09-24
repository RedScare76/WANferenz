import contextlib
import types

import torch


def _attention(graph, attn, x, main_x):

    M = graph.model
    bsz, block_size, _ = x.shape
    rd = attn.rope_head_dim
    main_kv = attn.kv_norm(attn.wkv(main_x))
    M.apply_rotary_emb(main_kv[..., -rd:], graph.main_freqs)
    M.act_quant(main_kv[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)
    q = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(q).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
    M.apply_rotary_emb(q[..., -rd:], graph.draft_freqs)
    kv = attn.kv_norm(attn.wkv(x))
    M.apply_rotary_emb(kv[..., -rd:], graph.draft_freqs)
    M.act_quant(kv[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)
    attn.kv_cache[:bsz].index_copy_(1, graph.slot, main_kv)
    kv = torch.cat([attn.kv_cache[:bsz], kv], dim=1)
    o = M.sparse_attn(q, kv, attn.attn_sink, graph.topk, attn.softmax_scale)
    M.apply_rotary_emb(o[..., -rd:], graph.draft_freqs, True)
    o = o.view(bsz, block_size, attn.n_local_groups, -1)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return attn.wo_b(o.flatten(2))


class CapturedDraft:
    def __init__(self, tail, model, ids, hidden):
        self.tail, self.model = tail, model
        self.ids, self.hidden = ids.clone(), hidden.clone()
        attn = tail.mtp[0].attn
        self.win, self.width = attn.window_size, tail.block_size
        self.pos = torch.zeros((), device=hidden.device, dtype=torch.long)

        topk_width = ((self.win + self.width + 63) // 64) * 64
        self.lanes = torch.arange(topk_width, device=hidden.device)
        self.main_freqs = attn.freqs_cis[:1].clone()
        self.draft_freqs = attn.freqs_cis[1 : 1 + self.width].clone()
        self.graph = None
        self.failed = False

    def _feed(self, ids, hidden, pos):
        self.ids.copy_(ids)
        self.hidden.copy_(hidden)
        self.pos.fill_(pos)
        freqs = self.tail.mtp[0].attn.freqs_cis
        self.main_freqs.copy_(freqs[pos : pos + 1])
        self.draft_freqs.copy_(freqs[pos + 1 : pos + 1 + self.width])

    @contextlib.contextmanager
    def _attention_overrides(self):
        previous = []
        try:
            for blk in self.tail.mtp:
                attn = blk.attn
                previous.append((attn, attn.__dict__.get("forward")))

                def forward(a, x, start_pos, main_x):
                    return _attention(self, a, x, main_x)

                attn.forward = types.MethodType(forward, attn)
            yield
        finally:
            for attn, prior in previous:
                if prior is None:
                    del attn.forward
                else:
                    attn.forward = prior

    def _forward(self):
        nwin = (self.pos + 1).clamp(max=self.win)
        idx = torch.where(
            self.lanes < nwin,
            self.lanes,
            torch.where(
                self.lanes < nwin + self.width, self.win + self.lanes - nwin, -1
            ),
        )
        self.topk = idx.int().view(1, 1, -1).expand(1, self.width, -1).contiguous()
        self.slot = self.pos.remainder(self.win).view(1)
        h, main_x = self.tail.mtp[0].forward_embed(self.hidden, self.ids)
        for blk in self.tail.mtp:
            h = blk(h, 1, self.ids, main_x)
        return self.tail.mtp[-1].forward_head(h, self.ids)

    def _capture(self):
        saved = [blk.attn.kv_cache.clone() for blk in self.tail.mtp]
        try:
            with self._attention_overrides():
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side), torch.no_grad():
                    for _ in range(3):
                        self._forward()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad():
                    out = self._forward()
                self.graph, self.out = graph, out
        finally:
            torch.cuda.synchronize()
            for blk, cache in zip(self.tail.mtp, saved):
                blk.attn.kv_cache.copy_(cache)
        print("[v4 dspark] complete drafter graph captured", flush=True)

    def run(self, ids, hidden, pos):
        self._feed(ids, hidden, pos)
        if self.graph is None:
            try:
                self._capture()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                self.failed = True
                print(
                    f"[v4 dspark] complete graph declined: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return None
        self.graph.replay()
        return tuple(t.clone() for t in self.out)
