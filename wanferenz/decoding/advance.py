import os

import torch


V4_DSPARK_FAST = os.environ.get("V4_DSPARK_FAST", "0") not in ("", "0")
V4_DSPARK_GRAPH = os.environ.get("V4_DSPARK_GRAPH", "0") not in ("", "0")

_REF_ADVANCE = None
V4_DSPARK_FULL_GRAPH = bool(int(os.environ.get("V4_DSPARK_FULL_GRAPH", "0") or 0))


def _model():

    import wanferenz.serving.partition as partition

    return partition.ref()


def _advance_cache_only(self, main_hidden_int, base_pos):

    M = _model()
    k = main_hidden_int.shape[1]
    bsz = main_hidden_int.shape[0]
    for j in range(k):
        pos = base_pos + j

        main_x = self.mtp[0].main_norm(
            self.mtp[0].main_proj(main_hidden_int[:, j : j + 1])
        )
        for blk in self.mtp:
            attn = blk.attn
            rd = attn.rope_head_dim
            win = attn.window_size
            main_kv = attn.kv_norm(attn.wkv(main_x))
            M.apply_rotary_emb(main_kv[..., -rd:], attn.freqs_cis[pos : pos + 1])
            M.act_quant(main_kv[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)
            attn.kv_cache[:bsz, pos % win] = main_kv.squeeze(1)


class _DraftOutputCapture:
    def __init__(self, head_block, b, block_size, hc_mult, dim, dtype, device):
        self.blk = head_block
        self.h = torch.zeros(b, block_size, hc_mult, dim, dtype=dtype, device=device)
        self.ids = torch.zeros(b, dtype=torch.long, device=device)
        self.graph = None
        self.out = None
        self.failed = False

    def _capture(self):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self.blk.forward_head(self.h, self.ids)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = self.blk.forward_head(self.h, self.ids)
        self.graph, self.out = g, out
        print("[v4 dspark] head-graph captured", flush=True)

    def run(self, h, input_ids):

        if self.graph is None:
            try:
                self._capture()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.failed = True
                print(
                    f"[v4 dspark] head-graph capture failed: {type(e).__name__}: {e} -> eager",
                    flush=True,
                )
                return None
        self.h.copy_(h)
        self.ids.copy_(input_ids)
        self.graph.replay()
        return tuple(t.clone() for t in self.out)


def _kept_forward_spec(self, input_ids, main_hidden, start_pos):

    if (
        V4_DSPARK_FULL_GRAPH
        and main_hidden.is_cuda
        and self.temperature == 0.0
        and main_hidden.shape[:2] == (1, 1)
    ):
        from wanferenz.graphs.drafter import CapturedDraft

        graph = getattr(self, "_full_draft_graph", None)
        if graph is None:
            graph = self._full_draft_graph = CapturedDraft(
                self, _model(), input_ids, main_hidden
            )
        if not graph.failed:
            out = graph.run(input_ids, main_hidden, start_pos)
            if out is not None:
                return out
    h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
    for blk in self.mtp:
        h = blk(h, start_pos, input_ids, main_x)
    if _graph_eligible(self, h):
        hg = self._fast_head_graph
        if hg is None:
            b, s, hc, dim = h.shape
            hg = _DraftOutputCapture(self.mtp[-1], b, s, hc, dim, h.dtype, h.device)
            self._fast_head_graph = hg
        if not hg.failed:
            out = hg.run(h, input_ids)
            if out is not None:
                return out
    return self.mtp[-1].forward_head(h, input_ids)


def _graph_eligible(self, h):

    return (
        V4_DSPARK_GRAPH
        and h.is_cuda
        and self.temperature == 0.0
        and h.shape[0] == 1
        and h.shape[1] == self.block_size
    )


def fast_advance_and_draft(self, input_ids_seq, main_hidden_seq, start_pos):

    if self._pos is None:
        raise RuntimeError(
            "v4 dspark: advance before prefill — the mtp window is empty, so the "
            "first draft would attend to zeros. Call prefill() after the ring's."
        )
    ids = self._seq_ids(input_ids_seq)
    n = ids.shape[1]
    mh = self._hidden(main_hidden_seq, want_s=n)
    if ids.shape[0] != mh.shape[0]:
        raise RuntimeError(
            f"v4 dspark: ids batch {ids.shape[0]} against main_hidden batch "
            f"{mh.shape[0]}"
        )
    if n > self.block_size + 1:
        raise RuntimeError(
            f"v4 dspark: an advance over {n} positions, but one round can commit at most "
            f"{self.block_size + 1} (g={self.block_size} accepted drafts plus the bonus). This "
            f"is the committed PREFIX of one verify round, not a whole chunk or several rounds."
        )
    if start_pos != self._pos + 1:
        raise RuntimeError(
            f"v4 dspark: advance at {start_pos} but the mtp cache stands at {self._pos}, so the "
            f"next position is {self._pos + 1}. A "
            f"{'gap' if start_pos > self._pos + 1 else 'overlap'} is an upstream protocol bug: "
            f"the drafter must be advanced over exactly the COMMITTED positions of every round, "
            f"no more and no less."
        )
    end = self._pos + n + self.block_size + 1
    if end > self.args.max_seq_len:
        raise RuntimeError(
            f"v4 dspark: a block drafted here would rope out to position {end}, "
            f"past max_seq_len {self.args.max_seq_len} — stop drafting before "
            f"the context limit, not inside the reference's freqs_cis slice"
        )
    with torch.no_grad():
        if n > 1:
            _advance_cache_only(self, mh[:, : n - 1], start_pos)
        pos = start_pos + n - 1
        out = _kept_forward_spec(self, ids[:, n - 1], mh[:, n - 1 : n], pos)
        self._pos = pos
    self.last_spec = out
    output_ids, _, confidence = out
    return output_ids[:, 1:], confidence


def install(module=None):

    global _REF_ADVANCE
    if not V4_DSPARK_FAST:
        return False
    if module is None:
        import wanferenz.decoding.dspark as module
    tail = module.DSparkPartition
    if getattr(tail.advance_and_draft, "_v4_dspark_fast", False):
        return False
    _REF_ADVANCE = tail.advance_and_draft
    fast_advance_and_draft._v4_dspark_fast = True
    tail.advance_and_draft = fast_advance_and_draft

    if not hasattr(tail, "_fast_head_graph"):
        tail._fast_head_graph = None
    return True


def uninstall(module=None):

    global _REF_ADVANCE
    if _REF_ADVANCE is None:
        return False
    if module is None:
        import wanferenz.decoding.dspark as module
    module.DSparkPartition.advance_and_draft = _REF_ADVANCE
    _REF_ADVANCE = None
    return True


def verify_locally():

    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import wanferenz.model.oracle as oracle
    import wanferenz.serving.partition as partition
    import wanferenz.decoding.dspark as D

    global V4_DSPARK_FAST
    args = oracle.miniature_parameters()
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

    def record():

        st, _ = build()
        p = 13
        ids = torch.randint(
            0, args.vocab_size, (1, p), generator=torch.Generator().manual_seed(3)
        )
        tok = st.logits_all(
            st.forward(st.embed(ids), ids, 0), full_logits=False
        ).argmax(-1)
        prefill_main = st.tail_main_hidden()
        seq = [int(tok)] + [7, 151, 293, 41, 88, 19, 200]
        rounds, pos, base = [], p, 0
        for run_len in (1, 3, 2):
            chunk = torch.tensor([seq[base : base + run_len]], dtype=torch.long)
            st.forward(st.embed(chunk), chunk, pos)
            committed = torch.tensor(
                [seq[base + 1 : base + run_len + 1]], dtype=torch.long
            )
            rounds.append((committed, st.tail_main_hidden().clone(), pos))
            pos += run_len
            base += run_len
        return tok, prefill_main, rounds

    tok, prefill_main, rounds = record()

    def replay(dr):
        dr.prefill(tok, prefill_main)
        blocks = []
        for committed, main_hidden, start_pos in rounds:
            blk, conf = dr.advance_and_draft(
                committed, main_hidden, start_pos=start_pos
            )
            blocks.append((blk.clone(), conf.clone(), dr.last_spec[1].clone()))
        return blocks, [b.attn.kv_cache.clone() for b in dr.mtp]

    V4_DSPARK_FAST = False
    _, dr = build()
    ref_blocks, ref_caches = replay(dr)

    V4_DSPARK_FAST = True
    assert install(D), "install must take when V4_DSPARK_FAST is on"
    _, dr2 = build()
    fast_blocks, fast_caches = replay(dr2)
    for i, ((rb, rc, rl), (fb, fc, fl)) in enumerate(zip(ref_blocks, fast_blocks)):
        assert torch.equal(rb, fb), f"draft ids diverged round {i}"
        assert torch.equal(rc, fc), f"confidence diverged round {i}"
        assert torch.equal(rl, fl), f"draft logits diverged round {i}"
    for i, (rk, fk) in enumerate(zip(ref_caches, fast_caches)):
        assert torch.equal(rk, fk), f"mtp {i} kv_cache diverged"
    print(
        "[v4 dspark] cache-advance-only: bit-exact vs reference advance_and_draft",
        flush=True,
    )
    uninstall(D)
    print("[v4 dspark] self-test OK", flush=True)


if __name__ == "__main__":
    verify_locally()
