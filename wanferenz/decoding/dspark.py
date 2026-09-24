import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _v4():

    import wanferenz.serving.partition as partition

    return partition


V4_DSPARK_BLOCK = int(os.environ.get("V4_DSPARK_BLOCK", "0") or 0)


V4_DRAFT_TOP2 = os.environ.get("V4_DRAFT_TOP2", "0") not in ("", "0")


ALIAS_KEYS = ("embed.weight", "head.weight")


def second_choices(tail):

    if tail.last_spec is None or tail.temperature != 0.0:
        return None
    return tail.last_spec[1][0].topk(2, dim=-1).indices[:, 1].tolist()


def plan_verify_round(drafts, replies):

    drafts, replies = [int(t) for t in drafts], [int(t) for t in replies]
    if len(replies) != len(drafts) + 1:
        raise RuntimeError(
            f"v4 dspark: {len(drafts)} drafts need {len(drafts) + 1} replies (one per chunk "
            f"position, the chunk being [cur] + drafts), got {len(replies)}"
        )
    n = 0
    for d, r in zip(drafts, replies):
        if d != r:
            break
        n += 1
    return n, drafts[:n] + [replies[n]]


class DSparkPartition:
    def __init__(self, stage, temperature=0.0):
        M = _v4().ref()
        a = stage.args
        if not stage.tail:
            raise RuntimeError(
                "v4 dspark: the drafter runs on the TAIL — main_hidden is the tap "
                "over the last layers and no other stage holds it"
            )
        if not a.dspark_block_size:
            raise RuntimeError(
                "v4 dspark: this config has dspark_block_size=0, i.e. no MTP stages. "
                "Serve greedily instead of building a drafter that cannot draft."
            )
        if stage.embed_tokens is None or getattr(stage, "lm_head", None) is None:
            raise RuntimeError(
                "v4 dspark: the tail has no embedding — DSparkBlock.embed/.head ALIAS "
                "the main model's modules, so build the tail LayerPartition "
                "with dspark=True"
            )
        self.stage = stage
        self.args = a
        self.device, self.dtype = stage.device, stage.dtype
        self.block_size = a.dspark_block_size
        self.hidden_dim = len(a.dspark_target_layer_ids) * a.dim

        with torch.device(self.device), M.set_dtype(self.dtype):
            self.mtp = torch.nn.ModuleList(
                [M.DSparkBlock(a.n_layers + k, a) for k in range(a.n_mtp_layers)]
            )
        for blk in self.mtp:
            blk.embed = stage.embed_tokens
            blk.head = stage.lm_head

            blk.temperature = float(temperature)
        self.temperature = float(temperature)

        if V4_DSPARK_BLOCK:
            if not 1 <= V4_DSPARK_BLOCK <= 32:
                raise ValueError(
                    f"v4 dspark: V4_DSPARK_BLOCK={V4_DSPARK_BLOCK} — the inference-time block "
                    f"width must be 1..32 (0/unset = the trained width, "
                    f"{a.dspark_block_size} in this checkpoint)"
                )
            self.block_size = int(V4_DSPARK_BLOCK)
            for blk in self.mtp:
                blk.block_size = self.block_size
        self.mtp.eval()
        self.alias_missing = []
        self._pos = None
        self.last_spec = None
        self.reset()

    def reset(self):

        with torch.no_grad():
            for blk in self.mtp:
                blk.attn.kv_cache.zero_()
        self._pos = None
        self.last_spec = None

    @property
    def pos(self):

        return self._pos

    def _forward_spec(self, input_ids, main_hidden, start_pos):

        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        for layer in self.mtp:
            h = layer(h, start_pos, input_ids, main_x)
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(h, input_ids)

    def _hidden(self, x, want_s=None):

        t = torch.as_tensor(x).to(device=self.device, dtype=self.dtype)
        if t.dim() != 3 or t.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                f"v4 dspark: main_hidden is {tuple(t.shape)}, expected [b, s, {self.hidden_dim}] "
                f"— that is LayerPartition.tail_main_hidden(), the {len(self.args.dspark_target_layer_ids)} "
                f"target-layer taps concatenated"
            )
        if t.shape[1] == 0:
            raise RuntimeError(
                "v4 dspark: main_hidden has no positions — a prefill needs the whole "
                "prompt's taps and an advance needs one per committed position"
            )
        if want_s is not None and t.shape[1] != want_s:
            raise RuntimeError(
                f"v4 dspark: main_hidden has {t.shape[1]} positions, the token ids "
                f"have {want_s} — one tap per committed position"
            )
        if t.shape[0] > self.args.max_batch_size:
            raise RuntimeError(
                f"v4 dspark: batch {t.shape[0]} exceeds max_batch_size "
                f"{self.args.max_batch_size} — the mtp kv_cache is sized for it"
            )
        return t

    def _flat_ids(self, x):

        t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        if t.dim() == 0:
            t = t.view(1)
        elif t.dim() == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.dim() != 1:
            raise RuntimeError(
                f"v4 dspark: token ids {tuple(t.shape)}, expected [b] or [b, 1]"
            )
        return t

    def _seq_ids(self, x):

        t = torch.as_tensor(x, dtype=torch.long, device=self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2 or t.shape[1] == 0:
            raise RuntimeError(
                f"v4 dspark: committed ids {tuple(t.shape)}, expected [b, n>0] "
                f"(a 1-D input is read as one batch row's run)"
            )
        return t

    def prefill(self, pred_ids, main_hidden):

        if self._pos is not None:
            raise RuntimeError(
                f"v4 dspark: already prefilled to position {self._pos} — reset() "
                f"before starting another sequence"
            )
        mh = self._hidden(main_hidden)
        ids = self._flat_ids(pred_ids)
        if ids.shape[0] != mh.shape[0]:
            raise RuntimeError(
                f"v4 dspark: {ids.shape[0]} token ids against a batch of "
                f"{mh.shape[0]} in main_hidden"
            )
        with torch.no_grad():
            out = self._forward_spec(ids, mh, 0)
        if out is not None:
            raise RuntimeError(
                "v4 dspark: forward_spec at start_pos=0 must be a prefill and return "
                "None — the vendored reference changed under us"
            )
        self._pos = mh.shape[1] - 1
        return self._pos

    def advance_and_draft(self, input_ids_seq, main_hidden_seq, start_pos):

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
        out = None
        with torch.no_grad():
            for j in range(n):
                pos = self._pos + 1
                out = self._forward_spec(ids[:, j], mh[:, j : j + 1], pos)
                self._pos = pos
        self.last_spec = out
        output_ids, _, confidence = out
        return output_ids[:, 1:], confidence

    def load(self, d=None):

        V4 = _v4()
        d = d or V4.V4_DIR
        wm = V4.tensor_locations(d)
        self.alias_missing = []
        for k, blk in enumerate(self.mtp):
            prefix = f"mtp.{k}."
            names = [n for n in wm if n.startswith(prefix)]
            if not names:
                raise RuntimeError(
                    f"v4 dspark: no tensor under {prefix!r} in {d!r} — this checkpoint carries no "
                    f"MTP stage {k}. The drafter's weights ship with the model (mtp.0/1/2.*); a "
                    f"checkpoint without them can only be served greedily."
                )
            state = {n[len(prefix) :]: V4.raw(n, d) for n in names}
            try:
                missing, unexpected = blk.load_state_dict(state, strict=False)
            finally:
                del state
                V4._close_tensor_sources(d)
            extra = sorted(set(missing) - set(ALIAS_KEYS))
            if extra or unexpected:
                raise RuntimeError(
                    f"v4 dspark: mtp stage {k} in {d!r} is not the checkpoint this config declares — "
                    f"missing {extra}, unexpected {sorted(unexpected)}. (Only {list(ALIAS_KEYS)} may "
                    f"be missing: they alias the tail's embed/head and convert.py skips them.)"
                )
            self.alias_missing.append(tuple(sorted(missing)))
        return self

    def __repr__(self):
        return (
            f"<V4DSparkTail x{len(self.mtp)} block={self.block_size} "
            f"targets={tuple(self.args.dspark_target_layer_ids)} {self.dtype} on {self.device} "
            f"pos={self._pos}>"
        )


class ChainDraftSource:
    def __init__(self, tail):
        self.tail = tail
        self._done = False

        self.pipelined = False
        self._cfront = None
        self._mfront = None

        self._last = None

    def on_chunk(self, msg, st, out):
        ids = torch.as_tensor(msg["ids"], dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise RuntimeError(
                f"v4 dspark: a drafted round is single-sequence today, got ids "
                f"{tuple(ids.shape)} — see ChainDraftSource's docstring"
            )
        if self.pipelined:
            return self._on_chunk_pipelined(ids, msg, st, out)
        start_pos = int(msg["start_pos"])
        main_hidden = st.tail_main_hidden()
        if start_pos == 0:
            self.tail.reset()
            self._done = False
            self.tail.prefill([int(out["token"])], main_hidden)
            return {}
        replies = out.get("tokens")
        if replies is None:
            raise RuntimeError(
                "v4 dspark: a dspark job's reset must arm `spec` as well — the accept rule needs the "
                "model's greedy token at EVERY chunk position, and the tail only computes those when "
                "_spec is armed (chain.decode_dspark sends both flags)"
            )
        n, committed = plan_verify_round(ids[0, 1:].tolist(), replies)

        end = start_pos + n + self.tail.block_size + 1
        self._done = self._done or end > self.tail.args.max_seq_len
        if self._done:
            return {"draft": [], "n": n}

        blk, conf = self.tail.advance_and_draft(
            [committed], main_hidden[:, : n + 1], start_pos=start_pos
        )
        r = {
            "draft": blk[0].tolist(),
            "n": n,
            "conf": [round(c, 4) for c in conf[0].float().tolist()],
        }
        if V4_DRAFT_TOP2:
            d2 = second_choices(self.tail)
            if d2 is not None:
                r["d2"] = d2
        return r

    def _on_chunk_pipelined(self, ids, msg, st, out):

        start_pos = int(msg["start_pos"])
        main_hidden = st.tail_main_hidden()
        if start_pos == 0:
            self.tail.reset()
            self._done = False
            self._last = None
            self.tail.prefill([int(out["token"])], main_hidden)
            self._cfront, self._mfront = main_hidden.shape[1] - 1, int(out["token"])
            return {"acc": True}
        if self._cfront is None:
            raise RuntimeError(
                "v4 dspark: a pipelined frame before the prefill — the mtp window is "
                "empty and there is no committed frontier to judge this frame against"
            )
        if ids.shape[1] != 1:
            raise RuntimeError(
                f"v4 dspark: a pipelined round streams s=1 frames, got s={ids.shape[1]} "
                f"— the whole point is that no stage replays a block position by "
                f"position, so a multi-token frame is a coordinator bug"
            )
        replies = out.get("tokens")
        if replies is None:
            raise RuntimeError(
                "v4 dspark: a pipelined job's reset must arm `spec` as well — the accept rule needs "
                "the model's greedy token at this frame's position, and the tail only computes those "
                "when _spec is armed"
            )
        if start_pos != self._cfront + 1 or int(ids[0, 0]) != self._mfront:
            return {"acc": False}
        m = int(replies[0])
        self._cfront, self._mfront = start_pos, m

        self._done = (
            self._done
            or (start_pos + self.tail.block_size + 1) > self.tail.args.max_seq_len
        )
        if self._done:
            return {"acc": True, "draft": []}
        if not wants_block(msg, m, self._last):
            t = self.tail
            if t.pos is None or start_pos != t.pos + 1:
                raise RuntimeError(
                    f"v4 dspark: a lazy advance at {start_pos} but the mtp cache stands at {t.pos} — "
                    f"the drafter's cursor must walk exactly the committed positions whether or not "
                    f"it drafts on them"
                )
            with torch.no_grad():
                _fast()._advance_cache_only(
                    t, t._hidden(main_hidden, want_s=1), start_pos
                )
            t._pos = start_pos
            return {"acc": True}
        blk, conf = self.tail.advance_and_draft([[m]], main_hidden, start_pos=start_pos)
        draft = blk[0].tolist()
        self._last = (start_pos, draft)
        r = {
            "acc": True,
            "draft": draft,
            "conf": [round(c, 4) for c in conf[0].float().tolist()],
        }
        if V4_DRAFT_TOP2:
            d2 = second_choices(self.tail)
            if d2 is not None:
                r["d2"] = d2
        return r


def _fast():

    global _FAST
    if _FAST is None:
        import wanferenz.decoding.advance as advance

        _FAST = advance
    return _FAST


_FAST = None


def wants_block(msg, m, last=None):

    nxt = msg.get("dnxt")
    if nxt is None and msg.get("dprev") and last is not None:
        at, blk = last
        if blk and at == int(msg["start_pos"]) - 1:
            nxt = blk[0]
    return nxt is None or int(nxt) != m


def ring_drafter(stage, ckpt_dir=None, temperature=0.0):

    import wanferenz.decoding.advance as advance

    advance.install(sys.modules[__name__])

    import wanferenz.serving.controls as controls

    live = getattr(DSparkPartition.advance_and_draft, "_v4_dspark_fast", False)
    controls.note("V4_DSPARK_FAST", live)
    print(
        f"[dspark] V4_DSPARK_FAST requested={advance.V4_DSPARK_FAST} "
        f"observed={'on' if live else 'off'}",
        file=sys.stderr,
        flush=True,
    )
    tail = DSparkPartition(stage, temperature=temperature)

    import wanferenz.decoding.expert_bank as expert_bank

    took = expert_bank.install_drafter(tail)
    controls.note("V4_DSPARK_MOE", took == len(tail.mtp) and took > 0)
    print(
        f"[dspark] V4_DSPARK_MOE requested={expert_bank.V4_DSPARK_MOE} "
        f"observed={took}/{len(tail.mtp)} drafter MoEs banked and bound "
        f"(draft width {tail.block_size}; width 1 uses the single-row class chain)",
        file=sys.stderr,
        flush=True,
    )

    controls.note("V4_DSPARK_BLOCK", str(tail.block_size) if V4_DSPARK_BLOCK else "off")
    if V4_DSPARK_BLOCK:
        print(
            f"[dspark] V4_DSPARK_BLOCK requested={V4_DSPARK_BLOCK} observed={tail.block_size} "
            f"(trained width {tail.args.dspark_block_size})",
            file=sys.stderr,
            flush=True,
        )

    controls.note("V4_DRAFT_TOP2", V4_DRAFT_TOP2 and tail.temperature == 0.0)
    if V4_DRAFT_TOP2:
        print(
            f"[dspark] V4_DRAFT_TOP2 requested=on observed="
            f"{'on' if tail.temperature == 0.0 else 'off (sampling drafter)'}",
            file=sys.stderr,
            flush=True,
        )
    if ckpt_dir is not None:
        tail.load(ckpt_dir)
    return ChainDraftSource(tail)


def verify_locally():

    import wanferenz.model.oracle as oracle

    args = oracle.miniature_parameters()
    prompt, rounds = 13, 3
    oracle = oracle.create_oracle(args, 0)
    st = _v4().LayerPartition(
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
    dr = DSparkPartition(st)
    for k, blk in enumerate(dr.mtp):
        sd = {
            n: v for n, v in oracle.mtp[k].state_dict().items() if n not in ALIAS_KEYS
        }
        assert set(blk.load_state_dict(sd, strict=False).missing_keys) == set(
            ALIAS_KEYS
        )
    print(st, flush=True)
    print(dr, flush=True)

    torch.manual_seed(0)
    ids = torch.randint(0, args.vocab_size, (1, prompt))
    o_tok, _, o_main = oracle(ids)
    assert oracle.forward_spec(o_tok, o_main) is None
    tok = st.logits_all(st.forward(st.embed(ids), ids, 0), full_logits=False).argmax(-1)
    assert torch.equal(tok, o_tok), "prefill token diverged"
    assert torch.equal(st.tail_main_hidden(), o_main), "prefill tap diverged"
    dr.prefill(tok, st.tail_main_hidden())
    for k, blk in enumerate(dr.mtp):
        assert torch.equal(blk.attn.kv_cache, oracle.mtp[k].attn.kv_cache), (
            f"mtp {k} cache diverged"
        )
    print(
        f"[v4] prefill  pos={dr.pos}  mtp window bit-identical to the reference",
        flush=True,
    )

    for i in range(prompt, prompt + rounds):
        o_tok, _, o_main = oracle(tok.unsqueeze(1), i)
        o_spec = oracle.forward_spec(o_tok, o_main, i)
        h = st.forward(st.embed(tok.unsqueeze(1)), tok.unsqueeze(1), i)
        tok = st.logits_all(h, full_logits=False).argmax(-1)
        drafts, conf = dr.advance_and_draft(
            tok.unsqueeze(1), st.tail_main_hidden(), start_pos=i
        )
        assert torch.equal(tok, o_tok), f"token diverged at {i}"
        for got, want, what in zip(
            dr.last_spec, o_spec, ("output_ids", "logits", "confidence")
        ):
            assert torch.equal(got, want), f"{what} diverged at {i}"
        print(
            f"[v4] round {i}  anchor {dr.last_spec[0][0, 0].item()} -> drafts "
            f"{drafts[0].tolist()}  confidence "
            f"{[round(c, 3) for c in conf[0].float().tolist()]}",
            flush=True,
        )
    print(
        f"[v4] {rounds} drafted rounds bit-identical to Transformer.forward_spec",
        flush=True,
    )


if __name__ == "__main__":
    verify_locally()
