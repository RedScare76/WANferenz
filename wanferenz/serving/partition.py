import argparse, gc, glob, json, os, sys, torch
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open


import wanferenz.serving.controls as controls


V4_DIR = os.environ.get("V4_DIR", "/root/v4")
dev = os.environ.get("V4_DEV", "cuda")


V4_DTYPE = os.environ.get("V4_DTYPE", "bfloat16")


V4_FAST_VERIFY = bool(int(os.environ.get("V4_FAST_VERIFY", "0") or 0))
V4_HEAD_GRAPH = os.environ.get("V4_HEAD_GRAPH", "0") not in ("", "0")
V4_SPEC_SNAPSHOT = os.environ.get("V4_SPEC_SNAPSHOT", "full")


V4_FAST_VERIFY_MAX = int(os.environ.get("V4_FAST_VERIFY_MAX", "16") or 16)


V4_CUDA_GRAPH = os.environ.get("V4_CUDA_GRAPH", "0")


def _capture_mode(v=None):

    v = V4_CUDA_GRAPH if v is None else v
    if v in (True, "1", "island", "on"):
        return "island"
    if v in ("whole", "2", "eager"):
        return "whole"
    return "off"


V4_GRAPH_MAX = int(os.environ.get("V4_GRAPH_MAX", "192"))
_GRAPH_COUNT = 0
_GRAPH_SKIPPED = 0

_REF = None
_ARGS = {}
_WM = {}
_HD = {}
_GLOBALS = None


def ref():

    global _REF
    if _REF is None:
        import wanferenz.model.oracle as oracle

        _REF = oracle.reference_module()
    return _REF


def config(d=None):

    d = d or V4_DIR
    if d not in _ARGS:
        with open(f"{d}/config.json") as f:
            _ARGS[d] = ref().ModelArgs(**json.load(f))
    return _ARGS[d]


def tensor_locations(d=None):

    d = d or V4_DIR
    if d not in _WM:
        files = sorted(glob.glob(os.path.join(d, "model*-mp*.safetensors")))
        if not files:
            raise RuntimeError(
                f"v4: no model*-mp*.safetensors in {d!r} — this loader reads convert.py's OUTPUT "
                f"format, not an HF release. Run deepseek_v4_ref/inference/convert.py first."
            )
        wm = {}
        for f in files:
            with safe_open(f, "pt", device="cpu") as h:
                for n in h.keys():
                    wm[n] = os.path.basename(f)
        _WM[d] = wm
    return _WM[d]


def raw(n, d=None):

    d = d or V4_DIR
    s = tensor_locations(d)[n]
    key = (d, s)
    if key not in _HD:
        _HD[key] = safe_open(os.path.join(d, s), "pt", device="cpu")
    return _HD[key].get_tensor(n)


def _close_tensor_sources(d=None):

    d = d or V4_DIR
    paths = []
    for key in [key for key in _HD if key[0] == d]:
        paths.append(os.path.join(*key))
        del _HD[key]
    gc.collect()
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        for path in paths:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def _configure_reference(M, args):

    global _GLOBALS
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if world_size != 1:
        raise RuntimeError(
            f"v4: world_size={world_size}. A LayerPartition is PIPELINE parallelism — the reference's "
            f"tensor-parallel path all_reduces inside RowParallelLinear/MoE/Indexer against a "
            f"process group a stage does not own. Run one rank per stage."
        )
    new = (
        world_size,
        rank,
        torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16,
        "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt,
        torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32,
    )
    if _GLOBALS is not None and _GLOBALS != new:
        raise RuntimeError(
            f"v4: model.py's globals are already {_GLOBALS} and this LayerPartition's args want {new}. They "
            f"are MODULE globals set once by Transformer.__init__, and every Linear read the old "
            f"ones at construction — rebinding them now would leave the existing stage's weights "
            f"in a format nothing agrees on. One ModelArgs per process."
        )
    M.world_size, M.rank, M.default_dtype, M.scale_fmt, M.scale_dtype = new
    _GLOBALS = new
    return new


class _DecodeWindow:
    def __init__(self, args, start_pos, s, bsz, device, cap):
        self.start_pos, self.s, self.bsz, self.device, self.cap = (
            start_pos,
            s,
            bsz,
            device,
            cap,
        )
        self.win = args.window_size
        self._win_rows, self._comp_rows, self._slots = {}, {}, None

    def _to(self, rows):

        t = torch.tensor(rows, dtype=torch.int32).view(self.s, -1)
        return t.unsqueeze(0).expand(self.bsz, -1, -1).contiguous().to(self.device)

    def ring_slots(self):

        if self._slots is None:
            self._slots = torch.tensor(
                [(self.start_pos + i) % self.win for i in range(self.s)],
                dtype=torch.long,
                device=self.device,
            )
        return self._slots

    def window_rows(self, base):

        if base not in self._win_rows:
            win, sp0, rows = self.win, self.start_pos, []
            for j in range(self.s):
                p = sp0 + j
                if p >= win - 1:
                    sp = p % win
                    slots = list(range(sp + 1, win)) + list(range(sp + 1))
                    pos = [p - win + 1 + k for k in range(win)]
                else:
                    slots = list(range(p + 1)) + [-1] * (win - p - 1)
                    pos = list(slots)
                rows.append(
                    [base + (q - sp0) if q >= sp0 else v for v, q in zip(slots, pos)]
                )
            self._win_rows[base] = self._to(rows)
        return self._win_rows[base]

    def lengths(self, ratio):

        return [(self.start_pos + j + 1) // ratio for j in range(self.s)]

    def groups(self, ratio):

        n, out, j0 = self.lengths(ratio), [], 0
        for j in range(1, self.s + 1):
            if j == self.s or n[j] != n[j0]:
                out.append((j0, j, n[j0]))
                j0 = j
        return out

    def compress_rows(self, ratio, offset):

        if (ratio, offset) not in self._comp_rows:
            n = self.lengths(ratio)
            wide = n[-1]
            rows = [[offset + t for t in range(k)] + [-1] * (wide - k) for k in n]
            self._comp_rows[(ratio, offset)] = (
                self._to(rows)
                if wide
                else torch.zeros(
                    self.bsz, self.s, 0, dtype=torch.int32, device=self.device
                )
            )
        return self._comp_rows[(ratio, offset)]


def _chunk_compressor(c, x, start_pos):

    for j in range(x.size(1)):
        c(x[:, j : j + 1].contiguous(), start_pos + j)


def _chunk_indexer(self, x, qr, start_pos, offset, capacity):

    M = ref()
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    ratio, rd = self.compress_ratio, self.rope_head_dim
    if self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis
    q = self.wq_b(qr)
    q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)
    q = M.rotate_activation(q)
    M.fp4_act_quant(q, M.fp4_block_size, True)
    _chunk_compressor(self.compressor, x, start_pos)
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    rows, wide = [], 0
    for j0, j1, n in capacity.groups(ratio):
        k = min(self.index_topk, n)
        if k == 0:
            rows.append(torch.zeros(bsz, j1 - j0, 0, dtype=torch.long, device=x.device))
            continue
        score = torch.einsum(
            "bshd,btd->bsht", q[:, j0:j1].contiguous(), self.kv_cache[:bsz, :n]
        )
        score = (score.relu_() * weights[:, j0:j1].unsqueeze(-1)).sum(dim=2)
        rows.append(score.topk(k, dim=-1)[1] + offset)
        wide = max(wide, k)
    if len(rows) == 1:
        return rows[0]
    return torch.cat(
        [
            r
            if r.size(-1) == wide
            else torch.cat(
                [r, r.new_full((bsz, r.size(1), wide - r.size(-1)), -1)], dim=-1
            )
            for r in rows
        ],
        dim=1,
    )


def _chunk_attention(self, x, start_pos, capacity):

    M = ref()
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim
    if self.compress_ratio and self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache[:, win:]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    qr = q = self.q_norm(self.wq_a(x))
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    M.apply_rotary_emb(q[..., -rd:], freqs_cis)

    kv = self.wkv(x)
    kv = self.kv_norm(kv)
    M.apply_rotary_emb(kv[..., -rd:], freqs_cis)
    M.act_quant(kv[..., :-rd], 64, M.scale_fmt, M.scale_dtype, True)
    base = self.kv_cache.size(1) - capacity.cap
    topk_idxs = capacity.window_rows(base)
    if self.compress_ratio:
        offset = win
        if self.indexer is not None:
            compress_topk_idxs = _chunk_indexer(
                self.indexer, x, qr, start_pos, offset, capacity
            ).int()
        else:
            compress_topk_idxs = capacity.compress_rows(ratio, offset)
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

    self.kv_cache[:bsz, base : base + seqlen] = kv
    if self.compress_ratio:
        _chunk_compressor(self.compressor, x, start_pos)
    o = M.sparse_attn(
        q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale
    )
    self.kv_cache[:bsz, capacity.ring_slots()] = kv
    M.apply_rotary_emb(o[..., -rd:], freqs_cis, True)

    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return self.wo_b(o.flatten(2))


_CHUNK_BLOCK = None


def chunk_decoder_type(M):

    global _CHUNK_BLOCK
    if _CHUNK_BLOCK is None or _CHUNK_BLOCK.__mro__[1] is not M.Block:

        class _ChunkAttention(M.Attention):
            _chunk = None

            def forward(self, x, start_pos):
                if self._chunk is None:
                    return super().forward(x, start_pos)
                return _chunk_attention(self, x, start_pos, self._chunk)

        class _ChunkBlock(M.Block):
            attention_cls = _ChunkAttention

        _CHUNK_BLOCK = _ChunkBlock
    return _CHUNK_BLOCK


class LayerPartition:
    def __init__(
        self,
        lo,
        hi,
        args=None,
        *,
        head=False,
        tail=False,
        dspark=False,
        device=None,
        dtype=None,
        spec_depth=None,
        fast_verify=None,
        snapshot_mode=None,
    ):
        self.lo, self.hi = lo, hi
        self.args = args if args is not None else config()
        self.device = device or dev
        self.dtype = dtype or getattr(torch, V4_DTYPE)
        self.head, self.tail, self._dspark = head, tail, dspark
        self._fast = V4_FAST_VERIFY if fast_verify is None else bool(fast_verify)
        self._snapshot_mode = (
            V4_SPEC_SNAPSHOT if snapshot_mode is None else snapshot_mode
        )
        if self._snapshot_mode not in ("full", "delta"):
            raise ValueError("V4_SPEC_SNAPSHOT must be full or delta")
        self._chunk_cap = V4_FAST_VERIFY_MAX if self._fast else 0
        M = ref()
        self._M = M
        self.globals = _configure_reference(M, self.args)
        a = self.args
        if not 0 <= lo < hi <= a.n_layers:
            raise RuntimeError(
                f"v4 stage[{lo}:{hi}) is not a range inside 0..{a.n_layers}"
            )
        block_cls = chunk_decoder_type(M) if self._fast else M.Block

        with torch.device(self.device), M.set_dtype(self.dtype):
            self.layers = torch.nn.ModuleList(
                [block_cls(li, a) for li in range(lo, hi)]
            )
            self.embed_tokens = (
                M.ParallelEmbedding(a.vocab_size, a.dim) if (head or dspark) else None
            )
            if tail:
                self.norm = M.RMSNorm(a.dim, a.norm_eps)
                self.lm_head = M.ParallelHead(a.vocab_size, a.dim, a.norm_eps, a.hc_eps)

                with M.set_dtype(torch.float32):
                    self.hc_head_fn = torch.nn.Parameter(
                        torch.empty(a.hc_mult, a.hc_mult * a.dim)
                    )
                    self.hc_head_base = torch.nn.Parameter(torch.empty(a.hc_mult))
                    self.hc_head_scale = torch.nn.Parameter(torch.empty(1))

        import wanferenz.kernels.fp4_grouped as fp4_grouped

        self._moe_banked = banked = fp4_grouped.bank_layout(self.layers, preserve=False)
        if banked:
            print(
                f"[v4] stage[{lo}:{hi}): grouped-MoE bank layout on {banked} layer(s) — the routed "
                f"experts ARE the bank, no duplicate",
                flush=True,
            )

        import wanferenz.kernels.fp8_vector as fp8_vector

        self._shared_banked = shared_banked = fp8_vector.shared_bank_layout(self.layers)
        if shared_banked:
            print(
                f"[v4] stage[{lo}:{hi}): shared-expert w13 bank on {shared_banked} layer(s) — "
                f"w1+w3 serve as ONE fp8 launch",
                flush=True,
            )
        if self._fast:
            self._reserve_chunk_scratch()
        for m in self._owned_modules():
            m.eval()

        self._tap_ids = tuple(
            sorted(li for li in a.dspark_target_layer_ids if lo <= li < hi)
        )
        self._spec = False

        self._spec_depth = (
            spec_depth
            if spec_depth is not None
            else int(os.environ.get("V4_SPEC_DEPTH", 16))
        )
        self._spec_ckpts = deque(maxlen=self._spec_depth)
        self._last_tap = {}
        self._pos = 0
        self._replaying = False
        self.reset()

        self._block_graphs = None
        self._capture_mode = _capture_mode()
        if self._capture_mode != "off":
            why = self._graph_refusal()
            if why:
                print(
                    f"[v4] GRAPH REFUSED for stage[{lo}:{hi}): {why} — staying eager",
                    flush=True,
                )
            elif self._capture_mode == "whole":
                import wanferenz.graphs.decoder as _wl

                mm = "graph" if _wl.V4_MOE_IN_GRAPH else "eager"
                self._block_graphs = [
                    _wl.CapturedDecoder(L, self, moe_mode=mm) for L in self.layers
                ]
            else:
                self._block_graphs = [_IslandSequence(L, self) for L in self.layers]

    def _graph_refusal(self):

        if not str(self.device).startswith("cuda"):
            return f"device is {self.device} (CUDA graphs are a GPU-only capture)"
        return None

    @property
    def _spec_ckpt(self):

        return self._spec_ckpts[-1] if self._spec_ckpts else None

    def _owned_modules(self):
        yield self.layers
        for n in ("embed_tokens", "norm", "lm_head"):
            if getattr(self, n, None) is not None:
                yield getattr(self, n)

    def _reserve_chunk_scratch(self):

        with torch.no_grad():
            for L in self.layers:
                a = L.attn
                b, n, d = a.kv_cache.shape
                a.kv_cache = a.kv_cache.new_zeros(b, n + self._chunk_cap, d)

    def _compressors(self):

        for L in self.layers:
            attn = L.attn
            if not attn.compress_ratio:
                continue
            yield attn.compressor, False
            if attn.indexer is not None:
                yield attn.indexer.compressor, True

    def reset(self):

        with torch.no_grad():
            for L in self.layers:
                L.attn.kv_cache.zero_()
                if L.attn.compress_ratio and L.attn.indexer is not None:
                    L.attn.indexer.kv_cache.zero_()
            for c, _ in self._compressors():
                c.kv_state.zero_()
                c.score_state.fill_(float("-inf"))
        self._pos = 0
        self._last_tap = {}
        self._spec_ckpts.clear()

    def _snapshot(self, window_row=None):

        win = self.args.window_size
        snap = []
        with torch.no_grad():
            for L in self.layers:
                if window_row is None:
                    snap.append({"win": L.attn.kv_cache[:, :win].clone()})
                else:
                    snap.append(
                        {
                            "win": L.attn.kv_cache[
                                :, window_row : window_row + 1
                            ].clone(),
                            "row": window_row,
                        }
                    )
            for c, _ in self._compressors():
                snap.append(
                    {
                        "kv_state": c.kv_state.clone(),
                        "score_state": c.score_state.clone(),
                    }
                )
        return snap

    def _restore_windows(self, snap):

        win = self.args.window_size
        with torch.no_grad():
            for L, e in zip(self.layers, snap):
                lo = e.get("row", 0)
                hi = lo + 1 if "row" in e else win
                L.attn.kv_cache[:, lo:hi].copy_(e["win"])

    def _restore(self, snap):

        self._restore_windows(snap)
        with torch.no_grad():
            n = len(self.layers)
            for (c, _), e in zip(self._compressors(), snap[n:]):
                c.kv_state.copy_(e["kv_state"])
                c.score_state.copy_(e["score_state"])

    def _replay(self, h, ids, start_pos):

        self._replaying = True
        try:
            with torch.no_grad():
                for i in range(h.shape[1]):
                    self._run(h[:, i : i + 1], ids[:, i : i + 1], start_pos + i, {})
        finally:
            self._replaying = False
        self._pos = start_pos + h.shape[1]

    def _seek(self, start_pos):

        if start_pos == self._pos:
            return
        if start_pos > self._pos:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: start_pos {start_pos} is ahead of the {self._pos} "
                f"tokens this stage has seen — a gap means the skipped tokens were never fed "
                f"through this block's layers (reset() first, or replay from {self._pos})"
            )
        ck = next(
            (
                c
                for c in reversed(self._spec_ckpts)
                if c["start_pos"] <= start_pos <= c["start_pos"] + c["s"]
            ),
            None,
        )
        if ck is None:
            covered = (
                "none"
                if not self._spec_ckpts
                else f"[{self._spec_ckpts[0]['start_pos']}, "
                f"{self._spec_ckpts[-1]['start_pos'] + self._spec_ckpts[-1]['s']}]"
            )
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: cannot rewind {self._pos} -> {start_pos}; the spec "
                f"checkpoint covers {covered} (the last W speculative frames' ring). A rollback only "
                f"rewinds inside that ring — arm _spec (the reset's `spec` flag does it) and rewind "
                f"before `commit` or the maxlen cap drops the checkpoint. reset() is the only other "
                f"way back."
            )
        if "row" in ck["state"][0]:
            for later in reversed(self._spec_ckpts):
                if later is ck:
                    break
                self._restore_windows(later["state"])
        self._restore(ck["state"])
        self._pos = ck["start_pos"]
        n = start_pos - ck["start_pos"]
        if n:
            self._replay(ck["h"][:, :n], ck["ids"][:, :n], ck["start_pos"])
        self._pos = start_pos
        while self._spec_ckpts and self._spec_ckpts[-1]["start_pos"] >= ck["start_pos"]:
            self._spec_ckpts.pop()

    def commit(self, pos):

        keep = deque(
            (c for c in self._spec_ckpts if c["start_pos"] + c["s"] > pos),
            maxlen=self._spec_ckpts.maxlen,
        )
        self._spec_ckpts = keep

    def embed(self, token_ids):

        if self.embed_tokens is None:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: no embedding — head=False "
                f"(pass dspark=True on a tail that needs one for the drafter)"
            )
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        with torch.no_grad():
            h = self.embed_tokens(ids)
            return h.unsqueeze(2).repeat(1, 1, self.args.hc_mult, 1)

    def _run(self, h, ids, start_pos, taps):

        bg = self._block_graphs
        graphed = (
            bg is not None and not self._replaying and start_pos > 0 and h.shape[1] == 1
        )
        for i, (li, L) in enumerate(zip(range(self.lo, self.hi), self.layers)):
            h = bg[i].run(h, ids, start_pos) if graphed else L(h, start_pos, ids)
            if self._dspark and li in self._tap_ids:
                taps.setdefault(li, []).append(h.mean(dim=2).detach().clone())

        return h.clone() if graphed else h

    def _chunk_ok(self, s):

        return self._fast and 1 < s <= min(self._chunk_cap, self.args.window_size)

    def _run_chunk(self, h, ids, start_pos, taps):

        capacity = _DecodeWindow(
            self.args, start_pos, h.shape[1], h.shape[0], self.device, self._chunk_cap
        )
        for L in self.layers:
            L.attn._chunk = capacity
        try:
            return self._run(h, ids, start_pos, taps)
        finally:
            for L in self.layers:
                L.attn._chunk = None

    def forward(self, h, ids, start_pos):

        if ids is None:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: forward() needs the token ids, not just the hidden "
                f"state — the first {self.args.n_hash_layers} layers route their MoE by "
                f"tid2eid[input_ids] (see this module's docstring). Carry them with the payload."
            )
        h = h.to(device=self.device, dtype=self.dtype)
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        s = h.shape[1]
        if ids.shape[:2] != h.shape[:2]:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: ids {tuple(ids.shape)} do not match "
                f"the payload's [b, s] = {tuple(h.shape[:2])}"
            )
        self._seek(start_pos)
        if self._spec and start_pos > 0:
            state = (
                self._snapshot(start_pos % self.args.window_size)
                if self._snapshot_mode == "delta" and s == 1
                else self._snapshot()
            )
            self._spec_ckpts.append(
                {
                    "start_pos": start_pos,
                    "s": s,
                    "state": state,
                    "h": h.clone(),
                    "ids": ids.clone(),
                }
            )
        taps = {}
        with torch.no_grad():
            if start_pos == 0 or s == 1:
                out = self._run(h, ids, start_pos, taps)
            elif self._chunk_ok(s):
                out = self._run_chunk(h, ids, start_pos, taps)
            else:
                out = torch.cat(
                    [
                        self._run(
                            h[:, i : i + 1], ids[:, i : i + 1], start_pos + i, taps
                        )
                        for i in range(s)
                    ],
                    dim=1,
                )
        self._last_tap = {li: torch.cat(v, dim=1) for li, v in taps.items()}
        self._pos = start_pos + s
        return out

    def greedy_head(self, h):

        if not (V4_HEAD_GRAPH and self.tail and h.is_cuda and h.shape[:2] == (1, 1)):
            return None
        from wanferenz.graphs.output_head import CapturedOutput

        graph = getattr(self, "_head_graph", None)
        if graph is None:
            graph = self._head_graph = CapturedOutput(self, h)
        return graph.run(h)

    def logits_all(self, h, full_logits=True):

        if not self.tail:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: logits_all() on a non-tail stage"
            )
        h = h.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            x = self.layers[-1].hc_head(
                h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
            )
            return self.lm_head(self.norm(x), full_logits=full_logits)

    def tail_main_hidden(self):

        want = tuple(sorted(self.args.dspark_target_layer_ids))
        missing = [li for li in want if not self.lo <= li < self.hi]
        if missing:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: dspark target layers {missing} are not in this "
                f"stage's range. The drafter consumes all of {want} concatenated, so they must land "
                f"on ONE stage — at V4's shape that means the tail owns at least "
                f"{max(want) - min(want) + 1} layers."
            )
        if not self._dspark:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: tail_main_hidden() with _dspark off "
                f"— arm the stage before the forward whose taps you want"
            )
        if not self._last_tap:
            raise RuntimeError(
                f"v4 stage[{self.lo}:{self.hi}]: no taps recorded — forward() first"
            )
        return torch.cat([self._last_tap[li] for li in want], dim=-1)

    def load(self, d=None):

        d = d or V4_DIR
        wm = tensor_locations(d)
        for li in range(self.lo, self.hi):
            prefix = f"layers.{li}."
            names = [n for n in wm if n.startswith(prefix)]
            if not names:
                raise RuntimeError(
                    f"v4 stage[{self.lo}:{self.hi}]: no tensor under {prefix!r} in "
                    f"{d!r} — wrong checkpoint, or convert.py was never run"
                )
            state = {n[len(prefix) :]: raw(n, d) for n in names}
            self.layers[li - self.lo].load_state_dict(state, strict=True)
            del state
            _close_tensor_sources(d)
        if self.embed_tokens is not None:
            state = {"weight": raw("embed.weight", d)}
            self.embed_tokens.load_state_dict(state)
            del state
            _close_tensor_sources(d)
        if self.tail:
            state = {"weight": raw("norm.weight", d)}
            self.norm.load_state_dict(state)
            del state
            _close_tensor_sources(d)
            state = {"weight": raw("head.weight", d)}
            self.lm_head.load_state_dict(state)
            del state
            _close_tensor_sources(d)

            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                t, p = raw(n, d), getattr(self, n)
                if tuple(t.shape) != tuple(p.shape):
                    raise RuntimeError(
                        f"v4 stage[{self.lo}:{self.hi}]: {n} is {tuple(t.shape)} in "
                        f"the checkpoint, this config declares {tuple(p.shape)}"
                    )
                with torch.no_grad():
                    p.data.copy_(t)
                del t
                _close_tensor_sources(d)
        return self

    def __repr__(self):
        import wanferenz.kernels.torch_ops as torch_ops

        kinds = "".join(
            "W"
            if not L.attn.compress_ratio
            else ("I" if L.attn.indexer is not None else "C")
            for L in self.layers
        )
        return (
            f"<V4Stage [{self.lo}:{self.hi}) {kinds} head={self.head} tail={self.tail} "
            f"{self.dtype} on {self.device} pos={self._pos} "
            f"kernels={torch_ops.backend()} "
            f"dspark={'on' if self._dspark else 'off'} taps={list(self._tap_ids)} "
            f"spec={'on' if self._spec else 'off'}/{self._spec_depth} snapshot={self._snapshot_mode} "
            f"graph={self._capture_mode if self._block_graphs is not None else 'off'} "
            f"fast_verify={f'<={self._chunk_cap}' if self._fast else 'off'} "
            f"moe={self._moe_status()} "
            f"ref_slim={self._ref_slim_status()} "
            f"levers={controls.summary(self)}>"
        )

    def _moe_status(self):

        chain = controls.moe_chain(ref())
        s = ">".join(chain) or "?"
        return f"{s}/{self._moe_banked}" if "grouped" in chain else s

    def _ref_slim_status(self):

        M = ref()
        on = [
            n
            for n, f in (("indexer", M.Indexer.forward), ("noqat", M.act_quant))
            if getattr(f, "_v4_ref_slim", False)
        ]
        return "+".join(on) if on else "off"


class _CapturedIsland:
    def __init__(self, fn, examples):
        global _GRAPH_COUNT
        self.fn = fn
        self.ins = tuple(e.clone() for e in examples)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                self.fn(*self.ins)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            outs = self.fn(*self.ins)
        self.outs = outs if isinstance(outs, tuple) else (outs,)
        torch.cuda.synchronize()
        _GRAPH_COUNT += 1

    def run(self, *args):
        for buf, a in zip(self.ins, args):
            buf.copy_(a)
        self.graph.replay()
        return self.outs if len(self.outs) > 1 else self.outs[0]


class _IslandSequence:
    def __init__(self, L, stage):
        self.L = L
        self.st = stage
        self.g = None
        self.eager = False

    def _g1_fn(self, h):
        y, post, comb = self.L.hc_pre(
            h, self.L.hc_attn_fn, self.L.hc_attn_scale, self.L.hc_attn_base
        )
        return self.L.attn_norm(y), post, comb

    def _g2_fn(self, attn_out, residual, post_a, comb_a):
        x2 = self.L.hc_post(attn_out, residual, post_a, comb_a)
        y, post_f, comb_f = self.L.hc_pre(
            x2, self.L.hc_ffn_fn, self.L.hc_ffn_scale, self.L.hc_ffn_base
        )
        return self.L.ffn_norm(y), x2, post_f, comb_f

    def _g3_fn(self, ffn_out, residual, post_f, comb_f):
        return self.L.hc_post(ffn_out, residual, post_f, comb_f)

    def _build(self):

        a = self.st.args
        dt, dev = self.st.dtype, self.st.device
        h = torch.zeros(1, 1, a.hc_mult, a.dim, dtype=dt, device=dev)
        with torch.no_grad():
            attn_in, post_a, comb_a = self._g1_fn(h)
            attn_out = torch.zeros_like(attn_in)
            ffn_in, x2, post_f, comb_f = self._g2_fn(attn_out, h, post_a, comb_a)
            ffn_out = torch.zeros_like(ffn_in)
        self.g = (
            _CapturedIsland(self._g1_fn, (h,)),
            _CapturedIsland(self._g2_fn, (attn_out, h, post_a, comb_a)),
            _CapturedIsland(self._g3_fn, (ffn_out, x2, post_f, comb_f)),
        )

    def run(self, h, ids, start_pos):

        global _GRAPH_SKIPPED
        if self.eager:
            return self.L(h, start_pos, ids)
        if self.g is None:
            if _GRAPH_COUNT + 3 > V4_GRAPH_MAX:
                self.eager, _GRAPH_SKIPPED = True, _GRAPH_SKIPPED + 3
                print(
                    f"[v4] graph budget V4_GRAPH_MAX={V4_GRAPH_MAX} spent — layer "
                    f"{self.L.layer_id} stays eager",
                    flush=True,
                )
                return self.L(h, start_pos, ids)
            try:
                self._build()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                torch.cuda.synchronize()
                self.eager, self.g, _GRAPH_SKIPPED = True, None, _GRAPH_SKIPPED + 3
                print(
                    f"[v4] graph capture failed for layer {self.L.layer_id}: "
                    f"{type(e).__name__}: {e} — layer stays eager",
                    flush=True,
                )
                return self.L(h, start_pos, ids)
        g1, g2, g3 = self.g
        attn_in, post_a, comb_a = g1.run(h)
        attn_out = self.L.attn(attn_in, start_pos)
        ffn_in, x2, post_f, comb_f = g2.run(attn_out, h, post_a, comb_a)
        ffn_out = self.L.ffn(ffn_in, ids)
        return g3.run(ffn_out, x2, post_f, comb_f)


def verify_locally(lo, hi, d):

    have = os.path.exists(f"{d}/config.json") and glob.glob(
        os.path.join(d, "model*-mp*.safetensors")
    )
    if have:
        args = config(d)
    else:
        import wanferenz.model.oracle as oracle

        args = oracle.miniature_parameters()
        print(
            f"[v4] no converted checkpoint at {d!r} — running at oracle.miniature_parameters() scale "
            f"(n_layers={args.n_layers} dim={args.dim}) with random weights",
            flush=True,
        )
        hi = min(hi, args.n_layers)
    st = LayerPartition(
        lo, hi, args, head=(lo == 0), tail=(hi == args.n_layers), device="cpu"
    )
    if have:
        st.load(d)
    else:
        holder = torch.nn.Module()
        holder.layers = st.layers
        if st.embed_tokens is not None:
            holder.embed_tokens = st.embed_tokens
        if st.tail:
            holder.norm, holder.lm_head = st.norm, st.lm_head
            for n in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                setattr(holder, n, getattr(st, n))
        oracle.initialize_parameters(holder, 0)
    print(st, flush=True)
    ids = torch.randint(0, args.vocab_size, (1, 9))
    h = (
        st.embed(ids)
        if st.head
        else torch.randn(1, 9, args.hc_mult, args.dim, dtype=st.dtype)
    )
    h = st.forward(h, ids, 0)
    per_tok = h.shape[2] * h.shape[3] * h.element_size()
    print(
        f"[v4] prefill h {tuple(h.shape)} — {per_tok / 1024:.1f} KiB/token on the wire "
        f"({h.shape[2]}x a plain transformer's hidden state)",
        flush=True,
    )
    nxt = torch.randint(0, args.vocab_size, (1, 1))
    h = st.forward(h[:, -1:], nxt, 9)
    if st.tail:
        print(f"[v4] decode logits {tuple(st.logits_all(h).shape)}", flush=True)
    print(f"[v4] {st}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=V4_DIR)
    ap.add_argument("--layers", type=int, nargs=2, default=[0, 4], metavar=("LO", "HI"))
    a = ap.parse_args()
    verify_locally(a.layers[0], a.layers[1], a.dir)
