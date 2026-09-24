import threading

import pytest
import torch

import wanferenz.serving.chain as VP

pytestmark = pytest.mark.integration
send_frame = VP.send_frame
PROMPT = [3, 9, 17, 2, 41]
NEW = 5
NEW_LONG = 10


def _accept_pattern(r):

    d = {
        k: r[k]
        for k in (
            "tokens",
            "frames",
            "drafted",
            "generated",
            "accepted",
            "cancels",
            "cycles",
            "g",
            "accept_hist",
            "max_inflight",
            "mean_inflight",
        )
    }
    d["fenced"] = r["stale_replies"] + r["unsent_frames"]
    return d


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):

    R = pytest.importorskip("wanferenz.model.oracle")
    pytest.importorskip("wanferenz.serving.partition")
    pytest.importorskip("safetensors.torch")
    args = R.miniature_parameters()
    d = str(tmp_path_factory.mktemp("v4pipe"))
    model = R.create_oracle(args)
    VP._save_test_checkpoint(d, args, model)
    return d, args, VP._oracle_tokens(model, PROMPT, NEW)


class _Ring:
    def __init__(
        self, ckpt_dir, args, ranges, receipts=False, tail_box_g=1, dspark=False
    ):
        self.n = len(ranges)
        self.layer_count = args.n_layers
        self.receipts = receipts
        ports = VP._reserve_test_ports(self.n)
        relay_i = self.n - tail_box_g
        events = [threading.Event() for _ in range(self.n)]
        self.threads = []
        for i, (lo, hi) in enumerate(ranges):
            nxt = None if i == self.n - 1 else f"127.0.0.1:{ports[i + 1]}"
            ret_relay = (
                f"127.0.0.1:{ports[-1]}" if (tail_box_g > 1 and i == relay_i) else None
            )
            t = threading.Thread(
                target=VP.run_partition,
                kwargs=dict(
                    stage=i,
                    nstages=self.n,
                    lo=lo,
                    hi=hi,
                    port=ports[i],
                    nxt=nxt,
                    ckpt_dir=ckpt_dir,
                    device="cpu",
                    receipts=receipts,
                    key_path=f"{ckpt_dir}/s{i}.key",
                    ret_relay=ret_relay,
                    dspark=dspark,
                    ready=events[i],
                ),
                daemon=True,
            )
            t.start()
            self.threads.append(t)
        for e in events:
            assert e.wait(120), "a stage never came up"

        tail_port = ports[relay_i] if tail_box_g > 1 else ports[-1]
        self.head_addr = f"127.0.0.1:{ports[0]}"
        self.tail_addr = f"127.0.0.1:{tail_port}"
        self.pipe, self.ret = VP.connect_chain(
            self.head_addr, self.tail_addr, timeout=60
        )

    def decode_greedy(self, prompt, max_new, nonce=None):
        return VP.decode_greedy(
            self.pipe,
            self.ret,
            prompt,
            max_new,
            nonce=nonce,
            receipts=self.receipts,
            layer_count=self.layer_count,
            timeout=60,
        )

    def spec(self, prompt, max_new, nonce=None, **kw):
        return VP.decode_proposals(
            self.pipe,
            self.ret,
            prompt,
            max_new,
            nonce=nonce,
            receipts=self.receipts,
            layer_count=self.layer_count,
            timeout=60,
            **kw,
        )

    def dspark(self, prompt, max_new, nonce=None, **kw):
        return VP.decode_dspark(
            self.pipe,
            self.ret,
            prompt,
            max_new,
            nonce=nonce,
            receipts=self.receipts,
            layer_count=self.layer_count,
            timeout=60,
            **kw,
        )

    def pipelined(self, prompt, max_new, nonce=None, **kw):
        return VP.stream_dspark(
            self.pipe,
            self.ret,
            prompt,
            max_new,
            nonce=nonce,
            receipts=self.receipts,
            layer_count=self.layer_count,
            timeout=60,
            **kw,
        )

    def close(self):
        try:
            send_frame(self.pipe, {"op": "stop"})
        except OSError:
            pass
        for t in self.threads:
            t.join(timeout=10)


@pytest.mark.parametrize(
    "ranges",
    [
        [(0, 3), (3, 6), (6, 8)],
        [(0, 4), (4, 8)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
    ],
)
def test_full_ring_greedy_matches_reference(tiny, ranges):

    d, args, ref = tiny
    ring = _Ring(d, args, ranges)
    try:
        got = ring.decode_greedy(list(PROMPT), NEW)["tokens"]
    finally:
        ring.close()
    assert got == ref, f"{ranges}: ring {got} != ref {ref}"


def test_full_ring_dspark_matches_reference(tiny):

    d, args, ref = tiny
    ring = _Ring(d, args, VP._draft_layer_ranges(args, 3), receipts=True, dspark=True)
    try:
        r = ring.dspark(list(PROMPT), NEW, nonce="dspark-nonce")
    finally:
        ring.close()
    assert r["tokens"] == ref, f"dspark ring {r['tokens']} != greedy {ref}"
    assert r["receipts_ok"] is True, "a drafted job must still settle its receipts"
    assert r["rounds"] > 1 and r["drafted"] == r["rounds"] - 1, (
        f"the drafter never proposed a block: {r['rounds']} rounds, {r['drafted']} drafted"
    )


@pytest.fixture(scope="module")
def long_ref(tiny):

    R = pytest.importorskip("wanferenz.model.oracle")
    _d, args, _ref = tiny
    return VP._oracle_tokens(R.create_oracle(args), PROMPT, NEW_LONG)


class _BlockScriptDrafter:
    def __init__(self, ckpt_dir, block_for):
        self.ckpt_dir = ckpt_dir
        self.block_for = block_for
        self.pipelined = False
        self._inner = None

    def on_chunk(self, msg, st, out):
        if self._inner is None:
            self._inner = VP._dspark().ring_drafter(st, self.ckpt_dir)
        self._inner.pipelined = self.pipelined
        r = self._inner.on_chunk(msg, st, out) or {}
        if r.get("draft"):
            r = dict(r, draft=list(self.block_for(int(msg["start_pos"]), r["draft"])))

            self._inner._last = (int(msg["start_pos"]), list(r["draft"]))
        return r


def _oracle_blocks(ref, poison_at=()):

    bad = set(poison_at)

    def at(p):
        i = p - len(PROMPT)
        return 1 if p in bad else (ref[i] if 0 <= i < len(ref) else 1)

    return lambda start_pos, real: [at(start_pos + 2 + i) for i in range(len(real))]


def test_pipelined_ring_matches_the_reference_and_the_serial_path(tiny):

    d, args, ref = tiny
    ring = _Ring(d, args, VP._draft_layer_ranges(args, 3), receipts=True, dspark=True)
    try:
        serial = ring.dspark(list(PROMPT), NEW, nonce="pipe-serial")
        piped = ring.pipelined(list(PROMPT), NEW, nonce="pipe-nonce")
    finally:
        ring.close()
    assert piped["tokens"] == ref, f"pipelined ring {piped['tokens']} != greedy {ref}"
    assert piped["tokens"] == serial["tokens"], "pipelined and serial dspark disagree"
    assert piped["receipts_ok"] is True, (
        "a pipelined job must still settle its receipts"
    )
    assert piped["max_inflight"] > 1, (
        "nothing was ever pipelined — this is greedy with extra steps"
    )
    assert piped["cancels"] > 0 and piped["stale_replies"] > 0, (
        "the rejection path (and with it the W-deep rewind) never ran"
    )


def _drafter_state(double):

    t = double._inner.tail
    return t.pos, [b.attn.kv_cache.clone() for b in t.mtp]


def _same_state(a, b):
    return (
        a[0] == b[0]
        and len(a[1]) == len(b[1])
        and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))
    )


def test_lazy_drafting_is_a_noop_in_the_zero_accept_regime(tiny):

    d, args, ref = tiny
    ring = _Ring(d, args, VP._draft_layer_ranges(args, 3), receipts=True, dspark=True)
    try:
        e = ring.pipelined(list(PROMPT), NEW, nonce="zero-eager")
        r = ring.pipelined(list(PROMPT), NEW, nonce="zero-lazy", lazy=True)
    finally:
        ring.close()
    assert r["tokens"] == ref == e["tokens"], (
        f"zero-accept lazy ring {r['tokens']} != {ref}"
    )
    assert _accept_pattern(r) == _accept_pattern(e), (
        "the lever moved a round it cannot help"
    )
    assert r["cancels"] > 0, "the rejection path never ran"
    assert r["drafts_issued"] == e["drafts_issued"], (
        "a round that consumes every block it is given must still be billed for every one"
    )


def test_lazy_drafting_under_a_floor_on_a_real_ring_is_bit_identical(tiny, long_ref):

    d, args, _ref = tiny
    eager = _BlockScriptDrafter(d, _oracle_blocks(long_ref))
    lazy = _BlockScriptDrafter(d, _oracle_blocks(long_ref))
    ring = _Ring(d, args, VP._draft_layer_ranges(args, 3), receipts=True, dspark=True)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(VP, "TAIL_DRAFTER", eager)
            e = ring.pipelined(list(PROMPT), NEW_LONG, nonce="floor-eager", floor=2)
        e_state = _drafter_state(eager)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(VP, "TAIL_DRAFTER", lazy)
            r = ring.pipelined(
                list(PROMPT), NEW_LONG, nonce="floor-lazy", floor=2, lazy=True
            )
        r_state = _drafter_state(lazy)
    finally:
        ring.close()
    assert r["tokens"] == long_ref, (
        f"lazy under floor=2: {r['tokens']} != greedy {long_ref}"
    )
    assert _accept_pattern(r) == _accept_pattern(e), (
        "lazy under a floor changed the round itself"
    )
    assert _same_state(r_state, e_state), (
        "the drafter's mtp window/cursor diverged under the floor"
    )
    assert r["drafts_issued"] <= e["drafts_issued"]
    assert r["receipts_ok"] is True


def test_pipelined_and_serial_dspark_jobs_interleave_on_one_warm_ring(tiny):

    d, args, ref = tiny
    ring = _Ring(d, args, VP._draft_layer_ranges(args, 3), dspark=True)
    try:
        first = ring.pipelined([7, 7, 7, 1, 2, 9], NEW)
        mid = ring.dspark(list(PROMPT), NEW)
        warm = ring.pipelined(list(PROMPT), NEW)
    finally:
        ring.close()
    assert first["tokens"] != ref, (
        "the warming job must be a DIFFERENT sequence to prove anything"
    )
    assert mid["tokens"] == ref and warm["tokens"] == ref
