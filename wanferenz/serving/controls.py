import os
import re
import sys


STAGE = "stage"
COORDINATOR = "coordinator"
BOTH = "both"


V4_LEVERS_STRICT = os.environ.get("V4_LEVERS_STRICT", "0") not in ("", "0")


class RuntimeView:
    def __init__(self, side, stage=None):
        self.side = side
        self.stage = stage

        rc = sys.modules.get("wanferenz.model.oracle")
        self.mod = getattr(rc, "_REF", None) if rc is not None else None
        if self.mod is None:
            self.mod = sys.modules.get("dsv4_model")


def _mod(name):

    return sys.modules.get(name)


def _flag(modname, attr, on="on", off="off"):

    m = _mod(modname)
    if m is None:
        return "absent"
    return on if getattr(m, attr, False) else off


_MOE_LAYERS = (
    ("multi", "wanferenz.kernels.expert_batch", "_v4_multi"),
    ("grouped", "wanferenz.kernels.fp4_grouped", "_v4_grouped"),
    ("decode", "wanferenz.kernels.expert_dispatch", "_v4_decode_fast"),
)


def moe_chain(mod):

    moe = getattr(mod, "MoE", None)
    fn = getattr(moe, "forward", None)
    out, seen = [], set()
    while fn is not None:
        if id(fn) in seen:
            out.append("CYCLE")
            break
        seen.add(id(fn))
        for label, modname, marker in _MOE_LAYERS:
            if getattr(fn, marker, False):
                out.append(label)
                owner = _mod(modname)
                if owner is None:
                    out.append("ORPHAN")
                    fn = None
                else:
                    fn = getattr(owner, "_REF_FORWARD", None)
                break
        else:
            out.append("ref")
            fn = None
    return out


def _moe_check(label, modname, attr):

    def check(ctx):
        req = _flag(modname, attr)
        if ctx.mod is None:
            return req, "unloaded", None
        chain = moe_chain(ctx.mod)
        broken = [w for w in ("CYCLE", "ORPHAN") if w in chain]
        if broken:
            return req, ">".join(chain), False
        obs = "on" if label in chain else "off"
        if label == "grouped" and obs == "on" and ctx.stage is not None:
            banked = getattr(ctx.stage, "_moe_banked", 0)
            return req, f"on/{banked}", (_agree(req, obs) and banked > 0)
        return req, obs, _agree(req, obs)

    return check


def _agree(req, obs):

    if req == "absent":
        return None if obs == "off" else False
    return req == obs


def _check_cuda_graph(ctx):

    st = _mod("wanferenz.serving.partition")
    req = st._capture_mode() if st is not None else "absent"
    if ctx.stage is None:
        return req, "no-stage", None
    obs = (
        ctx.stage._capture_mode
        if getattr(ctx.stage, "_block_graphs", None) is not None
        else "off"
    )
    st = _mod("wanferenz.serving.partition")
    skipped = getattr(st, "_GRAPH_SKIPPED", 0) if st is not None else 0
    return req, (f"{obs}(-{skipped})" if skipped else obs), _agree(req, obs)


def _check_moe_in_graph(ctx):

    wl = _mod("wanferenz.graphs.decoder")
    req = _flag("wanferenz.graphs.decoder", "V4_MOE_IN_GRAPH")
    if ctx.stage is None or wl is None:
        return req, "no-stage" if wl is not None else "unloaded", None
    if getattr(ctx.stage, "_block_graphs", None) is None:
        return req, "off", _agree(req, "off")
    got, refused, undecided = wl.moe_graph_coverage(ctx.stage)
    asked = got + refused + undecided
    if not asked:
        return req, "off", _agree(req, "off")
    if undecided == asked:
        return req, f"armed/{asked}", None
    return req, f"on/{got}-of-{asked}", (_agree(req, "on") and got > 0)


def _check_fp8_gemv(ctx):

    m = _mod("wanferenz.kernels.fp8_vector")
    if m is None:
        return "absent", "unloaded", None
    raw = getattr(m, "V4_FP8_GEMV", "")
    obs = m.gemv_status()
    if not raw or raw == "0":
        return "off", obs, _agree("off", obs)
    if getattr(m, "_GEMV_MODE", None) is None:
        return raw, obs, False
    if ctx.mod is not None and not getattr(ctx.mod.fp8_gemm, "_v4_fp8_gemv", False):
        return "on", "not-installed", False
    if obs == "armed":
        return "on", obs, None
    return "on", obs, obs.startswith("on/")


def _check_fp8_shared(ctx):

    m = _mod("wanferenz.kernels.fp8_vector")
    req = _flag("wanferenz.kernels.fp8_vector", "V4_FP8_SHARED")
    if m is None or ctx.mod is None:
        return req, "unloaded", None
    if not getattr(ctx.mod.Expert.forward, "_v4_fp8_shared", False):
        return req, "off", _agree(req, "off")
    obs = m.shared_status()
    if ctx.stage is not None:
        banked = getattr(ctx.stage, "_shared_banked", 0)
        return (
            req,
            f"{obs}/banked-{banked}",
            (
                None
                if obs == "armed"
                else (_agree(req, "on") and banked > 0 and obs.startswith("on/"))
            ),
        )
    if obs == "armed":
        return req, obs, None
    return req, obs, (_agree(req, "on") and obs.startswith("on/"))


def _check_fast_verify(ctx):
    st = _mod("wanferenz.serving.partition")
    req = (
        "on"
        if (st is not None and st.V4_FAST_VERIFY)
        else ("absent" if st is None else "off")
    )
    if ctx.stage is None:
        return req, "no-stage", None
    obs = "on" if getattr(ctx.stage, "_fast", False) else "off"
    return req, obs, _agree(req, obs)


def _check_spec_depth(ctx):

    if ctx.side == COORDINATOR:
        vp = _mod("wanferenz.serving.chain")
        req = str(vp.V4_SPEC_DEPTH) if vp is not None else "absent"
        fact = _NOTES.get("V4_SPEC_DEPTH")
        if fact is None:
            return req, "no-job-yet", None
        return req, fact, _agree(req, fact)
    req = os.environ.get("V4_SPEC_DEPTH", "16")
    if ctx.stage is None:
        return req, "no-stage", None
    obs = str(getattr(ctx.stage, "_spec_depth", ""))
    return req, obs, req == obs


def _check_fp8_wire(ctx):

    vp = _mod("wanferenz.serving.chain")
    req = _flag("wanferenz.serving.chain", "V4_FP8_WIRE")
    if vp is None:
        return req, "unloaded", None
    try:
        import torch

        frame, _ = vp._make_step_frame(torch.zeros(1, 1, 4, 8), [[0]], 0, None)
        obs = "on" if "h8" in frame else "off"
    except Exception as e:
        return req, f"unprobed({type(e).__name__})", None
    return req, obs, _agree(req, obs)


def _check_dspark_fast(ctx):

    req = _flag("wanferenz.decoding.advance", "V4_DSPARK_FAST")
    fact = _NOTES.get("V4_DSPARK_FAST")
    if fact is not None:
        return req, fact, _agree(req, fact)
    d = _mod("wanferenz.decoding.dspark")
    if d is None:
        return req, "unloaded", None
    obs = (
        "on"
        if getattr(d.DSparkPartition.advance_and_draft, "_v4_dspark_fast", False)
        else "off"
    )
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, obs, None
    return req, obs, _agree(req, obs)


def _check_dspark_moe(ctx):

    req = _flag("wanferenz.decoding.expert_bank", "V4_DSPARK_MOE")
    fact = _NOTES.get("V4_DSPARK_MOE")
    if fact is not None:
        return req, fact, _agree(req, fact)
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, "not-tail", None
    return req, "no-drafter-yet", None


def _check_dspark_block(ctx):

    d = _mod("wanferenz.decoding.dspark")
    if d is None:
        req = "absent"
    else:
        v = getattr(d, "V4_DSPARK_BLOCK", 0)
        req = str(v) if v else "off"
    fact = _NOTES.get("V4_DSPARK_BLOCK")
    if fact is not None:
        return (
            req,
            fact,
            (None if fact == "off" else False) if req == "absent" else req == fact,
        )
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, "not-tail", None
    return req, "no-drafter-yet", None


def _check_draft_top2(ctx):

    req = _flag("wanferenz.decoding.dspark", "V4_DRAFT_TOP2")
    fact = _NOTES.get("V4_DRAFT_TOP2")
    if fact is not None:
        return req, fact, _agree(req, fact)
    if ctx.stage is not None and not getattr(ctx.stage, "tail", False):
        return req, "not-tail", None
    return req, "no-drafter-yet", None


def _ref_slim_check(attr, marker_of):
    def check(ctx):
        req = _flag("wanferenz.kernels.short_context", attr)
        if ctx.mod is None:
            return req, "unloaded", None
        obs = "on" if marker_of(ctx.mod) else "off"
        return req, obs, _agree(req, obs)

    return check


def _check_lazy_draft(ctx):

    req = _flag("wanferenz.serving.chain", "V4_LAZY_DRAFT")
    fact = _NOTES.get("V4_LAZY_DRAFT")
    if fact is None:
        return req, "no-job-yet", None
    return req, fact, _agree(req, fact)


def _check_pipelined(ctx):
    req = _flag("wanferenz.serving.chain", "V4_PIPELINED_SPEC")
    fact = _NOTES.get("V4_PIPELINED_SPEC")
    if fact is None:
        return req, "no-job-yet", None

    return req, fact, not (req == "on" and fact == "off")


def _check_conf_gate(ctx):
    req = _flag("wanferenz.serving.chain", "V4_DSPARK_CONF_GATE")
    fact = _NOTES.get("V4_DSPARK_CONF_GATE")
    if fact is None:
        return req, "no-job-yet", None
    return req, fact, _agree(req, fact)


def _check_refill_floor(ctx):

    vp = _mod("wanferenz.serving.chain")
    req = str(vp.V4_REFILL_FLOOR) if vp is not None else "absent"
    fact = _NOTES.get("V4_REFILL_FLOOR")
    if fact is None:
        return req, "no-job-yet", None
    set_here = os.environ.get("V4_REFILL_FLOOR", "") not in ("", "0")
    return req, fact, (req == fact) if set_here else True


def _value_check(modname, attr):

    def check(ctx):
        m = _mod(modname)
        if m is None:
            return "-", "unloaded", None
        return str(getattr(m, attr, "?")), str(getattr(m, attr, "?")), None

    return check


class ControlRule:
    def __init__(self, env, side, owner, check, doc, kind="switch"):
        self.env = env
        self.side = side
        self.owner = owner
        self.check = check
        self.doc = doc

        self.kind = kind

    def wanted_here(self, side):
        return self.side in (side, BOTH)


LEVERS = (
    ControlRule(
        "V4_MOE_GROUPED",
        STAGE,
        "wanferenz.kernels.fp4_grouped",
        _moe_check("grouped", "wanferenz.kernels.fp4_grouped", "V4_MOE_GROUPED"),
        "grouped fp4 MoE kernel for the s==1 score-routed decode step (CUDA only)",
    ),
    ControlRule(
        "V4_FP8_GEMV",
        STAGE,
        "wanferenz.kernels.fp8_vector",
        _check_fp8_gemv,
        "occupancy-tiled fp8 GEMM at decode shapes (M<=32), self-gated torch.equal per (N,K)",
    ),
    ControlRule(
        "V4_FP8_SHARED",
        STAGE,
        "wanferenz.kernels.fp8_vector",
        _check_fp8_shared,
        "the shared expert's w1+w3 as one banked fp8 launch (+ one act_quant), gated per half",
    ),
    ControlRule(
        "V4_MOE_DECODE",
        STAGE,
        "wanferenz.kernels.expert_dispatch",
        _moe_check("decode", "wanferenz.kernels.expert_dispatch", "V4_MOE_DECODE"),
        "sync-free MoE dispatch at s==1 (DEFAULT ON)",
    ),
    ControlRule(
        "V4_MOE_MULTI",
        STAGE,
        "wanferenz.kernels.expert_batch",
        _moe_check("multi", "wanferenz.kernels.expert_batch", "V4_MOE_MULTI"),
        "sync-free MoE dispatch at the DSpark drafter's small block shape",
    ),
    ControlRule(
        "V4_CUDA_GRAPH",
        STAGE,
        "wanferenz.serving.partition",
        _check_cuda_graph,
        "decode-step CUDA graphs: off / island / whole",
    ),
    ControlRule(
        "V4_MOE_IN_GRAPH",
        STAGE,
        "wanferenz.graphs.decoder",
        _check_moe_in_graph,
        "capture the routed MoE INSIDE the whole-layer graph (needs whole mode + grouped)",
    ),
    ControlRule(
        "V4_FAST_VERIFY",
        STAGE,
        "wanferenz.serving.partition",
        _check_fast_verify,
        "chunked verify path (one pass per layer over a speculation chunk)",
    ),
    ControlRule(
        "V4_REF_SLIM",
        STAGE,
        "wanferenz.kernels.short_context",
        _ref_slim_check(
            "V4_REF_SLIM", lambda m: getattr(m.Indexer.forward, "_v4_ref_slim", False)
        ),
        "skip the Indexer's scoring while every compressed slot is selected",
    ),
    ControlRule(
        "V4_REF_SLIM_NOQAT",
        STAGE,
        "wanferenz.kernels.short_context",
        _ref_slim_check(
            "V4_REF_SLIM_NOQAT", lambda m: getattr(m.act_quant, "_v4_ref_slim", False)
        ),
        "skip the inplace KV/Q QAT quant-simulation (APPROXIMATE — not in the ring recipe)",
    ),
    ControlRule(
        "V4_DSPARK_FAST",
        STAGE,
        "wanferenz.decoding.advance",
        _check_dspark_fast,
        "tail only: cache-advance-only drafter forwards",
    ),
    ControlRule(
        "V4_DSPARK_MOE",
        STAGE,
        "wanferenz.decoding.expert_bank",
        _check_dspark_moe,
        "tail only: the drafter's block MoE as one grouped fp4 launch per matrix kind",
    ),
    ControlRule(
        "V4_DSPARK_BLOCK",
        STAGE,
        "wanferenz.decoding.dspark",
        _check_dspark_block,
        "tail only: draft the block at this width instead of the trained dspark_block_size — "
        "deeper proposals from the same tap, lifting the pipelined in-flight cap to width+1",
    ),
    ControlRule(
        "V4_DRAFT_TOP2",
        STAGE,
        "wanferenz.decoding.dspark",
        _check_draft_top2,
        "tail only: ship the drafter's runner-up token per block slot, so the coordinators can "
        "count the rescue rate that gates tree speculation",
    ),
    ControlRule(
        "V4_FP8_WIRE",
        STAGE,
        "wanferenz.serving.chain",
        _check_fp8_wire,
        "fp8-pack h on the forward leg (every non-tail stage packs its own output)",
    ),
    ControlRule(
        "V4_SPEC_DEPTH",
        BOTH,
        "wanferenz.serving.chain",
        _check_spec_depth,
        "pipelined speculation depth: the coordinator's window AND the stage's rollback ring",
    ),
    ControlRule(
        "V4_PIPELINED_SPEC",
        COORDINATOR,
        "wanferenz.serving.chain",
        _check_pipelined,
        "stream s=1 frames without waiting for their replies",
    ),
    ControlRule(
        "V4_LAZY_DRAFT",
        COORDINATOR,
        "wanferenz.serving.chain",
        _check_lazy_draft,
        "hint the tail to skip drafting a block the round will not consume",
    ),
    ControlRule(
        "V4_DSPARK_CONF_GATE",
        COORDINATOR,
        "wanferenz.serving.chain",
        _check_conf_gate,
        "serial DSpark only: trim the tail's offered block length by confidence",
    ),
    ControlRule(
        "V4_REFILL_FLOOR",
        COORDINATOR,
        "wanferenz.serving.chain",
        _check_refill_floor,
        "pipelined refill floor: consume a reply's block at or below this in-flight level "
        "(1 = drain-only, the shipped round)",
    ),
    ControlRule(
        "V4_SPEC_SNAPSHOT",
        STAGE,
        "wanferenz.serving.partition",
        _value_check("wanferenz.serving.partition", "V4_SPEC_SNAPSHOT"),
        "speculative window checkpoints: full buffer or single-token row journal",
        kind="knob",
    ),
    ControlRule(
        "V4_MOE_MULTI_MAX",
        STAGE,
        "wanferenz.kernels.expert_batch",
        _value_check("wanferenz.kernels.expert_batch", "V4_MOE_MULTI_MAX"),
        "widest block the multi-dispatch path claims",
        kind="knob",
    ),
    ControlRule(
        "V4_FAST_VERIFY_MAX",
        STAGE,
        "wanferenz.serving.partition",
        _value_check("wanferenz.serving.partition", "V4_FAST_VERIFY_MAX"),
        "chunk positions reserved per layer for the fast verify scratch",
        kind="knob",
    ),
    ControlRule(
        "V4_GRAPH_MAX",
        STAGE,
        "wanferenz.serving.partition",
        _value_check("wanferenz.serving.partition", "V4_GRAPH_MAX"),
        "process-wide captured-graph budget",
        kind="knob",
    ),
    ControlRule(
        "V4_KERNELS",
        STAGE,
        "wanferenz.kernels.torch_ops",
        _value_check("wanferenz.kernels.torch_ops", "V4_KERNELS"),
        "kernel backend selection (tilelang / cpu)",
        kind="knob",
    ),
)

LEVERS_BY_ENV = {lv.env: lv for lv in LEVERS}


NON_LEVER_ENV = {
    "V4_HEAD_GRAPH": "single-token main output head and greedy argmax CUDA graph",
    "V4_DSPARK_FULL_GRAPH": "complete greedy drafter CUDA graph, requires V4_DSPARK_FAST",
    "V4_DIR": "checkpoint directory (emitted by partition_command)",
    "V4_DEV": "torch device (emitted by partition_command)",
    "V4_DTYPE": "default construction dtype",
    "V4_MAX_SEQ": "stage build-out: kv cache length",
    "V4_MAX_BATCH": "stage build-out: batch width",
    "V4_KEEPWARM": "transport keep-warm",
    "V4_KEEPWARM_MS": "transport keep-warm period",
    "V4_DIAL_CONNECT_TIMEOUT": "inter-stage dial timeout",
    "V4_DIAL_RETRY_S": "inter-stage dial retry window",
    "V4_TIMING": "instrumentation",
    "V4_TIMING_EVERY": "instrumentation period",
    "V4_DSPARK_CONF_MIN": "conf-gate knob, consumed with V4_DSPARK_CONF_GATE",
    "V4_DSPARK_CONF_THRESH": "conf-gate knob, consumed with V4_DSPARK_CONF_GATE",
    "V4_DSPARK_GRAPH": "drafter head graph, rides on V4_DSPARK_FAST and is CUDA-only",
    "V4_LEVERS_STRICT": "this file: turn a finding into a refusal to serve",
}


_NOTES = {}


def note(env, value):

    _NOTES[env] = ("on" if value else "off") if isinstance(value, bool) else str(value)


def notes():
    return dict(_NOTES)


class ControlObservation:
    __slots__ = ("env", "side", "requested", "observed", "verdict", "why")

    def __init__(self, env, side, requested, observed, verdict, why=""):
        self.env, self.side = env, side
        self.requested, self.observed = requested, observed
        self.verdict, self.why = verdict, why

    @property
    def bad(self):

        return self.verdict in ("MISMATCH", "UNKNOWN")

    def __repr__(self):
        return f"<{self.env} req={self.requested} obs={self.observed} {self.verdict}>"


def _stray_env():

    known = set(LEVERS_BY_ENV) | set(NON_LEVER_ENV)
    return sorted(k for k in os.environ if k.startswith("V4_") and k not in known)


def audit(side=STAGE, stage=None):

    ctx = RuntimeView(side, stage)
    out = []
    for lv in LEVERS:
        set_here = os.environ.get(lv.env, "") not in ("", "0")
        if set_here and not lv.wanted_here(side):
            where = COORDINATOR if lv.side == COORDINATOR else STAGE
            out.append(
                ControlObservation(
                    lv.env,
                    lv.side,
                    os.environ[lv.env],
                    "n/a",
                    "OTHER SIDE",
                    f"{lv.side}-side lever: THIS {side} process never reads it. It is "
                    f"carried here because ENG_ENV propagates it; it configures the "
                    f"{where}, and only the {where}'s own audit can judge it.",
                )
            )
            continue
        if not lv.wanted_here(side):
            continue
        try:
            req, obs, ok = lv.check(ctx)
        except Exception as e:
            out.append(
                ControlObservation(
                    lv.env,
                    lv.side,
                    "?",
                    "?",
                    "UNJUDGED",
                    f"check raised {type(e).__name__}: {e}",
                )
            )
            continue

        if set_here and req == "absent" and ok is None:
            out.append(
                ControlObservation(
                    lv.env,
                    lv.side,
                    req,
                    obs,
                    "UNJUDGED",
                    f"{lv.owner} is not imported yet; no parsed value to compare",
                )
            )
            continue
        if set_here and req in ("off", "0", "False", "absent"):
            out.append(
                ControlObservation(
                    lv.env,
                    lv.side,
                    os.environ[lv.env],
                    f"module parsed {req}",
                    "MISMATCH",
                    f"{lv.env} is set in this process's environment but {lv.owner} parsed "
                    f"it {req} — it was almost certainly set after the module was imported",
                )
            )
            continue
        verdict = (
            "OK"
            if ok
            else ("VALUE" if lv.kind == "knob" else "UNJUDGED")
            if ok is None
            else "MISMATCH"
        )
        out.append(
            ControlObservation(
                lv.env,
                lv.side,
                req,
                obs,
                verdict,
                "requested and live state disagree" if verdict == "MISMATCH" else "",
            )
        )
    for name in _stray_env():
        out.append(
            ControlObservation(
                name,
                "?",
                os.environ[name],
                "nothing",
                "UNKNOWN",
                "set in this process but no v4 module reads this name",
            )
        )
    return out


def summary(stage=None, side=STAGE):

    try:
        bad = [f for f in audit(side, stage) if f.bad]
    except Exception as e:
        return f"audit-failed({type(e).__name__})"
    if not bad:
        return "ok"
    return "!" + ",".join(f"{f.env}:{f.verdict.split()[0].lower()}" for f in bad)


def report(side=STAGE, stage=None, strict=None, out=None):

    findings = audit(side, stage)
    bad = [f for f in findings if f.bad]
    w = max((len(f.env) for f in findings), default=10)
    lines = [f"{'=' * 26} V4 LEVER AUDIT ({side}) {'=' * 26}"]
    for f in findings:
        line = f"  {f.env:<{w}}  requested={f.requested:<10} observed={f.observed:<12} {f.verdict}"
        lines.append(line + (f" — {f.why}" if f.why else ""))
    if bad:
        lines.append(
            f"V4 LEVER AUDIT: {len(bad)} PROBLEM(S) — "
            + ", ".join(f"{f.env}({f.verdict})" for f in bad)
        )
        if not (V4_LEVERS_STRICT if strict is None else strict):
            lines.append(
                "V4 LEVER AUDIT: continuing anyway (set V4_LEVERS_STRICT=1 to refuse to serve)"
            )
    else:
        lines.append(f"V4 LEVER AUDIT: all {len(findings)} clean")
    lines.append("=" * 74)
    text = "\n".join(lines)
    print(text, file=(out if out is not None else sys.stderr), flush=True)
    if bad and (V4_LEVERS_STRICT if strict is None else strict):
        raise RuntimeError(
            "V4_LEVERS_STRICT: refusing to serve with "
            + ", ".join(f"{f.env}={f.verdict}" for f in bad)
            + ". Every number this process produced would be about a configuration nobody asked for."
        )
    return text


ENGINE_MODULES = (
    "serving/chain.py",
    "serving/command.py",
    "serving/controls.py",
    "serving/partition.py",
    "model/architecture.py",
    "model/assets.py",
    "model/chat.py",
    "model/conversion.py",
    "model/oracle.py",
    "model/tensor_layout.py",
    "kernels/expert_batch.py",
    "kernels/expert_dispatch.py",
    "kernels/fp4_grouped.py",
    "kernels/fp8_vector.py",
    "kernels/gb10.py",
    "kernels/reference.py",
    "kernels/short_context.py",
    "kernels/sparse_attention.py",
    "kernels/torch_ops.py",
    "decoding/advance.py",
    "decoding/dspark.py",
    "decoding/expert_bank.py",
    "decoding/ngram.py",
    "graphs/decoder.py",
    "graphs/drafter.py",
    "graphs/output_head.py",
)

_ENV_RE = re.compile(r"""environ(?:\.get)?[.(\[]+["'](V4_[A-Z0-9_]+)["']""")


def env_names_in_source(root=None):

    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = set()
    for name in ENGINE_MODULES:
        p = os.path.join(root, name)
        if os.path.exists(p):
            with open(p) as f:
                found |= set(_ENV_RE.findall(f.read()))
    return found


def verify_locally():

    class _F:
        pass

    class _MoE:
        pass

    class _Mod:
        MoE = _MoE

    def ref_fwd(self, x, ids):
        return x

    _MoE.forward = ref_fwd
    assert moe_chain(_Mod) == ["ref"], moe_chain(_Mod)

    import types

    dec = types.ModuleType("wanferenz.kernels.expert_dispatch")
    dec._REF_FORWARD = ref_fwd
    dec.V4_MOE_DECODE = True

    def dec_fwd(self, x, ids):
        return x

    dec_fwd._v4_decode_fast = True
    sys.modules["wanferenz.kernels.expert_dispatch"] = dec
    _MoE.forward = dec_fwd
    assert moe_chain(_Mod) == ["decode", "ref"], moe_chain(_Mod)

    mul = types.ModuleType("wanferenz.kernels.expert_batch")
    mul._REF_FORWARD = dec_fwd
    mul.V4_MOE_MULTI = True

    def mul_fwd(self, x, ids):
        return x

    mul_fwd._v4_multi = True
    sys.modules["wanferenz.kernels.expert_batch"] = mul
    _MoE.forward = mul_fwd
    chain = moe_chain(_Mod)
    assert chain == ["multi", "decode", "ref"], chain
    print(f"chain      {'>'.join(chain)}   (the shape that used to report 'ref')")

    mul._REF_FORWARD = mul_fwd
    assert moe_chain(_Mod) == ["multi", "CYCLE"], moe_chain(_Mod)
    ctx_cyc = RuntimeView(STAGE)
    ctx_cyc.mod = _Mod
    assert (
        _moe_check("multi", "wanferenz.kernels.expert_batch", "V4_MOE_MULTI")(ctx_cyc)[
            2
        ]
        is False
    )
    mul._REF_FORWARD = dec_fwd

    del sys.modules["wanferenz.kernels.expert_dispatch"]
    assert moe_chain(_Mod) == ["multi", "decode", "ORPHAN"], moe_chain(_Mod)
    sys.modules["wanferenz.kernels.expert_dispatch"] = dec

    ctx = RuntimeView(STAGE)
    ctx.mod = _Mod
    for label, modname, flag in (
        ("multi", "wanferenz.kernels.expert_batch", "V4_MOE_MULTI"),
        ("decode", "wanferenz.kernels.expert_dispatch", "V4_MOE_DECODE"),
    ):
        req, obs, ok = _moe_check(label, modname, flag)(ctx)
        assert ok is True, (label, req, obs, ok)

    dec.V4_MOE_DECODE = True
    _MoE.forward = ref_fwd
    req, obs, ok = _moe_check(
        "decode", "wanferenz.kernels.expert_dispatch", "V4_MOE_DECODE"
    )(ctx)
    assert (req, obs, ok) == ("on", "off", False), (req, obs, ok)
    print(f"finding    V4_MOE_DECODE requested={req} observed={obs} -> MISMATCH")
    print(f"registry   {len(LEVERS)} levers, {len(NON_LEVER_ENV)} non-lever knobs")
    print(
        f"sides      stage={sum(l.side == STAGE for l in LEVERS)} "
        f"coordinator={sum(l.side == COORDINATOR for l in LEVERS)} "
        f"both={sum(l.side == BOTH for l in LEVERS)}"
    )
    scraped = env_names_in_source()
    missing = scraped - set(LEVERS_BY_ENV) - set(NON_LEVER_ENV)
    assert not missing, f"unregistered levers in the source: {sorted(missing)}"
    print(f"source     {len(scraped)} V4_* names read by the engine, all registered")
    print("OK")


if __name__ == "__main__":
    verify_locally()
