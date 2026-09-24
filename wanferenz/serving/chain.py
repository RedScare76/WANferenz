import argparse
import collections
import json
import os
import queue
import select
import socket
import struct
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import torch


import wanferenz.serving.controls as controls

try:
    from wanferenz.protocol.framing import send_frame, receive_frame
except ImportError:
    from wanferenz.protocol.framing import send_frame, receive_frame
try:
    from wanferenz.protocol.attestation import (
        ActivationAttestor,
        ensure_identity,
        public_identity,
        validate_coverage,
        receipt_body,
        AttestationFailure,
    )
except ImportError:
    from wanferenz.protocol.attestation import (
        ActivationAttestor,
        ensure_identity,
        public_identity,
        validate_coverage,
        receipt_body,
        AttestationFailure,
    )


ENG_IN, FWD_RING, FWD_RET = 29610, 29611, 29612


ENG_LOCAL_BASE = 29620
NODELAY = (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


DIAL_RETRY_S = float(os.environ.get("V4_DIAL_RETRY_S", "0") or 0)
DIAL_CONNECT_TIMEOUT = float(os.environ.get("V4_DIAL_CONNECT_TIMEOUT", "5") or 5)
NODE_KEY_PATH = os.environ.get("WANFERENZ_NODE_KEY", "/root/.wanferenz_node_key")


SWARM_TOKEN = os.environ.get("WANFERENZ_SWARM_TOKEN") or None
RECEIPTS = os.environ.get("WANFERENZ_RECEIPTS", "") not in ("", "0")

V4_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
N_LAYERS = 43


V4_MAX_SEQ = int(os.environ.get("V4_MAX_SEQ", "8192") or 8192)
V4_MAX_BATCH = int(os.environ.get("V4_MAX_BATCH", "1") or 1)


_LEG_ERRORS = (OSError, EOFError)


V4_TIMING = bool(int(os.environ.get("V4_TIMING", "0") or 0))
V4_TIMING_EVERY = int(os.environ.get("V4_TIMING_EVERY", "0") or 0)


class _SilentClock:
    __slots__ = ()

    def lap(self, phase, obj=None):
        pass

    def sync(self):
        pass

    def frame(self, s=1):
        pass

    def report(self):
        pass

    def start(self):
        pass


_NO_TIMER = _SilentClock()


def _tensor_bytes(obj):

    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def _frame_bytes(obj):

    return obj if isinstance(obj, int) else _tensor_bytes(obj)


class _PhaseClock:
    __slots__ = ("tag", "phases", "cuda", "every", "acc", "by", "n", "npos", "t")

    def __init__(self, tag, phases, device=None, every=0):
        self.tag, self.phases = tag, phases
        self.cuda = str(device or "").startswith("cuda")
        self.every = every
        self.start()

    def start(self):
        self.acc = dict.fromkeys(self.phases, 0.0)
        self.by = dict.fromkeys(self.phases, 0)
        self.n = self.npos = 0
        self.t = time.perf_counter()

    def lap(self, phase, obj=None):
        now = time.perf_counter()
        self.acc[phase] += now - self.t
        self.t = now
        if obj is not None:
            self.by[phase] += _frame_bytes(obj)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def frame(self, s=1):
        self.n += 1
        self.npos += int(s)
        if self.every and self.n % self.every == 0:
            self.report()

    def _phase(self, p):
        ms = 1000.0 * self.acc[p] / self.n
        if not self.by[p]:
            return f"{p}={ms:.2f}"
        kb = self.by[p] / 1024.0 / self.n
        mbps = (self.by[p] / 1e6) / self.acc[p] if self.acc[p] > 0 else float("inf")
        return f"{p}={ms:.2f}[{kb:.0f}KB {mbps:.1f}MB/s]"

    def report(self):
        if not self.n:
            return
        on_box = sum(1000.0 * self.acc[p] / self.n for p in self.phases if p != "recv")
        print(
            f"{self.tag} timing n={self.n} s={self.npos / self.n:.1f} "
            + " ".join(self._phase(p) for p in self.phases)
            + f" on_box={on_box:.2f} ms/frame",
            flush=True,
        )


def _timer(tag, phases, device=None):

    if not V4_TIMING:
        return _NO_TIMER
    return _PhaseClock(tag, phases, device, V4_TIMING_EVERY)


def _v4():

    import wanferenz.serving.partition as partition

    return partition


def _dspark():

    import wanferenz.decoding.dspark as dspark

    return dspark


_BUILD_LOCK = threading.Lock()


def _tbytes(t):
    return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _ids_tensor(v):

    t = v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.int64)
    return t.to(torch.int64).contiguous()


def _payload_bytes(h, ids):

    return _tbytes(h) + _tbytes(ids)


V4_FP8_WIRE = bool(int(os.environ.get("V4_FP8_WIRE", "0") or 0))


def _pack_t(t):

    f = t.detach().float()
    scale = (
        (f.abs().amax(-1, keepdim=True) / 448.0)
        .clamp(min=1e-8, max=torch.finfo(torch.bfloat16).max)
        .to(torch.bfloat16)
    )
    q = (f / scale.float()).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    return q, scale.squeeze(-1).contiguous()


def _unpack_t(q, scale):

    return q.to(torch.bfloat16) * scale.unsqueeze(-1)


def _wire_bytes(h, ids, hs):

    return _tbytes(h) + _tbytes(ids) + _tbytes(hs)


def _recv_hids(msg, signer):

    h, ids = msg["h"], _ids_tensor(msg["ids"])
    packed = torch.is_tensor(h) and h.dtype == torch.float8_e4m3fn
    if packed != ("h8" in msg):
        raise ValueError(
            f"step frame: h is {getattr(h, 'dtype', type(h))} but h8 is "
            f"{'present' if 'h8' in msg else 'absent'}"
        )
    if packed:
        hs = msg["h8"]
        if not torch.is_tensor(hs) or hs.shape != h.shape[:-1]:
            raise ValueError(
                f"step frame: h8 {getattr(hs, 'shape', type(hs))} does not scale "
                f"h {tuple(h.shape)} per (position, stream)"
            )
        in_b = _wire_bytes(h, ids, hs) if signer is not None else None
        return _unpack_t(h, hs), ids, in_b
    return h, ids, (_payload_bytes(h, ids) if signer is not None else None)


def _make_step_frame(h, ids, start_pos, signer):

    ids = _ids_tensor(ids)
    if V4_FP8_WIRE:
        qh, sh = _pack_t(h)
        qh = qh.detach().cpu().contiguous()
        frame = {"op": "step", "h": qh, "h8": sh, "ids": ids, "start_pos": start_pos}
        return frame, (_wire_bytes(qh, ids, sh) if signer is not None else None)
    hc = h.detach().cpu().contiguous()
    frame = {"op": "step", "h": hc, "ids": ids, "start_pos": start_pos}
    return frame, (_payload_bytes(hc, ids) if signer is not None else None)


_PASSTHRU = ("epoch", "cpos", "dnxt", "dprev")


def _fenced(msg, epoch):

    return bool(msg.get("fenced")) or int(msg.get("epoch", 0)) < epoch


V4_KEEPWARM = bool(int(os.environ.get("V4_KEEPWARM", "0") or 0))
V4_KEEPWARM_MS = float(os.environ.get("V4_KEEPWARM_MS", "150") or 150)


class _IdleHeartbeat:
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
        self.last = time.monotonic()
        self._stop = False
        self.on = V4_KEEPWARM and sock is not None
        if self.on:
            self.period = V4_KEEPWARM_MS / 1000.0
            threading.Thread(target=self._run, daemon=True, name="v4-keepwarm").start()

    def attach(self, sock):
        with self.lock:
            self.sock = sock
            self.last = time.monotonic()

    def send(self, msg):
        if not self.on:
            return send_frame(self.sock, msg)
        with self.lock:
            n = send_frame(self.sock, msg)
            self.last = time.monotonic()
        return n

    def _run(self):
        while not self._stop:
            time.sleep(self.period / 2)
            if time.monotonic() - self.last < self.period:
                continue
            if self.lock.acquire(blocking=False):
                try:
                    if self.sock is not None:
                        send_frame(self.sock, {"op": "noop"})
                        self.last = time.monotonic()
                except Exception:
                    pass
                finally:
                    self.lock.release()

    def stop(self):
        self._stop = True


def sample_token(logits_row, temp=0.0, gen=None):

    if temp and temp > 0:
        probs = (logits_row.float() / float(temp)).softmax(-1)
        return int(torch.multinomial(probs, 1, generator=gen).item())
    return int(logits_row.argmax().item())


TAIL_DRAFTER = None


_SPEC_POS_MARGIN = 64


def _job_max_pos(prompt_ids, max_new, spec_margin=0):

    return len(prompt_ids) + int(max_new) + int(spec_margin)


def _set_job_horizon(max_pos):

    import wanferenz.kernels.short_context as short_context

    horizon = None if os.environ.get("V4_CUDA_GRAPH") == "whole" else max_pos
    short_context.set_job_max_pos(horizon)


def _dial_window(timeout):

    return DIAL_RETRY_S or float(timeout)


def _dial(host, port, timeout, retry_s=None):

    connect_timeout = min(float(timeout), DIAL_CONNECT_TIMEOUT)
    window = float(retry_s) if retry_s is not None else _dial_window(timeout)
    deadline = time.time() + window
    last = None
    while True:
        try:
            s = socket.create_connection((host, int(port)), timeout=connect_timeout)
            s.setsockopt(*NODELAY)

            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            for _opt, _val in (
                ("TCP_KEEPIDLE", 15),
                ("TCP_KEEPINTVL", 15),
                ("TCP_KEEPCNT", 4),
            ):
                if hasattr(socket, _opt):
                    try:
                        s.setsockopt(socket.IPPROTO_TCP, getattr(socket, _opt), _val)
                    except OSError:
                        pass
            s.settimeout(float(timeout))
            return s
        except OSError as e:
            last = e
            if time.time() >= deadline:
                break
            time.sleep(0.25)
    raise RuntimeError(
        f"v4 stage: could not connect to {host}:{port} within {window:.0f}s "
        f"({type(last).__name__}: {last})"
    )


def _fwd_open(kw, nxt, timeout, msg, tag="[s]"):

    try:
        kw.send(msg)
        return kw.sock
    except _LEG_ERRORS as e:
        if nxt is None:
            raise
        print(
            f"{tag} forward leg was dead at job open ({type(e).__name__}); "
            f"rebuilding -> {nxt} and re-sending {msg.get('op')!r}",
            flush=True,
        )
    try:
        if kw.sock is not None:
            kw.sock.close()
    except OSError:
        pass
    kw.attach(None)
    sock = _dial(*nxt.rsplit(":", 1), timeout=timeout)
    kw.attach(sock)
    if SWARM_TOKEN is not None:
        send_frame(sock, {"op": "hello_pred", "token": SWARM_TOKEN})
    kw.send(msg)
    return sock


def checkpoint_parameters(ckpt_dir):

    V4 = _v4()
    args = V4.config(ckpt_dir)
    with open(os.path.join(ckpt_dir or V4.V4_DIR, "config.json")) as f:
        declared = json.load(f)
    for field, env, default in (
        ("max_seq_len", "V4_MAX_SEQ", V4_MAX_SEQ),
        ("max_batch_size", "V4_MAX_BATCH", V4_MAX_BATCH),
    ):
        if os.environ.get(env):
            setattr(args, field, int(os.environ[env]))
        elif field not in declared:
            setattr(args, field, default)
    return args


def _is_return_hello(msg):

    if not (isinstance(msg, dict) and msg.get("op") == "hello_return"):
        return False
    return SWARM_TOKEN is None or msg.get("token") == SWARM_TOKEN


def _is_pred_hello(msg):

    if not (isinstance(msg, dict) and msg.get("op") == "hello_pred"):
        return False
    return SWARM_TOKEN is None or msg.get("token") == SWARM_TOKEN


def run_partition(
    stage,
    nstages,
    lo,
    hi,
    port,
    nxt=None,
    *,
    ckpt_dir=None,
    args=None,
    device=None,
    receipts=None,
    key_path=None,
    timeout=600.0,
    bind="127.0.0.1",
    ready=None,
    ret_relay=None,
    dspark=False,
):

    V4 = _v4()
    head, tail = (stage == 0), (stage == nstages - 1)
    args = args if args is not None else checkpoint_parameters(ckpt_dir)
    dev = device or getattr(V4, "dev", "cuda")
    receipts = RECEIPTS if receipts is None else receipts
    key_path = key_path or NODE_KEY_PATH
    if str(dev).startswith("cuda"):
        torch.set_default_device(dev)

        torch.set_default_dtype(torch.bfloat16)
    with _BUILD_LOCK:
        st = V4.LayerPartition(
            lo, hi, args, head=head, tail=tail, dspark=(dspark and tail), device=dev
        )
        if ckpt_dir is not None:
            st.load(ckpt_dir)
    node_key = ensure_identity(key_path) if receipts else None
    print(f"[s{stage}] {st}", flush=True)

    controls.report(side=controls.STAGE, stage=st)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind, port))
    srv.listen(4)

    nxt_sock = (
        _dial(*nxt.rsplit(":", 1), timeout=timeout) if (not tail and nxt) else None
    )
    if nxt_sock is not None and SWARM_TOKEN is not None:
        send_frame(nxt_sock, {"op": "hello_pred", "token": SWARM_TOKEN})
    print(
        f"[s{stage}] listening {bind}:{port}"
        + (f" -> {nxt}" if nxt_sock is not None else " (tail)")
        + (f" pub={public_identity(node_key)}" if node_key is not None else ""),
        flush=True,
    )
    if ready is not None:
        ready.set()

    try:
        if tail:
            _serve_tail(
                st,
                srv,
                lo,
                hi,
                node_key,
                receipts,
                timeout,
                ckpt_dir=(ckpt_dir if dspark else None),
            )
        elif ret_relay is not None:
            _serve_relay_ingress(
                st,
                stage,
                srv,
                nxt_sock,
                nxt,
                head,
                lo,
                hi,
                node_key,
                receipts,
                timeout,
                ret_relay,
            )
        else:
            _serve_forward(
                st, stage, srv, nxt_sock, nxt, head, lo, hi, node_key, receipts, timeout
            )
    finally:
        _set_job_horizon(None)


_STRAY = (ConnectionError, OSError, ValueError, KeyError, TypeError, struct.error)


def _accept_pred(srv, timeout):

    pending = []
    while True:
        ready, _, _ = select.select([srv] + pending, [], [])
        if srv in ready:
            conn, _ = srv.accept()
            conn.setsockopt(*NODELAY)
            conn.settimeout(timeout)
            pending.append(conn)
            continue
        conn = next(c for c in pending if c in ready)
        pending.remove(conn)
        try:
            first = receive_frame(conn)
        except _STRAY as e:
            print(f"[s] dropped stray inbound: {type(e).__name__}", flush=True)
            try:
                conn.close()
            except Exception:
                pass
            continue
        if SWARM_TOKEN is not None:
            if not _is_pred_hello(first):
                print("[s] dropped inbound w/o valid greeting", flush=True)
                try:
                    conn.close()
                except Exception:
                    pass
                continue
            return conn, None
        return conn, first


def _warm_until_accept(nxt_sock, period=5.0):

    if nxt_sock is None:
        return lambda: None
    ev = threading.Event()

    def run():
        while not ev.wait(period):
            try:
                send_frame(nxt_sock, {"op": "noop"})
            except OSError:
                return

    t = threading.Thread(target=run, daemon=True)
    t.start()

    def stop():
        ev.set()
        t.join(timeout=2)

    return stop


def _serve_forward(
    st, stage, srv, nxt_sock, nxt, head, lo, hi, node_key, receipts, timeout
):

    stop_warm = _warm_until_accept(nxt_sock)
    try:
        conn, queued = _accept_pred(srv, timeout)
    finally:
        stop_warm()
    print(f"[s{stage}] predecessor connected", flush=True)
    reaccept = (lambda: _accept_pred(srv, timeout)) if head else None
    _forward_loop(
        st,
        stage,
        nxt_sock,
        nxt,
        head,
        lo,
        hi,
        node_key,
        receipts,
        conn,
        queued,
        timeout,
        reaccept=reaccept,
    )


def _forward_loop(
    st,
    stage,
    nxt_sock,
    nxt,
    head,
    lo,
    hi,
    node_key,
    receipts,
    conn,
    queued,
    timeout=600.0,
    reaccept=None,
):

    signer = None
    kw = _IdleHeartbeat(nxt_sock)
    tag = f"[s{stage}]"
    epoch = 0
    timer = _timer(
        tag, ("recv", "pre", "fwd", "out", "send"), getattr(st, "device", None)
    )
    with torch.no_grad():
        while True:
            if queued is not None:
                msg, queued = queued, None
            else:
                try:
                    msg = receive_frame(conn)
                except _LEG_ERRORS as e:
                    if reaccept is None:
                        raise

                    print(
                        f"{tag} coordinator disconnected ({type(e).__name__}); re-accepting "
                        f"— ring stays warm",
                        flush=True,
                    )
                    try:
                        conn.close()
                    except OSError:
                        pass
                    conn, queued = reaccept()
                    print(f"{tag} coordinator reconnected", flush=True)
                    continue
            op = msg.get("op")
            if op == "noop":
                continue
            if op == "reset":
                timer.start()
                st.reset()
                _set_job_horizon(msg.get("max_pos"))
                st._spec = bool(msg.get("spec"))
                st._dspark = bool(msg.get("dspark"))
                epoch = 0
                signer = (
                    ActivationAttestor(
                        node_key,
                        msg.get("swarm_id", "swarm"),
                        msg.get("job_id", "job"),
                        lo,
                        hi,
                        nonce=msg.get("nonce"),
                    )
                    if receipts
                    else None
                )
                _fwd_open(kw, nxt, timeout, msg, tag)
                continue
            if op == "receipt":
                timer.report()
                if signer is not None:
                    msg.setdefault("receipts", []).append(
                        {"stage": stage, **signer.finalize()}
                    )
                _fwd_open(kw, nxt, timeout, msg, tag)
                continue
            if op == "step":
                timer.lap("recv", msg)
                if _fenced(msg, epoch):
                    msg["fenced"] = True
                    kw.send(msg)
                    continue
                epoch = max(epoch, int(msg.get("epoch", 0)))
                if "cpos" in msg:
                    st.commit(int(msg["cpos"]))
                if head:
                    ids = _ids_tensor(msg["ids"])
                    h = st.embed(ids)
                    in_b = _payload_bytes(h, ids) if signer is not None else None
                else:
                    h, ids, in_b = _recv_hids(msg, signer)
                start_pos = int(msg["start_pos"])
                timer.lap("pre")
                h = st.forward(h, ids, start_pos)
                timer.sync()
                timer.lap("fwd")
                frame, out_b = _make_step_frame(h, ids, start_pos, signer)
                for k in _PASSTHRU:
                    if k in msg:
                        frame[k] = msg[k]
                if signer is not None:
                    signer.observe(in_b, out_b)
                timer.lap("out")
                timer.lap("send", kw.send(frame))
                timer.frame(h.shape[1])
                continue
            if op == "stop":
                kw.stop()
                try:
                    kw.send(msg)
                except OSError:
                    pass
                conn.close()
                return
            raise RuntimeError(f"s{stage}: unknown op {op!r}")


def _tail_bringup(srv, timeout):

    ret = pred = queued = None
    pending = []
    while ret is None or pred is None:
        ready, _, _ = select.select([srv] + pending, [], [])
        if srv in ready:
            conn, _ = srv.accept()
            conn.setsockopt(*NODELAY)
            conn.settimeout(timeout)
            pending.append(conn)
            continue
        conn = next(c for c in pending if c in ready)
        pending.remove(conn)
        try:
            first = receive_frame(conn)
        except _STRAY as e:
            print(f"[tail] dropped stray inbound: {type(e).__name__}", flush=True)
            try:
                conn.close()
            except Exception:
                pass
            continue
        if _is_return_hello(first):
            ret = conn
            send_frame(ret, "ret_ok")
        elif _is_pred_hello(first):
            pred = conn
        else:
            pred, queued = conn, first
    return ret, pred, queued


def _tail_logit_rows(st, h, start_pos):

    if start_pos == 0:
        return [st.logits_all(h, full_logits=False)]
    return [
        st.logits_all(h[:, j : j + 1], full_logits=False) for j in range(h.shape[1])
    ]


def _tail_predictions(st, h, start_pos, temp=0.0, gen=None):
    token = None
    if start_pos > 0 and temp == 0.0 and hasattr(st, "greedy_head"):
        token = st.greedy_head(h)
    if token is not None:
        out = {"token": token}
        if getattr(st, "_spec", False):
            out["tokens"] = [token]
        return out
    rows = _tail_logit_rows(st, h, start_pos)
    out = {"token": sample_token(rows[-1][0], temp, gen)}
    if getattr(st, "_spec", False):
        out["tokens"] = [int(r[0].argmax()) for r in rows]
    return out


def _tail_drafter(st, ckpt_dir, cache):

    if TAIL_DRAFTER is not None:
        return TAIL_DRAFTER
    if cache.get("drafter") is None:
        cache["drafter"] = _dspark().ring_drafter(st, ckpt_dir)
        print(f"[tail] dspark drafter {cache['drafter'].tail}", flush=True)
    return cache["drafter"]


class _ReturnLink:
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()

    def send(self, msg):
        with self.lock:
            try:
                return send_frame(self.sock, msg)
            except _LEG_ERRORS:
                return 0

    def swap(self, sock):
        with self.lock:
            old, self.sock = self.sock, sock
        return old

    def close(self):
        with self.lock:
            sock = self.sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _tail_return_reaccept(srv, chan, timeout):

    while True:
        try:
            ready, _, _ = select.select([srv], [], [])
            if srv not in ready:
                continue
            conn, _ = srv.accept()
            conn.setsockopt(*NODELAY)
            conn.settimeout(timeout)
        except OSError:
            return
        try:
            first = receive_frame(conn)
        except _STRAY as e:
            print(f"[tail] dropped stray reconnect: {type(e).__name__}", flush=True)
            try:
                conn.close()
            except OSError:
                pass
            continue
        if not _is_return_hello(first):
            print("[tail] dropped non-return reconnect on the tail port", flush=True)
            try:
                conn.close()
            except OSError:
                pass
            continue
        old = chan.swap(conn)
        try:
            send_frame(conn, "ret_ok")
        except _LEG_ERRORS:
            pass
        if old is not None:
            try:
                old.close()
            except OSError:
                pass
        print(
            "[tail] coordinator-return re-accepted — ring survived a coordinator restart",
            flush=True,
        )


def _serve_tail(st, srv, lo, hi, node_key, receipts, timeout, ckpt_dir=None):

    ret, pred, queued = _tail_bringup(srv, timeout)
    print("[tail] predecessor + coord-return connected", flush=True)
    chan = _ReturnLink(ret)
    threading.Thread(
        target=_tail_return_reaccept,
        args=(srv, chan, timeout),
        daemon=True,
        name="v4-tail-reaccept",
    ).start()

    signer = None
    temp, gen = 0.0, None
    drafter, built = None, {}
    epoch = 0
    timer = _timer(
        "[tail]",
        ("recv", "pre", "fwd", "out", "logits", "draft", "send"),
        getattr(st, "device", None),
    )
    with torch.no_grad():
        while True:
            msg = queued if queued is not None else receive_frame(pred)
            queued = None
            op = msg.get("op")
            if op == "noop":
                continue
            if op == "reset":
                timer.start()
                st.reset()
                _set_job_horizon(msg.get("max_pos"))
                st._spec = bool(msg.get("spec"))
                st._dspark = bool(msg.get("dspark"))
                epoch = 0
                try:
                    drafter = _tail_drafter(st, ckpt_dir, built) if st._dspark else None
                    if drafter is not None:
                        drafter.pipelined = bool(msg.get("pipelined"))
                except Exception as e:
                    print(
                        f"[tail] dspark unavailable: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    chan.send({"ok": False, "error": f"{type(e).__name__}: {e}"})
                    continue
                signer = (
                    ActivationAttestor(
                        node_key,
                        msg.get("swarm_id", "swarm"),
                        msg.get("job_id", "job"),
                        lo,
                        hi,
                        nonce=msg.get("nonce"),
                    )
                    if receipts
                    else None
                )
                temp = float(msg.get("temp", 0.0))
                gen = (
                    torch.Generator(device="cpu").manual_seed(int(msg["seed"]))
                    if temp > 0 and msg.get("seed") is not None
                    else None
                )
                chan.send("ok")
                continue
            if op == "receipt":
                timer.report()
                if signer is not None:
                    msg.setdefault("receipts", []).append(
                        {"stage": "tail", **signer.finalize()}
                    )
                chan.send(msg.get("receipts", []))
                continue
            if op == "step":
                timer.lap("recv", msg)
                if _fenced(msg, epoch):
                    send_frame(
                        ret,
                        {
                            "fenced": True,
                            "epoch": int(msg.get("epoch", 0)),
                            "pos": int(msg["start_pos"]),
                        },
                    )
                    continue
                epoch = max(epoch, int(msg.get("epoch", 0)))
                if "cpos" in msg:
                    st.commit(int(msg["cpos"]))
                h, ids, in_b = _recv_hids(msg, signer)
                start_pos = int(msg["start_pos"])
                timer.lap("pre")
                h = st.forward(h, ids, start_pos)
                timer.sync()
                timer.lap("fwd")
                if signer is not None:
                    signer.observe(in_b, _payload_bytes(h, ids))
                timer.lap("out")
                out = _tail_predictions(st, h, start_pos, temp, gen)
                if "epoch" in msg:
                    out["epoch"], out["pos"] = int(msg["epoch"]), start_pos
                timer.sync()
                timer.lap("logits")
                if drafter is not None:
                    out.update(drafter.on_chunk(msg, st, out) or {})
                timer.sync()
                timer.lap("draft")
                timer.lap("send", chan.send(out))
                timer.frame(h.shape[1])
                continue
            if op == "stop":
                chan.send({"token": None})
                pred.close()
                chan.close()
                return
            raise RuntimeError(f"tail: unknown op {op!r}")


def _dial_return(addr, timeout):

    host, port = addr.rsplit(":", 1)
    s = _dial(host, port, timeout)
    s.settimeout(timeout)
    send_frame(
        s,
        {"op": "hello_return", "token": SWARM_TOKEN}
        if SWARM_TOKEN is not None
        else {"op": "hello_return"},
    )
    receive_frame(s)
    return s


def _serve_relay_ingress(
    st, stage, srv, nxt_sock, nxt, head, lo, hi, node_key, receipts, timeout, ret_relay
):

    ret_up, pred, queued = _tail_bringup(srv, timeout)
    ret_down = _dial_return(ret_relay, timeout)

    def _pump():
        try:
            while True:
                send_frame(ret_up, receive_frame(ret_down))
        except OSError:
            pass

    threading.Thread(target=_pump, daemon=True).start()
    print(
        f"[s{stage}] relay ingress: coord-return <-> box tail {ret_relay}", flush=True
    )
    _forward_loop(
        st,
        stage,
        nxt_sock,
        nxt,
        head,
        lo,
        hi,
        node_key,
        receipts,
        pred,
        queued,
        timeout,
    )


def _split_address(s):
    h, _, p = s.rpartition(":")
    return h or "127.0.0.1", int(p)


def connect_chain(head, tail, timeout=600.0, token=None, retry_s=300):

    deadline = time.time() + retry_s
    last = None
    while time.time() < deadline:
        pipe = ret = None
        try:
            pipe = socket.create_connection(_split_address(head), timeout=timeout)
            pipe.setsockopt(*NODELAY)
            ret = socket.create_connection(_split_address(tail), timeout=timeout)
            ret.setsockopt(*NODELAY)
            ret.settimeout(timeout)
            if token:
                send_frame(pipe, {"op": "hello_pred", "token": token})
                send_frame(ret, {"op": "hello_return", "token": token})
            else:
                send_frame(ret, {"op": "hello_return"})
            receive_frame(ret)
            return pipe, ret
        except Exception as e:
            last = e
            for s in (pipe, ret):
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            time.sleep(min(2.0, max(0.25, deadline - time.time())))
    raise ConnectionError(
        f"v4 ring not reachable after {retry_s}s: {type(last).__name__}: {last}"
    )


def _sweep_receipts(pipe, ret, layer_count, nonce):

    send_frame(pipe, {"op": "receipt", "receipts": []})
    recs = receive_frame(ret) or []
    if not recs or layer_count is None:
        return recs, None
    wired = [receipt_body(r) for r in recs]
    try:
        validate_coverage(
            wired, int(layer_count), expected_nonce=nonce, check_chain=True
        )
        return recs, True
    except AttestationFailure:
        return recs, False


def decode_greedy(
    pipe,
    ret,
    prompt_ids,
    max_new,
    *,
    eos_ids=(),
    nonce=None,
    swarm_id="swarm",
    job_id="job",
    layer_count=None,
    receipts=False,
    temp=0.0,
    seed=0,
    timeout=600.0,
    on_token=None,
):

    ret.settimeout(timeout)
    send_frame(
        pipe,
        {
            "op": "reset",
            "swarm_id": swarm_id,
            "job_id": job_id,
            "nonce": nonce,
            "temp": float(temp),
            "seed": int(seed),
            "max_pos": _job_max_pos(prompt_ids, max_new),
        },
    )
    ack = receive_frame(ret)
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)
    pos = 0
    toks = []
    for _ in range(max_new):
        send_frame(pipe, {"op": "step", "ids": [ids], "start_pos": pos})
        pos += len(ids)
        rep = receive_frame(ret)
        tid = int(rep["token"]) if isinstance(rep, dict) else int(rep)
        toks.append(tid)
        if on_token is not None:
            on_token(tid)
        if tid in eos:
            break
        ids = [tid]

    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    return {
        "ok": True,
        "tokens": toks,
        "prompt_tokens": len(prompt_ids),
        "receipts": recs,
        "receipts_ok": receipts_ok,
    }


class _LastTokenProposal:
    def propose(self, seq, k):
        return [seq[-1] if seq else 0] * k


def _drafter_propose(drafter, ng):

    if drafter is None:
        try:
            from wanferenz.decoding.ngram import NgramHistory

            drafter = NgramHistory(ng=ng, margin=0)
        except ImportError:
            drafter = _LastTokenProposal()
    return drafter.propose if hasattr(drafter, "propose") else drafter


def decode_proposals(
    pipe,
    ret,
    prompt_ids,
    max_new,
    *,
    eos_ids=(),
    nonce=None,
    swarm_id="swarm",
    job_id="job",
    layer_count=None,
    receipts=False,
    timeout=600.0,
    on_token=None,
    K=4,
    ng=3,
    drafter=None,
):

    plan_verify_round = _dspark().plan_verify_round
    propose = _drafter_propose(drafter, ng)
    ret.settimeout(timeout)
    send_frame(
        pipe,
        {
            "op": "reset",
            "swarm_id": swarm_id,
            "job_id": job_id,
            "nonce": nonce,
            "temp": 0.0,
            "seed": 0,
            "spec": True,
            "max_pos": _job_max_pos(prompt_ids, max_new, K + 1),
        },
    )
    ack = receive_frame(ret)
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 spec ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)
    toks = []
    rounds, accepted_total = 0, 0
    hist = {}

    send_frame(pipe, {"op": "step", "ids": [ids], "start_pos": 0})
    rep = receive_frame(ret)
    pos = len(ids)
    cur = int(rep["token"] if isinstance(rep, dict) else rep)
    ids.append(cur)
    toks.append(cur)
    if on_token is not None:
        on_token(cur)

    while len(toks) < max_new and cur not in eos:
        drafts = propose(ids, K)
        rounds += 1
        send_frame(pipe, {"op": "step", "ids": [[cur] + drafts], "start_pos": pos})
        r = receive_frame(ret)["tokens"]
        n, committed = plan_verify_round(drafts, r)
        accepted_total += n
        hist[n] = hist.get(n, 0) + 1
        stop = False
        for t in committed:
            toks.append(int(t))
            ids.append(int(t))
            if on_token is not None:
                on_token(int(t))
            if int(t) in eos or len(toks) >= max_new:
                stop = True
                break
        cur = ids[-1]
        pos = len(ids) - 1
        if stop:
            break

    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    gen = len(toks)
    return {
        "ok": True,
        "tokens": toks,
        "prompt_tokens": len(prompt_ids),
        "receipts": recs,
        "receipts_ok": receipts_ok,
        "rounds": rounds,
        "generated": gen,
        "accepted": accepted_total,
        "g": (gen / rounds) if rounds else float(gen),
        "accept_hist": hist,
    }


V4_DSPARK_CONF_GATE = os.environ.get("V4_DSPARK_CONF_GATE", "0") not in ("", "0")
V4_DSPARK_CONF_THRESH = float(os.environ.get("V4_DSPARK_CONF_THRESH", "0") or 0)
V4_DSPARK_CONF_MIN = int(os.environ.get("V4_DSPARK_CONF_MIN", "1") or 1)


def _conf_send_len(confs, thresh, min_send):

    if not confs:
        return 0
    k = 0
    for c in confs:
        if float(c) < thresh:
            break
        k += 1
    return min(len(confs), max(min_send, k))


def decode_dspark(
    pipe,
    ret,
    prompt_ids,
    max_new,
    *,
    eos_ids=(),
    nonce=None,
    swarm_id="swarm",
    job_id="job",
    layer_count=None,
    receipts=False,
    timeout=600.0,
    on_token=None,
    conf_gate=None,
    conf_thresh=None,
    conf_min=None,
    conf_probe=None,
):

    plan_verify_round = _dspark().plan_verify_round
    gate = V4_DSPARK_CONF_GATE if conf_gate is None else bool(conf_gate)
    thresh = V4_DSPARK_CONF_THRESH if conf_thresh is None else float(conf_thresh)
    min_send = V4_DSPARK_CONF_MIN if conf_min is None else int(conf_min)
    controls.note("V4_DSPARK_CONF_GATE", gate)
    controls.note("V4_PIPELINED_SPEC", False)

    controls.note("V4_LAZY_DRAFT", False)
    ret.settimeout(timeout)
    send_frame(
        pipe,
        {
            "op": "reset",
            "swarm_id": swarm_id,
            "job_id": job_id,
            "nonce": nonce,
            "temp": 0.0,
            "seed": 0,
            "spec": True,
            "dspark": True,
            "max_pos": _job_max_pos(prompt_ids, max_new, _SPEC_POS_MARGIN),
        },
    )
    ack = receive_frame(ret)
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 dspark ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)
    toks = []
    rounds, accepted_total = 0, 0
    hist = {}

    send_frame(pipe, {"op": "step", "ids": [ids], "start_pos": 0})
    rep = receive_frame(ret)
    pos = len(ids)
    cur = int(rep["token"] if isinstance(rep, dict) else rep)
    ids.append(cur)
    toks.append(cur)
    if on_token is not None:
        on_token(cur)

    block, confs = [], []
    d2s = []
    rescue_by_depth = {}
    drafted, sent = 0, 0
    send_hist = {}
    while len(toks) < max_new and cur not in eos:
        rounds += 1

        drafts = block[: _conf_send_len(confs, thresh, min_send)] if gate else block
        drafted += bool(drafts)
        sent += len(drafts)
        send_hist[len(drafts)] = send_hist.get(len(drafts), 0) + 1
        send_frame(pipe, {"op": "step", "ids": [[cur] + drafts], "start_pos": pos})
        rep = receive_frame(ret)
        if "n" not in rep:
            raise RuntimeError(
                "v4 dspark: the tail's reply carries no accept length, so nothing is drafting on it "
                "— launch the tail stage with --dspark so it builds the MTP speculator"
            )
        n, committed = plan_verify_round(drafts, rep["tokens"])
        if int(rep["n"]) != n:
            raise RuntimeError(
                f"v4 dspark: the tail accepted {rep['n']} of round {rounds}'s {len(drafts)} drafts "
                f"and this coordinator accepted {n}, off the same drafts and the same replies. The "
                f"two ends of a lossless round have diverged, so the drafter is advancing over a "
                f"history the ring is not taking — one accept rule, and it is plan_verify_round."
            )
        accepted_total += n
        hist[n] = hist.get(n, 0) + 1
        if n < len(drafts) and n < len(d2s):
            slot = rescue_by_depth.setdefault(n + 1, [0, 0])
            slot[1] += 1
            slot[0] += int(d2s[n] == int(rep["tokens"][n]))
        if conf_probe is not None and block:
            conf_probe(list(confs), n)
        stop = False
        for t in committed:
            toks.append(int(t))
            ids.append(int(t))
            if on_token is not None:
                on_token(int(t))
            if int(t) in eos or len(toks) >= max_new:
                stop = True
                break
        cur = ids[-1]
        pos = len(ids) - 1
        block = [int(t) for t in (rep.get("draft") or [])]
        confs = [float(c) for c in (rep.get("conf") or [])]
        d2s = [int(t) for t in (rep.get("d2") or [])]
        if stop:
            break

    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    gen = len(toks)
    return {
        "ok": True,
        "tokens": toks,
        "prompt_tokens": len(prompt_ids),
        "receipts": recs,
        "receipts_ok": receipts_ok,
        "rounds": rounds,
        "drafted": drafted,
        "generated": gen,
        "accepted": accepted_total,
        "g": (gen / rounds) if rounds else float(gen),
        "accept_hist": hist,
        "rescue_by_depth": {d: tuple(v) for d, v in sorted(rescue_by_depth.items())},
        "sent": sent,
        "send_hist": send_hist,
    }


V4_PIPELINED_SPEC = bool(int(os.environ.get("V4_PIPELINED_SPEC", "0") or 0))


V4_SPEC_DEPTH = int(os.environ.get("V4_SPEC_DEPTH", "16") or 16)


V4_LAZY_DRAFT = bool(int(os.environ.get("V4_LAZY_DRAFT", "0") or 0))


V4_REFILL_FLOOR = int(os.environ.get("V4_REFILL_FLOOR", "1") or 1)


class _SpeculationWindow:
    def __init__(self):
        self.lock = threading.Lock()
        self.epoch = 0
        self.outstanding = collections.deque()
        self.pending = 0
        self.unsent = 0


class _OutboundQueue(threading.Thread):
    def __init__(self, pipe, state):
        super().__init__(daemon=True, name="v4-pipe-sender")
        self.pipe = pipe
        self.st = state
        self.q = queue.Queue()
        self.err = None
        self._stopped = False

    def put(self, frame, epoch):
        with self.st.lock:
            self.st.pending += 1
        self.q.put((frame, epoch))

    def stop(self):

        if self._stopped:
            return
        self._stopped = True
        self.q.put(None)
        self.join(timeout=10)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            frame, ep = item
            with self.st.lock:
                if ep != self.st.epoch:
                    self.st.pending -= 1
                    self.st.unsent += 1
                    continue
                self.st.outstanding.append((int(frame["start_pos"]), ep))
            try:
                send_frame(self.pipe, frame)
            except Exception as e:
                self.err = e
                return


def stream_dspark(
    pipe,
    ret,
    prompt_ids,
    max_new,
    *,
    eos_ids=(),
    nonce=None,
    swarm_id="swarm",
    job_id="job",
    layer_count=None,
    receipts=False,
    timeout=600.0,
    on_token=None,
    depth=None,
    lazy=None,
    floor=None,
):

    W = int(depth or V4_SPEC_DEPTH)
    F = int(floor if floor is not None else V4_REFILL_FLOOR)
    if F < 1:
        raise ValueError(
            f"v4 pipelined dspark: refill floor {F} — the pipe refills at or below the "
            f"floor, so it must be at least 1 (1 = the shipped drain-only behaviour)"
        )
    lazy = V4_LAZY_DRAFT if lazy is None else bool(lazy)

    controls.note("V4_LAZY_DRAFT", lazy)
    controls.note("V4_SPEC_DEPTH", W)
    controls.note("V4_PIPELINED_SPEC", True)
    controls.note("V4_REFILL_FLOOR", F)
    ret.settimeout(timeout)
    send_frame(
        pipe,
        {
            "op": "reset",
            "swarm_id": swarm_id,
            "job_id": job_id,
            "nonce": nonce,
            "temp": 0.0,
            "seed": 0,
            "spec": True,
            "dspark": True,
            "pipelined": True,
        },
    )
    ack = receive_frame(ret)
    if not (ack == "ok" or (isinstance(ack, dict) and ack.get("ok"))):
        raise RuntimeError(f"v4 dspark ring reset not acked: {ack!r}")

    eos = set(eos_ids)
    ids = list(prompt_ids)
    toks = []

    send_frame(pipe, {"op": "step", "ids": [ids], "start_pos": 0, "epoch": 0})
    rep = receive_frame(ret)
    if not isinstance(rep, dict) or "acc" not in rep:
        raise RuntimeError(
            "v4 pipelined dspark: the tail's prefill reply carries no `acc`, so nothing is drafting "
            "on it in pipelined mode — launch the tail stage with --dspark (and check it is a build "
            "new enough to honour the reset's `pipelined` flag)"
        )
    cur = int(rep["token"])
    c = len(ids)
    ids.append(cur)
    toks.append(cur)
    if on_token is not None:
        on_token(cur)
    stop = cur in eos or len(toks) >= max_new

    st8 = _SpeculationWindow()
    sender = _OutboundQueue(pipe, st8)
    sender.start()
    sent = {}
    ddepth = {}
    dsrc = {}
    dalt = {}
    horizon = c - 1
    frames = drafted = accepted = cancels = run = stale = issued = 0
    topups = topup_frames = topup_agree = topup_disagree = 0
    hist, depths = {}, []
    by_depth, tu_by_depth = {}, {}
    rescue_by_depth = {}
    prevhint = set()
    blen = 0

    tick = {"t": None, "area": 0.0, "span": 0.0}

    def _mark():

        now = time.monotonic()
        if tick["t"] is not None:
            dt = now - tick["t"]
            tick["area"] += (horizon - c + 1) * dt
            tick["span"] += dt
        tick["t"] = now

    def _feed(pos, tok, nxt=None, prev=False, dep=0, src="block", alt=None):

        nonlocal horizon, frames, topup_frames
        sent[pos] = int(tok)
        ddepth[pos] = int(dep)
        dsrc[pos] = src
        dalt[pos] = None if alt is None else int(alt)
        if src == "topup":
            topup_frames += 1
        _mark()
        horizon = max(horizon, pos)
        frames += 1
        f = {
            "op": "step",
            "ids": [[int(tok)]],
            "start_pos": pos,
            "epoch": st8.epoch,
            "cpos": c - 1,
        }
        if lazy and nxt is not None:
            f["dnxt"] = int(nxt)
        if lazy and prev:
            f["dprev"] = True
            prevhint.add(pos)
        sender.put(f, st8.epoch)

    def _hints(span):

        return [
            span[i + 1] if i <= len(span) - F - 2 else None for i in range(len(span))
        ]

    if not stop:
        _feed(c, cur)
    while True:
        with st8.lock:
            waiting = st8.pending
        if waiting == 0:
            if stop:
                break
            raise RuntimeError(
                f"v4 pipelined dspark: nothing in flight at position {c} with {len(toks)} of "
                f"{max_new} tokens generated — the pipeline emptied without stopping"
            )
        try:
            rep = receive_frame(ret)
        except Exception:
            if sender.err is not None:
                raise RuntimeError(
                    f"v4 pipelined dspark: the frame sender died: "
                    f"{type(sender.err).__name__}: {sender.err}"
                ) from sender.err
            raise
        with st8.lock:
            pos, ep = st8.outstanding.popleft()
            st8.pending -= 1
        if int(rep.get("pos", pos)) != pos or int(rep.get("epoch", ep)) != ep:
            raise RuntimeError(
                f"v4 pipelined dspark: reply {rep.get('pos')}@e{rep.get('epoch')} for the frame "
                f"{pos}@e{ep} — the return channel reordered or dropped a frame, and every accept "
                f"after it would be attributed to the wrong position"
            )
        if rep.get("fenced") or ep != st8.epoch:
            stale += 1
            continue
        if stop:
            continue
        if pos != c:
            raise RuntimeError(
                f"v4 pipelined dspark: reply for position {pos} with the committed "
                f"frontier at {c} — replies must land in committed order"
            )

        if sent.get(pos) != ids[pos]:
            raise RuntimeError(
                f"v4 pipelined dspark: about to judge the reply of the frame at {pos}, which fed "
                f"{sent.get(pos)!r} while the committed token there is {ids[pos]!r} — a fenced frame "
                f"reached the accept path, and every token after it would answer a discarded history"
            )
        if not rep.get("acc"):
            raise RuntimeError(
                f"v4 pipelined dspark: the tail judged the frame at {pos} speculative and this "
                f"coordinator judged it committed, off the same frame and the same replies. The two "
                f"ends of a lossless round have diverged, so the drafter is advancing over a history "
                f"the ring is not taking — one accept rule, and it is plan_verify_round."
            )
        m = int(rep["tokens"][0])
        fed = sent.get(pos + 1)
        fdep = ddepth.get(pos + 1, 0)
        _mark()
        ids.append(m)
        c = pos + 1
        toks.append(m)
        if on_token is not None:
            on_token(m)
        if fdep:
            book = tu_by_depth if dsrc.get(pos + 1) == "topup" else by_depth
            slot = book.setdefault(fdep, [0, 0])
            slot[1] += 1
            if fed == m:
                slot[0] += 1
            elif dalt.get(pos + 1) is not None:
                slot2 = rescue_by_depth.setdefault(fdep, [0, 0])
                slot2[1] += 1
                slot2[0] += int(dalt[pos + 1] == m)
        if fed == m:
            accepted += 1
            run += 1
        stop = m in eos or len(toks) >= max_new
        if stop:
            with st8.lock:
                st8.epoch += 1
            sender.stop()
            continue
        blk = [int(t) for t in (rep.get("draft") or [])]
        alt2 = [int(t) for t in (rep.get("d2") or [])]
        issued += bool(blk)
        blen = max(blen, len(blk))

        need = fed is None or fed != m or horizon - c + 1 <= F
        if lazy and need and "draft" not in rep and pos not in prevhint:
            raise RuntimeError(
                f"v4 pipelined dspark: the reply for the frame at {pos} carries no block and this "
                f"round needs one (frontier {c}, horizon {horizon}) — the tail skipped a draft the "
                f"lazy-draft hint did not license it to skip"
            )

        nblk = min(len(blk), max(W - 1, 0))
        hints = _hints([m] + blk[:nblk])
        if fed is None:
            _feed(c, m, hints[0])
        elif fed != m:
            cancels += 1
            hist[run] = hist.get(run, 0) + 1
            run = 0
            with st8.lock:
                st8.epoch += 1
            for p in [p for p in sent if p > pos]:
                del sent[p]
                ddepth.pop(p, None)
                dsrc.pop(p, None)
                dalt.pop(p, None)
            _mark()
            horizon = pos

            _feed(c, m, hints[0])

        if blk and horizon - c + 1 <= F:
            base = horizon
            topup = base > c
            drafted += 1
            topups += topup
            for i, d in enumerate(blk[:nblk]):
                p = pos + 2 + i
                if p <= base:
                    topup_agree += int(d) == sent[p]
                    topup_disagree += int(d) != sent[p]
                    continue

                _feed(
                    p,
                    d,
                    hints[i + 1],
                    prev=(F == 1 and i == nblk - 1 and nblk >= 2),
                    dep=p - c,
                    src="topup" if topup else "block",
                    alt=alt2[i] if i < len(alt2) else None,
                )

        depths.append(horizon - c + 1)
        sent.pop(pos - 1, None)
        ddepth.pop(pos - 1, None)
        dsrc.pop(pos - 1, None)
        dalt.pop(pos - 1, None)
        prevhint.discard(pos)

    hist[run] = hist.get(run, 0) + 1
    with st8.lock:
        st8.epoch += 1
    sender.stop()
    if sender.err is not None:
        raise RuntimeError(
            f"v4 pipelined dspark: the frame sender died: "
            f"{type(sender.err).__name__}: {sender.err}"
        ) from sender.err
    recs, receipts_ok = [], None
    if receipts:
        recs, receipts_ok = _sweep_receipts(pipe, ret, layer_count, nonce)
    gen = len(toks)
    cycles = cancels + 1
    return {
        "ok": True,
        "tokens": toks,
        "prompt_tokens": len(prompt_ids),
        "receipts": recs,
        "receipts_ok": receipts_ok,
        "frames": frames,
        "drafted": drafted,
        "drafts_issued": issued,
        "lazy": lazy,
        "floor": F,
        "generated": gen,
        "accepted": accepted,
        "cancels": cancels,
        "cycles": cycles,
        "rounds": cycles,
        "g": (gen / cycles) if cycles else float(gen),
        "accept_hist": hist,
        "accept_by_depth": {d: tuple(v) for d, v in sorted(by_depth.items())},
        "topups": topups,
        "topup_frames": topup_frames,
        "topup_accept_by_depth": {d: tuple(v) for d, v in sorted(tu_by_depth.items())},
        "topup_agree": topup_agree,
        "topup_disagree": topup_disagree,
        "rescue_by_depth": {d: tuple(v) for d, v in sorted(rescue_by_depth.items())},
        "max_inflight": max(depths) if depths else 0,
        "mean_inflight": round(sum(depths) / len(depths), 2) if depths else 0.0,
        "inflight_time_avg": round(tick["area"] / tick["span"], 2)
        if tick["span"] > 0
        else 0.0,
        "decode_wall_s": round(tick["span"], 3),
        "frames_per_token": round(frames / gen, 3) if gen else 0.0,
        "block_len": blen,
        "stale_replies": stale,
        "unsent_frames": st8.unsent,
    }


def balanced_ranges(n_layers, nstages):

    if nstages < 1 or nstages > n_layers:
        raise ValueError(f"cannot tile {n_layers} layers over {nstages} stages")
    base, extra = divmod(n_layers, nstages)
    ranges, lo = [], 0
    for k in range(nstages):
        hi = lo + base + (1 if k < extra else 0)
        ranges.append((lo, hi))
        lo = hi
    return ranges


def assign_layer_ranges(nodes, rtt, model_id=V4_MODEL_ID, *, per_gpu=False, **kw):

    try:
        from wanferenz.protocol.capacity import plan_chain, lookup_capacity
    except ImportError:
        from wanferenz.protocol.capacity import plan_chain, lookup_capacity
    try:
        lookup_capacity(model_id)
    except ValueError as e:
        raise RuntimeError(
            f"no engine profile for {model_id!r} in wanferenz/protocol/capacity.py PROFILES ({e}) — register a "
            f"measured V4_PROFILE (layer_vram_mb / load_peak_extra_mb / layer_ms_base / "
            f"decode_bytes = hc_mult*dim*2) before planning a V4 ring; balanced_ranges() is the "
            f"offline fallback"
        ) from e
    capacity = plan_chain(nodes, rtt, model=model_id, **kw)
    if not capacity:
        return None
    if not per_gpu:
        return capacity["stages"]
    return distribute_local_gpus(
        capacity["stages"], {nd["id"]: int(nd.get("gpus", 1)) for nd in nodes}
    )


def relay_routes(k, n, maddrs, ret_maddr=None):

    inbound = f"127.0.0.1:{ENG_IN}" if k > 0 else ""
    forwards = []
    if k < n - 1:
        forwards.append(f"127.0.0.1:{FWD_RING}={maddrs[k + 1]}")
    if k == 0:
        forwards.append(f"127.0.0.1:{FWD_RET}={ret_maddr or maddrs[-1]}")
    allow = None
    if k > 0:
        allow = [_remote_peer(maddrs[k - 1])]
        if k == n - 1:
            allow.append(_remote_peer(maddrs[0]))
    return inbound, forwards, allow


def _remote_peer(maddr):
    return maddr.rsplit("/p2p/", 1)[-1].split("/p2p-circuit")[0]


GRAPH_MODE_VALUES = frozenset({"0", "1", "island", "on", "whole", "2", "eager"})


ENG_ENV = [
    "V4_HEAD_GRAPH",
    "V4_DSPARK_FULL_GRAPH",
    "V4_SPEC_SNAPSHOT",
    "V4_FP8_WIRE",
    "V4_PIPELINED_SPEC",
    "V4_SPEC_DEPTH",
    "V4_LAZY_DRAFT",
    "V4_REFILL_FLOOR",
    "V4_MOE_GROUPED",
    "V4_MOE_DECODE",
    "V4_MOE_MULTI",
    "V4_MOE_MULTI_MAX",
    "V4_FP8_GEMV",
    "V4_FP8_SHARED",
    "V4_GRAPH_MAX",
    "V4_MOE_IN_GRAPH",
    "V4_DSPARK_FAST",
    "V4_DSPARK_GRAPH",
    "V4_DSPARK_MOE",
    "V4_DSPARK_BLOCK",
    "V4_DRAFT_TOP2",
    "V4_DSPARK_CONF_GATE",
    "V4_DSPARK_CONF_MIN",
    "V4_DSPARK_CONF_THRESH",
    "V4_REF_SLIM",
    "V4_REF_SLIM_NOQAT",
    "V4_FAST_VERIFY",
    "V4_FAST_VERIFY_MAX",
    "V4_KERNELS",
    "V4_DTYPE",
    "V4_MAX_SEQ",
    "V4_MAX_BATCH",
    "V4_KEEPWARM",
    "V4_KEEPWARM_MS",
    "V4_DIAL_CONNECT_TIMEOUT",
    "V4_DIAL_RETRY_S",
    "V4_TIMING",
    "V4_TIMING_EVERY",
    "V4_LEVERS_STRICT",
]


def _eng_env():
    return "".join(f"{k}={os.environ[k]} " for k in ENG_ENV if k in os.environ)


def _graph_env_conflict(gp_mode):

    v = os.environ.get("V4_CUDA_GRAPH")
    if v is not None and v != gp_mode:
        raise ValueError(
            f"V4_CUDA_GRAPH={v!r} is exported on the launcher but this launch would send "
            f"V4_CUDA_GRAPH={gp_mode!r} to the stages. Refusing to pick for you: pass "
            f"cuda_graph={v!r} to mean it, or unset the export to use the launch default. "
            f"(Unlike the ENG_ENV levers this one is always emitted, so it is never dropped — "
            f"only ever ambiguous.)"
        )


def partition_command(
    stage,
    nstages,
    lo,
    hi,
    *,
    model_dir="/root/v4",
    receipts=False,
    device="cuda",
    token=None,
    extra_env="",
    gpu=None,
    port=None,
    nxt_addr=None,
    ret_relay=None,
    dspark=False,
    cuda_graph=True,
):

    port = port or ENG_IN
    if nxt_addr is not None:
        nxt = f"--next {nxt_addr}"
    else:
        nxt = "" if stage == nstages - 1 else f"--next 127.0.0.1:{FWD_RING}"
    rr = f"--ret-relay {ret_relay} " if ret_relay else ""
    ds = "--dspark " if dspark else ""
    rc = "WANFERENZ_RECEIPTS=1 " if receipts else ""
    tk = f"WANFERENZ_SWARM_TOKEN={token} " if token else ""
    cvd = f"CUDA_VISIBLE_DEVICES={int(gpu)} " if gpu is not None else ""

    gp_mode = (
        "1" if cuda_graph is True else "0" if cuda_graph is False else str(cuda_graph)
    )
    if gp_mode not in GRAPH_MODE_VALUES:
        raise ValueError(
            f"partition_command: V4_CUDA_GRAPH={gp_mode!r} is not a mode the partition runtime "
            f"recognises, so the stage would resolve it to OFF. Use one of "
            f"{sorted(GRAPH_MODE_VALUES)} — a typo here launches a ring that measures "
            f"eager and reports graphed."
        )
    _graph_env_conflict(gp_mode)
    gp = f"V4_CUDA_GRAPH={gp_mode} "

    log = f"/root/v4_stage_{port}.log"
    inner = (
        f"python3 -m wanferenz.serving.chain stage --stage {stage} --nstages {nstages} --lo {lo} --hi {hi} "
        f"--port {port} {nxt} {rr}{ds}--dir {model_dir} > {log} 2>&1"
    )
    return (
        f"{rc}{tk}{cvd}{gp}{_eng_env()}{extra_env}V4_DIR={model_dir} V4_DEV={device} "
        f"WANFERENZ_ENGINE_BIND=127.0.0.1 setsid bash -c '{inner}' </dev/null >/dev/null 2>&1 &"
    )


def local_eng_port(local_index):

    return ENG_IN if local_index == 0 else ENG_LOCAL_BASE + local_index


def distribute_local_gpus(node_stages, gpus):

    def _g(k, nd):
        if isinstance(gpus, dict):
            return int(gpus.get(nd["id"], 1))
        if isinstance(gpus, (list, tuple)):
            return int(gpus[k])
        return int(gpus)

    subs = []
    for k, nd in enumerate(node_stages):
        G = _g(k, nd)
        if G < 1:
            raise ValueError(f"box {nd.get('id', k)!r} announced {G} GPUs")
        for j, (slo, shi) in enumerate(balanced_ranges(nd["hi"] - nd["lo"], G)):
            subs.append(
                {
                    "id": nd["id"],
                    "box_index": k,
                    "gpu": j,
                    "local_index": j,
                    "nlocal": G,
                    "lo": nd["lo"] + slo,
                    "hi": nd["lo"] + shi,
                    "layers": shi - slo,
                    "box_head": j == 0,
                    "box_tail": j == G - 1,
                }
            )
    n = len(subs)
    for g, s in enumerate(subs):
        s["global_index"], s["nstages"] = g, n
        s["head"], s["tail"] = g == 0, g == n - 1
    return subs


def box_stage_wiring(sub):

    eng_port = local_eng_port(sub["local_index"])
    if sub["tail"]:
        return eng_port, None, "tail"
    if sub["box_tail"]:
        return eng_port, f"127.0.0.1:{FWD_RING}", "wan"
    return eng_port, f"127.0.0.1:{local_eng_port(sub['local_index'] + 1)}", "loopback"


def box_return_relay(sub, tail_box_index):

    if sub["box_head"] and sub["nlocal"] > 1 and sub["box_index"] == tail_box_index:
        return f"127.0.0.1:{local_eng_port(sub['nlocal'] - 1)}"
    return None


def box_ring_launch(
    node_stages,
    gpus,
    box_maddrs=None,
    *,
    model_dir="/root/v4",
    receipts=False,
    token=None,
    ret_maddr=None,
    dspark=False,
):

    subs = distribute_local_gpus(node_stages, gpus)
    tail_box = subs[-1]["box_index"]
    stages = []
    for sub in subs:
        eng_port, nxt, link = box_stage_wiring(sub)
        ret_relay = box_return_relay(sub, tail_box)
        cmd = partition_command(
            sub["global_index"],
            sub["nstages"],
            sub["lo"],
            sub["hi"],
            model_dir=model_dir,
            receipts=receipts,
            token=token,
            gpu=sub["gpu"],
            port=eng_port,
            nxt_addr=nxt,
            ret_relay=ret_relay,
            dspark=(dspark and sub["tail"]),
        )
        stages.append(
            {
                **sub,
                "eng_port": eng_port,
                "next": nxt,
                "link": link,
                "ret_relay": ret_relay,
                "cmd": cmd,
            }
        )
    B = len(node_stages)
    sidecars = (
        [relay_routes(b, B, box_maddrs, ret_maddr=ret_maddr) for b in range(B)]
        if box_maddrs is not None
        else None
    )
    return {"stages": stages, "sidecars": sidecars}


def _save_test_checkpoint(d, args, model):

    import dataclasses
    import safetensors.torch as ST

    os.makedirs(d, exist_ok=True)
    tensors = {
        k: v.detach().clone().contiguous() for k, v in model.state_dict().items()
    }
    ST.save_file(tensors, os.path.join(d, "model0-mp1.safetensors"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(args), f)
    return d


def _oracle_tokens(model, prompt, max_new):

    ids = list(prompt)
    with torch.inference_mode():
        out, _, _ = model(torch.tensor([ids]))
        tid = int(out.reshape(-1)[-1])
        toks = [tid]
        pos = len(ids)
        while len(toks) < max_new:
            out, _, _ = model(torch.tensor([[tid]]), pos)
            pos += 1
            tid = int(out.reshape(-1)[-1])
            toks.append(tid)
    return toks


def _reserve_test_ports(n):

    socks = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(n)]
    ports = []
    for s in socks:
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def _expected_cover(ranges):
    return sorted((lo, hi) for lo, hi in ranges)


def _start_test_chain(d, ranges, tail_box_g=1, dspark=False, tag="s"):

    n = len(ranges)
    ports = _reserve_test_ports(n)
    relay_i = n - tail_box_g
    events = [threading.Event() for _ in range(n)]
    for i, (lo, hi) in enumerate(ranges):
        nxt = None if i == n - 1 else f"127.0.0.1:{ports[i + 1]}"
        ret_relay = (
            f"127.0.0.1:{ports[-1]}" if (tail_box_g > 1 and i == relay_i) else None
        )
        threading.Thread(
            target=run_partition,
            kwargs=dict(
                stage=i,
                nstages=n,
                lo=lo,
                hi=hi,
                port=ports[i],
                nxt=nxt,
                ckpt_dir=d,
                device="cpu",
                receipts=True,
                key_path=f"{d}/{tag}{i}.key",
                ret_relay=ret_relay,
                dspark=dspark,
                ready=events[i],
            ),
            daemon=True,
        ).start()
    for e in events:
        e.wait(120)
    tail_port = ports[relay_i] if tail_box_g > 1 else ports[-1]
    return connect_chain(f"127.0.0.1:{ports[0]}", f"127.0.0.1:{tail_port}", timeout=120)


def _draft_layer_ranges(args, nstages):

    lo = min(args.dspark_target_layer_ids)
    if nstages != 3 or lo < 2:
        raise ValueError(
            f"the drafted selftest ring wants 3 stages and targets from layer 2 up "
            f"(got {nstages} stages, first target {lo})"
        )
    return [(0, lo // 2), (lo // 2, lo), (lo, args.n_layers)]


def selftest(nstages=3, prompt=(168, 15, 493, 72, 22), max_new=6, tail_box_g=1):

    import tempfile
    import wanferenz.model.oracle as R

    os.environ["WANFERENZ_RECEIPTS"] = "1"
    global RECEIPTS
    RECEIPTS = True

    args = R.miniature_parameters()
    n_layers = args.n_layers
    d = tempfile.mkdtemp(prefix="v4pipe_")
    model = R.create_oracle(args)
    _save_test_checkpoint(d, args, model)
    os.environ["V4_DIR"] = d

    ref_tokens = _oracle_tokens(model, list(prompt), max_new)
    ranges = balanced_ranges(n_layers, nstages)
    pipe, ret = _start_test_chain(d, ranges, tail_box_g)
    r = decode_greedy(
        pipe,
        ret,
        list(prompt),
        max_new,
        nonce="settle-nonce-0",
        receipts=True,
        layer_count=n_layers,
        timeout=120,
    )
    s = decode_proposals(
        pipe,
        ret,
        list(prompt),
        max_new,
        nonce="spec-nonce-0",
        receipts=True,
        layer_count=n_layers,
        timeout=120,
        K=4,
        drafter=_LastTokenProposal(),
    )

    def perfect(seq, K):

        nxt = ref_tokens[len(seq) - len(prompt) :][:K]
        return list(nxt) + [0] * (K - len(nxt))

    p = decode_proposals(
        pipe,
        ret,
        list(prompt),
        max_new,
        nonce="spec-nonce-1",
        receipts=True,
        layer_count=n_layers,
        timeout=120,
        K=2,
        drafter=perfect,
    )
    send_frame(pipe, {"op": "stop"})

    d_ranges = _draft_layer_ranges(args, nstages)
    d_pipe, d_ret = _start_test_chain(d, d_ranges, tail_box_g, dspark=True, tag="d")
    k = decode_dspark(
        d_pipe,
        d_ret,
        list(prompt),
        max_new,
        nonce="dspark-nonce-0",
        receipts=True,
        layer_count=n_layers,
        timeout=120,
    )
    q = stream_dspark(
        d_pipe,
        d_ret,
        list(prompt),
        max_new,
        nonce="dspark-nonce-1",
        receipts=True,
        layer_count=n_layers,
        timeout=120,
    )

    y = {
        w: stream_dspark(
            d_pipe,
            d_ret,
            list(prompt),
            max_new,
            nonce=f"eager-{w}",
            receipts=True,
            layer_count=n_layers,
            timeout=120,
            depth=w,
        )
        for w in (2, 3, 16)
    }
    z = {
        w: stream_dspark(
            d_pipe,
            d_ret,
            list(prompt),
            max_new,
            nonce=f"lazy-{w}",
            receipts=True,
            layer_count=n_layers,
            timeout=120,
            depth=w,
            lazy=True,
        )
        for w in (2, 3, 16)
    }

    fl = {
        f: stream_dspark(
            d_pipe,
            d_ret,
            list(prompt),
            max_new,
            nonce=f"floor-{f}",
            receipts=True,
            layer_count=n_layers,
            timeout=120,
            floor=f,
        )
        for f in (2, 3, 5)
    }
    send_frame(d_pipe, {"op": "stop"})

    def cover(res, rs):
        return sorted(
            (c["layer_start"], c["layer_end"]) for c in res["receipts"]
        ) == _expected_cover(rs)

    checks = {
        "reference_stream_is_a_fingerprint": (
            len(set(ref_tokens)) >= max(4, max_new - 1)
        ),
        "full_ring_matches_reference": (r["tokens"] == ref_tokens),
        "receipts_settle": (r["receipts_ok"] is True),
        "coverage_tiles_all_layers": cover(r, ranges),
        "spec_lossless_vs_greedy": (s["tokens"] == ref_tokens),
        "spec_receipts_settle": (s["receipts_ok"] is True and cover(s, ranges)),
        "spec_full_accept_commits_a_block": (
            p["tokens"] == ref_tokens and p["g"] > 1.0 and max(p["accept_hist"]) == 2
        ),
        "dspark_lossless_vs_greedy": (k["tokens"] == ref_tokens),
        "dspark_receipts_settle": (k["receipts_ok"] is True and cover(k, d_ranges)),
        "dspark_drafted_real_blocks": (
            k["rounds"] > 1 and k["drafted"] == k["rounds"] - 1
        ),
        "pipelined_lossless_vs_greedy": (q["tokens"] == ref_tokens),
        "pipelined_matches_the_serial_dspark_path": (q["tokens"] == k["tokens"]),
        "pipelined_receipts_settle": (q["receipts_ok"] is True and cover(q, d_ranges)),
        "pipelined_actually_pipelined": (q["max_inflight"] > 1),
        "lazy_lossless_vs_greedy_at_every_depth": all(
            v["tokens"] == ref_tokens for v in z.values()
        ),
        "lazy_runs_the_same_round_as_eager": all(
            z[w]["drafted"] == y[w]["drafted"]
            and z[w]["cancels"] == y[w]["cancels"]
            and z[w]["frames"] == y[w]["frames"]
            and z[w]["max_inflight"] == y[w]["max_inflight"]
            for w in z
        ),
        "lazy_never_drafts_more_than_eager": all(
            z[w]["drafts_issued"] <= y[w]["drafts_issued"] for w in z
        ),
        "lazy_receipts_settle": all(
            v["receipts_ok"] is True and cover(v, d_ranges) for v in z.values()
        ),
        "floor_lossless_at_every_setting": all(
            v["tokens"] == ref_tokens for v in fl.values()
        ),
        "floor_at_zero_accept_is_the_shipped_round": all(
            v["frames"] == q["frames"]
            and v["cancels"] == q["cancels"]
            and v["max_inflight"] == q["max_inflight"]
            and v["stale_replies"] + v["unsent_frames"]
            == q["stale_replies"] + q["unsent_frames"]
            and v["topups"] == 0
            and v["topup_frames"] == 0
            for v in fl.values()
        ),
        "floor_receipts_settle": all(
            v["receipts_ok"] is True and cover(v, d_ranges) for v in fl.values()
        ),
        "accept_by_depth_scores_the_misses": all(
            v["accept_by_depth"]
            and all(h == 0 for h, _ in v["accept_by_depth"].values())
            and v["topup_accept_by_depth"] == {}
            for v in fl.values()
        ),
    }
    tag = f", tail box = {tail_box_g} GPUs (return relay)" if tail_box_g > 1 else ""
    print("\n=== V4 pipe offline selftest (CPU tiny config) ===")
    print(f"  ring:   {nstages} stages over {n_layers} layers {ranges}{tag}")
    print(
        f"  dspark: {d_ranges} — the tail owns targets {tuple(args.dspark_target_layer_ids)}, "
        f"block={args.dspark_block_size}"
    )
    print(f"  tokens (ring)   {r['tokens']}\n  tokens (spec)   {s['tokens']}")
    print(f"  tokens (dspark) {k['tokens']}\n  tokens (ref)    {ref_tokens}")
    print(
        f"  spec   rounds={s['rounds']} accepted={s['accepted']} g={s['g']:.2f} "
        f"hist={s['accept_hist']}  (repeat-last drafter: rejects, so every round rewinds)"
    )
    print(
        f"  spec   rounds={p['rounds']} accepted={p['accepted']} g={p['g']:.2f} "
        f"hist={p['accept_hist']}  (perfect drafter: full accepts, no rewind)"
    )
    print(
        f"  dspark rounds={k['rounds']} drafted={k['drafted']} accepted={k['accepted']} "
        f"g={k['g']:.2f} hist={k['accept_hist']}"
    )
    print(f"  tokens (pipe)   {q['tokens']}")
    print(
        f"  pipe   frames={q['frames']} cycles={q['cycles']} accepted={q['accepted']} "
        f"cancels={q['cancels']} g={q['g']:.2f} inflight max={q['max_inflight']} "
        f"mean={q['mean_inflight']} stale={q['stale_replies']} unsent={q['unsent_frames']}"
    )
    print(
        "  drafts issued/consumed  "
        + "   ".join(
            f"W={w} eager {y[w]['drafts_issued']}/{y[w]['drafted']} "
            f"lazy {z[w]['drafts_issued']}/{z[w]['drafted']}"
            for w in z
        )
        + "   (zero accept: every block is consumed, so there is nothing to skip)"
    )
    print(
        "  floor  "
        + "   ".join(
            f"F={f} frames={v['frames']} cancels={v['cancels']} topups={v['topups']} "
            f"by_depth={v['accept_by_depth']}"
            for f, v in fl.items()
        )
    )
    for name, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    ok = all(checks.values())
    print(f"\n  {'ALL PASS' if ok else 'FAILURES PRESENT'}", flush=True)
    os._exit(0 if ok else 1)


def _publish_event(tag, **fields):
    sys.stdout.write(tag + " " + json.dumps(fields) + "\n")
    sys.stdout.flush()


from wanferenz.model.assets import reference_directory


def _tokenize_job(tok, job):

    if not job.get("messages"):
        return job["promptIds"]
    enc_dir = reference_directory()
    if enc_dir not in sys.path:
        sys.path.insert(0, enc_dir)
    from wanferenz.model.chat import encode_messages

    text = encode_messages(
        job["messages"],
        "thinking" if job.get("thinking") else "chat",
        reasoning_effort=job.get("reasoningEffort"),
    )
    return tok.encode(text, add_special_tokens=False)


def _serve_job_stream(a):

    os.environ.setdefault("V4_DIR", a.dir)
    layer_count = checkpoint_parameters(a.dir).n_layers
    try:
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast.from_pretrained(a.dir, fix_mistral_regex=True)
    except Exception as e:
        _publish_event("WANFERENZ_JOB_FATAL", error=f"tokenizer load failed: {e}")
        return 1
    eos = tok.eos_token_id
    eos_ids = (
        tuple(eos)
        if isinstance(eos, (list, tuple))
        else ((eos,) if eos is not None else ())
    )
    pipe, ret = connect_chain(
        a.head, a.tail, timeout=a.timeout, token=SWARM_TOKEN, retry_s=a.connect_retry
    )

    controls.report(side=controls.COORDINATOR)
    _publish_event(
        "WANFERENZ_COORD_READY", head=a.head, tail=a.tail, receipts=a.receipts
    )
    audited = False

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
            job_id = job["jobId"]
        except (ValueError, KeyError) as e:
            _publish_event("WANFERENZ_JOB_FATAL", error=f"unparseable job line: {e}")
            continue
        max_new = max(1, min(int(job.get("maxNew") or 512), 4096))
        _publish_event("WANFERENZ_JOB_START", jobId=job_id, maxNew=job.get("maxNew"))
        state = {"n": 0, "t0": None, "tft": None}

        def _on_token(_tid, _job=job_id, _st=state):
            _st["n"] += 1
            if _st["tft"] is None:
                _st["tft"] = time.time() - _st["t0"]
            _publish_event("WANFERENZ_JOB_TOKEN", jobId=_job, delta=_st["n"])

        try:
            prompt_ids = _tokenize_job(tok, job)
            state["t0"] = time.time()
            if job.get("dspark"):
                pipelined = bool(V4_PIPELINED_SPEC or job.get("pipelined"))
                wants_conf = (
                    job.get("confGate")
                    if job.get("confGate") is not None
                    else V4_DSPARK_CONF_GATE
                )
                if pipelined and wants_conf:
                    raise ValueError(
                        "confGate/V4_DSPARK_CONF_GATE gates the serial DSpark path's block length; "
                        "the pipelined coordinator streams s=1 frames and has no block to trim. "
                        "Run one or the other, not both."
                    )
                if pipelined:
                    r = stream_dspark(
                        pipe,
                        ret,
                        prompt_ids,
                        max_new,
                        eos_ids=eos_ids,
                        nonce=job.get("nonce"),
                        swarm_id=job.get("swarmId") or "swarm",
                        job_id=job_id,
                        layer_count=layer_count,
                        receipts=a.receipts,
                        timeout=a.timeout,
                        on_token=_on_token,
                    )
                else:
                    r = decode_dspark(
                        pipe,
                        ret,
                        prompt_ids,
                        max_new,
                        eos_ids=eos_ids,
                        nonce=job.get("nonce"),
                        swarm_id=job.get("swarmId") or "swarm",
                        job_id=job_id,
                        layer_count=layer_count,
                        receipts=a.receipts,
                        timeout=a.timeout,
                        on_token=_on_token,
                        conf_gate=job.get("confGate"),
                        conf_thresh=job.get("confThresh"),
                        conf_min=job.get("confMin"),
                    )
            elif job.get("spec"):
                r = decode_proposals(
                    pipe,
                    ret,
                    prompt_ids,
                    max_new,
                    eos_ids=eos_ids,
                    nonce=job.get("nonce"),
                    swarm_id=job.get("swarmId") or "swarm",
                    job_id=job_id,
                    layer_count=layer_count,
                    receipts=a.receipts,
                    timeout=a.timeout,
                    on_token=_on_token,
                    K=int(job.get("specK", 4)),
                    ng=int(job.get("specNg", 3)),
                )
            else:
                r = decode_greedy(
                    pipe,
                    ret,
                    prompt_ids,
                    max_new,
                    eos_ids=eos_ids,
                    nonce=job.get("nonce"),
                    swarm_id=job.get("swarmId") or "swarm",
                    job_id=job_id,
                    layer_count=layer_count,
                    receipts=a.receipts,
                    temp=float(job.get("temperature", 0.0)),
                    timeout=a.timeout,
                    on_token=_on_token,
                )
            elapsed = time.time() - state["t0"]
            ngen = len(r["tokens"])
            _publish_event(
                "WANFERENZ_JOB_DONE",
                jobId=job_id,
                ok=True,
                response=tok.decode(r["tokens"], skip_special_tokens=True),
                tokensGenerated=ngen,
                tokPerSec=round(ngen / elapsed, 3) if elapsed > 0 else None,
                firstTokenMs=round((state["tft"] or 0) * 1000, 1),
                elapsedS=round(elapsed, 2),
                spec=bool(job.get("spec") or job.get("dspark")),
                dspark=bool(job.get("dspark")),
                g=r.get("g"),
                rounds=r.get("rounds"),
                acceptHist=r.get("accept_hist"),
                receipts=[receipt_body(rr) for rr in (r["receipts"] or [])],
                receiptsOk=r["receipts_ok"],
                nonce=job.get("nonce"),
            )
        except Exception as e:
            _publish_event(
                "WANFERENZ_JOB_FATAL", jobId=job_id, error=f"{type(e).__name__}: {e}"
            )
            continue
        if not audited:
            audited = True
            controls.report(side=controls.COORDINATOR)
    return 0


def run_command():
    from wanferenz.serving.command import execute

    return execute(sys.modules[__name__])


if __name__ == "__main__":
    run_command()
