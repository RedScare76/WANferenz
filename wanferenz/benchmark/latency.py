import json
import re
import socket
import statistics
import sys
import time
import urllib.request


def api(base, path, payload=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def measure_echo(address, count=12):
    host, port = address.rsplit(":", 1)
    times = []
    with socket.create_connection((host, int(port)), timeout=10) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for position in range(count + 2):
            payload = b"v4-latency-probe!"
            start = time.perf_counter()
            sock.sendall(payload)
            reply = b""
            while len(reply) < len(payload):
                chunk = sock.recv(len(payload) - len(reply))
                if not chunk:
                    raise ConnectionError("Echo peer closed before replying")
                reply += chunk
            if reply != payload:
                raise ValueError("Echo payload mismatch")
            if position >= 2:
                times.append((time.perf_counter() - start) * 1000)
    return {
        "samplesMs": times,
        "medianMs": statistics.median(times),
        "avgMs": statistics.mean(times),
        "minMs": min(times),
        "maxMs": max(times),
    }


def configure_delay(capacity, total_ms):
    up = int(total_ms) // 2
    down = int(total_ms) - up
    for name in capacity["proxyNames"]:
        for stream, delay in [("upstream", up), ("downstream", down)]:
            api(
                capacity["api"],
                f"/proxies/{name}/toxics/{stream}",
                {"attributes": {"latency": delay, "jitter": 0}},
            )
    return {"upstreamMs": up, "downstreamMs": down}


def calibrate(capacity, target):
    configure_delay(capacity, 0)
    baseline = measure_echo(capacity["echoAddress"])
    total = max(0, round(target - baseline["medianMs"]))
    trials = []
    for _ in range(5):
        delays = configure_delay(capacity, total)
        observed = measure_echo(capacity["echoAddress"])
        trials.append({"delay": delays, "observed": observed})
        error = target - observed["medianMs"]
        if abs(error) <= capacity.get("toleranceMs", 1.5):
            return {
                "targetMs": target,
                "baseline": baseline,
                "trials": trials,
                **delays,
                "observed": observed,
            }
        total = max(0, total + round(error))
    raise ValueError(f"Could not calibrate {target} ms RTT: {trials}")


class LatencyMismatch(ValueError):
    def __init__(self, attestation):
        self.attestation = attestation
        super().__init__(
            f"Observed proxy RTT {attestation['observed']['medianMs']:.3f} differs from target {attestation['targetMs']}"
        )


def snapshot(capacity, target):
    observed = measure_echo(capacity["echoAddress"])
    attestation = {
        "targetMs": target,
        "observed": observed,
        "proxies": [
            api(capacity["api"], "/proxies/" + name)
            for name in capacity.get("proxyNames", [])
        ],
    }
    if abs(observed["medianMs"] - target) > capacity.get("toleranceMs", 1.5):
        raise LatencyMismatch(attestation)
    return attestation


def prepare_latency_sample(capacity, target, emit):
    for attempt in range(5):
        try:
            return snapshot(capacity, target)
        except LatencyMismatch as exc:
            emit(
                "WANFERENZ_RTT_DRIFT",
                phase="before",
                attempt=attempt + 1,
                **exc.attestation,
            )
            emit("WANFERENZ_RTT_CALIBRATED", **calibrate(capacity, target))
    return snapshot(capacity, target)


def attach_latency_checks(p, capacity):
    emit = p._publish_event
    current = None
    accepted = None
    source = sys.stdin

    def jobs():
        nonlocal accepted
        for line in source:
            if not line.strip():
                continue
            job = json.loads(line)
            for attempt in range(1, 6):
                accepted = None
                emit("WANFERENZ_RTT_ATTEMPT", jobId=job["jobId"], attempt=attempt)
                yield line
                if accepted is True:
                    break
                if accepted is None:
                    raise RuntimeError(
                        f"Job {job['jobId']} failed without a timing receipt"
                    )
            else:
                raise RuntimeError(
                    f"RTT drift persisted through five attempts of {job['jobId']}"
                )

    sys.stdin = jobs()

    def wrapped(tag, **fields):
        nonlocal current, accepted
        if tag == "WANFERENZ_JOB_START":
            target = int(re.search("-r(\\d+)-", fields["jobId"])[1])
            if target != current:
                attestation = calibrate(capacity, target)
                current = target
                emit("WANFERENZ_RTT_CALIBRATED", **attestation)
            emit(
                "WANFERENZ_RTT_BEFORE",
                jobId=fields["jobId"],
                **prepare_latency_sample(capacity, target, emit),
            )
        if tag == "WANFERENZ_JOB_DONE":
            try:
                after = snapshot(capacity, current)
            except LatencyMismatch as exc:
                accepted = False
                emit(
                    "WANFERENZ_JOB_REJECTED_RTT", **fields, networkAfter=exc.attestation
                )
                return
            accepted = True
            emit(tag, **fields)
            emit("WANFERENZ_RTT_AFTER", jobId=fields["jobId"], **after)
        else:
            emit(tag, **fields)

    p._publish_event = wrapped
