import pytest

from wanferenz.benchmark import latency as tox
from wanferenz.benchmark.workload import compose_workload
from wanferenz.benchmark.cluster import latency_jobs


def test_sweep_runs_all_points_with_full_warmups_and_four_passes():
    w = compose_workload(
        {"renderedPrompt": "x", "promptIds": [1, 2], "tokenizerSha256": "x"}
    )
    jobs = latency_jobs({"spec": {}}, w, list(range(5, 101, 5)))
    assert len(jobs) == 200
    assert jobs[0]["jobId"] == "greedy-r005-warmup"
    assert jobs[-1]["jobId"] == "greedy-r100-4"
    assert all(j["maxNew"] == 128 for j in jobs)
    assert len({j["jobId"] for j in jobs}) == 200


def test_delay_is_split_across_directions_not_doubled(monkeypatch):
    calls = []
    monkeypatch.setattr(tox, "api", lambda *args: calls.append(args))
    capacity = {
        "api": "http://local",
        "proxyNames": ["forward", "return", "echo", "rpc"],
    }
    assert tox.configure_delay(capacity, 9) == {"upstreamMs": 4, "downstreamMs": 5}
    assert len(calls) == 8
    assert [c[2]["attributes"]["latency"] for c in calls] == [4, 5] * 4


def test_calibration_accounts_for_baseline_and_checks_observed_rtt(monkeypatch):
    delays = []
    monkeypatch.setattr(
        tox,
        "configure_delay",
        lambda p, d: (
            delays.append(d) or {"upstreamMs": d // 2, "downstreamMs": d - d // 2}
        ),
    )
    observed = iter([{"medianMs": 1}, {"medianMs": 10.4}])
    monkeypatch.setattr(tox, "measure_echo", lambda a: next(observed))
    r = tox.calibrate({"echoAddress": "unused"}, 10)
    assert delays == [0, 9] and r["observed"]["medianMs"] == 10.4


def test_drift_is_rejected(monkeypatch):
    monkeypatch.setattr(tox, "measure_echo", lambda a: {"medianMs": 40})
    with pytest.raises(ValueError, match="differs from target"):
        tox.snapshot({"echoAddress": "unused"}, 80)


def test_hook_retries_drifted_job_and_preserves_rejected_receipt(monkeypatch):
    import io
    import json
    import sys
    from types import SimpleNamespace

    emitted = []
    p = SimpleNamespace(
        _publish_event=lambda tag, **fields: emitted.append((tag, fields))
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"jobId":"greedy-r005-1"}\n'))
    monkeypatch.setattr(tox, "calibrate", lambda *args: {"targetMs": 5})
    good = {"targetMs": 5, "observed": {"medianMs": 5}, "proxies": []}
    bad = {"targetMs": 5, "observed": {"medianMs": 8}, "proxies": []}
    sequence = iter([good, tox.LatencyMismatch(bad), good, good])

    def probe(*args):
        value = next(sequence)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(tox, "snapshot", probe)
    tox.attach_latency_checks(p, {})
    jobs = []
    for line in sys.stdin:
        job = json.loads(line)
        jobs.append(job)
        p._publish_event("WANFERENZ_JOB_START", **job)
        p._publish_event("WANFERENZ_JOB_DONE", **job, timings={"raw": "preserved"})
    assert len(jobs) == 2 and jobs[0] == jobs[1]
    rejected = [f for t, f in emitted if t == "WANFERENZ_JOB_REJECTED_RTT"]
    assert len(rejected) == 1 and rejected[0]["networkAfter"] == bad
    assert rejected[0]["timings"] == {"raw": "preserved"}
    assert sum(t == "WANFERENZ_JOB_DONE" for t, f in emitted) == 1


def test_before_job_drift_recalibrates_and_records_original_sample(monkeypatch):
    good = {"targetMs": 5, "observed": {"medianMs": 5}}
    bad = {"targetMs": 5, "observed": {"medianMs": 8}}
    samples = iter([tox.LatencyMismatch(bad), good])

    def probe(*args):
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value

    events = []
    calibrated = []
    monkeypatch.setattr(tox, "snapshot", probe)
    monkeypatch.setattr(
        tox, "calibrate", lambda p, t: calibrated.append(t) or {"targetMs": t}
    )
    assert (
        tox.prepare_latency_sample({}, 5, lambda tag, **f: events.append((tag, f)))
        == good
    )
    assert calibrated == [5]
    assert events[0][1]["observed"]["medianMs"] == 8
