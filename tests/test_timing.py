import json
import types

import pytest

from wanferenz.benchmark import timing as metrics
from wanferenz.benchmark.cluster import summarize


def measurement(times=(10, 12, 15, 15.1, 15.1, 16, 20)):
    clock = iter(times)
    meter = metrics.GenerationClock(clock=lambda: next(clock))
    meter.sent({"op": "reset"})
    meter.sent({"op": "step", "start_pos": 0})
    for _ in range(4):
        meter.token()
    return meter.finish(30, 4)


def test_separates_prefill_decode_and_drain_and_preserves_bursts():
    r = measurement()
    assert r["prefillMs"] == 3000
    assert r["promptProcessingTokensPerSec"] == r["prefillTokensPerSec"] == 10
    assert r["timeToFirstTokenMs"] == 5000
    assert r["beforePrefillMs"] == 2000
    assert r["decodeTokensPerSec"] == 3
    assert r["decodeMs"] == 1000
    assert r["postLastTokenMs"] == 4000
    assert r["tokenTimesMs"] == pytest.approx([5000, 5100, 5100, 6000])
    assert r["interTokenLatencyMs"] == pytest.approx([100, 0, 900])
    assert r["interTokenLatencySummaryMs"] == pytest.approx(
        {"mean": 1000 / 3, "median": 100, "min": 0, "max": 900, "p95": 820, "p99": 884}
    )


def test_single_token_has_prefill_but_no_decode_intervals():
    clock = iter([0, 1, 2, 3])
    meter = metrics.GenerationClock(clock=lambda: next(clock))
    meter.sent({"op": "step", "start_pos": 0})
    meter.token()
    r = meter.finish(26, 1)
    assert r["prefillTokensPerSec"] == 26
    assert r["decodeTokensPerSec"] is None
    assert r["interTokenLatencyMs"] == []
    assert all(v is None for v in r["interTokenLatencySummaryMs"].values())


def test_missing_token_callbacks_reject_timing_receipt():
    meter = metrics.GenerationClock()
    meter.sent({"op": "step", "start_pos": 0})
    meter.token()
    with pytest.raises(ValueError, match="Incomplete"):
        meter.finish(26, 2)


def test_summary_excludes_warmups_and_keeps_historical_metrics_unavailable():
    rows = [
        {
            "jobId": name,
            "ok": True,
            "tokensGenerated": 4,
            "tokenIds": [1, 2, 3, 4],
            "tokPerSec": 1,
            "firstTokenMs": 5000,
            "timings": measurement(times),
        }
        for name, times in [
            ("greedy-warmup", (0, 1, 11, 12, 13, 14, 15)),
            ("greedy-1", (10, 12, 15, 15.1, 15.1, 16, 20)),
        ]
    ]
    log = "\n".join("WANFERENZ_JOB_DONE " + json.dumps(r) for r in rows)
    result = summarize(log, rows, require_timings=True)
    assert result["ok"]
    assert result["arms"]["greedy"]["timings"]["medianDecodeTokensPerSec"] == 3
    assert result["arms"]["greedy"]["timings"]["medianPrefillTokensPerSec"] == 10
    for row in rows:
        del row["timings"]
    log = "\n".join("WANFERENZ_JOB_DONE " + json.dumps(r) for r in rows)
    assert summarize(log, rows)["arms"]["greedy"]["timings"] is None
    assert not summarize(log, rows, require_timings=True)["ok"]


def test_wrapper_preserves_tokens_restores_send_and_resets_timing_between_jobs(
    monkeypatch,
):
    clock = iter([0, 1, 3, 4, 8, 10, 11, 13, 14, 18, 20])
    cls = metrics.GenerationClock
    monkeypatch.setattr(
        metrics, "GenerationClock", lambda: cls(clock=lambda: next(clock))
    )
    emitted, tokens, messages = [], [], []

    def send(sock, message):
        messages.append(message)

    def decode_greedy(*args, on_token, job_id):
        if job_id == "fail-1":
            raise RuntimeError("test failure")
        p.send_frame(None, {"op": "reset"})
        p.send_frame(None, {"op": "step", "start_pos": 0})
        on_token(17)
        on_token(18)
        return {"tokens": [17, 18], "prompt_tokens": 26, "receipts": []}

    p = types.SimpleNamespace(
        decode_greedy=decode_greedy,
        send_frame=send,
        _publish_event=lambda tag, **kw: emitted.append(kw),
    )
    metrics.install(p, {})
    for job in ("greedy-1", "greedy-2"):
        result = p.decode_greedy(job_id=job, on_token=tokens.append)
        p._publish_event("WANFERENZ_JOB_DONE", jobId=job)
        assert result["tokens"] == [17, 18]
        assert p.send_frame is send
    assert tokens == [17, 18, 17, 18]
    assert len(messages) == 4
    assert emitted[0]["timings"] == emitted[1]["timings"]
    assert emitted[0]["timings"]["decodeTokensPerSec"] == 1
    with pytest.raises(RuntimeError, match="test failure"):
        p.decode_greedy(job_id="fail-1")
    assert p.send_frame is send


@pytest.mark.parametrize("name", ["decode_greedy", "stream_dspark"])
def test_actual_coordinator_records_prefill_for_single_token_eos(monkeypatch, name):
    pytest.importorskip("torch")
    import wanferenz.serving.chain as p

    replies = iter([{"ok": True}, {"token": 17, "acc": True}])
    monkeypatch.setattr(p, "send_frame", lambda *a, **k: None)
    monkeypatch.setattr(p, "receive_frame", lambda *a, **k: next(replies))
    monkeypatch.setattr(p, "_publish_event", lambda *a, **k: None)
    for attr in ("decode_greedy", "decode_proposals", "decode_dspark", "stream_dspark"):
        monkeypatch.setattr(p, attr, getattr(p, attr))
    events = []
    monkeypatch.setattr(p, "_publish_event", lambda tag, **kw: events.append(kw))
    metrics.install(p, {})
    seen = []
    r = getattr(p, name)(
        None,
        types.SimpleNamespace(settimeout=lambda _: None),
        [1, 2],
        128,
        eos_ids=[17],
        job_id="greedy-1" if name == "decode_greedy" else "spec-1",
        on_token=seen.append,
    )
    p._publish_event("WANFERENZ_JOB_DONE")
    assert seen == r["tokens"] == [17]
    assert events[-1]["timings"]["prefillTokensPerSec"] > 0
    assert events[-1]["timings"]["decodeTokensPerSec"] is None
