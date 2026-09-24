import json

from wanferenz.benchmark.cluster import summarize


def _run(tokens):
    jobs = [{"jobId": name} for name in ("greedy-1", "greedy-2", "spec-1")]
    log = "\n".join(
        "WANFERENZ_JOB_DONE "
        + json.dumps(
            {
                **job,
                "ok": True,
                "tokenIds": stream,
                "tokensGenerated": len(stream),
                "tokPerSec": 10 + i,
                "firstTokenMs": 100,
            }
        )
        for i, (job, stream) in enumerate(zip(jobs, tokens))
    )
    return jobs, log


def test_v4_benchmark_compares_exact_ids_across_greedy_and_speculation():
    jobs, log = _run([[1, 2, 3]] * 3)
    result = summarize(log, jobs)
    assert result["ok"] and result["tokenStreamsIdentical"]
    assert result["arms"]["greedy"]["medianTokPerSec"] == 10.5
    assert result["arms"]["spec"]["medianTokPerSec"] == 12


def test_v4_benchmark_rejects_a_fast_but_different_speculative_stream():
    jobs, log = _run([[1, 2, 3], [1, 2, 3], [1, 2, 4]])
    result = summarize(log, jobs)
    assert not result["ok"] and not result["tokenStreamsIdentical"]


def test_v4_benchmark_requires_every_job_and_no_fatal_events():
    jobs, log = _run([[1, 2, 3]] * 3)
    assert not summarize(log, jobs + [{"jobId": "spec-2"}])["ok"]
    assert not summarize(log + '\nWANFERENZ_JOB_FATAL {"error": "reset failed"}', jobs)[
        "ok"
    ]


def test_v4_benchmark_rejects_failed_jobs_even_with_matching_tokens():
    jobs, log = _run([[1, 2, 3]] * 3)
    assert not summarize(log.replace('"ok": true', '"ok": false', 1), jobs)["ok"]


def test_runtime_comparison_warms_all_modes_then_reverses_measurement_order():
    from wanferenz.benchmark.cluster import generation_jobs

    modes = ["greedy", "spec-base", "spec-delta", "spec-graph", "spec-combined"]
    jobs = generation_jobs(
        dict.fromkeys(modes[1:]), runs=2, tokens=128, compare_runtime=True
    )
    assert [j["jobId"] for j in jobs[:5]] == [m + "-warmup" for m in modes]
    assert [j["jobId"] for j in jobs[5:10]] == [m + "-1" for m in modes]
    assert [j["jobId"] for j in jobs[10:]] == [m + "-2" for m in reversed(modes)]
    assert all(j["maxNew"] == 128 for j in jobs)
    assert len({j["jobId"] for j in jobs}) == 15
