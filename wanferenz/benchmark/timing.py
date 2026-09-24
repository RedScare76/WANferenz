import math
import statistics
import time

TIMING_DEFINITIONS = {
    "clock": "Coordinator time.perf_counter; seconds stored relative to job start.",
    "prefill": "First prompt send to first token callback, including transport and output sampling; excludes reset and tokenisation. Not GPU-only compute time.",
    "promptProcessing": "Alias of distributed prefill: prompt tokens / prefill seconds.",
    "decode": "Tokens after the first / first-to-last token callback seconds; excludes prefill and final drain.",
    "interTokenLatency": "Consecutive committed-token callback gaps, including communication and scheduling; speculative bursts retain their short gaps. Not browser/client arrival timing.",
    "percentiles": "Linear interpolation at (count - 1) * percentile; no intervals for a one-token output.",
    "aggregation": "Per-arm medians across measured jobs, excluding warmups. ITL summaries are medians of per-job statistics, not pooled percentiles.",
}


def summarize_intervals(values):
    if not values:
        return {key: None for key in ("mean", "median", "p95", "p99", "min", "max")}
    ranked = sorted(values)

    def percentile(q):
        index = (len(ranked) - 1) * q
        lo, hi = (math.floor(index), math.ceil(index))
        return ranked[lo] + (ranked[hi] - ranked[lo]) * (index - lo)

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": ranked[0],
        "max": ranked[-1],
    }


class GenerationClock:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.start = clock()
        self.prompt_sent = None
        self.tokens = []

    def sent(self, message):
        if (
            self.prompt_sent is None
            and isinstance(message, dict)
            and (message.get("op") == "step")
            and (message.get("start_pos") == 0)
        ):
            self.prompt_sent = self.clock()

    def token(self):
        self.tokens.append(self.clock())

    def finish(self, prompt_tokens, generated_tokens):
        end = self.clock()
        if (
            self.prompt_sent is None
            or not self.tokens
            or len(self.tokens) != generated_tokens
        ):
            raise ValueError(
                "Incomplete benchmark timing: prompt send or token callbacks missing"
            )
        first, last = (self.tokens[0], self.tokens[-1])
        prefill = first - self.prompt_sent
        decode = last - first
        if (
            prefill <= 0
            or self.prompt_sent < self.start
            or end < last
            or any((b < options for options, b in zip(self.tokens, self.tokens[1:])))
        ):
            raise ValueError("Invalid benchmark clock ordering")
        intervals = [
            (b - options) * 1000 for options, b in zip(self.tokens, self.tokens[1:])
        ]
        return {
            "version": 1,
            "promptTokens": prompt_tokens,
            "outputTokens": generated_tokens,
            "prefillMs": prefill * 1000,
            "prefillTokensPerSec": prompt_tokens / prefill,
            "promptProcessingTokensPerSec": prompt_tokens / prefill,
            "decodeMs": decode * 1000,
            "decodeTokensPerSec": (generated_tokens - 1) / decode
            if decode > 0
            else None,
            "timeToFirstTokenMs": (first - self.start) * 1000,
            "beforePrefillMs": (self.prompt_sent - self.start) * 1000,
            "postLastTokenMs": (end - last) * 1000,
            "tokenTimesMs": [(t - self.start) * 1000 for t in self.tokens],
            "interTokenLatencyMs": intervals,
            "interTokenLatencySummaryMs": summarize_intervals(intervals),
        }


def install(p, arms):
    last, timing = ({}, {})
    for name in ("decode_greedy", "decode_proposals", "decode_dspark", "stream_dspark"):
        if not hasattr(p, name):
            continue
        original = getattr(p, name)

        def capture(*args, _original=original, _name=name, **kwargs):
            last.clear()
            timing.clear()
            if _name == "stream_dspark":
                arm = kwargs["job_id"].rsplit("-", 1)[0]
                kwargs.update(arms.get(arm, {}))
            meter = GenerationClock()
            original_send = p.send_frame
            original_token = kwargs.get("on_token")

            def send(sock, message, *a, **kw):
                meter.sent(message)
                return original_send(sock, message, *a, **kw)

            def token(tid):
                meter.token()
                if original_token is not None:
                    original_token(tid)

            p.send_frame = send
            kwargs["on_token"] = token
            try:
                outcome = _original(*args, **kwargs)
                timing.update(
                    meter.finish(outcome["prompt_tokens"], len(outcome["tokens"]))
                )
                last.update(
                    {
                        entry_key: entry_value
                        for entry_key, entry_value in outcome.items()
                        if entry_key != "receipts"
                    }
                )
                return outcome
            finally:
                p.send_frame = original_send

        setattr(p, name, capture)
    original_emit = p._publish_event

    def emit(tag, **fields):
        if tag == "WANFERENZ_JOB_DONE":
            fields["tokenIds"] = last.get("tokens")
            fields["benchmarkStats"] = {
                entry_key: entry_value
                for entry_key, entry_value in last.items()
                if entry_key != "tokens"
            }
            fields["timings"] = dict(timing)
        original_emit(tag, **fields)

    p._publish_event = emit


def aggregate_timings(rows):
    measured = [outcome["timings"] for outcome in rows if outcome.get("timings")]
    if len(measured) != len(rows):
        return None

    def median(values):
        return (
            statistics.median(values)
            if all((entry_value is not None for entry_value in values))
            else None
        )

    result = {
        "median" + key[0].upper() + key[1:]: median([m[key] for m in measured])
        for key in (
            "prefillMs",
            "prefillTokensPerSec",
            "promptProcessingTokensPerSec",
            "decodeMs",
            "decodeTokensPerSec",
            "timeToFirstTokenMs",
        )
    }
    result["medianPerJobInterTokenLatencyMs"] = {
        key: median([m["interTokenLatencySummaryMs"][key] for m in measured])
        for key in ("mean", "median", "p95", "p99", "min", "max")
    }
    return result
