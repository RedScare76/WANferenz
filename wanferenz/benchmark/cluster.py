import argparse
import datetime
import getpass
import json
import pathlib
import secrets
import shlex
import statistics
import subprocess
import sys
import time
from wanferenz.benchmark.timing import TIMING_DEFINITIONS, aggregate_timings
from wanferenz.benchmark.workload import read_workload

IMAGE = "ghcr.io/anemll/dspark-vllm-gx10:0.1.1"
PROMPT = "Write a compact technical explanation of pipeline parallelism, its communication cost, and its main bottleneck. Continue until complete."
COORDINATOR = "from wanferenz.benchmark.instrumentation import run_coordinator\nrun_coordinator()\n"
STAGE = "from wanferenz.benchmark.instrumentation import run_worker\nrun_worker()\n"


def generation_jobs(arms, runs, tokens, compare_runtime=False, workload=None):
    modes = ["greedy", *arms]

    def job(mode, i):
        outcome = {
            "jobId": f"{mode}-{i}",
            "messages": [{"role": "user", "content": PROMPT}],
            "maxNew": tokens if compare_runtime or i != "warmup" else 8,
            "temperature": 0,
            "dspark": mode != "greedy",
            "pipelined": mode != "greedy",
        }
        if workload is not None:
            outcome.pop("messages")
            outcome["promptIds"] = workload["promptIds"]
            outcome["maxNew"] = workload["outputTokens"]
        return outcome

    if compare_runtime:
        jobs = [job(mode, "warmup") for mode in modes]
        for position in range(1, runs + 1):
            jobs.extend(
                (
                    job(mode, position)
                    for mode in (modes if position % 2 else reversed(modes))
                )
            )
        return jobs
    return [
        job(mode, "warmup" if position == 0 else position)
        for mode in modes
        for position in range(runs + 1)
    ]


def latency_jobs(arms, workload, targets):
    jobs = []
    for target in targets:
        for job in generation_jobs(
            arms, workload["runs"], workload["outputTokens"], True, workload
        ):
            mode, repetition = job["jobId"].rsplit("-", 1)
            job["jobId"] = f"{mode}-r{target:03d}-{repetition}"
            jobs.append(job)
    return jobs


def summarize(log, jobs, *, require_timings=False):
    done, errors = ([], [])
    for line in log.splitlines():
        if line.startswith("WANFERENZ_JOB_DONE "):
            row = json.loads(line.split(" ", 1)[1])
            done.append(row)
            if row.get("ok") is not True:
                errors.append({"error": "Job reported failure", "jobId": row["jobId"]})
        elif line.startswith("WANFERENZ_JOB_FATAL "):
            errors.append(json.loads(line.split(" ", 1)[1]))
    expected = [row_index["jobId"] for row_index in jobs]
    if [outcome["jobId"] for outcome in done] != expected:
        errors.append({"error": "Completed jobs do not match the submitted sequence"})
    measured = [outcome for outcome in done if "warmup" not in outcome["jobId"]]
    if require_timings:
        for row in done:
            timing = row.get("timings") or {}
            if (
                timing.get("version") != 1
                or len(timing.get("tokenTimesMs", [])) != row["tokensGenerated"]
                or timing.get("prefillTokensPerSec") is None
            ):
                errors.append(
                    {
                        "error": "Missing or incomplete benchmark timings",
                        "jobId": row["jobId"],
                    }
                )
    streams = [outcome.get("tokenIds") for outcome in measured]
    exact = bool(streams) and all(
        (isinstance(text_value, list) and text_value for text_value in streams)
    )
    exact = exact and all((text_value == streams[0] for text_value in streams))
    arms = {}
    for mode in dict.fromkeys(
        (outcome["jobId"].rsplit("-", 1)[0] for outcome in measured)
    ):
        records = [
            outcome
            for outcome in measured
            if outcome["jobId"].rsplit("-", 1)[0] == mode
        ]
        if records:
            arms[mode] = {
                "runs": len(records),
                "medianTokPerSec": statistics.median(
                    (outcome["tokPerSec"] for outcome in records)
                ),
                "medianFirstTokenMs": statistics.median(
                    (outcome["firstTokenMs"] for outcome in records)
                ),
                "tokensGenerated": [outcome["tokensGenerated"] for outcome in records],
                "timings": aggregate_timings(records),
            }
    return {
        "ok": not errors and exact,
        "tokenStreamsIdentical": exact,
        "arms": arms,
        "errors": errors,
        "jobs": done,
        "timingDefinitions": TIMING_DEFINITIONS,
    }


def validate_partition_modes(log, jobs, arms, *, warmed_policies=False):
    observed, errors, current = ([], [], None)
    for line in log.splitlines():
        if line.startswith("WANFERENZ_BENCH_STAGE_ARM "):
            row = json.loads(line.split(" ", 1)[1])
            current = row["jobId"]
            observed.append(current)
            expected = arms.get(current.rsplit("-", 1)[0])
            if {
                entry_key: entry_value
                for entry_key, entry_value in row.items()
                if entry_key != "jobId"
            } != expected:
                errors.append(f"Wrong stage settings for {current}")
        elif (
            warmed_policies
            and current
            and ("warmup" not in current)
            and ("V4_FP8_GEMV=" in line)
            and ("PROVEN" in line)
        ):
            errors.append(f"FP8 safety probe repeated during measured job {current}")
    if observed != [row_index["jobId"] for row_index in jobs]:
        errors.append("LayerPartition reset sequence differs from submitted jobs")
    return errors


def run_command():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--profile", choices=("eager", "optimized", "moe-graph"), default="optimized"
    )
    argument_parser.add_argument("--dspark", action="store_true")
    argument_parser.add_argument(
        "--ref-slim", action="store_true", help="A/B the indexer-only shortcut"
    )
    argument_parser.add_argument(
        "--all-optimizations",
        action="store_true",
        help="Enable moe-graph, ref-slim, DSpark fast/MoE/head graph and lazy drafting",
    )
    argument_parser.add_argument("--dspark-moe", action="store_true")
    argument_parser.add_argument("--dspark-graph", action="store_true")
    argument_parser.add_argument("--lazy-draft", action="store_true")
    argument_parser.add_argument(
        "--timing", action="store_true", help="Record per-stage serving phase timings"
    )
    argument_parser.add_argument(
        "--snapshot-mode", choices=("full", "delta"), default="full"
    )
    argument_parser.add_argument("--dspark-full-graph", action="store_true")
    argument_parser.add_argument(
        "--compare-runtime",
        action="store_true",
        help="Alternate baseline, delta checkpoints, complete drafter graph and both on one ring",
    )
    argument_parser.add_argument(
        "--compare-kernels-head",
        action="store_true",
        help="Alternate DSpark baseline, GB10 tiles, main head graph and both",
    )
    argument_parser.add_argument("--fp8-gemv", default="auto", choices=("auto", "gb10"))
    argument_parser.add_argument("--head-graph", action="store_true")
    argument_parser.add_argument("--draft-width", type=int, default=5)
    argument_parser.add_argument("--refill-floor", type=int, default=1)
    argument_parser.add_argument(
        "--sweep",
        action="store_true",
        help="Compare depths 2/4/8/16 and refill floors 1/3/5",
    )
    argument_parser.add_argument(
        "--sweep-depths", type=int, nargs="+", default=[2, 4, 8, 16]
    )
    argument_parser.add_argument(
        "--sweep-floors", type=int, nargs="+", default=[1, 3, 5]
    )
    argument_parser.add_argument(
        "--split", type=int, default=22, help="First layer owned by the tail (22 or 23)"
    )
    argument_parser.add_argument(
        "--reference",
        type=pathlib.Path,
        help="Prior summary with matching prompt/token count",
    )
    argument_parser.add_argument(
        "--workload",
        type=pathlib.Path,
        help="Hashed workload JSON, such as benchmark-workloads/v4-flash-128.json",
    )
    argument_parser.add_argument(
        "--sweep-plan",
        type=pathlib.Path,
        help="Authorized Toxiproxy latency sweep configuration",
    )
    argument_parser.add_argument("--spec-depth", type=int, default=4)
    argument_parser.add_argument("--runs", type=int, default=3)
    argument_parser.add_argument(
        "--interleave",
        action="store_true",
        help="Full-length warmups and alternating measured passes for all modes",
    )
    argument_parser.add_argument("--tokens", type=int, default=128)
    argument_parser.add_argument("--output", type=pathlib.Path)
    argument_parser.add_argument("--remote", default="192.168.10.154")
    argument_parser.add_argument("--user", default=getpass.getuser())
    argument_parser.add_argument("--host-key-alias")
    argument_parser.add_argument("--repo", default=str(pathlib.Path.cwd()))
    argument_parser.add_argument("--model", default=str(pathlib.Path.home() / "v4-wanferenz"))
    argument_parser.add_argument("--key", default=str(pathlib.Path.home() / ".ssh/id_ed25519"))
    options = argument_parser.parse_args()
    workload = read_workload(options.workload) if options.workload else None
    sweep_plan = (
        json.loads(options.sweep_plan.read_text()) if options.sweep_plan else None
    )
    if sweep_plan and (
        not workload
        or options.sweep
        or options.compare_runtime
        or options.compare_kernels_head
    ):
        argument_parser.error(
            "Latency sweep needs --workload and cannot combine with runtime/kernel/scheduling sweeps"
        )
    if workload is not None:
        options.tokens, options.runs, options.interleave = (
            workload["outputTokens"],
            workload["runs"],
            True,
        )
        import hashlib

        if (
            hashlib.sha256(
                (pathlib.Path(options.model) / "tokenizer.json").read_bytes()
            ).hexdigest()
            != workload["tokenizerSha256"]
        ):
            argument_parser.error(
                "Model tokenizer differs from the shared workload tokenizer"
            )
    if options.all_optimizations:
        options.profile = "moe-graph"
        options.dspark = options.ref_slim = options.dspark_moe = (
            options.dspark_graph
        ) = options.lazy_draft = True
    if (
        options.dspark_moe
        or options.dspark_graph
        or options.dspark_full_graph
        or options.compare_runtime
        or options.compare_kernels_head
        or options.lazy_draft
        or options.sweep
    ) and (not options.dspark):
        argument_parser.error("drafter options require --dspark")
    if sum((options.compare_runtime, options.compare_kernels_head, options.sweep)) > 1:
        argument_parser.error(
            "runtime, kernel/head and scheduling comparisons are separate experiments"
        )
    if options.compare_kernels_head and (
        options.profile != "moe-graph" or not options.dspark_full_graph
    ):
        argument_parser.error(
            "kernel/head comparison requires --profile moe-graph --dspark-full-graph"
        )
    compare = options.compare_runtime or options.compare_kernels_head
    if options.split not in (22, 23) or not (
        1 <= options.draft_width <= 5 and 1 <= options.refill_floor <= 5
    ):
        argument_parser.error(
            "split must be 22 or 23; draft width and refill floor must be 1..5"
        )
    if not all((1 <= document <= 32 for document in options.sweep_depths)) or not all(
        (1 <= stream <= 5 for stream in options.sweep_floors)
    ):
        argument_parser.error("sweep depths must be 1..32 and floors 1..5")
    if not (
        1 <= options.spec_depth <= 32
        and 1 <= options.tokens <= 256
        and (options.runs >= 1)
    ):
        argument_parser.error(
            "depth must be 1..32, tokens 1..256 (512-position cache), and runs positive"
        )
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_value = (
        options.output
        or pathlib.Path("benchmark-results") / f"v4-{options.profile}-{stamp}"
    )
    output_value.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(32)
    ssh = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        f"HostKeyAlias={options.host_key_alias or options.remote}",
        "-i",
        options.key,
        f"{options.user}@{options.remote}",
    ]
    names = [f"wanferenz-v4-{stamp}-head", f"wanferenz-v4-{stamp}-tail"]
    coord_name = f"wanferenz-v4-{stamp}-coord"
    started = []
    arms = {}
    if options.dspark:
        if options.sweep:
            arms = {
                f"spec-d{document}-f{stream}-lazy": {
                    "depth": document,
                    "floor": stream,
                    "lazy": True,
                }
                for document in options.sweep_depths
                for stream in options.sweep_floors
            }
            arms["spec-d4-f1-eager"] = {"depth": 4, "floor": 1, "lazy": False}
        else:
            arms["spec"] = {
                "depth": options.spec_depth,
                "floor": options.refill_floor,
                "lazy": options.lazy_draft,
            }
    capacity = max(
        [options.spec_depth] + [entry_value["depth"] for entry_value in arms.values()]
    )
    stage_arms = {}
    if options.compare_runtime:
        settings = {
            "depth": options.spec_depth,
            "floor": options.refill_floor,
            "lazy": options.lazy_draft,
        }
        arms = {
            mode: dict(settings)
            for mode in ("spec-base", "spec-delta", "spec-graph", "spec-combined")
        }
        stage_arms = {
            mode: {
                "snapshot": "delta"
                if mode in ("spec-delta", "spec-combined")
                else "full",
                "fullGraph": mode in ("spec-graph", "spec-combined"),
            }
            for mode in ["greedy", *arms]
        }
    if options.compare_kernels_head:
        settings = {
            "depth": options.spec_depth,
            "floor": options.refill_floor,
            "lazy": options.lazy_draft,
        }
        arms = {
            mode: dict(settings)
            for mode in ("spec-base", "spec-tiles", "spec-head", "spec-both")
        }
        stage_arms = {
            mode: {
                "snapshot": "full",
                "fullGraph": True,
                "fp8": "gb10" if mode in ("spec-tiles", "spec-both") else "auto",
                "headGraph": mode in ("spec-head", "spec-both"),
            }
            for mode in ["greedy", *arms]
        }

    def run(argv, stage=0, timeout=60, check=True):
        outcome = subprocess.run(
            ssh + [shlex.join(argv)] if stage else argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and outcome.returncode:
            raise RuntimeError(
                (outcome.stdout + outcome.stderr).replace(token, "[redacted]")
            )
        return outcome

    env = {
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": "/workspace/wanferenz",
        "WANFERENZ_SWARM_TOKEN": token,
        "V4_LEVERS_STRICT": "1",
        "V4_SPEC_DEPTH": str(capacity),
    }
    stage_env = {
        **env,
        "V4_MAX_SEQ": "512",
        "V4_MAX_BATCH": "1",
        "V4_MOE_DECODE": "1",
        "V4_CUDA_GRAPH": "0" if options.profile == "eager" else "whole",
        "V4_DIAL_RETRY_S": "1800",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if options.profile != "eager":
        stage_env.update(
            V4_MOE_GROUPED="1", V4_FP8_GEMV=options.fp8_gemv, V4_FP8_SHARED="1"
        )
    if options.profile == "moe-graph":
        stage_env["V4_MOE_IN_GRAPH"] = "1"
    if options.dspark:
        stage_env["V4_DSPARK_FAST"] = "1"
        stage_env["V4_DSPARK_BLOCK"] = str(options.draft_width)
    if options.dspark_moe:
        stage_env["V4_DSPARK_MOE"] = "1"
    if options.dspark_graph:
        stage_env["V4_DSPARK_GRAPH"] = "1"
    if options.dspark_full_graph:
        stage_env["V4_DSPARK_FULL_GRAPH"] = "1"
    if options.ref_slim:
        stage_env["V4_REF_SLIM"] = "1"
    if options.timing:
        stage_env["V4_TIMING"] = "1"
        stage_env["V4_TIMING_EVERY"] = "64"
    stage_env["V4_SPEC_SNAPSHOT"] = options.snapshot_mode
    if options.head_graph:
        stage_env["V4_HEAD_GRAPH"] = "1"
    if compare:
        stage_env["WANFERENZ_BENCH_STAGE_ARMS"] = json.dumps(stage_arms)
    env["WANFERENZ_BENCH_ARMS"] = json.dumps(arms)
    if sweep_plan:
        env["WANFERENZ_BENCH_SWEEP_PLAN"] = json.dumps(sweep_plan)
        env["WANFERENZ_BENCH_ARMS"] = json.dumps(
            {
                f"{mode}-r{target:03d}": settings
                for mode, settings in arms.items()
                for target in sweep_plan["targetsMs"]
            }
        )
    if workload is not None:
        stage_env["V4_MAX_SEQ"] = str(workload["context"])
    (output_value / "config.json").write_text(
        json.dumps(
            {
                "profile": options.profile,
                "dspark": options.dspark,
                "image": IMAGE,
                "containers": {
                    "head": names[0],
                    "tail": names[1],
                    "coordinator": coord_name,
                },
                "stageEnv": {
                    entry_key: entry_value
                    for entry_key, entry_value in stage_env.items()
                    if entry_key != "WANFERENZ_SWARM_TOKEN"
                },
                "layers": [[0, options.split], [options.split, 43]],
                "remote": options.remote,
                "specArms": arms,
                "reference": str(options.reference) if options.reference else None,
                "stageArms": stage_arms,
                "timingDefinitions": TIMING_DEFINITIONS,
                "workload": workload,
                "modelPath": options.model,
                "sweepPlan": sweep_plan,
            },
            indent=2,
        )
        + "\n"
    )
    ping = run(["ping", "-n", "-c", "20", "-i", "0.2", options.remote], check=False)
    (output_value / "network-rtt.txt").write_text(ping.stdout + ping.stderr)

    def container(name, is_stage):
        argv = [
            "docker",
            "run",
            "--name",
            name,
            "--network",
            "host",
            "--pull",
            "never",
            "--entrypoint",
            "python3",
        ]
        argv += ["-d", "--gpus", "all", "--ipc", "host"] if is_stage else ["-i", "--rm"]
        argv += [
            "-v",
            f"{options.repo}:/workspace/wanferenz:ro",
            "-v",
            f"{options.model}:/model:ro",
            "-v",
            "/tmp/wanferenz-v4-tilelang:/root/.tilelang",
            "-w",
            "/workspace/wanferenz",
        ]
        for entry_key, entry_value in (stage_env if is_stage else env).items():
            argv += ["-e", f"{entry_key}={entry_value}"]
        argv += [IMAGE]
        return argv + (
            (["-c", STAGE] if compare else ["wanferenz/serving/chain.py"])
            if is_stage
            else ["-c", COORDINATOR]
        )

    try:
        for text_value in (1, 0):
            argv = container(names[text_value], True) + [
                "stage",
                "--stage",
                str(text_value),
                "--nstages",
                "2",
                "--lo",
                str(options.split if text_value else 0),
                "--hi",
                str(43 if text_value else options.split),
                "--port",
                "29610",
                "--bind",
                options.remote if text_value else "127.0.0.1",
                "--dir",
                "/model",
                "--device",
                "cuda",
            ]
            if not text_value:
                argv += [
                    "--next",
                    sweep_plan["forwardAddress"]
                    if sweep_plan
                    else f"{options.remote}:29610",
                ]
            elif options.dspark:
                argv += ["--dspark"]
            run(argv, text_value)
            started.append(text_value)
        deadline = time.monotonic() + 1800
        while True:
            ready = []
            for text_value in (0, 1):
                outcome = run(
                    ["docker", "logs", "--tail", "200", names[text_value]], text_value
                )
                log = outcome.stdout + outcome.stderr
                (output_value / f"stage{text_value}-loading.log").write_text(
                    log.replace(token, "[redacted]")
                )
                status = run(
                    [
                        "docker",
                        "inspect",
                        "--format",
                        "{{.State.Status}}",
                        names[text_value],
                    ],
                    text_value,
                ).stdout.strip()
                print(
                    f"stage{text_value}: {status}; "
                    + (log.splitlines()[-1] if log else "loading"),
                    flush=True,
                )
                if status != "running":
                    raise RuntimeError(f"stage{text_value} exited; see its log")
                ready.append(f"[s{text_value}] listening" in log)
            if all(ready):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Model loading exceeded 30 minutes")
            time.sleep(30)
        jobs = generation_jobs(
            arms, options.runs, options.tokens, compare or options.interleave, workload
        )
        if sweep_plan:
            jobs = latency_jobs(
                arms,
                workload,
                sweep_plan.get("wanferenzTargetsMs", sweep_plan["targetsMs"]),
            )
        (output_value / "jobs.jsonl").write_text(
            "".join((json.dumps(row_index) + "\n" for row_index in jobs))
        )
        argv = container(coord_name, False) + [
            "coord",
            "--head",
            "127.0.0.1:29610",
            "--tail",
            sweep_plan["returnAddress"] if sweep_plan else f"{options.remote}:29610",
            "--dir",
            "/model",
            "--timeout",
            "1800",
        ]
        with (
            (output_value / "jobs.jsonl").open() as src,
            (output_value / "coordinator.log").open("w") as log,
        ):
            proc = subprocess.Popen(
                argv, stdin=src, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                proc.wait(timeout=14400 if sweep_plan else 2400)
            finally:
                if proc.poll() is None:
                    run(["docker", "stop", "--timeout", "10", coord_name], check=False)
                    proc.wait(timeout=30)
        summary = summarize(
            (output_value / "coordinator.log").read_text(), jobs, require_timings=True
        )
        if workload is not None:
            for row in summary["jobs"]:
                if row["tokensGenerated"] != workload["outputTokens"] or row["timings"][
                    "promptTokens"
                ] != len(workload["promptIds"]):
                    summary["ok"] = False
                    summary["errors"].append(
                        {
                            "jobId": row["jobId"],
                            "error": "Workload token count mismatch",
                        }
                    )
        if options.reference:
            reference = json.loads(options.reference.read_text())
            ref_ids = next(
                (
                    outcome["tokenIds"]
                    for outcome in reference["jobs"]
                    if outcome["jobId"] == "greedy-1"
                )
            )
            measured = [
                outcome
                for outcome in summary["jobs"]
                if "warmup" not in outcome["jobId"]
            ]
            matches = bool(measured) and all(
                (outcome["tokenIds"] == ref_ids for outcome in measured)
            )
            summary["referenceTokensIdentical"] = matches
            if not matches:
                summary["ok"] = False
                summary["errors"].append(
                    {"error": "Token stream differs from the saved reference"}
                )
        if proc.returncode:
            summary["ok"] = False
            summary["errors"].append({"error": f"Coordinator exited {proc.returncode}"})
        if options.dspark_full_graph or options.compare_runtime:
            report = run(["docker", "logs", names[1]], 1)
            tail_log = report.stdout + report.stderr
            observed = (
                "[v4 dspark] complete drafter graph captured" in tail_log
                and "[v4 dspark] complete graph declined:" not in tail_log
            )
            summary["completeDrafterGraphCaptured"] = observed
            if not observed:
                summary["ok"] = False
                summary["errors"].append(
                    {"error": "Complete drafter graph did not capture successfully"}
                )
        if options.head_graph or options.compare_kernels_head:
            report = run(["docker", "logs", names[1]], 1)
            log = report.stdout + report.stderr
            observed = (
                "[v4] main head graph captured" in log
                and "[v4] main head graph declined:" not in log
            )
            summary["mainHeadGraphCaptured"] = observed
            if not observed:
                summary["ok"] = False
                summary["errors"].append(
                    {"error": "Main head graph did not capture successfully"}
                )
        if compare:
            settings_ok = True
            for side in (0, 1):
                report = run(["docker", "logs", names[side]], side)
                errors = validate_partition_modes(
                    report.stdout + report.stderr,
                    jobs,
                    stage_arms,
                    warmed_policies=options.compare_kernels_head,
                )
                if errors:
                    settings_ok = summary["ok"] = False
                    summary["errors"].extend(
                        ({"stage": side, "error": failure} for failure in errors)
                    )
            summary["stageSettingsVerified"] = settings_ok
        (output_value / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps(
                {
                    entry_key: entry_value
                    for entry_key, entry_value in summary.items()
                    if entry_key != "jobs"
                },
                indent=2,
            ),
            flush=True,
        )
        return 0 if summary["ok"] else 1
    finally:
        cleanup_errors = []
        for text_value in (0, 1):
            if text_value not in started:
                continue
            try:
                outcome = run(
                    ["docker", "logs", names[text_value]], text_value, check=False
                )
                (output_value / f"stage{text_value}.log").write_text(
                    (outcome.stdout + outcome.stderr).replace(token, "[redacted]")
                )
            except Exception as exc:
                print(f"Could not save stage{text_value} log: {exc}", file=sys.stderr)
            finally:
                try:
                    run(
                        ["docker", "stop", "--timeout", "30", names[text_value]],
                        text_value,
                    )
                except Exception as exc:
                    cleanup_errors.append(f"Could not stop {names[text_value]}: {exc}")
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))


if __name__ == "__main__":
    raise SystemExit(run_command())
