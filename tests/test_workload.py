import json

import pytest

from wanferenz.benchmark.workload import compose_workload, check_workload


def workload():
    return compose_workload(
        {
            "renderedPrompt": "<bos>test",
            "promptIds": [1, 2, 3],
            "tokenizerSha256": "test-tokenizer",
        },
        "test",
        tokens=4,
        runs=2,
    )


def test_workload_hash_and_context_are_checked():
    w = workload()
    w["promptIds"].append(99)
    with pytest.raises(ValueError, match="hash mismatch"):
        check_workload(w)
    with pytest.raises(ValueError, match="exceed context"):
        compose_workload(
            {
                "renderedPrompt": "long",
                "promptIds": list(range(400)),
                "tokenizerSha256": "x",
            }
        )


def test_recorded_workload_matches_isolated_encoder():
    import pathlib
    import subprocess
    import sys

    from wanferenz.benchmark import cluster, workload as definition

    root = pathlib.Path(__file__).resolve().parents[1]
    saved = definition.read_workload(root / "benchmark-workloads/v4-flash-128.json")
    assert cluster.PROMPT == definition.PROMPT == saved["prompt"]
    script = (
        definition.ENCODE.replace("/workspace/wanferenz", str(root))
        .replace("'/model'", repr(str(root / "wanferenz/model/data")))
        .replace(
            "'/model/tokenizer.json'",
            repr(str(root / "wanferenz/model/data/tokenizer.json")),
        )
    )
    process = subprocess.run(
        [sys.executable, "-I", "-c", script],
        input=json.dumps(saved["prompt"]),
        text=True,
        capture_output=True,
        check=True,
        cwd="/tmp",
    )
    regenerated = definition.compose_workload(json.loads(process.stdout))
    assert regenerated == saved
