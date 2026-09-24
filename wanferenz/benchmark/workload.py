import hashlib
import json
from pathlib import Path

PROMPT = "Write a compact technical explanation of pipeline parallelism, its communication cost, and its main bottleneck. Continue until complete."
ENCODE = "import hashlib, json, sys\nfrom pathlib import Path\nsys.path.insert(0, '/workspace/wanferenz')\nfrom wanferenz.model.chat import encode_messages\nfrom transformers import PreTrainedTokenizerFast\nprompt = json.load(sys.stdin)\nrendered = encode_messages([{'role': 'user', 'content': prompt}], 'chat', reasoning_effort=None)\ntok = PreTrainedTokenizerFast.from_pretrained('/model', fix_mistral_regex=True)\nprint(json.dumps({'renderedPrompt': rendered, 'promptIds': tok.encode(rendered, add_special_tokens=False), 'tokenizerSha256': hashlib.sha256(Path('/model/tokenizer.json').read_bytes()).hexdigest()}))\n"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def compose_workload(encoded, prompt=PROMPT, *, tokens=128, runs=4, context=512):
    workload = {
        "version": 1,
        "prompt": prompt,
        **encoded,
        "outputTokens": tokens,
        "runs": runs,
        "context": context,
        "temperature": 0,
        "seed": 0,
        "ignoreEos": False,
        "thinking": False,
        "batchSize": 1,
        "warmups": 1,
        "cachePrompt": False,
        "format": "DeepSeek official encode_messages(chat), no explicit system message",
    }
    workload["id"] = digest(workload)
    check_workload(workload)
    return workload


def check_workload(w):
    if w.get("id") != digest(
        {
            entry_key: entry_value
            for entry_key, entry_value in w.items()
            if entry_key != "id"
        }
    ):
        raise ValueError(
            "Workload hash mismatch; generate a new workload instead of editing it"
        )
    fixed = {
        "version": 1,
        "temperature": 0,
        "seed": 0,
        "ignoreEos": False,
        "thinking": False,
        "batchSize": 1,
        "warmups": 1,
        "cachePrompt": False,
    }
    if any(
        (w.get(entry_key) != entry_value for entry_key, entry_value in fixed.items())
    ):
        raise ValueError("Unsupported workload settings")
    ids = w.get("promptIds")
    if (
        not isinstance(ids, list)
        or not ids
        or any((type(t) is not int or t < 0 for t in ids))
    ):
        raise ValueError("Workload requires a nonempty list of token IDs")
    if (
        type(w.get("outputTokens")) is not int
        or not 2 <= w["outputTokens"] <= 256
        or type(w.get("runs")) is not int
        or (w["runs"] < 2)
        or (type(w.get("context")) is not int)
        or (not 512 <= w["context"] <= 16384)
    ):
        raise ValueError(
            "Require 2..256 output tokens, at least two runs, and 512..16384 context"
        )
    if len(ids) + w["outputTokens"] + 32 > w["context"]:
        raise ValueError(
            "Prompt/output exceed context after reserving 32 speculative positions"
        )
    if not w.get("renderedPrompt") or not w.get("tokenizerSha256"):
        raise ValueError("Missing rendered prompt or tokenizer identity")
    return w


def read_workload(path):
    return check_workload(json.loads(Path(path).read_text()))
