import json
import sys
from wanferenz.protocol.placement import choose_chain

DEFAULT_CAPACITY = {
    "n_layers": 62,
    "layer_vram_mb": 2330.0,
    "kv_mb_per_layer": 150.0,
    "layer_ms_base": 0.65,
    "reserve_mb": 1500.0,
    "head_reserve_mb": 4096.0,
    "tail_reserve_mb": 1400.0,
    "cap_layers": 12,
    "head_layer_ms_mult": 1.3,
    "prefill_bytes": 4096 * 3072 * 2.0,
    "decode_bytes": 9 * 3072 * 2.0,
    "decode_steps": 256,
    "prefill_chunks": 1,
}
PROFILES = {}


def lookup_capacity(model_id: str) -> dict:
    try:
        return dict(PROFILES[model_id])
    except KeyError:
        raise ValueError(
            f"no engine profile for model_id {model_id!r} (known: {', '.join(sorted(PROFILES))})"
        ) from None


_SLACK = 3
_UNREACHABLE = 9000.0
_PROVEN_CAP_VRAM_MB = 32768.0


def scale_layer_limit(cap_layers, total_vram_mb):
    return max(
        0, int(round(int(cap_layers) * float(total_vram_mb) / _PROVEN_CAP_VRAM_MB))
    )


def plan_chain(nodes, rtt, model=None, *, slack=None, privacy=None):
    if isinstance(model, str):
        model = lookup_capacity(model)
    m = {**DEFAULT_CAPACITY, **(model or {})}
    item_count = len(nodes)
    if item_count == 0:
        return None
    ids = [nd["id"] for nd in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node id in `nodes`")
    layer_vram, kv = (float(m["layer_vram_mb"]), float(m["kv_mb_per_layer"]))
    cap_layers = int(m["cap_layers"])
    lv = {
        position: float(nodes[position].get("layer_vram_mb") or layer_vram)
        for position in range(item_count)
    }
    per_layer = {position: lv[position] + kv for position in range(item_count)}

    def _node_cap(i):
        if nodes[i].get("cap_layers") is not None:
            return int(nodes[i]["cap_layers"])
        total = float(nodes[i].get("total_vram_mb") or 0.0)
        return scale_layer_limit(cap_layers, total) if total > 0 else cap_layers

    free = {
        position: min(
            max(
                nodes[position]["free_vram_mb"]
                - float(m["reserve_mb"])
                - float(
                    nodes[position].get("load_peak_extra_mb")
                    or m.get("load_peak_extra_mb")
                    or 0.0
                ),
                0.0,
            ),
            _node_cap(position) * per_layer[position],
        )
        for position in range(item_count)
    }
    cap_ok = [
        position
        for position in range(item_count)
        if free[position] >= per_layer[position]
    ]
    if not cap_ok:
        return None
    pin = privacy is not None
    trusted = (
        {
            position
            for position in range(item_count)
            if nodes[position].get("trusted") is True
        }
        if pin
        else None
    )
    head_pool = (
        [position for position in cap_ok if position in trusted] if pin else cap_ok
    )
    if not head_pool:
        return None

    def centrality(i):
        return sum(
            (
                min(float(rtt[i][row_index]), _UNREACHABLE)
                for row_index in range(item_count)
                if row_index != i
            )
        )

    def _connected_cap(i):
        return int(free[i] // per_layer[i]) + sum(
            (
                int(free[row_index] // per_layer[row_index])
                for row_index in cap_ok
                if row_index != i
                and rtt[i][row_index] < _UNREACHABLE
                and (rtt[row_index][i] < _UNREACHABLE)
            )
        )

    head_pool = [
        position
        for position in head_pool
        if _connected_cap(position) >= int(m["n_layers"])
    ]
    if not head_pool:
        return None
    head = min(head_pool, key=centrality)
    free[head] = max(free[head] - float(m["head_reserve_mb"]), 0.0)
    layer_ms = {
        position: float(nodes[position]["layer_ms"])
        if nodes[position].get("layer_ms") is not None
        else float(m["layer_ms_base"]) * float(nodes[position].get("cpu_factor", 1.0))
        for position in range(item_count)
    }
    layer_ms[head] *= float(m["head_layer_ms_mult"])
    c_out = [
        rtt[head][position] if position != head else 1.0
        for position in range(item_count)
    ]
    c_in = [
        rtt[position][head] if position != head else 1.0
        for position in range(item_count)
    ]
    subnet = {position: nodes[position]["subnet"] for position in range(item_count)}
    ups = [nodes[position].get("up_mbps") for position in range(item_count)]
    aware = all((u is not None for u in ups))
    extra = {}
    if aware:
        extra = {
            "up_mbps": {
                position: float(ups[position]) for position in range(item_count)
            },
            "prefill_bytes": float(m.get("prefill_bytes", 0.0)),
            "decode_bytes": float(m.get("decode_bytes", 0.0)),
            "decode_steps": int(m.get("decode_steps", 1)),
            "prefill_chunks": int(m.get("prefill_chunks", 1)),
        }
    if pin:
        extra["trusted"] = trusted
        extra["boundary_in"] = int(privacy.get("boundary_in", 0))
        extra["boundary_out"] = int(privacy.get("boundary_out", 0))
    tail_reserve = float(m.get("tail_reserve_mb", 0.0))
    base_free = dict(free)
    docked = set()
    spec = None
    for _ in range(item_count + 1):
        spec = choose_chain(
            range(item_count),
            rtt,
            c_out,
            c_in,
            free_vram_mb=free,
            layer_ms=layer_ms,
            subnet=subnet,
            n_layers=int(m["n_layers"]),
            layer_vram_mb=lv,
            kv_mb_per_layer=kv,
            slack=min(item_count, _SLACK) if slack is None else int(slack),
            require=head,
            **extra,
        )
        if spec is None:
            return None
        tail_i = spec["order"][-1]
        lo, hi = spec["blocks"][tail_i]
        if (
            tail_reserve == 0.0
            or base_free[tail_i] >= (hi - lo) * per_layer[tail_i] + tail_reserve
        ):
            break
        if tail_i in docked:
            return None
        docked.add(tail_i)
        free[tail_i] = max(base_free[tail_i] - tail_reserve, 0.0)
    else:
        return None
    assert spec["order"][0] == head, (
        "choose_chain must return a head-first (deployable) order"
    )
    _o = spec["order"]
    if any((rtt[options][b] >= _UNREACHABLE for options, b in zip(_o, _o[1:]))) or (
        _o[-1] != head and rtt[_o[-1]][head] >= _UNREACHABLE
    ):
        return None
    for position in spec["order"]:
        lo, hi = spec["blocks"][position]
        need = (hi - lo) * per_layer[position] + (
            tail_reserve if position == spec["order"][-1] else 0.0
        )
        if need > base_free[position] + 1e-06:
            raise RuntimeError(
                f"planned block [{lo}:{hi}) needs {need:.0f} MB on node {ids[position]!r} whose budget is {base_free[position]:.0f} MB"
            )
    boundary = set(spec.get("boundary", []))
    order = [ids[position] for position in spec["order"]]
    last = len(spec["order"]) - 1
    stages = []
    for entry_key, position in enumerate(spec["order"]):
        lo, hi = spec["blocks"][position]
        state = {
            "id": ids[position],
            "index": entry_key,
            "lo": lo,
            "hi": hi,
            "head": entry_key == 0,
            "tail": entry_key == last,
            "layers": hi - lo,
        }
        if pin:
            state["boundary"] = position in boundary
        stages.append(state)
    output_value = {
        "order": order,
        "head": ids[head],
        "stages": stages,
        "dropped": [ids[position] for position in spec["dropped"]],
        "step_ms": spec["step_ms"],
        "tok_s_per_g": spec["tok_s_per_g"],
        "k": spec["k"],
    }
    if aware:
        output_value["request_ms"] = spec.get("request_ms")
        output_value["prefill_ms"] = spec.get("prefill_ms")
        output_value["roles"] = {
            ids[int(position)]: outcome
            for position, outcome in spec.get("roles", {}).items()
        }
    if pin:
        output_value["privacy"] = {
            "boundary_in": extra["boundary_in"],
            "boundary_out": extra["boundary_out"],
            "boundary_stages": [
                ids[position] for position in spec["order"] if position in boundary
            ],
        }
    return output_value


def _main() -> int:
    try:
        req = json.load(sys.stdin)
    except Exception as failure:
        json.dump({"error": f"bad request json: {failure}"}, sys.stdout)
        return 2
    try:
        capacity = plan_chain(
            req["nodes"],
            req["rtt"],
            req.get("model"),
            slack=req.get("slack"),
            privacy=req.get("privacy"),
        )
    except KeyError as failure:
        json.dump({"error": f"missing field: {failure}"}, sys.stdout)
        return 2
    except Exception as failure:
        json.dump({"error": f"plan failed: {failure}"}, sys.stdout)
        return 1
    json.dump(capacity, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
