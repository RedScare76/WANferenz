from itertools import combinations, permutations

INF = float("inf")
_TRIM = 12


def _up(up_mbps, n):
    entry_value = up_mbps.get(n, 0.0) if hasattr(up_mbps, "get") else up_mbps[n]
    return max(float(entry_value), 0.5)


def _xfer_ms(nbytes, up_mbps, n):
    return nbytes * 8.0 / (_up(up_mbps, n) * 1000.0)


def loop_cost(order, L, c_out, c_in):
    if not order:
        return 0.0
    cost = c_out[order[0]] + c_in[order[-1]]
    for options, b in zip(order, order[1:]):
        cost += L[options][b]
    return cost


def _held_karp(nodes, L, c_out, c_in):
    idx = list(nodes)
    entry_key = len(idx)
    if entry_key == 1:
        return (idx, c_out[idx[0]] + c_in[idx[0]])
    offset = {item_count: position for position, item_count in enumerate(idx)}
    dp = [[INF] * entry_key for _ in range(1 << entry_key)]
    par = [[-1] * entry_key for _ in range(1 << entry_key)]
    for row_index in range(entry_key):
        dp[1 << row_index][row_index] = c_out[idx[row_index]]
    for mask in range(1 << entry_key):
        for row_index in range(entry_key):
            if dp[mask][row_index] == INF or not mask >> row_index & 1:
                continue
            base = dp[mask][row_index]
            for m in range(entry_key):
                if mask >> m & 1:
                    continue
                nmask = mask | 1 << m
                cand = base + L[idx[row_index]][idx[m]]
                if cand < dp[nmask][m]:
                    dp[nmask][m] = cand
                    par[nmask][m] = row_index
    full = (1 << entry_key) - 1
    best, bj = (INF, -1)
    for row_index in range(entry_key):
        c = dp[full][row_index] + c_in[idx[row_index]]
        if c < best:
            best, bj = (c, row_index)
    order, mask, row_index = ([], full, bj)
    while row_index != -1:
        order.append(idx[row_index])
        pj = par[mask][row_index]
        mask ^= 1 << row_index
        row_index = pj
    order.reverse()
    return (order, best)


def _pin_order_oversize(nodes, L, c_out, c_in, heads, ends):
    best_o, best_c = (None, INF)
    for h in heads:
        for t in ends:
            if t == h:
                continue
            mid, tour = (set(nodes) - {h, t}, [h])
            while mid:
                nxt = min(mid, key=lambda n: L[tour[-1]][n])
                tour.append(nxt)
                mid.discard(nxt)
            tour.append(t)
            improved = True
            while improved:
                improved = False
                for position in range(1, len(tour) - 2):
                    for row_index in range(position + 1, len(tour) - 1):
                        cand = (
                            tour[:position]
                            + tour[position : row_index + 1][::-1]
                            + tour[row_index + 1 :]
                        )
                        if loop_cost(cand, L, c_out, c_in) + 1e-09 < loop_cost(
                            tour, L, c_out, c_in
                        ):
                            tour, improved = (cand, True)
            c = loop_cost(tour, L, c_out, c_in)
            if c < best_c:
                best_o, best_c = (tour, c)
    return (best_o, best_c)


def _nn_2opt(nodes, L, c_out, c_in, rounds=4):
    idx = list(nodes)
    start = min(idx, key=lambda n: c_out[n])
    tour, rest = ([start], set(idx) - {start})
    while rest:
        last = tour[-1]
        nxt = min(rest, key=lambda n: L[last][n])
        tour.append(nxt)
        rest.discard(nxt)
    best = loop_cost(tour, L, c_out, c_in)
    improved = True
    while improved:
        improved = False
        for position in range(len(tour) - 1):
            for row_index in range(position + 1, len(tour)):
                cand = (
                    tour[:position]
                    + tour[position : row_index + 1][::-1]
                    + tour[row_index + 1 :]
                )
                c = loop_cost(cand, L, c_out, c_in)
                if c + 1e-09 < best:
                    tour, best, improved = (cand, c, True)
    return (tour, best)


def optimal_loop(nodes, L, c_out, c_in):
    nodes = list(nodes)
    if len(nodes) <= 16:
        return _held_karp(nodes, L, c_out, c_in)
    return _nn_2opt(nodes, L, c_out, c_in)


def select_and_order(nodes, L, c_out, c_in, k):
    nodes = list(nodes)
    if k >= len(nodes):
        return optimal_loop(nodes, L, c_out, c_in)
    if len(nodes) <= 16:
        best_order, best_cost = (None, INF)
        for subset in combinations(nodes, k):
            order, cost = _held_karp(subset, L, c_out, c_in)
            if cost < best_cost:
                best_order, best_cost = (order, cost)
        return (best_order, best_cost)
    order, _ = _nn_2opt(nodes, L, c_out, c_in)
    while len(order) > k:
        drop = min(
            range(len(order)),
            key=lambda i: loop_cost(order[:i] + order[i + 1 :], L, c_out, c_in),
        )
        order = order[:drop] + order[drop + 1 :]
    return _nn_2opt(order, L, c_out, c_in)


def predict_step_ms(
    order, layers, L, c_out, c_in, layer_ms, up_mbps=None, decode_bytes=0.0
):
    ms = loop_cost(order, L, c_out, c_in) + sum(
        (layers[item_count] * layer_ms[item_count] for item_count in order)
    )
    if up_mbps is not None and decode_bytes:
        ms += sum((_xfer_ms(decode_bytes, up_mbps, item_count) for item_count in order))
    return ms


def predict_prefill_ms(
    order,
    layers,
    L,
    c_out,
    c_in,
    up_mbps,
    prefill_bytes,
    prefill_chunks=1,
    prefill_layer_ms=None,
):
    lat = loop_cost(order, L, c_out, c_in)
    fwd = order[:-1]
    C = max(1, int(prefill_chunks))
    if fwd:
        us = [_xfer_ms(prefill_bytes, up_mbps, item_count) for item_count in fwd]
        framing = (sum(us) + (C - 1) * max(us)) / C
    else:
        framing = 0.0
    compute = (
        sum((layers[item_count] * prefill_layer_ms[item_count] for item_count in order))
        if prefill_layer_ms
        else 0.0
    )
    return lat + framing + compute


def _boundary_nodes(order, alloc, n_layers, boundary_in, boundary_out):
    output_value, lo = ({order[0], order[-1]}, 0)
    hi_cut = n_layers - max(0, boundary_out)
    for item_count in order:
        hi = lo + alloc[item_count]
        if lo < boundary_in or hi > hi_cut:
            output_value.add(item_count)
        lo = hi
    return output_value


def _relegate(
    order, dropped, caps, subnet, up_mbps, layer_ms, trusted=None, boundary_subnets=()
):
    if not order:
        return {}
    ring_subnets = {subnet[item_count] for item_count in order}
    ring_best_up = max((_up(up_mbps, item_count) for item_count in order))
    ring_worst_compute = max((layer_ms[item_count] for item_count in order))
    roles = {}
    for item_count in dropped:
        if caps.get(item_count, 0) == 0:
            roles[item_count] = "weight-seeder"
        elif (
            _up(up_mbps, item_count) >= ring_best_up
            and subnet[item_count] not in ring_subnets
        ):
            roles[item_count] = "aggregator"
        elif subnet[item_count] in ring_subnets and (
            trusted is None
            or item_count in trusted
            or subnet[item_count] not in boundary_subnets
        ):
            roles[item_count] = "hot-standby"
        elif layer_ms[item_count] <= ring_worst_compute:
            roles[item_count] = "decode-only-replica"
        else:
            roles[item_count] = "spot-check-verifier"
    return roles


def _head_first(order, head, L, c_out, c_in):
    if order[0] == head:
        return list(order)
    rest = [item_count for item_count in order if item_count != head]
    if len(rest) <= 7:
        best = min(
            permutations(rest), key=lambda p: loop_cost([head, *p], L, c_out, c_in)
        )
        return [head, *best]
    cand = ([head] + rest, [head] + rest[::-1])
    return list(min(cand, key=lambda p: loop_cost(p, L, c_out, c_in)))


def node_capacity(free_vram_mb, layer_vram_mb, kv_mb_per_layer=0):
    per = layer_vram_mb + kv_mb_per_layer
    return int(free_vram_mb // per) if per > 0 else 0


def assign_layers(order, n_layers, caps, layer_ms, floors=None):
    base = {
        item_count: max(1, (floors or {}).get(item_count, 1)) for item_count in order
    }
    need = sum(base.values())
    if n_layers <= 0 or n_layers < need:
        return None
    if any((base[item_count] > caps[item_count] for item_count in order)):
        return None
    if sum((caps[item_count] for item_count in order)) < n_layers:
        return None
    alloc = dict(base)
    rem = n_layers - need
    for item_count in sorted(order, key=lambda n: layer_ms[n]):
        take = min(caps[item_count] - alloc[item_count], rem)
        if take > 0:
            alloc[item_count] += take
            rem -= take
        if rem <= 0:
            break
    return alloc if rem == 0 else None


def choose_chain(
    nodes,
    L,
    c_out,
    c_in,
    *,
    free_vram_mb,
    layer_ms,
    subnet,
    n_layers,
    layer_vram_mb,
    kv_mb_per_layer=0,
    slack=2,
    exclude=None,
    require=None,
    up_mbps=None,
    prefill_bytes=0.0,
    decode_bytes=0.0,
    decode_steps=1,
    prefill_chunks=1,
    prefill_layer_ms=None,
    relegate=True,
    trusted=None,
    boundary_in=0,
    boundary_out=0,
):
    if require is not None and exclude and (require in set(exclude)):
        raise ValueError("`require` and `exclude` name the same node")
    pin = trusted is not None
    trust = set(trusted) if pin else set()
    b_in = min(max(0, int(boundary_in)), n_layers) if pin else 0
    b_out = min(max(0, int(boundary_out)), n_layers) if pin else 0
    nodes = [
        item_count for item_count in nodes if not exclude or item_count not in exclude
    ]

    def _lv(n):
        return layer_vram_mb[n] if isinstance(layer_vram_mb, dict) else layer_vram_mb

    caps = {
        item_count: node_capacity(
            free_vram_mb[item_count], _lv(item_count), kv_mb_per_layer
        )
        for item_count in nodes
    }
    usable = [item_count for item_count in nodes if caps[item_count] > 0]
    if require is not None and require not in usable:
        return None
    if pin:
        if require is not None and require not in trust:
            return None
        if not any((item_count in trust for item_count in usable)):
            return None

    def feasible_cap(pool):
        best = {}
        for item_count in pool:
            best[subnet[item_count]] = max(
                best.get(subnet[item_count], 0), caps[item_count]
            )
        return sum(best.values())

    if feasible_cap(usable) < n_layers:
        return None
    by_cap = sorted(usable, key=lambda n: caps[n], reverse=True)
    acc, k_min, used_sub = (0, 0, set())
    for item_count in by_cap:
        if subnet[item_count] in used_sub:
            continue
        used_sub.add(subnet[item_count])
        acc += caps[item_count]
        k_min += 1
        if acc >= n_layers:
            break
    if len(usable) > _TRIM:
        keep = sorted(usable, key=lambda n: c_out[n] + c_in[n])[:_TRIM]
        must, observed = (set(), set())
        for m in by_cap:
            if subnet[m] in observed:
                continue
            observed.add(subnet[m])
            must.add(m)
            if len(must) >= k_min + slack:
                break
        if require is not None:
            must.add(require)
            observed, acc = ({subnet[require]}, caps[require])
            for m in by_cap:
                if subnet[m] in observed:
                    continue
                observed.add(subnet[m])
                must.add(m)
                acc += caps[m]
                if acc >= n_layers:
                    break
        if pin:
            observed, kept_t = (set(), 0)
            for m in by_cap:
                if m not in trust or subnet[m] in observed:
                    continue
                observed.add(subnet[m])
                must.add(m)
                kept_t += 1
                if kept_t >= 2 + slack:
                    break
        usable = keep + [item_count for item_count in must if item_count not in keep]
    aware = up_mbps is not None
    D = max(0, int(decode_steps))
    if aware:
        EL = {options: {} for options in usable}
        Eout, Ein = ({}, {})
        for options in usable:
            pf_a = _xfer_ms(prefill_bytes, up_mbps, options)
            dc_a = _xfer_ms(decode_bytes, up_mbps, options)
            Eout[options] = (1 + D) * c_out[options]
            Ein[options] = (1 + D) * c_in[options] + D * dc_a
            for b in usable:
                if options != b:
                    EL[options][b] = (1 + D) * L[options][b] + pf_a + D * dc_a

    def _score(order, alloc):
        step = predict_step_ms(
            order,
            alloc,
            L,
            c_out,
            c_in,
            layer_ms,
            up_mbps if aware else None,
            decode_bytes,
        )
        if not aware:
            return (step, step, 0.0)
        pf = predict_prefill_ms(
            order,
            alloc,
            L,
            c_out,
            c_in,
            up_mbps,
            prefill_bytes,
            prefill_chunks,
            prefill_layer_ms,
        )
        return (pf + D * step, step, pf)

    def _pin_floors(order, subset_caps):
        if order[0] not in trust or order[-1] not in trust:
            return None
        if b_in + b_out >= n_layers:
            return {} if all((item_count in trust for item_count in order)) else None
        floors = {}
        need, position = (b_in, 0)
        while need > 0 and position < len(order):
            nd = order[position]
            if nd not in trust:
                return None
            take = min(subset_caps[nd], need)
            floors[nd] = max(floors.get(nd, 0), take)
            need -= take
            position += 1
        if need > 0:
            return None
        need, row_index = (b_out, len(order) - 1)
        while need > 0 and row_index >= 0:
            nd = order[row_index]
            if nd not in trust:
                return None
            take = min(subset_caps[nd], need)
            floors[nd] = max(floors.get(nd, 0), take)
            need -= take
            row_index -= 1
        if need > 0:
            return None
        return floors

    def _pin_orders(subset, k):
        Lm, om, im = (EL, Eout, Ein) if aware else (L, c_out, c_in)
        heads = (
            [require]
            if require is not None
            else [item_count for item_count in subset if item_count in trust]
        )
        ends = {item_count for item_count in subset if item_count in trust}
        if k == 1:
            return [[h] for h in heads if h in ends]
        if k - 1 > 7:
            o = _pin_order_oversize(subset, Lm, om, im, heads, ends)[0]
            return [o] if o else []
        cand = []
        for h in heads:
            mids = [item_count for item_count in subset if item_count != h]
            for perm in permutations(mids):
                if perm[-1] not in ends:
                    continue
                order = [h, *perm]
                cand.append((loop_cost(order, Lm, om, im), order))
        cand.sort(key=lambda x: x[0])
        return [o for _, o in cand]

    def _search(k_lo, k_hi):
        found = None
        for entry_key in range(k_lo, min(k_hi, len(usable)) + 1):
            for subset in combinations(usable, entry_key):
                if require is not None and require not in subset:
                    continue
                if pin and sum((1 for item_count in subset if item_count in trust)) < (
                    1 if entry_key == 1 else 2
                ):
                    continue
                if len(set((subnet[item_count] for item_count in subset))) < entry_key:
                    continue
                subset_caps = {item_count: caps[item_count] for item_count in subset}
                if pin:
                    for order in _pin_orders(subset, entry_key):
                        floors = _pin_floors(order, subset_caps)
                        if floors is None:
                            continue
                        alloc = assign_layers(
                            order, n_layers, subset_caps, layer_ms, floors
                        )
                        if alloc is None:
                            continue
                        if (
                            not _boundary_nodes(order, alloc, n_layers, b_in, b_out)
                            <= trust
                        ):
                            continue
                        rank, step, pf = _score(order, alloc)
                        if found is None or rank < found[0]:
                            found = (rank, order, alloc, entry_key, step, pf)
                        break
                    continue
                order, _ = (
                    optimal_loop(subset, EL, Eout, Ein)
                    if aware
                    else optimal_loop(subset, L, c_out, c_in)
                )
                if require is not None and order[0] != require:
                    order = (
                        _head_first(order, require, EL, Eout, Ein)
                        if aware
                        else _head_first(order, require, L, c_out, c_in)
                    )
                alloc = assign_layers(order, n_layers, subset_caps, layer_ms)
                if alloc is None:
                    continue
                rank, step, pf = _score(order, alloc)
                if found is None or rank < found[0]:
                    found = (rank, order, alloc, entry_key, step, pf)
        return found

    kmax = k_min + slack
    best = _search(k_min, kmax)
    if best is None:
        best = _search(kmax + 1, len(usable))
    if best is None:
        return None
    rank, order, alloc, entry_key, step, pf = best
    blocks, lo = ({}, 0)
    for item_count in order:
        blocks[item_count] = (lo, lo + alloc[item_count])
        lo += alloc[item_count]
    dropped = [item_count for item_count in nodes if item_count not in order]
    spec = {
        "order": order,
        "blocks": blocks,
        "layers": alloc,
        "step_ms": round(step, 1),
        "tok_s_per_g": round(1000.0 / step, 2) if step > 0 else INF,
        "dropped": dropped,
        "k": entry_key,
    }
    b_nodes = _boundary_nodes(order, alloc, n_layers, b_in, b_out) if pin else set()
    if pin:
        spec["boundary"] = [item_count for item_count in order if item_count in b_nodes]
    if aware:
        spec["prefill_ms"] = round(pf, 1)
        spec["request_ms"] = round(rank, 1)
        if relegate:
            spec["roles"] = _relegate(
                order,
                dropped,
                caps,
                subnet,
                up_mbps,
                layer_ms,
                trust if pin else None,
                {subnet[item_count] for item_count in b_nodes},
            )
    return spec


if __name__ == "__main__":
    cities = ["WA", "OR", "CA", "TX", "KS", "IL", "GA", "NC", "VA", "NY"]
    xy = {
        "WA": (0, 9),
        "OR": (0, 7),
        "CA": (1, 3),
        "TX": (5, 1),
        "KS": (6, 5),
        "IL": (8, 6),
        "GA": (9, 2),
        "NC": (11, 3),
        "VA": (11, 5),
        "NY": (12, 8),
    }

    def base(a, b):
        (x1, y1), (x2, y2) = (xy[a], xy[b])
        return 4.0 + 2.3 * ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    n = len(cities)
    L = [[0.0] * n for _ in range(n)]
    for i, a in enumerate(cities):
        for j, b in enumerate(cities):
            if i != j:
                L[i][j] = base(a, b) + (3.0 if (i + 2 * j) % 5 == 0 else 0.0)
    L[2][8] = L[8][2] = 9.0
    c_out = [base("WA", c) for c in cities]
    c_in = [base("WA", c) * 0.9 for c in cities]
    nodes = list(range(n))
    geo = loop_cost(nodes, L, c_out, c_in)
    join = [4, 9, 2, 7, 0, 5, 8, 3, 6, 1]
    join_cost = loop_cost(join, L, c_out, c_in)
    order, cost = optimal_loop(nodes, L, c_out, c_in)
    name = lambda o: " -> ".join((cities[i] for i in o))
    print(f"arbitrary join order {join_cost:6.1f} ms   {name(join)}")
    print(f"geographic guess     {geo:6.1f} ms   {name(nodes)}")
    print(f"OPTIMAL loop         {cost:6.1f} ms   {name(order)}")
    print(
        f"  -> {join_cost / cost:.2f}x vs how nodes actually join, {geo / cost:.2f}x vs a hand geo-guess"
    )
    sub_order, sub_cost = select_and_order(nodes, L, c_out, c_in, k=6)
    print(f"best 6 of {n}          {sub_cost:6.1f} ms   {name(sub_order)}")
