class NgramHistory:
    def __init__(self, ng=3, max_ext=64, max_cand=48, margin=256, min_match=1):
        self.ng = ng
        self.max_ext = max_ext
        self.max_cand = max_cand

        self.min_match = min_match

        self.margin = margin
        self.indexed = 0
        self.table = {}
        self._pending = None
        self.matched = False

    def request(self, ids, k):
        self._pending = (list(ids), k)

    def fetch(self):
        ids, k = self._pending
        return self.propose(ids, k)

    def cancel(self):
        self._pending = None

    def _sync(self, seq):

        stable = len(seq) - self.margin
        if stable <= self.indexed:
            return
        ng, tbl = self.ng, self.table
        for p in range(max(self.indexed, ng), stable):
            tbl.setdefault(tuple(seq[p - ng : p]), []).append(p)
        self.indexed = stable

    def propose(self, seq, k):
        self._sync(seq)
        n = len(seq)
        self.matched = False
        if n < self.ng:
            return [seq[-1] if seq else 0] * k
        cands = self.table.get(tuple(seq[n - self.ng : n]))
        if not cands:
            return [seq[-1]] * k

        ng, me = self.ng, self.max_ext
        best_p, best_len = None, -1
        for p in cands[-self.max_cand :][::-1]:
            if p >= n:
                continue
            L = 0
            while (
                L < me
                and p - ng - 1 - L >= 0
                and seq[p - ng - 1 - L] == seq[n - ng - 1 - L]
            ):
                L += 1
            if L > best_len:
                best_len, best_p = L, p
                if L == me:
                    break
        if best_p is None:
            return [seq[-1]] * k
        self.matched = best_len >= self.min_match
        cont = seq[best_p : best_p + k]
        if len(cont) < k:
            cont = cont + [cont[-1] if cont else seq[-1]] * (k - len(cont))
        return cont


def simulate_g(seq_ids, prompt_len, ng=3, k=4):

    d = NgramHistory(ng=ng, margin=0)
    i = prompt_len
    end = len(seq_ids)
    traversals = 0
    while i < end:
        ds = d.propose(seq_ids[:i], k)
        acc = 0
        for j in range(min(k, end - i - 1)):
            if ds[j] == seq_ids[i + j]:
                acc += 1
            else:
                break
        i += acc + 1
        traversals += 1
    return (end - prompt_len) / max(traversals, 1), traversals
