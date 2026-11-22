"""Benchmark + ablation harness for submission.py (not used by optimize).

Run:  python bench.py                 (larger corpus: 20 circuits per kind)
      python bench.py --quick         (the 35-circuit corpus used in README and REPORT)
      python bench.py --optimality    (also the 2- and 3-qubit optimality checks)
      python bench.py --ablation      (also remove each step in turn)
Flags combine, e.g.  python bench.py --quick --optimality --ablation

Reports, per weight profile, the mean cost of input / Stage 2 / Stage 4 and
the runtime, plus the gap to the exact optimum on small 2- and 3-qubit
circuits. --ablation removes one step at a time EVERYWHERE it runs (greedy
loop, candidate list and beam-search moves), not just its FEATURES switch.
Deterministic: fixed seeds throughout.
"""
import random
import sys
import time
from collections import Counter

import numpy as np

import stage2_baseline as stage2
import submission as s4

PROFILES = {
    "w2-heavy (1,6,0.2)": {"w1": 1.0, "w2": 6.0, "wd": 0.2},
    "balanced (1,1,1)": {"w1": 1.0, "w2": 1.0, "wd": 1.0},
    "w1-heavy (1,0,0)": {"w1": 1.0, "w2": 0.0, "wd": 0.0},
    "wd-heavy (0,0,1)": {"w1": 0.0, "w2": 0.0, "wd": 1.0},
}
TOFFOLI = [("H", (2,)), ("CX", (1, 2)), ("TDG", (2,)), ("CX", (0, 2)), ("T", (2,)),
           ("CX", (1, 2)), ("TDG", (2,)), ("CX", (0, 2)), ("T", (1,)), ("T", (2,)),
           ("H", (2,)), ("CX", (0, 1)), ("T", (0,)), ("TDG", (1,)), ("CX", (0, 1))]
INV = {"S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T"}


def _rand(rng, n, length, pool):
    out = []
    for _ in range(length):
        g = rng.choice(pool)
        if g == "CX":
            if n < 2:
                continue
            out.append(("CX", tuple(rng.sample(range(n), 2))))
        else:
            out.append((g, (rng.randrange(n),)))
    return out


def corpus(per_kind):
    rng = random.Random(2026)
    allg = sorted(s4.ALLOWED_GATES)
    kinds = {
        "general": allg,
        "H-heavy": ["H", "H", "H", "S", "T", "X", "CX"],
        "phase-heavy": ["T", "TDG", "S", "SDG", "Z", "CX", "CX", "H"],
        "CX-heavy": ["CX", "CX", "CX", "H", "T", "S"],
    }
    out = []
    for kind, pool in kinds.items():
        for _ in range(per_kind):
            n = rng.randint(2, 6)
            out.append((kind, n, _rand(rng, n, rng.randint(10, 60), pool)))
    out.append(("Toffoli", 3, TOFFOLI))
    out.append(("Toffoli x2", 3, TOFFOLI + TOFFOLI))
    r = _rand(rng, 3, 14, ["CX", "X", "Z", "S", "SDG", "T", "TDG"])
    out.append(("compute/uncompute", 4, r + [("H", (3,))] + [(INV.get(g, g), q) for g, q in reversed(r)]))
    return out


def evaluate(circuits):
    """Mean costs and runtimes per profile. The two-qubit Clifford table is
    built before timing (a one-time cost per process, reported separately)."""
    s4._cliff2_table()
    s4._z_basis_changers()
    rows = {}
    for pname, cfg in PROFILES.items():
        tin = t2 = t4 = 0.0
        secs = []
        for _, n, gates in circuits:
            c = s4._from_pairs(n, gates)
            tin += s4.circuit_cost(c, cfg)
            t2 += s4.circuit_cost(stage2.optimize(c, cfg), cfg)
            start = time.perf_counter()
            out = s4.optimize(c, cfg)
            secs.append(time.perf_counter() - start)
            t4 += s4.circuit_cost(out, cfg)
        k = len(circuits)
        secs.sort()
        rows[pname] = (tin / k, t2 / k, t4 / k, sum(secs) / k, secs[len(secs) // 2], secs[-1])
    return rows


def print_rows(title, rows):
    print(f"\n{title}")
    print(f"{'profile':22s} {'input':>8s} {'stage2':>8s} {'stage4':>8s} {'s4/in':>7s} "
          f"{'s4/s2':>7s} {'mean s':>7s} {'median s':>9s} {'max s':>7s}")
    for p, (a, b, c, mean, med, mx) in rows.items():
        print(f"{p:22s} {a:8.2f} {b:8.2f} {c:8.2f} {c / a if a else 1:7.3f} "
              f"{c / b if b else 1:7.3f} {mean:7.3f} {med:9.3f} {mx:7.3f}")
    total = sum(r[3] for r in rows.values()) / len(rows)
    print(f"{'mean over all profiles':22s} {'':44s} {total:7.3f}")


def _unitary_key(u):
    """Hashable key for a unitary up to global phase (rounded floats; used only
    for the brute-force optimum search below)."""
    flat = u.flatten()
    pivot = int(np.argmax(np.abs(flat) > 1e-9))
    u = u / (flat[pivot] / abs(flat[pivot]))
    return tuple(np.round(u.flatten(), 6).view(float).tolist())


def optimality_gap(samples=3000, max_len=4):
    """Exact minimum gate count for 2-qubit unitaries reachable in <= max_len gates."""
    gates = ([(g, (q,)) for g in sorted(s4.SINGLE_QUBIT_GATES) for q in (0, 1)]
             + [("CX", (0, 1)), ("CX", (1, 0))])
    mats = {g: s4._full_gate_matrix(2, {"gate": g[0], "qubits": list(g[1])}) for g in gates}
    best = {_unitary_key(np.eye(4, dtype=complex)): 0}
    frontier = [np.eye(4, dtype=complex)]
    for length in range(1, max_len + 1):
        nxt = []
        for u in frontier:
            for g in gates:
                v = mats[g] @ u
                k = _unitary_key(v)
                if k not in best:
                    best[k] = length
                    nxt.append(v)
        frontier = nxt
    rng = random.Random(7)
    cfg = {"w1": 1.0, "w2": 1.0, "wd": 0.0}
    gaps = Counter()
    for _ in range(samples):
        g = _rand(rng, 2, rng.randint(6, 16), sorted(s4.ALLOWED_GATES))
        k = _unitary_key(s4._fast_unitary(2, g))
        if k in best:
            out = s4.optimize(s4._from_pairs(2, g), cfg)
            gaps[len(out["gates"]) - best[k]] += 1
    total = sum(gaps.values())
    print(f"\nOptimality gap, 2-qubit circuits with optimum <= {max_len} gates "
          f"({total} cases, gate-count weights):")
    print(f"  exactly optimal: {gaps[0]}/{total} ({100 * gaps[0] / total:.1f}%);"
          f" gap distribution: {dict(sorted(gaps.items()))}")


# Each ablation step: the FEATURES switches it turns off and the beam-search
# moves it removes. A switch alone only affects the greedy loop / candidates,
# so the beam would still apply the step and the ablation would understate it.
ABLATION_STEPS = {
    "fold": (["fold"], ["fold-first", "fold-last", "fold-depth"]),
    "depth_fold": (["depth_fold"], []),
    "h_reduce": (["h_reduce"], ["h_reduce"]),
    "regions": (["regions"], ["regions"]),
    "convex_regions": (["convex_regions"], ["convex_regions"]),
    "cross_regions": ([], ["cross_regions"]),
    "blocks": (["blocks"], ["blocks"]),
    "tblocks": ([], ["tblocks"]),
    "pipeline_b": (["pipeline_b"], ["pipeline_b"]),
    "local_moves": ([], []),       # special: _local_moves returns nothing
    "beam": (["beam"], []),
}


def run_without(step, circuits):
    """evaluate() with one step removed everywhere; restores the module after."""
    features, moves = ABLATION_STEPS[step]
    saved = (dict(s4.FEATURES), s4._PASS_MOVES, s4._local_moves)
    names = {name for name, _ in s4._PASS_MOVES}
    assert all(f in s4.FEATURES for f in features) and all(m in names for m in moves), step
    try:
        for f in features:
            s4.FEATURES[f] = False
        s4._PASS_MOVES = tuple(m for m in s4._PASS_MOVES if m[0] not in moves)
        if step == "local_moves":
            s4._local_moves = lambda gates: []
        return evaluate(circuits)
    finally:
        s4.FEATURES.clear(); s4.FEATURES.update(saved[0])
        s4._PASS_MOVES, s4._local_moves = saved[1], saved[2]


def _key_batch(mats):
    """Global-phase-free keys (rounded) for a stack of unitaries, one per row."""
    flat = mats.reshape(len(mats), -1)
    pivot = np.argmax(np.abs(flat) > 1e-9, axis=1)
    phase = flat[np.arange(len(flat)), pivot]
    norm = np.round(flat / (phase / np.abs(phase))[:, None], 6) + 0.0   # +0.0: no -0.0
    return [row.tobytes() for row in norm]


def optimality_gap_3q(samples=400, at_five=200, max_len=4):
    """Exact optimality on 3-qubit circuits.

    Breadth-first search gives the minimum gate count of every 3-qubit
    unitary reachable with <= max_len gates (27 generators). Then:
      (a) random 3-qubit circuits of 5-10 gates whose unitary is in that table
          (true optimum <= 4): compare optimize()'s gate count to the optimum;
      (b) random 5-gate circuits whose unitary is NOT in the table: their
          optimum is exactly 5, so the output must have 5 gates.
    Gate-count weights (1, 1, 0). Deterministic seeds.
    """
    n = 3
    gens = ([(g, (q,)) for g in sorted(s4.SINGLE_QUBIT_GATES) for q in range(n)]
            + [("CX", (c, t)) for c in range(n) for t in range(n) if c != t])
    gmats = np.array([s4._fast_unitary(n, [g]) for g in gens])
    best = {_key_batch(np.eye(8, dtype=complex)[None])[0]: 0}
    frontier = np.eye(8, dtype=complex)[None]
    for length in range(1, max_len + 1):
        new = []
        for gm in gmats:
            cand = np.einsum("ij,njk->nik", gm, frontier)
            for k, m in zip(_key_batch(cand), cand):
                if k not in best:
                    best[k] = length
                    new.append(m)
        frontier = np.array(new)
    cfg = {"w1": 1.0, "w2": 1.0, "wd": 0.0}
    pool = sorted(s4.ALLOWED_GATES)
    rng = random.Random(33)
    gaps, gaps4 = Counter(), Counter()
    while sum(gaps.values()) < samples:
        g = _rand(rng, n, rng.randint(5, 10), pool)
        k = _key_batch(s4._fast_unitary(n, g)[None])[0]
        if k in best:
            out = s4.optimize(s4._from_pairs(n, g), cfg)
            gap = len(out["gates"]) - best[k]
            gaps[gap] += 1
            if best[k] == 4:
                gaps4[gap] += 1
    rng = random.Random(55)
    gaps5 = Counter()
    while sum(gaps5.values()) < at_five:
        g = _rand(rng, n, 5, pool)
        if len(g) == 5 and _key_batch(s4._fast_unitary(n, g)[None])[0] not in best:
            gaps5[len(s4.optimize(s4._from_pairs(n, g), cfg)["gates"]) - 5] += 1
    print(f"\nOptimality, 3-qubit circuits (gate-count weights; {len(best)} unitaries need <= {max_len} gates):")
    for label, c in ((f"optimum <= {max_len}", gaps), (f"optimum exactly {max_len}", gaps4),
                     (f"optimum exactly {max_len + 1}", gaps5)):
        t = sum(c.values())
        print(f"  {label}: exactly optimal {c[0]}/{t}; gap distribution {dict(sorted(c.items()))}")
    return gaps, gaps4, gaps5


def main():
    quick = "--quick" in sys.argv
    start = time.perf_counter()
    s4._cliff2_table()
    print(f"two-qubit Clifford table built in {time.perf_counter() - start:.2f} s (once per process)")
    circuits = corpus(8 if quick else 20)
    print(f"corpus: {len(circuits)} circuits")
    print_rows("All features on", evaluate(circuits))
    if "--optimality" in sys.argv:
        optimality_gap(3000)
        optimality_gap_3q()
    if "--ablation" in sys.argv:
        for step in ABLATION_STEPS:
            print_rows(f"Without {step} (removed everywhere)", run_without(step, circuits))
    if any(s4._BUDGET_HITS.values()):
        print("\nWARNING: work budget reached:", s4._BUDGET_HITS)


if __name__ == "__main__":
    main()
