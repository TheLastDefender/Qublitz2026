"""Stage 4 tests (sub-stages 4a-4f) for submission_stage4.py.

Run:  python -m pytest test_stage4.py -q
"""

import copy
import itertools
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

import submission_stage4 as s4

E = s4.equivalent_up_to_global_phase
# Default sizes keep the whole file to a few minutes. STAGE4_FULL=1 runs the
# spec's full sizes (500 circuits x 6 configs, 200 idempotence circuits).
FULL = os.environ.get("STAGE4_FULL") == "1"
CFG = {"w1": 1.0, "w2": 6.0, "wd": 0.2}
WEIGHTS = [
    {"w1": 1.0, "w2": 6.0, "wd": 0.2},
    {"w1": 1.0, "w2": 1.0, "wd": 1.0},
    {"w1": 0.0, "w2": 1.0, "wd": 0.0},
    {"w1": 1.0, "w2": 0.0, "wd": 0.0},
    {"w1": 0.0, "w2": 0.0, "wd": 1.0},
    {"w1": 3.0, "w2": 10.0, "wd": 0.5},
]
TOFFOLI = [("H", (2,)), ("CX", (1, 2)), ("TDG", (2,)), ("CX", (0, 2)), ("T", (2,)),
           ("CX", (1, 2)), ("TDG", (2,)), ("CX", (0, 2)), ("T", (1,)), ("T", (2,)),
           ("H", (2,)), ("CX", (0, 1)), ("T", (0,)), ("TDG", (1,)), ("CX", (0, 1))]


def D(n, gates):
    """(name, qubits) pairs -> circuit dict."""
    return {"n_qubits": n, "gates": [{"gate": g, "qubits": list(q)} for g, q in gates]}


def P(name, *qs):
    return (name, tuple(qs))


def rand_pairs(rng, n, length, pool):
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


ALL = sorted(s4.ALLOWED_GATES)
HFREE = ["CX", "X", "Z", "S", "SDG", "T", "TDG"]
CLIFF = ["H", "S", "SDG", "X", "Z", "CX"]


def t_count(gates):
    return sum(1 for g, _ in gates if g in ("T", "TDG"))


# =============================================================================
# 4a  H-reduction
# =============================================================================
H_RULES = [
    ([P("H", 0), P("H", 1), P("CX", 0, 1), P("H", 0), P("H", 1)], [P("CX", 1, 0)]),
    ([P("H", 0), P("S", 0), P("H", 0)], [P("SDG", 0), P("H", 0), P("SDG", 0)]),
    ([P("H", 0), P("SDG", 0), P("H", 0)], [P("S", 0), P("H", 0), P("S", 0)]),
    ([P("H", 0), P("X", 0), P("H", 0)], [P("Z", 0)]),
    ([P("H", 0), P("Z", 0), P("H", 0)], [P("X", 0)]),
    ([P("X", 0), P("CX", 0, 1), P("X", 0)], [P("CX", 0, 1), P("X", 1)]),
    ([P("Z", 1), P("CX", 0, 1), P("Z", 1)], [P("Z", 0), P("CX", 0, 1)]),
]


@pytest.mark.parametrize("lhs,rhs", H_RULES)
def test_4a_rule_identities(lhs, rhs):
    assert E(D(2, lhs), D(2, rhs))


@pytest.mark.parametrize("lhs,rhs", H_RULES)
def test_4a_pass_applies_each_rule(lhs, rhs):
    out = s4._h_reduce(lhs, CFG)
    assert out == rhs


def test_4a_random_never_more_h_equivalent_and_stable():
    rng = random.Random(41)
    for _ in range(500):
        n = rng.randint(1, 4)
        g = rand_pairs(rng, n, rng.randint(0, 40), ["H", "H", "X", "Z", "S", "SDG", "CX"])
        out = s4._h_reduce(g, CFG)
        assert s4._h_count(out) <= s4._h_count(g)
        assert E(D(n, g), D(n, out))
        assert s4._h_reduce(out, CFG) == out


# =============================================================================
# 4b  regions: exact representation and reconstruction
# =============================================================================
def test_4b_x_before_t_negates_exponent():
    rows, b, terms = s4._region_build([P("X", 0), P("T", 0)], [0])
    assert b == 1 and terms == {1: 7}           # T on (1+x0) == TDG on x0 (+ phase)


def test_4b_cx_updates_rows():
    rows, b, terms = s4._region_build([P("CX", 0, 1), P("T", 1)], [0, 1])
    assert rows == [0b01, 0b11] and terms == {0b11: 1}


def test_4b_reconstruct_roundtrip_random():
    rng = random.Random(42)
    for _ in range(1000):
        n = rng.randint(1, 4)
        g = rand_pairs(rng, n, rng.randint(0, 25), HFREE)
        wires = list(range(n))
        rows, b, terms = s4._region_build(g, wires)
        for rec in (s4._reconstruct_gauss(rows, b, terms, wires),
                    s4._reconstruct_bfs(rows, b, terms, wires)):
            assert E(D(n, g), D(n, rec))


def test_4b_canon_iff_unitary():
    rng = random.Random(43)
    # The verified counterexample: a non-empty term list that is the identity.
    ident = [P("Z", 0), P("Z", 1), P("CX", 0, 1), P("Z", 1), P("CX", 0, 1)]
    assert s4._region_build(ident, [0, 1])[2] != {}
    assert s4._region_canon(ident, [0, 1]) == s4._region_canon([], [0, 1])
    equal_pairs = unequal_pairs = 0
    for _ in range(600):
        n = rng.randint(1, 3)
        a = rand_pairs(rng, n, rng.randint(0, 8), HFREE)
        # Half the time compare with an equivalent rewrite, else a random circuit.
        b = s4._stage12(n, a) if rng.random() < 0.5 else rand_pairs(rng, n, rng.randint(0, 8), HFREE)
        same_canon = s4._region_canon(a, list(range(n))) == s4._region_canon(b, list(range(n)))
        same_unitary = E(D(n, a), D(n, b))
        assert same_canon == same_unitary
        equal_pairs += same_unitary
        unequal_pairs += not same_unitary
    assert equal_pairs > 100 and unequal_pairs > 100


def test_4b_canonical_terms_roundtrip():
    rng = random.Random(44)
    for _ in range(3000):
        m = rng.randint(1, 6)
        terms = {}
        for _ in range(rng.randint(0, 8)):
            y = rng.randrange(1, 1 << m)
            terms[y] = (terms.get(y, 0) + rng.randrange(8)) % 8
        terms = {y: c for y, c in terms.items() if c}
        table = s4._phase_table(m, terms)
        canonical = s4._canonical_terms(m, table)
        assert canonical is not None and s4._phase_table(m, canonical) == table


# =============================================================================
# 4c  phase merging + resynthesis
# =============================================================================
def test_4c_bfs_table_is_correct():
    for m in (1, 2, 3, 4):
        table = s4._linear_bfs(m)
        assert len(table) == {1: 1, 2: 6, 3: 168, 4: 20160}[m]  # |GL(m, 2)|
        for state, path in table.items():
            rows = [1 << i for i in range(m)]
            for c, t in path:
                rows[t] ^= rows[c]
            assert tuple(rows) == state


def test_4c_gauss_linear_ops_correct():
    for m in (2, 3, 4):
        for state in list(s4._linear_bfs(m))[:500]:
            rows = [1 << i for i in range(m)]
            for c, t in s4._linear_ops_gauss(list(state)):
                rows[t] ^= rows[c]
            assert tuple(rows) == state


def test_4c_merges_same_parity_t_that_stage2_misses():
    # Both halves apply T to the parity x0 XOR x1, through different CX's.
    # Stage 2 cannot merge them; the region view sums them into one S.
    g = [P("CX", 0, 1), P("T", 1), P("CX", 0, 1), P("CX", 1, 0), P("T", 0), P("CX", 1, 0)]
    s2 = s4._stage12(2, g)
    assert t_count(s2) == 2
    out = s4._region_pass(2, g, CFG)
    assert E(D(2, g), D(2, out))
    assert s4._cost_key(out, CFG) < s4._cost_key(s2, CFG)


def test_4c_region_pass_t_count_and_cost():
    rng = random.Random(45)
    for _ in range(300):
        n = rng.randint(1, 4)
        g = rand_pairs(rng, n, rng.randint(0, 30), HFREE)
        for cfg in WEIGHTS[:3]:
            out = s4._region_pass(n, g, cfg)
            assert E(D(n, g), D(n, out))
            assert s4._cost_key(out, cfg) <= s4._cost_key(g, cfg)


# =============================================================================
# 4d  cross-region simplification
# =============================================================================
def test_4d_toffoli_twice_is_identity():
    g = TOFFOLI + TOFFOLI
    stage2 = s4._stage12(3, g)
    assert len(stage2) == 27
    out = s4.optimize(D(3, g), CFG)
    assert E(D(3, g), out)
    assert len(out["gates"]) == 0


def _inverse(gates):
    inv = {"S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T"}
    return [(inv.get(g, g), q) for g, q in reversed(gates)]


def test_4d_compute_uncompute_cancels():
    rng = random.Random(46)
    for _ in range(50):
        r = rand_pairs(rng, 2, rng.randint(1, 12), HFREE)   # on wires 0,1
        g = r + [P("H", 2)] + _inverse(r)
        out = s4._full_optimize(3, g, CFG)
        assert E(D(3, g), D(3, out))
        assert out == [P("H", 2)]


def test_4d_cross_region_pass_preserves_unitary():
    rng = random.Random(47)
    for _ in range(300):
        n = rng.randint(2, 4)
        g = rand_pairs(rng, n, rng.randint(0, 40), ALL)
        out = s4._cross_region_pass(n, g, CFG)
        assert E(D(n, g), D(n, out))
        assert s4._cost_key(out, CFG) <= s4._cost_key(g, CFG)


def test_4d_h_delay_order_lengthens_regions():
    # T1 H0 T1: H0 is on another wire, so the two T1's can share one region.
    g = [P("T", 1), P("H", 0), P("T", 1)]
    assert s4._h_delay_order(g) == [P("T", 1), P("T", 1), P("H", 0)]
    assert s4._h_delay_order(g, eager=True) == [P("H", 0), P("T", 1), P("T", 1)]
    assert len(s4._region_spans(s4._region_order(g))) == 1
    # Delaying is not always better (found by this test): a leading H that a
    # later CX needs would be moved into the middle. _region_order keeps the
    # original then, so it never has more regions than the input.
    lead = [P("H", 1), P("X", 0), P("CX", 0, 1), P("T", 1)]
    assert len(s4._region_spans(s4._h_delay_order(lead))) == 2
    assert len(s4._region_spans(s4._region_order(lead))) == 1
    rng = random.Random(56)
    for _ in range(300):
        n = rng.randint(1, 5)
        g = rand_pairs(rng, n, rng.randint(0, 40), ALL)
        out = s4._region_order(g)
        assert len(s4._region_spans(out)) <= len(s4._region_spans(g))
        for q in range(n):
            assert [x for x in g if q in x[1]] == [x for x in out if q in x[1]]


def test_4d_h_delay_order_keeps_wire_order():
    rng = random.Random(48)
    for _ in range(300):
        n = rng.randint(1, 5)
        g = rand_pairs(rng, n, rng.randint(0, 40), ALL)
        out = s4._h_delay_order(g)
        assert sorted(out) == sorted(g)
        for q in range(n):
            assert [x for x in g if q in x[1]] == [x for x in out if q in x[1]]
        assert s4._metrics_pairs(out) == s4._metrics_pairs(g)


# =============================================================================
# 4e  Pipeline B (Pauli rotations)
# =============================================================================
_SIGMA = {(0, 0): np.eye(2), (1, 0): np.array([[0, 1], [1, 0]]),
          (0, 1): np.array([[1, 0], [0, -1]]), (1, 1): np.array([[0, -1j], [1j, 0]])}


def pauli_matrix(n, p):
    x, z, s = p
    m = np.array([[1]], dtype=complex)
    for q in range(n):  # qubit 0 = most significant, as in the checker
        m = np.kron(m, _SIGMA[((x >> q) & 1, (z >> q) & 1)])
    return (-1) ** s * m


def gate_matrix(n, g):
    return s4._full_gate_matrix(n, {"gate": g[0], "qubits": list(g[1])})


def all_paulis(n):
    for x in range(1 << n):
        for z in range(1 << n):
            for s in (0, 1):
                yield (x, z, s)


def test_4e_tableau_rules_match_matrices():
    n = 3
    gates = [(g, (q,)) for g in ("H", "S", "SDG", "X", "Z") for q in range(n)]
    gates += [("CX", (c, t)) for c in range(n) for t in range(n) if c != t]
    for g in gates:
        u = gate_matrix(n, g)
        for p in all_paulis(n):
            fwd = s4._pauli_forward(p, *g)
            assert np.allclose(u @ pauli_matrix(n, p) @ u.conj().T, pauli_matrix(n, fwd))
            back = s4._pauli_backward(p, *g)
            assert np.allclose(u.conj().T @ pauli_matrix(n, p) @ u, pauli_matrix(n, back))


def test_4e_pauli_commutation_matches_matrices():
    n = 2
    ps = list(all_paulis(n))
    for p in ps:
        for q in ps:
            a, b = pauli_matrix(n, p), pauli_matrix(n, q)
            assert s4._paulis_anticommute(p, q) == (not np.allclose(a @ b, b @ a))


def test_4e_conjugated_z_matches_matrix():
    rng = random.Random(49)
    for _ in range(200):
        n = rng.randint(1, 3)
        cliff = rand_pairs(rng, n, rng.randint(0, 15), CLIFF)
        q = rng.randrange(n)
        p = (0, 1 << q, 0)
        for g in reversed(cliff):
            p = s4._pauli_backward(p, *g)
        u = s4.circuit_unitary(D(n, cliff))
        assert np.allclose(u.conj().T @ pauli_matrix(n, (0, 1 << q, 0)) @ u, pauli_matrix(n, p))


def _b_without_merge(n, g):
    rotations, clifford = s4._to_rotations(g)
    out = []
    for p, k in rotations:
        out += s4._rotation_gates(n, p, k)
    return out + clifford


def test_4e_build_synthesize_without_merge():
    rng = random.Random(50)
    for _ in range(500):
        n = rng.randint(1, 4)
        g = rand_pairs(rng, n, rng.randint(0, 25), ALL)
        assert E(D(n, g), D(n, _b_without_merge(n, g)))


def test_4e_merging_preserves_unitary_and_never_adds_rotations():
    rng = random.Random(51)
    for _ in range(500):
        n = rng.randint(1, 4)
        g = rand_pairs(rng, n, rng.randint(0, 30), ALL)
        out = s4._pipeline_b(n, g)
        assert E(D(n, g), D(n, out))
        rotations = s4._to_rotations(g)[0]
        assert len(s4._merge_rotations(rotations)) <= len(rotations)


def test_4e_even_k_merge_case():
    # Two T's on the same axis merge to k=2 (a Clifford S-rotation).
    g = [P("H", 0), P("T", 0), P("H", 0), P("CX", 0, 1), P("H", 0), P("T", 0), P("H", 0)]
    out = s4._pipeline_b(2, g)
    assert E(D(2, g), D(2, out))
    assert len(s4._merge_rotations(s4._to_rotations(g)[0])) <= 2


# =============================================================================
# 4f  selection, determinism, guarantees
# =============================================================================
def _random_circuit(rng):
    n = rng.randint(1, 6)
    return D(n, rand_pairs(rng, n, rng.randint(0, 60), ALL))


def test_4f_never_worse_than_stage2_and_valid():
    import submission as stage2_module
    rng = random.Random(52)
    for _ in range(500 if FULL else 120):
        c = _random_circuit(rng)
        for cfg in WEIGHTS:
            c_copy, cfg_copy = copy.deepcopy(c), copy.deepcopy(cfg)
            out = s4.optimize(c, cfg)
            assert c == c_copy and cfg == cfg_copy
            s4.validate_circuit(out)
            assert out["n_qubits"] == c["n_qubits"]
            assert E(c, out)
            ref = stage2_module.optimize(c, cfg)
            assert s4.circuit_cost(out, cfg) <= s4.circuit_cost(ref, cfg) + 1e-12
            assert s4.circuit_cost(out, cfg) <= s4.circuit_cost(c, cfg) + 1e-12


def test_4f_idempotent():
    # optimize repeats its selection until nothing is strictly cheaper, and
    # ties keep the input, so a second call must return the same circuit.
    rng = random.Random(53)
    for _ in range(200 if FULL else 80):
        c = _random_circuit(rng)
        for cfg in (CFG, {"w1": 1.0, "w2": 0.0, "wd": 0.0}):
            once = s4.optimize(c, cfg)
            assert s4.optimize(once, cfg) == once


def test_4f_unchanged_unless_strictly_cheaper():
    # Ties keep the input: if optimize found nothing strictly cheaper (same
    # cost tuple), it must return the input exactly, not an equal-cost rewrite.
    rng = random.Random(55)
    ties = 0
    for _ in range(150):
        c = _random_circuit(rng)
        for cfg in (CFG, {"w1": 1.0, "w2": 0.0, "wd": 0.0}):
            out = s4.optimize(c, cfg)
            pairs_in, pairs_out = s4._as_pairs(c), s4._as_pairs(out)
            if s4._cost_key(pairs_out, cfg) == s4._cost_key(pairs_in, cfg):
                ties += 1
                assert out == c
    assert ties > 10  # the property is actually exercised


def test_4f_weights_change_the_choice():
    rng = random.Random(54)
    w2_heavy = {"w1": 0.0, "w2": 1.0, "wd": 0.0}
    w1_heavy = {"w1": 1.0, "w2": 0.0, "wd": 0.0}
    differs = 0
    for _ in range(100):
        c = _random_circuit(rng)
        differs += s4.optimize(c, w2_heavy) != s4.optimize(c, w1_heavy)
    assert differs > 0


_DETERMINISM_SCRIPT = r"""
import json, random, sys
sys.path.insert(0, %r)
import submission_stage4 as s4
rng = random.Random(7)
pool = sorted(s4.ALLOWED_GATES)
results = []
for _ in range(25):
    n = rng.randint(1, 6)
    gates = []
    for _ in range(rng.randint(0, 50)):
        g = rng.choice(pool)
        if g == "CX":
            if n < 2:
                continue
            gates.append({"gate": "CX", "qubits": rng.sample(range(n), 2)})
        else:
            gates.append({"gate": g, "qubits": [rng.randrange(n)]})
    c = {"n_qubits": n, "gates": gates}
    for cfg in ({"w1": 1.0, "w2": 6.0, "wd": 0.2}, {"w1": 1.0, "w2": 0.0, "wd": 0.0}):
        results.append(s4.optimize(c, cfg))
print(json.dumps(results, sort_keys=True))
"""


def test_4f_deterministic_across_processes_and_hash_seeds():
    here = os.path.dirname(os.path.abspath(__file__))
    script = _DETERMINISM_SCRIPT % here
    outputs = []
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run([sys.executable, "-c", script], env=env,
                             capture_output=True, text=True, check=True, cwd=here)
        outputs.append(res.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    # And identical to this process.
    assert json.loads(outputs[0]) == json.loads(
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       check=True, cwd=here).stdout)


# =============================================================================
# Stage 5, step A: fast verifier, strict mode, budgets
# =============================================================================
def test_A_fast_unitary_matches_official():
    rng = random.Random(60)
    for _ in range(300):
        n = rng.randint(1, 6)
        g = rand_pairs(rng, n, rng.randint(0, 40), ALL)
        assert np.allclose(s4._fast_unitary(n, g), s4.circuit_unitary(D(n, g)), atol=1e-9)


def test_A_fast_equivalent_matches_official():
    rng = random.Random(61)
    same = 0
    for _ in range(600):
        n = rng.randint(1, 4)
        a = rand_pairs(rng, n, rng.randint(0, 20), ALL)
        b = s4._stage12(n, a) if rng.random() < 0.5 else rand_pairs(rng, n, rng.randint(0, 20), ALL)
        official = E(D(n, a), D(n, b))
        assert s4._fast_equivalent(s4._fast_unitary(n, a), n, b) == official
        same += official
    assert same > 200


def test_A_strict_mode_raises_on_a_broken_pipeline(monkeypatch):
    real = s4._pipeline_b
    monkeypatch.setattr(s4, "_pipeline_b", lambda n, g: real(n, g) + [P("X", 0)])
    c = D(2, [P("H", 0), P("CX", 0, 1), P("T", 1)])
    monkeypatch.setenv("QUBLITZ_STRICT", "1")
    with pytest.raises(AssertionError):
        s4.optimize(c, CFG)
    monkeypatch.setenv("QUBLITZ_STRICT", "0")
    out = s4.optimize(c, CFG)          # non-strict: broken candidate dropped
    assert E(c, out)


def test_A_work_budgets_never_reached():
    rng = random.Random(62)
    for key in s4._BUDGET_HITS:
        s4._BUDGET_HITS[key] = 0
    for _ in range(60):
        c = _random_circuit(rng)
        for cfg in (CFG, {"w1": 1.0, "w2": 0.0, "wd": 0.0}, {"w1": 0.0, "w2": 0.0, "wd": 1.0}):
            s4.optimize(c, cfg)
    assert all(v == 0 for v in s4._BUDGET_HITS.values()), s4._BUDGET_HITS
