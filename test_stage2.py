"""Stage 2 tests: commutation-aware cancellation and single-wire window rewrites.

Run:  python -m pytest test_stage2.py -q
"""

import copy
import random

import pytest

from submission import (
    _seq_matrix,
    _unitary_key,
    _window_table,
    _commutes,
    _stage1_cancel,
    circuit_cost,
    equivalent_up_to_global_phase,
    optimize,
)

CFG = {"w1": 1.0, "w2": 6.0, "wd": 0.2}
WEIGHTS = [
    {"w1": 1.0, "w2": 6.0, "wd": 0.2},
    {"w1": 1.0, "w2": 1.0, "wd": 1.0},
    {"w1": 0.0, "w2": 1.0, "wd": 0.0},
    {"w1": 1.0, "w2": 0.0, "wd": 0.0},
    {"w1": 0.0, "w2": 0.0, "wd": 1.0},
    {"w1": 3.0, "w2": 10.0, "wd": 0.5},
]


def circ(n, *gates):
    return {"n_qubits": n,
            "gates": [{"gate": g, "qubits": list(q)} for g, *q in gates]}


def names(c):
    return [(g["gate"], tuple(g["qubits"])) for g in c["gates"]]


def unitary_equal(c1, c2):
    return equivalent_up_to_global_phase(c1, c2)


# --------------------------------------------------------------------------
# Commutation rules: each claimed rule is checked against the exact unitary.
# --------------------------------------------------------------------------
COMMUTE_YES = [
    (("Z", (0,)), ("CX", (0, 1))),     # diagonal on control
    (("S", (0,)), ("CX", (0, 1))),
    (("SDG", (0,)), ("CX", (0, 1))),
    (("T", (0,)), ("CX", (0, 1))),
    (("TDG", (0,)), ("CX", (0, 1))),
    (("X", (1,)), ("CX", (0, 1))),     # X on target
    (("CX", (0, 1)), ("CX", (0, 2))),  # shared control
    (("CX", (0, 2)), ("CX", (1, 2))),  # shared target
    (("H", (2,)), ("CX", (0, 1))),     # disjoint wires
]
COMMUTE_NO = [
    (("X", (0,)), ("CX", (0, 1))),     # X on control
    (("Z", (1,)), ("CX", (0, 1))),     # Z on target
    (("H", (0,)), ("CX", (0, 1))),
    (("CX", (0, 1)), ("CX", (1, 0))),  # reversed orientation
]


@pytest.mark.parametrize("a,b", COMMUTE_YES)
def test_commutation_rules_hold(a, b):
    assert _commutes(a, b)


@pytest.mark.parametrize("a,b", COMMUTE_NO)
def test_non_commutation_rules_hold(a, b):
    assert not _commutes(a, b)


# --------------------------------------------------------------------------
# Single-wire windows: validity and table quality.
# --------------------------------------------------------------------------
def _key(seq):
    return _unitary_key(_seq_matrix(seq))


SANDWICHES = {
    ("H", "X", "H"): ["Z"],
    ("H", "Z", "H"): ["X"],
    ("X", "T", "X"): ["TDG"],
    ("X", "S", "X"): ["SDG"],
    ("T", "T", "T"): ["S", "T"],
}


@pytest.mark.parametrize("seq,expected", SANDWICHES.items())
def test_sandwich_table_entries(seq, expected):
    assert _window_table()[_key(list(seq))] == expected


def test_window_table_entries_are_equivalent_and_not_longer():
    """Every table entry must equal its key, and be no longer than the sequence
    that produced it (checked with the official equivalence function)."""
    table = _window_table()
    from itertools import product
    singles = ["H", "X", "Z", "S", "SDG", "T", "TDG"]
    for length in range(1, 5):
        for seq in product(singles, repeat=length):
            best = table[_key(list(seq))]
            assert len(best) <= length
            c_in = circ(1, *[(g, 0) for g in seq])
            c_out = circ(1, *[(g, 0) for g in best])
            assert equivalent_up_to_global_phase(c_in, c_out), seq


# --------------------------------------------------------------------------
# Worked cases.
# --------------------------------------------------------------------------
def test_pdf_worked_example():
    # Guidelines section 5: CX Z CX -> Z, cost 13.6 -> 1.2 under example weights.
    c = circ(2, ("CX", 0, 1), ("Z", 0), ("CX", 0, 1))
    out = optimize(c, CFG)
    assert names(out) == [("Z", (0,))]
    assert circuit_cost(out, CFG) == pytest.approx(1.2)


def test_cx_x_target_cx_cancels():
    # X1 commutes with the CX target, so the CX pair cancels around it.
    out = optimize(circ(2, ("CX", 0, 1), ("X", 1), ("CX", 0, 1)), CFG)
    assert names(out) == [("X", (1,))]


def test_cx_x_control_cx_not_wrongly_cancelled():
    # X on the control does NOT commute with CX, so dropping the CX pair and
    # keeping X0 alone would be wrong. The equivalence check catches that.
    # (A correct rewrite exists: CX01 X0 CX01 = X0 X1; it is allowed.)
    c = circ(2, ("CX", 0, 1), ("X", 0), ("CX", 0, 1))
    out = optimize(c, CFG)
    assert unitary_equal(c, out)
    assert names(out) != [("X", (0,))]
    assert circuit_cost(out, CFG) <= circuit_cost(c, CFG)


def test_cx_z_target_cx_not_wrongly_cancelled():
    # Z on the target does NOT commute with CX (CX01 Z1 CX01 = Z0 Z1).
    c = circ(2, ("CX", 0, 1), ("Z", 1), ("CX", 0, 1))
    out = optimize(c, CFG)
    assert unitary_equal(c, out)
    assert names(out) != [("Z", (1,))]
    assert circuit_cost(out, CFG) <= circuit_cost(c, CFG)


def test_cx_shared_control_cancels_through_other_cx():
    # CX01 and CX02 share a control, so CX01 can slide past CX02.
    c = circ(3, ("CX", 0, 1), ("CX", 0, 2), ("CX", 0, 1))
    out = optimize(c, CFG)
    assert unitary_equal(c, out)
    assert names(out) == [("CX", (0, 2))]


def test_sandwich_h_x_h_becomes_z():
    assert names(optimize(circ(1, ("H", 0), ("X", 0), ("H", 0)), CFG)) == [("Z", (0,))]


def test_x_t_x_becomes_tdg_inside_larger_circuit():
    c = circ(2, ("CX", 0, 1), ("X", 0), ("T", 0), ("X", 0), ("CX", 1, 0))
    out = optimize(c, CFG)
    assert unitary_equal(c, out)
    assert circuit_cost(out, CFG) < circuit_cost(c, CFG)


# --------------------------------------------------------------------------
# Random circuits, including CX-heavy and diagonal-heavy generators that
# actually exercise commutation.
# --------------------------------------------------------------------------
def _random(rng, pool, max_n, max_len):
    n = rng.randint(1, max_n)
    gates = []
    for _ in range(rng.randint(0, max_len)):
        g = rng.choice(pool)
        if g == "CX":
            if n < 2:
                continue
            c, t = rng.sample(range(n), 2)
            gates.append({"gate": "CX", "qubits": [c, t]})
        else:
            gates.append({"gate": g, "qubits": [rng.randrange(n)]})
    return {"n_qubits": n, "gates": gates}


POOLS = [
    (["H", "X", "Z", "S", "SDG", "T", "TDG", "CX"], 6, 60),
    (["Z", "S", "SDG", "T", "TDG", "CX", "X"], 3, 60),  # commutation-heavy
    (["H", "X", "T", "TDG", "CX"], 2, 60),              # sandwich-heavy
]


@pytest.mark.parametrize("pool,max_n,max_len", POOLS)
def test_random_circuits(pool, max_n, max_len):
    # Each config is passed to optimize, and the output is judged only under
    # that config: equivalent, no costlier than the input, and no costlier
    # than Stage 1 alone.
    rng = random.Random(20260601 + max_n)
    shrunk = 0
    for _ in range(60):
        c = _random(rng, pool, max_n, max_len)
        s1 = _stage1_cancel(c)
        for cfg in WEIGHTS:
            c_copy, cfg_copy = copy.deepcopy(c), copy.deepcopy(cfg)
            out = optimize(c, cfg)

            assert c == c_copy and cfg == cfg_copy
            assert out is not c and out["gates"] is not c["gates"]
            assert set(out) == {"n_qubits", "gates"} and out["n_qubits"] == c["n_qubits"]
            for g in out["gates"]:
                assert set(g) == {"gate", "qubits"}

            assert unitary_equal(c, out)
            assert circuit_cost(out, cfg) <= circuit_cost(c, cfg) + 1e-12
            assert circuit_cost(out, cfg) <= circuit_cost(s1, cfg) + 1e-12
            shrunk += circuit_cost(out, cfg) < circuit_cost(c, cfg) - 1e-12
    assert shrunk > 30


def test_idempotent_stage2_pools():
    rng = random.Random(99)
    for pool, max_n, max_len in POOLS:
        for _ in range(150):
            c = _random(rng, pool, max_n, max_len)
            once = optimize(c, CFG)
            assert optimize(once, CFG) == once
