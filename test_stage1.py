"""Stage 1 tests: self-inverse cancellation + phase-run merging.

Run:  python -m pytest test_stage1.py -q
"""

import copy
import itertools
import random

import numpy as np
import pytest

from submission import (
    ALLOWED_GATES,
    SINGLE_QUBIT_GATES,
    circuit_cost,
    equivalent_up_to_global_phase,
    optimize,
)

CFG = {"w1": 1.0, "w2": 6.0, "wd": 0.2}
SQ = sorted(SINGLE_QUBIT_GATES)  # deterministic order


def circ(n, *gates):
    """circ(2, ("CX", 0, 1), ("H", 0)) -> circuit dict."""
    return {"n_qubits": n,
            "gates": [{"gate": g, "qubits": list(q)} for g, *q in gates]}


def names(c):
    return [(g["gate"], tuple(g["qubits"])) for g in c["gates"]]


# --------------------------------------------------------------------------
# 1. Single-qubit identity table
# --------------------------------------------------------------------------
def _true_min_lengths():
    """BFS over 1-qubit gate sequences: shortest length for each unitary class
    (up to global phase). Independent reference for how short *any* rewrite
    could get; Stage 1 is not expected to reach it (sandwiches are Stage 2)."""
    def key(c):
        from submission import circuit_unitary
        u = circuit_unitary(c)
        flat = u.flatten()
        i = int(np.argmax(np.abs(flat) > 1e-9))
        u = u / (flat[i] / abs(flat[i]))
        return tuple(np.round(u.flatten(), 6).tolist())

    best = {key(circ(1)): 0}
    frontier = [[]]
    for length in range(1, 4):
        nxt = []
        for seq in frontier:
            for g in SQ:
                s = seq + [g]
                k = key(circ(1, *[(x, 0) for x in s]))
                if k not in best:
                    best[k] = length
                nxt.append(s)
        frontier = nxt
    return key, best


@pytest.mark.parametrize("length", [2, 3])
def test_single_qubit_table(length):
    key, best = _true_min_lengths()
    gap = 0
    for seq in itertools.product(SQ, repeat=length):
        c = circ(1, *[(g, 0) for g in seq])
        out = optimize(c, CFG)
        assert equivalent_up_to_global_phase(c, out), seq
        assert len(out["gates"]) <= length, seq
        gap += len(out["gates"]) - best[key(c)]
    # Informational: how far Stage 1 is from the true optimum (Stage 2 closes it).
    print(f"length {length}: total gap to optimum = {gap}")


# --------------------------------------------------------------------------
# 2. Adjacency cases
# --------------------------------------------------------------------------
def test_nested_h_cx_cx_h():
    out = optimize(circ(2, ("H", 0), ("CX", 0, 1), ("CX", 0, 1), ("H", 0)), CFG)
    assert out["gates"] == []


def test_cx_x_other_wire_cx():
    # X on qubit 2 does not touch qubits 0,1, so the CX pair still cancels.
    out = optimize(circ(3, ("CX", 0, 1), ("X", 2), ("CX", 0, 1)), CFG)
    assert names(out) == [("X", (2,))]


def test_cx_opposite_orientation_not_cancelled():
    c = circ(2, ("CX", 0, 1), ("CX", 1, 0))
    out = optimize(c, CFG)
    assert names(out) == names(c)


def test_cx_blocked_by_gate_on_shared_wire_stage1_only():
    # Stage 1 alone must NOT look past Z0 (that needs commutation).
    from submission import _stage1_cancel
    c = circ(2, ("CX", 0, 1), ("Z", 0), ("CX", 0, 1))
    assert names(_stage1_cancel(c)) == names(c)


def test_cx_blocked_by_gate_on_shared_wire_now_stage2():
    # Stage 2: Z0 commutes with the CX control, so the CX pair cancels.
    c = circ(2, ("CX", 0, 1), ("Z", 0), ("CX", 0, 1))
    assert names(optimize(c, CFG)) == [("Z", (0,))]


def test_phase_runs():
    # T Z T -> SDG (needs the run summed)
    assert names(optimize(circ(1, ("T", 0), ("Z", 0), ("T", 0)), CFG)) == [("SDG", (0,))]
    # T T -> S, S S -> Z, T TDG -> I
    assert names(optimize(circ(1, ("T", 0), ("T", 0)), CFG)) == [("S", (0,))]
    assert names(optimize(circ(1, ("S", 0), ("S", 0)), CFG)) == [("Z", (0,))]
    assert optimize(circ(1, ("T", 0), ("TDG", 0)), CFG)["gates"] == []
    # k=3 and k=5 expand to two gates, in table order
    assert [g for g, _ in names(optimize(circ(1, ("S", 0), ("T", 0)), CFG))] == ["S", "T"]
    assert [g for g, _ in names(optimize(circ(1, ("Z", 0), ("T", 0)), CFG))] == ["Z", "T"]


def test_phase_run_survives_other_wire_gates():
    out = optimize(circ(2, ("T", 0), ("H", 1), ("T", 0)), CFG)
    assert names(out) == [("S", (0,)), ("H", (1,))] or \
           names(out) == [("H", (1,)), ("S", (0,))]


def test_phase_blocked_by_h():
    c = circ(1, ("T", 0), ("H", 0), ("T", 0))
    assert names(optimize(c, CFG)) == names(c)


def test_phase_on_cx_wire_merged_across_cx_control_stage2():
    # T0 is diagonal, so it commutes with a CX control: T0 CX01 T0 -> S0 CX01.
    c = circ(2, ("T", 0), ("CX", 0, 1), ("T", 0))
    out = optimize(c, CFG)
    assert sorted(names(out)) == sorted([("S", (0,)), ("CX", (0, 1))])
    assert equivalent_up_to_global_phase(c, out)


def test_phase_on_cx_target_not_merged_across_cx():
    # T1 is on the target: it does not commute with CX, so nothing changes.
    c = circ(2, ("T", 1), ("CX", 0, 1), ("T", 1))
    assert names(optimize(c, CFG)) == names(c)


# --------------------------------------------------------------------------
# 3. Random circuits
# --------------------------------------------------------------------------
def random_circuit(rng):
    n = rng.randint(1, 6)
    gates = []
    for _ in range(rng.randint(0, 60)):
        g = rng.choice(sorted(ALLOWED_GATES))
        if g == "CX":
            if n < 2:
                g = rng.choice(SQ)
            else:
                c, t = rng.sample(range(n), 2)
                gates.append({"gate": "CX", "qubits": [c, t]})
                continue
        gates.append({"gate": g, "qubits": [rng.randrange(n)]})
    return {"n_qubits": n, "gates": gates}


def _biased_circuit(rng):
    """Fewer qubits + small gate pool => many cancellation opportunities."""
    n = rng.randint(1, 3)
    pool = ["H", "X", "T", "TDG", "S", "Z", "CX"]
    gates = []
    for _ in range(rng.randint(0, 60)):
        g = rng.choice(pool)
        if g == "CX" and n >= 2:
            c, t = rng.sample(range(n), 2)
            gates.append({"gate": "CX", "qubits": [c, t]})
        else:
            g = g if g != "CX" else "H"
            gates.append({"gate": g, "qubits": [rng.randrange(n)]})
    return {"n_qubits": n, "gates": gates}


WEIGHT_CONFIGS = [
    {"w1": 1.0, "w2": 6.0, "wd": 0.2},
    {"w1": 1.0, "w2": 1.0, "wd": 1.0},
    {"w1": 0.0, "w2": 1.0, "wd": 0.0},
    {"w1": 1.0, "w2": 0.0, "wd": 0.0},
    {"w1": 0.0, "w2": 0.0, "wd": 1.0},
    {"w1": 3.0, "w2": 10.0, "wd": 0.5},
]


@pytest.mark.parametrize("make", [random_circuit, _biased_circuit])
def test_random_circuits(make):
    # The goal: called with weights W, the output must be equivalent and no
    # costlier than the input UNDER W. Each config is passed to optimize;
    # an output is never scored under weights it was not optimized for.
    rng = random.Random(20260521)
    reduced = 0
    for _ in range(100):
        c = make(rng)
        for cfg in WEIGHT_CONFIGS:
            c_before, cfg_before = copy.deepcopy(c), copy.deepcopy(cfg)
            out = optimize(c, cfg)

            # inputs untouched
            assert c == c_before and cfg == cfg_before
            # output is a fresh object, not aliased into the input
            assert out is not c and out["gates"] is not c["gates"]
            # format
            assert set(out) == {"n_qubits", "gates"}
            assert out["n_qubits"] == c["n_qubits"]
            for g in out["gates"]:
                assert set(g) == {"gate", "qubits"} and g["gate"] in ALLOWED_GATES

            assert equivalent_up_to_global_phase(c, out)
            assert circuit_cost(out, cfg) <= circuit_cost(c, cfg) + 1e-12
            reduced += circuit_cost(out, cfg) < circuit_cost(c, cfg) - 1e-12
    # sanity: the test actually exercises cost reductions
    assert reduced > 50


# --------------------------------------------------------------------------
# 4. Idempotence
# --------------------------------------------------------------------------
def test_idempotent():
    rng = random.Random(7)
    for make in (random_circuit, _biased_circuit):
        for _ in range(500):
            c = make(rng)
            once = optimize(c, CFG)
            assert optimize(once, CFG) == once
