"""Compare submission.py with Qiskit (and PyZX, if installed) on the benchmark corpus.

Optional (not needed by the submission):  python -m pip install qiskit pyzx
Run:  python compare_qiskit.py             (a few minutes; runs optimize on 35 x 4)

Fair comparison: each corpus circuit is transpiled by Qiskit at optimization
levels 0-3 into the same gate set (h, x, z, s, sdg, t, tdg, cx), and optimized
by PyZX with its two strategies (full_optimize; full_reduce + extract). Every
output is checked with the organizers' equivalence function and gate set;
invalid outputs are discarded. For each circuit and weight profile each tool is
credited with the CHEAPEST of: the input unchanged and its valid outputs.
Neither tool takes our weights, so their outputs are scored under each profile;
ours are produced with the profile passed in.

Reported per tool: mean cost, ours / tool (ratio of mean costs), the average
per-circuit saving (mean over circuits of (tool - ours) / tool), and how often
ours is cheaper or costlier; for all 35 circuits and for the 32 random ones
(without the three textbook circuits).
"""
import os
import sys
import time
import warnings

from qiskit import QuantumCircuit, transpile

import bench
from submission import circuit_cost, equivalent_up_to_global_phase, optimize

BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "cx"]


def qiskit_outputs(n, gates):
    qc = QuantumCircuit(n)
    for g, q in gates:
        getattr(qc, g.lower())(*q)
    original = bench.s4._from_pairs(n, gates)
    outs = []
    for level in range(4):
        try:
            res = transpile(qc, basis_gates=BASIS, optimization_level=level, seed_transpiler=1)
            out = {"n_qubits": n, "gates": [
                {"gate": i.operation.name.upper(), "qubits": [res.find_bit(b).index for b in i.qubits]}
                for i in res.data]}
            if (all(g["gate"] in bench.s4.ALLOWED_GATES for g in out["gates"])
                    and equivalent_up_to_global_phase(original, out)):
                outs.append(out)
        except Exception:
            pass
    return original, outs


def pyzx_outputs(original):
    """PyZX's two strategies, or [] if PyZX is not installed."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks"))
        import compare_standard as cs
        import pyzx as zx
    except ImportError:
        return None
    circ = cs.to_pyzx(original)
    candidates = [zx.optimize.full_optimize(circ.copy())]
    g = circ.to_graph()
    zx.simplify.full_reduce(g)
    candidates.append(zx.optimize.basic_optimization(zx.extract_circuit(g.copy()).to_basic_gates()))
    outs = []
    for o in candidates:
        o = cs.from_pyzx(o)
        if (o is not None and o["n_qubits"] == original["n_qubits"]
                and equivalent_up_to_global_phase(original, o)):
            outs.append(o)
    return outs


def summary(label, rows, tool):
    """rows: (tool cost, our cost) per case."""
    saving = [100 * (t - o) / t for t, o in rows if t > 0]
    return (f"   vs {tool:6s} {label:16s} average saving per circuit {sum(saving) / len(saving):5.1f}%, "
            f"total cost {100 * (1 - sum(o for _, o in rows) / sum(t for t, _ in rows)):5.1f}%, "
            f"ours cheaper {sum(o < t - 1e-12 for t, o in rows)}/{len(rows)}, "
            f"costlier {sum(o > t + 1e-12 for t, o in rows)}")


def main():
    warnings.filterwarnings("ignore")
    circuits = bench.corpus(8)
    data = []
    for kind, n, g in circuits:
        original, qk = qiskit_outputs(n, g)
        data.append((kind, original, qk, pyzx_outputs(original)))
    with_pyzx = all(d[3] is not None for d in data)
    tools = ["Qiskit"] + (["PyZX"] if with_pyzx else [])
    if not with_pyzx:
        print("(PyZX not installed: Qiskit only)")
    rows = {t: [] for t in tools}          # (profile, random?, tool cost, our cost)
    print(f"{'weights':22s} {'input':>8s} {'Qiskit best':>12s}" + (f" {'PyZX best':>10s}" if with_pyzx else "")
          + f" {'ours':>8s} {'ours/Qiskit':>12s}" + (f" {'ours/PyZX':>10s}" if with_pyzx else ""))
    for name, cfg in bench.PROFILES.items():
        tin = to = 0.0
        tt = {t: 0.0 for t in tools}
        for kind, original, qk, pz in data:
            base = circuit_cost(original, cfg)
            ours = optimize(original, cfg)
            assert equivalent_up_to_global_phase(original, ours)
            o = circuit_cost(ours, cfg)
            tin += base
            to += o
            for t, outs in (("Qiskit", qk), ("PyZX", pz)):
                if t in tools:
                    c = min([base] + [circuit_cost(x, cfg) for x in outs])
                    tt[t] += c
                    rows[t].append((name, kind in ("general", "H-heavy", "phase-heavy", "CX-heavy"), c, o))
        k = len(data)
        print(f"{name:22s} {tin / k:8.2f} {tt['Qiskit'] / k:12.2f}" + (f" {tt['PyZX'] / k:10.2f}" if with_pyzx else "")
              + f" {to / k:8.2f} {to / tt['Qiskit']:12.3f}" + (f" {to / tt['PyZX']:10.3f}" if with_pyzx else ""))
    print()
    for t in tools:
        print(summary("all 35 circuits", [(c, o) for _, _, c, o in rows[t]], t))
        print(summary("32 random ones", [(c, o) for _, r, c, o in rows[t] if r], t))
        for name in bench.PROFILES:
            print(summary(name, [(c, o) for p, _, c, o in rows[t] if p == name], t))


if __name__ == "__main__":
    main()
