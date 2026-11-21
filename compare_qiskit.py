"""Compare submission.py with Qiskit's transpiler on the benchmark corpus.

Optional (not needed by the submission):  python -m pip install qiskit
Run:  python compare_qiskit.py             (a few minutes; runs optimize on 35 x 4)

Fair comparison: each corpus circuit is transpiled by Qiskit at optimization
levels 0-3 into the same gate set (h, x, z, s, sdg, t, tdg, cx). Every Qiskit
output is checked with the organizers' equivalence function and gate set;
invalid outputs are discarded. For each circuit and weight profile Qiskit is
credited with the CHEAPEST of: the input unchanged and its valid level 0-3
outputs. Qiskit's transpiler does not take our weights, so its outputs are
scored under each profile; ours are produced with the profile passed in.
"""
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


def main():
    warnings.filterwarnings("ignore")
    circuits = bench.corpus(8)
    data = [qiskit_outputs(n, g) for _, n, g in circuits]
    print(f"{'weights':22s} {'input':>8s} {'Qiskit best':>12s} {'ours':>8s} {'ours/Qiskit':>12s}")
    for name, cfg in bench.PROFILES.items():
        tin = tq = to = 0.0
        cheaper = worse = 0
        for original, outs in data:
            q = min([circuit_cost(original, cfg)] + [circuit_cost(o, cfg) for o in outs])
            ours = optimize(original, cfg)
            assert equivalent_up_to_global_phase(original, ours)
            o = circuit_cost(ours, cfg)
            tin += circuit_cost(original, cfg)
            tq += q
            to += o
            cheaper += o < q - 1e-12
            worse += o > q + 1e-12
        k = len(data)
        print(f"{name:22s} {tin / k:8.2f} {tq / k:12.2f} {to / k:8.2f} {to / tq:12.3f}"
              f"   ours cheaper on {cheaper}/{k}, Qiskit cheaper on {worse}/{k}")


if __name__ == "__main__":
    main()
