"""Compare submission.py with PyZX and Qiskit on standard benchmark circuits.

Optional tooling, not used by the submission. Needs:  python -m pip install pyzx qiskit

    python benchmarks/compare_standard.py                 # compare on benchmarks/circuits/
    python benchmarks/compare_standard.py --scan DIR      # list every circuit in DIR (e.g. a
                                                          # clone of PyZX's circuits/ folder)
                                                          # that fits the challenge

"Fits the challenge" = at most 6 qubits and exactly expressible in the gate set
{H, X, Z, S, SDG, T, TDG, CX} (every phase a multiple of pi/4). Each conversion
is checked against PyZX's own tensor of the original circuit.

Fairness: for each circuit and weight profile, PyZX is credited with the
cheapest of its two strategies (full_optimize; full_reduce + extract +
basic_optimization) and the unchanged input; Qiskit with the cheapest of its
optimization levels 0-3 and the input. Every output from every tool is checked
with the organizers' equivalence function; invalid outputs are discarded.
Neither tool takes our weights, so their outputs are scored under each profile;
ours is produced with the profile passed in.
"""
import os
import sys
import time
import warnings
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
warnings.filterwarnings("ignore")

import pyzx as zx                                    # noqa: E402
from qiskit import QuantumCircuit, transpile         # noqa: E402

import bench                                         # noqa: E402
from submission import (_PHASE_OUT, circuit_cost, circuit_metrics,  # noqa: E402
                        equivalent_up_to_global_phase as equivalent, optimize)

BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "cx"]
CIRCUITS = ["tof_3_before", "barenco_tof_3_before", "mod5_4_before", "qft_4.qc"]
# Published optimized versions of the same circuits, as distributed in PyZX's
# circuits/ folder: Nam et al. 2018 (light and heavy), T-par (Amy, Maslov &
# Mosca 2014), Feynman, and PyZX's own stored result. Each is scored only if it
# is equivalent to the input under the organizers' check.
PUBLISHED = {
    "tof_3_before": [("Nam light", "tof_3_after_light"), ("Nam heavy", "tof_3_after_heavy"),
                     ("T-par", "tof_3_tpar.qc"), ("Feynman", "tof_3_feynman.qc"), ("PyZX stored", "tof_3_pyzx.qc")],
    "barenco_tof_3_before": [("Nam light", "barenco_tof_3_after_light"), ("Nam heavy", "barenco_tof_3_after_heavy"),
                             ("T-par", "barenco_tof_3_tpar.qc"), ("Feynman", "barenco_tof_3_feynman.qc"),
                             ("PyZX stored", "barenco_tof_3_pyzx.qc")],
    "mod5_4_before": [("Nam light", "mod5_4_after_light"), ("Nam heavy", "mod5_4_after_heavy"),
                      ("T-par", "mod5_4_tpar.qc"), ("Feynman", "mod5_4_feynman.qc"), ("PyZX stored", "mod5_4_pyzx.qc")],
    "qft_4.qc": [("Feynman", "qft_4_feynman.qc"), ("PyZX stored", "qft_4_pyzx.qc")],
}
_FIXED_PHASE = {"Z": Fraction(1), "S": Fraction(1, 2), "T": Fraction(1, 4)}


def from_pyzx(circ):
    """PyZX circuit -> our circuit dict, or None if it does not fit the gate set."""
    circ = circ.to_basic_gates()
    gates = []
    for g in circ.gates:
        name = g.name
        if name == "HAD":
            gates.append(("H", [g.target]))
        elif name == "NOT":
            gates.append(("X", [g.target]))
        elif name == "CNOT":
            gates.append(("CX", [g.control, g.target]))
        elif name == "CZ":
            gates += [("H", [g.target]), ("CX", [g.control, g.target]), ("H", [g.target])]
        elif name == "SWAP":
            a, b = g.control, g.target
            gates += [("CX", [a, b]), ("CX", [b, a]), ("CX", [a, b])]
        elif name in ("Z", "S", "T", "ZPhase", "XPhase"):
            phase = Fraction(g.phase) if name in ("ZPhase", "XPhase") else _FIXED_PHASE[name]
            if name in ("S", "T") and getattr(g, "adjoint", False):
                phase = -phase
            k = phase * 4                              # phase in units of pi/4
            if k.denominator != 1:
                return None                            # not a multiple of pi/4
            ph = [(x, [g.target]) for x in _PHASE_OUT[int(k) % 8]]
            gates += [("H", [g.target])] + ph + [("H", [g.target])] if name == "XPhase" else ph
        else:
            return None
    return {"n_qubits": circ.qubits, "gates": [{"gate": a, "qubits": q} for a, q in gates]}


def to_pyzx(c):
    """Our circuit dict -> PyZX circuit (used to check conversions exactly)."""
    out = zx.Circuit(c["n_qubits"])
    k = {"T": Fraction(1, 4), "TDG": Fraction(-1, 4), "S": Fraction(1, 2), "SDG": Fraction(-1, 2), "Z": Fraction(1)}
    for g in c["gates"]:
        name, q = g["gate"], g["qubits"]
        if name == "H":
            out.add_gate("HAD", q[0])
        elif name == "X":
            out.add_gate("NOT", q[0])
        elif name == "CX":
            out.add_gate("CNOT", q[0], q[1])
        else:
            out.add_gate("ZPhase", q[0], phase=k[name])
    return out


def load(path):
    """Load and convert; None unless it fits the challenge and converts exactly."""
    try:
        circ = zx.Circuit.load(path)
    except Exception:
        return None
    if not 1 <= circ.qubits <= 6:
        return None
    c = from_pyzx(circ)
    if c is None or not c["gates"]:
        return None
    if not zx.compare_tensors(circ, to_pyzx(c), preserve_scalar=False):
        raise AssertionError(f"conversion of {path} changed the circuit")
    return c


def qiskit_outputs(c):
    qc = QuantumCircuit(c["n_qubits"])
    for g in c["gates"]:
        getattr(qc, g["gate"].lower())(*g["qubits"])
    outs = []
    for level in range(4):
        r = transpile(qc, basis_gates=BASIS, optimization_level=level, seed_transpiler=1)
        o = {"n_qubits": c["n_qubits"], "gates": [
            {"gate": i.operation.name.upper(), "qubits": [r.find_bit(b).index for b in i.qubits]} for i in r.data]}
        if equivalent(c, o):
            outs.append(o)
    return outs


def pyzx_outputs(path, c):
    circ = zx.Circuit.load(path).to_basic_gates()
    outs = []
    candidates = [zx.optimize.full_optimize(circ)]
    g = circ.to_graph()
    zx.simplify.full_reduce(g)
    candidates.append(zx.optimize.basic_optimization(zx.extract_circuit(g.copy()).to_basic_gates()))
    for o in candidates:
        o = from_pyzx(o)
        if o is not None and o["n_qubits"] == c["n_qubits"] and equivalent(c, o):
            outs.append(o)
    return outs


def tcount(c):
    return sum(g["gate"] in ("T", "TDG") for g in c["gates"])


def summary(c):
    m = circuit_metrics(c)
    return f"{len(c['gates'])} gates, {m['n_2q']} CX, T {tcount(c)}, depth {m['depth']}"


def scan(root):
    print("Circuits that fit the challenge (<= 6 qubits, exact Clifford+T):")
    for d, _, files in os.walk(root):
        for f in sorted(files):
            path = os.path.join(d, f)
            c = load(path)
            if c is not None:
                print(f"  {os.path.relpath(path, root)}: {c['n_qubits']} qubits, {summary(c)}")


def compare():
    directory = os.path.join(HERE, "circuits")
    wins = {"PyZX": 0, "Qiskit": 0, "published": 0}
    ties = {"published": 0}
    savings = {"PyZX": [], "Qiskit": [], "published": []}
    cases = 0
    for name in CIRCUITS:
        path = os.path.join(directory, name)
        c = load(path)
        pz, qk = pyzx_outputs(path, c), qiskit_outputs(c)
        print(f"\n=== {name}: {c['n_qubits']} qubits; input {summary(c)}")
        published = []
        for label, fname in PUBLISHED.get(name, []):
            o = load(os.path.join(directory, fname))
            ok = o is not None and o["n_qubits"] == c["n_qubits"] and equivalent(c, o)
            print(f"   published {label:12s} {summary(o) if o else '(not loadable)'}"
                  f"{'' if ok else '  [NOT equivalent to the input: excluded]'}")
            if ok:
                published.append(o)
        for label, outs in (("PyZX", pz), ("Qiskit", qk)):
            for o in outs:
                print(f"   {label:7s} output: {summary(o)}")
        print(f"   {'weights':20s} {'input':>8s} {'PyZX best':>10s} {'Qiskit best':>12s} {'published':>15s} {'ours':>8s}   ours")
        for profile, cfg in bench.PROFILES.items():
            base = circuit_cost(c, cfg)
            best_pz = min([base] + [circuit_cost(o, cfg) for o in pz])
            best_qk = min([base] + [circuit_cost(o, cfg) for o in qk])
            best_pub = min([base] + [circuit_cost(o, cfg) for o in published])
            start = time.perf_counter()
            out = optimize(c, cfg)
            secs = time.perf_counter() - start
            assert equivalent(c, out)
            ours = circuit_cost(out, cfg)
            cases += 1
            wins["PyZX"] += ours < best_pz - 1e-12
            wins["Qiskit"] += ours < best_qk - 1e-12
            for tool, best in (("PyZX", best_pz), ("Qiskit", best_qk), ("published", best_pub)):
                savings[tool].append(100 * (best - ours) / best)
            wins["published"] += ours < best_pub - 1e-12
            ties["published"] += abs(ours - best_pub) <= 1e-12
            print(f"   {profile:20s} {base:8.1f} {best_pz:10.1f} {best_qk:12.1f} {best_pub:15.1f} {ours:8.1f}   "
                  f"{summary(out)} ({secs:.1f} s)")
    print(f"\nOurs strictly cheapest: vs PyZX in {wins['PyZX']}/{cases} cases, "
          f"vs Qiskit in {wins['Qiskit']}/{cases} cases, vs the published versions in benchmarks/circuits in "
          f"{wins['published']}/{cases} cases (tied in {ties['published']}).")
    print("Average saving per circuit-and-weight case: " + ", ".join(
        f"vs {t} {sum(v) / len(v):.1f}%" for t, v in savings.items()))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--scan":
        scan(sys.argv[2])
    else:
        compare()
