"""Qublitz 2026 - Quantum Circuit Golf participant starter.

Install:  python -m pip install numpy
Try it:   python submission.py

Organizers will import `optimize(circuit, config)` from this file.
Change the body of `optimize` to implement your own algorithm.

Circuit format:
    {"n_qubits": 2, "gates": [{"gate": "H", "qubits": [0]}, ...]}

Supported gates: H, X, Z, S, SDG, T, TDG, CX.
CX qubit order: [control, target]. Qubit 0 is the top/MSB wire.
Gates execute in listed order. One through six qubits are supported.

Config format:
    {"w1": 1.0, "w2": 6.0, "wd": 0.2}

This includes a LOCAL checker, not the organizers' private tests.
"""

import copy
import functools
import math

import numpy as np


SINGLE_QUBIT_GATES = frozenset(("H", "X", "Z", "S", "SDG", "T", "TDG"))
ALLOWED_GATES = SINGLE_QUBIT_GATES | {"CX"}


def optimize(circuit, config):
    """Optimizes a quantum circuit to reduce its cost according to the given weights.

    Inputs:
        circuit: A dictionary representing the quantum circuit to be optimized.
        config: A dictionary containing the weights for one-qubit gates, two-qubit 
        gates, and circuit depth.
    """
    validate_circuit(circuit)
    validate_config(config)

    # Stage 1: per-wire stack cancellation (adjacent pairs + phase-run merging).
    # Stage 2: commutation-aware cancellation and single-wire window rewrites,
    # iterated until no rewrite shortens the circuit.
    n = circuit["n_qubits"]
    gates = _as_pairs(_stage1_cancel(circuit))
    # Stage 1 can expand a phase run into two gates with no net saving (e.g.
    # SDG,TDG -> Z,T). Stage 2 does not undo that, so re-run Stage 1 on the
    # result. Iterate until Stage 2 changes nothing: then the output is a fixed
    # point, and optimize(optimize(c)) == optimize(c).
    # Terminates: Stage 2 returns its input unchanged unless it found a strictly
    # shorter circuit, and Stage 1 never lengthens, so each extra round removes
    # at least one gate.
    while True:
        reduced = _stage2_reduce(gates)
        if reduced == gates:
            return _from_pairs(n, gates)
        gates = _as_pairs(_stage1_cancel(_from_pairs(n, reduced)))


def _as_pairs(circuit):
    return [(g["gate"], tuple(g["qubits"])) for g in circuit["gates"]]


def _from_pairs(n, gates):
    return {
        "n_qubits": n,
        "gates": [{"gate": name, "qubits": list(qubits)} for name, qubits in gates],
    }


# --- Stage 1 helpers --------------------------------------------------------
#
# Every phase gate is a power of T = diag(1, e^{i*pi/4}): gate = T^k, k mod 8.
# Powers of T on one wire commute and add, so a whole run collapses to one
# exponent. T Z T = T^(1+4+1) = T^6 = SDG. A pairwise check never sees this:
# (T,Z) and (Z,T) don't cancel on their own, and T,T isn't adjacent once Z
# sits between them. Summing the *run* is what reveals the net exponent.
_PHASE_K = {"T": 1, "S": 2, "Z": 4, "SDG": 6, "TDG": 7}

# Shortest equivalent gate list for each total exponent k mod 8.
_PHASE_OUT = {
    0: [],
    1: ["T"],
    2: ["S"],
    3: ["S", "T"],
    4: ["Z"],
    5: ["Z", "T"],
    6: ["SDG"],
    7: ["TDG"],
}


def _stage1_cancel(circuit):
    """Per-wire stack cancellation. Returns a NEW circuit dict.

    One stack per qubit. A stack entry is a tuple
        (creation_index, payload, qubits)
    where payload is a gate name ("H", "X", "CX") or, for a phase gate,
    the integer exponent k (gate = T^k). A CX entry is the *same tuple object*
    on both its control and target stacks, so `is` tells us two tops are
    one and the same gate.

    "Adjacent" = nothing between the two gates touches their qubits. That is
    exactly "the earlier gate is on top of the stack of every wire the later
    gate uses". Gates on other wires never enter those stacks, so they cannot
    block a match. Only the top needs checking: a cancellation exposes the
    entry beneath, and that is checked when the next gate arrives
    (H0 CX01 CX01 H0 -> I).
    """
    n = circuit["n_qubits"]
    stacks = [[] for _ in range(n)]

    for index, op in enumerate(circuit["gates"]):
        name = op["gate"]

        if name == "CX":
            c, t = op["qubits"]
            top_c = stacks[c][-1] if stacks[c] else None
            top_t = stacks[t][-1] if stacks[t] else None
            # Same entry on both tops AND same orientation: CX[0,1] and
            # CX[1,0] are different gates and must not cancel.
            if (top_c is not None and top_c is top_t
                    and top_c[1] == "CX" and top_c[2] == (c, t)):
                stacks[c].pop()
                stacks[t].pop()
            else:
                entry = (index, "CX", (c, t))
                stacks[c].append(entry)
                stacks[t].append(entry)
            continue

        q = op["qubits"][0]
        stack = stacks[q]
        top = stack[-1] if stack else None

        if name in _PHASE_K:
            k = _PHASE_K[name]
            if top is not None and isinstance(top[1], int):
                # Merge into the run: exponents add mod 8. The merged gate
                # keeps the earlier creation index (it holds the earlier slot).
                stack.pop()
                k = (k + top[1]) % 8
                if k != 0:
                    stack.append((top[0], k, (q,)))
                # k == 0: the run cancels entirely, push nothing.
            else:
                stack.append((index, k, (q,)))
        else:  # H or X: self-inverse
            if top is not None and top[1] == name:
                stack.pop()
            else:
                stack.append((index, name, (q,)))

    # Collect survivors. CX entries sit on two stacks: de-dup by creation index.
    survivors = {}
    for stack in stacks:
        for entry in stack:
            survivors[entry[0]] = entry

    gates = []
    for index in sorted(survivors):
        _, payload, qubits = survivors[index]
        if isinstance(payload, int):
            # Phase run: expand the exponent into its shortest gate sequence.
            for gate_name in _PHASE_OUT[payload]:
                gates.append({"gate": gate_name, "qubits": [qubits[0]]})
        else:
            gates.append({"gate": payload, "qubits": list(qubits)})

    return {"n_qubits": n, "gates": gates}


# --- Stage 2 helpers --------------------------------------------------------
#
# Stage 2 works on a flat list of (name, qubits) tuples. Two mechanisms:
#
# (a) COMMUTATION. Sequences below are in circuit (time) order, as listed.
#     If gate g commutes with every gate M between g and a
#     partner p on the same wire(s), then g M p = M g p. If p = g, that is
#     M g g = M, so both vanish. If p is a phase gate on the same wire, the
#     two phases fuse. A gate is only moved in the algebra, never in the
#     output: the output just drops g and rewrites p in place.
#     Gates on disjoint qubits always commute and are skipped.
#     `_commutes` checks each pair exactly against the unitary, so no
#     commutation rule is trusted without verification.
#
# (b) SINGLE-WIRE WINDOWS. A run of consecutive single-qubit gates on one
#     wire, of length 2..4, is replaced by the shortest gate list with the same
#     one-qubit unitary up to phase. This is the "sandwich" family: H X H -> Z,
#     X T X -> TDG, etc. The table is built by enumeration, not typed in.
#
# Termination: every accepted rewrite strictly lowers the gate count, so the
# loop always ends. Merges are accepted only when the merged phase needs at
# most one gate, so a pairwise merge never just relabels the pair.

_SINGLE_NAMES = sorted(SINGLE_QUBIT_GATES)


def _unitary_key(u):
    """Hashable key for a 2x2 or 4x4 unitary, normalized to remove global phase."""
    flat = u.flatten()
    pivot = int(np.argmax(np.abs(flat) > 1e-9))
    u = u / (flat[pivot] / abs(flat[pivot]))
    return tuple(np.round(u.flatten(), 6).view(float).tolist())


def _build_window_table(max_len=4):
    """Shortest one-qubit gate list for each unitary class, lengths 0..max_len."""
    table = {_unitary_key(np.eye(2, dtype=complex)): []}
    frontier = [[]]
    for _ in range(max_len):
        next_frontier = []
        for seq in frontier:
            for name in _SINGLE_NAMES:
                new_seq = seq + [name]
                key = _unitary_key(_seq_matrix(new_seq))
                if key not in table:
                    table[key] = new_seq
                    next_frontier.append(new_seq)
        frontier = next_frontier
    return table


def _seq_matrix(names):
    """Product matrix for a sequence of one-qubit gate names (first acts first)."""
    u = np.eye(2, dtype=complex)
    for name in names:
        u = _ONE_QUBIT_MATRICES[name] @ u
    return u


_WINDOW_TABLE = None


def _window_table():
    """Build the window table on first use (it needs _ONE_QUBIT_MATRICES, defined below)."""
    global _WINDOW_TABLE
    if _WINDOW_TABLE is None:
        _WINDOW_TABLE = _build_window_table(4)
    return _WINDOW_TABLE


def _commutes(first, second):
    """Exact check: do two gates commute as operators on their union of qubits?"""
    union = sorted(set(first[1]) | set(second[1]))
    index = {q: i for i, q in enumerate(union)}
    return _commute_relabelled(
        first[0], tuple(index[q] for q in first[1]),
        second[0], tuple(index[q] for q in second[1]),
        len(union))


@functools.lru_cache(maxsize=None)
def _commute_relabelled(name_a, qubits_a, name_b, qubits_b, m):
    """Cached matrix commutator test on m local qubits (depends only on the pattern)."""
    ua = _full_gate_matrix(m, {"gate": name_a, "qubits": list(qubits_a)})
    ub = _full_gate_matrix(m, {"gate": name_b, "qubits": list(qubits_b)})
    return bool(np.allclose(ua @ ub, ub @ ua, atol=1e-9, rtol=0.0))


def _is_phase(name):
    return name in _PHASE_K


def _commute_cancel_pass(gates):
    """One sweep: each gate looks forward for a partner it can reach.

    Returns a new list. Only ever shorter or equal in length.
    """
    cur = list(gates)
    alive = [True] * len(cur)
    changed = False
    for i in range(len(cur)):
        if not alive[i]:
            continue
        gi = cur[i]
        for j in range(i + 1, len(cur)):
            if not alive[j]:
                continue
            gj = cur[j]
            if not set(gi[1]) & set(gj[1]):
                continue  # disjoint gates commute trivially
            # Partner: self-inverse cancellation.
            if gi == gj and gi[0] in ("H", "X", "CX"):
                alive[i] = alive[j] = False
                changed = True
                break
            # Partner: phase merge on the same single wire.
            if (_is_phase(gi[0]) and _is_phase(gj[0]) and gi[1] == gj[1]):
                k = (_PHASE_K[gi[0]] + _PHASE_K[gj[0]]) % 8
                out = _PHASE_OUT[k]
                if len(out) <= 1:  # only merge when it strictly shortens
                    alive[i] = False
                    changed = True
                    if out:
                        cur[j] = (out[0], gj[1])
                    else:
                        alive[j] = False
                    break
            # Otherwise gj is a blocker unless it commutes with gi.
            if not _commutes(gi, gj):
                break
    return [cur[k] for k in range(len(cur)) if alive[k]] if changed else list(gates)


def _window_pass(gates):
    """Replace the first improvable one-qubit window. Returns (gates, changed)."""
    n_gates = len(gates)
    for q in sorted({w for _, qs in gates for w in qs}):
        positions = [k for k in range(n_gates) if q in gates[k][1]]
        # Walk maximal runs of single-qubit gates on this wire.
        run = []
        runs = []
        for k in positions:
            name, qs = gates[k]
            if len(qs) == 1:
                run.append(k)
            else:
                if run:
                    runs.append(run)
                run = []
        if run:
            runs.append(run)
        for run in runs:
            for length in (4, 3, 2):
                for s in range(len(run) - length + 1):
                    window = run[s:s + length]
                    seq = [gates[k][0] for k in window]
                    best = _window_table()[_unitary_key(_seq_matrix(seq))]
                    if len(best) < length:
                        new = [(name, (q,)) for name in best]
                        at = window[0]
                        out = []
                        for k in range(n_gates):
                            if k == at:
                                out.extend(new)
                            elif k not in window:
                                out.append(gates[k])
                        return out, True
    return list(gates), False


def _stage2_reduce(gates):
    """Apply commutation cancellation and window rewrites until neither shortens."""
    gates = list(gates)
    while True:
        shorter = _commute_cancel_pass(gates)
        if len(shorter) < len(gates):
            gates = shorter
            continue
        shorter, changed = _window_pass(gates)
        if changed and len(shorter) < len(gates):
            gates = shorter
            continue
        return gates


def validate_circuit(circuit):
    """Raise ValueError if a circuit violates the public format."""
    if not isinstance(circuit, dict) or set(circuit) != {"n_qubits", "gates"}:
        raise ValueError("Circuit must have exactly 'n_qubits' and 'gates'.")

    n = circuit["n_qubits"]
    if type(n) is not int or not 1 <= n <= 6:
        raise ValueError("n_qubits must be an integer from 1 through 6.")

    gates = circuit["gates"]
    if not isinstance(gates, list):
        raise ValueError("gates must be a list.")

    for i, operation in enumerate(gates):
        if not isinstance(operation, dict) or set(operation) != {"gate", "qubits"}:
            raise ValueError(f"Gate {i}: expected exactly 'gate' and 'qubits'.")

        name = operation["gate"]
        qubits = operation["qubits"]
        if not isinstance(name, str) or name not in ALLOWED_GATES:
            raise ValueError(f"Gate {i}: unsupported gate {name!r}.")
        if not isinstance(qubits, list):
            raise ValueError(f"Gate {i}: qubits must be a list.")

        arity = 2 if name == "CX" else 1
        if len(qubits) != arity:
            raise ValueError(f"Gate {i}: {name} requires {arity} qubit(s).")
        if any(type(q) is not int or not 0 <= q < n for q in qubits):
            raise ValueError(f"Gate {i}: invalid qubit index.")
        if len(set(qubits)) != len(qubits):
            raise ValueError(f"Gate {i}: qubit indices must be distinct.")

    return True


def validate_config(config):
    """Check nonnegative finite resource weights w1, w2, wd."""
    if not isinstance(config, dict) or set(config) != {"w1", "w2", "wd"}:
        raise ValueError("config must have exactly 'w1', 'w2', and 'wd'.")

    for key in ("w1", "w2", "wd"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a real number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")

    if not any(config[key] > 0 for key in ("w1", "w2", "wd")):
        raise ValueError("At least one weight must be positive.")

    return True


_ROOT2 = math.sqrt(2)
_ONE_QUBIT_MATRICES = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / _ROOT2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "S": np.diag([1, 1j]).astype(complex),
    "SDG": np.diag([1, -1j]).astype(complex),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex),
    "TDG": np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex),
}


def _full_gate_matrix(n, operation):
    """Matrix for one gate, with qubit 0 as the most significant bit."""
    name = operation["gate"]
    qubits = operation["qubits"]
    if name != "CX":
        result = np.array([[1]], dtype=complex)
        for q in range(n):
            factor = (_ONE_QUBIT_MATRICES[name]
                      if q == qubits[0] else np.eye(2, dtype=complex))
            result = np.kron(result, factor)
        return result

    control, target = qubits
    dimension = 1 << n
    result = np.zeros((dimension, dimension), dtype=complex)
    control_bit = 1 << (n - 1 - control)
    target_bit = 1 << (n - 1 - target)
    for input_state in range(dimension):
        output_state = (input_state ^ target_bit
                        if input_state & control_bit else input_state)
        result[output_state, input_state] = 1
    return result


def circuit_unitary(circuit):
    """Calculate the complete unitary for a supported 1-6 qubit circuit.

    Earlier listed gates act first: [A, B] has unitary U_B @ U_A.
    """
    validate_circuit(circuit)
    n = circuit["n_qubits"]
    result = np.eye(1 << n, dtype=complex)
    for operation in circuit["gates"]:
        result = _full_gate_matrix(n, operation) @ result
    return result


def equivalent_up_to_global_phase(first, second, atol=1e-9):
    """Return True if both circuits implement the same unitary up to phase."""
    validate_circuit(first)
    validate_circuit(second)
    if first["n_qubits"] != second["n_qubits"]:
        return False

    a = circuit_unitary(first)
    b = circuit_unitary(second)
    pivot = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[pivot] / a[pivot]
    return (np.isclose(abs(phase), 1.0, atol=atol, rtol=0.0)
            and np.allclose(b, phase * a, atol=atol, rtol=0.0))


def circuit_metrics(circuit):
    """Count gates and minimum depth, scheduling disjoint gates in parallel.

    Every gate takes one time unit. Gates sharing a qubit are serialized.
    """
    validate_circuit(circuit)
    available = [0] * circuit["n_qubits"]
    one_qubit = 0
    two_qubit = 0

    for operation in circuit["gates"]:
        qubits = operation["qubits"]
        end = 1 + max(available[q] for q in qubits)
        for q in qubits:
            available[q] = end
        if operation["gate"] == "CX":
            two_qubit += 1
        else:
            one_qubit += 1

    return {"n_1q": one_qubit, "n_2q": two_qubit, "depth": max(available)}


def circuit_cost(circuit, config):
    """Weighted cost C = w1*N_1q + w2*N_2q + wd*depth."""
    validate_config(config)
    metrics = circuit_metrics(circuit)
    return (config["w1"] * metrics["n_1q"]
            + config["w2"] * metrics["n_2q"]
            + config["wd"] * metrics["depth"])


def evaluate_optimization(original, optimized, config):
    """Return local correctness and resource comparisons for one input."""
    equivalent = equivalent_up_to_global_phase(original, optimized)
    before = circuit_cost(original, config)
    after = circuit_cost(optimized, config)
    return {
        "equivalent": equivalent,
        "input_metrics": circuit_metrics(original),
        "output_metrics": circuit_metrics(optimized),
        "input_cost": before,
        "output_cost": after,
        "cost_reduction": before - after,
    }


if __name__ == "__main__":
    example = {
        "n_qubits": 2,
        "gates": [
            {"gate": "CX", "qubits": [0, 1]},
            {"gate": "Z", "qubits": [0]},
            {"gate": "CX", "qubits": [0, 1]},
        ],
    }
    weights = {"w1": 1.0, "w2": 6.0, "wd": 0.2}
    result = optimize(example, weights)
    print("Your optimizer output:")
    print(result)
    print("Local assessment:")
    print(evaluate_optimization(example, result, weights))

    # A known equivalent circuit to demonstrate the checker:
    expected = {"n_qubits": 2, "gates": [{"gate": "Z", "qubits": [0]}]}
    print("Known simplification (CX-Z-CX -> Z):")
    print(evaluate_optimization(example, expected, weights))