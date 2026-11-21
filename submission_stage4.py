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
import os

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

    # Stage 4 (4f): build several candidates, FULLY optimize each with the same
    # convergent loop, verify each against the input unitary, then pick the
    # cheapest under `config` (ties broken by a total order, see _key).
    n = circuit["n_qubits"]
    original = _as_pairs(circuit)
    reference = _fast_unitary(n, original)   # input unitary, computed once
    current = original
    # Repeat the selection from its own result until it returns its input.
    # That makes optimize(optimize(c)) == optimize(c) exactly: the second call
    # starts from a fixed point (unless the MAX_ROUNDS budget ran out).
    for _ in range(MAX_ROUNDS):
        best = _select(n, reference, current, config)
        if best == current:
            break
        current = best
    else:
        _BUDGET_HITS["select"] += 1
    result = _from_pairs(n, current)
    # Final check with the OFFICIAL function. Every rewrite is an exact
    # identity, so this should never fail; if it does, that is a bug: raise
    # in strict mode (tests), otherwise return the input unchanged.
    if not equivalent_up_to_global_phase(circuit, result):
        if _strict():
            raise AssertionError("optimize produced an inequivalent circuit")
        return copy.deepcopy(circuit)
    return result


def _select(n, reference, start, config):
    """One round of 4f: fully optimize every candidate, verify, pick cheapest."""
    s2 = _stage12(n, start)                                # Stages 1-2 only
    candidates = [start, s2]
    a = _full_optimize(n, s2, config)                      # Pipeline A
    candidates.append(a)
    if FEATURES["pipeline_b"]:
        candidates.append(_full_optimize(n, _pipeline_b(n, s2), config))  # B
        candidates.append(_full_optimize(n, _pipeline_b(n, a), config))   # B after A
    # Each candidate is checked against the ORIGINAL input's unitary. Every
    # pipeline is built from exact identities, so a rejection means a bug:
    # strict mode (used by the tests) raises; otherwise the candidate is
    # dropped. `start` is the input or an earlier verified result.
    verified = []
    for i, c in enumerate(candidates):
        if _fast_equivalent(reference, n, c):
            verified.append((i, c))
        elif _strict():
            raise AssertionError(f"candidate {i} failed verification")
    # Ties on cost go to the EARLIEST candidate (start first): the circuit is
    # only changed when something is strictly cheaper. Still a total order.
    return min(verified, key=lambda ic: (_cost_key(ic[1], config), ic[0]))[1]


def _stage12(n, gates):
    """Stages 1-2 iterated to a fixed point (exactly the Stage 2 `optimize`).

    Terminates: Stage 2 returns its input unchanged unless it found a strictly
    shorter circuit, and Stage 1 never lengthens, so each round removes a gate.
    """
    gates = _as_pairs(_stage1_cancel(_from_pairs(n, gates)))
    while True:
        reduced = _stage2_reduce(gates)
        if reduced == gates:
            return gates
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


# =============================================================================
# Stage 4: region resynthesis + Pauli-rotation alternative
# =============================================================================
#
# Determinism: no randomness, no wall-clock limits (work budgets below), no
# iteration over sets to produce output, every tie broken explicitly. All
# decisions use exact integers: GF(2) rows as int bitmasks, phase exponents
# mod 8, Pauli strings as (x bits, z bits, sign bit). Floats appear only in
# the final unitary check in `optimize`.
#
# Exactness: every rewrite is an identity up to global phase. Region rewrites
# are checked exactly with `_region_canon` (integer arithmetic, proven iff);
# whole candidates are checked against the input unitary in `optimize`.

# Work budgets (deterministic: counts, never time). Termination never depends
# on them: every loop strictly lowers a well-ordered cost tuple. They bound
# runtime; `_BUDGET_HITS` records any loop that stopped because of a cap, and
# a test asserts that this never happens on the test corpus.
MAX_ROUNDS = 8          # rounds of the full loop per candidate (work budget)
MAX_CROSS_MOVES = 64    # accepted cross-region moves per round (work budget)
BFS_MAX_WIRES = 4       # exact CX-optimal linear synthesis up to this width
_REGION_GATES = frozenset(("CX", "X", "Z", "S", "SDG", "T", "TDG"))
_BUDGET_HITS = {"select": 0, "full": 0, "cross": 0, "beam": 0}

# Feature switches, used only by the ablation harness (bench.py) to measure
# each component's contribution. All are on in normal use.
FEATURES = {
    "h_reduce": True,
    "regions": True,
    "cross_regions": True,
    "pipeline_b": True,
    "fold": True,
    "blocks": True,
}


def _strict():
    """Strict mode (QUBLITZ_STRICT=1, set by the tests): fail loudly on bugs."""
    return os.environ.get("QUBLITZ_STRICT") == "1"


# --- fast exact verifier ------------------------------------------------------

def _fast_unitary(n, gates):
    """Circuit unitary by applying each gate to the matrix's rows directly.

    Same convention as circuit_unitary (qubit 0 = most significant bit, first
    gate acts first) but without building a 2^n x 2^n Kronecker product per
    gate: CX permutes rows, a one-qubit gate mixes pairs of rows.
    """
    dim = 1 << n
    u = np.eye(dim, dtype=complex)
    index = np.arange(dim)
    for name, qubits in gates:
        if name == "CX":
            cbit, tbit = 1 << (n - 1 - qubits[0]), 1 << (n - 1 - qubits[1])
            u = u[np.where(index & cbit, index ^ tbit, index)]
        else:
            q = qubits[0]
            u = u.reshape(1 << q, 2, dim >> (q + 1), dim)
            u = np.einsum("ab,ibjk->iajk", _ONE_QUBIT_MATRICES[name], u).reshape(dim, dim)
    return u


def _fast_equivalent(reference, n, gates, atol=1e-9):
    """Same global-phase test as equivalent_up_to_global_phase."""
    b = _fast_unitary(n, gates)
    a = reference
    pivot = np.unravel_index(np.argmax(np.abs(a)), a.shape)
    phase = b[pivot] / a[pivot]
    return bool(np.isclose(abs(phase), 1.0, atol=atol, rtol=0.0)
                and np.allclose(b, phase * a, atol=atol, rtol=0.0))


# --- cost and total order ----------------------------------------------------

def _metrics_pairs(gates):
    """Same semantics as circuit_metrics, on (name, qubits) tuples."""
    avail = {}
    n1 = n2 = 0
    for name, qubits in gates:
        end = 1 + max(avail.get(q, 0) for q in qubits)
        for q in qubits:
            avail[q] = end
        if name == "CX":
            n2 += 1
        else:
            n1 += 1
    return n1, n2, max(avail.values(), default=0)


def _cost_key(gates, config):
    """(cost, n_2q, depth, n_1q). Same arithmetic order as circuit_cost."""
    n1, n2, depth = _metrics_pairs(gates)
    cost = config["w1"] * n1 + config["w2"] * n2 + config["wd"] * depth
    return (cost, n2, depth, n1)


def _key(gates, config):
    """Total order on candidates: cost tuple, then the gate list itself."""
    return _cost_key(gates, config) + (tuple(gates),)


def _prev_on(gates, k, q):
    for i in range(k - 1, -1, -1):
        if q in gates[i][1]:
            return i
    return None


def _next_on(gates, k, q):
    for i in range(k + 1, len(gates)):
        if q in gates[i][1]:
            return i
    return None


# --- 4a: H-reduction -----------------------------------------------------------
#
# Rules (each verified in tests):
#   H_c H_t CX(c,t) H_c H_t -> CX(t,c)        H count -4
#   H S H -> SDG H SDG,  H SDG H -> S H S      H count -1 (same gate count)
#   H X H -> Z,  H Z H -> X                    H count -2
#   X_c CX(c,t) X_c -> CX(c,t) X_t            Pauli propagation, -1 gate
#   Z_t CX(c,t) Z_t -> Z_c CX(c,t)            Pauli propagation, -1 gate
# A rewrite is accepted only if (H count, cost tuple) strictly decreases,
# which guarantees termination.

def _h_count(gates):
    return sum(1 for name, _ in gates if name == "H")


def _h_rules_at(gates, k):
    """Candidate rewrites anchored at gate k, in a fixed order."""
    name, qubits = gates[k]
    out = []
    if name == "CX":
        c, t = qubits
        pc, pt = _prev_on(gates, k, c), _prev_on(gates, k, t)
        nc, nt = _next_on(gates, k, c), _next_on(gates, k, t)
        h_c, h_t = ("H", (c,)), ("H", (t,))
        if (None not in (pc, pt, nc, nt) and gates[pc] == h_c and gates[nc] == h_c
                and gates[pt] == h_t and gates[nt] == h_t):
            drop = {pc, pt, nc, nt}
            out.append([("CX", (t, c)) if i == k else g
                        for i, g in enumerate(gates) if i not in drop])
        x_c = ("X", (c,))
        if pc is not None and nc is not None and gates[pc] == x_c and gates[nc] == x_c:
            new = []
            for i, g in enumerate(gates):
                if i in (pc, nc):
                    continue
                new.append(g)
                if i == k:
                    new.append(("X", (t,)))
            out.append(new)
        z_t = ("Z", (t,))
        if pt is not None and nt is not None and gates[pt] == z_t and gates[nt] == z_t:
            new = []
            for i, g in enumerate(gates):
                if i in (pt, nt):
                    continue
                if i == k:
                    new.append(("Z", (c,)))
                new.append(g)
            out.append(new)
    elif name == "H":
        q = qubits[0]
        j = _next_on(gates, k, q)
        l = _next_on(gates, j, q) if j is not None else None
        if l is not None and len(gates[j][1]) == 1 and gates[l] == ("H", (q,)):
            mid = gates[j][0]
            swap = {"S": ("SDG", "H", "SDG"), "SDG": ("S", "H", "S")}
            if mid in swap:
                new = list(gates)
                new[k], new[j], new[l] = ((swap[mid][0], (q,)), (swap[mid][1], (q,)),
                                          (swap[mid][2], (q,)))
                out.append(new)
            elif mid in ("X", "Z"):
                repl = ("Z", (q,)) if mid == "X" else ("X", (q,))
                out.append([repl if i == j else g
                            for i, g in enumerate(gates) if i not in (k, l)])
    return out


def _h_reduce(gates, config):
    gates = list(gates)
    score = (_h_count(gates), _cost_key(gates, config))
    while True:
        for k in range(len(gates)):
            accepted = None
            for cand in _h_rules_at(gates, k):
                cand_score = (_h_count(cand), _cost_key(cand, config))
                if cand_score < score:
                    accepted = (cand, cand_score)
                    break
            if accepted:
                gates, score = accepted
                break
        else:
            return gates


# --- 4b: regions, exact representation, reconstruction -------------------------
#
# An H-free region maps |x> -> w^{f(x)} |A x + b>, w = e^{i pi/4}, over GF(2),
# with f(x) = sum_y c_y (y.x mod 2) mod 8. `rows[i]` is the bitmask of input
# bits XORed into local wire i; `b` the flip bits; `terms` maps parity y -> c_y.

def _region_build(gates, wires):
    index = {q: i for i, q in enumerate(wires)}
    rows = [1 << i for i in range(len(wires))]
    b = 0
    terms = {}
    for name, qubits in gates:
        if name == "CX":
            c, t = index[qubits[0]], index[qubits[1]]
            rows[t] ^= rows[c]
            if (b >> c) & 1:
                b ^= 1 << t
        elif name == "X":
            b ^= 1 << index[qubits[0]]
        elif name in _PHASE_K:
            q = index[qubits[0]]
            k = _PHASE_K[name]
            if (b >> q) & 1:
                # Phase on (1 + p): k*(1-p) = k - k*p; constant k is global phase.
                k = -k
            terms[rows[q]] = (terms.get(rows[q], 0) + k) % 8
        else:
            raise ValueError(f"{name} is not allowed inside an H-free region")
    terms = {y: c for y, c in sorted(terms.items()) if c % 8}
    return rows, b, terms


def _phase_table(m, terms):
    """f(x) mod 8 for every x. f(0) = 0 always, so no constant to remove."""
    return tuple(sum(c * ((y & x).bit_count() & 1) for y, c in terms.items()) % 8
                 for x in range(1 << m))


def _region_canon(gates, wires):
    """Exact canonical form. Equal iff equal unitaries up to global phase."""
    rows, b, terms = _region_build(gates, wires)
    return tuple(rows), b, _phase_table(len(wires), terms)


def _gadget(y, c, wires):
    """Gates applying w^{c (y.x)} using wires that currently hold x."""
    support = [i for i in range(len(wires)) if (y >> i) & 1]
    target, others = support[0], support[1:]
    ladder = [("CX", (wires[o], wires[target])) for o in others]
    phase = [(name, (wires[target],)) for name in _PHASE_OUT[c]]
    return ladder + phase + ladder[::-1]


def _linear_ops_gauss(rows):
    """CX list (local (c, t)) realizing the linear map, by Gauss-Jordan."""
    rows = list(rows)
    m = len(rows)
    ops = []
    for j in range(m):
        if not (rows[j] >> j) & 1:
            r = next(r for r in range(j + 1, m) if (rows[r] >> j) & 1)
            rows[j] ^= rows[r]
            ops.append((r, j))
        for i in range(m):
            if i != j and (rows[i] >> j) & 1:
                rows[i] ^= rows[j]
                ops.append((j, i))
    # ops reduce A to I; applying them in reverse order builds A from I.
    return ops[::-1]


@functools.lru_cache(maxsize=None)
def _linear_bfs(m):
    """Shortest CX sequence for every invertible m x m GF(2) map (m <= 4)."""
    from collections import deque
    start = tuple(1 << i for i in range(m))
    parent = {start: None}
    queue = deque([start])
    moves = [(c, t) for c in range(m) for t in range(m) if c != t]
    while queue:
        state = queue.popleft()
        for c, t in moves:
            nxt = list(state)
            nxt[t] ^= nxt[c]
            nxt = tuple(nxt)
            if nxt not in parent:
                parent[nxt] = (state, (c, t))
                queue.append(nxt)
    paths = {}
    for state in parent:
        path = []
        cur = state
        while parent[cur] is not None:
            cur, move = parent[cur]
            path.append(move)
        paths[state] = tuple(path[::-1])
    return paths


def _reconstruct_gauss(rows, b, terms, wires):
    gates = []
    for y, c in sorted(terms.items()):
        gates += _gadget(y, c, wires)
    gates += [("CX", (wires[c], wires[t])) for c, t in _linear_ops_gauss(rows)]
    gates += [("X", (wires[i],)) for i in range(len(wires)) if (b >> i) & 1]
    return gates


def _reconstruct_bfs(rows, b, terms, wires):
    """CX-optimal linear part; phases placed where a wire already holds them."""
    m = len(wires)
    pending = dict(terms)
    cur = [1 << i for i in range(m)]
    body = []

    def place():
        for i in range(m):
            if cur[i] in pending:
                body.extend((name, (wires[i],)) for name in _PHASE_OUT[pending.pop(cur[i])])

    place()
    for c, t in _linear_bfs(m)[tuple(rows)]:
        body.append(("CX", (wires[c], wires[t])))
        cur[t] ^= cur[c]
        place()
    pre = []
    for y, c in sorted(pending.items()):  # parities never seen: gadget at start
        pre += _gadget(y, c, wires)
    return pre + body + [("X", (wires[i],)) for i in range(m) if (b >> i) & 1]


# --- 4c: phase merging (automatic in `terms`) and resynthesis --------------------

def _z_variants(terms):
    """Exact rewrites of the term list using 4a + 4b = 4(a XOR b) (mod 8)."""
    variants = [dict(terms)]
    for selection in ((4,), (3, 4, 5)):
        chosen = [y for y, c in sorted(terms.items()) if c in selection]
        if not chosen:
            continue
        new = dict(terms)
        combined = 0
        for y in chosen:
            new[y] = (new[y] + 4) % 8
            combined ^= y
        if combined:
            new[combined] = (new.get(combined, 0) + 4) % 8
        new = {y: c for y, c in sorted(new.items()) if c}
        if new not in variants:
            variants.append(new)
    return variants


def _canonical_terms(m, table):
    """Term list derived from the phase TABLE alone (so it is unique).

    1. Mobius transform mod 8 gives the unique multilinear (AND-monomial) form
       f = sum_S a_S prod_{i in S} x_i.
    2. Each monomial is rewritten in parities with the exact identity
       2^{d-1} prod_{i in S} x_i = sum_{T nonempty, T in S} (-1)^{|T|+1} p_T,
       which needs 2^{d-1} | a_S. That holds for every phase function a
       circuit of these gates can make; the final table check confirms it.
    Returns None if the table is not of that form (it never should be).
    """
    a = list(table)
    for i in range(m):
        for x in range(1 << m):
            if (x >> i) & 1:
                a[x] = (a[x] - a[x ^ (1 << i)]) % 8
    terms = {}
    for monomial in range(1, 1 << m):
        coeff = a[monomial]
        if not coeff:
            continue
        scale = 1 << (monomial.bit_count() - 1)
        if coeff % scale:
            return None
        q = coeff // scale
        sub = monomial
        while sub:  # every nonempty subset T of the monomial
            sign = 1 if sub.bit_count() % 2 else -1
            terms[sub] = (terms.get(sub, 0) + sign * q) % 8
            sub = (sub - 1) & monomial
    terms = {y: c for y, c in sorted(terms.items()) if c}
    if _phase_table(m, terms) != tuple(table):
        return None
    return terms


def _wires_of(gates):
    return sorted({q for _, qubits in gates for q in qubits})


def _region_candidates(n, gates, wires):
    """All exact rewrites of one region, verified with the canonical form."""
    rows, b, terms = _region_build(gates, wires)
    canon = (tuple(rows), b, _phase_table(len(wires), terms))
    raw = [list(gates)]  # cleaned copy of the original is also tried
    bases = [terms]
    canonical = _canonical_terms(len(wires), canon[2])
    if canonical is not None and canonical != terms:
        bases.append(canonical)  # e.g. an identity region gets NO terms
    variants = []
    for base in bases:
        for variant in _z_variants(base):
            if variant not in variants:
                variants.append(variant)
    for variant in variants:
        raw.append(_reconstruct_gauss(rows, b, variant, wires))
        if len(wires) <= BFS_MAX_WIRES:
            raw.append(_reconstruct_bfs(rows, b, variant, wires))
    out = [list(gates)]  # the untouched original: always valid, never empty
    for cand in raw:
        cand = _stage12(n, cand)
        if any(name not in _REGION_GATES for name, _ in cand):
            continue  # cleanup never adds H, but stay strict
        if _region_canon(cand, wires) == canon:
            out.append(cand)
        elif _strict():
            raise AssertionError("region rewrite changed the region's unitary")
    return out


def _best_region(n, gates, config):
    """Cheapest verified rewrite of an H-free segment, judged on its own."""
    if not gates:
        return []
    wires = _wires_of(gates)
    return min(_region_candidates(n, gates, wires), key=lambda c: _key(c, config))


def _h_delay_order(gates, eager=False):
    """Topological re-sort that postpones H gates as long as possible
    (or, with eager=True, emits each H as soon as it is ready).

    Keeps every wire's gate order, so the unitary and all metrics are
    unchanged. Ties are broken by original index, so it is deterministic.
    Neither variant is always best: postponing an H that is needed later can
    move a boundary from the start into the middle. `_region_order` picks.
    """
    import heapq
    preds = [0] * len(gates)
    succs = [[] for _ in gates]
    last = {}
    for i, (_, qubits) in enumerate(gates):
        for q in qubits:
            if q in last and i not in succs[last[q]]:
                succs[last[q]].append(i)
                preds[i] += 1
            last[q] = i
    ready_other, ready_h = [], []
    for i in range(len(gates)):
        if preds[i] == 0:
            heapq.heappush(ready_h if gates[i][0] == "H" else ready_other, i)
    order = []

    def release(i):
        for s in succs[i]:
            preds[s] -= 1
            if preds[s] == 0:
                heapq.heappush(ready_h if gates[s][0] == "H" else ready_other, s)

    while ready_other or ready_h:
        if ready_other and not (eager and ready_h):
            i = heapq.heappop(ready_other)
            order.append(i)
            release(i)
        else:
            batch = [heapq.heappop(ready_h) for _ in range(len(ready_h))]
            for i in batch:
                order.append(i)
            for i in batch:
                release(i)
    return [gates[i] for i in order]


def _region_order(gates):
    """Wire-order-preserving reordering with the fewest H-free regions.

    Candidates: the given order, H-delayed, H-eager. Fewest regions wins;
    ties go to the earliest candidate, so the given order is kept unless
    another is strictly better. Never more regions than the input.
    """
    options = [list(gates), _h_delay_order(gates), _h_delay_order(gates, eager=True)]
    return min(enumerate(options), key=lambda io: (len(_region_spans(io[1])), io[0]))[1]


def _region_spans(gates):
    """(start, end) of each maximal H-free run, in order."""
    spans = []
    start = None
    for i, (name, _) in enumerate(gates):
        if name == "H":
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(gates)))
    return spans


def _region_pass(n, gates, config):
    """Replace every region by its cheapest verified rewrite (in context)."""
    gates = _region_order(gates)
    for s, e in reversed(_region_spans(gates)):  # right to left keeps indices valid
        segment = gates[s:e]
        wires = _wires_of(segment)
        best = None
        for cand in _region_candidates(n, segment, wires):
            full = gates[:s] + cand + gates[e:]
            if best is None or _key(full, config) < _key(best, config):
                best = full
        gates = best
    return gates


# --- 4d: cross-region simplification ---------------------------------------------
#
# (1) Backward term floating: a phase term of region R_{i+1} acts on its INPUT
#     wires (the terms are applied first), so it can be written as a gadget at
#     the very start of R_{i+1}. If no separating H touches the term's wires,
#     the gadget commutes past the separators and joins the end of R_i, where it
#     may merge with R_i's terms. Moving terms only backward needs no inverse
#     matrix, and still lets any two terms meet.
# (2) Region merging (includes inverse-region cancellation): if no separating H
#     touches either region's wires, both regions commute past the separators
#     and form one region; if they are mutually inverse it resynthesizes to [].

def _cross_region_pass(n, gates, config):
    gates = _region_order(gates)
    moves = 0
    changed = True
    while changed and moves < MAX_CROSS_MOVES:
        changed = False
        spans = _region_spans(gates)
        base = _key(gates, config)
        for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
            left, seps, right = gates[s0:e0], gates[e0:s1], gates[s1:e1]
            if any(name != "H" for name, _ in seps):
                continue
            sep_wires = {q for _, qs in seps for q in qs}
            prefix, suffix = gates[:s0], gates[e1:]
            candidates = []
            if not sep_wires & set(_wires_of(left) + _wires_of(right)):
                candidates.append(prefix + seps + _best_region(n, left + right, config)
                                  + suffix)
            r_wires = _wires_of(right)
            rows, b, terms = _region_build(right, r_wires)
            for y, c in sorted(terms.items()):
                y_wires = {r_wires[i] for i in range(len(r_wires)) if (y >> i) & 1}
                if y_wires & sep_wires:
                    continue
                rest_terms = {yy: cc for yy, cc in terms.items() if yy != y}
                right_rest = _reconstruct_gauss(rows, b, rest_terms, r_wires)
                gadget = _gadget(y, c, r_wires)
                # Exact check of R_{i+1} == gadget, then the rest.
                if _region_canon(gadget + right_rest, r_wires) != _region_canon(right, r_wires):
                    continue
                candidates.append(prefix + _best_region(n, left + gadget, config) + seps
                                  + _best_region(n, right_rest, config) + suffix)
            best = min(candidates, key=lambda g: _key(g, config), default=None)
            if best is not None and _cost_key(best, config) < base[:4]:
                gates = _region_order(best)
                moves += 1
                changed = True
                break
    if changed and moves >= MAX_CROSS_MOVES:
        _BUDGET_HITS["cross"] += 1
    return gates


# --- 4e: Pipeline B, Pauli-rotation form (Zhang & Chen, 2019) -----------------------
#
# A Pauli string is (x, z, s): bit q of x/z gives X/Z on qubit q (both = Y),
# and the operator is (-1)^s times the tensor product.

def _pauli_forward(p, name, qubits):
    """G P G^dagger for one Clifford gate G (Aaronson-Gottesman rules)."""
    x, z, s = p
    if name == "SDG":
        for _ in range(3):
            x, z, s = _pauli_forward((x, z, s), "S", qubits)
        return x, z, s
    if name == "CX":
        c, t = qubits
        xc, zc, xt, zt = (x >> c) & 1, (z >> c) & 1, (x >> t) & 1, (z >> t) & 1
        s ^= xc & zt & (xt ^ zc ^ 1)
        x ^= xc << t
        z ^= zt << c
        return x, z, s
    q = qubits[0]
    xq, zq = (x >> q) & 1, (z >> q) & 1
    if name == "H":
        s ^= xq & zq
        x = (x & ~(1 << q)) | (zq << q)
        z = (z & ~(1 << q)) | (xq << q)
    elif name == "S":
        s ^= xq & zq
        z ^= xq << q
    elif name == "X":
        s ^= zq
    elif name == "Z":
        s ^= xq
    else:
        raise ValueError(f"{name} is not Clifford")
    return x, z, s


def _pauli_backward(p, name, qubits):
    """G^dagger P G: forward conjugation by the inverse gate."""
    inverse = {"S": "SDG", "SDG": "S"}.get(name, name)
    return _pauli_forward(p, inverse, qubits)


def _paulis_anticommute(p, q):
    return ((p[0] & q[1]) ^ (p[1] & q[0])).bit_count() & 1


def _to_rotations(gates):
    """Split into (rotations in time order, Clifford gates in order).

    U = C_final * R_m ... R_1 with P_j = C_j^dagger Z_q C_j, where C_j is the
    Clifford prefix before the j-th T/TDG (T = R_Z(pi/4) up to global phase).
    """
    clifford, rotations = [], []
    for name, qubits in gates:
        if name in ("T", "TDG"):
            p = (0, 1 << qubits[0], 0)
            for g in reversed(clifford):
                p = _pauli_backward(p, *g)
            rotations.append((p, 1 if name == "T" else 7))
        else:
            clifford.append((name, qubits))
    return rotations, clifford


def _merge_rotations(rotations):
    """Merge equal-axis rotations that can slide together (exact)."""
    out = []
    for p, k in rotations:
        for m in range(len(out) - 1, -1, -1):
            q, kq = out[m]
            if q[0] == p[0] and q[1] == p[1]:
                total = (kq + (k if q[2] == p[2] else -k)) % 8
                if total:
                    out[m] = (q, total)
                else:
                    del out[m]
                break
            if _paulis_anticommute(q, p):
                out.append((p, k % 8))
                break
        else:
            out.append((p, k % 8))
    return out


def _rotation_gates(n, p, k):
    """R_P(k pi/4) as gates: basis change, CX ladder, T^k, undo."""
    x, z, s = p
    if s:
        k = (-k) % 8
    support = [q for q in range(n) if ((x | z) >> q) & 1]
    pre, post = [], []
    for q in support:
        if (x >> q) & 1 and not (z >> q) & 1:      # X
            pre.append(("H", (q,)))
            post.append(("H", (q,)))
        elif (x >> q) & 1:                          # Y
            pre += [("SDG", (q,)), ("H", (q,))]
            post += [("H", (q,)), ("S", (q,))]
    target = support[0]
    ladder = [("CX", (o, target)) for o in support[1:]]
    phase = [(name, (target,)) for name in _PHASE_OUT[k]]
    return pre + ladder + phase + ladder[::-1] + post


def _pipeline_b(n, gates):
    rotations, clifford = _to_rotations(gates)
    out = []
    for p, k in _merge_rotations(rotations):
        out += _rotation_gates(n, p, k)
    return _stage12(n, out + clifford)


# --- Stage 5 B: phase folding across H gates ----------------------------------
#
# Sum-over-paths view (as in T-par, Amy-Maslov-Mosca 2014): describe each
# wire's value as a parity (XOR) of PATH VARIABLES plus a constant bit. The
# path variables are the n inputs plus one fresh variable for every H gate.
#   CX(c,t): value[t] ^= value[c]      X(q): flip value[q]'s constant
#   H(q):    value[q] = a fresh variable
# The circuit's amplitudes are a sum over all variable assignments of
# w^{P}, where P adds k*(parity) for every phase gate T^k (plus terms from
# the H gates, which folding never touches). P is a SUM, so every phase gate
# applied to the same parity can be combined into one, applied at any one of
# their positions: P, and therefore the unitary, is unchanged.
# A constant bit of 1 means the gate acts on (1 + parity) = k - k*parity, so
# its contribution to the parity term is -k (the constant k is global phase).

def _phase_fold(gates, host="first"):
    """Merge all phase gates on the same path-variable parity into one place."""
    value = {}
    next_var = 0
    groups = {}      # parity mask -> [(index, constant bit, signed exponent)]
    for i, (name, qubits) in enumerate(gates):
        for q in qubits:
            if q not in value:
                value[q] = (1 << next_var, 0)
                next_var += 1
        if name == "CX":
            c, t = qubits
            value[t] = (value[t][0] ^ value[c][0], value[t][1] ^ value[c][1])
        elif name == "X":
            q = qubits[0]
            value[q] = (value[q][0], value[q][1] ^ 1)
        elif name == "H":
            value[qubits[0]] = (1 << next_var, 0)
            next_var += 1
        else:  # phase gate
            mask, const = value[qubits[0]]
            k = _PHASE_K[name]
            groups.setdefault(mask, []).append((i, const, -k if const else k))
    replace = {}
    drop = set()
    for occurrences in groups.values():      # insertion order: deterministic
        if len(occurrences) < 2:
            continue
        total = sum(k for _, _, k in occurrences) % 8
        index, const, _ = occurrences[0] if host == "first" else occurrences[-1]
        exponent = (-total) % 8 if const else total
        wire = gates[index][1]
        replace[index] = [(name, wire) for name in _PHASE_OUT[exponent]]
        drop.update(i for i, _, _ in occurrences if i != index)
    out = []
    for i, gate in enumerate(gates):
        if i in drop:
            continue
        out.extend(replace.get(i, [gate]))
    return out


def _fold_pass(n, gates, config):
    """Best of {unchanged, fold to first position, fold to last position}."""
    options = [list(gates)]
    for host in ("first", "last"):
        options.append(_stage12(n, _phase_fold(gates, host)))
    return min(enumerate(options), key=lambda io: (_cost_key(io[1], config), io[0]))[1]


# --- Stage 5 C: exact 2-qubit Clifford block resynthesis -----------------------
#
# A two-qubit Clifford operation (gates H, S, SDG, X, Z, CX only) is fixed, up
# to global phase, by where it sends the four Paulis X0, Z0, X1, Z1 (with
# signs). That gives an EXACT integer key. There are 11,520 such operations.
# For each one we store every Pareto-optimal implementation in
# (CX count, one-qubit count, depth of wire 0, depth of wire 1), so the
# cheapest one can be chosen for any weights.

_CLIFFORD_NAMES = ("H", "S", "SDG", "X", "Z")
_CLIFF2_GENERATORS = ([(g, (q,)) for g in _CLIFFORD_NAMES for q in (0, 1)]
                      + [("CX", (0, 1)), ("CX", (1, 0))])
_PAULI_GENS_2Q = ((1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 2, 0))  # X0, Z0, X1, Z1


def _cliff2_key(gates):
    """Exact key: images of X0, Z0, X1, Z1 under U P U^dagger."""
    images = []
    for p in _PAULI_GENS_2Q:
        for g in gates:
            p = _pauli_forward(p, *g)
        images.append(p)
    return tuple(images)


def _key_after(key, gate):
    """Key of (circuit + gate) from the circuit's key."""
    return tuple(_pauli_forward(p, *gate) for p in key)


def _dominates(a, b):
    return all(x <= y for x, y in zip(a, b))


@functools.lru_cache(maxsize=None)
def _cliff2_table():
    """key -> list of (label, gates); label = (n2, n1, depth0, depth1). Pareto."""
    start = _cliff2_key([])
    table = {start: [((0, 0, 0, 0), ())]}
    layer = [(start, (0, 0, 0, 0), ())]
    while layer:
        nxt = []
        for key, (n2, n1, a0, a1), gates in layer:
            for g in _CLIFF2_GENERATORS:
                if g[0] == "CX":
                    top = max(a0, a1) + 1
                    label = (n2 + 1, n1, top, top)
                elif g[1] == (0,):
                    label = (n2, n1 + 1, a0 + 1, a1)
                else:
                    label = (n2, n1 + 1, a0, a1 + 1)
                k2 = _key_after(key, g)
                entries = table.setdefault(k2, [])
                if any(_dominates(old, label) for old, _ in entries):
                    continue
                entries[:] = [(old, seq) for old, seq in entries if not _dominates(label, old)]
                entries.append((label, gates + (g,)))
                nxt.append((k2, label, gates + (g,)))
        layer = nxt
    return table


def _label_cost(label, config):
    n2, n1, a0, a1 = label
    return config["w1"] * n1 + config["w2"] * n2 + config["wd"] * max(a0, a1)


def _local_label(gates):
    a0 = a1 = n1 = n2 = 0
    for name, qubits in gates:
        if name == "CX":
            a0 = a1 = max(a0, a1) + 1
            n2 += 1
        elif qubits == (0,):
            a0 += 1; n1 += 1
        else:
            a1 += 1; n1 += 1
    return (n2, n1, a0, a1)


def _clifford_blocks(gates, a, b):
    """Maximal runs (list order, skipping gates on other wires) of Clifford
    gates acting only within {a, b}. Yields lists of indices."""
    block = []
    for i, (name, qubits) in enumerate(gates):
        touches = [q for q in qubits if q in (a, b)]
        if not touches:
            continue
        if name in ("T", "TDG") or len(touches) < len(qubits):
            if len(block) >= 2:
                yield block
            block = []
        else:
            block.append(i)
    if len(block) >= 2:
        yield block


def _block_pass(n, gates, config):
    """Replace 2-qubit Clifford blocks by their cheapest exact implementation."""
    table = _cliff2_table()
    gates = list(gates)
    changed = True
    while changed:
        changed = False
        base = _key(gates, config)
        for a in range(n):
            for b in range(a + 1, n):
                for block in _clifford_blocks(gates, a, b):
                    local = {a: 0, b: 1}
                    seq = [(gates[i][0], tuple(local[q] for q in gates[i][1])) for i in block]
                    options = table[_cliff2_key(seq)]
                    label, best = min(options, key=lambda e: (_label_cost(e[0], config), e[0]))
                    if _label_cost(label, config) >= _label_cost(_local_label(seq), config):
                        continue
                    back = {0: a, 1: b}
                    new = [(g, tuple(back[q] for q in qs)) for g, qs in best]
                    drop = set(block)
                    out = []
                    for i, g in enumerate(gates):
                        if i == block[-1]:
                            out.extend(new)
                        elif i not in drop:
                            out.append(g)
                    if _key(out, config) < base:
                        gates = out
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return gates


# --- the shared, convergent optimization loop (used by every candidate) -------------

def _full_optimize(n, gates, config):
    """Stages 1-2, phase folding, 4a, 4c, 4d and 2-qubit blocks, repeated until
    the cost tuple stops dropping."""
    gates = _stage12(n, gates)
    for _ in range(MAX_ROUNDS):
        g = gates
        if FEATURES["fold"]:
            g = _stage12(n, _fold_pass(n, g, config))
        if FEATURES["h_reduce"]:
            g = _stage12(n, _h_reduce(g, config))
        if FEATURES["regions"]:
            g = _stage12(n, _region_pass(n, g, config))
        if FEATURES["cross_regions"]:
            g = _stage12(n, _cross_region_pass(n, g, config))
        if FEATURES["blocks"]:
            g = _stage12(n, _block_pass(n, g, config))
        if _cost_key(g, config) < _cost_key(gates, config):
            gates = g
        else:
            break
    else:
        _BUDGET_HITS["full"] += 1
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