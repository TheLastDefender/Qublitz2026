"""Qublitz 2026 - Quantum Circuit Golf: the submitted optimizer.

Install:  python -m pip install numpy
Try it:   python submission.py

`optimize(circuit, config)` returns an equivalent circuit (up to global phase)
with lower weighted cost C = w1*N_1q + w2*N_2q + wd*depth. See README.md for
the full algorithm, every identity used, citations and design trade-offs.
The validators, unitary checker and cost functions at the end of this file
are the organizers' starter code, unchanged.

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
import heapq
import itertools
import math
import os
from collections import deque

import numpy as np


SINGLE_QUBIT_GATES = frozenset(("H", "X", "Z", "S", "SDG", "T", "TDG"))
ALLOWED_GATES = SINGLE_QUBIT_GATES | {"CX"}


def optimize(circuit, config, beam_width=None):
    """Optimizes a quantum circuit to reduce its cost according to the given weights.

    Inputs:
        circuit: A dictionary representing the quantum circuit to be optimized.
        config: A dictionary containing the weights for one-qubit gates, two-qubit 
        gates, and circuit depth.
        beam_width: optional. Circuits kept per beam-search step; None means
        DEFAULT_BEAM_WIDTH, 0 turns the beam search off. Larger is slower and
        usually cheaper. The result is deterministic for any given value.
    """
    validate_circuit(circuit)
    validate_config(config)
    if beam_width is not None and (isinstance(beam_width, bool)
                                   or not isinstance(beam_width, int) or beam_width < 0):
        raise ValueError("beam_width must be a non-negative integer or None.")

    # Each selection round builds several candidates, fully optimizes each
    # with the same greedy loop, verifies them against the input unitary,
    # picks the cheapest under `config` (ties keep the round's input), then
    # runs beam search from it (see _select).
    n = circuit["n_qubits"]
    original = _as_pairs(circuit)
    reference = _fast_unitary(n, original)   # input unitary, computed once
    current = original
    # Repeat the selection from its own result until it returns its input.
    # That makes optimize(optimize(c)) == optimize(c) exactly: the second call
    # starts from a fixed point (unless the MAX_SELECT_ROUNDS budget ran out).
    for _ in range(MAX_SELECT_ROUNDS):
        best = _select(n, reference, current, config, beam_width)
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


def _select(n, reference, start, config, beam_width=None):
    """One selection round: build candidates (the round's input, Pipeline A,
    Pipeline B, B after A, A on the given listing), verify each against the
    input unitary, keep the cheapest, then beam-search from it."""
    s2 = _stage12(n, start)                                # Stages 1-2 cleanup
    a = _full_optimize(n, s2, config)                      # Pipeline A
    candidates = [start, a]
    if FEATURES["pipeline_b"]:
        candidates.append(_full_optimize(n, _pipeline_b(n, s2), config))  # B
        candidates.append(_full_optimize(n, _pipeline_b(n, a), config))   # B after A
    # Pipeline A with list regions on the given listing (no H-delayed/H-eager
    # reordering). The fewest-regions reordering helps some weights and hurts
    # others; as an extra candidate (last, so ties keep the others) it made
    # the bench corpus no worse under any profile: -0.2% w2-heavy, -1.1%
    # balanced, -0.3% w1-heavy, 0 wd-heavy.
    candidates.append(_full_optimize(n, s2, config, reorder_regions=False))
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
    best = min(verified, key=lambda ic: (_cost_key(ic[1], config), ic[0]))[1]
    # Beam search from the best greedy result. Accepted only if
    # strictly cheaper AND verified, so it can never make the result worse.
    if FEATURES["beam"] and _beam_settings(beam_width)[0] > 0:
        beamed = _beam_search(n, best, config, beam_width)
        if _cost_key(beamed, config) < _cost_key(best, config):
            if _fast_equivalent(reference, n, beamed):
                return beamed
            if _strict():
                raise AssertionError("beam search result failed verification")
    return best


def _stage12(n, gates):
    """Stages 1-2 iterated to a fixed point.

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
# Stage 2 works on a flat list of (name, qubits) tuples. Sequences below are in
# circuit (time) order, as listed. Three mechanisms:
#
# (a) COMMUTATION. If gate g commutes with every gate M between g and a
#     partner p on the same wire(s), then g M p = M g p. If p = g and g is
#     self-inverse, M g g = M, so both vanish. A gate is only moved in the
#     algebra, never in the output: the output just drops the pair. Gates on
#     disjoint qubits always commute and are skipped. `_commutes` checks each
#     pair against the exact matrices, so no commutation rule is assumed.
# (b) PHASE SEGMENTS. On one wire, all phase gates between two blockers
#     (H or X on that wire, or a CX targeting it) are summed as T-exponents
#     mod 8 and re-emitted in shortest form (_phase_segment_pass).
# (c) ONE-QUBIT RUNS. A run of consecutive one-qubit gates on a wire is
#     replaced by its shortest equivalent (_best_run): exact table lookup for
#     operators needing <= 7 gates, otherwise a best split, also tried on the
#     Matsumoto-Amano normal form. This covers sandwiches like H X H -> Z.
#
# Termination: every accepted rewrite strictly lowers the gate count.

_SINGLE_NAMES = sorted(SINGLE_QUBIT_GATES)


@functools.lru_cache(maxsize=None)
def _commutes(first, second):
    """Exact check: do two gates commute as operators on their union of qubits?
    Cached on the gate pair; the relabelled matrix test below is cached too."""
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
            # (Phase gates are merged by _phase_segment_pass, not here.)
            # Otherwise gj is a blocker unless it commutes with gi.
            if not _commutes(gi, gj):
                break
    return [cur[k] for k in range(len(cur)) if alive[k]] if changed else list(gates)



# --- Matsumoto-Amano normal form (exact single-qubit Clifford+T) ---------------
#
# Matsumoto & Amano (arXiv:0806.3834); streamlined by Giles & Selinger
# (arXiv:1312.6584). Every single-qubit Clifford+T operator has a UNIQUE
# normal form: a Clifford, followed by syllables HT or SHT, with an optional
# T, and it has the minimum possible number of T gates.
#
# Exact arithmetic: an operator acts on the Bloch sphere as a 3x3 rotation
# R (U sigma_j U^dagger = sum_i R[i][j] sigma_i), which forgets global phase
# exactly. Every entry lies in Z[1/sqrt2], so R = (1/sqrt2)^k * N with N's
# entries a + b*sqrt2 (integers a, b), stored as (a, b). (a + b sqrt2) is
# divisible by sqrt2 exactly when a is even: (a + b sqrt2)/sqrt2 = b + (a/2) sqrt2.
# Reducing k as far as possible gives the "denominator exponent"; the reduced
# (k, N) is unique, so it is an EXACT key for the operator up to global phase.

def _zr_mul(x, y):
    return (x[0] * y[0] + 2 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def _zr_add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def _so3_reduce(k, n):
    while k > 0 and all(a % 2 == 0 for row in n for a, _ in row):
        n = tuple(tuple((b, a // 2) for a, b in row) for row in n)
        k -= 1
    return k, n


def _so3_mul(left, right):
    """(k1, N1) * (k2, N2) as matrices."""
    (k1, a), (k2, b) = left, right
    prod = tuple(tuple(
        _zr_add(_zr_add(_zr_mul(a[i][0], b[0][j]), _zr_mul(a[i][1], b[1][j])),
                _zr_mul(a[i][2], b[2][j]))
        for j in range(3)) for i in range(3))
    return _so3_reduce(k1 + k2, prod)


def _so3_transpose(m):
    k, n = m
    return k, tuple(tuple(n[j][i] for j in range(3)) for i in range(3))


def _so3_int(rows):
    return 0, tuple(tuple((v, 0) for v in row) for row in rows)


_SO3_GATES = {
    "H": _so3_int(((0, 0, 1), (0, -1, 0), (1, 0, 0))),
    "S": _so3_int(((0, -1, 0), (1, 0, 0), (0, 0, 1))),
    "X": _so3_int(((1, 0, 0), (0, -1, 0), (0, 0, -1))),
    "Z": _so3_int(((-1, 0, 0), (0, -1, 0), (0, 0, 1))),
    # T: rotation by pi/4 about Z, entries 1/sqrt2 = (1/sqrt2)^1 * 1, and the
    # Z-Z entry 1 = (1/sqrt2)^1 * sqrt2.
    "T": (1, (((1, 0), (-1, 0), (0, 0)), ((1, 0), (1, 0), (0, 0)), ((0, 0), (0, 0), (0, 1)))),
}
_SO3_GATES["SDG"] = _so3_transpose(_SO3_GATES["S"])
_SO3_GATES["TDG"] = _so3_transpose(_SO3_GATES["T"])
_SO3_ID = _so3_int(((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def _so3_of(names):
    """Exact key of a one-qubit gate list (first gate acts first)."""
    m = _SO3_ID
    for name in names:
        m = _so3_mul(_SO3_GATES[name], m)
    return m


@functools.lru_cache(maxsize=None)
def _clifford1_table():
    """Exact key -> shortest gate list, for the 24 one-qubit Cliffords."""
    table = {_SO3_ID: ()}
    frontier = [()]
    while frontier:
        nxt = []
        for seq in frontier:
            for name in ("H", "S", "SDG", "X", "Z"):
                new = seq + (name,)
                key = _so3_of(new)
                if key not in table:
                    table[key] = new
                    nxt.append(new)
        frontier = nxt
    return table


# Syllables as operators, with their gates in TIME order.
_MA_SYLLABLES = (
    (("T",), _so3_of(("T",))),
    (("T", "H"), _so3_of(("T", "H"))),          # operator H*T
    (("T", "H", "S"), _so3_of(("T", "H", "S"))),  # operator S*H*T
)


def _ma_normal_form(names):
    """Matsumoto-Amano normal form of a one-qubit gate list (time order).

    Repeatedly strips the syllable (T, HT or SHT) whose removal lowers the
    denominator exponent; what remains has exponent 0, i.e. is Clifford.
    Returns the gate list, or None if no syllable reduces (never expected;
    strict mode raises instead).
    """
    m = _so3_of(names)
    left = []                       # syllables, outermost (latest) first
    while m[0] > 0:
        for gates, syl in _MA_SYLLABLES:
            reduced = _so3_mul(_so3_transpose(syl), m)
            if reduced[0] < m[0]:
                left.append(gates)
                m = reduced
                break
        else:
            if _strict():
                raise AssertionError("Matsumoto-Amano reduction failed")
            return None
    out = list(_clifford1_table()[m])
    for gates in reversed(left):
        out += list(gates)
    return out


LONG_WINDOW = 7   # one-qubit table depth: 376 distinct operators


@functools.lru_cache(maxsize=None)
def _window_table_long():
    """EXACT key (see _so3_of) -> shortest gate list, for every one-qubit
    operator that needs at most LONG_WINDOW gates. No floating point."""
    table = {_SO3_ID: ()}
    frontier = [()]
    for _ in range(LONG_WINDOW):
        nxt = []
        for seq in frontier:
            for name in _SINGLE_NAMES:
                new = seq + (name,)
                key = _so3_of(new)
                if key not in table:
                    table[key] = new
                    nxt.append(new)
        frontier = nxt
    return table


def _best_run(names):
    """Cached wrapper: identical runs recur across thousands of beam children."""
    return list(_best_run_cached(tuple(names)))


@functools.lru_cache(maxsize=65536)
def _best_run_cached(names):
    """Shortest replacement for a run of one-qubit gates on one wire.

    If the run's operator needs at most LONG_WINDOW gates, the table gives
    the exact optimum. Otherwise a dynamic program splits a sequence into
    chunks of at most LONG_WINDOW gates, each replaced by its optimum, and
    this is done both for the run itself and for its Matsumoto-Amano normal
    form (minimum T count); the shorter result is kept.
    """
    table = _window_table_long()
    whole = table.get(_so3_of(names))
    if whole is not None:
        return tuple(whole)

    def split(seq):
        best = [[]]
        for i in range(1, len(seq) + 1):
            best.append(min((best[j] + list(table[_so3_of(seq[j:i])])
                             for j in range(max(0, i - LONG_WINDOW), i)), key=len))
        return best[-1]

    options = [split(list(names))]
    normal = _ma_normal_form(list(names))
    if normal is not None:
        options.append(split(normal))
    return tuple(min(options, key=len))


def _splice(gates, drop, at, new):
    """Copy of `gates` without the indices in `drop`, with `new` inserted where
    index `at` was (`at` is one of the dropped indices). Every block rewrite
    uses this: the dropped gates are consecutive on their wires, so moving
    the replacement to one of their positions keeps each wire's order."""
    out = []
    for i, g in enumerate(gates):
        if i == at:
            out.extend(new)
        elif i not in drop:
            out.append(g)
    return out


def _window_pass(gates):
    """Replace the first one-qubit run that has a shorter equivalent."""
    n_gates = len(gates)
    for q in sorted({w for _, qs in gates for w in qs}):
        runs, run = [], []
        for k in range(n_gates):
            if q not in gates[k][1]:
                continue
            if len(gates[k][1]) == 1:
                run.append(k)
            else:
                if len(run) >= 2:
                    runs.append(run)
                run = []
        if len(run) >= 2:
            runs.append(run)
        for run in runs:
            best = _best_run([gates[k][0] for k in run])
            if len(best) < len(run):
                return _splice(gates, set(run), run[0], [(name, (q,)) for name in best]), True
    return list(gates), False


def _phase_segment_pass(gates):
    """Sum each wire's phase segments in one go.

    On wire q, a phase gate commutes with everything on q except H(q), X(q)
    and a CX TARGETING q. Between two such blockers every phase gate on q can
    slide next to the first one, so the whole segment is one exponent sum
    mod 8. Only applied when it strictly shortens the circuit.
    """
    n_gates = len(gates)
    for q in sorted({w for _, qs in gates for w in qs}):
        segment = []
        for k in list(range(n_gates)) + [None]:
            if k is not None and q not in gates[k][1]:
                continue
            blocker = (k is None or gates[k][0] in ("H", "X")
                       or (gates[k][0] == "CX" and gates[k][1][1] == q))
            if k is not None and gates[k][0] in _PHASE_K:
                segment.append(k)
                continue
            if not blocker:
                continue                       # CX with q as control: phases pass
            if len(segment) >= 2:
                total = sum(_PHASE_K[gates[i][0]] for i in segment) % 8
                if len(_PHASE_OUT[total]) < len(segment):
                    # The merged phase may sit at any of the segment's phase
                    # positions (all equivalent). Gate count is the same at
                    # each, so pick the lowest depth (ties: earliest).
                    merged = [(name, (q,)) for name in _PHASE_OUT[total]]
                    options = [_splice(gates, set(segment), at, merged) for at in segment]
                    return min(options, key=lambda o: _metrics_pairs(o)[2]), True
            segment = []
    return list(gates), False


def _stage2_reduce(gates):
    """Phase segments, commutation cancellation and one-qubit runs, until
    nothing shortens."""
    gates = list(gates)
    while True:
        shorter, changed = _phase_segment_pass(gates)
        if changed:
            gates = shorter
            continue
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
# Stage 4 and later: regions, phase folding, exact 2-qubit blocks, Pauli
# rotations, beam search
# =============================================================================
#
# Determinism: no randomness, no wall-clock limits (work budgets below), no
# iteration over sets to produce output, every tie broken explicitly. Region
# synthesis, phase folding, Pauli tableaux and the one-qubit tables use exact
# integers (GF(2) bitmasks, exponents mod 8, Pauli bits + sign, Z[1/sqrt2]).
# Floats are used only where a matrix is compared: the cached commutation
# test and unitary verification.
#
# Exactness: every rewrite is an identity up to global phase. Region rewrites
# are checked exactly with `_region_canon` (integer arithmetic, proven iff);
# every candidate is checked against the input unitary in `_select`, and the
# final result with the organizers' checker in `optimize`.

# Work budgets (deterministic: counts, never time). Termination never depends
# on them: every loop strictly lowers a well-ordered cost tuple. They bound
# runtime; `_BUDGET_HITS` records any loop that stopped because of a cap, and
# a test asserts that this never happens on the test corpus.
MAX_ROUNDS = 8          # rounds of the full loop per candidate (work budget)
MAX_SELECT_ROUNDS = 8   # selection rounds in optimize (work budget)
MAX_CROSS_MOVES = 64    # accepted cross-region moves per round (work budget)
BFS_MAX_WIRES = 4       # exact CX-optimal linear synthesis up to this width
_REGION_GATES = frozenset(("CX", "X", "Z", "S", "SDG", "T", "TDG"))
_BUDGET_HITS = {"select": 0, "full": 0, "cross": 0, "regions": 0, "tblocks": 0, "beam": 0}

# Feature switches. bench.py toggles them to measure each component's
# contribution. The *_greedy switches control the greedy loop only; those
# passes are always available as beam-search moves.
FEATURES = {
    "h_reduce": True,
    "regions": True,
    # Off in the greedy loop: measured (bench.py --ablation) to make results
    # WORSE there (0.725 -> 0.718 of Stage 2 cost when removed), because an
    # immediately-accepted move blocks better later ones. Still used as a
    # beam-search move, where a bad early choice can be abandoned.
    "cross_regions_greedy": False,
    "pipeline_b": True,
    "fold": True,
    "blocks": True,
    "convex_regions": True,
    "beam": True,
    # Off in the greedy loop, on as a beam move: measured better that way
    # (w1-heavy 9.51 -> 9.40, balanced 23.43 -> 23.23 on the quick corpus).
    "tblocks_greedy": False,
    "depth_fold": True,
}


def _depth_focus(config):
    """Depth-only moves are used when depth outweighs the gate weights.
    Measured (bench --quick): depth-placed folding improved wd-heavy
    (0,0,1) by 2.2% but made balanced (1,1,1) 0.9% worse, because a
    greedy depth choice can crowd out a gate saving. This rule turns it on
    for the former and off for the latter; it was chosen from that corpus."""
    return FEATURES["depth_fold"] and config["wd"] > config["w1"] + config["w2"]


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
    """Apply the rules above until none lowers (H count, cost tuple)."""
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
    """(rows, b, terms) of an H-free gate list on the given wires (see above)."""
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
                # Phase on (1 XOR p) = 1 - p: k(1-p) = k - kp; k is global phase.
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


def _parity_tree(support, target):
    """CX gates that XOR every wire of `support` onto `target`, arranged as a
    tree of disjoint CX gates: depth ceil(log2 |support|) instead of the
    ladder's |support| - 1, with the same number of CX gates. Intermediate
    wires are modified too, so callers undo it with the reversed list."""
    nodes = [target] + [q for q in support if q != target]
    out = []
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes) - 1, 2):
            out.append(("CX", (nodes[i + 1], nodes[i])))
            nxt.append(nodes[i])
        if len(nodes) % 2:
            nxt.append(nodes[-1])
        nodes = nxt
    return out


def _gadget(y, c, wires):
    """Gates applying w^{c (y.x)} using wires that currently hold x."""
    support = [wires[i] for i in range(len(wires)) if (y >> i) & 1]
    tree = _parity_tree(support, support[0])
    phase = [(name, (support[0],)) for name in _PHASE_OUT[c]]
    return tree + phase + tree[::-1]


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
    """Baseline rebuild: one gadget per term, then the linear map by
    Gauss-Jordan elimination, then X gates for b."""
    gates = []
    for y, c in sorted(terms.items()):
        gates += _gadget(y, c, wires)
    gates += [("CX", (wires[c], wires[t])) for c, t in _linear_ops_gauss(rows)]
    gates += [("X", (wires[i],)) for i in range(len(wires)) if (b >> i) & 1]
    return gates


def _walk_cx_path(path, terms, wires):
    """Walk a CX path (local (c, t) pairs) from the identity, applying each
    phase term the first time its parity sits on a wire (lowest wire first).
    Returns (gates, terms whose parity never appeared)."""
    m = len(wires)
    pending = dict(terms)
    cur = [1 << i for i in range(m)]
    out = []

    def place():
        for i in range(m):
            if cur[i] in pending:
                out.extend((name, (wires[i],)) for name in _PHASE_OUT[pending.pop(cur[i])])

    place()
    for c, t in path:
        out.append(("CX", (wires[c], wires[t])))
        cur[t] ^= cur[c]
        place()
    return out, pending


def _flips(b, wires):
    """X gates for the flip bits b."""
    return [("X", (wires[i],)) for i in range(len(wires)) if (b >> i) & 1]


def _reconstruct_bfs(rows, b, terms, wires):
    """CX-optimal linear part; phases placed where a wire already holds them,
    any parity that never appears as a gadget at the start."""
    body, pending = _walk_cx_path(_linear_bfs(len(wires))[tuple(rows)], terms, wires)
    pre = []
    for y, c in sorted(pending.items()):
        pre += _gadget(y, c, wires)
    return pre + body + _flips(b, wires)


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
    """All exact rewrites of one region (cached: beam states share regions)."""
    return [list(c) for c in _region_candidates_cached(n, tuple(gates), tuple(wires))]


@functools.lru_cache(maxsize=16384)
def _region_candidates_cached(n, gates, wires):
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
        if len(wires) <= 3:
            raw.append(_reconstruct_exact_small(rows, b, variant, wires))
        raw.append(_reconstruct_graysynth(rows, b, variant, wires))
    out = [list(gates)]  # the untouched original: always valid, never empty
    for cand in raw:
        if cand is None:
            continue  # a synthesizer that does not apply to this region
        cand = _stage12(n, cand)
        if any(name not in _REGION_GATES for name, _ in cand):
            continue  # cleanup never adds H, but stay strict
        if _region_canon(cand, wires) == canon:
            out.append(cand)
        elif _strict():
            raise AssertionError("region rewrite changed the region's unitary")
    return tuple(tuple(c) for c in out)


def _best_region(n, gates, config):
    """Cheapest verified rewrite of an H-free segment, judged on its own."""
    if not gates:
        return []
    wires = _wires_of(gates)
    return min(_region_candidates(n, gates, wires), key=lambda c: _key(c, config))


def _wire_successors(gates):
    """For each gate, the indices of the next gate on each of its wires
    (without duplicates): the dependency graph used for reordering."""
    succs = [[] for _ in gates]
    last = {}
    for i, (_, qubits) in enumerate(gates):
        for q in qubits:
            if q in last and i not in succs[last[q]]:
                succs[last[q]].append(i)
            last[q] = i
    return succs


def _h_delay_order(gates, eager=False):
    """Topological re-sort that postpones H gates as long as possible
    (or, with eager=True, emits each H as soon as it is ready).

    Keeps every wire's gate order, so the unitary and all metrics are
    unchanged. Ties are broken by original index, so it is deterministic.
    Neither variant is always best: postponing an H that is needed later can
    move a boundary from the start into the middle. `_region_order` picks.
    """
    succs = _wire_successors(gates)
    preds = [0] * len(gates)
    for nexts in succs:
        for i in nexts:
            preds[i] += 1
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


def _region_pass(n, gates, config, reorder=True):
    """Replace every region by its cheapest verified rewrite (in context).
    With reorder=False the given listing is kept (see _select)."""
    if reorder:
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
    """Merge adjacent regions or float terms backward across H gates, when
    strictly cheaper (see the section comment). Used as a beam move."""
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
    """R_P(k pi/4) as gates: basis change, CX parity tree, T^k, undo."""
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
    tree = _parity_tree(support, target)
    phase = [(name, (target,)) for name in _PHASE_OUT[k]]
    return pre + tree + phase + tree[::-1] + post


def _pipeline_b(n, gates):
    """Rotations first (merged, then synthesized), then the Clifford gates."""
    rotations, clifford = _to_rotations(gates)
    out = []
    for p, k in _merge_rotations(rotations):
        out += _rotation_gates(n, p, k)
    return _stage12(n, out + clifford)


# --- phase folding across H gates ---------------------------------------------
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

    def build():
        out = []
        for i, gate in enumerate(gates):
            if i in drop:
                continue
            out.extend(replace.get(i, [gate]))
        return out

    for occurrences in groups.values():      # insertion order: deterministic
        if len(occurrences) < 2:
            continue
        total = sum(k for _, _, k in occurrences) % 8
        if host == "first":
            hosts = [occurrences[0]]
        elif host == "last":
            hosts = [occurrences[-1]]
        else:   # "depth": any occurrence may host; keep the lowest depth
            hosts = occurrences
        best = None
        for index, const, _ in hosts:
            exponent = (-total) % 8 if const else total
            wire = gates[index][1]
            replace[index] = [(name, wire) for name in _PHASE_OUT[exponent]]
            dropped = {i for i, _, _ in occurrences if i != index}
            drop.update(dropped)
            depth = _metrics_pairs(build())[2] if len(hosts) > 1 else 0
            if best is None or depth < best[0]:
                best = (depth, index, replace[index], dropped)
            del replace[index]
            drop.difference_update(dropped)
        _, index, gates_at_host, dropped = best
        replace[index] = gates_at_host
        drop.update(dropped)
    return build()


def _fold_pass(n, gates, config):
    """Cheapest of {unchanged, fold to first, fold to last} occurrence, plus
    fold to the lowest-depth occurrence when depth dominates the weights."""
    options = [list(gates)]
    hosts = ("first", "last", "depth") if _depth_focus(config) else ("first", "last")
    for host in hosts:
        options.append(_stage12(n, _phase_fold(gates, host)))
    return min(enumerate(options), key=lambda io: (_cost_key(io[1], config), io[0]))[1]


# --- exact 2-qubit Clifford block resynthesis ---------------------------------
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
    """key -> list of (label, gates); label = (n2, n1, depth0, depth1). Pareto.
    About 1.8 s to build, once per process (lru_cache), on first use: by
    _block_pass or the T-block pass, so never for one-qubit circuits."""
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
    """Weighted cost of a 2-qubit circuit label (n2, n1, depth0, depth1)."""
    n2, n1, a0, a1 = label
    return config["w1"] * n1 + config["w2"] * n2 + config["wd"] * max(a0, a1)


def _local_label(gates):
    """Label (n2, n1, depth0, depth1) of a gate list on local qubits 0, 1."""
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


def _pair_blocks(gates, a, b, clifford_only=False):
    """Maximal runs (list order, skipping gates on other wires) of gates acting
    only within {a, b}. With clifford_only, a T/TDG gate also ends a block.
    Yields lists of indices (blocks of at least 2 gates)."""
    block = []
    for i, (name, qubits) in enumerate(gates):
        touches = [q for q in qubits if q in (a, b)]
        if not touches:
            continue
        if (clifford_only and name in ("T", "TDG")) or len(touches) < len(qubits):
            if len(block) >= 2:
                yield block
            block = []
        else:
            block.append(i)
    if len(block) >= 2:
        yield block


def _block_pass(n, gates, config):
    """Replace 2-qubit Clifford blocks by their cheapest exact implementation.
    The table is fetched only when a block exists, but _t_block_pass also
    needs it, so in practice it is built on the first multi-qubit circuit."""
    gates = list(gates)
    changed = True
    while changed:
        changed = False
        base = _key(gates, config)
        for a in range(n):
            for b in range(a + 1, n):
                for block in _pair_blocks(gates, a, b, clifford_only=True):
                    local = {a: 0, b: 1}
                    seq = [(gates[i][0], tuple(local[q] for q in gates[i][1])) for i in block]
                    options = _cliff2_table()[_cliff2_key(seq)]
                    label, best = min(options, key=lambda e: (_label_cost(e[0], config), e[0]))
                    if _label_cost(label, config) >= _label_cost(_local_label(seq), config):
                        continue
                    back = {0: a, 1: b}
                    new = [(g, tuple(back[q] for q in qs)) for g, qs in best]
                    out = _splice(gates, set(block), block[-1], new)
                    if _key(out, config) < base:
                        gates = out
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return gates


# --- exact small-region synthesis, GraySynth, convex regions ------------------

def _gf2_mul(a, b):
    """Row-bitmask matrix product a*b over GF(2)."""
    out = []
    for row in a:
        acc = 0
        k = 0
        while row:
            if row & 1:
                acc ^= b[k]
            row >>= 1
            k += 1
        out.append(acc)
    return out


def _gf2_inv(rows):
    """Inverse over GF(2). _linear_ops_gauss returns the elimination steps in
    build order; applied in elimination order to the identity they give A^-1."""
    inv = [1 << k for k in range(len(rows))]
    for c, t in reversed(_linear_ops_gauss(rows)):
        inv[t] ^= inv[c]
    return inv


@functools.lru_cache(maxsize=None)
def _exact_small_paths(m, needed):
    """Shortest CX sequences (m <= 3 wires) that make every parity in `needed`
    appear on some wire at some moment, for every final linear map.

    Breadth-first search over (wire contents, parities not yet seen): at most
    168 x 2^7 = 21,504 states for 3 wires. Returns {final rows: move list}.
    """
    moves = [(c, t) for c in range(m) for t in range(m) if c != t]

    def absorb(rows, rem):
        return frozenset(y for y in rem if y not in rows)

    start_rows = tuple(1 << i for i in range(m))
    start = (start_rows, absorb(start_rows, needed))
    parent = {start: None}
    queue = deque([start])
    done = {}
    while queue:
        state = queue.popleft()
        rows, rem = state
        if not rem and rows not in done:
            done[rows] = state
        for c, t in moves:
            nxt = list(rows)
            nxt[t] ^= nxt[c]
            nxt = tuple(nxt)
            child = (nxt, absorb(nxt, rem))
            if child not in parent:
                parent[child] = (state, (c, t))
                queue.append(child)
    paths = {}
    for rows, state in done.items():
        path = []
        while parent[state] is not None:
            state, move = parent[state]
            path.append(move)
        paths[rows] = tuple(path[::-1])
    return paths


def _emit_along(path, terms, b, wires):
    """The path's circuit if every phase is reached along it, else None."""
    out, pending = _walk_cx_path(path, terms, wires)
    return None if pending else out + _flips(b, wires)


def _reconstruct_exact_small(rows, b, terms, wires):
    """CX-optimal for this term list (m <= 3): exact search, not greedy."""
    path = _exact_small_paths(len(wires), frozenset(terms)).get(tuple(rows))
    return None if path is None else _emit_along(path, terms, b, wires)


def _reconstruct_graysynth(rows, b, terms, wires):
    """GraySynth-style parity network (Amy, Azimzadeh & Mosca, arXiv:1712.01859).

    Parities are tracked in CURRENT-wire coordinates: column y means "XOR of
    the wires whose bits are set". CX(control j, target i) updates every
    column by col_j ^= col_i, and a column equal to a single bit e_k means
    wire k holds that parity, so its phase is applied there. Columns are split
    recursively on the row that best separates them (Gray-code idea), and each
    group is collapsed onto one target wire.
    """
    m = len(wires)
    cols = {y: y for y in terms}            # term -> current coordinates
    cur = [1 << i for i in range(m)]
    out = []

    def place():
        for y in sorted(cols):
            col = cols[y]
            if col and col & (col - 1) == 0:
                k = col.bit_length() - 1
                out.extend((nm, (wires[k],)) for nm in _PHASE_OUT[terms[y]])
                del cols[y]

    def cx(j, i):
        out.append(("CX", (wires[j], wires[i])))
        cur[i] ^= cur[j]
        for y in cols:
            if (cols[y] >> i) & 1:
                cols[y] ^= 1 << j

    place()
    stack = [(tuple(sorted(terms)), tuple(range(m)), None)]
    while stack:
        group, free, target = stack.pop()
        group = tuple(y for y in group if y in cols)
        if not group:
            continue
        if target is not None:
            changed = True
            while changed:
                changed = False
                for j in range(m):
                    if j != target and group and all((cols[y] >> j) & 1 for y in group):
                        cx(j, target)
                        place()
                        group = tuple(y for y in group if y in cols)
                        changed = True
                        break
            if not group:
                continue
        if not free:
            continue
        j = max(free, key=lambda r: (max(sum((cols[y] >> r) & 1 for y in group),
                                         sum(1 - ((cols[y] >> r) & 1) for y in group)), -r))
        rest = tuple(r for r in free if r != j)
        zeros = tuple(y for y in group if not (cols[y] >> j) & 1)
        ones = tuple(y for y in group if (cols[y] >> j) & 1)
        stack.append((zeros, rest, target))
        stack.append((ones, rest, j if target is None else target))
    if cols:
        return None
    fix = _gf2_mul(list(rows), _gf2_inv(cur))          # M with M * cur = A
    out += [("CX", (wires[c], wires[t])) for c, t in _linear_ops_gauss(fix)]
    return out + [("X", (wires[i],)) for i in range(m) if (b >> i) & 1]


# --- convex H-free regions ------------------------------------------------------

def _reach_masks(gates):
    """Descendant and ancestor bitmasks for every gate (wire dependencies)."""
    count = len(gates)
    nxt = _wire_successors(gates)
    desc = [0] * count
    for i in range(count - 1, -1, -1):
        for s in nxt[i]:
            desc[i] |= (1 << s) | desc[s]
    anc = [0] * count
    for i in range(count):
        for s in nxt[i]:
            anc[s] |= (1 << i) | anc[i]
    return desc, anc


def _convex_regions(gates):
    """Greedy maximal convex H-free regions (lists of indices, in order).

    A region is convex when no dependency path leaves it and comes back in:
    no outside gate is both a descendant and an ancestor of region gates.
    Regions grow from the earliest unassigned non-H gate, adding wire
    neighbours (before or after) in index order while convexity holds.
    """
    desc, anc = _reach_masks(gates)
    assigned = [False] * len(gates)
    neighbours = [set() for _ in gates]
    for q in sorted({q for _, qs in gates for q in qs}):
        on_wire = [i for i, (_, qs) in enumerate(gates) if q in qs]
        for x, y in zip(on_wire, on_wire[1:]):
            neighbours[x].add(y)
            neighbours[y].add(x)
    regions = []
    for i, (name, _) in enumerate(gates):
        if assigned[i] or name == "H":
            continue
        members, mask, d, a = [i], 1 << i, desc[i], anc[i]
        assigned[i] = True
        heap = sorted(neighbours[i])
        heapq.heapify(heap)
        while heap:
            j = heapq.heappop(heap)
            if assigned[j] or gates[j][0] == "H":
                continue
            m2, d2, a2 = mask | (1 << j), d | desc[j], a | anc[j]
            if d2 & a2 & ~m2:
                continue
            members.append(j)
            mask, d, a = m2, d2, a2
            assigned[j] = True
            for k in sorted(neighbours[j]):
                if not assigned[k]:
                    heapq.heappush(heap, k)
        regions.append(sorted(members))
    return regions


def _make_contiguous(gates, region):
    """Reorder (wire orders kept) so the convex region is one contiguous slice.
    Returns (new gate list, start, end)."""
    rset = set(region)
    _, anc = _reach_masks(gates)
    region_anc = 0
    for r in region:
        region_anc |= anc[r]
    before = [i for i in range(len(gates)) if i not in rset and (region_anc >> i) & 1]
    after = [i for i in range(len(gates)) if i not in rset and not (region_anc >> i) & 1]
    order = before + sorted(region) + after
    return [gates[i] for i in order], len(before), len(before) + len(region)


MAX_REGION_MOVES = 64   # accepted convex-region rewrites per pass (work budget)


def _convex_region_pass(n, gates, config):
    """Rewrite convex H-free regions (made contiguous first) when strictly cheaper."""
    gates = list(gates)
    for _ in range(MAX_REGION_MOVES):
        base = _cost_key(gates, config)
        accepted = None
        for region in _convex_regions(gates):
            if len(region) < 2:
                continue
            ordered, s, e = _make_contiguous(gates, region)
            segment = ordered[s:e]
            for cand in _region_candidates(n, segment, _wires_of(segment)):
                full = ordered[:s] + cand + ordered[e:]
                if _cost_key(full, config) < base and (
                        accepted is None or _key(full, config) < _key(accepted, config)):
                    accepted = full
            if accepted is not None:
                break
        if accepted is None:
            return gates
        gates = accepted
    _BUDGET_HITS["regions"] += 1
    return gates


# --- beam search over exact moves ---------------------------------------------
#
# Greedy passes only accept immediate improvements, so they stop in local
# minima. The beam keeps the `width` best distinct circuits at each step,
# including ones that are temporarily worse, and expands each with every move.
# Every move is an exact identity, every child is cleaned by Stages 1-2, and
# the result is verified like any other candidate.

# Beam width is a parameter of optimize (beam_width=...). The default was
# calibrated with bench.py: on the quick corpus, width 16 / patience 5 gave
# 2.6% lower cost than width 6 / patience 3 at a median of ~0.7 s per call
# (max ~10 s on the largest circuits), and never hit the evaluation budget.
# The evaluation budget scales with the width. QUBLITZ_BEAM=fast (used only
# by the bulk test suites, so they finish in minutes) sets width 4,
# patience 2 when no width is passed; a dedicated test runs the default.
DEFAULT_BEAM_WIDTH = 16
BEAM_PATIENCE = 5
EVALS_PER_WIDTH = 1000


def _beam_settings(width=None):
    """(width, patience, max evaluations) for a requested width (None = default)."""
    if width is None:
        if os.environ.get("QUBLITZ_BEAM") == "fast":
            return 4, 2, 1000
        width = DEFAULT_BEAM_WIDTH
    return width, BEAM_PATIENCE, width * EVALS_PER_WIDTH


def _canon(gates):
    """Per-wire gate sequences: equal exactly when the circuits have the same
    gates in the same order on every wire (same DAG), whatever the listing."""
    wires = {}
    for g in gates:
        for q in g[1]:
            wires.setdefault(q, []).append(g)
    return tuple((q, tuple(wires[q])) for q in sorted(wires))


def _local_moves(gates):
    """Exact local rewrites, including ones that raise cost. Fixed order."""
    out = []
    for k, (name, qubits) in enumerate(gates):
        if name == "CX":
            c, t = qubits
            nc, nt = _next_on(gates, k, c), _next_on(gates, k, t)
            pc, pt = _prev_on(gates, k, c), _prev_on(gates, k, t)
            # Pauli push: X_c CX = CX X_c X_t ; CX X_c = X_c X_t CX
            #             Z_t CX = CX Z_c Z_t ; CX Z_t = Z_c Z_t CX
            if pc is not None and gates[pc] == ("X", (c,)):
                out.append(_move(gates, drop={pc}, after={k: [("X", (c,)), ("X", (t,))]}))
            if nc is not None and gates[nc] == ("X", (c,)):
                out.append(_move(gates, drop={nc}, before={k: [("X", (c,)), ("X", (t,))]}))
            if pt is not None and gates[pt] == ("Z", (t,)):
                out.append(_move(gates, drop={pt}, after={k: [("Z", (c,)), ("Z", (t,))]}))
            if nt is not None and gates[nt] == ("Z", (t,)):
                out.append(_move(gates, drop={nt}, before={k: [("Z", (c,)), ("Z", (t,))]}))
            # CX reversal: CX(c,t) = H_c H_t CX(t,c) H_c H_t
            hh = [("H", (c,)), ("H", (t,))]
            out.append(_move(gates, replace={k: hh + [("CX", (t, c))] + hh}))
        elif name in ("H", "S", "SDG"):
            q = qubits[0]
            j = _next_on(gates, k, q)
            l = _next_on(gates, j, q) if j is not None else None
            if l is not None and len(gates[j][1]) == 1 and len(gates[l][1]) == 1:
                triple = (name, gates[j][0], gates[l][0])
                swap = {("H", "S", "H"): ("SDG", "H", "SDG"), ("SDG", "H", "SDG"): ("H", "S", "H"),
                        ("H", "SDG", "H"): ("S", "H", "S"), ("S", "H", "S"): ("H", "SDG", "H")}
                if triple in swap:
                    new = list(gates)
                    for idx, nm in zip((k, j, l), swap[triple]):
                        new[idx] = (nm, (q,))
                    out.append(new)
        # Commute-swap with the next gate on a shared wire (changes the DAG).
        for q in qubits:
            j = _next_on(gates, k, q)
            if j is not None and _commutes(gates[k], gates[j]):
                # gates[j] must not depend on anything between k and j on its other wires
                blocked = any(_prev_on(gates, j, w) not in (k, None) and _prev_on(gates, j, w) > k
                              for w in gates[j][1])
                if not blocked:
                    new = list(gates)
                    new.insert(k, new.pop(j))
                    out.append(new)
    return out


def _move(gates, drop=(), before=None, after=None, replace=None):
    """Copy of `gates` with indices dropped, or gates inserted before/after or
    replacing given indices."""
    before, after, replace = before or {}, after or {}, replace or {}
    out = []
    for i, g in enumerate(gates):
        if i in drop:
            continue
        out.extend(before.get(i, []))
        out.extend(replace.get(i, [g]))
        out.extend(after.get(i, []))
    return out


# --- two-qubit blocks that contain T gates -------------------------------------
#
# A block on qubits (a, b) with T gates is written exactly as
#     U = C_final * R_m ... R_1        (rotations act first)
# where each R_j is a pi/4-multiple rotation about a two-qubit Pauli P_j (the
# Pipeline B view, restricted to the pair) and C_final is Clifford. Rotations
# about the same axis merge. Each rotation is rebuilt as  V^dagger, T^k on q,
# V  with V a Clifford sending Z_q to P_j, so the block becomes
#     [Clifford] T [Clifford] T ... [Clifford]
# and every Clifford stretch is replaced by its cheapest implementation from
# the exact two-qubit Clifford table. The qubit q that carries each T is
# chosen by trying every combination (up to 2^TBLOCK_MAX_CHOICES).

TBLOCK_MAX_CHOICES = 6


def _inverse_gates(gates):
    inv = {"S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T"}
    return [(inv.get(g, g), q) for g, q in reversed(gates)]


@functools.lru_cache(maxsize=None)
def _z_basis_changers():
    """(q, signed Pauli) -> cheapest 2-qubit Clifford circuit V with
    V Z_q V^dagger = that Pauli. Taken from the exact Clifford table."""
    out = {}
    for key, entries in _cliff2_table().items():
        label, gates = min(entries, key=lambda e: (e[0][0], e[0][1], max(e[0][2:])))
        for q, image in ((0, key[1]), (1, key[3])):
            cost = (label[0], label[1])
            if (q, image) not in out or cost < out[(q, image)][0]:
                out[(q, image)] = (cost, tuple(gates))
    return {k: v[1] for k, v in out.items()}


def _best_clifford(seq, config):
    """Cheapest table implementation of a 2-qubit Clifford gate list."""
    if not seq:
        return []
    label, gates = min(_cliff2_table()[_cliff2_key(seq)],
                       key=lambda e: (_label_cost(e[0], config), e[0]))
    return list(gates)


def _resynth_t_block(local, config):
    """Exact cheaper rewrite of a 2-qubit gate list (local qubits 0, 1), or None.
    Cached on (gates, weights): beam states share most blocks."""
    out = _resynth_t_block_cached(tuple(local), config["w1"], config["w2"], config["wd"])
    return None if out is None else list(out)


@functools.lru_cache(maxsize=16384)
def _resynth_t_block_cached(local, w1, w2, wd):
    config = {"w1": w1, "w2": w2, "wd": wd}
    rotations, clifford = _to_rotations(local)
    rotations = _merge_rotations(rotations)
    changers = _z_basis_changers()
    odd = [i for i, (_, k) in enumerate(rotations) if k % 2]
    choices = [()] if not odd else (
        itertools.product((0, 1), repeat=len(odd))
        if len(odd) <= TBLOCK_MAX_CHOICES else [tuple(0 for _ in odd)])
    best = None
    for choice in choices:
        pick = dict(zip(odd, choice))
        stream = []
        for i, (p, k) in enumerate(rotations):
            q = pick.get(i, 0)
            v = list(changers[(q, p)])
            stream += _inverse_gates(v)
            stream += [(name, (q,)) for name in _PHASE_OUT[k]]
            stream += v
        stream += clifford
        out, segment = [], []
        for g in stream:
            if g[0] in ("T", "TDG"):
                out += _best_clifford(segment, config)
                out.append(g)
                segment = []
            else:
                segment.append(g)
        out += _best_clifford(segment, config)
        if best is None or _label_cost(_local_label(out), config) < _label_cost(_local_label(best), config):
            best = out
    if best is None or _label_cost(_local_label(best), config) >= _label_cost(_local_label(local), config):
        return None
    return tuple(best)


def _t_block_pass(n, gates, config):
    """Rewrite 2-qubit blocks containing T gates when strictly cheaper."""
    gates = list(gates)
    for _ in range(MAX_REGION_MOVES):
        base = _key(gates, config)
        accepted = None
        for a in range(n):
            for b in range(a + 1, n):
                for block in _pair_blocks(gates, a, b):
                    if not any(gates[i][0] in ("T", "TDG") for i in block):
                        continue   # pure Clifford blocks: handled by _block_pass
                    local = {a: 0, b: 1}
                    seq = [(gates[i][0], tuple(local[q] for q in gates[i][1])) for i in block]
                    new_local = _resynth_t_block(seq, config)
                    if new_local is None:
                        continue
                    if not _fast_equivalent(_fast_unitary(2, seq), 2, new_local):
                        if _strict():
                            raise AssertionError("T-block rewrite changed the unitary")
                        continue
                    back = {0: a, 1: b}
                    new = [(g, tuple(back[q] for q in qs)) for g, qs in new_local]
                    out = _splice(gates, set(block), block[-1], new)
                    if _key(out, config) < base:
                        accepted = out
                        break
                if accepted:
                    break
            if accepted:
                break
        if accepted is None:
            return gates
        gates = accepted
    _BUDGET_HITS["tblocks"] += 1
    return gates


# Whole-pass moves, each applied once to a state. A cheap applicability check
# skips moves that provably cannot change the circuit (e.g. folding with fewer
# than two phase gates); the skip is exact, so it never loses a result.
# (Tried and measured, then removed: (a) applying each pass as a closure so
# that re-applying it is provably useless and can be skipped; (b) ranking
# children by their cost after a follow-up block pass. On the bench corpus
# (a) gave the same results ~45% slower and (b) gave worse results.)
_PASS_MOVES = (
    ("fold-first", lambda n, g, c: _phase_fold(g, "first")),
    ("fold-last", lambda n, g, c: _phase_fold(g, "last")),
    ("fold-depth", lambda n, g, c: _phase_fold(g, "depth")),
    ("h_reduce", lambda n, g, c: _h_reduce(g, c)),
    ("regions", lambda n, g, c: _region_pass(n, g, c)),
    ("convex_regions", lambda n, g, c: _convex_region_pass(n, g, c)),
    ("cross_regions", lambda n, g, c: _cross_region_pass(n, g, c)),  # beam-only
    ("blocks", lambda n, g, c: _block_pass(n, g, c)),
    ("tblocks", lambda n, g, c: _t_block_pass(n, g, c)),
    ("pipeline_b", lambda n, g, c: _pipeline_b(n, g)),
)


def _pass_applicable(name, gates, config):
    """False only when the move provably cannot change the circuit, or (for
    depth-only moves) when depth does not dominate the weights."""
    names = [g for g, _ in gates]
    if name == "fold-depth" and not _depth_focus(config):
        return False
    if name.startswith("fold"):
        return sum(1 for g in names if g in _PHASE_K) >= 2
    if name == "h_reduce":
        return "H" in names
    if name in ("pipeline_b", "tblocks"):
        return "T" in names or "TDG" in names
    if name in ("blocks", "cross_regions"):
        return "CX" in names
    return len(gates) >= 2


def _beam_children(n, state, config):
    """Every applicable whole-pass move and local move, each cleaned up."""
    out = [move(n, state, config) for name, move in _PASS_MOVES
           if _pass_applicable(name, state, config)]
    return [_stage12(n, child) for child in out + _local_moves(state)]


def _beam_search(n, start, config, beam_width=None):
    """Keep the `width` cheapest distinct circuits per step; stop after
    `patience` steps without a new best, or when the work budget is spent."""
    width, patience, max_evals = _beam_settings(beam_width)
    best = list(start)
    beam = [best]
    seen = {_canon(best)}
    evals = 0
    stale = 0
    while beam and stale < patience:
        children = []
        for state in beam:
            for child in _beam_children(n, state, config):
                key = _canon(child)
                if key in seen:
                    continue
                seen.add(key)
                children.append(child)
                evals += 1
                if evals >= max_evals:
                    break
            if evals >= max_evals:
                break
        children.sort(key=lambda c: _key(c, config))
        beam = children[:width]
        if beam and _cost_key(beam[0], config) < _cost_key(best, config):
            best, stale = beam[0], 0
        else:
            stale += 1
        if evals >= max_evals:
            _BUDGET_HITS["beam"] += 1
            break
    return best


# --- the shared, convergent optimization loop (used by every candidate) -------------

def _full_optimize(n, gates, config, reorder_regions=True):
    """The greedy loop used by every candidate: Stages 1-2, phase folding,
    H-reduction, regions (list and convex) and 2-qubit Clifford blocks, plus
    any passes switched on in FEATURES, repeated until the cost tuple stops
    dropping. (Cross-region moves and T-blocks are beam moves by default.)
    reorder_regions=False makes the list region pass keep the given listing
    instead of reordering for the fewest regions (the fifth candidate)."""
    gates = _stage12(n, gates)
    for _ in range(MAX_ROUNDS):
        g = gates
        if FEATURES["fold"]:
            g = _stage12(n, _fold_pass(n, g, config))
        if FEATURES["h_reduce"]:
            g = _stage12(n, _h_reduce(g, config))
        if FEATURES["regions"]:
            g = _stage12(n, _region_pass(n, g, config, reorder_regions))
        if FEATURES["convex_regions"]:
            g = _stage12(n, _convex_region_pass(n, g, config))
        if FEATURES["cross_regions_greedy"]:
            g = _stage12(n, _cross_region_pass(n, g, config))
        if FEATURES["blocks"]:
            g = _stage12(n, _block_pass(n, g, config))
        if FEATURES["tblocks_greedy"]:
            g = _stage12(n, _t_block_pass(n, g, config))
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