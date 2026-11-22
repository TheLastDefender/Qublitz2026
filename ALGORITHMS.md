# Algorithms: precise reference

This file specifies every algorithm in `submission.py`: precise pseudocode,
every parameter and heuristic, and the reason for each choice (with the
measurement behind it where one exists). README §4 is the narrative overview;
this is the reference. Function names match the code.

Measurements are on the benchmark corpus (`bench.corpus(8)`: 35 circuits × 4
weight profiles = 140 cases) unless stated otherwise. "w2-heavy" = (1, 6, 0.2),
"balanced" = (1, 1, 1), "w1-heavy" = (1, 0, 0), "wd-heavy" = (0, 0, 1).

## Contents

0. [Conventions](#0-conventions)
1. [Top level: `optimize`](#1-top-level-optimize)
2. [One selection round: `_select`](#2-one-selection-round-_select)
3. [Cleanup: `_stage12`](#3-cleanup-_stage12)
4. [One-qubit runs: exact keys, table, best split, normal form](#4-one-qubit-runs)
5. [The greedy loop: `_full_optimize`](#5-the-greedy-loop-_full_optimize)
6. [Phase folding](#6-phase-folding)
7. [H-reduction](#7-h-reduction)
8. [Phase polynomials and region rebuilding](#8-phase-polynomials-and-region-rebuilding)
9. [Finding regions: listings, convex regions, cross-region moves](#9-finding-regions)
10. [Two-qubit Clifford table and block pass](#10-two-qubit-clifford-table-and-block-pass)
11. [Pipeline B: Pauli rotations](#11-pipeline-b-pauli-rotations)
12. [T-blocks](#12-t-blocks)
13. [Beam search](#13-beam-search)
14. [Verification](#14-verification)
15. [Termination, determinism, idempotence](#15-termination-determinism-idempotence)
16. [All parameters](#16-all-parameters)

---

## 0. Conventions

**Circuit.** A list of `(name, qubits)` tuples in time order (entry 0 acts
first); `qubits` is `(q,)` or `(control, target)`. Only gates sharing a qubit
have a meaningful order; `_canon` (one gate sequence per wire) identifies two
listings of the same circuit.

**Matrices.** Qubit 0 is the most significant bit of a basis index. A circuit
`g1 … gm` has unitary `U = Gm ⋯ G1`.

**Phase exponents.** Every phase gate is `T^k`, k mod 8:

| Gate | T | S | Z | SDG | TDG |
|---|---|---|---|---|---|
| k | 1 | 2 | 4 | 6 | 7 |

`_PHASE_OUT[k]` is the shortest spelling: 0 → nothing, 1 → T, 2 → S,
3 → S T, 4 → Z, 5 → Z T, 6 → SDG, 7 → TDG.

**Cost and order.**
```text
_metrics_pairs(g) = (n1, n2, depth)        # same semantics as circuit_metrics
_cost_key(g)      = (w1·n1 + w2·n2 + wd·depth, n2, depth, n1)
_key(g)           = _cost_key(g) + (tuple(g),)   # total order: ties broken by the gate list
```
"Strictly cheaper" always means a strictly smaller `_cost_key`.

**Exactness rule.** Every rewrite is an identity up to global phase
(the 23 identities of README §5). Nothing approximate is ever accepted.

**Determinism rule.** No randomness, no wall-clock limits, no output that
depends on set iteration order, every tie broken explicitly. Limits are work
budgets (counts), listed in §16.

---

## 1. Top level: `optimize`

```text
optimize(circuit, config, beam_width=None):
    validate_circuit(circuit); validate_config(config)
    n         = circuit.n_qubits
    original  = as_pairs(circuit)
    reference = _fast_unitary(n, original)            # computed once
    current   = original
    repeat at most MAX_SELECT_ROUNDS (8) times:
        best = _select(n, reference, current, config, beam_width)
        if best == current: break                     # fixed point reached
        current = best
    else: _BUDGET_HITS["select"] += 1
    result = from_pairs(n, current)
    if not equivalent_up_to_global_phase(circuit, result):   # organizers' function
        if strict mode: raise
        return deep copy of circuit
    return result
```

**Why.**
- *Repeat until a round returns its input*: this makes the output a fixed point
  of `_select`, so `optimize(optimize(c)) == optimize(c)` (§15).
- *Final check with the organizers' own function*: the output is judged by
  their code, not ours. A failure can only be a bug; returning the input keeps
  the result valid and never costlier.
- *Measured*: at most 2 selection rounds have ever been needed.

---

## 2. One selection round: `_select`

```text
_select(n, reference, start, config, beam_width):
    s2 = _stage12(n, start)
    A  = _full_optimize(n, s2, config)
    candidates = [start, A]
    if FEATURES.pipeline_b:
        candidates += [_full_optimize(n, _pipeline_b(n, s2), config),     # B
                       _full_optimize(n, _pipeline_b(n, A),  config)]     # B after A
    candidates += [_full_optimize(n, s2, config, reorder_regions=False)]  # A, given listing
    verified = [(i, c) for i, c in enumerate(candidates)
                if _fast_equivalent(reference, n, c)]       # strict mode: raise on failure
    best = argmin over verified of (_cost_key(c), i)        # ties: earliest candidate
    if FEATURES.beam and beam width > 0:
        beamed = _beam_search(n, best, config, beam_width)
        if _cost_key(beamed) < _cost_key(best) and _fast_equivalent(reference, n, beamed):
            return beamed
    return best
```

**Why each candidate** (first round, before beam search, 140 cases):

| Candidate | Only cheapest candidate in | Reason it exists |
|---|---|---|
| start | 0 | guarantees "never worse"; ties go to it, which gives the fixed point |
| A | 46 | the main greedy path |
| B | 6 | merges T gates hidden by H gates (§11), which A cannot see |
| B after A | 11 | B applied to an already-simplified circuit finds different merges |
| A, given listing | (added later) | the fewest-regions reordering (§9.1) helps some weights and hurts others |

- *Fifth candidate, measured*: as an extra candidate it made no profile worse
  (−0.2% w2-heavy, −1.1% balanced, −0.3% w1-heavy, 0 wd-heavy; 6 cases cheaper,
  1 costlier). Replacing the reordering outright instead gave −2.0% / −1.4% /
  +2.4% / −0.4%, so it was rejected.
- *Removed candidates*: the improved Stage 1–2 result can never beat A (A starts
  from it and accepts only strict improvements); the legacy Stage 2 path was
  never cheaper than it (0 of 140). Removing both changed 7 of 140 outputs, all
  at equal cost.
- *More candidates are not free*: adding four variants at once was worse than
  adding the fifth alone, because extra ties change which circuit the beam
  starts from.
- *Verification of every candidate*: catches implementation bugs at their
  source, drops only the bad candidate, compares against the original input
  every round (no drift); costs about 0.3% of runtime.

---

## 3. Cleanup: `_stage12`

```text
_stage12(n, g):
    g = stage1(g)
    loop:
        r = _stage2_reduce(g)
        if r == g: return g
        g = stage1(r)
```
Runs after almost every other step. It only ever removes gates or replaces a
run with a shorter one, so it never raises the cost and terminates (each
iteration removes at least one gate). Under 0.2% of runtime.

### 3.1 Stage 1: per-wire stacks (`_stage1_cancel`)

```text
stacks[q] = [] for every qubit
for index, gate in enumerate(circuit):
    if gate is CX(c, t):
        if top(stacks[c]) is top(stacks[t]) and it is CX with the same (c, t):
            pop both                                    # CX CX = I
        else:
            e = (index, "CX", (c, t)); push e on stacks[c] and stacks[t]   # one shared object
    elif gate is a phase gate T^k on q:
        if top(stacks[q]) is a phase entry with exponent k0:
            pop it; k = (k + k0) mod 8
            if k != 0: push (k0's index, k, q)          # merged gate keeps the earlier slot
        else: push (index, k, q)
    else:  # H or X on q
        if top(stacks[q]) is the same gate: pop         # H H = X X = I
        else: push (index, name, q)
survivors = entries of all stacks, de-duplicated by index, sorted by index
emit each survivor; a phase entry k emits _PHASE_OUT[k]
```
"Same object on top of both stacks" means nothing touched either wire between
the two CX gates. Phases merge only when directly consecutive on a wire.

### 3.2 Stage 2 (`_stage2_reduce`)

```text
loop:
    g', changed = _phase_segment_pass(g);  if changed: g = g'; continue
    g' = _commute_cancel_pass(g);          if len(g') < len(g): g = g'; continue
    g', changed = _window_pass(g);         if changed and shorter: g = g'; continue
    return g
```

**Phase segments** (`_phase_segment_pass`, returns after the first change):
```text
for wire q in sorted order:
    segment = []
    for each gate on q, then an end marker:
        if gate is a phase gate: segment.append(it); continue
        blocker = end marker or H(q) or X(q) or CX with target q
        if not blocker: continue                 # a CX with control q: phases pass through
        if len(segment) >= 2:
            total = sum of exponents mod 8
            if len(_PHASE_OUT[total]) < len(segment):
                options = [splice(segment removed, _PHASE_OUT[total] at position p)
                           for p in segment positions]
                return argmin over options of depth (ties: earliest), changed
        segment = []
```
*Why*: summing a whole segment catches totals that no pairwise check sees
(T Z T = SDG). *Placement*: the merged phase may sit at any segment position
(all equivalent); the lowest-depth one is chosen because placing it first
raised depth from 6 to 8 in a found case.

**Commutation cancellation** (`_commute_cancel_pass`, one sweep):
```text
for i in order, if gate i is alive:
    for j > i, alive:
        if gates i and j share no qubit: continue
        if gate j == gate i and it is H, X or CX: kill both; break    # g M g = M g g = M
        if not _commutes(gate i, gate j): break                       # blocked
```
Pairwise checks suffice: if g commutes with each of M1 … Mk it commutes with
their product, so it can slide next to its partner. The check is conservative
(it stops at the first gate that does not commute on its own); whole-block
cases are left to regions, the Clifford table, folding and the beam.

**Commutation test** (`_commutes`): renumber the union of the two gates' qubits
to 0, 1, …; build both gates' matrices on those qubits with the starter code's
`_full_gate_matrix`; return `A·B ≈ B·A` (tolerance 1e-9). Cached by pattern
(gate names and renumbered qubits), so each pattern is computed once. No
commutation rule is assumed by hand.

**One-qubit runs** (`_window_pass`): for each wire in sorted order, collect the
maximal runs of consecutive one-qubit gates on it (a CX on that wire ends a
run; gates on other wires do not); for the first run of length ≥ 2 whose
`_best_run` is shorter, splice the replacement in at the run's first position.
See §4.

---

## 4. One-qubit runs

### 4.1 Exact key (`_so3_of`)

Up to global phase, a one-qubit operator U is the Bloch rotation R with
`U σj U† = Σi R[i][j] σi`; the phase cancels in U·U†, and equal R ⇔ equal U up
to phase. For this gate set every entry of R lies in Z[1/√2], so

```text
R = (1/√2)^k · N,   each entry of N = a + b√2 with integers (a, b)
```
Gate matrices: H, S, X, Z have k = 0 and integer entries; T has k = 1.
Products use exact integer arithmetic (`_zr_mul`, `_zr_add`, `_so3_mul`).

```text
_so3_reduce(k, N):
    while k > 0 and every entry's a is even:
        replace each entry (a, b) by (b, a/2)     # (a + b√2)/√2 = b + (a/2)√2
        k -= 1
```
The reduced (k, N) is unique, so it is an exact key: equal keys ⇔ same
operator up to global phase. No floating point. k is the *denominator
exponent*, which Matsumoto–Amano relate to the T-count.

### 4.2 The table (`_window_table_long`)

```text
table = {key(identity): ()}; frontier = [()]
repeat LONG_WINDOW (7) times:
    for seq in frontier, for name in sorted gate names (H, S, SDG, T, TDG, X, Z):
        new = seq + (name,)
        if key(new) not in table: table[key(new)] = new; next_frontier.append(new)
```
Breadth-first, so the first sequence stored for an operator is a shortest one.
376 operators need at most 7 gates. A key **not** in the table proves the
operator needs at least 8 gates.

### 4.3 Shortest replacement (`_best_run`, cached by sequence)

```text
_best_run(names):
    if key(names) in table: return table[key(names)]          # proven shortest
    options = [split(names)]
    nf = _ma_normal_form(names)
    if nf is not None: options.append(split(nf))
    return the shortest option (ties: the first)

split(seq):                                     # dynamic program over cut points
    best[0] = []
    for i in 1 .. len(seq):
        best[i] = shortest over j in max(0, i−7) .. i−1 of
                  best[j] + table[key(seq[j:i])]              # any ≤ 7 gates are in the table
    return best[len(seq)]
```
`split` finds the best of **all** ways to cut `seq` into pieces of at most 7
gates (the best cutting of a prefix ends with some last piece, and the part
before it is a shorter prefix whose best cutting is already known), using at
most 7 table lookups per position. *Not proven optimal*: a shorter spelling may
exist that no cutting of either starting point produces. *Certificate*: when
the result has exactly 8 gates it is optimal, since the operator needs ≥ 8.

*Why fewest gates*: on one wire every gate costs w1 and the run's length is that
wire's depth there, so fewer gates is never costlier under any weights.

### 4.4 Matsumoto–Amano normal form (`_ma_normal_form`)

```text
m = key(names); left = []
while m.k > 0:
    for (gates, S) in syllables [ (T), (T H), (T H S) ]:      # time order; operators T, H·T, S·H·T
        r = transpose(S) · m                                   # strip S from the latest end
        if r.k < m.k: left.append(gates); m = r; break
    else: return None                                          # never expected (strict: raise)
return _clifford1_table[m] + concatenation of reversed(left)
```
`_clifford1_table`: breadth-first table of the 24 one-qubit Cliffords over
H, S, SDG, X, Z. The normal form is unique and has the minimum T-count
(Matsumoto & Amano 2008; Giles & Selinger 2013). Used as a second starting
point for `split`, because its chunks can collapse in ways the original's
cannot.

---

## 5. The greedy loop: `_full_optimize`

```text
_full_optimize(n, g, config, reorder_regions=True):
    g = _stage12(n, g)
    repeat at most MAX_ROUNDS (8) times:
        h = g
        if FEATURES.fold:                 h = _stage12(n, _fold_pass(n, h, config))
        if FEATURES.h_reduce:             h = _stage12(n, _h_reduce(h, config))
        if FEATURES.regions:              h = _stage12(n, _region_pass(n, h, config, reorder_regions))
        if FEATURES.convex_regions:       h = _stage12(n, _convex_region_pass(n, h, config))
        if FEATURES.cross_regions_greedy: h = _stage12(n, _cross_region_pass(n, h, config))   # off
        if FEATURES.blocks:               h = _stage12(n, _block_pass(n, h, config))
        if FEATURES.tblocks_greedy:       h = _stage12(n, _t_block_pass(n, h, config))         # off
        if _cost_key(h) < _cost_key(g): g = h
        else: break
    else: _BUDGET_HITS["full"] += 1
    return g
```
*Greedy*: takes every improvement immediately, never a worse step.

*Pass order* (folding → H-reduction → list regions → convex regions → blocks):
chosen by reasoning — folding is cheap and removes phase gates; fewer H gates
make bigger regions; the convex pass catches regions the list pass misses;
blocks then work on the CX structure the rebuilds leave. *Measured*: regions
first gave identical outputs; blocks first and the reversed order were slightly
worse. Beam search applies any pass in any order, which compensates for the
fixed order.

*Off in the greedy loop, on as beam moves*: cross-region moves (greedy use made
results worse: 0.725 → 0.718 of Stage 2 cost when removed) and T-blocks
(beam-only measured better: w1-heavy 9.51 → 9.40, balanced 23.43 → 23.23),
because an immediately accepted move can block a better later one.

*Measured rounds*: at most 2 improving rounds.

---

## 6. Phase folding

### 6.1 `_phase_fold(g, host)` (Amy, Maslov & Mosca 2014)

```text
value[q] = (mask, const)   # parity of path variables, plus a flip bit
next_var = 0; groups = {}  # parity mask -> [(index, const, signed exponent)]
for i, gate in enumerate(g):
    for q in gate.qubits not yet seen: value[q] = (1 << next_var, 0); next_var += 1
    CX(c, t): value[t] = value[t] XOR value[c]              # masks and consts
    X(q):     value[q].const ^= 1
    H(q):     value[q] = (1 << next_var, 0); next_var += 1   # fresh path variable
    T^k on q: (mask, const) = value[q]
              groups[mask].append((i, const, −k if const else k))
for each group (insertion order) with ≥ 2 occurrences:
    total = sum of signed exponents mod 8
    hosts = [first occurrence] | [last occurrence] | all occurrences (host == "depth")
    for each host h: replace gate h by _PHASE_OUT[(−total) mod 8 if h.const else total]
                     on h's wire, delete the group's other gates
    keep the host with the lowest circuit depth (only computed for "depth"; ties: earliest)
return the rebuilt list
```
*Why it is exact*: the circuit's amplitudes are a sum over path variables of
ω^P, and P adds k·(parity) for each phase gate; P is a sum, so all phase gates
on the same parity combine at any one of their positions. A constant bit 1
means the gate acts on (1 ⊕ p), and k(1 − p) = k − kp, the k being global
phase, hence the sign flip.

### 6.2 `_fold_pass`

```text
options = [g] + [_stage12(n, _phase_fold(g, host))
                 for host in ("first", "last") + (("depth",) if _depth_focus(config) else ())]
return argmin over options of (_cost_key, index)
_depth_focus(config) = FEATURES.depth_fold and wd > w1 + w2
```
*Why three placements*: where the merged gate lands decides which later
rewrites are possible. In `CX01 T1 CX01 H2 CX12 H2 CX10 T0` (cost 29.2) the
first position blocks a CX01 pair (28.0); the last lets it cancel (15.8).
*Why the depth rule*: depth placement improved wd-heavy by 2.2% but made
balanced 0.9% worse when always on.

---

## 7. H-reduction

### 7.1 Rules (`_h_rules_at(g, k)`, in this order)

With `prev/next on q` = the nearest gate on wire q before/after index k:

| Anchor at k | Condition | Rewrite | H count |
|---|---|---|---|
| CX(c, t) | H on c and on t immediately before and after | CX(t, c), the four H removed | −4 |
| CX(c, t) | X on c immediately before and after | remove both X, add X on t after the CX | 0 (one gate fewer) |
| CX(c, t) | Z on t immediately before and after | remove both Z, add Z on c before the CX | 0 (one gate fewer) |
| H on q | next gate on q is S (one-qubit), then H on q | SDG H SDG | −1 |
| H on q | next is SDG, then H | S H S | −1 |
| H on q | next is X (or Z), then H | Z (or X), both H removed | −2 |

### 7.2 Pass (`_h_reduce`)

```text
score = (h_count(g), _cost_key(g))
loop:
    for k in 0 .. len(g)−1:
        for cand in _h_rules_at(g, k):
            if (h_count(cand), _cost_key(cand)) < score: g, score = cand, its score; restart loop
    no rewrite accepted: return g
```
*Why H count first*: every H ends an H-free region; fewer H gates mean bigger
regions for §8. Terminates because the score strictly decreases.

---

## 8. Phase polynomials and region rebuilding

### 8.1 Representation (`_region_build`, `_phase_table`, `_region_canon`)

An H-free region on local wires 0 … m−1 maps `|x⟩ → ω^f(x) |Ax ⊕ b⟩`,
ω = e^(iπ/4).

```text
rows = [1 << i for i in 0..m−1]; b = 0; terms = {}
for gate in region:
    CX(c, t): rows[t] ^= rows[c]; if bit c of b: b ^= 1 << t
    X(q):     b ^= 1 << q
    T^k on q: k' = −k if bit q of b else k                  # sign rule
              terms[rows[q]] = (terms[rows[q]] + k') mod 8
drop zero terms
_phase_table(m, terms)[x] = Σ_y c_y · parity(y AND x) mod 8, for x in 0 .. 2^m − 1
_region_canon = (rows, b, phase table)        # f(0) = 0: the global phase is gone
```
Equal canonical forms ⇔ equal unitaries up to global phase. This is the exact
check every region rewrite must pass.

### 8.2 Term lists

**Canonical terms** (`_canonical_terms`, from the table alone):
```text
a = table; for each bit i, for each x with bit i set: a[x] = (a[x] − a[x XOR bit i]) mod 8
                                                   # Möbius transform: f = Σ_S a_S Π_{i∈S} x_i
for each monomial S with a_S ≠ 0:
    scale = 2^(|S|−1); if a_S mod scale ≠ 0: return None
    for each nonempty T ⊆ S: terms[T] += (+1 if |T| odd else −1) · a_S/scale   (mod 8)
return terms if their table equals the input table, else None
```
Uses 2^(d−1)·x1⋯xd = Σ_{∅≠T⊆S} (−1)^(|T|+1) (⊕_{i∈T} x_i). Unique, so a region
equal to the identity gets no terms and rebuilds to nothing.

**Z-variants** (`_z_variants`):
```text
variants = [terms]
for selection in ((4,), (3, 4, 5)):
    chosen = terms whose coefficient is in selection (sorted)
    if none: continue
    new = terms with 4 added to each chosen term; combined = XOR of chosen parities
    if combined ≠ 0: new[combined] += 4
    drop zeros; add new if not already present
```
Exact because Σ 4·p_i ≡ 4·(⊕ p_i) mod 8 (the case of two terms is
4a + 4b ≡ 4(a ⊕ b)), so the added amount is ≡ 0.

### 8.3 Synthesizers

**Gadget** (`_gadget(y, c)`): parity tree of disjoint CX gates collecting the
wires of y onto its first wire (depth ⌈log2 |y|⌉, same CX count as a ladder),
then `_PHASE_OUT[c]` on that wire, then the tree reversed.

| Synthesizer | Pseudocode | Scope |
|---|---|---|
| `_reconstruct_gauss` | gadget for every term (sorted), then the CX gates of `_linear_ops_gauss(rows)`, then X for each bit of b | any width |
| `_reconstruct_bfs` | path = `_linear_bfs(m)[rows]`; walk it from the identity, applying each term the moment its parity sits on a wire (lowest wire first); unreached terms become gadgets at the start; X gates for b at the end | m ≤ 4 |
| `_reconstruct_exact_small` | path = `_exact_small_paths(m, terms)[rows]`; walk it as above; None if a term is not reached | m ≤ 3 |
| `_reconstruct_graysynth` | see below | any width |

`_linear_ops_gauss(rows)`: Gauss–Jordan elimination to the identity, recording
each row operation as a CX; reversed, they build A from I.

`_linear_bfs(m)` (cached per m): breadth-first search from the identity over all
CX moves (c, t) in fixed order, with parent pointers; gives a shortest CX
sequence for every invertible m × m bit matrix (20,160 of them for m = 4).

`_exact_small_paths(m, needed)` (cached):
```text
state = (rows, parities in needed not yet seen on any wire)
start = (identity, needed minus those present in the identity's rows)
breadth-first search over CX moves; each move updates rows and removes newly seen parities
done[rows] = the first state reached with nothing left to see
return {rows: path to done[rows]}
```
At most 168 × 2^7 = 21,504 states for 3 wires. Gives the minimum number of CX
gates that both visits every needed parity and ends at A: CX-optimal for that
term list.

`_reconstruct_graysynth` (Amy, Azimzadeh & Mosca 2018):
```text
cols[y] = y for each term     # parity in current-wire coordinates
place(): every column equal to a single bit e_k: apply its phase on wire k, delete it
cx(j, i): CX(control j, target i); cur[i] ^= cur[j]; every column with bit i gets bit j toggled
place()
stack = [(all terms, all wires, no target)]
while stack:
    group, free, target = pop; keep only terms still pending
    if target is set:
        while some wire j ≠ target has bit j set in every column of the group:
            cx(j, target); place(); refresh group
    if group empty or free empty: continue
    j = wire in free maximising max(#ones, #zeros) of row j over the group (ties: lowest j)
    push (columns with bit j = 0, free − j, target)
    push (columns with bit j = 1, free − j, j if target unset else target)
if any term left: return None
fix = A · cur⁻¹ (over GF(2)); append the Gauss–Jordan CX gates for fix; X gates for b
```

### 8.4 `_region_candidates(n, region, wires)` (cached)

```text
(rows, b, terms) = _region_build(region); canon = (rows, b, table)
bases    = [terms] + ([canonical] if canonical exists and differs)
variants = union of _z_variants(base) over bases, in order
raw      = [region]
for v in variants:
    raw += [gauss(v)] + ([bfs(v)] if m ≤ 4) + ([exact_small(v)] if m ≤ 3) + [graysynth(v)]
out = [region]
for cand in raw, skipping None:
    cand = _stage12(n, cand)
    keep it only if it has no H gate and _region_canon(cand) == canon   # strict: raise otherwise
return out
```
*Why so many*: different term lists and synthesizers win on different regions,
and the cheapest is judged in the context of the whole circuit (§9.1).
*Why exact search only up to 3 wires*: the same search at 4 wires has about
20,160 × 2^15 ≈ 660 million states. *Rejected*: a depth-optimal CX-network
candidate (+0.6% w1-heavy, −0.9% w2-heavy even at equal CX count).

---

## 9. Finding regions

### 9.1 List regions (`_region_order`, `_region_pass`)

```text
_h_delay_order(g, eager):              # topological re-sort keeping every wire's order
    ready_other, ready_h = min-heaps of ready gate indices (non-H, H)
    while any ready:
        if ready_other and not (eager and ready_h): emit the smallest ready non-H gate
        else: emit every ready H gate at once
_region_order(g) = argmin over [g, delay(g), eager(g)] of (number of H-free spans, index)

_region_pass(n, g, config, reorder=True):
    if reorder: g = _region_order(g)
    for each maximal H-free span (s, e), right to left:
        g = argmin over cand in _region_candidates(g[s:e]) of _key(g[:s] + cand + g[e:])
    return g
```
*Notes*: the choice uses `_key`, so equal-cost alternatives are picked by the
gate list (deterministic), and an equal-cost rewrite can set up later gains.
Neither ordering rule is always best: `CX02 H0 CX21` needs H delayed,
`CX10 H2 CX12 T1 CX21` needs H early. *Rejected*: choosing the listing by cost
right after the region pass (−0.2% / −1.5% / +1.5% / 0), because the final
result depends on later steps.

### 9.2 Convex regions (`_reach_masks`, `_convex_regions`, `_make_contiguous`, `_convex_region_pass`)

```text
_reach_masks: desc[i] = bitmask of every gate after i along wires (descendants)
              anc[i]  = bitmask of every gate before i along wires (ancestors)

_convex_regions(g):
    neighbours = adjacent gates on the same wire
    for each unassigned non-H gate i, in index order:
        members = {i}; mask, d, a = bit i, desc[i], anc[i]
        heap = neighbours of i
        while heap: j = pop smallest
            if j assigned or H: continue
            if (d | desc[j]) & (a | anc[j]) & ~(mask | bit j): continue   # would not be convex
            add j; update mask, d, a; push j's unassigned neighbours
        yield sorted(members)

_make_contiguous(g, region): relist as [outside ancestors of the region] + region + [everything else]

_convex_region_pass(n, g, config):
    repeat at most MAX_REGION_MOVES (64) times:
        for region in _convex_regions(g) with ≥ 2 gates:
            h, s, e = _make_contiguous(g, region)
            best = argmin by _key over candidates whose full circuit is strictly cheaper than g
            if best: g = best; restart
        none found: return g
    _BUDGET_HITS["regions"] += 1
```
*Convex* = no outside gate is both a descendant and an ancestor of region
gates; only then can the region be made one contiguous block. It catches
regions that interleaved gates hide from the list pass, e.g. a 6-gate region in
`CX01 T1 CX01 H2 CX12 H2 CX10 T0` (cost 29.2 → 15.6).

### 9.3 Cross-region moves (`_cross_region_pass`, beam move only)

```text
g = _region_order(g)
repeat while a move was accepted, at most MAX_CROSS_MOVES (64) moves:
    for consecutive spans (left, separators, right) where separators are only H gates:
        candidates = []
        if the separators touch neither region's wires:
            candidates += [prefix + separators + best_region(left + right) + suffix]   # merge
        for each term (y, c) of right whose wires the separators do not touch:
            right_rest = gauss rebuild of right without that term
            if canon(gadget(y, c) + right_rest) == canon(right):          # exact check
                candidates += [prefix + best_region(left + gadget) + separators
                               + best_region(right_rest) + suffix]          # float term backward
        accept the best candidate if strictly cheaper; g = _region_order(it)
```

---

## 10. Two-qubit Clifford table and block pass

### 10.1 Exact key (`_pauli_forward`, `_cliff2_key`)

A Pauli string is (x mask, z mask, sign). `_pauli_forward(p, G)` returns
G·p·G† by the Aaronson–Gottesman rules (H swaps x and z bits; S adds x into z;
CX propagates x from control to target and z from target to control, with the
sign rule s ^= xc·zt·(xt ⊕ zc ⊕ 1); X and Z flip the sign on anticommuting
bits; SDG = S three times). `_cliff2_key(gates)` = the images of X0, Z0, X1, Z1.
Equal keys ⇔ equal operators up to phase.

### 10.2 Table (`_cliff2_table`, built once per process on first use, about 1.8 s)

```text
generators = H, S, SDG, X, Z on each qubit + CX(0,1), CX(1,0)          # 12
label = (n2, n1, depth0, depth1)
table = {key(identity): [((0,0,0,0), ())]}; layer = [(identity, label, ())]
while layer:
    for (key, label, gates) in layer, for g in generators:
        new label: CX → both depths = max + 1, n2 + 1; one-qubit → that wire's depth + 1, n1 + 1
        k2 = key after g
        if some stored label for k2 dominates the new one (≤ in every component): skip
        remove stored labels the new one dominates; store it; add to next layer
```
Result: all 11,520 two-qubit Clifford operations, each with every
Pareto-optimal implementation.
`_label_cost(label) = w1·n1 + w2·n2 + wd·max(depth0, depth1)`.

### 10.3 Blocks and the pass (`_pair_blocks`, `_block_pass`)

```text
_pair_blocks(g, a, b, clifford_only):
    scan gates; skip gates touching neither a nor b
    a gate touching a qubit outside {a, b}, or (clifford_only and T/TDG), closes the block
    yield blocks of ≥ 2 gates

_block_pass(n, g, config):
    repeat while something changed:
        for pairs a < b, for block in _pair_blocks(g, a, b, clifford_only=True):
            (label, best) = argmin over table[key(block)] of (_label_cost, label)
            if _label_cost(label) ≥ _label_cost(block's own label): continue
            out = splice(block removed, best at the block's last position)
            if _key(out) < _key(g): g = out; restart
    return g
```
*Why*: CX gates dominate cost under typical weights, and for a two-qubit
Clifford block the true optimum is a lookup; Pareto sets let the choice follow
the weights. *Measured*: removing the pass raises cost 3.0–6.8%.
*Trade-off*: the 1.8 s build per process; no three-qubit table.

---

## 11. Pipeline B: Pauli rotations

```text
_to_rotations(g):                    # U = C_final · R_m ⋯ R_1
    clifford = []; rotations = []
    for gate in g:
        if T or TDG on q:
            p = Z on q; for c in reversed(clifford): p = c† · p · c   # axis C†·Z·C
            rotations.append((p, 1 if T else 7))
        else: clifford.append(gate)
    return rotations, clifford

_merge_rotations(rotations):
    out = []
    for (p, k) in rotations:
        scan out from the end backwards:
            same axis (same x and z masks): k' = k if same sign else −k
                total = (k_out + k') mod 8; replace, or delete if 0; stop
            anticommutes with p: append (p, k); stop
        reached the start: append (p, k)

_rotation_gates(n, p, k):
    if p's sign is negative: k = −k
    basis change on each qubit of p: X → H; Y → SDG then H (undone by H then S)
    parity tree onto the first qubit of p; _PHASE_OUT[k]; tree reversed; undo basis change

_pipeline_b(n, g) = _stage12(n, concat(_rotation_gates(p, k) for merged rotations) + clifford)
```
*Why* (Zhang & Chen 2019): merges T gates about X and Y axes, i.e. T gates
sandwiched by H, which folding cannot see. Example: `H0 T0 H0 CX10 H0 T0 H0`
(cost 13.4) → `H0 S0 H0 CX10` (9.8); nothing else in the optimizer finds it.
*Rejected*: choosing the parity-ladder target per rotation (no end-to-end change).

---

## 12. T-blocks

```text
_t_block_pass(n, g, config):                      # beam move only
    repeat at most MAX_REGION_MOVES (64) times:
        for pairs a < b, for block in _pair_blocks(g, a, b) containing a T or TDG:
            new = _resynth_t_block(block in local qubits 0, 1)
            if new is None: continue
            if not _fast_equivalent(block, new): continue        # strict: raise
            out = splice(block removed, new at the block's last position)
            if _key(out) < _key(g): g = out; restart
        none accepted: return g

_resynth_t_block(local, config):                  # cached on (gates, w1, w2, wd)
    rotations, clifford = _to_rotations(local); rotations = _merge_rotations(rotations)
    changers = _z_basis_changers()   # (q, P) → cheapest Clifford V with V·Z_q·V† = P, from the table
    odd = indices of rotations with odd exponent
    choices = all of {0,1}^|odd| if |odd| ≤ TBLOCK_MAX_CHOICES (6) else [all zeros]
    for choice in choices:
        stream = []
        for (P, k) at index i: q = choice for i if i is odd, else 0
            stream += inverse(V) + _PHASE_OUT[k] on q + V          # V = changers[(q, P)]; = R_P
        stream += clifford
        out = split stream at T/TDG gates; replace every Clifford stretch with its
              cheapest table entry; keep the T gates
        keep the out with the lowest _label_cost
    return best if strictly cheaper than local's label cost else None
```
*Why*: the T-count stays fixed, and every Clifford gate around the T gates
becomes the cheapest the exact table allows; neighbouring V and V† merge inside
one stretch. *Cap*: 2^6 = 64 carrier assignments bounds the work.

---

## 13. Beam search

```text
_beam_settings(width):
    width None: (4, 2, 1000) if QUBLITZ_BEAM == "fast" else (16, BEAM_PATIENCE = 5, 16 × 1000)
    else:       (width, 5, width × EVALS_PER_WIDTH (1000))

_beam_search(n, start, config, width):
    best = start; beam = [start]; seen = {_canon(start)}; evals = stale = 0
    while beam and stale < patience:
        children = []
        for state in beam, for child in _beam_children(state):
            if _canon(child) in seen: continue
            seen.add; children.append(child); evals += 1
            stop expanding if evals ≥ max_evals
        beam = the `width` children with the smallest _key           # may be costlier than parents
        if beam and _cost_key(beam[0]) < _cost_key(best): best = beam[0]; stale = 0
        else: stale += 1
        if evals ≥ max_evals: _BUDGET_HITS["beam"] += 1; break
    return best

_beam_children(state) = [_stage12(n, child) for child in
    [move(state) for each whole-pass move whose _pass_applicable is true] + _local_moves(state)]
```

**Whole-pass moves**: fold-first, fold-last, fold-depth, h_reduce, regions,
convex_regions, cross_regions, blocks, tblocks, pipeline_b.

**`_pass_applicable`** skips a move only when it provably cannot change the
circuit: fold-depth unless `_depth_focus`; folding with fewer than two phase
gates; H-reduction without H; Pipeline B and T-blocks without T/TDG; blocks
and cross-region without CX; anything else with fewer than 2 gates.

**`_local_moves`** (all exact, may raise cost):

| At | Move |
|---|---|
| CX(c, t), X on c just before | remove it, add X_c X_t after the CX |
| CX(c, t), X on c just after | remove it, add X_c X_t before the CX |
| CX(c, t), Z on t just before | remove it, add Z_c Z_t after the CX |
| CX(c, t), Z on t just after | remove it, add Z_c Z_t before the CX |
| CX(c, t) | replace by H_c H_t CX(t, c) H_c H_t |
| H S H, SDG H SDG, H SDG H, S H S on one wire | swap to the equivalent triple |
| gate k and the next gate j on a shared wire | if they commute and j does not wait on a gate between them on its other wires: move j before k |

*Why*: greedy passes stop at local minima; the beam keeps costlier circuits
that may lead to cheaper ones. *Measured*: removing the beam raises cost
6–12%; removing only the local moves 6–10%; width 16 beat width 6 by 2.6%;
width 48 gained only 0.4% more; the evaluation budget has never been reached.
*Rejected*: skipping re-application of a pass already applied (same cost, about
45% slower); scoring children by a follow-up block pass (worse).

---

## 14. Verification

**`_fast_unitary(n, gates)`**: start from I; for each gate, left-multiply
without building the expanded matrix:
```text
CX(c, t): u = u[perm], perm[x] = x with bit t flipped if bit c is set      # a row permutation
G on q:   view u as shape (2^q, 2, 2^(n−q−1), 2^n); apply G on axis 1 (einsum "ab,ibjk->iajk")
```
About 4^n operations per gate instead of 8^n. Matches the organizers'
`circuit_unitary` (tested).

**`_fast_equivalent(reference, n, gates)`** (the organizers' rule):
```text
B = _fast_unitary(n, gates); p = position of the largest |entry| of reference
λ = B[p] / reference[p]
return |λ| ≈ 1 and B ≈ λ · reference      (absolute tolerance 1e-9)
```
The largest entry of a unitary is at least 2^(−n/2), so the division is
well-conditioned.

**Three layers**: region canonical forms (§8.1, exact integers); every
candidate and the beam result (`_fast_equivalent` against the original input);
the final output (`equivalent_up_to_global_phase`). None has ever rejected a
rewrite; in strict mode (`QUBLITZ_STRICT=1`, used by the tests) a rejection
raises instead of being dropped.

---

## 15. Termination, determinism, idempotence

- **Termination.** Every loop accepts a change only if it strictly lowers
  `_cost_key` (cleanup strictly shortens; H-reduction lowers (H count, cost)).
  Costs are sums of whole-number gate counts times fixed non-negative weights,
  so only finitely many cost values lie below any start, and a strictly
  decreasing sequence stops. The caps (§16) only bound worst-case runtime.
- **Determinism.** No randomness or time limits; heaps and sorts on indices;
  every argmin has an explicit tie-break; caches store only results of
  functions that always return the same output for the same input. Outputs are
  identical across processes and hash seeds (tested).
- **Idempotence.** `optimize` returns only a fixed point x of `_select`
  (`_select(x) == x`); a second call computes `_select(x)` first and stops.
  Internal caps cannot break this, because they are part of `_select`'s
  deterministic definition. Only reaching the outer cap (8 selection rounds
  still changing) could, which is recorded in `_BUDGET_HITS` and tested never
  to happen on the test circuits.

---

## 16. All parameters

| Constant | Value | Role | Basis |
|---|---|---|---|
| `MAX_SELECT_ROUNDS` | 8 | outer rounds | measured ≤ 2 |
| `MAX_ROUNDS` | 8 | greedy-loop rounds | measured ≤ 2 improving |
| `MAX_REGION_MOVES` | 64 | accepted convex-region and T-block rewrites per pass | budget, never reached |
| `MAX_CROSS_MOVES` | 64 | accepted cross-region moves per pass | budget, never reached |
| `BFS_MAX_WIRES` | 4 | `_reconstruct_bfs` width limit | 20,160 matrices at 4 wires |
| exact-small limit | 3 wires | `_reconstruct_exact_small` | 21,504 states; 4 wires ≈ 660 million |
| `LONG_WINDOW` | 7 | one-qubit table depth | 376 operators |
| `DEFAULT_BEAM_WIDTH` | 16 | beam width | +2.6% over 6; 48 adds 0.4% |
| `BEAM_PATIENCE` | 5 | steps without a new best | calibrated with width |
| `EVALS_PER_WIDTH` | 1000 | evaluation budget per unit of width | never reached |
| `TBLOCK_MAX_CHOICES` | 6 | carrier-qubit choices tried exhaustively (2^6) | work bound |
| `_depth_focus` | wd > w1 + w2 | enables depth placement | +2.2% wd-heavy, −0.9% balanced if always on |
| tolerance | 1e-9 | matrix comparisons | rounding ≈ 1e-15 |
| caches | `_best_run` 65,536; `_region_candidates` and `_resynth_t_block` 16,384 entries; tables unbounded | speed only | outputs byte-identical with and without |

`FEATURES` switches (all on except two): `fold`, `h_reduce`, `regions`,
`convex_regions`, `blocks`, `pipeline_b`, `beam`, `depth_fold`;
`cross_regions_greedy` and `tblocks_greedy` are off (beam moves only, §5).
`python bench.py --quick --ablation` removes each step everywhere it runs and
reports the effect; the current numbers are in README §8.
