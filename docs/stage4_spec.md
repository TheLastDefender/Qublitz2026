# Stage 4: Region Resynthesis + Pauli-Rotation Alternative (Implementation Spec)

> **File names (updated):** this document was written while the full optimizer lived in `submission_stage4.py`. That file is now **`submission.py`** (the file the organizers import), and the former Stage 1-2 `submission.py` is now **`stage2_baseline.py`**, kept only as the reference for tests and benchmarks. Read file names below accordingly.

Implement inside `submission.py` (helpers in the same file; organizers import
only this file). Build in sub-stages **4a → 4f**, one at a time, each with its
own tests passing before the next starts. Stage 1–2 code stays and is reused.

---

## 0. Hard requirements (apply to every sub-stage)

### 0.1 Unitary preservation
- Every transformation is an exact identity (up to global phase). No
  approximation and no "probably equal".
- **Final gate:** before `optimize` returns, compute
  `equivalent_up_to_global_phase(input, output)`. If it is False, return the
  best earlier result that *was* verified. The chain is: Stage 4 result →
  Stage 2 result → `copy.deepcopy(input)`. Each fallback is itself checked.
- **Candidate gate:** each region rewrite and each whole-circuit candidate
  (Pipeline A, Pipeline B) is checked by unitary before it is compared on cost.
  A candidate that fails the check is discarded, never repaired.
- Bugs must surface in tests, not be hidden by the fallback. Tests call the
  internal pipelines directly and assert equivalence, so a pipeline that only
  "works" because of the fallback fails its tests.

### 0.2 Determinism
Same input and same config give byte-identical output on any machine and any run.
- **No randomness:** no `random` and no hash-seeded ordering.
- **No wall-clock limits:** a time budget makes the output machine-dependent.
  Use a **work budget** instead: a fixed cap on loop iterations, candidate
  counts or search nodes, given as module constants.
- **Stable ordering everywhere:**
  - never iterate over a `set` or `frozenset` to produce output. Iterate over `sorted(...)`;
  - dict iteration only over insertion-ordered dicts built in sorted or circuit order;
  - every tie-break is spelled out (see 0.4).
- **Exact arithmetic in decisions:**
  - GF(2) matrices as Python `int` bitmasks;
  - phase exponents as `int` mod 8;
  - Pauli tableaux as bits plus an integer sign;
  - floats only in the final unitary verification, which never steers a choice.
  The Stage 2 window table and `_commutes` use floats and are left unchanged,
  since they are built deterministically and are covered by tests.

### 0.3 Validity
- Output passes `validate_circuit`, has the same `n_qubits`, uses `ALLOWED_GATES`
  only, and has exactly the keys `{"n_qubits", "gates"}` / `{"gate", "qubits"}`.
- Do not mutate `circuit` or `config`.
- `circuit_cost(output, cfg) <= circuit_cost(input, cfg)` and
  `<= circuit_cost(stage2_output, cfg)` for the given cfg. This is guaranteed by
  the final selection (Section 4f), not just hoped for.

### 0.4 Tie-breaking (total order on candidates)
- **Final selection (4f):** compare `(cost, n_2q, depth, n_1q, candidate index)`.
  The candidate list is in a fixed order with the round's starting circuit first,
  so on equal cost the circuit is **left unchanged**: `optimize` changes a
  circuit only when it finds something strictly cheaper. (Idempotence comes
  from the repeat loop in 4f, not from this tie-break. A mutation test showed
  that a gate-tuple tie-break is also idempotent, but it swaps equal-cost
  circuits. `test_4f_unchanged_unless_strictly_cheaper` pins this property.)
- **Inside a pass** (choosing among region rewrites): compare
  `(cost, n_2q, depth, n_1q, gate_list_as_tuple)`. That is a total order.

---

## 4a. H-reduction pre-pass

**Goal:** fewer H gates means fewer region boundaries in 4b.

Rules. Each one is verified in tests against the official equivalence
function, and each strictly lowers the H count:

| Rule | H change |
|---|---|
| `H0 H1 CX01 H0 H1 → CX10` (any wires; the four H's must be wire-adjacent to the CX) | −4 |
| `H S H → SDG H SDG` | −1 |
| `H SDG H → S H S` | −1 |
| `H X H → Z`, `H Z H → X` (already in Stage 2 windows) | −2 |

Pauli propagation, used only when it lets two H's become adjacent and cancel:
`X0 CX01 X0 → CX01 X1` and `Z1 CX01 Z1 → Z0 CX01`.

- **Acceptance:** a rewrite is applied only if (H count, cost) strictly
  decreases lexicographically. That guarantees termination.
- **Scan order:** circuit order, then lowest qubit index. Restart the scan
  after each accepted rewrite.

**Tests (4a):**
- each rule's LHS and RHS are equivalent;
- the H count never increases;
- the output is equivalent on 500 random circuits;
- running the pass twice gives the same output.

---

## 4b. Region extraction and exact reconstruction (no optimization yet)

**Goal:** prove the representation is lossless before optimizing with it.

### Region
A **region** is a maximal run of consecutive gates (in list order) containing
no H. Gates allowed in a region: `CX, X, Z, S, SDG, T, TDG`. H gates sit
between regions as fixed separators.

### Representation
On basis state `|x⟩` (x ∈ {0,1}^n), an H-free region acts as

```
|x⟩  →  ω^{f(x)} |A·x ⊕ b⟩,     ω = e^{iπ/4}
f(x) = Σ_y  c_y · (y·x mod 2)    (mod 8, global phase dropped)
```

- `A` is an n×n invertible GF(2) matrix: `rows[i]` is an int bitmask saying
  which input bits XOR into wire i.
- `b` is an n-bit int: which wires carry an extra flip, from X gates.
- `terms` maps parity y (an int bitmask, nonzero) to `c_y` in 1..7. Zero
  coefficients are dropped.

### Building it (walk the region in order, tracking each wire's current affine function)
- `CX c,t`: `rows[t] ^= rows[c]`, and `b_t ^= b_c`.
- `X q`: `b_q ^= 1`.
- Phase `T^k` on wire q with current value `rows[q]·x ⊕ b_q`:
  - if `b_q = 0`: `terms[rows[q]] += k`;
  - if `b_q = 1`: `terms[rows[q]] -= k` (because `k·(1 ⊕ p) = k − k·p`; the
    constant `k` is a global phase and is dropped). This is the verified
    identity `X T = TDG X`.
- All coefficients are reduced mod 8.

### Canonical form
`terms` is **not** unique, because parity functions are not independent mod 8.
Verified counterexample: `4·x0 + 4·x1 + 4·(x0⊕x1)` is the identity
(`Z0 · Z1 · Z(x0⊕x1) = I`). So do not compare term lists.

Use the phase **table** instead:
`canon = (A as tuple of rows, b, tuple((f(x) − f(0)) mod 8 for x in 0 .. 2^m − 1))`,
where m is the number of region wires (at most 6, so at most 64 integers).

`U|x⟩ = ω^{f(x)}|Ax⊕b⟩`. Two such maps are equal up to global phase iff they
have the same permutation (A, b) and the same f up to an additive constant.
Subtracting f(0) removes exactly that constant. So two regions are equal up to
global phase **if and only if** their `canon` values are equal. This is exact
integer arithmetic.

### Exact reconstruction (baseline synthesizer, no cleverness)
Emit phase gates first, in sorted parity order:
1. For each term, bring parity y onto its lowest set wire using CX's from the
   other set wires (in increasing order).
2. Apply the phase as the shortest list from `_PHASE_OUT[c_y]`.
3. Undo those CX's in reverse order.

Then synthesize the linear map `A` by GF(2) Gaussian elimination (column by
column, lowest pivot first), and finally apply X on each set bit of b.

**Tests (4b):**
- for 1000 random H-free circuits, extract → reconstruct → equivalent;
- `canon` is equal for two circuits iff they are equivalent (check against the
  unitary, both directions, on random pairs). Include the identity
  `Z0 Z1 CX01 Z1 CX01`, whose term list is non-empty but whose `canon` must equal
  the empty circuit's;
- unit tests: X before T flips the sign; CX updates the rows correctly.

---

## 4c. In-region phase merging and resynthesis

Merging is automatic: equal parities add in `terms`. That is the whole point of
the representation. `T` on `x0⊕x1` at two different places in a region merges
even with CX's between them.

**Z-term reduction (exact):**
- split each coefficient as `c_y = (c_y mod 4) + 4·e_y`, with `e_y ∈ {0,1}`;
- since `4a + 4b ≡ 4·(a ⊕ b) mod 8` for bits a, b, all coefficient-4 parts
  collapse into one Z-term on the parity `⊕{y : e_y = 1}` (none if that XOR is 0);
- verified: `Z(x0) · Z(x0⊕x1) = Z(x1)`.

There is no such rule for S-type (coefficient-2) terms: `S(x0) S(x1) S(x0⊕x1)`
is **not** the identity (verified). Do not invent one.

### Resynthesis candidates for one region (generate all, keep the cheapest)
1. **The original gates** (always a candidate, so the rewrite is never worse).
2. **Baseline reconstruction** (4b).
3. **Exact CX-optimal linear part for n_region ≤ 4:**
   - BFS over GF(2) states (at most 20,160 for 4 wires), with generators
     CX(i,j) in fixed order `(i, j)` ascending, built once per wire count
     and cached;
   - phase terms are then placed **greedily**: walk the BFS-optimal CX
     sequence, and at each step, if some wire's current row equals a pending
     parity y, emit its phase there (lowest wire first);
   - any parity never seen during the walk falls back to 4b-style CX
     conjugation.
4. *(Optional, later)* a GraySynth-style parity network (Amy, Azimzadeh & Mosca,
   2018) for regions with more than 4 wires.

Region wires are those touched by the region. Map them to local indices
0..m−1 in increasing global qubit order.

**Selection:** each candidate is checked by unitary on the region's wires, then
compared by the 0.4 tuple using the **actual config**. Region-level cost is
computed by splicing the candidate into the full circuit and calling
`circuit_cost`, because depth interacts with neighbouring gates.

**Tests (4c):**
- the T-count of the result is at most that of the input, on random H-free circuits;
- a hand-built case: `T0 CX01 T1 CX01 TDG0 …` where two T's on the same
  parity, separated by CX's, merge (Stage 2 must miss this case);
- the BFS table is a correct linear map for every entry;
- the selected candidate is never costlier than the original under 6 weight configs.

---

## 4d. Recombination and cross-boundary simplification

After 4c, rebuild the gate list (regions and H separators in order), then:

1. **Boundary cleanup:** run Stages 1–2 on the whole circuit.
2. **Floating terms across H gates:**
   - a phase term can be placed at the *end* of its region as a parity over
     output wires: `y_out = (A⁻¹)ᵀ·y`, with the sign flipped if `y_out·b = 1`;
   - if no H between this region and the next touches any wire in `y_out`,
     move the term into the next region's `terms` (that region's input wires
     are the previous region's output wires);
   - process boundaries left to right, terms in sorted order;
   - accept the move only if the total cost after re-resynthesizing both
     regions strictly drops.
3. **Inverse-region detection:**
   - regions R1 and R2 cancel if feeding R1's gates and then R2's gates through
     the same 4b builder gives `canon` equal to the identity's (A = I, b = 0,
     table all zeros);
   - this composition approach needs no separate inverse formula, so there is
     nothing extra to get wrong;
   - it applies only when everything between them commutes with both. Start with
     the simple version: they are separated only by H gates on wires that
     neither region touches;
   - if they cancel, remove both.

**Tests (4d):**
- **Toffoli ×2 (standard 7-T decomposition, twice):** the output has strictly
  fewer gates than Stage 2's 27. Target: 0. Record the actual result.
- a constructed compute/uncompute circuit `R · H_k · R⁻¹` (k untouched by R) → `H_k`;
- term floating preserves the unitary on 500 random circuits with H gates.

---

## 4e. Pipeline B: Pauli-rotation form (Zhang & Chen, 2019)

### Representation
Every circuit equals `C_final · R_{P_m}(θ_m) ⋯ R_{P_1}(θ_1)` (rotations act first,
in index order), where:
- `C_final` is a Clifford, stored as a **stabilizer tableau** (Aaronson &
  Gottesman, 2004): for each qubit, the images of `X_q` and `Z_q` as Pauli
  strings (x-bits, z-bits, sign ∈ {+1, −1}), all ints.
- each `P_j` is an n-qubit Pauli string with sign, and `θ_j = k_j·π/4` with
  `k_j` odd (only T and TDG are non-Clifford).

### Building (walk the gates in order; C starts as the identity)
- **Clifford gate** (H, S, SDG, X, Z, CX): update the tableau with the standard
  rules (S = Z-quarter-turn; SDG = S³; X and Z only flip signs).
- **T/TDG on q:** `T_q = R_{Z_q}(π/4)` up to phase, and
  `T_q · C = C · R_{C†Z_qC}(π/4)`. So append a rotation with
  `P = C† Z_q C` (conjugate `Z_q` backwards through the current tableau), with
  `k = +1` for T and `−1` for TDG.

### Merging
- Scan rotations in index order. Rotation j merges into an earlier rotation i
  with the same Pauli (`P_j = ±P_i`, where `−P` negates k) if every rotation
  strictly between them **commutes** with `P_j`. Two Pauli strings commute iff
  the number of positions where they anticommute is even.
- After merging, apply the merged `k` mod 8:
  - **0:** delete.
  - **odd:** keep.
  - **2, 4 or 6:** this is a Clifford. Absorb it into `C_final`, and conjugate
    every *later* rotation's Pauli by it, so the order stays correct.

### Synthesis back to gates
1. **Each rotation in order:**
   - basis change per qubit: X → H, Y → SDG·H (undone with H·S), Z → nothing;
   - CX ladder onto the lowest qubit in the support (other support qubits in
     increasing order);
   - `T^k` (from `_PHASE_OUT`);
   - undo the ladder and the basis change. Apply sign by negating k.
2. **`C_final`:** deterministic tableau synthesis (Aaronson–Gottesman
   elimination with fixed pivot order). Then run Stages 1–2 on the whole
   result to clean up.

**Expected behaviour:** B often lowers the T count and almost always adds CX
and H. Whether it wins depends on the weights, which is exactly why it is a
candidate and not the default.

**Tests (4e):**
- tableau update rules match matrix conjugation for every gate on 1–3 qubits;
- `C† Z_q C` from the tableau equals the matrix computation on random Cliffords;
- build → synthesize (no merging) is equivalent on 500 random circuits;
- merging preserves the unitary;
- the commutation test on Pauli strings matches matrices;
- the k = 2/4/6 absorption case preserves the unitary;
- T-count after merging is at most the input T-count.

---

## 4f. Selection loop

```
optimize(circuit, config):
    validate inputs
    current = input
    repeat (at most MAX_ROUNDS):
        s2 = Stages 1–2 (current)
        A  = FULL(s2)                  # FULL = loop {Stages 1–2, 4a, 4c, 4d}
        B  = FULL(4e(s2))              #   until the cost tuple stops dropping
        AB = FULL(4e(A))
        best = cheapest of [current, s2, A, B, AB] that passes verification
               against the ORIGINAL input (ties keep the earliest, i.e. current)
        if best == current: stop
        current = best
    return current
```

- **Every candidate is fully optimized before comparison.** All of them go
  through the same convergent `FULL` loop. An earlier draft gave Pipeline B
  only a Stages 1–2 cleanup, which could make it lose comparisons it would win
  once properly cleaned up.
- **Repeating the selection** until nothing is strictly cheaper makes
  `optimize(optimize(c)) == optimize(c)` exact, unless the budget runs out.
- `MAX_ROUNDS` and all search caps are module constants (a work budget, not
  time).
- Selection runs per config. Different weights may legitimately pick different
  candidates, and a test must show this happens for at least one circuit.

**Tests (4f):**
- **Determinism:** run `optimize` twice in the same process and in a fresh
  subprocess (also with a different `PYTHONHASHSEED`). Outputs are identical.
- `cost(optimize(c)) <= cost(Stage 2(c))` under 6 configs, on 500 random circuits.
- `optimize(optimize(c)) == optimize(c)` exactly. This is guaranteed by the
  repeat-the-selection-until-nothing-changes loop: the second call starts
  from a fixed point. (It holds unless the `MAX_ROUNDS` budget runs out.)
- On a cost tie, the input is returned unchanged (keep-current tie-break).
- All Stage 1 and Stage 2 tests pass against both files. Those tests check
  only the goal: called with weights W, the output is equivalent and no
  costlier under W.
- Inputs are unchanged, and the output format is valid.

---

## Done when
- Every sub-stage's tests pass, including the determinism test across hash seeds.
- A results table in `docs/stage4.md`:
  - **circuits:** random, Toffoli, Toffoli×2, compute/uncompute, plus any
    structured benchmarks we add;
  - **configs:** at least `w2`-heavy, `w1`-heavy and `wd`-heavy;
  - **columns:** cost for input, Stage 2, Pipeline A, Pipeline B and the final pick.
- You can explain, without notes:
  - why an H gate ends a phase-polynomial region;
  - why X before T negates the exponent;
  - why a floated term must be translated by `(A⁻¹)ᵀ`;
  - why Pipeline B can lower T count yet raise total cost.

## Out of scope for Stage 4
- Approximate synthesis or any non-exact rewrite.
- Ancilla qubits (not allowed by the challenge).
- Changes to `validate_*`, `circuit_metrics` or `equivalent_up_to_global_phase`.

## Implementation notes: where `submission_stage4.py` differs from the text above

Each change keeps the result exact and deterministic. Most make the code simpler.

1. **Canonical terms from the phase table (4b/4c).** Resynthesizing from the
   built term list missed identities. In Toffoli×2 the merged region is the
   identity, but its term list was non-empty. `_canonical_terms` derives a unique
   term list from the table:
   - a Möbius transform mod 8 gives the AND-monomial form;
   - each monomial is expanded exactly, using
     `2^{d-1}·x_{i1}…x_{id} = Σ_{T⊆S, T≠∅} (−1)^{|T|+1} p_T`.

   Both term lists are tried, and every result is checked against the table.
2. **Region-friendly ordering (4b).** Before slicing into regions, three
   wire-order-preserving orderings are compared: the given order, H gates
   postponed, and H gates emitted eagerly. The one with the fewest H-free
   regions is used, and the given order wins ties. "Always postpone H" alone
   was tried first, and a test showed it can move a leading H into the middle
   and *add* a region. Every wire's order is unchanged, so the unitary and all
   metrics are identical.
3. **Backward-only floating (4d).** A region's terms act on its *input* wires,
   so a term of R_{i+1} is already a gadget at R_{i+1}'s start. It moves into
   R_i with no translation. Forward floating (the `(A⁻¹)ᵀ` translation) isn't
   needed, because any two terms can still meet by moving left.
4. **Inverse detection is region merging (4d).** If no separating H touches
   either region, both regions are resynthesized as one. A mutually inverse pair
   then resynthesizes to nothing.
5. **No tableau synthesis in Pipeline B (4e).** `C_final` is the input's own
   Clifford gates in order, with the T gates removed. That's exactly right,
   because `U = C_final · R_m ⋯ R_1`. The `FULL` loop then optimizes it.
   Merged rotations with even k are synthesized directly (S/Z/SDG in place of
   T^k) rather than absorbed. Both are exact, and both avoid a tableau
   synthesizer whose correctness would need its own proof.

## References (verified)
- Y. Nam, N. J. Ross, Y. Su, A. M. Childs, D. Maslov, *Automated optimization of large quantum circuits with continuous parameters*, npj Quantum Information **4**, 23 (2018). arXiv:1710.07345. Pipeline structure: H-reduction, rotation merging, cancellation, iterate.
- M. Amy, D. Maslov, M. Mosca, *Polynomial-time T-depth optimization of Clifford+T circuits via matroid partitioning*, IEEE TCAD **33**(10), 1476–1489 (2014). arXiv:1303.2042. Phase polynomials / T-par.
- M. Amy, P. Azimzadeh, M. Mosca, *On the CNOT-complexity of CNOT-phase circuits*, arXiv:1712.01859 (2017, rev. 2018). GraySynth.
- F. Zhang, J. Chen, *Optimizing T gates in Clifford+T circuit as π/4 rotations around Paulis*, arXiv:1903.12456 (2019). Worst case O(n·k²), with k the number of T gates.
- S. Aaronson, D. Gottesman, *Improved simulation of stabilizer circuits*, Phys. Rev. A **70**, 052328 (2004). arXiv:quant-ph/0406196. Tableau conjugation rules.
