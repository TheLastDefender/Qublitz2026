# Stage 5 Plan: Bigger Improvements (implementation plan, not started)

Target file: `submission_stage4.py`. `submission.py` stays the frozen,
verified Stage 2 baseline until the final one-step replacement (option B).
Every rule from the Stage 4 spec section 0 still applies: exact identities
only, deterministic output (work budgets, never wall-clock), each rewrite
verified, and output never costlier than the input or Stage 2 under the
weights passed in.

---

## 0. Baseline measurements (taken before planning)

| Measurement | Result |
|---|---|
| 2-qubit circuits whose optimum is ≤ 4 gates (1,429 cases) | Stage 4 exactly optimal on 92%. Misses are mostly one-qubit gates, plus 37 cases with extra CX. |
| 3-wire regions vs exact minimum CX for the same term list (400) | Optimal 53%; 1–6 extra CX in 42% |
| One-qubit runs vs whole-run optimum (3,000) | Stage 2 longer in 4.1%; Stage 4 longer in 1.3% |
| Pairwise vs summed phase merging (3,000) | Never worse (0 cases) |
| Rounds used | `_select`: 1–2 rounds; FULL: at most 2 improving rounds; cap of 8 never approached |
| Time per call, 60 gates on 6 qubits | FULL ≈ 6 ms; Stage 1+2 ≈ 0.34 ms; final unitary check ≈ 6.5 ms |

The 2-qubit gap was measured only on *easy* instances. Real gaps on
larger circuits are unknown, which is why step 1 builds the measurement
harness first.

---

## 1. Priority order (most impactful first)

Priority is set by how much cost each change can remove under typical
weights, where CX gates usually dominate (`w2 = 6` in the example config).
Step 1 measures the actual contribution of each item, so the order can be
corrected with data.

| # | Item | Expected impact | Why |
|---|---|---|---|
| 1 | **Benchmark and ablation harness** | enables everything | Without it, "bigger improvements" can't be shown. It gives exact optima for small circuits and a per-change contribution table. |
| 2 | **Faster exact verifier** | runtime, enables 5 | Verification costs as much as a full optimization round. Search needs it to be cheap. |
| 3 | **Phase folding across H gates** | high (phase-heavy circuits) | Merges every phase on the same parity anywhere in the circuit in one shot. Today's term floating is pairwise, local, and only accepted if it pays off at once. |
| 4 | **Exact 2-qubit block resynthesis** | high (CX-heavy) | It targets CX directly, which is the most expensive resource. Clifford blocks are exactly optimal from a table. |
| 5 | **Beam search instead of strict-only acceptance** | high, broad | Lets the optimizer take temporarily worse moves that unlock larger savings. Costs runtime (section 5). |
| 6 | **Exact combined CX + phase synthesis (≤ 3 wires), GraySynth (4–6)** | medium (CX) | 42% of 3-wire regions have 1–6 extra CX. |
| 7 | **Greedy convex H-free regions** | medium | Bigger regions feed 6. Replaces list slicing plus the three orderings. |
| 8 | **Depth-aware optimization** | medium under depth-heavy weights | Today depth only breaks ties between candidates. |
| 9 | **Optimal windows (whole-run lookup + best split)** | low–medium | 1.3% of one-qubit runs (in Stage 4) are still suboptimal. |
| 10 | **Phase segment summing** | clarity only | Measured: same results as pairwise. Simpler correctness argument. |
| 11 | **Strict verification mode; round-cap test** | safety | Bugs fail loudly in tests, and a hit cap is reported, not silent. |

---

## 2. Item details

### 1. Benchmark and ablation harness (`bench.py`, not imported by `optimize`)
- **Exact optima for small circuits:** breadth-first search over 2-qubit
  circuits, storing for each distinct unitary (up to global phase) the
  **Pareto set** of (CX count, one-qubit count, depth). The best entry for
  any weights can then be read off.
- **Corpus:** random circuits (1–6 qubits, up to 60 gates), Toffoli,
  Toffoli×2, compute/uncompute, and H-heavy, phase-heavy and CX-heavy
  generators.
- **Weight profiles:** w2-heavy (1, 6, 0.2), w1-heavy (1, 0, 0),
  wd-heavy (0, 0, 1), balanced (1, 1, 1).
- **Report:** mean cost reduction per profile; the gap to optimal where
  known; and an **ablation table**, meaning the cost change when each item
  below is switched off.
- **Done when:** the harness reproduces the section 0 numbers.

### 2. Faster exact verifier
- **How:** apply gates directly to the 2^n × 2^n matrix. X and CX permute
  rows, diagonal gates scale rows, and H mixes pairs of rows. That replaces
  building a Kronecker product per gate. Then compare with the same
  global-phase rule as the official checker.
- **Test:** agrees with `equivalent_up_to_global_phase` on 5,000 random
  pairs, half equivalent and half not.
- **Rule:** the official function remains the final check in `optimize`;
  the fast one is used inside search.

### 3. Phase folding across H gates
- **Idea (T-par, Amy–Maslov–Mosca 2014):** each wire's value is written as
  a parity over *path variables*, meaning the inputs plus one fresh
  variable for every H gate.
  - **X** flips the parity's constant bit, so the exponent is negated.
  - **CX** XORs the parities.
  - **H on wire q** gives q a fresh variable.
  - **Merging:** phase gates applied to the *same* parity merge into one,
    at the position of the first. Their exponents add mod 8, and a zero
    total removes it.
- **Before implementing:** read the exact validity condition in the source
  and confirm it numerically on hand-built cases, including parities that
  survive on a different wire after an H.
- **Tests:**
  - equivalence on 2,000 random circuits;
  - cases that today's region floating misses;
  - phase-gate count never higher than with the old method.

### 4. Exact 2-qubit block resynthesis
- **Block:** a maximal convex group of gates touching only one qubit pair
  (a, b).
- **Clifford-only blocks** (H, S, SDG, X, Z, CX):
  - the 2-qubit Clifford group has 11,520 elements up to phase, so build a
    BFS table over the gate set (CX in both directions);
  - for each element, keep the Pareto set of (CX count, one-qubit count,
    depth);
  - pick the cheapest entry under the given weights. This is exactly
    optimal for that block.
- **Blocks with T gates:**
  - a BFS table up to length 5 holds about 20,000 distinct unitaries
    (measured during the Stage 2 research);
  - longer blocks: pull the phases out with phase folding first, then
    resynthesize the Clifford parts between them.
- **Tests:**
  - table entries are equivalent to their keys;
  - the table size is exactly 11,520;
  - block rewrites preserve the unitary;
  - never costlier.

### 5. Beam search over moves (replaces strict-only acceptance)
- **State:** a circuit in canonical form, defined below (deduplication).
- **Moves:** every verified local rewrite, *including ones that raise
  cost*, for example:
  - Pauli propagation in both directions;
  - `H S H ↔ SDG H SDG`;
  - CX reversal with H gates;
  - moving a gate past one it commutes with;
  - splitting a phase;
  - plus the composite passes (folding, region resynthesis, block resynthesis).
- **Search:**
  1. keep the **B** best states by cost tuple;
  2. expand each one;
  3. drop repeats;
  4. keep the B best;
  5. stop after a fixed number of steps without a new overall best, or
     when the work budget is spent.
- **Determinism:**
  - moves are generated in a fixed order;
  - ties are broken by the canonical form;
  - no randomness;
  - the budget counts evaluated states, not time.
- **Canonical form:** each wire's gate sequence, which is the same for
  every valid reordering of gates on separate wires. Two circuits that
  differ only in how independent gates are listed count as one state.
- **Tests:**
  - never worse than the greedy result, because greedy is the first beam
    member;
  - deterministic across hash seeds;
  - finds the known improvement on hand-built local-minimum cases (circuits
    where the only path to a saving goes through a costlier intermediate);
  - budget respected.

### 6. Exact CX + phase synthesis (≤ 3 wires) and GraySynth (4–6 wires)
- **≤ 3 wires:** breadth-first search over (wire contents, set of phases
  still needed). There are 168 × 2⁷ = 21,504 states, the same search used
  for the 3-wire measurement. Run it for each term-list variant (built,
  canonical, Z-variants) and keep the best.
- **4–6 wires:** implement GraySynth from Amy, Azimzadeh & Mosca
  (arXiv:1712.01859), following the paper's algorithm. The existing
  candidates stay, so it can only help.
- **Tests:**
  - matches brute force on all 3-wire cases in the measurement set;
  - GraySynth output is equivalent;
  - never costlier than the current candidates.

### 7. Greedy convex H-free regions
- **Growing a region:** start from the earliest unassigned non-H gate. Add
  the next non-H gate along any wire of the region if the region stays
  **convex**, meaning no path in the gate dependency graph leaves the
  region and comes back in. Check that by reachability.
- **Order:** fixed, by wire and then gate index.
- **Tests:**
  - every region is convex and H-free;
  - the counterexample `CX01 H1 CX10` is not merged;
  - never more regions than today's method.

### 8. Depth-aware optimization
- **Synthesis:** prefer, among equal-count options, the one with lower
  depth (BFS tables store depth; GraySynth orderings are scored by depth).
- **Scheduling move:** add a move to beam search that reorders independent
  gates to shorten the critical path. Only gates that commute are swapped.
- **Weights:** depth is optimized only to the extent `wd` rewards it.
- **Tests:** depth-heavy profile improvements, measured in the harness;
  never costlier.

### 9–11. As described before
- **Optimal windows:** whole-run lookup for runs whose optimum is ≤ 7
  gates, using a 376-class table; otherwise a dynamic program for the best
  split into chunks of ≤ 7.
- **Phase segment summing:** split each wire at the gates that block
  phases, sum each segment mod 8, and emit the shortest form.
- **Strict mode:** a module flag the tests set, so a failed verification
  *raises* instead of dropping the candidate.
- **Round-cap test:** fails if any test circuit reaches `MAX_ROUNDS` or
  the beam budget without converging.

---

## 3. Order of work and checkpoints

Each step ends with:
- the full test suite passing (all three suites);
- the harness numbers recorded in `docs/stage5.md`;
- a commit.

| Step | Contents | Checkpoint |
|---|---|---|
| A | Items 1, 2, 11 (harness, fast verifier, safety) | baseline table reproduced |
| B | Item 3 (phase folding) | ablation shows its gain |
| C | Item 4 (2-qubit blocks) | CX gap reduced in the harness |
| D | Item 5 (beam search) + decide the budget | gain vs runtime table |
| E | Items 6, 7 (regions and synthesis) | 3-wire CX gap ≈ 0 |
| F | Items 8, 9, 10 | depth-profile gains; window gap ≈ 0 |

---

## 4. Decisions needed before step D
- **Time budget per circuit:** for example ≤ 1 s at 60 gates. It sets the
  beam width and the work budget.
- Whether the beam budget should scale with circuit size, which is still
  deterministic.

## 5. Beam search runtime model

**Cost ≈ (steps) × B × (moves per state) × (time per evaluation).**

Today, one evaluation is about 0.35 ms (Stage 1+2 cleanup plus cost).
A full verification is about 6.5 ms, but it is only needed for the final
candidates once the fast verifier exists.

| Configuration | Estimated time per call (60 gates) |
|---|---|
| Greedy (today) | ~40 ms |
| Beam B = 4 | ~0.5–1 s |
| Beam B = 8 | ~1–2 s |
| 2-move lookahead (all moves) | ~3–10 s |
| 3-move lookahead (all moves) | not feasible (≈ 27,000 evaluations per decision) |

Lookahead grows exponentially with depth and beam search linearly with
width, which is why the plan uses beam search with only local lookahead.
