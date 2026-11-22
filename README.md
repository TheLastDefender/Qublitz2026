# Quantum Circuit Golf: Exact, Cost-Aware Circuit Optimizer (Qublitz 2026)

`submission.py` exports `optimize(circuit, config)`, the interface the
organizers import. Given a circuit over {H, X, Z, S, SDG, T, TDG, CX} on 1-6
qubits and weights `{"w1", "w2", "wd"}`, it returns a circuit that implements
**the same unitary up to global phase** and has the lowest weighted cost it can
find:

    C = w1 * N_1q + w2 * N_2q + wd * depth

Every rewrite is an **exact** identity. There is no approximation and no
randomness. The output is **deterministic**, **idempotent**
(`optimize(optimize(c)) == optimize(c)`), and **never costlier** than the
input, under the weights passed in.

---

## Contents
1. [Quick start](#1-quick-start)
2. [Requirements and how they are met](#2-requirements-and-how-they-are-met)
3. [Sources of truth: how correctness is established](#3-sources-of-truth-how-correctness-is-established)
4. [The algorithm, from input to output](#4-the-algorithm-from-input-to-output)
5. [Every identity used, with justification](#5-every-identity-used-with-justification)
6. [Existing algorithms used, with citations](#6-existing-algorithms-used-with-citations)
7. [Trade-offs and design decisions](#7-trade-offs-and-design-decisions)
8. [Results](#8-results)
9. [Limitations](#9-limitations)
10. [Testing and reproducibility](#10-testing-and-reproducibility)
11. [Repository map](#11-repository-map)
12. [Attribution and references](#12-attribution-and-references)

---

## 1. Quick start

Requirements: Python 3.10 or newer (the code uses `int.bit_count`; it was
developed and tested on 3.13) and NumPy. The tests also need pytest.

```bash
python -m pip install numpy pytest
```

```bash
python submission.py
```

```python
from submission import optimize, equivalent_up_to_global_phase, circuit_cost

circuit = {"n_qubits": 2, "gates": [
    {"gate": "CX", "qubits": [0, 1]}, {"gate": "Z", "qubits": [0]},
    {"gate": "CX", "qubits": [0, 1]}]}
config = {"w1": 1.0, "w2": 6.0, "wd": 0.2}
out = optimize(circuit, config)          # -> {"n_qubits": 2, "gates": [Z on 0]}
assert equivalent_up_to_global_phase(circuit, out)
```

An optional third argument sets the beam-search width:
`optimize(circuit, config, beam_width=16)`. The default is 16, `0` disables
the search, and larger values are slower and usually cheaper.

`pyproject.toml` declares Python >= 3.10 and NumPy >= 2.0 (tested with Python
3.13 and NumPy 2.5.3); `pip install -e ".[test]"` also installs pytest.

---

## 2. Requirements and how they are met

Source: the challenge document (*CircuitChallenge.pdf*) and the technical
guidelines (*SubmissionGuidelineCircuits.pdf*).

| Requirement | How it is met |
|---|---|
| `submission.py` defines `optimize(circuit, config)` returning a new circuit dict | Yes. The starter's validators, unitary checker and cost functions are kept unchanged at the end of the file. |
| Do not modify the input `circuit` or `config` | The input is converted to tuples once and never written. Tested with deep-copy comparisons. |
| Same `n_qubits`, exact format, allowed gates only | Every rewrite emits only allowed gates. The output passes `validate_circuit` (tested). |
| `U_out = e^{iφ} U_in` | Exact identities only, checked three ways (§3). |
| Use the supplied weights; different configurations are evaluated | Every choice compares the weighted cost under the given `config`. Different weights produce different circuits (tested). |
| Generated automatically for new circuits | No per-circuit tuning. Fixed, deterministic work budgets. |
| Level 1: basic cancellation (HH = XX = TT† = I, phase runs) | Stage 1, §4.3 |
| Level 2: hidden redundancies (commutation, CX Z₀ CX = Z₀) | Stage 2, §4.4 |
| Level 3: a general compiler (repeated rewriting, larger patterns, combined methods, heuristics) | Regions, phase folding, exact 2-qubit blocks, Pauli rotations, beam search: §4.5–§4.10 |
| Level 4: changing resource weights | Cost-aware selection everywhere, Pareto tables, depth moves that follow `wd`: §4.11 |
| Reliability and runtime are considered | Deterministic, never worse than the input; about 0.3–0.5 s median per circuit (§8) |
| Attribution of external code, algorithms, libraries and AI | §6 and §12 |

---

## 3. Sources of truth: how correctness is established

There are three independent layers, from cheapest to most authoritative.

1. **Exact algebra while rewriting** (integer arithmetic only):
   - **H-free regions** are compared by an exact canonical form: the linear map,
     the flips, and the full phase table (§4.6). Two regions are equal up to
     global phase **if and only if** these match.
   - **Two-qubit Clifford operations** are compared by their exact stabilizer
     tableau: where X₀, Z₀, X₁ and Z₁ are sent, with signs (§4.8).
   - **One-qubit operators** are compared by their exact Bloch-sphere (SO(3))
     matrix over Z[1/√2], reduced to lowest terms (§4.12).
   - These checks are exact, so a rewrite is either proven equal or rejected.
2. **Every candidate circuit** in each selection round is checked against the
   input's unitary with a fast exact simulator (`_fast_unitary`). It applies
   each gate to the matrix's rows instead of building Kronecker products, and
   uses the same global-phase test as the starter.
3. **The final output** is checked with the organizers' own
   `equivalent_up_to_global_phase`.

Every transformation is an identity, so layers 2 and 3 should never reject
anything. They are defence in depth against *implementation bugs*. In 1,455
measured checks they rejected nothing. Under the tests, `QUBLITZ_STRICT=1`
makes any rejection **raise an error**, so a bug fails loudly rather than
being silently dropped. In normal use, a rejected candidate is discarded, and
if the final check ever failed, `optimize` would return the input unchanged
(valid, and never costlier).

---

## 4. The algorithm, from input to output

This section is the overview. **`ALGORITHMS.md`** is the precise reference: pseudocode for every algorithm, every parameter and heuristic, and the reason and measurement behind each choice.

### 4.1 Top level: `optimize`
1. **Validate** the circuit and config (starter code), then validate
   `beam_width`.
2. **Convert** the circuit to a list of `(name, qubits)` tuples, and compute
   the input unitary once, as the reference for layer 2.
3. **Repeat selection rounds** (`_select`) from the current circuit until a
   round returns its own input. That is what makes `optimize` idempotent: a
   second call starts from a fixed point. The cap is 8 rounds; the measured
   maximum is 2.
4. **Final check** with the official equivalence function, then return the
   dict.

### 4.2 One selection round: `_select`
1. **Build five candidates.** The four pipelines all use the same greedy loop
   (`_full_optimize`, §4.13):

   | Candidate | What it is |
   |---|---|
   | the round's input | unchanged, so the result is never costlier than the input |
   | Pipeline A | Stages 1–2 cleanup, then the greedy loop |
   | Pipeline B | the Pauli-rotation rewrite (§4.9) of the cleaned input, then the greedy loop |
   | B after A | Pipeline B applied to A's result, then the greedy loop |
   | A, given listing | Pipeline A with the list region pass on the circuit as listed (no H-delayed/H-eager reordering) |

2. **Verify** every candidate against the input unitary.
3. **Keep the cheapest** under the weights, by
   `(cost, n_2q, depth, n_1q, candidate index)`. Ties go to the round's input,
   so a circuit only changes when something is *strictly* cheaper.
4. **Run beam search** (§4.10) from that circuit. Its result replaces the
   winner only if it is strictly cheaper *and* passes verification.

### 4.3 Stage 1: per-wire stack cancellation (`_stage1_cancel`)
- **Setup:** one stack per qubit; a CX sits on both of its wires' stacks as
  the same object.
- **Phase gates** are powers of T, with k = Z:4, S:2, SDG:6, T:1, TDG:7.
  Consecutive phases on a wire add their exponents mod 8.
- **H, X and CX:** an identical gate on top (for CX, on *both* stacks) cancels.
- **Output:** the survivors in their original order, each phase sum written in
  its shortest form (k=3 → `S T`, k=5 → `Z T`).
- **Summing the whole run** is what catches `T Z T = T⁶ = SDG`, where a
  pairwise check finds nothing.

### 4.4 Stage 2: commutation, phase segments, one-qubit runs (`_stage2_reduce`)
These passes repeat while any of them shortens the circuit:
- **Commutation cancellation.** Each gate looks ahead for an identical
  self-inverse partner, skipping gates it commutes with (checked against the
  exact matrices). If found, both vanish: `g M g = M g g = M`.
- **Phase segments.** On each wire, every phase gate between two blockers is
  summed mod 8 in one step. The blockers are H or X on that wire, or a CX
  targeting it; phases pass through a CX *control*. The merged phase is placed
  at the lowest-depth position in the segment.
- **One-qubit runs.** Each run of consecutive one-qubit gates on a wire is
  replaced by its shortest equivalent:
  - an exact lookup when its operator needs at most 7 gates (376 classes,
    exact keys);
  - otherwise the best split into chunks of at most 7 gates, by dynamic
    programming, also tried on the run's Matsumoto–Amano normal form.

`_stage12` alternates Stage 1 and Stage 2 until neither changes anything.

### 4.5 H-reduction (`_h_reduce`)
Each H is a boundary for region synthesis (§4.6), so fewer H gates means
larger regions. Rules (§5, identities 9–12):
- `H H CX H H → CX` reversed;
- `H S H → SDG H SDG` and `H SDG H → S H S`;
- `H X H → Z` and `H Z H → X`;
- Pauli propagation through a CX.

A rewrite is accepted only if (H count, cost) strictly decreases, which
guarantees termination.

### 4.6 H-free regions and exact resynthesis (`_region_pass`, `_convex_region_pass`)
**The representation.** A run of gates from {CX, X, Z, S, SDG, T, TDG} acts on
a basis state as

    |x⟩ → ω^{f(x)} |A·x ⊕ b⟩,   ω = e^{iπ/4},   f(x) = Σ_y c_y·(y·x mod 2)  (mod 8)

- A is an invertible matrix over GF(2), b a flip vector, and f a phase
  polynomial.
- It is built by tracking each wire's value as an XOR of input bits
  (`_region_build`).

**Finding regions,** two ways:
- **List slicing:** cut at H gates, after choosing whichever of three
  wire-order-preserving orderings gives the fewest regions. The fifth
  selection candidate (§4.2) runs the greedy loop once more with this
  reordering switched off, because neither choice is best for all weights.
- **Convex regions:** grown greedily over the gate dependency graph. A region
  must be *convex*: no dependency path may leave it and come back in. That is
  checked with ancestor and descendant bitmasks. The region is then made
  contiguous by a valid reordering.

**Rebuilding a region.** Candidates come from several term lists:
- the built list;
- a **canonical** list derived from the phase table (identity 18). An
  identity region therefore gets no terms at all;
- **Z-variants** of each (identity 17).

Each term list goes through several synthesizers:

| Synthesizer | When | Notes |
|---|---|---|
| Gauss–Jordan | always | one phase gadget per term |
| BFS-optimal linear map, phases placed greedily | ≤ 4 wires | |
| Exact minimum-CX search | ≤ 3 wires | searches (wire contents, parities not yet seen) |
| GraySynth-style parity network | always | |

**Selection.** Every candidate is cleaned up, checked against the region's
exact canonical form, and the cheapest is kept *in the context of the whole
circuit*. The original region is always a candidate, so a region rewrite is
never worse.

### 4.7 Phase folding across H gates (`_phase_fold`)
- **Path variables.** Each wire's value is written as an XOR of *path
  variables*: the inputs plus one fresh variable per H gate.
- **Merging.** All phase gates acting on the same expression are merged into
  one, anywhere in the circuit, even across H gates (identity 20).
- **Host position.** The merged gate is placed at the first occurrence, the
  last, or (when depth dominates the weights) the lowest-depth occurrence.

### 4.8 Exact two-qubit Clifford blocks (`_block_pass`)
- **The table.** All **11,520** two-qubit Clifford operations (up to phase)
  are enumerated on first use (about 1.8 s, once per process; in practice on
  the first multi-qubit circuit, never for one-qubit ones). Each is keyed
  exactly by its tableau, and every Pareto-optimal implementation in
  (CX count, one-qubit count, per-wire depth) is stored.
- **The pass.** Any maximal block of Clifford gates on a qubit pair is
  replaced by its cheapest entry *under the given weights*. That rewrite is
  provably optimal for that block.
- **T-containing pair blocks** (`_t_block_pass`, used as a beam move):
  1. rewrite the block as T rotations about Pauli axes plus a Clifford
     (§4.9), and merge the rotations;
  2. rebuild each rotation as `V† · T · V`, with V a Clifford mapping Z to the
     rotation's axis;
  3. replace every Clifford stretch with its table optimum, trying both
     qubits for each T.

### 4.9 Pipeline B: Pauli-rotation form (`_pipeline_b`)
1. **Pull the T gates out.** Walk the gates, keeping the Cliffords in order.
   Each T or TDG becomes a π/4 rotation about the Pauli axis `C†·Z·C`, where C
   is the Clifford prefix, computed with tableau update rules. The circuit
   equals `C_final · R_m ⋯ R_1`.
2. **Merge rotations** about the same axis whenever every rotation between
   them commutes with it. Unlike region synthesis, this is not blocked by H
   gates.
3. **Rebuild each rotation** as a basis change, a CX parity tree, a phase,
   then the undo, and append `C_final`.

### 4.10 Beam search (`_beam_search`)
Greedy passes accept only immediate improvements and stop in local minima.
The beam fixes that:
- **State:** up to `width` circuits (default 16), deduplicated by their
  per-wire gate sequences.
- **Expansion:** each circuit is expanded by
  - every **whole-pass move**: folding (first, last, depth), H-reduction, both
    region passes, cross-region moves, Clifford blocks, T-blocks, Pipeline B;
  - every **local move**, including ones that temporarily raise cost:
    - Pauli push through a CX;
    - CX reversal with H gates;
    - `H S H ↔ SDG H SDG`;
    - swapping commuting gates that share a wire, which changes the structure,
      and with it the depth.
- **Pruning:** moves that provably cannot change the circuit are skipped, for
  example folding with fewer than two phase gates.
- **Selection:** the `width` cheapest children are kept, even if they are
  costlier than their parent.
- **Stopping:** after 5 steps without a new best, or when the deterministic
  evaluation budget (1,000 × width) runs out.

### 4.11 Adapting to the weights
- **Every choice is cost-driven:** candidate selection, region rebuilds, table
  entries and beam ranking all compare the cost under the given `config`. The
  Pareto tables mean, for example, that a block can be rebuilt with fewer CX
  gates under CX-heavy weights, or with fewer one-qubit gates under w1-heavy
  weights.
- **Depth-only moves** (lowest-depth fold placement) are used only when depth
  outweighs the gate weights, `wd > w1 + w2`. Elsewhere they were measured to
  crowd out gate savings (§7).
- **CX parity trees** replace ladders everywhere: the same CX count with
  logarithmic depth.

### 4.12 Matsumoto–Amano normal form (`_ma_normal_form`)
- **Exact form.** A one-qubit operator is stored as its Bloch rotation
  `(1/√2)^k · N`, where N has entries `a + b√2`. While k > 0, the syllable
  (T, HT or SHT) whose removal lowers k is stripped off; the remainder is one
  of 24 one-qubit Cliffords.
- **Properties:** the result is unique and has the minimum T count (§6).
- **Uses:**
  1. an **exact key** for one-qubit equivalence, with no floating-point
     rounding;
  2. a second split candidate for runs that need more than 7 gates.

### 4.13 The greedy loop: `_full_optimize`
Repeated while the cost tuple strictly drops (at most 8 rounds; the measured
maximum is 2 improving rounds):
1. Stages 1–2;
2. phase folding;
3. H-reduction;
4. list regions (reordered for the fewest regions, or as listed for the
   fifth candidate);
5. convex regions;
6. Clifford blocks.

Each pass is followed by Stage 1–2 cleanup.

### 4.14 A traced example: two Toffoli gates (30 gates → 0)
1. **Stages 1–2:** the two adjacent H gates on qubit 2 cancel, leaving 27
   gates.
2. **Region ordering:** delaying H gates gives `H₂ [25 H-free gates] H₂`,
   which is one region.
3. **That region is CCZ·CCZ = I:** its linear map is the identity, b = 0, and
   its phase table is all zeros. The built term list is *not* empty, but the
   canonical list derived from the table is.
4. **The canonical candidate rebuilds to `[]`.** It passes the exact check and
   is cheapest.
5. **Cleanup:** `H₂ H₂` cancel. Result: **0 gates**, verified.

---

## 5. Every identity used, with justification

Notation: sequences are in time order, written left to right (`A B` means A
first). `≅` means equal up to global phase. CX = |0⟩⟨0|⊗I + |1⟩⟨1|⊗X.

**One qubit**

| # | Identity | Justification | Used in |
|---|---|---|---|
| 1 | H H = X X = Z Z = I; S SDG = T TDG = I | Involutions and inverse pairs (direct matrix product) | Stage 1–2 |
| 2 | Z = T⁴, S = T², SDG = T⁶, TDG = T⁷, T⁸ = I | Diagonal matrices: exponents of e^{iπ/4} add | Stage 1, phase segments, folding, regions |
| 3 | H X H = Z, H Z H = X | H maps the X basis to the Z basis (H X H† = Z) | windows, H-reduction |
| 4 | X T X ≅ TDG, X S X ≅ SDG, X Z X = −Z | X swaps \|0⟩ and \|1⟩: diag(1, e^{iθ}) → e^{iθ}·diag(1, e^{−iθ}) | windows; region/fold sign rule |
| 5 | X T ≅ TDG X (phase on the complemented bit negates the exponent) | Consequence of 4; k·(1 ⊕ p) = k − k·p, and the constant is global phase | `_region_build`, `_phase_fold` |
| 6 | H S H ≅ SDG H SDG, H SDG H ≅ S H S | Clifford conjugation (verified numerically in the tests) | H-reduction, beam |
| 7 | Any one-qubit run ≅ its table entry or normal form | Equal exact SO(3) keys ⇔ equal operators up to phase | one-qubit runs |

**Two qubits**

| # | Identity | Justification | Used in |
|---|---|---|---|
| 8 | CX CX = I | Involution | everywhere |
| 9 | A diagonal gate (Z, S, SDG, T, TDG) on the control commutes with CX | It commutes with both projectors \|0⟩⟨0\| and \|1⟩⟨1\| | Stage 2, segments |
| 10 | X on the target commutes with CX | X commutes with both I and X | Stage 2 |
| 11 | CX gates sharing only a control, or only a target, commute | The same control bit is read and not changed; XORs on the target commute | Stage 2, beam |
| 12 | Gates on disjoint qubits commute | They act on different tensor factors | everywhere (reordering) |
| 13 | X_c CX = CX X_c X_t; Z_t CX = CX Z_c Z_t (and so X_c CX X_c = CX X_t, CX Z_t CX = Z_c Z_t) | Stabilizer conjugation rules for CX | H-reduction, beam |
| 14 | (H⊗H) CX₀₁ (H⊗H) = CX₁₀ | H swaps the X and Z bases on both qubits, swapping control and target roles | H-reduction, beam, blocks |
| 15 | Any two-qubit Clifford ≅ its table entry | Equal tableau images of X₀, Z₀, X₁, Z₁ (with signs) ⇔ equal operators up to phase | Clifford blocks |

**Many qubits**

| # | Identity | Justification | Used in |
|---|---|---|---|
| 16 | An H-free circuit = ω^{f(x)}\|Ax⊕b⟩; two are equal ⇔ same A, b and f up to a constant | CX, X and phases act as affine maps and diagonal phases; ω^{f(0)} is the global phase | regions |
| 17 | 4a + 4b ≡ 4(a ⊕ b) (mod 8) for bits a, b | 4a + 4b − 4(a ⊕ b) = 8ab | Z-variants |
| 18 | 2^{d−1}·x_{i₁}⋯x_{i_d} = Σ_{∅≠T⊆S} (−1)^{\|T\|+1}·(⊕_{i∈T} x_i) | Inclusion–exclusion; with the Möbius transform this gives a unique term list | canonical terms |
| 19 | (parity tree) · phase on target · (tree reversed) = phase on the parity | The tree XORs the parity onto the target and the reverse undoes it; ladders and trees are equivalent | gadgets, Pipeline B |
| 20 | Phase gates acting on the same path-variable parity can be combined at any one of their positions | The circuit's sum-over-paths phase is a *sum* over phase gates, so regrouping terms leaves it unchanged | phase folding |
| 21 | T ≅ R_Z(π/4); C·R_Z·C† = R_{CZC†}; R_P(a)·R_P(b) = R_P(a+b); R_{−P}(θ) = R_P(−θ); rotations whose Paulis commute can be reordered | Rotations are exponentials of Paulis; Pauli algebra | Pipeline B, T-blocks |
| 22 | R_X = H·R_Z·H; R_Y = (SDG H) · R_Z · (H S) in time order | Basis change (verified numerically) | Pipeline B |
| 23 | Reordering gates while keeping each wire's order leaves the unitary and all metrics unchanged | Only disjoint gates swap (identity 12); depth depends only on the dependency graph | region ordering, convex regions |

---

## 6. Existing algorithms used, with citations

Full references are in §12.

| Method | Where in `submission.py` | What we took and what we built |
|---|---|---|
| **Gate cancellation via commutation, Hadamard reduction, rotation merging** [Nam et al. 2018] | `_commute_cancel_pass`, `_h_reduce`, phase merging | The overall idea of a rule-based optimizer that commutes gates to cancel them, reduces H gates to enlarge phase-mergeable regions, and merges phase rotations, iterated. Our rules, data structures and pass ordering are our own. Every rule is listed in §5 and tested. |
| **Phase polynomials / sum-over-paths, T-par** [Amy, Maslov & Mosca 2014] | `_region_build` (regions), `_phase_fold` (folding across H gates) | The representation of CNOT+phase circuits as an affine map plus phase polynomial, and path variables for H gates. We use it to merge phases and resynthesize regions, minimizing *weighted cost*. We do not implement the matroid-partitioning T-depth algorithm. |
| **GraySynth** [Amy, Azimzadeh & Mosca 2018] | `_reconstruct_graysynth` | The idea of synthesizing a parity network by recursively splitting the parities on the most decisive row and collapsing each group onto one target wire (Gray-code-like). Our implementation follows that structure; it is not a line-by-line transcription of the paper's pseudocode. |
| **Pauli-rotation T merging** [Zhang & Chen 2019] | `_to_rotations`, `_merge_rotations`, `_pipeline_b`, `_resynth_t_block` | Rewriting T gates as π/4 rotations about conjugated Paulis and merging commuting equal-axis rotations. Our synthesis back to gates, and the use of the exact two-qubit Clifford table in T-blocks, are our own. |
| **Stabilizer tableau update rules** [Aaronson & Gottesman 2004] | `_pauli_forward`, `_pauli_backward`, `_cliff2_key` | The conjugation rules for H, S and CX on Pauli strings, including the sign rule for CX. |
| **Matsumoto–Amano normal form** [Matsumoto & Amano 2008; Giles & Selinger 2013] | `_ma_normal_form`, `_so3_of` | Uniqueness and T-optimality of the normal form (Clifford, then HT/SHT syllables, with an optional T). We implemented the reduction by stripping, in exact SO(3) arithmetic, the syllable that lowers the denominator exponent. |
| **Beam search** [Lowerre 1976] | `_beam_search` | Keep the best *k* partial solutions at each step. Our moves, deduplication and budget are our own. |
| Breadth-first search; Gauss–Jordan elimination over GF(2); Möbius transform on subsets; Pareto (multi-objective) labels; dependency-graph convexity | `_linear_bfs`, `_exact_small_paths`, `_cliff2_table`, `_window_table_long`, `_linear_ops_gauss`, `_canonical_terms`, `_convex_regions` | Standard textbook techniques, implemented here. |
| Organizers' starter code | end of both `.py` files | `validate_circuit`, `validate_config`, `circuit_unitary`, `equivalent_up_to_global_phase`, `circuit_metrics`, `circuit_cost`, `evaluate_optimization`: unchanged. |

Facts quoted in this README: the 7-T Toffoli is T-optimal without ancillae
[Gosset et al. 2013], and needs 6 CX [Shende & Markov 2008]. That's why the
optimizer keeps a single Toffoli at 7 T and 6 CX, improving only its depth.
A Toffoli with T-depth 3 exists [Amy, Maslov, Mosca & Roetteler 2013]; T-depth
is not part of this challenge's cost.

---

## 7. Trade-offs and design decisions

Every decision below was measured with `bench.py` (35 circuits × 4 weight
profiles) unless noted otherwise.

| Decision | Reason (measured where noted) |
|---|---|
| **Exact identities only** | The challenge gives no credit for inequivalent output. Exactness also makes every step explainable. |
| **Deterministic work budgets, not time limits** | Same input, same output on any machine; a time limit would make results depend on machine speed. |
| **Three verification layers** (§3) | Defence in depth against bugs. 1,455 checks, 0 rejections. Strict mode makes the tests fail loudly. |
| **Candidates: the input, A, B, B after A, A on the given listing** | Each of A, B and B after A is the only cheapest result in some cases (A 46, B 6, B after A 11 of 140, first round). The fifth candidate exists because the fewest-regions reordering helps some weights and hurts others (replacing it outright: −2.0% w2-heavy but +2.4% w1-heavy); as an extra candidate it changed no profile for the worse (−0.2% w2-heavy, −1.1% balanced, −0.3% w1-heavy, 0 wd-heavy). Choosing the listing by cost after the region pass alone, and alternative pass orders, were tested and were worse. Two earlier candidates were removed: the improved Stage 1–2 result can never beat A, which starts from it, and the legacy Stage 2 path was never cheaper than it. Removing both changed 7 of 140 outputs, all at equal cost. "Never worse than Stage 2" is now checked by tests against `stage2_baseline.py` rather than holding by construction. |
| **Segment summing places the merged phase at the lowest-depth position** | Placing it first raised depth from 6 to 8 in a found case. |
| **Beam search on top of greedy passes** | Without it, mean cost rises 6–12% (table in §8). Width 16 vs 6: 2.6% lower cost at about 0.7 s median per call (measured before the caching row below). Width 48 gained only 0.4% more. Runtime is about 2× because exact idempotence needs a confirmation round. |
| **Cross-region moves and T-blocks: beam-only** | Applied greedily they *hurt* (cross-regions 0.725 → 0.718 of Stage 2 when removed; T-blocks hurt w1-heavy), because an immediately accepted move blocks better later ones. |
| **Depth placement only when `wd > w1 + w2`** | +2.2% on depth-heavy weights, but −0.9% on balanced weights when always on. This rule was chosen from the benchmark corpus. |
| **Tables built in code on first use** | About 1.8 s once per process (the two-qubit Clifford table), but nothing is shipped as data and everything is reproducible from code. Only matters if each circuit runs in a fresh process. |
| **Exact keys for one-qubit tables** (Matsumoto–Amano) | Removes floating-point rounding from decisions. Every table lookup uses exact keys; floats appear only in the commutation test and in verification. |
| **Tried and removed:** closure-based move skipping, lookahead scoring, Pipeline B ladder-target choice, a depth-optimal CX-network candidate | Measured respectively: same cost ~45% slower; worse cost; no end-to-end change; +0.6% w1-heavy but −0.9% w2-heavy even at equal CX count. |
| **Kept although they look redundant:** list regions, H-reduction, "B after A", the BFS-greedy rebuild for ≤ 3 wires, Stage 1 inside the Stage 1–2 cleanup | Removing each one made at least one weight profile worse. Equal-cost choices change which later improvements are reachable. |
| **Caching** (`_best_run`, `_commutes`; later `_region_candidates`, `_resynth_t_block`) | 33% faster, then a further ~1.6× (beam states share most regions and blocks), each time with all 140 benchmark outputs byte-identical. |

---

## 8. Results

The full report with example circuits, shown before and after, is **`REPORT.md`**
(generated by `python report.py`). Summary:

`bench.py --quick`: 35 circuits (random general, H-heavy, phase-heavy,
CX-heavy; 2–6 qubits; 10–60 gates; plus Toffoli, Toffoli×2 and
compute/uncompute). Mean cost as a fraction of the input, and of our Stage 1–2
baseline:

| Weights (w1, w2, wd) | vs input | vs Stage 2 baseline |
|---|---|---|
| CX-heavy (1, 6, 0.2) | 0.452 | 0.658 |
| balanced (1, 1, 1) | 0.425 | 0.675 |
| one-qubit only (1, 0, 0) | 0.396 | 0.678 |
| depth only (0, 0, 1) | 0.380 | 0.602 |

**Optimality on small circuits** (exhaustive search for the true minimum;
gate-count weights; `python bench.py --quick --optimality`):

| Circuits | Exactly optimal |
|---|---|
| 2-qubit, optimum ≤ 4 gates (random circuits of 6–16 gates) | 99.8% (1037/1039); the other 2 are 1 gate over |
| 3-qubit, optimum ≤ 4 gates (random circuits of 5–10 gates) | 400/400, including 222/222 with optimum exactly 4 |
| 3-qubit, optimum exactly 5 (random 5-gate circuits outside the ≤ 4 table) | 200/200 |

The 3-qubit search enumerates all 45,632 unitaries that need at most 4 gates.

**What each step contributes** (`python bench.py --quick --ablation`: change
in mean cost when the step is removed everywhere it runs — greedy loop,
candidates and beam moves):

| Removed | CX-heavy | balanced | one-qubit | depth |
|---|---|---|---|---|
| beam search | +6.4% | +8.6% | +9.1% | +11.8% |
| beam local moves | +5.6% | +7.5% | +8.2% | +9.6% |
| Clifford blocks | +6.8% | +3.0% | +6.4% | +3.7% |
| list regions | +0.2% | +1.9% | +3.4% | +3.0% |
| convex regions | +1.6% | +0.6% | +1.2% | −0.4% |
| Pipeline B | +0.2% | −0.7% | +0.6% | +0.4% |
| phase folding | +0.3% | +0.2% | +0.0% | +0.4% |
| depth placement | 0 | 0 | 0 | +0.4% |
| cross-region moves | 0 | +0.1% | 0 | +0.4% |
| T-blocks | +0.1% | +0.1% | +0.3% | 0 |
| H-reduction | −0.1% | −0.2% | +1.2% | 0 |

Every step helps at least one profile. Pipeline B, H-reduction and convex
regions help under some weights and cost a little under others; with 35
circuits per profile those differences are a few circuits, so all are kept.

**Known cases** (all pinned by tests):
- the guidelines' example: `CX01 Z0 CX01 → Z0`, cost 13.6 → 1.2;
- Toffoli × 2: 30 gates → 0;
- compute / H / uncompute → `H`;
- SWAP stays at 3 CX (optimal);
- a single Toffoli keeps exactly the same 15 gates (6 CX, 2 H, 4 T, 3 TDG), but depth drops from 11 to 9 (cost 47.2 → 46.8):
  - **how:** Toffoli = H·CCZ·H, and CCZ is a phase polynomial over all 7 parities of 3 bits. Pipeline B rebuilds it with a different 6-CX network, and beam-search moves then slide phase gates into idle time slots. No single greedy pass finds this;
  - **T-depth** stays 4. Amy, Maslov, Mosca & Roetteler (2013) report a Toffoli with T-depth 3 and total depth 9 (from a figure caption; their full paper was not accessible here, so their depth model and CX count are unconfirmed). Ours matches their total depth, but not their T-depth. Our cost does not count T-depth, so it is not targeted. This reproduces a known result; it is not a new one.

**Standard benchmark circuits, against PyZX, Qiskit and published results**
(`python benchmarks/compare_standard.py`; needs `pip install pyzx qiskit`).
Of the standard benchmark suite, exactly four distinct inputs fit the
challenge (≤ 6 qubits, exact Clifford+T): `tof_3`, `barenco_tof_3`, `mod5_4`
and `qft_4`, all on 5 qubits; every other one needs more qubits
(`--scan` lists them; files and sources in `benchmarks/circuits/`). For each
circuit and weight profile, PyZX 0.10.8 is credited with the cheaper of its two
strategies, Qiskit 2.6.2 with the cheapest of its levels 0–3, and "published"
with the cheapest of the optimized versions distributed with PyZX: Nam et al.
2018 (light and heavy), T-par, Feynman and PyZX's stored result. Each is also
credited with the unchanged input, and every output from every source passed
the organizers' equivalence check.

| Circuit (input: gates, CX, T) | Weights | Input | PyZX | Qiskit | Published | Ours |
|---|---|---|---|---|---|---|
| tof_3 (45, 18, 21) | (1, 6, 0.2) | 141.2 | 141.2 | 140.2 | 110.8 | **106.0** |
| | balanced | 76.0 | 76.0 | 75.0 | 64.0 | **61.0** |
| | one-qubit | 27.0 | 26.0 | 26.0 | **21.0** | 23.0 |
| | depth | 31.0 | 31.0 | 31.0 | 29.0 | **26.0** |
| barenco_tof_3 (58, 24, 28) | (1, 6, 0.2) | 186.4 | 186.4 | 184.4 | 137.4 | **126.8** |
| | balanced | 100.0 | 100.0 | 98.0 | 77.0 | **74.0** |
| | one-qubit | 34.0 | 34.0 | 32.0 | **22.0** | 24.0 |
| | depth | 42.0 | 42.0 | 42.0 | 37.0 | **33.0** |
| mod5_4 (63, 28, 28) | (1, 6, 0.2) | 212.6 | 106.8 | 209.6 | 187.8 | **94.6** |
| | balanced | 111.0 | 52.0 | 108.0 | 93.0 | **44.0** |
| | one-qubit | 35.0 | **16.0** | 32.0 | 23.0 | 17.0 |
| | depth | 48.0 | 19.0 | 48.0 | 44.0 | **13.0** |
| qft_4 (179, 46, 69) | (1, 6, 0.2) | 438.2 | 438.2 | 438.2 | 436.2 | **373.6** |
| | balanced | 325.0 | 325.0 | 325.0 | 325.0 | **282.0** |
| | one-qubit | 133.0 | 133.0 | 133.0 | 127.0 | **115.0** |
| | depth | 146.0 | 146.0 | 146.0 | 146.0 | **119.0** |

- **Result:** ours is strictly cheapest against Qiskit in 16 of 16 cases,
  against PyZX in 15 of 16, and against the published versions listed above in
  14 of 16. Average saving per case: 29.2% against Qiskit, 18.3% against PyZX. The exceptions are all one-qubit-only weights: Nam et al. and
  Feynman on `tof_3` and `barenco_tof_3`, and PyZX on `mod5_4`.
- **In standard measures** (our output under (1, 6, 0.2)): CX 13, 16, 13 and
  39, against 14, 18, 26 and 45 for the best of those published versions
  (PyZX run live reaches 14 on `mod5_4`); T-count 15, 16, 8 and 67, equal to the
  lowest among them; we do not target T at all.
- **Scope of the "published" column:** it covers only the optimized versions
  distributed with PyZX, from 2014–2018 (Nam et al., T-par, Feynman, PyZX's
  stored result). Newer optimizers (for example VOQC, Quartz, QUESO) report
  results on these circuits that were not compared, and none of the published
  tools optimizes our weighted cost, so this is not a state-of-the-art claim.
- **Speed:** PyZX and Qiskit take well under a second per circuit; we take
  1–4 s on the three Toffoli-based circuits and 17–30 s on the 179-gate `qft_4`,
  three times the size of our own benchmark circuits.

**Comparison with Qiskit and PyZX on our own benchmark** (`python
compare_qiskit.py`; PyZX is included when installed). This is the 35-circuit
corpus above, which was also used during development to tune the depth rule,
the beam width and the fifth candidate, so it is breadth evidence; the
standard circuits above are the headline. Each tool is credited per circuit
with the cheapest valid result (Qiskit: levels 0–3; PyZX: its two strategies)
or the unchanged input. All outputs passed the organizers' equivalence check.

| Weights | Input | Qiskit (best) | PyZX (best) | Ours | Ours / Qiskit | Ours / PyZX |
|---|---|---|---|---|---|---|
| CX-heavy (1, 6, 0.2) | 87.90 | 58.29 | 49.88 | 39.75 | 0.682 | 0.797 |
| balanced (1, 1, 1) | 54.06 | 34.03 | 29.20 | 22.97 | 0.675 | 0.787 |
| one-qubit only (1, 0, 0) | 23.66 | 14.34 | 11.77 | 9.37 | 0.653 | 0.796 |
| depth only (0, 0, 1) | 20.37 | 12.71 | 10.86 | 7.74 | 0.609 | 0.713 |

- **Average saving per circuit** (mean over cases of (tool − ours) / tool):
  30.7% against Qiskit and 18.4% against PyZX over all 140 cases; 30.3% and
  19.3% over the 32 random circuits (128 cases).
- **Never costlier than Qiskit** (cheaper in 126 of 140, the rest ties).
  Against PyZX: cheaper in 102, costlier in 1 (a 2-qubit, 60-gate CX-heavy
  circuit under (1, 6, 0.2): 66.2 against 56.4).
- **Qiskit's levels:** its level 1 (cancellation and commutation) was its best
  single level. Levels 2–3 resynthesize blocks into arbitrary rotations and
  then rebuild them from H, S and T, which made circuits about 3× *costlier*
  in this gate set.
- **Speed:** both tools are far faster (Qiskit about 0.5 s for all 35
  circuits) and neither takes our cost weights into account.

**Runtime:** median about 0.3–0.5 s per circuit (up to 60 gates, 6 qubits;
0.27–0.55 s depending on the weight profile), and up to about 3–4.5 s on the
largest benchmark circuits. Add about 1.7–1.8 s for the Clifford table on the
first multi-qubit call in a process. Cost grows roughly linearly with
beam width, and with gates × moves per beam step.

The development history, with per-step measurements, is in `docs/`
(`stage2.md`, `stage4.md`, `stage5.md`, `stage6_plan.md`).

---

## 9. Limitations

- **Not globally optimal.** Exact optimization is intractable in general.
  Beam search is a heuristic, and wider beams still help slightly.
- **CX synthesis for regions of 4–6 wires** uses GraySynth-style and
  Gauss–Jordan networks, not an exact search. Combined CX-plus-phase synthesis
  is exact only for ≤ 3 wires.
- **One-qubit runs needing more than 7 gates** use the best split, not a
  proven optimum.
- **Exact two-qubit tables cover Clifford blocks only.** T-containing blocks
  use the rotation method, and three-qubit Clifford blocks have no exact
  table.
- **Depth is optimized indirectly** (placement, parity trees, commuting
  swaps). A dedicated depth scheduler is future work.
- **Some rules were tuned on our benchmark corpus** (the depth rule, the beam
  width). They may be suboptimal for very different inputs, though the
  "never worse" guarantee always holds.

---

## 10. Testing and reproducibility

```bash
python -m pytest test_stage1.py test_stage2.py test_stage4.py -q
```

That runs 120 tests in about 7 minutes. `conftest.py` turns on strict
mode, and uses a reduced beam for the bulk tests;
`test_D_full_beam_guarantees` runs the production beam.

```bash
SUBMISSION_MODULE=stage2_baseline python -m pytest test_stage1.py test_stage2.py -q
```

That runs the same Stage 1/2 tests against the frozen baseline.

**Reproducing every number in this README.** All runs are deterministic
(fixed seeds, no time limits), so they print the same numbers on any machine;
only the seconds vary.

| Numbers | Command | Section |
|---|---|---|
| corpus results vs input and Stage 2 baseline; runtime per call | `python bench.py --quick` | §8 |
| 2- and 3-qubit optimality (1037/1039, 400/400, 222/222, 200/200) | `python bench.py --quick --optimality` | §8 |
| what each step contributes | `python bench.py --quick --ablation` | §8 |
| example circuits, corpus table, runtime by size (`REPORT.md`) | `python report.py` | §8 |
| Qiskit and PyZX on our corpus (table, average savings, cheaper/costlier counts) | `python compare_qiskit.py` | §8 |
| PyZX, Qiskit and published results on the standard circuits | `python benchmarks/compare_standard.py` | §8 |
| which standard circuits fit the challenge | `python benchmarks/compare_standard.py --scan <PyZX clone>/circuits` | §8 |

The last three need `python -m pip install qiskit pyzx` (Qiskit 2.6.2 and
PyZX 0.10.8 were used); the optimizer itself needs only NumPy.

**What the tests check:**
- every identity in §5;
- exact canonical forms against real unitaries, in both directions;
- table sizes: 11,520 two-qubit Cliffords, 24 one-qubit Cliffords,
  |GL(4,2)| = 20,160;
- Matsumoto–Amano uniqueness and T-optimality against brute force;
- equivalence, validity and "never costlier than the input or Stage 2" on
  random circuits under 6 weight configurations;
- unchanged inputs and exact idempotence;
- byte-identical output across processes with different `PYTHONHASHSEED`;
- the work budget is never reached.

**Mutation testing:** deliberately broken variants of the code were caught by
the tests. Two tests were added after mutants initially survived.

---

## 11. Repository map

| File | Purpose |
|---|---|
| `submission.py` | **The submission**: the full optimizer plus the organizers' starter helpers |
| `ALGORITHMS.md` | Precise reference: pseudocode, parameters, heuristics and the reason for each choice |
| `stage2_baseline.py` | Frozen Stage 1–2 optimizer, the reference for tests and benchmarks. Not submitted. |
| `report.py` -> `REPORT.md` | The results report: example circuits for every level shown before and after, with equivalence checks, resources, weight comparisons, corpus results and runtime scaling. Regenerate with `python report.py`. |
| `compare_qiskit.py` | Optional comparison with Qiskit's transpiler on our corpus (needs `pip install qiskit`) |
| `benchmarks/compare_standard.py` | Optional comparison with PyZX, Qiskit and published results on the standard benchmark circuits (needs `pip install pyzx qiskit`) |
| `benchmarks/circuits/` | The standard benchmark circuits and published optimized versions, copied from PyZX (Apache-2.0), with sources |
| `bench.py` | Benchmark, optimality-gap and ablation harness |
| `test_stage1.py`, `test_stage2.py` | Tests from Stages 1–2, phrased as goal checks (equivalence, cost under the given weights); run against either module |
| `test_stage4.py` | Tests of every component of the full optimizer |
| `conftest.py` | Strict mode, fast beam for the bulk tests, module switch |
| `docs/` | Design specs, plans and measured results by stage |

---

## 12. Attribution and references

**Starter code.** Qublitz 2026 organizers: the circuit format, validators,
unitary checker, metrics and cost function (kept unchanged in `submission.py`
and `stage2_baseline.py`).

**Libraries.**
- NumPy: C. R. Harris et al., "Array programming with NumPy", *Nature* 585, 357–362 (2020). doi:10.1038/s41586-020-2649-2.
- pytest: tests only.
- Qiskit (optional, comparison script only, not used by the optimizer): Qiskit contributors, *Qiskit: An Open-source Framework for Quantum Computing*, https://github.com/Qiskit/qiskit.
- PyZX (optional, comparison script only, not used by the optimizer): A. Kissinger, J. van de Wetering, "PyZX: Large Scale Automated Diagrammatic Reasoning," *EPTCS* 318, 229–241 (2020), [doi:10.4204/EPTCS.318.14](https://doi.org/10.4204/EPTCS.318.14); https://github.com/zxcalc/pyzx. The standard benchmark circuits in `benchmarks/circuits/` are copied from its repository (Apache License 2.0).

**AI assistance.** This project was developed with Claude (Anthropic), used
through Claude Code. Claude:
- researched and verified identities and citations;
- proposed the designs;
- wrote the implementation, tests and documentation;
- ran the measurements.

The team specified each stage's goals and constraints (exactness,
determinism, cost-awareness, "no loss" rules for changes), reviewed the
designs and results, and directed every decision recorded in §7 and in
`docs/`.

**Published work** (verified bibliographic details; DOIs checked 22 November 2026). Journal DOIs where one exists; arXiv DOIs for preprints without a journal version, and for the two *Quantum Inf. Comput.* papers, whose journal DOI could not be confirmed:
- Y. Nam, N. J. Ross, Y. Su, A. M. Childs, D. Maslov, "Automated optimization of large quantum circuits with continuous parameters," *npj Quantum Information* **4**, 23 (2018). [doi:10.1038/s41534-018-0072-4](https://doi.org/10.1038/s41534-018-0072-4); arXiv:1710.07345.
- M. Amy, D. Maslov, M. Mosca, "Polynomial-time T-depth optimization of Clifford+T circuits via matroid partitioning," *IEEE Trans. CAD* **33**(10), 1476–1489 (2014). [doi:10.1109/TCAD.2014.2341953](https://doi.org/10.1109/TCAD.2014.2341953); arXiv:1303.2042.
- M. Amy, P. Azimzadeh, M. Mosca, "On the CNOT-complexity of CNOT-PHASE circuits," *Quantum Sci. Technol.* **4**(1), 015002 (2018). [doi:10.1088/2058-9565/aad8ca](https://doi.org/10.1088/2058-9565/aad8ca); arXiv:1712.01859.
- F. Zhang, J. Chen, "Optimizing T gates in Clifford+T circuit as π/4 rotations around Paulis," arXiv:1903.12456 (2019). [doi:10.48550/arXiv.1903.12456](https://doi.org/10.48550/arXiv.1903.12456).
- S. Aaronson, D. Gottesman, "Improved simulation of stabilizer circuits," *Phys. Rev. A* **70**, 052328 (2004). [doi:10.1103/PhysRevA.70.052328](https://doi.org/10.1103/PhysRevA.70.052328); arXiv:quant-ph/0406196.
- K. Matsumoto, K. Amano, "Representation of quantum circuits with Clifford and π/8 gates," arXiv:0806.3834 (2008). [doi:10.48550/arXiv.0806.3834](https://doi.org/10.48550/arXiv.0806.3834).
- B. Giles, P. Selinger, "Remarks on Matsumoto and Amano's normal form for single-qubit Clifford+T operators," arXiv:1312.6584 (2013). [doi:10.48550/arXiv.1312.6584](https://doi.org/10.48550/arXiv.1312.6584).
- B. T. Lowerre, "The HARPY speech recognition system," Ph.D. thesis, Carnegie Mellon University (1976). (No DOI.)
- D. Gosset, V. Kliuchnikov, M. Mosca, V. Russo, "An algorithm for the T-count," *Quantum Inf. Comput.* **14**, 1261–1276 (2014). arXiv:1308.4134, [doi:10.48550/arXiv.1308.4134](https://doi.org/10.48550/arXiv.1308.4134).
- V. V. Shende, I. L. Markov, "On the CNOT-cost of TOFFOLI gates," *Quantum Inf. Comput.* **9**(5–6), 461–486 (2009). arXiv:0803.2316, [doi:10.48550/arXiv.0803.2316](https://doi.org/10.48550/arXiv.0803.2316).
- M. Amy, D. Maslov, M. Mosca, M. Roetteler, "A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum circuits," *IEEE Trans. CAD* **32**(6), 818–830 (2013). [doi:10.1109/TCAD.2013.2244643](https://doi.org/10.1109/TCAD.2013.2244643); arXiv:1206.0758.
