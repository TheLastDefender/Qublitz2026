# Stage 2: Commutation and Sandwich Identities

Stage 1 removed gates that are **adjacent** on their wires. Stage 2 removes
gates that only become adjacent once you account for **commutation**, and it
collapses short runs on a single wire using **sandwich identities**.

Notation: gate sequences are written in circuit (time) order, e.g. `CX01 Z0 CX01`
means CX first. `CX01` means control 0, target 1. "Equal" always means equal
up to a global phase, which the challenge allows (`U_out = e^{iφ} U_in`).

---

## 1. How the simplifications were found

Instead of listing identities from memory, I ran an exhaustive search over
2-qubit circuits (16 gate choices: 7 one-qubit gates on each wire plus `CX01`
and `CX10`). The search looked for sequences with a strictly shorter equivalent
and grouped them:

| Length | What the search found | Handled by |
|---|---|---|
| 2 | `H H`, `X X`, `CX CX` → I, phase merges (`T T` → S, `S SDG` → I, …) | Stage 1 |
| 3, one wire | sandwiches: `H X H` → Z, `X T X` → TDG, … | Stage 2 windows |
| 3, with CX | `S0 CX01 S0` → `Z0 CX01` (phase passes the control) | Stage 2 commutation |
| 3–4, with CX | `X0 CX01 X0` → `CX01 X1`, `H0 H1 CX01 H0` → `CX10 H1` | **not yet** (Stage 3) |

---

## 2. Commutation rules

If gate `g` commutes with everything between it and a partner `p` on the same
wires, `g` can slide next to `p`:

```
g M p  =  M g p        (valid because g·M = M·g)
```

* If `p = g` and `g` is self-inverse (H, X, CX): `M g g = M`, so both vanish.
* If `g` and `p` are phase gates on the same wire: `M (g p)`, so they fuse into one phase.

The code never actually moves `g` in the output. It drops `g` and rewrites `p`
in place, which is the same unitary by the line above.

### Which gates commute with CX, and why

Write CX as "do nothing if the control is 0, apply X to the target if it is 1":

```
CX = |0⟩⟨0| ⊗ I  +  |1⟩⟨1| ⊗ X
```

| Rule | Why it holds |
|---|---|
| Diagonal gate (Z, S, SDG, T, TDG) on the **control** commutes | A diagonal gate is `a|0⟩⟨0| + b|1⟩⟨1|`, which commutes with both projectors `|0⟩⟨0|` and `|1⟩⟨1|`. It only adds a phase depending on the control bit, and CX never changes the control bit. |
| X on the **target** commutes | In each branch the target gets `I` or `X`, and X commutes with both. Flipping the target before or after a conditional flip ends the same way. |
| CX's sharing a **control** commute (`CX01`, `CX02`) | Both read the same control bit, which neither changes, and they flip different targets. |
| CX's sharing a **target** commute (`CX02`, `CX12`) | The target becomes `t ⊕ a ⊕ b` in either order, because XOR is commutative. |
| Gates on **disjoint** wires commute | They act on separate tensor factors. |

### What does *not* commute (and must not be cancelled through)

| Non-rule | Why it fails |
|---|---|
| X on the **control** | It flips the condition: `X0 CX01 = CX01 X0 X1`. The X leaks onto the target. |
| Z on the **target** | Z anticommutes with X: `Z1 CX01 = CX01 Z0 Z1`. The Z leaks onto the control. |
| `CX01 CX10` | Opposite orientations. Their product is two-thirds of a SWAP, not identity. |
| H with CX on either wire | H swaps the X and Z bases, so neither of the rules above applies. |

### Safety: commutation is checked, not assumed

`_commutes(a, b)` builds both gates' matrices on their shared qubits and tests
`A·B == B·A` exactly. The table above explains *why* the answers come out as
they do, and `test_stage2.py` pins each row (both the "yes" and the "no" lists).

### Example: the PDF worked example

`CX01 Z0 CX01`: Z0 is diagonal on the control, so the first CX slides past it,
meets the second CX, and both cancel. The result is `Z0`, and the cost drops
from 13.6 to 1.2 under the example weights.

---

## 3. Sandwich identities (single-wire windows)

A **window** is 2–4 consecutive one-qubit gates on one wire. Gates on other
wires in between don't matter, because they act on other qubits. Each window is
replaced by the shortest gate list with the same 2×2 unitary up to phase.

Why the key examples hold:

| Identity | Reason |
|---|---|
| `H X H = Z` | H maps the X basis to the Z basis: `H X H† = Z` (and `H† = H`). |
| `H Z H = X` | Same fact, the other way round. |
| `X T X = TDG` | X swaps `|0⟩` and `|1⟩`, so `diag(1, e^{iπ/4})` becomes `diag(e^{iπ/4}, 1) = e^{iπ/4}·diag(1, e^{-iπ/4})`. That is TDG times a global phase. |
| `X S X = SDG` | Same argument with `i` in place of `e^{iπ/4}`. |
| `X Z X = Z` | Same argument: `diag(-1, 1) = -Z`. |
| `T T T = S T` | Phase exponents add: `T^3 = T^2·T`, and that needs two gates. |

**How the table is built:** the code enumerates every one-qubit sequence of
length 0–4 (about 2,800 sequences). It keys each one by its unitary with the
global phase normalized away (`_unitary_key`) and keeps the shortest sequence
per key. That gives 98 distinct unitaries. No identity is typed in by hand,
and a test confirms every table entry is equivalent (via the official checker)
and no longer than its source.

---

## 4. Putting it together

```
optimize:
    Stage 1 (adjacent stack cancellation)
    repeat:
        Stage 2: commutation-cancel and window passes, until neither shortens
        if Stage 2 changed nothing: return
        Stage 1 again on the result
```

* **Termination.** Every accepted Stage 2 rewrite strictly removes gates. A
  pairwise phase merge is accepted only if the result is at most one gate, so
  `S, T → S, T` can never loop. Stage 1 never lengthens. Each round therefore
  shrinks the circuit, and the loop must end.
* **Idempotence.** Stage 1 can rewrite `SDG, TDG` into `Z, T`: same length,
  different gates. If Stage 1 ran only once, a second `optimize` call would
  still change the output. Looping until Stage 2 finds nothing makes the result
  a fixed point, so `optimize(optimize(c)) == optimize(c)`.
* **Cost never increases.** Every rewrite only deletes or shortens, so
  `N_1q`, `N_2q`, and depth never grow. That holds under any non-negative
  weights, and the tests check it for six weight configs.

---

## 5. Verification

* `test_stage2.py`:
  * every commutation and non-commutation row;
  * every sandwich entry and the whole window table;
  * the PDF example and other worked cases;
  * 900 random circuits from three generators (general, commutation-heavy,
    sandwich-heavy). Each checks equivalence, non-increasing metrics and cost
    under six weight configs, unchanged inputs, and "never worse than Stage 1";
  * idempotence on 450 more circuits.
* `test_stage1.py` still passes. Two tests that pinned "no commutation" now
  assert the Stage 1 helper alone, plus the new Stage 2 result.
* **Mutation checks:** making every pair "commute" causes 16 failures, and
  disabling windows causes 2. So the tests catch broken commutation and broken
  sandwiches.
* **Coverage against the exhaustive search** (samples of 400 rules each):
  optimal on 385/400 rules at length 3 and 370/400 at length 4. Every miss
  involves a CX-transforming identity (Section 6).
* **Speed:** a 1,000-gate, 6-qubit random circuit takes about 0.2 s.

---

## 6. Limitations / Stage 3 candidates

These identities are **verified** (checked numerically) but **not implemented**.
They aren't commutations: the gate changes as it passes the CX, so they need
their own rules, and some only pay off for certain weights.

| Identity | Effect | Why not in Stage 2 |
|---|---|---|
| `X0 CX01 X0 = CX01 X1` | 3 → 2 gates | Pauli propagation: X on the control *becomes* X on both wires |
| `Z1 CX01 Z1 = Z0 CX01` | 3 → 2 gates | Same, for Z on the target |
| `H0 H1 CX01 H0 H1 = CX10` | 5 → 1 gate | Hadamards reverse the CX orientation |
| `CX01 CX12 CX01 = CX12 CX02` | 3 → 2 CX | CX-network (linear reversible) resynthesis |
| `CX01 Z1 CX01 = Z0 Z1` | trades 2 CX for 1 extra one-qubit gate | Worth it only for some weights, so it needs the cost-aware choice |

Other current limits:

* Windows are capped at 4 gates. Longer runs are reduced window by window, which may not find the global shortest.
* Stage 2 doesn't read `config` yet. The guidelines ask for weight-aware optimization, which becomes relevant once rewrites can *trade* one resource for another (Stage 3).
