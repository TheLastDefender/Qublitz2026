# Stage 6: Further Gains and Depth Optimization (plan and status)

Order: lowest work and highest expected gain first. Every item is measured on
`bench.py --quick` before it is kept.

| # | Item | Status | Measured effect |
|---|---|---|---|
| 1 | Beam efficiency: provable no-op skipping (closures), lookahead scoring | **tried, removed** | closures: same cost, ~45% slower; lookahead: worse (33.03 vs 32.77). Kept only exact applicability checks. |
| 2 | 3-qubit optimality gap | **measured** | exactly optimal on 400/400 (optimum ≤ 4 gates), 123/124 (4), 200/200 (5). A much stronger beam (width 48) gains only 0.4%. So the remaining gains need *new rewrites*, not more search. |
| 3 | 2-qubit blocks containing T | **done** | as a beam move: w1-heavy 9.46 → 9.40, balanced 23.51 → 23.23; w2-heavy unchanged. Hurt when also used greedily, so beam-only. |
| 4 | Pipeline B: ladder targets chosen by cost | **tried, removed** | B's own output 0.5% cheaper, but end-to-end results identical with fixed targets (wd 7.914, balanced 23.229, w1 9.400): the beam's other moves already recover it. Reverted to keep the code simpler. |
| 5 | 3-qubit Clifford blocks | pending | high effort; the 3-qubit gap measurement found no misses on small instances |
| 6 | Depth optimization driven by `wd` | **plan below** | |

## Depth optimization plan (item 6)

Depth matters only as much as `wd` makes it matter. The plan scales the
effort accordingly.

**0. Gate on the weight.**
- If `wd == 0`, depth costs nothing, so skip every depth-only move. That's
  faster and changes no result.
- Otherwise, compute the depth share of the current cost,
  `wd · depth / cost`. When it is small (say under 5%), depth-only moves get a
  smaller budget.

**1. Choose positions by depth wherever placement is free (no gate-count cost).**
- **Phase folding:** a merged phase may sit at *any* of its occurrences. Today
  it is placed at the first or last one; instead, try every occurrence and keep
  the lowest-depth one, as the segment pass already does.
- **Synthesis:** the 2-qubit Clifford table already stores depth (Pareto
  entries). Extend the same idea to region synthesis: among
  equal-CX-count candidates, prefer the lowest depth. Add a *depth-optimal*
  linear-map candidate for ≤ 4 wires by searching over *layers* of disjoint
  CX gates instead of single CX gates. That's a small search, and it gives the
  minimum depth for the CX network.

**2. Critical-path moves (beam search).** Restrict depth moves to gates on the
current critical path, which keeps branching low:
- commute-swap (exists);
- sliding a diagonal gate along a CX control, or an X along a CX target, into
  an idle slot;
- reordering commuting CX gates (shared control or shared target) so
  independent ones run in parallel.

**3. Trade gates for depth only when the weights pay for it.**
- A rewrite that adds gates but removes depth (for example a log-depth CX tree
  instead of a ladder in Pipeline B or GraySynth) is generated only when
  `wd · (depth saved) > w1·(1q added) + w2·(2q added)` can hold, which needs
  `wd` to be large relative to `w1` and `w2`.
- Candidates are still judged by the full cost, so nothing is accepted on a
  guess.

**4. Measure.** Use the benchmark's wd-heavy (0, 0, 1) and balanced (1, 1, 1)
profiles. Add a depth lower bound for each circuit (the longest single-wire
gate count of the *optimized* circuit is not a bound; use the exact optimum on
small circuits instead) to see how far from optimal depth is.
