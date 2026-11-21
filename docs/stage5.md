# Stage 5: Results

Implementation: `submission_stage4.py`. `submission.py` is unchanged: it is the
frozen Stage 2 baseline (option B). The plan is in `docs/stage5_plan.md`.
All numbers come from `bench.py --quick` (35 circuits, fixed seed).

## Progress by step

The first table gives Stage 4's cost as a fraction of Stage 2's (lower is better).

| After step | w2-heavy (1,6,0.2) | balanced (1,1,1) | w1-heavy (1,0,0) | wd-heavy (0,0,1) | 2-qubit exactly optimal* |
|---|---|---|---|---|---|
| Start of Stage 5 (Stage 4 as of `f776bf8`) | 0.838 | 0.836 | 0.845 | 0.802 | 90.3% |
| B+C: phase folding, 2-qubit Clifford blocks | 0.730 | 0.751 | 0.750 | 0.698 | 97.9% |
| E+F: exact ≤3-wire synthesis, GraySynth, convex regions, optimal windows, segment summing | 0.718 | 0.738 | 0.750 | 0.680 | 97.9% |
| D: beam search, width 6 | 0.677 | 0.700 | 0.715 | 0.631 | 99.6% |
| **D: beam search, width 16 (production)** | **0.659** | **0.691** | **0.684** | **0.613** | **99.6%** |

\* Measured on 2-qubit circuits whose optimum is ≤ 4 gates (524 cases),
counting gates.

Final costs against the **input**: 0.453 (w2-heavy), 0.435 (balanced),
0.400 (w1-heavy), 0.387 (wd-heavy). Overall, the cost falls to about 39–45%
of the input.

## Runtime (production settings)

- **Per `optimize` call:** median 0.76–1.26 s and maximum about 11 s,
  depending on the weight profile.
- **One-time setup:** building the Clifford table takes about 1.7 s, on the
  first call in a process.
- **Budget:** the beam's evaluation budget was never reached. Every search
  stopped because it ran out of improvements.

## What each step contributed (ablation before the beam)

- **2-qubit Clifford blocks:** the largest single gain. Removing them takes
  w2-heavy from 0.725 back to 0.825.
- **Pipeline B:** now matters (0.754 without it), because its output feeds the
  block pass.
- **Convex regions:** 0.737 without them.
- **Phase folding, H-reduction and list-slicing regions:** small but positive.
- **Cross-region moves:** made results worse inside the greedy loop
  (0.718 without them vs 0.725 with them). They are off there and used only as
  a beam-search move.

## Problems found and fixed during this stage

- **Segment summing raised depth.** Placing the merged phase at the first
  position increased depth from 6 to 8 in a found case. The pass now places it
  at the lowest-depth position.
- **"Never worse than Stage 2" stopped holding.** The Stage 4 file's improved
  Stage 2 differs from `submission.py`, so the guarantee broke (57.6 vs 57.2).
  A legacy path that reproduces `submission.py`'s Stage 2 exactly is now always
  a candidate. A test pins it to the real file.

## Guarantees (all tested)

- **Correctness:** every move is an exact identity, every candidate is verified,
  and the official check runs on the final result. Strict mode, used in the
  tests, raises on any verification failure.
- **Never worse:** under the weights passed in, the result is never costlier
  than the input or than Stage 2.
- **Idempotence and determinism:** the result is idempotent and deterministic
  across processes and hash seeds.
- **Full-beam check:** `test_D_full_beam_guarantees` runs the production beam
  settings. The bulk tests use the reduced "fast" profile, so the suites finish
  in about 11 minutes.

## Remaining limits

- **Wide regions:** CX synthesis for regions of 4–6 wires uses GraySynth, a
  heuristic, not an exact search.
- **One-qubit runs:** runs whose optimum needs more than 7 gates use the best
  split into chunks, not a proven optimum.
- **Beam search:** it is a heuristic. A wider beam keeps helping
  (width 6 → 16 gave −2.6%) at more runtime.
- **Two-qubit blocks:** only Clifford blocks have an exact table. Blocks
  containing T rely on phase folding and the other passes.
