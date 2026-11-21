# Stage 4: Results

Implementation: `submission_stage4.py` (same as `submission.py` except for the
Stage 4 code and the new `optimize`). Spec: `docs/stage4_spec.md`; its
"Implementation notes" section lists every place the code differs from the spec.

## How `optimize` works now

1. Build five candidates:
   - the input;
   - Stages 1–2;
   - Pipeline A (H-reduction, then phase-polynomial regions, then cross-region moves);
   - Pipeline B (Zhang & Chen Pauli rotations);
   - B applied to A's result.
2. Optimize **every** candidate with the same convergent loop.
3. Check each against the input unitary, and keep the cheapest under the
   given weights. On a tie, keep the earliest candidate.
4. Repeat from the winner until nothing is strictly cheaper.

## Results

Costs below are `w1·N_1q + w2·N_2q + wd·depth`. The "final" column is what
`optimize` returns. Every final result was verified equivalent to its input.

| circuit | config | input | Stage 2 | Pipeline A | Pipeline B | final |
|---|---|---|---|---|---|---|
| Toffoli (7-T) | w2-heavy (1,6,0.2) | 47.2 | 47.2 | 47.2 | 59.4 | **47.2** |
| Toffoli (7-T) | w1-heavy (1,0,0) | 9 | 9 | 9 | 9 | **9** |
| Toffoli (7-T) | wd-heavy (0,0,1) | 11 | 11 | 11 | 12 | **11** |
| Toffoli x2 | w2-heavy (1,6,0.2) | 94.2 | 91.2 | 0 | 0 | **0** |
| Toffoli x2 | w1-heavy (1,0,0) | 18 | 15 | 0 | 0 | **0** |
| Toffoli x2 | wd-heavy (0,0,1) | 21 | 21 | 0 | 0 | **0** |
| compute / H / uncompute | w2-heavy (1,6,0.2) | 62.6 | 1.2 | 1.2 | 1.2 | **1.2** |
| compute / H / uncompute | w1-heavy (1,0,0) | 23 | 1 | 1 | 1 | **1** |
| compute / H / uncompute | wd-heavy (0,0,1) | 18 | 1 | 1 | 1 | **1** |
| random 4q x 40 | w2-heavy (1,6,0.2) | 57.8 | 36 | 36 | 36.6 | **36** |
| random 4q x 40 | w1-heavy (1,0,0) | 37 | 16 | 10 | 10 | **10** |
| random 4q x 40 | wd-heavy (0,0,1) | 14 | 10 | 8 | 8 | **8** |
| random 6q x 60 | w2-heavy (1,6,0.2) | 99.2 | 65 | 63 | 193.4 | **63** |
| random 6q x 60 | w1-heavy (1,0,0) | 53 | 21 | 18 | 36 | **18** |
| random 6q x 60 | wd-heavy (0,0,1) | 21 | 10 | 10 | 37 | **10** |
| random H-free 3q x 40 | w2-heavy (1,6,0.2) | 84.2 | 60.8 | 34.4 | 34.4 | **34.4** |
| random H-free 3q x 40 | w1-heavy (1,0,0) | 32 | 10 | 8 | 8 | **8** |
| random H-free 3q x 40 | wd-heavy (0,0,1) | 21 | 14 | 7 | 7 | **7** |

What the table shows:

- **Toffoli×2** (its two copies multiply to the identity): Stage 2 managed 30 → 27 gates,
  while Stage 4 reaches **0**. The H between the copies cancels. The middle then becomes
  one H-free region whose phase table is all zeros (CCZ·CCZ = I), and it resynthesizes to nothing.
- **Single Toffoli:** unchanged. Its 7 T gates are known to be optimal for this
  decomposition, so that is the expected result, not a missed optimization.
- **Pipeline B never won outright here.** It tied A on the H-free and
  Toffoli×2 rows, and lost badly on `random 6q` (193 vs 63) because its CX
  ladders cost a lot under w2-heavy weights. It stays as a scored candidate:
  it can never make the result worse, only slower.
- **Weights matter:** `optimize` chooses its output for the weights it is
  given, and different weights produce different output circuits (tested).
  Each output is only ever scored under the weights it was produced for.

On 750 circuits from the Stage 1/2 test generators: Stage 4 was **never**
costlier than Stage 2, strictly cheaper on 321, and idempotent on all 750.

## Verification

- `test_stage4.py` has 40 tests across 4a–4f:
  - every rule identity;
  - representation round-trips (1,000 circuits);
  - `canon` agreeing with the unitary in both directions;
  - the BFS table, which reaches all 20,160 maps for 4 wires (|GL(4,2)|);
  - tableau rules against matrices for every gate and every 3-qubit Pauli;
  - Toffoli×2 → 0 and compute/uncompute → H;
  - random circuits × 6 weight configs: never worse than Stage 2 or the input, valid output, inputs untouched (120 circuits by default; `STAGE4_FULL=1` runs the spec's 500);
  - exact idempotence, plus "unchanged unless strictly cheaper";
  - weight sensitivity;
  - byte-identical output across 3 processes with different `PYTHONHASHSEED`.
- **Mutation testing:** 8 deliberately injected bugs. The tests caught
  6 immediately. The other 2 survived at first, and each exposed a test gap:

  | Mutant | Result |
  |---|---|
  | X-before-T sign flip removed | caught |
  | Pauli CX sign rule removed | caught |
  | Gauss ops not reversed | caught |
  | Rotation merge ignores blockers | caught |
  | Canonical terms: wrong sign | caught |
  | Y-basis undo uses SDG instead of S | caught |
  | Final tie-break by gate tuple | **survived**. I had wrongly credited this tie-break with idempotence; the repeat loop provides that. Its real job is "return the input unchanged unless something is strictly cheaper". New test `test_4f_unchanged_unless_strictly_cheaper` now catches it. |
  | H-delay emits H gates eagerly | **survived**, because 4d region merging hid it. The new direct test also found that *always* delaying H can be worse: a leading H can move into the middle. `_region_order` now picks the order with the fewest regions among {given, H-delayed, H-eager}, never more than the input. Now caught. |

- **Stage 1/2 tests run against both files** (`SUBMISSION_MODULE=submission_stage4`
  switches the module). The tests check the actual goal and nothing else.
  Each weight setting W is passed to `optimize`, and the output must be
  equivalent and no costlier *under W* than the input, and than Stage 1.
  Earlier versions asserted Stage 1–2 side properties instead: no metric ever
  increases, never more gates than Stage 1, "stays 3 gates", and costs under
  weights the output was not optimized for. Those failed by design for
  Stage 4 and were rewritten.

## Limitations

- Exact CX-optimal linear synthesis covers regions of up to 4 wires. Wider
  regions use Gauss–Jordan, which may lose to the original. The original is
  always a candidate, so this never makes a result worse.
- Phase placement during resynthesis is greedy, not GraySynth.
- Regions are found in list order, after choosing the best of three
  wire-order-preserving orderings. That is not the maximal "H-free
  sub-circuit" in the DAG sense.
- Runtime: median about 27 ms per `optimize` call at up to 60 gates (max seen 135 ms).
  The default Stage 4 test file takes about 1.5 minutes.
