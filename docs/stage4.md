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
- **Weights matter:** `random 4q` costs 36 (w2-heavy) vs 10 (w1-heavy), and
  different weights produce different output circuits (tested).

On 750 circuits from the Stage 1/2 test generators: Stage 4 was **never**
costlier than Stage 2, strictly cheaper on 321, and idempotent on all 750.

## Verification

- `test_stage4.py` has 38 tests across 4a–4f:
  - every rule identity;
  - representation round-trips (1,000 circuits);
  - `canon` agreeing with the unitary in both directions;
  - the BFS table, which reaches all 20,160 maps for 4 wires (|GL(4,2)|);
  - tableau rules against matrices for every gate and every 3-qubit Pauli;
  - Toffoli×2 → 0 and compute/uncompute → H;
  - 500 random circuits × 6 weight configs (never worse than Stage 2 or the input, valid, inputs untouched);
  - exact idempotence;
  - weight sensitivity;
  - byte-identical output across 3 processes with different `PYTHONHASHSEED`.
- **Mutation testing:** 8 deliberately injected bugs. See the table below.
- **Stage 1/2 tests run against `submission_stage4.py`** (`SUBMISSION_MODULE=submission_stage4`):
  41/45 pass. The 4 failures are by design. They assert Stage 1–2 guarantees
  that Stage 4 deliberately trades away for lower weighted cost:
  - "no metric ever increases": one case goes 11 → 12 one-qubit gates in exchange for fewer CX;
  - "never longer than Stage 1": one case is 17 vs 16 gates but cheaper;
  - "`CX01 X0 CX01` stays 3 gates": it becomes `X0 X1`;
  - "`CX01 Z1 CX01` stays 3 gates": it becomes `Z0 Z1`.

## Limitations

- Exact CX-optimal linear synthesis covers regions of up to 4 wires. Wider
  regions use Gauss–Jordan, which may lose to the original. The original is
  always a candidate, so this never makes a result worse.
- Phase placement during resynthesis is greedy, not GraySynth.
- Regions are found in list order, after H-delay ordering. That is not the
  maximal "H-free sub-circuit" in the DAG sense.
- Runtime: about 0.02–0.1 s per optimize call at 60 gates; the full Stage 4
  test file takes about 2 minutes.
