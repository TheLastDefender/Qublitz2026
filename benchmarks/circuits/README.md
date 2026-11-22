# Standard benchmark circuits

Copied unchanged from the `circuits/` folder of the PyZX repository
(https://github.com/zxcalc/pyzx, commit 48954857eac13eea3ae8bcf879e41434756d3686,
Apache License 2.0). They are used only by `benchmarks/compare_standard.py`,
never by the submission.

**Selection.** `python benchmarks/compare_standard.py --scan <pyzx>/circuits`
lists every circuit in that folder that fits the challenge (at most 6 qubits,
exactly expressible in H, X, Z, S, SDG, T, TDG, CX). Apart from trivial 2-gate
test circuits and alternative spellings, exactly four distinct benchmark
inputs fit: `tof_3`, `barenco_tof_3`, `mod5_4` and `qft_4`. All other standard
benchmarks need more than 6 qubits.

| File | What it is | Source in the PyZX repository |
|---|---|---|
| `tof_3_before`, `barenco_tof_3_before`, `mod5_4_before` | benchmark inputs (the set used by Nam, Ross, Su, Childs & Maslov 2018) | `Arithmetic_and_Toffoli/` |
| `qft_4.qc` | benchmark input (Clifford+T quantum Fourier transform on 4 qubits plus one ancilla) | `Fast/` |
| `*_after_light`, `*_after_heavy` | Nam et al.'s published optimized circuits (light and heavy settings) | `Arithmetic_and_Toffoli/` |
| `*_tpar.qc` | T-par optimized circuits (Amy, Maslov & Mosca 2014) | `Arithmetic_and_Toffoli/` |
| `*_feynman.qc` | Feynman optimizer output (renamed from `feyn_bench/out/<name>.qc`) | `feyn_bench/out/` |
| `*_pyzx.qc` | PyZX's stored optimized circuits | `optimized/` |

References: Y. Nam et al., *npj Quantum Information* 4, 23 (2018),
doi:10.1038/s41534-018-0072-4; M. Amy, D. Maslov, M. Mosca, *IEEE Trans. CAD*
33(10), 1476–1489 (2014), doi:10.1109/TCAD.2014.2341953.
