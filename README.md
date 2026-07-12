# Analog gravity on computational substrates — reproducibility package

Code and reference outputs to reproduce every figure and the experimental-record table of:

> **Analog Gravity on Computational Substrates: Robust Kinematic Horizons, Einstein
> Dynamics Confined to the Holographic Error-Correcting Class, and a Gravitating
> Critical Line**, O. Yilmaz (in submission, *Journal of Physics Communications*; an earlier version is under review at *Entropy*).

The paper's split: effective-metric **kinematics** hold generically on computational
substrates (a transport horizon at the gain-field zero set coincides with the clock-freeze
surface), Einstein **dynamics** fails in four tests across three dissipative substrate families, and is
recovered only on the holographic / quantum-error-correcting class (a $[[5,1,3]]$
perfect-tensor network: Ryu–Takayanagi density $\eta_c=1$, a first-law/FLM analogue). The
broader framework is the Computational Wave Field Theory book (Zenodo DOI
[10.5281/zenodo.20632349](https://doi.org/10.5281/zenodo.20632349)).

## Environment

CPU-only Python (no GPU). Tested on Python 3.10.

```
pip install -r requirements.txt     # numpy, scipy, matplotlib, networkx, stim
```

## How to run

Run any script from the repository root; each writes its figure(s) into the root and
updates its record in `results.json`:

```
python3 cwf_test6b.py            # Test 6 (kinematics: horizon = clock-freeze)
python3 cwf_clausius2d.py        # Clausius/area-law failure (2D reservoir)
python3 cwf_a3b2_happy_perfect.py  # Ryu-Takayanagi eta_c = 1 (perfect tensor)
python3 cwf_a3b4_first_law.py    # first law / linearised Einstein analogue
```

The paper's four figures are committed under `figures/` for reference; running the scripts
regenerates them in the repository root for comparison.

## The paper's four figures

| Figure | Script | `results.json` key |
|---|---|---|
| Test 6 — transport horizon = clock-freeze | `cwf_test6b.py` | `Test6` |
| Clausius/area-law failure (2D) | `cwf_clausius2d.py` | `ClausiusTest2D` |
| Ryu–Takayanagi $\eta_c=1$ (HaPPY $[[5,1,3]]$) | `cwf_a3b2_happy_perfect.py` | `A3b2_happy_perfect` |
| First law / FLM analogue | `cwf_a3b4_first_law.py` | `A3b4_first_law` |

## Full experimental-record map

| Paper result | Script | `results.json` key |
|---|---|---|
| Test 6 (kinematics) | `cwf_test6b.py` | `Test6` |
| Test 5 (reconstruction threshold) | `cwf_test5.py` | `Test5` |
| Test 3 (Page curve, diagnostic) | `cwf_pagecurve.py` | `Test3_PageCurve` |
| Clausius 2D (area-law fails) | `cwf_clausius2d.py` | `ClausiusTest2D` |
| A1 hysteresis (irreversible) | `cwf_a1_hysteresis.py` | `A1_hysteresis` |
| A2 phase 1 (fast scrambling) | `cwf_a2_phase1_scrambling.py` | `A2_phase1_powerlaw_scrambling` |
| A2 phase 2 (non-local Clausius) | `cwf_a2_phase2_clausius_nonlocal.py` | `A2_phase2_clausius_nonlocal` |
| A3 (Clifford lattice) | `cwf_a3_clifford_horizon.py` | `A3_clifford_horizon` |
| A3a phase 1 (Hayden–Preskill) | `cwf_hp_phase1_decoding.py` | `A3a_phase1_hp_decoding` |
| A3a phase 2 (Page curves) | `cwf_hp_phase2_pagecurve.py` | `A3a_phase2_pagecurves` |
| A3a phase 3b (tree-doubling code) | `cwf_hp_phase3b_happy.py` | `A3a_phase3b_happy_tree` |
| A3a phase 4 (scrambling time) | `cwf_hp_phase4_scrambling.py` | `A3a_phase4_scrambling_time` |
| A3a phase 5 (comparison) | `cwf_hp_phase5_compare.py` | — |
| A3b (emergent-geometry RT form) | `cwf_a3b_emergent_geometry.py` | `A3b_emergent_geometry` |
| A3b-quant (perfect tensor, $\eta_c=1$) | `cwf_a3b2_happy_perfect.py` | `A3b2_happy_perfect` |
| $\Lambda_c$ probe | `cwf_a3b3_cosmological.py` | `A3b3_cosmological` |
| A3b-dyn (first law) | `cwf_a3b4_first_law.py` | `A3b4_first_law` |

Shared modules (imported, not run directly): `cwf_substrate.py` (reservoir / Rule-110
base), `cwf_experiments.py`, `cwf_hp_lib.py` (stabilizer / HaPPY helpers), `cwf_pagecurve.py`.

## Reproducibility note

Experiments are seeded. The **stabilizer and perfect-tensor results** (Page curves,
Hayden–Preskill, $\eta_c$, the first law) are exact and **bit-reproducible**. The
**reservoir gain-field tests** carry last-digit nondeterminism from threaded LAPACK
eigenvalue routines, which can shift a discrete horizon site by $\pm1$ — e.g. Test 6's
clock-freeze/transport-horizon separation is **2–3 sites across runs** (always far inside
the well half-width of 18). This does not affect any conclusion; pin `OMP_NUM_THREADS=1`
for bit-identical reservoir output if desired. `results.json` holds the reference values.

## License

Released under the MIT License (see `LICENSE`).

## Citation

Please cite the paper (above) and the Computational Wave Field Theory book
(Zenodo DOI 10.5281/zenodo.20632349).
