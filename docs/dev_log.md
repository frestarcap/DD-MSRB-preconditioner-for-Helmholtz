# Development log

Construction of this repository from the authors' research repository (private,
commit 097edc8, 2026-10-24).  The private repository is never modified; old code
runs only from an exported copy.

## 2026-10-24 — P0/P1: paper data and figures (checkpoint C1)

- `paper_data/` (26 files, 9.9 MB) exported from the frozen run outputs with a
  private helper; only plotted or tabulated quantities, copied without rounding.
  No paths, machine names or job identifiers.
- `figures/`: `style.py`, `exp1.py`, `exp2.py`, `exp3.py`, `appendixA.py`, `make_all.py`
  port the published plotting code; `waveguide_geometry.m` (MATLAB) draws the two
  isosurface renderings, which cannot be reproduced exactly with matplotlib.
- Check: all 27 figures pixel-identical to the published files (pdftoppm, 100 dpi),
  all six table bodies row-identical to the manuscript (Table 3 was hand-copied in
  the manuscript; it now has a generator).  `tests/python/test_paper_figures.py`
  repeats the check against `tests/reference/`.

## 2026-10-25 — P2: half-space core and equivalence tests

- New core in `src/+dmsrb` (compress, greedy_ranks, project/lift/galerkin, arnoldi_step,
  verified_snapshot, apply_rom, prefix, train, train_nt, solution_pod, solve, sl.gmres, sl.ilu)
  and `src/+dmsrb/+halfspace` (problem, operator, sl_mass, partition, family).
- L0/L1: matrices, source and the three refined partitions are bitwise equal to commit 097edc8.
- L3a: the refactored trainer reproduces the original trainer of 097edc8 bitwise
  (snapshots of all three iterations, global and local ranks, reduced operators).
- L3b: against the frozen paper models (trained 2026-09-05) the ranks are identical:
  global (7,11,19), DD (44,80,123) with all ten local ranks.  The snapshots of iterations
  2 and 3 differ by 1.7e-9 / 7.3e-5 (global) and 1.6e-10 / 5.0e-6 (DD): that training used
  one-pass Gram--Schmidt and no residual check of the recurrence; the refactored code
  follows Algorithm 1 of the paper (two-pass, verified recurrence).
- L4: the frozen paper models solved by the refactored online code reproduce the stored
  ROM fields (<1e-9) and the published SL step counts of all 23 queries and five methods.
- L5: SS-ROM rebuilt from scratch spans the paper space (7 modes).  NT-ROM agrees in
  iterations 1-2; its iteration-3 space keeps roundoff-level modes (sigma_19 ~ 2e-18)
  that are not reproducible, yet all 23 ROM fields agree to 1e-9.

## 2026-10-25 — P2 (scattering) and P3: experiment drivers and full re-runs (checkpoint C3)

- `src/+dmsrb/+ecc` (geometry, ghost-point assembly, admissibility, near/far/PML partition).
  The 22 test systems are bitwise equal to the stored ones; the trainer reproduces the
  original trainer of 097edc8 bitwise (one thread); against the frozen 2026-09-20 models
  the ranks agree, snapshots of iterations 2-3 to 1.7e-7 / 3.6e-4 (the original code
  itself does not reproduce them more closely today); all SL step counts agree.
- `experiments/exp1_halfspace/run_exp1.m`, `experiments/exp2_ecc/run_exp2.m`,
  `dmsrb.benchmark`, `dmsrb.storage`, `dmsrb.deploy`; `tools/compare_to_paper.py`.
- Full re-runs on an Apple M4 Max, one thread (Exp. 1: 17 models, about 1.5 h in all;
  Exp. 2: about 6 min).  `reference_outputs/comparison_report.md`: 57 exact and 21
  tolerance quantities agree; 6 marginal flips (SS-ROM: five queries one SL step apart,
  the last SL residual lies within 10% of 1e-12 in both runs); timing ratios differ by
  -22% to +93% (the SL baseline time varies most between sessions; Table 1's SL
  baseline of the paper was 0.533 s, the re-run 0.418 s, while ROM, NT, SS and direct
  medians agree within 1-2%).
- Table 2 with ILU (as stated in the paper): online means about twice the published ones,
  which used exact LU.

## 2026-10-25 — P4/P5: waveguide, Appendix A, documentation (checkpoints C4, C5)

- `src/+dmsrb/+waveguide` (problem, operator, mass, partition, reduce), `dmsrb.sl.mg`,
  `src/mex` (optional CGS2 kernel), `experiments/exp3_waveguide` (exp3_config,
  train_waveguide with checkpoints and concurrent solves, run_exp3, Slurm templates),
  `experiments/appendixA` (run_appA, mg_response, transverse_roughness).
- L6 (`test_waveguide`): at omega = 6 and 12 the whole pipeline agrees with the cluster
  code (partition, source, global and local ranks of all three iterations, reduced
  operators to 1e-9, ROM residuals to 1e-6 relative, SL step counts of 3 queries x 3
  methods).  `test_appendixA`: MG component contractions and roughness agree to 1e-10.
- `run_exp3('verify')` runs end to end at omega = 6 in 15 s.  The MEX kernel was not
  built here (Xcode license not accepted on this machine); the MATLAB path is used.
- Documentation: README, docs/REPRODUCING.md, docs/METHOD_NOTES.md,
  docs/CHANGES_FROM_PAPER_RUNS.md, experiments/exp3_waveguide/README.md,
  tests/equivalence/README.md, LICENSE (MIT), CITATION.cff.

## 2026-10-25 — release preparation at the author's request

- Repository named "DD-MSRB preconditioner for Helmholtz".  The original research code
  (commit 097edc8: `code/`, `scripts/`) and the cluster source bundle v7 are in `legacy/`;
  only account-specific strings (cluster user paths, partition, SSH alias, Overleaf URLs)
  were replaced.  `legacy/README.md` gives the original-code commands for Experiment 3 and
  Appendix A.
- The manuscript now takes Table 1, Figure 2 (right), Figure 3 (left) and all 2D offline
  times from the re-run with this code; `paper_data` and `tests/reference` follow.
- New `offline` stages in `run_exp1` and `run_exp2`: complete trainings (DD partition
  included), median of three after one untimed warm-up, one thread.  Table 2 offline
  0.74/0.76, 2.12/1.54, 4.28/2.96 s; Experiment 2 global/DD 7.60/7.57 s.
  A second quiet re-run of Table 1 reproduced the per-query pattern (the five queries with
  c_right < 0.83 have a slower SL baseline in both runs) and gave speedups within 4%.
- The comparison report compares the re-run with the values of the original runs.
