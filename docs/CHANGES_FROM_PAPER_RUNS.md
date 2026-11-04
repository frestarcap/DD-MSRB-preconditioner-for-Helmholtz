# Differences between this code and the runs reported in the paper

The rewritten code implements the method as stated in the paper; where the runs of the
original code (`legacy/`) differed from that statement, or did extra work, the difference
is listed here with its effect.  `reference_outputs/comparison_report.md` compares a
complete re-run of Experiments 1 and 2 with the numbers of the original runs.

**Paper numbers from the rewritten code.**  Table 1, Figure 3 (left)
and all 2D offline times of the paper (the Table 2 offline column and the Experiment 2
offline times, 7.57 s DD and 7.60 s global) come from the re-run with this code.  The
earlier manuscript version reported a slower SL baseline in Table 1 (median 0.5328 s
instead of 0.4177 s, hence speedups 75.5/67.7 instead of 59.7/53.6) and offline times that
included unused computations (below): Table 2 1.14/1.36, 3.28/2.74, 6.15/5.13 s instead of
0.74/0.76, 2.12/1.54, 4.28/2.96 s, Experiment 2 11.58/14.46 s.  Offline times are the
median of three complete trainings after an untimed warm-up (stage `offline`; the three
runs differ by at most 1.5%).  A second re-run of Table 1 on the same machine gave
speedups 57.2 and 51.4.

**Figure synchronization (2026-11-04).** The solution panels now use the physical
domain and the current manuscript colormap. Figure 2 (right) follows the
2026-10-22 original-code residual histories used by the current manuscript
renderer; the rewritten-code histories remain in `reference_outputs/exp1`.
Its legend is lowered to avoid covering the plain GMRES curve.

## Algorithms

1. **Snapshot recurrence (Algorithm 1).**  The code verifies every recurrence snapshot
   (‖A z − q‖ ≤ 10⁻¹⁰, otherwise a direct solve) and orthogonalizes with two passes, as
   stated in the paper.  The Experiment 1 models of the paper (trained before the check
   was added) used one Gram–Schmidt pass without the check.  The ranks are identical —
   global (7, 11, 19), DD (44, 80, 123) and all ten local ranks — while the snapshots of
   iterations 2 and 3 differ by 1.7·10⁻⁹ and 7.3·10⁻⁵ (global), 1.6·10⁻¹⁰ and
   5.0·10⁻⁶ (DD) in relative Frobenius norm.  For the same training data the rewrite
   reproduces the original trainer bitwise (`tests/equivalence`).
2. **Partition and frequency studies of Experiment 1.**  The paper trained these models
   with a direct solve A\q for every snapshot; the code uses the verified recurrence
   throughout.  All ranks and payloads agree (merged (17, 29, 44), signed/depth
   (49, 87, 132), uniform (72, 137, 201); ω = 80.25, 107, 214).
3. **Roundoff sensitivity of Experiment 2.**  The reduced systems of the scattering
   problem are ill-conditioned.  The original code, re-run today, reproduces its own
   2026-09-20 snapshots of iterations 2 and 3 only to 1.7·10⁻⁷ and 3.6·10⁻⁴; the
   rewrite agrees with today's original code bitwise.  Ranks (25, 42, 42) and
   (41, 71, 75) and all SL step counts are unaffected.
4. **NT-ROM, iteration 3.**  The equal-rank control keeps 19 modes although its
   snapshot spectrum reaches roundoff (σ₁₉ ≈ 2·10⁻¹⁸); these directions are not
   reproducible, but the ROM iterates agree to 10⁻⁹ for all 23 queries.
5. **Early stop of the ROM steps.**  The online ROM steps stop when the true residual
   reaches the tolerance τ (Algorithm 2); the Experiment 1 runs tracked 10⁻¹⁴.  No query
   reaches τ before the last ROM step, so nothing changes.  An empty reduced space (a
   converged training trajectory, e.g. n_s = 6) ends the ROM steps.
6. **3D rank allocation.**  The greedy allocation sums the remaining tails directly; the
   cluster runs subtracted retained energy from the total.  Both give the same ranks
   (checked on reduced grids, `tests/equivalence/test_waveguide.m`).

## Work that is no longer done (offline times are lower)

The research solvers also computed, inside the timed offline stage, quantities that the
paper does not use: a time-domain (WaveHoltz) lift of every snapshot, a dense POD of
every DD layer, a "direct" POD of all reference solutions, re-solves of every training
sample after each layer, and, online, a 60-step unpreconditioned GMRES per query.  All
WaveHoltz code is removed.  Offline times of the re-run are therefore lower; the paper
now reports the 2D offline times of the rewritten code.

## Settings

7. **Table 2 time column.**  The training-coverage timings of the paper use an exact LU
   factorization of the shifted operator (the default of the research solver); the paper
   states this.  The rewritten code uses ILU everywhere, so its Table 2 times are about
   twice as long.
8. **One timing protocol.**  All 2D timings use one untimed warm-up and five repetitions
   with the method order rotated per query and repetition.  Table 3 of the paper timed
   each partition variant separately, and the baselines of Figure 5 were timed in a
   separate run; the Experiment 2 rotation also contained two unpublished models.
9. **MATLAB object size.**  The deployed model is a plain struct (`dmsrb.deploy`); the
   `whos` size differs slightly from the research layout (see the comparison report).

Experiment 3 and Appendix A were not re-run with the rewritten code; their numbers come
from the original code in `legacy/`.
