# Experiment 3 — 3D Gaussian waveguide

The waveguide problems are large: at ω = 48 the physical grid has 287 × 95 × 95
unknowns. Adding 18 PML cells per face gives 323 × 131 × 131 = 5,543,003 unknowns.  The paper ran them on single nodes of a Slurm cluster (64 cores,
Hygon 7285 at 2.0 GHz) with MATLAB R2024a.  On a workstation, `run_exp3('verify')`
trains and solves the same pipeline at ω = 6 (N = 11,849) in a few minutes.

## Requirements

| ω | N | training (8 concurrent solves × 4 threads) | peak memory | online, 3 queries |
|---:|---:|---:|---:|---:|
| 24 | 680,225 | 0.5 h | 25 GB | 4 min |
| 36 | 2,381,643 | 1.5 h | 46 GB | ~0.5 h |
| 48 | 5,543,003 | 3.3 h | 96 GB | ~1.5 h |
| 60 | 11,026,125 | 31 h (4 workers, 120 GMRES cycles per training solve) | 110 GB | ~9 h |

Training needs the Parallel Computing Toolbox for concurrent solves (it runs serially
without it).  Build the optional MEX kernel of the SL GMRES on the machine that runs it:
`build_mex` (needs a C compiler); without it a MATLAB implementation is used.

## Running

1. Copy `slurm/run_omega.slurm.template` and `slurm/validation.slurm.template`, and fill
   in the account, partition, repository path and MATLAB command.
2. For ω in 24, 36, 48, 60: `sbatch --export=OMEGA=<ω> run_omega.slurm`
   (ω = 60: `--time=36:00:00 --mem=224G`).  Training checkpoints after every iteration
   and every accepted training solve, so a resubmitted job resumes.
3. `sbatch validation.slurm` (ten queries at ω = 48).
4. Appendix A, for every ω: `matlab -batch "startup; run_appA('probe', <ω>)"`
   (ω = 48: about 0.5 h, 20 GB).
5. Copy `results/exp3` back and run `run_exp3('export')` and `run_appA('export')`.

## Settings

All settings are in `exp3_config.m`: eight training parameters μ ∈ [0.2, 0.6], test
queries 0.237, 0.317, 0.483 (and seven added queries at ω = 48), tail tolerances
η = (0.003, 0.008, 0.02), tolerance 10⁻⁶, SL shift β = 2 with a six-level geometric
multigrid V-cycle (Galerkin coarse operators, two weighted-Jacobi sweeps before and
after, coarse grids of at most 12,000 unknowns), restart 40, at most 40 cycles online.
Training solves are SL-MG GMRES solves to a relative residual of 10⁻⁷.
