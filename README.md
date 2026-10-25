# DD-MSRB preconditioner for Helmholtz

MATLAB code and data for the paper

> An Ping, Zhichao Peng, Pengliang Yang.
> *A Trajectory-Aware Domain-Decomposed Multispace Reduced Basis Preconditioner for
> Parameterized Helmholtz Problems* (submitted to the Journal of Computational Physics).

For a parameterized Helmholtz--PML system A(μ)u = b(μ), the method trains one reduced
space per FGMRES iteration along the iteration trajectory that the compressed
preconditioners themselves generate (trajectory-aware training).  Online, the first k
FGMRES steps are preconditioned by the reduced inverse actions
z_j = Ψ_j (Ψ_jᴴ A(μ) Ψ_j)⁻¹ Ψ_jᴴ q_j; if the true residual is still above the
tolerance, FGMRES restarts from the ROM iterate with a shifted-Laplacian (SL)
preconditioner.  In the domain-decomposed variant (DD-MSRB) every subdomain, including
the PML subdomains, has its own local basis, with local ranks chosen greedily under one
global snapshot-error budget; the online reduced system keeps all subdomain couplings.

## Original and rewritten code

The first author continues to use the original research code in other ongoing work, so
**Experiment 3 and Appendix A of the paper were computed with the original code**, which
is included unchanged in [`legacy/`](legacy/README.md).  For readability, this repository
also provides a rewritten implementation of the method.

* **Original code** (`legacy/`): harder to read; the source of the Experiment 3 and
  Appendix A data, and of the original runs of Experiments 1 and 2.
* **Rewritten code** (`src/`, `experiments/`): easier to read; one MATLAB package and one
  driver per experiment.  Table 1, Figure 2 (right), Figure 3 (left) and the 2D offline
  times of the paper (Table 2, Experiment 2) come from its re-run; for all other quantities of
  Experiments 1 and 2 it reproduces the ranks, storage and SL step counts of the original
  runs (`reference_outputs/comparison_report.md`).  Its Experiment 3 pipeline agrees with
  the cluster code on reduced grids but was not run at full scale.

## Reproduction commands

| Experiment | Rewritten code (MATLAB, after `startup` in the repository root) | Original code |
|---|---|---|
| Figures and tables from the published numbers | `python figures/make_all.py` (and `waveguide_geometry` in MATLAB for two 3D renderings) | — |
| 1 — half-space (≈1.5 h, one thread) | `run_exp1('all')` | `legacy/research_code` (as in the original study) |
| 2 — elliptic-obstacle scattering (≈6 min) | `run_exp2('all')` | `legacy/research_code` |
| 3 — 3D waveguide (cluster) | `run_exp3('train', w)`, `run_exp3('online', w)`, w = 24, 36, 48, 60; `run_exp3('validation', 48)`; `run_exp3('export')` | `sbatch scripts/scnet_3d.slurm full w 8 4 waveguide profiles/beta2_coarse12000_restart40.json`, … — see [`legacy/README.md`](legacy/README.md) |
| Appendix A (cluster) | `run_appA('probe', w)`, then `run_appA('export')` | `scripts/scnet_dd_mechanism_fast.slurm`, … — see `legacy/README.md` |

After a re-run with the rewritten code:

```bash
python figures/make_all.py --data results/paper --out build/rerun   # the paper figures from your run
python tools/compare_to_paper.py --paper <paper values>             # comparison report
```

`docs/REPRODUCING.md` lists, for every table and figure, the stage that computes it, the
data file it is drawn from and the expected run time.  Experiment 3 needs a node with
64 cores and 192 GB (224 GB at ω = 60); `experiments/exp3_waveguide/README.md` has the
Slurm templates, and `run_exp3('verify')` runs the same pipeline at ω = 6 on a workstation.
The differences between the rewritten code and the runs of the original code are listed
in `docs/CHANGES_FROM_PAPER_RUNS.md`.

## Layout

```
src/+dmsrb/              method: training (train, train_nt, solution_pod), compression
                         (compress, greedy_ranks), reduced spaces (project, lift, galerkin),
                         online solve (prefix, apply_rom, solve), storage, timing (benchmark)
src/+dmsrb/+sl/          shifted-Laplacian preconditioners (ILU, geometric multigrid) and GMRES
src/+dmsrb/+fd/          PML profile and 2D/3D finite-difference operators
src/+dmsrb/+halfspace/   Experiment 1 problem and partitions
src/+dmsrb/+ecc/         Experiment 2 problem (ghost-point elliptic obstacle) and partition
src/+dmsrb/+waveguide/   Experiment 3 problem, partition and reduced assembly
src/mex/                 optional BLAS kernel of the 3D GMRES (build_mex)
experiments/             one driver per experiment: run_exp1, run_exp2, run_exp3, run_appA
figures/                 Python figures and tables (make_all.py); waveguide_geometry.m
paper_data/              the numbers behind every figure and table of the paper (10 MB)
reference_outputs/       our re-run of Experiments 1-2 with the rewritten code, and its comparison
legacy/                  the original research code (Experiment 3, Appendix A)
tests/                   unit, smoke, equivalence (against the original code) and figure tests
docs/                    reproduction guide, method notes, differences to the original runs
```

## Tests

```bash
matlab -batch "startup; runtests('tests/unit'); runtests('tests/smoke')"
python -m pytest tests/python          # figures and tables reproduce the published ones
```

The equivalence tests in `tests/equivalence` compare the rewritten code with the original
code and the frozen paper models; they are skipped unless those are available
(see `tests/equivalence/README.md`).

## License

MIT, see [LICENSE](LICENSE).  If you use this code, please cite the paper above
([CITATION.cff](CITATION.cff)).
