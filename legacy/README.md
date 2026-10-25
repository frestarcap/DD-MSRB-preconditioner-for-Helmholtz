# Original research code

The first author continues to use the original research code in other ongoing work.
For this reason, **Experiment 3 (the 3D Gaussian waveguide) and Appendix A of the paper
were computed with this original code**, and it is kept here unchanged.  For readability,
the repository root contains a rewritten implementation of the same method
(`src/`, `experiments/`).

| | Original code (this folder) | Rewritten code (repository root) |
|---|---|---|
| Readability | harder to read: research history, many study variants and options | easier to read: one package, one driver per experiment |
| Paper data | source of Experiment 3 and Appendix A; source of the original runs of Experiments 1–2 | source of Table 1, Figure 2 (right), Figure 3 (left) and the 2D offline times (Table 2, Experiment 2); re-runs Experiments 1–2 with the same ranks, storage and SL steps (`reference_outputs/comparison_report.md`) |
| Experiment 3 | as run on the cluster | the same pipeline, checked against `cluster_v7/` on reduced grids (`tests/equivalence/test_waveguide.m`), not run at full scale |

Contents:

* `research_code/code`, `research_code/scripts`: the complete research code (MATLAB,
  Python post-processing, Slurm scripts) at the state of the submitted manuscript,
  including studies that are not in the paper (photonic crystal, lens, Marmousi,
  WaveHoltz, …).  `scripts/analyze_components.py` was kept next to the Appendix A results
  and is added here.  `profiles/` holds the solver profile of the 3D runs.
* `cluster_v7`: the frozen source bundle that ran the 3D training and online jobs on the
  cluster (2026-09-09); `source_manifest.json` lists its file hashes.  Its files agree
  with those of `research_code` except `rom_reduced_space_utils.m` (later, numerically
  equivalent changes) and `plot_helmholtz_3d_paper_figures.m`.

Only account-specific strings were replaced in the copies (user name and paths on the
cluster, partition name, SSH alias, Overleaf project URLs); set `MATLAB_ENV_SCRIPT`,
`MATLAB_EXE`, `REMOTE_RUN_ROOT` and the `#SBATCH --partition` line before submitting.
Numerical code is unchanged.

## Reproducing Experiment 3 and Appendix A with the original code

The original code writes to `outputs/` next to `code/` (see `code/helm_output_path.m`).
Run from `research_code/` (or, for steps 1–2, from `cluster_v7/`), on nodes with 64 cores
and 192 GB (224 GB for ω = 60), MATLAB R2024a.

1. **Training and online solves, ω = 24, 36, 48** (paper jobs 43868638, 43864386, 43864387):
   ```bash
   mkdir -p logs
   sbatch scripts/scnet_3d.slurm full 24 8 4 waveguide profiles/beta2_coarse12000_restart40.json
   sbatch scripts/scnet_3d.slurm full 36 8 4 waveguide profiles/beta2_coarse12000_restart40.json
   sbatch scripts/scnet_3d.slurm full 48 8 4 waveguide profiles/beta2_coarse12000_restart40.json
   ```
   Each run writes `report.json`, the trained models (`*_training.mat`) and the results
   (`helmholtz_3d_candidates_*.mat`) under `outputs/scnet_3d_20260909/`.
2. **ω = 60** (four workers, 120 GMRES cycles per training solve; paper job 44057113, which
   reused seven first-iteration snapshots of the interrupted job 43935834 from the
   snapshot cache):
   ```bash
   sbatch scripts/scnet_w60_recovery.slurm     # settings in scripts/scnet_w60_recovery.json
   ```
3. **Ten-query validation at ω = 48** (Table and Figure of Section 5.3.1): create
   `sources.tsv` with one line `48<TAB><training MAT of step 1><TAB><report.json of step 1>`,
   copy the compiled kernel to `kernel/`, then
   ```bash
   sbatch --array=1-10 scripts/scnet_jcp_revision_validation.slurm production
   python scripts/analyze_jcp_revision_3d.py      # writes the cohort summary (summary.json)
   python scripts/write_jcp_3d_validation.py      # table rows
   ```
4. **MATLAB object sizes** (storage panel): `sources.tsv` with one line per ω, then
   `sbatch scripts/scnet_waveguide_whos.slurm` and `python scripts/summarize_waveguide_whos.py`.
5. **Appendix A** for every ω: residual composition and multigrid response, then the
   interface audit and the central-plane slices:
   ```bash
   sbatch scripts/scnet_dd_mechanism_fast.slurm <training MAT> <results MAT> results/fast_w<ω> full
   sbatch scripts/scnet_dd_interfaces_flexible.slurm interfaces_fast_manifest.tsv
   python analyze_components.py                   # copy it next to results/; writes final_layer_components.csv
   python scripts/analyze_dd_interface_evidence.py
   ```
6. **Tables and figures**: `python scripts/plot_waveguide_frequency_paper.py`,
   `python scripts/compute_waveguide_break_even.py` (frequency table, break-even counts),
   `python scripts/plot_jcp_publication_layout.py` (all layout figures).

The numbers these steps produced are in `paper_data/exp3` and `paper_data/appendixA`;
`python figures/make_all.py` draws the published figures and tables from them.

## Reproducing every experiment with the rewritten code

From the repository root, in MATLAB after `startup`:

| Experiment | Command |
|---|---|
| 1 — half-space | `run_exp1('all')` |
| 2 — elliptic-obstacle scattering | `run_exp2('all')` |
| 3 — 3D waveguide (cluster) | `run_exp3('train', w)`, `run_exp3('online', w)` for w = 24, 36, 48, 60; `run_exp3('validation', 48)`; `run_exp3('export')` (Slurm templates in `experiments/exp3_waveguide/slurm`) |
| Appendix A (cluster) | `run_appA('probe', w)` for each w, then `run_appA('export')` |
| Figures and tables | `python figures/make_all.py --data results/paper` |

See `docs/REPRODUCING.md` for run times and the data file behind each table and figure.
