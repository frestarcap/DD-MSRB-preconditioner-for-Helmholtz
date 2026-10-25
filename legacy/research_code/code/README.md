# Active Correction ROM Code

This directory keeps the active Helm/WaveHoltz correction-ROM workflows.
Legacy standalone experiments and older variants have been moved to
`archive_legacy/`.

## Current JSC major revision (2026-09-07)

See [`md/jsc_major_revision_reproduction.md`](../md/jsc_major_revision_reproduction.md)
for the authoritative data map and separate A/B manuscript routes. New main
PC uses six blocks (15.1% basis / 11.8% payload savings) and true Helmholtz
acceptance; new main lens uses both tails `1e-4` (21.9% / 16.2%, zero completion
at the two development parameters). The historical 7.4% PC and 46.4% lens
results below remain provenance, not the current primary configurations.

`run_jsc_halfspace_study` adds fixed solution-POD initialization, equal-target
completion, training counts 6/12/24, and a storage--time sweep. Fixed POD is a
strong competitor and is included in the manuscript. `run_jsc_lens_tradeoff`
and `run_jsc_pc_study` retain unfavorable configurations as evidence.
`scripts/reproduce_jsc_major_revision.sh` separates training, serial repeated
timing, and final export/check phases. No files in `archive_legacy/` were edited.

## Main entry points

- `run_marmousi_rom_experiment.m`
  - Research-only Helm feasibility study using the Devito Marmousi gridded
    velocity model, frequency reuse, and held-out pointwise/smooth velocity
    perturbations. Compares shifted-ILUTP, sparse direct, Global TA, and DD TA
    with the same true residual target and complete online timing.
  - `scripts/reproduce_marmousi.sh` downloads a checksum-pinned input;
    `verify_marmousi_rom_experiment` checks saved fields and reduced actions.
    This tests heterogeneous isotropic acoustic media, not VTI/TTI physics.
    See `agent_runs/2026-09-20_marmousi.md` for configuration and provenance.

- `run_ecc_shifted_tuning.m` and `benchmark_ecc_shifted_tuning.m`
  - Research-only Experiment 2 study of training coverage, layer depth, and
    truncation under the unchanged shifted-ILUTP baseline and `1e-10` true
    residual target. Includes the original cohort and 22 fresh geometries.
  - The selected configuration has 96 training points, three layers, and
    `[1e-6,1e-6,1e-6]` tails. `scripts/reproduce_ecc_shifted_tuning.sh` separates
    training, screening, repeated timing, production verification, and reporting.
  - Saved pilot models are frozen caches. Use a new output directory after
    changing the numerical model or training procedure. See
    [`md/ecc_shifted_tuning_20260919.md`](../md/ecc_shifted_tuning_20260919.md).

- `helm_waveholtz_correction_rom.m`
  - Main half-space Helm/WaveHoltz correction-ROM implementation.
  - External local dependencies:
    - `rom_reduced_space_utils.m`
    - `pml_stiffness_2d.m`
    - `prepare_wave_pml_2d.m`
    - `build_shifted_laplacian_preconditioner.m`
    - `helm_output_path.m`
    - `helm_project_root.m`

- `waveholtz_2d_pml_scattering_ecc_correction_rom.m`
  - Main eccentric/scattering correction-ROM implementation.
  - External local dependencies:
    - `rom_reduced_space_utils.m`
    - `pml_stiffness_2d.m`
    - `prepare_wave_pml_2d.m`
    - `build_shifted_laplacian_preconditioner.m`
    - `helm_output_path.m`
    - `helm_project_root.m`

- `photonic_crystal_waveguide_correction_rom.m`
  - Photonic-crystal wrapper/configuration for the shared Helm correction-ROM
    implementation.
  - External local dependencies:
    - `helm_waveholtz_correction_rom.m`
    - `rom_reduced_space_utils.m`
    - `helm_output_path.m`
    - `helm_project_root.m`

## Convenience wrappers

- `waveholtz_correction_rom.m`
  - Dispatches by `cfg.problem_type` to Helm, ECC/scattering, or PC.

- `run_helm_trajectory_evidence_experiment.m`
  - Reproduces the Helm half-space trajectory evidence outputs.
  - Its DD configuration now uses the accepted ten-block partition. For the
    adopted timing batch and current paper figures, use the entry below.

- `prepare_halfspace_dd_paper.m`
  - Current Experiment 1 entry: two physical blocks plus eight material/depth
    PML groups, DOF-weighted allocation, 28.6% basis / 23.4% ROM payload saving.
  - `assemble` reuses matching saved Global/DD and tolerance-diagnostic runs,
    generates missing runs, and adds an equal-rank NR/plain-baseline ablation.
    It preserves the accepted main timing batch (0.707 / 0.744 s) separately
    from the auxiliary baseline batch; no cross-batch speedup is inferred.
  - `redraw` reloads only `outputs/halfspace_dd_paper/paper_data.mat`, writes
    all six current half-space figures, and copies them into the LaTeX project.
    No old three-block evidence or old threshold sweep is read.
  - CSV tables, full-field solutions, spectra, residual histories, partition
    masks, configurations, and provenance are saved with the paper data.
    Full basis/snapshot/operator arrays remain in the referenced refinement
    MAT files. Use `halfspace_paper_dd_config` for the accepted partition.
  - `bash scripts/reproduce_halfspace_dd_paper.sh assemble` reproduces the
    result; use `redraw` for figures only and `MATLAB_BIN` to select MATLAB.

- `run_helm_direct_shifted_laplacian_scaling_pilot.m`
  - Runs a fixed-points-per-wavelength half-space decision pilot comparing
    sparse direct factorization with LU- or ILUTP-based shifted-Laplacian
    GMRES.

- `run_helm_ta_shifted_scaling_experiment.m`
  - Runs the fixed-PPW direct/plain-shifted/TA-ROM-plus-shifted comparison.
  - Reports setup, offline, online, factorization, solve, iteration, rank, and
    basis-storage costs separately.

- `run_helmholtz_3d_candidate_experiments.m`
  - Runs the Gaussian-lens and long Gaussian-waveguide 3D candidates.
  - Compares direct solution, shifted-Laplacian geometric-multigrid GMRES,
    and global/DD trajectory-aware ROM pre-correction with the same shifted
    completion.
  - The default physics-aware DD partition separates the material-active and
    physical far-field regions and splits the PML by face normal (`pml_x`,
    `pml_y`, and `pml_z`). Set `dd_partition_mode='active_far_pml'` to recover
    the original three-block partition.
  - Research partition modes additionally support signed PML faces, inner/outer
    PML depth, and face/edge/corner topology. Set
    `dd_partition_mode='signed_depth_pml'` and adjust
    `pml_depth_fraction` to use the strongest tested PML refinement.
  - Set `physical_partition_mode='profile_bands'` to split the physical domain
    by affine material-profile levels. `profile_partition_thresholds` defines
    the levels and `physical_axial_sections` optionally subdivides each band
    along the configured propagation axis.
  - The default `marginal_storage` rank allocator greedily selects the next
    local singular vector by discarded-energy reduction per stored entry while
    enforcing one global snapshot-tail tolerance. `dof_weighted` and the exact
    diagnostic `minimum_storage` selector remain available.

- `helmholtz_3d_40pct_config.m` and
  `run_helmholtz_3d_40pct_reproduction.m`
  - Freeze the lens and waveguide configurations that produced 46.4% and
    41.0% DD basis-storage reductions, respectively.
  - Train the global and DD representations together and check both storage
    reduction and final online residuals. Run them through
    `scripts/reproduce_3d_dd_memory_saving_40pct.sh`.
  - The candidate MAT files store complete layer bases, affine reduced
    operators, local singular spectra, snapshot diagnostics, online histories,
    and a separate lightweight `paper_plot_data` payload. The reproduction
    wrapper reloads each MAT file and validates these artifacts after saving.
  - Actual grid/partition data, the normalized right-hand side, precorrection
    fields, and final complex solution fields are also retained. A tagged
    training checkpoint is written before online tests. See
    `md/3d_saved_results_and_redraw.md` for the MAT field map.

- `plot_helmholtz_3d_paper_figures.m`
  - Reads the frozen reproduction MAT files and draws separate 3D material
    isosurfaces, DD cuts, representative solution fields, and residual histories
    for Experiments 4 and 5. Fonts and solver colors match the 2D paper figures.
  - Copies publication PNGs into
    `LaTeX_DL_468198_240419/figures/helmholtz_3d/` by default.
  - Run directly or through `scripts/redraw_3d_paper_figures.sh`; the general
    `redraw_paper_output_figures('all')` workflow also includes these figures.
  - Uses the configuration inside the saved MAT, exports the online-results
    CSV, and accepts `data_dir`/`output_dir` for archived result pairs.
  - On the first redraw, extracts a compact field/partition cache from full
    `results`; subsequent redraws do not load bases. Exports reference fields,
    pointwise differences, and a full-domain field-comparison CSV as diagnostics.

- `verify_helmholtz_3d_saved_results.m`
  - Reloads a completed MAT and verifies saved basis/reduced-operator actions
    and the residuals of saved solution fields without retraining or solving.

- `run_waveholtz_3d_forward_timing.m`
  - Times one streamed, matrix-free 3D WaveHoltz period and optionally solves
    the full split-field `I-Pi` system with GMRES.
  - Compares the same parameter and grid with shifted-Laplacian MG-GMRES and
    estimates the two-layer, six-parameter ROM snapshot cost. This remains a
    feasibility implementation pending a dedicated split-field PML accuracy
    study. Run the recorded full solves through
    `scripts/time_3d_waveholtz_forward.sh`.

- `helmholtz_3d_problem_interface.m`
  - Geometry-pluggable 3D Helmholtz/PML interface with vectorized seven-point
    assembly, matrix-action, shifted-operator, and DD-partition callbacks.
  - Supports built-in half-space and spherical-inclusion smoke geometries;
    publication geometry can be supplied through callbacks later.

- `run_pc_ecc_residual_ablation_experiment.m`
  - Reproduces PC/ECC residual-ablation evidence outputs.
  - Defaults to marginal-storage greedy allocation. An explicit policy override
    is respected. `output_tag` isolates outputs and enables per-model checkpoints.

- `prepare_halfspace_dd_paper.m` and `prepare_greedy_ecc_pc_paper.m`
  - Current paper adoption uses greedy selection in all five examples.
    Half-space accepts the true Helmholtz residual at `1e-12` before completion;
    native WaveHoltz targets `1e-14`, `1e-13`, and `1e-12` remain diagnostics.
    ECC retains its `1e-8` relative-update criterion with relaxation `0.15`,
    with separate checks of actual Helmholtz residuals and reference errors.
  - Run `bash scripts/reproduce_greedy_paper.sh [all|halfspace|ecc|pc|redraw]`.
    `pc` explicitly retrains; `all` can reuse the validated saved greedy PC run.
    Tagged data are reused only with unchanged numerical code; use a fresh tag
    after numerical changes. Figures are redrawn from MAT data and published
    only to existing manuscript figure paths (or through the half-space publisher).

- `run_rom_threshold_balance_probe.m`
  - Research-only Global/DD sweep over simple three-layer ROM truncation
    schedules.  Records residual, completion time, ranks, and DD basis-entry
    savings without overwriting paper evidence MAT files.
  - Accepts an explicit `corr_block3_global_tail_strategy` through
    `solver_cfg`; use `output_tag` to keep allocation-policy reruns separate.

- `plot_half_ecc_dof_weighted_comparison.m`
  - Compares the equal-absolute and DOF-weighted Half-space/ECC paper-grid
    reruns in storage and three-step residual.

- `run_halfspace_dd_refinement.m`
  - Independent half-space DD partition/allocation study; preserves the
    manuscript grid, frequency, PML thickness, and `[1e-5,1e-5,1e-5]` tails.
  - `pilot` uses all 24 training parameters and five held-out parameters;
    `paper` uses all 23 original held-out parameters. Every candidate retrains
    its own three-layer trajectory and uses the same WaveHoltz completion.
  - Compares signed/depth/material-aligned PML partitions, physical slabs,
    Cartesian blocks, source rings, and symmetry-grouped exterior regions.
  - Saves per-candidate MAT artifacts with bases, affine reduced matrices,
    snapshots, spectra, final solutions, and residual histories. Cached runs
    are reused only when the requested configuration agrees exactly.
  - Reports basis storage separately from a deployable ROM payload estimate
    (basis + two affine reduced operators + one shared index map). This is
    not peak process memory and excludes training artifacts/full-order data.
  - Run `bash scripts/reproduce_halfspace_dd_refinement.sh pilot`;
    `test_halfspace_refined_partition` checks partition and projection algebra.
  - `paper` reproduces five selected representations; `redraw` reconstructs
    their figures from MAT files without solving. `plot_halfspace_dd_refinement`
    also exports per-block diagnostics. `diagnose_halfspace_dd_rank_limit`
    compares greedy and exact allocation on fixed saved training trajectories.
  - The historical 28.6% basis / 23.4% payload option uses two physical blocks,
    eight material-aligned PML groups, and the existing DOF-weighted policy.
    Marginal greedy with four physical strips reaches 33.4% / 27.1%, with
    additional completion iterations at the original `1e-14` stopping target.
    See `agent_runs/2026-09-05_halfspace_dd_refinement.md` for the full audit.

- `run_dd_rank_policy_comparison.m`
  - Compares DOF-weighted and marginal-storage allocation on the fixed paper
    half-space and 3D waveguide partitions, with complete trajectory retraining.
    Uses all 23 half-space queries and both publication waveguide queries.
  - Saves raw bases/fields through the existing drivers and writes comparison
    MAT/CSV files under `outputs/dd_rank_policy_comparison/<tag>/`.
    Run `scripts/compare_dd_rank_policies.sh [all|halfspace|waveguide] [tag]`.
    Reuse a tag only with unchanged numerical code; matching caches are resumed.
  - The waveguide crosscheck raises the snapshot iteration ceiling from 400
    to 800 steps for every representation, without changing the `1e-7`
    tolerance, and rejects artifacts whose actual snapshot residuals miss it.
    This exploration does not overwrite the publication configurations.
  - `plot_halfspace_dd_policy_residuals` redraws the saved half-space histories
    as separate Helmholtz-precorrection and native WaveHoltz-completion panels.
    It saves representative/worst-gap PNG and FIG files, lightweight plot data,
    and all-query diagnostics without rerunning the solver.

- `plot_ecc_global_spectral_energy.m`
  - Reads the primary ECC Global snapshot spectra and plots normalized mode
    energies and relative Frobenius tails with the actual thresholds/ranks.
    Saves the curves and audit metrics alongside PNG/vector PDF figures;
    included in `redraw_paper_output_figures('ecc')` and the full redraw.

- `run_waveguide_threshold_tradeoff.m`
  - Fixed-grid Experiment 3 sweep with paired layer thresholds
    `alpha * [0.0056,0.0278]`, `alpha = [0.25,0.5,1,2,4]`.
    Retrains both Global/DD trajectories with verified `1e-7` snapshots.
  - Train with `matlab -singleCompThread -batch "addpath('code'); run_waveguide_threshold_tradeoff('train');"`,
    then run `python3 scripts/run_waveguide_isolated_timings.py` after all
    training processes finish. An optional second MATLAB argument selects
    training indices; `plot` collects the completed isolated measurements.
  - Each distinct model uses a fresh MATLAB process, with one warm-up and
    three serial repetitions per query measuring online time,
    including required MG setup, at true residual `1e-6`. The main figure
    compares online acceleration with persistent ROM storage, including DD
    speed relative to Global. Iterations and post-ROM residuals are auxiliary.
  - Training outputs go to `outputs/waveguide_threshold_tradeoff_20260908/`;
    primary timings/figures are in its `isolated_timing/` subdirectory.
    Earlier shared-process timings are diagnostic data and are excluded.
    `plot_waveguide_threshold_tradeoff` redraws PNG/vector PDF from saved data.

- `run_pc_dd_storage_exploration.m`
  - Research-only PC comparison of three/four-block partitions and global-tail
    allocation rules.  The diagnostic `dof_weighted` rule assigns squared
    tail budget in proportion to block DOFs and preserves the requested global
    Frobenius-tail bound without a discrete optimization.
  - The exact `minimum_storage` selector is available only through
    `include_discrete_diagnostics=true`; it is a small-problem diagnostic whose
    Pareto state count is not assumed to scale polynomially.

- `run_pc_saved_rank_pruning_probe.m`
  - Tests rank deletions on a saved PC DD-ROM and recomputes the actual
    three-step Helmholtz FGMRES residual without retraining the references.

- `waveholtz_2d_pml_v5.m`
  - Verifies matched WaveHoltz/PML phasor recovery for five distinct source
    types on one fixed PML problem.
  - Saves MAT/CSV/settings outputs under `outputs/pml_v5_consistency/` and
    paper figures under `outputs/figures/pml_v5_consistency/`.

- `redraw_paper_output_figures.m`
  - Reloads saved MAT files and redraws paper figures.

- `plot_waveholtz_corrected_filter_polynomials.m`
  - Plots representative corrected filter polynomials and compares their
    maximum unit-circle gain with the standard \(1/4\) shift.

## Current paper truncation conventions

The paper scripts distinguish:

- Offline basis truncation tolerances choose correction-ROM ranks.
- Online WaveHoltz completion tolerances stop the post-correction completion
  iteration.
- `online_helm_accept_tol` optionally accepts the frequency-domain ROM output
  before lifting or calling WaveHoltz. The half-space paper uses `1e-12`.
  Skipped WaveHoltz states and residuals are stored as NaN, histories are empty,
  and `online_fp_stop_reason` is `helmholtz_residual`; these are not zero WH
  residuals. Fixed-point timing now includes final checking and phasor recovery.

Current offline basis truncation settings:

- The global TA-ROM correction basis uses the layerwise relative
  Frobenius-tail tolerances `corr_pod_singular_value_tols` or
  `corr_rank_rtols`.  The paper schedules are Half-space
  `[1e-5, 1e-5, 1e-5]`, ECC `[1e-3, 1e-2, 1e-1]`, and PC
  `[1e-4, 1e-3, 1e-2]`.
- The global TA-ROM correction basis applies the relative Frobenius-tail rule:
  choose the smallest rank `r` such that
  `norm(sigma(r+1:end)) / norm(sigma) <= tol`.
- DD-ROM applies relative Frobenius-tail truncation blockwise.  Within each
  experiment, DD uses the same layerwise global snapshot-tail budgets as the
  Global basis. The `marginal_storage` rule repeatedly selects the largest
  next singular-value square per additional basis entry. No block tolerance
  is hand tuned. After the spectra are available, selection costs
  `O(sum(p_b) + B*R)` with a scan of `B` blocks at each of `R` retained modes.
  This fast heuristic does not claim exact integer storage optimality.
- Current paper DD partitions are:
  - Helm half-space: left/right physical material blocks plus eight PML
    groups, aligned with material side and inner/outer attenuation depth.
    Upper/lower counterparts are grouped without imposing field symmetry.
  - ECC/scattering: near-obstacle, far-field, and PML blocks.
  - Photonic crystal: narrow channel core, physical cladding, left/right PML,
    and top/bottom-plus-corner PML blocks, with channel half-width
    `0.75 * pc_lattice_a`.
  - 3D Gaussian lens and waveguide: five material-profile bands with axial
    sections, plus signed PML faces split into inner and outer depth layers.
    These cases use `marginal_storage` allocation under the same global
    Frobenius-tail budget.

## Running paper code

Run commands from the project root, not from `code/`, so relative paths and
LaTeX figure references remain easy to inspect.  All generated MAT files,
figures, and CSV files are written under the top-level `outputs/` directory via
`helm_output_path(...)`.

Quick smoke checks:

```bash
matlab -batch "addpath('code'); run_helm_trajectory_evidence_experiment('smoke');"
matlab -batch "addpath('code'); run_pc_ecc_residual_ablation_experiment('both','smoke');"
matlab -batch "addpath('code'); run_helm_direct_shifted_laplacian_scaling_pilot('smoke');"
matlab -batch "addpath('code'); run_helm_ta_shifted_scaling_experiment('smoke');"
matlab -batch "addpath('code'); test_helmholtz_3d_problem_interface;"
matlab -batch "addpath('code'); test_shifted_laplacian_multigrid_3d;"
matlab -batch "addpath('code'); run_helmholtz_3d_candidate_experiments('smoke');"
```

Run the full fixed-PPW direct/shifted-Laplacian decision pilot with:

```bash
matlab -batch "addpath('code'); run_helm_direct_shifted_laplacian_scaling_pilot('pilot');"
matlab -batch "addpath('code'); run_helm_ta_shifted_scaling_experiment('paper');"
```

Full paper runs:

```bash
bash scripts/reproduce_halfspace_dd_paper.sh assemble
matlab -batch "addpath('code'); run_helm_shifted_laplacian_completion_experiment('paper');"
matlab -batch "addpath('code'); run_pc_ecc_residual_ablation_experiment('both','paper',struct('run_tradeoff',false));"
```

The shifted-Laplacian paper run uses the common `1e-12` Helmholtz-residual
target and accepts an already converged ROM field without completion. Its
MAT archive is `outputs/helm_shifted_laplacian_completion/helm_shifted_laplacian_completion_paper_tol1e12.mat`,
separate from the older roundoff-target run. Regenerate Figure 6 without
solving again using `redraw_paper_output_figures('shifted')`.

To run one PC/ECC case at a time:

```bash
matlab -batch "addpath('code'); run_pc_ecc_residual_ablation_experiment('pc','paper',struct('run_tradeoff',false));"
matlab -batch "addpath('code'); run_pc_ecc_residual_ablation_experiment('ecc','paper',struct('run_tradeoff',false));"
```

The half-space paper figures now show the accepted layerwise and physical/PML
storage counts and a two-tolerance completion-cost diagnostic. The former
three-block ROM-tail sweep and old speedup curve are retired from the paper.
Use `run_rom_threshold_balance_probe` for internal threshold studies only.

After rerunning experiments, redraw all paper figures and partition schematics
from the validated MAT files with:

```bash
matlab -batch "addpath('code'); redraw_paper_output_figures;"
```

## Path/output helpers

To redraw only the solution fields in all five experiments, without changing
partitions, residuals, spectra, or tables:

```bash
matlab -batch "addpath('code'); redraw_paper_output_figures('fields');"
```

Signed real/imaginary fields share `helm_solution_colormap`: blue for negative
amplitudes, white at zero, red for positive amplitudes, with asinh-enhanced
contrast. Pass the existing color limits to preserve their ranges and the
neutral zero even when those limits are asymmetric. Field values and colorbar
labels are untransformed. Magnitude/error maps keep their sequential palettes.

For the paired half-space residual figures and the PC display panels only:

```matlab
plot_halfspace_dd_paper([], [], 'residual');
plot_helm_shifted_laplacian_completion_figure('paper', ...
    'helm_shifted_laplacian_completion', 0.8622);
redraw_paper_output_figures('pc_display');
```

Half-space and PC residual plots now use two panels with unscaled
frequency-domain and native WaveHoltz histories; the old combined-history
inset is retired. The PC solution PNG retains its historical `_real_` filename for paper
compatibility but now contains real and imaginary panels with a shared range.

- `helm_project_root.m`
  - Returns the project root, one level above `code/`.

- `helm_output_path.m`
  - Builds absolute project-root-relative paths, so outputs are written to
    the top-level `outputs/` directory even if MATLAB is run from `code/`.
# Historical reviewer-audited evidence (2026-09-05)

The `review_repaired_20260905` cohorts supersede the earlier greedy-adoption
2D results. Training and reduced online FGMRES use two-pass MGS; recurrent
snapshots are residual-checked and replaced by direct inverse actions when
needed. PC now uses direct Helmholtz training references. Its updated basis
saving is 7.4%, not the earlier 13.8%; half-space and ECC retain 31.8% and 48.3%.

Use `scripts/reproduce_review_repairs.sh` for the repaired runs and unit tests.
`audit_revision_evidence('legacy')` and `audit_revision_evidence('repaired')`
keep the two provenance chains separate. `write_revision_audit_tables` generates
the manuscript audit tables from saved data. `run_3d_frozen_holdout_audit` loads
existing complete 3D models, verifies their configuration and evaluates newly
selected interpolation/extrapolation queries without retraining.

For 3D persistent ROM storage, count one shared DD index map in addition to
bases and affine reduced matrices. Historical MAT counters omitted this map;
the revised manuscript savings are 39.0% / 34.7%, while basis-only savings
remain 46.4% / 41.0%. See the point-by-point record in `agent_runs/` for remaining
benchmark, timing and public-release limitations.

### Half-space unified true-residual figure

`run_halfspace_true_residual_figure` reruns the representative half-space query
with the accepted training configuration and writes a separate diagnostic MAT.
`run_halfspace_true_residual_figure('redraw')` only redraws it; the standard
half-space paper redraw also uses this diagnostic when available.

The opt-in `cfg.record_true_helm_history=true` requires streamed fixed-point
completion. It saves `online_true_helm_relres_hist` and
`plain_true_helm_relres_hist` separately from native stopping residuals.
The homogeneous period accumulates the phasor alongside the original filter;
a forced zero-state phasor is computed once per solve and added by linearity.
Thus there is one extra recovery period per solve, plus projection sums and
one Helmholtz matrix-vector product per recorded state, not an extra period
at every update. Original timing datasets are not overwritten.
The plot retains both ROM and recovered-field transition values without
scaling. `test_halfspace_true_residual_history` checks unchanged native
trajectories and independent recovered-field residuals.

### Actual process memory measurements

`prepare_paper_memory_benchmark` exports model-only files before measurement.
Run `python3 scripts/measure_paper_memory.py` from the repository to launch
three fresh, single-computational-thread MATLAB processes per method and
problem, in alternating Global/DD order. The driver uses representative
parameters and the original reduced/completion kernels through optional
`cfg.deployment_file` entry points, without offline training or reference
solves. Three-dimensional runs use the existing deployment driver with one
warmup and one recorded solve, with inverse-action diagnostics disabled.
`/usr/bin/time -l` peak RSS includes startup,
loading, assembly and solve workspaces; `whos` reports loaded model size.
Raw logs and results are in `outputs/paper_memory_benchmark/`. These peaks
are not full-test-set maxima, and model savings must not be substituted for
whole-process RSS savings.

After all 18 runs, `validate_paper_memory_benchmark` checks residuals and repeat
counts; `python3 scripts/summarize_paper_memory.py` writes the measured table.

### September 9 reviewer sensitivity evidence

`run_reviewer_20260909_study` adds independent Helm half-space frequency,
two-material, and partition studies using the active PML assembly and reduced
space/Krylov utilities. `train` and `timing` are separate phases; the
`explore_train`/`explore_timing` pair adds the complete 12-configuration
parameter-range, training-density, and threshold scan. `diagnostics` computes
conditional error-model constants outside timed solves. The `holdout` mode
reads a previously frozen `selected_holdout.mat` and evaluates the independent
validation grid. `validate_reviewer_20260909_study` checks affine projection,
replayed inverse-action targets, greedy saturation and infeasible caps, and
PML-free ECC boundary forcing in all archived primary systems.

`run_halfspace_default_model_retime` (2026-10-24) unifies the default DD model
of Experiment 1.  The partition and omega=53.5 frequency results used a
separately trained default model with ranks `[44 80 122]`; the driver converts
the main model (`halfspace_dd_refinement/review_repaired_20260905`, ranks
`[44 80 123]`) to the reviewer format and checks its reduced operators.  It then
times all partition variants and both baselines in one session with the
reviewer protocol and writes to `outputs/halfspace_default_retime_20261024/`.

The optional second argument `'frequency_4'` restricts `train` or `timing` to
the omega=214 algebraic extension and writes separate summary files.
`validate_reviewer_frequency_214` checks its saved operator, projected actions,
replayed snapshots, and payload accounting without expanding all DD columns
into a global dense matrix. Timing remains one warm-up plus five repetitions
on the original five queries, with shifted ILU as baseline and completion.

The September 10 DD online-cost audit uses `rom_replay_inverse_action` to
reconstruct only the returned iterate; block lifting now allocates the output
with the coefficient type to avoid real-to-complex conversion. The frozen-model
`benchmark_dd_online_optimization` checks bitwise equality of the complete
returned trajectory and rotates before/after Global/DD timing order. It writes
a separate cohort under `outputs/dd_online_audit_20260910`, using archived
`*_before.m` helpers in that directory. `run_dd_online_audit` provides component
timings. See `agent_runs/2026-09-10_dd_online_optimization.md` for the protocol;
existing paper timing CSVs are preserved.

`run_reviewer_high_frequency_waveholtz` separately exercises the active full
Helm/WaveHoltz workflow at omega=53.5,107,214 (fixed PPW, through 570,025 DOFs),
including WaveHoltz-GMRES baselines and optional completion. High-frequency
stress-test timings are single-pass and kept separate from repeated algebraic
sensitivity measurements. See `md/reviewer_20260909_reproduction.md` and the
September 9 work log for configurations, selection rules, and complete results.

`run_omega214_solver_comparison` compares zero-start WaveHoltz-GMRES,
shifted-Laplacian ILUTP-GMRES, and fresh sparse direct solution on the archived
omega=214 grid and three paper queries. Run `smoke`, then `algebraic` and
`waveholtz` with `-singleCompThread`. The opt-in `cfg.baseline_probe_solver`
branch bypasses ROM training and reuses the active solvers. Every solve must
meet true residual 1e-10; setup and recovery are included in online timing.
See `agent_runs/2026-09-10_omega214_waveholtz_vs_shifted.md` for the protocol.

`run_reviewer_waveholtz_prefix_probe` uses the opt-in
`cfg.frozen_online_file` branch of the active Helm entry point to evaluate a
frozen layer prefix without training or reference diagnostics. It checks exact
operator/RHS compatibility, applies the original lift and verified WaveHoltz
completion, and counts actual completion iterations. This one-layer high-frequency
ablation is separate from the three-layer frequency study. The associated
`plot_reviewer_high_frequency` keeps true Helmholtz and native fixed-point
histories on separate axes.

`test_reviewer_frozen_online` validates the saved successful prefix smoke and
rejects mismatched operators or forcing before online work. The selected
two-material model is `explore_w05_n8_e-4`; its hashes were frozen before the
nine-query, five-repeat independent validation. Every development candidate
remains in the full-scan CSV and manuscript appendix.

### 64 CPU cluster / 3D frequency study

`run_scnet_3d_experiment` adapts the 3D affine half-space and Gaussian-waveguide workflows to
headless Slurm runs. `run_scnet_3d_scaling` measures parameter-task concurrency;
`test_helm_3d_hpc` checks the optimized coupled projection and fixed-linear-MG
GMRES path against the original algebra. See
[`md/scnet_3d_parallel_study_20260909.md`](../md/scnet_3d_parallel_study_20260909.md)
and [`scripts/scnet_3d.slurm`](../scripts/scnet_3d.slurm). These 3D runs use shifted
MG completion; the separate split-field WaveHoltz probe has its own validation scope.

The cluster entry now saves per-parameter inverse actions and atomic per-layer
checkpoints. Re-running the same case directory resumes accepted work after
checking operator/configuration/RHS identity and recomputing true residuals.
`test_helm_3d_checkpoint` verifies both partial-layer cache reuse and layer resume
against a fresh solve. `scripts/scnet_3d_sweep.slurm` runs four cases with at most
one 64-CPU allocation active; its minimum and maximum time limits are identical.

`run_scnet_3d_solver_probe` compares beta, restart and direct coarse-grid size
on an unchanged retained difficult RHS, preserving every passing and failing
configuration. The optional sixth argument of `scripts/scnet_3d.slurm` loads
a solver-options JSON; selected settings must then pass complete ROM training
and the original held-out queries. The measured production layout is eight
workers with four threads each. `test_helm_3d_failure_collection` checks that
all numerical failures in a layer are saved before that layer is rejected.

### Consolidated half-space manuscript (September 10)

`benchmark_algebraic_frequency_baselines` refreshes direct and shifted-ILU
measurements at all four frozen frequencies, with one warm-up and five
repetitions on each of the same five queries. The results join the optimized
Global/DD audit in `outputs/dd_online_audit_20260910/`. The paper generator
uses these online measurements with the original frozen model storage and
offline construction costs. The manuscript now groups representative-case,
frequency, and two-parameter evidence as parts (a)--(c) of Section 5.1; the
separate high-frequency WaveHoltz comparison is archived. Existing WaveHoltz
drivers remain available for reproducing the historical evidence.

### Half-space extension figures

`export_halfspace_frequency_figure` evaluates the saved omega214 DD model on
one prescribed query and independently checks the field against a direct
solve. `scripts/plot_halfspace_extensions.py` generates the solution, frequency
trends, two-parameter schematic/sample grid, and nine-query validation figures.
The revised waveguide threshold plot displays P2--P5 while retaining all five
raw comparison rows. See `agent_runs/2026-09-10_halfspace_results_figures.md`.

### ECC third-layer necessity audit

`run_ecc_layer3_audit('residual')` independently recomputes the full-domain
residuals of all saved training/test iterates in the current Experiment 2
cohort. The `'completion'` mode also replays frozen two- and three-layer
prefixes through the original ECC online kernel and WaveHoltz completion,
with unchanged stopping settings. It checks prefix agreement and reproduces
the original three-layer completion counts. The opt-in `deployment_audit`
flag returns diagnostic fields from a single-query deployment solve.
`scripts/summarize_ecc_layer3_audit.py` writes a comparison figure and report.
Outputs remain separate in `outputs/ecc_layer3_audit_20260911/`; timings are
single-pass exploratory measurements, not replacement paper benchmarks.

### Current Experiment 2: two-layer paper cohort

Run `benchmark_ecc_two_layer_paper`, then `prepare_ecc_two_layer_paper` and
`validate_ecc_two_layer_paper`. The benchmark uses four independent query
workers (one thread each), a per-method warm-up on every worker, and three
repetitions at all 11 queries for Global, DD, and plain WaveHoltz. It applies
the frozen first two layers without retraining. Preparation writes the current
`outputs/ecc_two_layer_paper_20260911/paper_data.mat`, lean deployed models,
storage counts, and updated paper figures. The default ECC redraw now requires
this cohort; historical three-layer data are used explicitly for the compact
layer-depth reference. `plot_ecc_global_spectral_energy` follows the supplied
model's actual depth. The older `run_pc_ecc_residual_ablation_experiment` paper
defaults reproduce the historical three-layer cohort.

`scripts/plot_waveguide_frequency_paper.py` renders the verified matched-node
omega24/36/48 cohort for Experiment 3(b). It writes figures into the LaTeX
project and a generated table into `outputs/experiment23_revision_20260911/`;
the table is included inline in `template.tex` between the frequency-table
markers. No new manuscript fragment files are needed.

## Deployment object measurement (2026-09-13)

`measure_waveguide_rom_whos(source, output_file, report_file)` loads frozen
3D Global/DD training models and reproduces the legacy lean deployment
layout with per-layer indices. It verifies unchanged bases, affine matrices and
indices, then records whole-object and numeric-array `whos` bytes. Per-layer
index fields are retained; the one-map accounting is verified separately.
No training, full-order assembly, projection solve or completion is run.
The optional fourth argument checks exact equality with existing representative
deployment files. HPC cohort execution uses MATLAB R2024a via
`scripts/scnet_waveguide_whos.slurm`; see the dated run record.


`build_jsc_3d_deployment_model` now supplies the lean exporter with one top-level
`shared_block_indices` cell array when every blockn layer has exactly the same
partition. The layer block structures omit `idx`; bases, ranks and affine
matrices remain unchanged. Differing partitions retain the legacy layout.
`run_jsc_3d_deployment` and `rom_true_action_diagnostics` accept both layouts;
`rom_reduced_space_utils` project/lift take an optional shared index map.
The training model and historical paper measurements remain unchanged.

`audit_waveguide_shared_indices(source, output_file, reference_dir)` measures
both actual objects and verifies identical retained values, projection, lift,
and reduced inverse actions at the three paper queries. The optional reference
directory checks the legacy object against existing deployment MAT files.
`validate_shared_index_deployment(reference_file, output_dir)` verifies MAT
save/load and an entire existing query including completion and diagnostics.
New measurements are separate from the historical per-layer-index evidence;
logical byte reductions do not imply equal reductions in physical RAM or RSS.

### ECC strict DD tail-budget audit (2026-09-19)

`run_ecc_dd_budget_study('audit')` checks the historical three-layer ranks.
`'train'` retrains four matched Global/DD tolerance schedules with no independent
local tail constraints and audits the actual snapshot projection errors.
`'screen'` compares all schedules on the original development cohort;
`'confirm'` and `'validate'` compare the uniform and selected third-layer-relaxed
schedules. Validation excludes training, original tests, and the previous tuning
validation cohort. `verify_ecc_dd_budget` checks production deployment replay.
See `md/ecc_dd_budget_20260919.md`; outputs use a separate directory.
The historical `run_ecc_shifted_tuning` includes additional local tail caps and
must not be described as the paper's zero-start global-budget allocation.

For the requested all-three-layers `1e-3` follow-up, run
`scripts/reproduce_ecc_uniform3.sh`. It trains Global/DD from scratch under the
same global-only truncation rule and compares against the previous layerwise
schedule on the frozen 22-query cohort, with five paired repeats. This is a
follow-up on an existing cohort, not a new independent holdout. Results use
`uniform3_*` filenames and preserve the original study tables.

The two-layer `1e-6` ablation is `scripts/reproduce_ecc_two6.sh`: retrain two
layers, audit their snapshot tails, compare to the previous three-layer
`[1e-6,1e-6,1e-3]` models on the frozen cohort, and verify production replay.
Files use the `two6_*` prefix; verification supports the actual model depth.

The all-three-layers `1e-5` follow-up is
`scripts/reproduce_ecc_uniform5.sh`. It retrains both representations, audits
actual truncation tails, then compares against the paper's layerwise schedule
on the same frozen 22-query cohort with five paired repeats. Output names
start with `uniform5_`; this follow-up does not change the paper configuration.
