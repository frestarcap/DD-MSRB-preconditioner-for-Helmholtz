# Reproducible experiment scripts

Run these commands from any directory. Set `MATLAB_BIN` if MATLAB is not on
`PATH`.

## <cluster> supercomputer

Read [the <cluster> calling rules](../md/<cluster>_usage.md) before submitting
cluster jobs. They document SSH, Slurm resources, the verified MATLAB R2024a
startup, job monitoring, and result collection. Local wrappers that use
`-batch` need their launch flags adapted for this environment; setting
`MATLAB_BIN` alone changes only the executable path.

## 3D DD memory saving

```bash
./scripts/reproduce_3d_dd_memory_saving_40pct.sh lens
./scripts/reproduce_3d_dd_memory_saving_40pct.sh waveguide
./scripts/reproduce_3d_dd_memory_saving_40pct.sh all
```

Each case trains global and DD trajectory-aware reduced inverse-action spaces
with identical training parameters and layer tolerances. Expected results are:

| Case | Unknowns | Global basis | DD basis | Saving |
| --- | ---: | ---: | ---: | ---: |
| Gaussian lens | 658,503 | 70.34 MiB | 37.73 MiB | 46.4% |
| Gaussian waveguide | 680,225 | 93.41 MiB | 55.14 MiB | 41.0% |

Results are written under `outputs/helmholtz_3d_candidates/large/`. The entry
point fails if the DD saving falls more than two percentage points below the
recorded value or an online DD solve misses the `1e-6` residual tolerance.
Each candidate MAT file contains the complete global/DD layer bases, reduced
operators, local spectra, training diagnostics, online residual histories, and
a lightweight `paper_plot_data` variable. Grid coordinates, actual DD indices,
the normalized right-hand side, precorrection fields, and final solution fields
are also saved. A `*_training.mat` checkpoint preserves the trained models
before online tests begin. See `md/3d_saved_results_and_redraw.md` for loading
and validating these arrays.

Redraw both three-dimensional experiments without rerunning a solve:

```bash
./scripts/redraw_3d_paper_figures.sh
```

The same redraw is available from MATLAB through
`redraw_paper_output_figures('3d')`; `redraw_paper_output_figures('all')`
includes it with the two-dimensional figures.
Each case now has its own profile isosurface, DD-cut, solution, and residual
figure. The first redraw extracts a compact field cache; later redraws reuse
it. Reference-field and difference plots are also exported for verification.

## 3D WaveHoltz timing

```bash
./scripts/time_3d_waveholtz_forward.sh lens
./scripts/time_3d_waveholtz_forward.sh waveguide
./scripts/time_3d_waveholtz_forward.sh all
```

This runs a full split-field WaveHoltz GMRES solve to `1e-7`, a one-period
timing probe, and the shifted-Laplacian MG-GMRES baseline on the same grid and
parameter. Results are written under `outputs/waveholtz_3d_timing/`.

The WaveHoltz implementation streams the time filter and does not retain the
period history. It is a feasibility implementation; use the current shifted
completion for publication runs until the 3D split-field PML has a documented
accuracy and refinement study.

## September 9 reviewer evidence

After the MATLAB studies in `md/reviewer_20260909_reproduction.md`, run
`python3 scripts/write_reviewer_20260909_evidence.py` to regenerate the
supplementary frequency, two-parameter, partition, and conditional-bound tables
from CSVs. The full development scan preserves all candidates and repeated
samples, including adverse storage/time results. Primary manuscript figures and
historical numerical datasets retain their existing provenance.

For the omega=214 algebraic extension, run
`python3 scripts/summarize_reviewer_frequency_214.py` after MATLAB validation
and timing. It checks all 100 recorded samples and writes Global/DD storage,
offline/online costs, and per-query timing variability. Use
`python3 scripts/write_reviewer_20260909_evidence.py --frequency-only` to update
only the algebraic frequency tables, including the new row.

`package_reviewer_20260909.py` creates local response/manuscript PDFs, a
source/evidence ZIP, and a separate supplementary numerical-data TAR with
SHA-256 manifests. Run it after numerical checks, table/figure generation,
and LaTeX compilation. It requires ReportLab; scientific figures use
Matplotlib. Packages stay under `outputs/reviewer_20260909/delivery/`.

`plot_reviewer_claims.py` draws three PNG/SVG figure pairs from the completed
reviewer CSVs: high-frequency verification, independent two-material validation,
and all 12 development configurations. It performs no solves or manuscript
edits. Input hashes, query coordinates, and figures are saved under
`outputs/reviewer_20260909/figures/claim_visuals_20260909/`; see
`md/reviewer_claim_figure_map_20260909.md` for the exact manuscript locations.

The separate local omega=214 half-space baseline comparison is summarized by
`python3 scripts/summarize_omega214_solver_comparison.py` after the MATLAB
`run_omega214_solver_comparison('algebraic')` and `('waveholtz')` stages finish.
It checks all 21 recorded samples and writes per-query CSV and aggregate JSON
under `outputs/omega214_solver_comparison_20260910/`.

### <cluster> 3D Slurm entry

`scnet_3d.slurm qualify` runs numerical qualification and worker/thread layout
calibration on one 64-CPU node with a 192-GiB allocation. After qualification,
`scnet_3d.slurm full 24 8 4` runs the 3D waveguide at omega=24 with eight workers
and a computational thread limit of four per worker, the selected calibration layout. Submit with `sbatch` from an isolated
repository copy containing `code/`, `scripts/`, and `logs/`. See
`md/scnet_3d_parallel_study_20260909.md` for fixed PPW and residual acceptance.

`scnet_3d_sweep.slurm` submits a serial Slurm array for waveguide omega=24,
halfspace omega=48, waveguide omega=36, and waveguide omega=48. Each case uses
64 CPUs / 192 GiB and the measured 8-worker / 4-thread layout. TimeMin equals
TimeLimit (four hours), so backfill cannot silently shorten the requested limit.
Use `fetch_scnet_3d.sh REMOTE_RUN LOCAL_RESULTS` to retrieve lightweight results,
then `plot_scnet_3d.py` (or `--problem halfspace`) to generate scientific figures.

`monitor_scnet_3d.py --watch` reads the existing array described in
`outputs/scnet_3d_20260909/expected_cases.json`, collects lightweight results,
draws each completed report, and updates `results_audit_zh.md`. It does not submit
or alter jobs. Omit `--watch` for one collection pass. The plotted 3D view embeds
an actual exported midplane solution in the physical domain; it does not infer
unexported volume values. Full solution fields and trained bases remain in the
remote MAT files.

`scnet_3d_solver_probe.slurm RETAINED_SNAPSHOT_MAT OUTPUT_TAG` performs eight
beta/restart/coarse-grid diagnostics on the exact saved RHS. Failed settings
remain in the MAT/JSON outputs and `plot_scnet_3d_solver_probe.py` figures.
For full runs, the optional sixth argument of `scnet_3d.slurm` is a solver
profile JSON; the fifth argument selects `waveguide` or `halfspace`. See the
study notes for frozen profiles, source versions and job IDs. The initial
default solver stalled at two high-frequency cases, so the original sweep
is retained as diagnostic evidence and followed by separately tagged retries.

`verify_scnet_3d_results.py` cross-checks saved numerical reports, CSV rows,
training logs, input geometry and Slurm accounting. Use
`--require-uniform-complete` for the final acceptance of all matching reruns;
the JSON verification record preserves incomplete/failed cases and input hashes.
`plot_scnet_3d_cost.py` separates standalone training, online queries, actual
workflow wall time and batch MaxRSS. It accepts `--profile`, `--node`, and
`--output` like `plot_scnet_3d.py`.
`redraw_scnet_3d_study.py --require-uniform-complete` rebuilds the full evidence
audit and all plots locally from the collected files, using NumPy and Matplotlib.
It does not contact the cluster or rerun the numerical experiments.

For the high-frequency study, use the frozen profile rather than
omitting the solver-options argument:

```bash
sbatch scripts/scnet_3d.slurm full 48 8 4 waveguide profiles/beta2_coarse12000_restart40.json
```

The profile is included in the frozen deployment archive; its local evidence
copy is under `outputs/scnet_3d_20260909/profiles/`.
After final verification, `package_scnet_3d_evidence.py` bundles the collected
lightweight evidence, all figures, the frozen HPC source archive, and offline
analysis scripts. It verifies the per-file archive hashes and excludes MAT
files, MATLAB binaries and SSH connection state.

### September 10 consolidated frequency tables

`write_reviewer_20260909_evidence.py --frequency-only` now requires all four
frozen algebraic frequency models, the complete paired ROM timing audit, and
the refreshed baseline CSV produced by `benchmark_algebraic_frequency_baselines`.
It selects the optimized Global/DD rows for every frequency and preserves the
original storage and construction measurements. The obsolete separate
WaveHoltz high-frequency table generator has been removed; its previous source
is archived under `outputs/paper_restructure_20260910/before/`. Other study
tables retain their own historical timing cohorts.

`plot_halfspace_extensions.py` reads the verified MATLAB field export and
saved frequency/holdout CSVs to generate four publication figures. It requires
NumPy, SciPy, and Matplotlib (available in the reviewer tooling virtualenv)
and writes figure provenance and the exact validation query key. The frequency
table generator now writes one main-text summary and two detailed appendix
tables, preserving the original main-table labels.

### Four-frequency MATLAB object sizes

`scnet_waveguide_whos.slurm` runs one MATLAB R2024a inspection process per
frequency using an explicit `sources.tsv` manifest in the submission directory.
It hashes original training files and reports, records `whos` object and array
bytes, and logs inspection RSS separately from solver performance.
`summarize_waveguide_whos.py` verifies all eight objects, their formula checks,
source report hashes, common MATLAB version and successful job exit, then
writes `outputs/waveguide_whos_20260913/summary.json` and CSV.
`plot_waveguide_frequency_paper.py` uses that verified summary alongside the
accepted frequency reports; Table 12 compares accounted and measured storage,
and Figure 20(a) shows deployed-object `whos` measurements.


`scnet_waveguide_shared_indices.slurm` repeats the four-frequency inspection
with both the legacy and shared-index deployment representations under R2024a.
It calls `audit_waveguide_shared_indices` to compare retained fields and reduced
inverse actions, without retraining or high-frequency completion solves.
The submission directory needs the same `sources.tsv` manifest. Before/after
measurements are stored separately in `outputs/waveguide_shared_indices_20260913`;
the earlier manuscript tables still describe the legacy deployment format.

### Unified waveguide paper figures (2026-09-13)

plot_waveguide_frequency_paper.py now draws all three paired Global/DD
online time ratios, the omega48 representative field/ROM-history figure,
and generates waveguide_representative_online.tex and
waveguide_representative_storage.tex from the saved frequency cohort.
It preserves the four-frequency summaries and uses atomic figure replacement.
No training or online solves are performed by this script. The omega24
two-layer sensitivity study is included separately through
waveguide_threshold_appendix.tex.

## JSC / JCP Overleaf synchronization

The original manuscript is now `JSC/template.tex`. The old directory name
`LaTeX_DL_468198_240419` is a compatibility symlink to `JSC`, so existing
experiment/export scripts still target the JSC manuscript.
The JCP adaptation is `JCP/elsarticle-template-num.tex` (Elsevier preprint,
numerical references). Select this file as the Overleaf main document. The two other original root
template filenames forward to it, so existing Overleaf main-file choices also
compile the manuscript.

```bash
./scripts/overleaf_pull.sh JSC
./scripts/overleaf_push.sh JSC "Sync JSC manuscript"
./scripts/overleaf_pull.sh JCP
./scripts/overleaf_push.sh JCP "Sync JCP manuscript"
```

No project argument defaults to JSC for backward compatibility. Alternatively,
set `OVERLEAF_PROJECT=JCP`. Existing environment overrides remain supported;
`OVERLEAF_REMOTE_SUBDIR=''` now correctly selects the remote root.
JSC uses `.overleaf/paper` and its existing remote nested source directory;
JCP uses `.overleaf/jcp` and the new remote project root. The JCP clone's Git
metadata is kept in that mirror, following the existing mirror/rsync design.
Pull refuses to overwrite uncommitted local manuscript changes; push commits
in the mirror, rebases, then pushes without force. Generated manuscript PDFs
and LaTeX auxiliary files are excluded; publication figure PDFs are included.

Local JCP build:

```bash
cd JCP
latexmk -pdf elsarticle-template-num.tex
```

On the current Mac, newer user-installed LaTeX core files conflict with the
system format. The verified local invocation prioritizes system packages:

```bash
TEXINPUTS='/usr/local/texlive/2026basic/texmf-dist/tex/latex//:' latexmk -pdf elsarticle-template-num.tex
```

This environment override is local only; Overleaf does not need it. The supplied
`elsarticle.cls` was generated from the cloned template's `elsarticle.ins`.

## JCP representative half-space / ECC shifted-Laplacian comparison

```bash
matlab -batch "addpath('code'); run_jcp_shifted_experiments('smoke');"
matlab -batch "addpath('code'); run_jcp_shifted_experiments('paper');"
python3.12 scripts/summarize_jcp_shifted.py
```

The runner preserves the manuscript's original Global/DD training spaces and
23 half-space / 11 ECC test parameters. The half-space NR control is reconstructed
with the active equal-rank trainer; fixed POD uses the saved seven-dimensional
Galerkin basis. All benchmark solves use frequency-domain matrices and shifted
ILU (beta 0.2, drop tolerance 1e-3, restart 40, up to 20 cycles). True residual
targets are 1e-12 for half-space and 1e-10 for ECC. No WaveHoltz iteration is used.

Timings run serially with one computational thread, one warm-up and five measured
repeats per method/query, rotating method order. Required preconditioner setup is
included. Common full-order assembly, model loading and offline training are
excluded. Sparse direct solves are timed as an additional reference baseline.
Saved prefix fields and residual acceptance are checked for every query.

Results, raw timings, residual histories, verification and plots are written to
`outputs/jcp_shifted_20260914/`; no historical evidence or manuscript is changed.
The `frozen_cohorts.mat` cache contains the exact tested matrices, bases, references
and queries. `provenance.json` records original data and dependency hashes for the
September 14 run. The report generator requires matplotlib and NumPy.

For the JCP manuscript replacements, export the saved frequency-domain fields
and regenerate the publication assets without rerunning the solves:

```bash
matlab -batch "addpath('code'); export_jcp_shifted_fields;"
python3.12 scripts/write_jcp_shifted_paper.py
python3.12 scripts/write_jcp_waveguide_table.py
```

This writes JCP-only timing inputs, field/residual/regional-storage figures,
and the waveguide frequency table with separate calculated/measured storage
panels and independent DD/Global timing columns. Paired high-frequency evidence
is extracted to `outputs/jcp_dd_presentation_20260914/`. See
`md/jcp_shifted_reproduction.md` for data paths and the complete protocol.

### JCP table generators: preserve the approved prose and layout

The six tables below reproduce the current manuscript text, including the 12pt
font, column spacing, captions, and SL-step wording. Do not copy output from the
legacy `write_reviewer_20260909_evidence.py`: it targets JSC through the
compatibility symlink and intentionally retains a different narrative.

| JCP output | Generator (table-only command) |
|---|---|
| `jcp_halfspace_timing.tex` | `python3.12 scripts/write_jcp_shifted_paper.py --tables-only` |
| `reviewer_selected_validation.tex`, `reviewer_parameter_partition_tables.tex` | `python3.12 scripts/write_jcp_reviewer_tables.py` |
| `jcp_ecc_timing.tex` | `python3.12 scripts/write_ecc_budget_paper.py --timing-only` |
| `jcp_waveguide_frequency.tex` | `python3.12 scripts/write_jcp_waveguide_table.py --table-only` |
| `jcp_waveguide_validation.tex` | `python3.12 scripts/write_jcp_3d_validation.py --table-only` |

The JCP reviewer generator reads `summary_train.csv`, the frozen selection,
`summary_explore_train.csv`, and the timing/holdout CSVs. Its editable prose and
layout live in `scripts/templates/jcp/*.tex.in`; numerical rows and narrative
metrics are regenerated from those sources. Edit those templates when changing
the corresponding captions or paragraphs. The other four tables keep their
presentation strings in their listed Python generators. The shifted-paper
writer no longer writes the obsolete 11-query ECC table; uniform5 is its sole
current timing-table source.

Check all six without rewriting manuscript files or figures:

```bash
python3.12 scripts/check_jcp_table_generators.py
```

This regenerates into a temporary directory and compares the complete output
byte for byte, including numerical cells, whitespace, captions, and prose.


### Current JCP publication panels (September 25)

`plot_jcp_publication_layout.py` redraws the current title-free individual
panels for LaTeX subfigure typesetting from frozen MATLAB/CSV/JSON data.
Run `export_jcp_layout_data` in MATLAB first. The `waveguide_geometry` target
of `plot_helmholtz_3d_paper_figures` exports the cached 3D geometry to JCP.
See `md/jcp_shifted_reproduction.md` for the complete sequence and required
saved inputs. These are plotting-only commands and do not run experiments.
