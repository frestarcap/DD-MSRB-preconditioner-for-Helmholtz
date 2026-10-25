# Paper data

The numbers behind every figure and table of the paper, exported from the runs
reported there. `figures/make_all.py` draws all figures and table bodies from this
directory; a fresh run of the experiments writes the same files under `results/paper/`.

Times are in seconds, storage in MiB (2^20 bytes), residuals are true relative residuals
‖b − Au‖₂/‖b‖₂ on the full PML-extended grid. `rho_rom` is the residual after the
ROM-preconditioned iterations, before any shifted-Laplacian (SL) step. Method keys:
`global` (global MSRB), `dd` (DD-MSRB), `nt` (NT-ROM), `ss` (SS-ROM),
`shifted` (SL baseline, k = 0), `direct` (sparse direct solver), `plain_gmres`.
`MANIFEST.json` lists the SHA-256 of every file.  `exp1/online_timings.csv`,
`exp1/residual_histories_query10.csv` and the offline times in `exp2/storage.csv` come
from the re-run with the rewritten code (2026-10-24), as in the paper; all other files
come from the runs of the original code (`legacy/`).

## exp1 — half-space

| File | Content | Used in |
|---|---|---|
| `field_omega53.5.mat` | `U` (solution, ω = 53.5, c_right = 0.8622), grid `x`, `y`, subdomain `labels`, singular values `spectrum_1..3`, `source` | Fig. 1, Fig. 2 left |
| `field_omega214.mat` | `U`, `x`, `y`, `source`, `rho` at ω = 214 | Fig. 1 |
| `online_timings.csv` | one row per method, query (23), repetition (5): `total_s`, `ROM_s`, `setup_s`, `solve_s`, `sl_steps`, `rho_rom`, `true_residual`, `relative_error` | Table 1, Fig. 3 left |
| `residual_histories_query10.csv` | `method`, `step`, `relative_residual` (ROM steps followed by SL steps) | Fig. 2 right |
| `storage.csv` | basis, payload and MATLAB object size | Fig. 3 right |
| `training_coverage.csv` | n_s = 6, 12, 24: ranks, payload, offline time, mean online time, max ρ_ROM (nine queries) | Table 2 |
| `partitions.csv` | four partitions and global: ranks, basis/operator/index/payload MiB, mean time, max ρ_ROM (five queries) | Table 3 |
| `frequency_storage.csv` | payload per ω ∈ {53.5, 80.25, 107, 214} | Fig. 5 left |
| `frequency_timings.csv` | mean online time per ω, method and query (five repetitions) | Fig. 5 right |

## exp2 — elliptic-obstacle scattering

| File | Content | Used in |
|---|---|---|
| `field.mat` | DD-MSRB solution `U`, grid `x`, `y`, obstacle `aspect_ratio`, `rho` | field and partition panels |
| `tail_spectra.csv` | relative Frobenius tail per iteration and retained rank (global snapshots) | spectra panel |
| `residual_histories_query1.csv` | residual histories of the representative geometry | residual panel |
| `online_timings.csv` | one row per method, geometry (22), repetition (5) | cost table, time panel |
| `rank_storage.csv` | per-iteration global dimension, DD local ranks (near field, far field, PML), basis entries | spectra markers, text |
| `storage.csv` | basis, payload and MATLAB object size | storage panel |

## exp3 — 3D Gaussian waveguide

| File | Content | Used in |
|---|---|---|
| `frequency_queries.csv` | ω ∈ {24, 36, 48, 60}, three common queries μ ∈ {0.237, 0.317, 0.483}: online time, setup/solve/ROM time, SL steps, ρ_ROM, true residual | frequency table and time panel |
| `residual_histories_omega48_mu0.317.csv` | ROM + SL residual histories | residual panel |
| `field_omega48_mu0.317.mat` | real part of the DD-MSRB solution on z = 0.5 | field panel |
| `validation_omega48.csv` | ten queries at ω = 48 (three common queries with three repetitions, seven added with one): mean/SD time, SL steps, payload, for global (k = 3), global (k = 2), DD (k = 3), SL baseline | validation table and panel |
| `storage_omega48.csv`, `iteration_storage_omega48.csv` | basis, payload, object size; basis per iteration | storage panel |
| `frequency_storage.csv` | payload per ω | frequency storage panel |
| `offline_times.csv` | aggregate time of all training solves per ω | frequency table |
| `waveguide_geometry.mat` | geometry and subdomain classes for the two 3D renderings | `figures/waveguide_geometry.m` |

At ω = 48 the paper reports the repeated timings of `validation_omega48.csv` for the
three common queries; the single timings in `frequency_queries.csv` are kept for reference.

## appendixA — residual composition (ω = 48 and frequency sweep)

| File | Content | Used in |
|---|---|---|
| `residual_slices_omega48.mat` | normalized \|r\| on z = 0.5 before (`*_r0`) and after (`*_rafter`) the ROM steps, grid, spacing, subdomain labels | slice panels |
| `residual_components.csv` | per ω, μ, method: ρ_ROM, transverse roughness, optimal contraction of the MG pre-smoother, coarse correction and full cycle | roughness and response panels |
