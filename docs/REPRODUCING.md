# Reproducing the paper

Start MATLAB in the repository root and run `startup` first.  Every stage writes to
`results/` (set `DMSRB_RESULTS` to use another folder); `export` stages convert the
results to the schema of `paper_data/`, so the same figure code draws both:

```bash
python figures/make_all.py                                  # from paper_data/
python figures/make_all.py --data results/paper --out build/rerun
matlab -batch "addpath('figures'); waveguide_geometry()"   # two 3D renderings
```

All 2D computations use one computational thread (`maxNumCompThreads(1)`), as in the
paper.  Times below are for an Apple M4 Max; they scale with single-core speed.

## Experiment 1 — half-space (`experiments/exp1_halfspace`)

```matlab
run_exp1('train')        % 17 models, ~15 min (omega = 214: N = 570,025)
run_exp1('main')         % ~5 min
run_exp1('coverage')     % ~2 min
run_exp1('sensitivity')  % ~40 min (omega = 214 dominates)
run_exp1('offline')      % ~1 min: offline column of Table 2 (median of three trainings)
run_exp1('export')
```

| Paper item | Stage | Data file (`paper_data/exp1/`) | Drawn by |
|---|---|---|---|
| Fig. 1 (fields, partition) | `main`, `sensitivity` | `field_omega53.5.mat`, `field_omega214.mat` | `figures/exp1.py: fields` |
| Fig. 2 (spectra, residual histories) | `train`, `main` | `field_omega53.5.mat`, `residual_histories_query10.csv` | `fields`, `residuals` |
| Table 1, Fig. 3 (online cost, storage) | `main`, `train` | `online_timings.csv`, `storage.csv` | `table_cost`, `cost` |
| Table 2 (training coverage) | `coverage`, `offline` | `training_coverage.csv` | `table_coverage` |
| Table 3, partition figure | `sensitivity` | `partitions.csv` | `table_partitions` |
| Fig. 5 (frequency study) | `sensitivity` | `frequency_storage.csv`, `frequency_timings.csv` | `frequency` |

## Experiment 2 — elliptic-obstacle scattering (`experiments/exp2_ecc`)

```matlab
run_exp2('train')   % ~1 min
run_exp2('time')    % ~3 min
run_exp2('offline') % ~1 min: offline times (median of three trainings)
run_exp2('export')
```

| Paper item | Data file (`paper_data/exp2/`) | Drawn by |
|---|---|---|
| Geometry/partition and field figure | `field.mat` | `figures/exp2.py: field_and_partition` |
| Spectra and residual histories | `tail_spectra.csv`, `rank_storage.csv`, `residual_histories_query1.csv` | `spectra`, `residuals` |
| Cost table, time and storage panels, offline times | `online_timings.csv`, `storage.csv` | `table_cost`, `cost` |

The 96 training and 22 test aspect ratios are frozen in `exp2_params.json`
(`dmsrb.ecc.admissible` confirms that every geometry has an unambiguous ghost stencil).

## Experiment 3 and Appendix A — 3D waveguide

See `experiments/exp3_waveguide/README.md`.  The stages are `run_exp3('train', omega)`,
`run_exp3('online', omega)` for omega = 24, 36, 48, 60, `run_exp3('validation', 48)`,
`run_appA('probe', omega)`, and finally `run_exp3('export')` and `run_appA('export')`.

| Paper item | Data file (`paper_data/exp3/`, `appendixA/`) | Drawn by |
|---|---|---|
| Geometry and partition renderings | `waveguide_geometry.mat` | `figures/waveguide_geometry.m` |
| Field and residual histories at omega = 48 | `field_omega48_mu0.317.mat`, `residual_histories_omega48_mu0.317.csv` | `figures/exp3.py: field`, `residuals` |
| Ten-query table and time panel | `validation_omega48.csv` | `table_validation`, `validation` |
| Storage panel | `storage_omega48.csv` | `storage` |
| Frequency table and panels | `frequency_queries.csv`, `frequency_storage.csv`, `offline_times.csv` | `table_frequency`, `frequency_time` |
| Appendix A figures | `residual_components.csv`, `residual_slices_omega48.mat` | `figures/appendixA.py` |

## Checks

```matlab
runtests('tests/unit'); runtests('tests/smoke');
```
```bash
python -m pytest tests/python      # 25 figures pixel-identical, six tables identical
python tools/compare_to_paper.py   # re-run of Experiments 1-2 versus the paper
```
