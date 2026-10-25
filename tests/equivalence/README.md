# Equivalence tests

These tests compare this code with the research code that produced the paper and
with the frozen paper models.  They are skipped unless the environment provides:

| Variable | Contents | Used by |
|---|---|---|
| `DMSRB_ORACLE` | a checkout of the research code (`code/`) with the frozen run outputs (`outputs/`) | half-space, scattering and Appendix A tests |
| `DMSRB_V7` | the source tree that ran Experiment 3 on the cluster (`code/`) | `test_waveguide.m` |

| Test | Level | What is compared |
|---|---|---|
| `test_halfspace_assembly` | L0, L1 | matrices, source, partitions (bitwise) |
| `test_halfspace_training` | L3 | trainer vs original trainer (bitwise); ranks of the paper models |
| `test_halfspace_online` | L4 | paper models solved by this code: ROM fields, SL step counts of all 23 queries |
| `test_halfspace_controls` | L5 | NT-ROM and SS-ROM rebuilt from scratch |
| `test_ecc` | L0, L3, L4 | test systems (bitwise), trainer vs original (bitwise), paper ranks, SL steps |
| `test_waveguide` | L6 | full 3D pipeline vs the cluster code at omega = 6 and 12 |
| `test_appendixA` | — | multigrid component contractions and roughness |

```matlab
setenv('DMSRB_ORACLE', '/path/to/research'); setenv('DMSRB_V7', '/path/to/v7');
startup; addpath('tests/equivalence'); results = runtests('tests/equivalence')
```
