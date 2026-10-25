# Frozen 3D Helmholtz / trajectory-aware ROM source (v7)

This is the source actually used by <cluster> jobs 43864385, 43864386 and
43864387 and the matching-frequency control 43868638, together with the selected solver profile. The MATLAB/C/Slurm file
hashes match source_manifest.json. It is a research implementation with a
shifted-MG completion solver; the separate 3D WaveHoltz probe has a different
validation scope. Numerical results are stored separately from this archive.

On the configured <cluster> account, extract this archive and run from helm3d_v7:

    mkdir -p logs
    sbatch scripts/scnet_3d.slurm full 48 8 4 halfspace profiles/beta2_coarse12000_restart40.json
    sbatch scripts/scnet_3d.slurm full 36 8 4 waveguide profiles/beta2_coarse12000_restart40.json
    sbatch scripts/scnet_3d.slurm full 24 8 4 waveguide profiles/beta2_coarse12000_restart40.json
    sbatch scripts/scnet_3d.slurm full 48 8 4 waveguide profiles/beta2_coarse12000_restart40.json

These example submissions are independent. To reproduce the study's serial
resource usage, submit each after the preceding job finishes, or use Slurm
job dependencies. Each requests 64 CPUs / 192 GiB and a strict 8-hour limit.
The launch script uses the configured account-specific MATLAB R2024a paths,
compiles the CGS2 BLAS MEX on a compute node, and checks it before solving.
No compiled binary, credential, numerical MAT file or login-node computation
is included. On another account/cluster, adapt the module and MATLAB paths.

Eight workers each use four computational threads during training. Snapshots
must pass 1e-7 true residual; the three original online queries must pass 1e-6.
The profile sets beta=2, snapshot/online restart=40, maximum cycles=40, and
the direct coarse-grid size threshold to 12000. Three ROM layers and all eight
training parameters remain fixed. Original default settings and failed
experimental points remain in the separate study audit and data archives.

Accepted snapshots and completed layers are checkpointed under outputs/.
A repeat with identical settings resumes accepted work after checking input
identity and true residuals. A resumed run does not measure fresh offline cost.


The controlled v7 frequency series is pinned to node g02r1n09 (64 physical
CPUs, Hygon C86 7285, lscpu maximum 2.0 GHz). Add
`--nodelist=g02r1n09` to sbatch to request the same node. Other nodes in this
partition include 7285H / 2.5 GHz hardware; record the actual CPU type when
reproducing timings elsewhere. Original runs on other hardware are retained
as separate configurations. Each method speedup uses same-node timings.

Relevant numerical checks (inside MATLAB with addpath('code')):

    test_helm_3d_hpc
    test_helm_3d_checkpoint
    test_helm_3d_failure_collection

The Slurm launch compiles and validates the MEX; local tests can use MATLAB
CGS2 if a compiler is unavailable. The complete reference results, scientific
figures and resource audit are delivered separately as results_audit_zh.md.
