# Method notes

How the paper's definitions map to the code.

## Training (Algorithm 1, `dmsrb.train`)

For training parameters μ_1, …, μ_ns and k reduced spaces:

1. q_{1,i} = b_i/‖b_i‖ and the exact inverse action z_{1,i} = u_i/‖b_i‖ with u_i = A_i \ b_i.
2. The snapshots [z_{1,1}, …, z_{1,ns}] are compressed into the first space Ψ_1.
3. For j = 1, …, k−1 and every sample: ẑ = P_j(μ_i) q_{j,i} with the *compressed*
   layer j, w = A_i ẑ is orthogonalized against q_{1,i}, …, q_{j,i} (two-pass
   Gram–Schmidt, `dmsrb.arnoldi_step`), and the inverse action of the new vector,
   z_{j+1,i} = (ẑ − Σ_l h_l z_{l,i}) / h_{j+1}, follows from A_i z_{l,i} = q_{l,i}
   without a solve.  It is accepted if ‖A_i z − q‖ ≤ 10⁻¹⁰ and otherwise replaced by
   A_i \ q (`dmsrb.verified_snapshot`).  After all samples are advanced the snapshots
   are compressed once into Ψ_{j+1}.

Experiment 3 (`experiments/exp3_waveguide/train_waveguide.m`) computes every
snapshot by an SL-MG GMRES solve to 10⁻⁷ instead of the recurrence, shares the first
iteration's snapshots between the global and the DD model, and advances the
trajectories with one Gram–Schmidt pass.

## Compression (`dmsrb.compress`, `dmsrb.greedy_ranks`)

* Global MSRB: the POD basis of the smallest rank r with
  ‖σ_{r+1:}‖ ≤ η‖σ‖ (relative Frobenius tail).
* DD-MSRB: per subdomain b the SVD of the snapshot rows, local ranks from zero, and the
  next mode taken from argmax_b σ²_{b,r_b+1}/n_b until the remaining energy over all
  subdomains is at most η² times the total snapshot energy; ties go to the lower
  subdomain number.  The reduced Galerkin matrix couples all subdomains
  (`dmsrb.galerkin`; `dmsrb.waveguide.reduce` in 3D).

## Online solve (Algorithm 2, `dmsrb.solve`)

ROM-preconditioned FGMRES from u_0 = 0 (`dmsrb.prefix`), stopping early when the true
relative residual ‖b − Au‖/‖b‖ reaches τ or the Arnoldi process breaks down; then, only
if ρ_ROM > τ, the SL preconditioner is built (ILU of A + iβ M in 2D, a geometric
multigrid V-cycle in 3D) and restarted GMRES continues from the ROM iterate until the
true residual meets τ.  Affine problems assemble the reduced matrix from precomputed
terms Σ θ_q(μ) Ψᴴ A_q Ψ; the scattering problem assembles Ψᴴ A(μ) Ψ online.

## Comparison methods (Experiment 1)

* **NT-ROM** (`dmsrb.train_nt`): the same compression, but the snapshots are the exact
  inverse actions A_i \ q_{j,i} of the *unpreconditioned* Arnoldi vectors
  (q_{j+1} ∝ A q_j), and the ranks equal the trajectory-aware global ranks.
* **SS-ROM** (`dmsrb.solution_pod`): one POD space of the training solutions
  (η = 10⁻⁵, seven modes), reused as the reduced space of all k ROM steps.
* **SL baseline**: the hybrid solve without ROM steps (k = 0); **sparse direct**: `A\b`.

## Storage (`dmsrb.storage`)

* basis = 16 bytes per complex basis entry;
* payload = basis + affine reduced operators (16 bytes per entry, two terms per layer)
  + one index map of 8 bytes per grid node for a DD model, shared by all layers;
* MATLAB object = `whos` size of the deployed model (`dmsrb.deploy`), in which every
  layer keeps its own subdomain indices.

## Timing (`dmsrb.benchmark`)

Per query, one untimed warm-up of every method, then five timed repetitions with the
method order circularly shifted by query and repetition.  A timed solve includes the
ROM steps, the true-residual check, the SL matrix and preconditioner construction when
needed, the SL steps and the final residual; it excludes the assembly of A(μ) and b(μ)
and model loading.  2D runs use one computational thread.
