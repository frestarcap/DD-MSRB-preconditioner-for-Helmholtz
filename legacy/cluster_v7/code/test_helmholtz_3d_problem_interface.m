function result = test_helmholtz_3d_problem_interface()
%TEST_HELMHOLTZ_3D_PROBLEM_INTERFACE Smoke-test assembly and callbacks.

    cfg = struct('Nphys', [9, 8, 7], 'npml', [2, 2, 2], ...
        'omega', 8, 'material_model', 'sphere', ...
        'sphere_radius', 0.2, 'source_location', [0.2, 0.5, 0.5]);
    problem = helmholtz_3d_problem_interface(cfg);
    [A, b, meta] = problem.assemble(0.85);

    rng(1);
    u = randn(problem.N, 1) + 1i * randn(problem.N, 1);
    apply_error = norm(A * u - problem.apply(0.85, u)) / norm(A * u);
    assert(apply_error < 20 * eps, '3D assembled/apply mismatch: %.3e.', apply_error);

    Ashift = problem.assemble_shifted(0.85, 0.2);
    assert(isequal(size(Ashift), [problem.N, problem.N]));
    assert(nnz(A) <= 7 * problem.N);
    assert(nnz(b) == 1);
    assert(meta.min_speed == 0.85 && meta.max_speed == 1.0);

    partition = problem.partition(0.85);
    covered = sort(vertcat(partition.indices{:}));
    assert(isequal(covered, (1:problem.N).'));

    u_direct = A \ b;
    direct_residual = norm(A * u_direct - b) / norm(b);
    assert(direct_residual < 1e-10);

    gaussian_models = {'gaussian_lens', 'gaussian_waveguide', 'affine_halfspace'};
    affine_errors = zeros(numel(gaussian_models), 1);
    for k = 1:numel(gaussian_models)
        gaussian_cfg = struct('Nphys', [8, 7, 6], 'npml', [2, 2, 2], ...
            'omega', 7, 'material_model', gaussian_models{k}, ...
            'source_model', 'gaussian', 'source_location', [0.2, 0.5, 0.5]);
        gaussian_problem = helmholtz_3d_problem_interface(gaussian_cfg);
        mu = 0.35;
        [Ag, bg] = gaussian_problem.assemble(mu);
        assert(gaussian_problem.affine.is_affine);
        assert(nnz(bg) > 1);
        ug = randn(gaussian_problem.N, 1) + 1i * randn(gaussian_problem.N, 1);
        affine_action = gaussian_problem.K_pml * ug ...
            + gaussian_problem.affine.mass_base * ug ...
            + mu * gaussian_problem.affine.mass_parameter_diagonal .* ug;
        affine_errors(k) = norm(Ag * ug - affine_action) / norm(Ag * ug);
        assert(affine_errors(k) < 20 * eps, ...
            '%s affine action mismatch: %.3e.', gaussian_models{k}, affine_errors(k));
    end
    % The affine parameter is squared-slowness contrast, not right speed.
    % It must reproduce the existing halfspace operator after the mapping.
    affine_cfg=gaussian_cfg;
    affine_cfg.material_model='affine_halfspace';
    pa=helmholtz_3d_problem_interface(affine_cfg);
    affine_cfg.material_model='halfspace';
    ps=helmholtz_3d_problem_interface(affine_cfg);
    Aa=pa.assemble(.35); As=ps.assemble(1/sqrt(1+.35));
    assert(norm(Aa-As,'fro')/norm(As,'fro')<1e-14);

    result = struct('N', problem.N, 'nnz', nnz(A), ...
        'apply_relative_error', apply_error, ...
        'direct_relative_residual', direct_residual, ...
        'partition_labels', partition.unique_labels, ...
        'gaussian_affine_errors', affine_errors);
    fprintf('3D interface N=%d, nnz=%d, apply err %.3e, direct residual %.3e\n', ...
        result.N, result.nnz, result.apply_relative_error, result.direct_relative_residual);
end
