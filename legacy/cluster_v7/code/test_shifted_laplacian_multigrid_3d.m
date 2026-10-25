function result = test_shifted_laplacian_multigrid_3d()
%TEST_SHIFTED_LAPLACIAN_MULTIGRID_3D Smoke-test the 3D V-cycle baseline.

    cfg = struct('Nphys', [15, 15, 15], 'npml', [3, 3, 3], ...
        'omega', 8, 'material_model', 'gaussian_lens', ...
        'source_model', 'gaussian');
    problem = helmholtz_3d_problem_interface(cfg);
    mu = 0.3;
    beta = 0.4;
    [A, b] = problem.assemble(mu);
    Ashift = problem.assemble_shifted(mu, beta);

    prec_cfg = struct('solver', 'mg', 'grid_size', problem.n, ...
        'mg_coarse_max_dofs', 200);
    [Msolve, setup] = build_shifted_laplacian_preconditioner(Ashift, prec_cfg);
    opts = struct('restart', 20, 'maxit', 8, 'tol', 1e-6);
    [x, info] = shifted_laplacian_gmres(@(v) A * v, Msolve, b, [], opts);
    true_relres = norm(b - A * x) / norm(b);

    assert(setup.multigrid.level_count >= 2);
    assert(info.flag == 0, '3D shifted-Laplacian MG did not converge.');
    assert(true_relres <= 1.1 * opts.tol, ...
        'Unexpected true residual %.3e.', true_relres);

    result = struct('N', problem.N, 'levels', setup.multigrid.level_count, ...
        'iterations', info.iter_count, 'relative_residual', true_relres, ...
        'setup_time', setup.setup_time, 'solve_time', info.total_time);
    fprintf(['3D shifted-Laplacian MG N=%d, levels=%d, iterations=%d, ' ...
        'residual=%.3e\n'], result.N, result.levels, result.iterations, ...
        result.relative_residual);
end
