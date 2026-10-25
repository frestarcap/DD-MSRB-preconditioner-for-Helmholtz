function results = run_helm_direct_shifted_laplacian_scaling_pilot(mode, user_cfg)
%RUN_HELM_DIRECT_SHIFTED_LAPLACIAN_SCALING_PILOT Half-space scaling pilot.
%
% The physical mesh and frequency are refined together, keeping the points
% per shortest wavelength approximately constant. At one representative
% material parameter, the pilot separates operator assembly, sparse-direct
% factorization/solve, and shifted-Laplacian setup/GMRES costs. The shifted
% inverse action can use sparse LU or ILUTP for a scalable pilot.

    if nargin < 1 || isempty(mode)
        mode = "pilot";
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end

    mode = lower(string(mode));
    out_dir = helm_output_path('outputs', 'helm_direct_shifted_scaling_pilot');
    if mode == "redraw"
        mat_file = fullfile(out_dir, 'helm_direct_shifted_scaling_pilot.mat');
        loaded = load(mat_file, 'results');
        results = loaded.results;
        fig_file = fullfile(out_dir, 'helm_direct_shifted_scaling_pilot.png');
        plot_results(results.table, fig_file);
        fprintf('Redrew %s\n', fig_file);
        return;
    end

    cfg = default_cfg(mode);
    cfg = merge_structs(cfg, user_cfg);
    validate_cfg(cfg);

    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    fprintf('=== Half-space direct/shifted-Laplacian scaling pilot: %s ===\n', mode);
    fprintf('scale factors = %s, c = %.4f, tol = %.1e\n', ...
        mat2str(cfg.scale_factors), cfg.c_probe, cfg.tol);

    cases = repmat(empty_case(), numel(cfg.scale_factors), 1);
    selected_beta = NaN;

    for iscale = 1:numel(cfg.scale_factors)
        scale = cfg.scale_factors(iscale);
        nphys = round((cfg.base_nphys + 1) * scale) - 1;
        omega = cfg.base_omega * scale;

        setup_cfg = struct();
        setup_cfg.setup_only = true;
        setup_cfg.material_model = 'halfspace';
        setup_cfg.Nphys = nphys;
        setup_cfg.omega = omega;
        setup_cfg.train_params = [cfg.c_probe - 0.01, cfg.c_probe + 0.01];
        setup_cfg.test_params = cfg.c_probe;
        setup_cfg.parallel_train = false;
        setup_cfg.parallel_test = false;
        setup_cfg.do_plots = false;
        setup_cfg.save_plots = false;
        setup_cfg.save_benchmark = false;

        fprintf('\n--- scale %.2f: Nphys=%d, omega=%.3f ---\n', scale, nphys, omega);
        t_assembly = tic;
        setup = helm_waveholtz_correction_rom(setup_cfg);
        assembly_time = toc(t_assembly);
        prob = setup.problem;
        rhs = setup.rhs;

        alpha = (1 / cfg.c_probe^2) - 1;
        A = prob.A_base + alpha * prob.A_right;
        mass_part = (prob.omega ^ 2) * speye(prob.Ntot) + alpha * prob.A_right;
        ppw = 2 * pi * cfg.c_probe / (prob.omega * min(prob.hx, prob.hy));

        fprintf('N=%d, nnz(A)=%d, ppw(shortest)=%.2f, assembly=%.3fs\n', ...
            prob.Ntot, nnz(A), ppw, assembly_time);

        direct = benchmark_direct(A, rhs, cfg.solve_repeats);
        fprintf(['direct: factor %.3fs, solve median %.3fs, total %.3fs, ' ...
            'residual %.3e\n'], direct.factor_time, direct.solve_time_median, ...
            direct.total_time, direct.relres);

        if iscale == 1 || ~cfg.tune_beta_on_first_scale
            betas = cfg.beta_candidates;
        else
            betas = selected_beta;
        end

        shifted = repmat(empty_shifted(), numel(betas), 1);
        for ibeta = 1:numel(betas)
            shifted(ibeta) = benchmark_shifted(A, mass_part, rhs, betas(ibeta), cfg);
            fprintf(['shift beta %.3f: factor %.3fs, GMRES %.3fs, total %.3fs, ' ...
                'iters %d, residual %.3e, flag %d\n'], ...
                betas(ibeta), shifted(ibeta).factor_time, shifted(ibeta).solve_time, ...
                shifted(ibeta).total_time, shifted(ibeta).iter_count, ...
                shifted(ibeta).relres, shifted(ibeta).flag);
        end

        if iscale == 1 && cfg.tune_beta_on_first_scale
            selected_beta = choose_beta(shifted);
            fprintf('selected beta for larger scales: %.3f\n', selected_beta);
        end

        one = empty_case();
        one.scale = scale;
        one.nphys = nphys;
        one.omega = omega;
        one.npml = prob.npml;
        one.N = prob.Ntot;
        one.nnz_A = nnz(A);
        one.ppw_shortest = ppw;
        one.assembly_time = assembly_time;
        one.direct = direct;
        one.shifted = shifted;
        cases(iscale) = one;

        clear A mass_part prob rhs setup
    end

    results = struct();
    results.mode = char(mode);
    results.cfg = cfg;
    results.selected_beta = selected_beta;
    results.cases = cases;
    results.table = make_results_table(cases);
    results.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));
    results.matlab_version = version;

    tag = cfg.output_tag;
    mat_file = fullfile(out_dir, ['helm_direct_shifted_scaling_' tag '.mat']);
    csv_file = fullfile(out_dir, ['helm_direct_shifted_scaling_' tag '.csv']);
    fig_file = fullfile(out_dir, ['helm_direct_shifted_scaling_' tag '.png']);
    results.output_files = struct('mat', mat_file, 'csv', csv_file, 'figure', fig_file);
    save(mat_file, 'results', '-v7.3');
    writetable(results.table, csv_file);
    plot_results(results.table, fig_file);

    fprintf('\nSaved %s\nSaved %s\nSaved %s\n', mat_file, csv_file, fig_file);
    disp(results.table);
end

function cfg = default_cfg(mode)
    cfg = struct();
    cfg.base_nphys = 125;
    cfg.base_omega = 53.5;
    cfg.c_probe = 0.875;
    cfg.beta_candidates = [0.1, 0.2, 0.4];
    cfg.tune_beta_on_first_scale = true;
    cfg.restart = 30;
    cfg.maxit = 6;
    cfg.tol = 1e-8;
    cfg.reorthogonalize = false;
    cfg.solve_repeats = 3;
    cfg.shifted_solver = 'lu';
    cfg.ilu_type = 'ilutp';
    cfg.ilu_droptol = 1e-2;
    cfg.ilu_thresh = 1.0;
    cfg.ilu_milu = 'off';

    switch mode
        case "smoke"
            cfg.scale_factors = 1;
            cfg.beta_candidates = [0.1, 0.2];
            cfg.maxit = 3;
            cfg.solve_repeats = 1;
        case "pilot"
            cfg.scale_factors = [1, 2, 3, 4];
        otherwise
            error('mode must be smoke or pilot.');
    end
    cfg.output_tag = char(mode);
end

function out = merge_structs(base, override)
    out = base;
    names = fieldnames(override);
    for k = 1:numel(names)
        out.(names{k}) = override.(names{k});
    end
end

function validate_cfg(cfg)
    if any(~isfinite(cfg.scale_factors)) || any(cfg.scale_factors <= 0)
        error('scale_factors must be finite and positive.');
    end
    if any(~isfinite(cfg.beta_candidates)) || any(cfg.beta_candidates <= 0)
        error('beta_candidates must be finite and positive.');
    end
    if ~(isscalar(cfg.c_probe) && isfinite(cfg.c_probe) && cfg.c_probe > 0)
        error('c_probe must be a positive finite scalar.');
    end
end

function out = benchmark_direct(A, rhs, solve_repeats)
    out = struct();
    t_factor = tic;
    D = decomposition(A, 'lu');
    out.factor_time = toc(t_factor);

    solve_times = zeros(solve_repeats, 1);
    x = zeros(size(rhs));
    for k = 1:solve_repeats
        t_solve = tic;
        x = D \ rhs;
        solve_times(k) = toc(t_solve);
    end
    out.solve_times = solve_times;
    out.solve_time_median = median(solve_times);
    out.total_time = out.factor_time + out.solve_time_median;
    out.relres = norm(rhs - A * x) / max(norm(rhs), eps);
    factor_info = whos('D');
    out.reported_factor_bytes = factor_info.bytes;
    clear D
end

function out = benchmark_shifted(A, mass_part, rhs, beta, cfg)
    out = empty_shifted();
    out.beta = beta;
    Ashift = A + 1i * beta * mass_part;

    try
        prec_opts = struct('solver', cfg.shifted_solver, ...
            'ilu_type', cfg.ilu_type, 'ilu_droptol', cfg.ilu_droptol, ...
            'ilu_thresh', cfg.ilu_thresh, 'ilu_milu', cfg.ilu_milu);
        [Msolve, setup_info] = build_shifted_laplacian_preconditioner(Ashift, prec_opts);
        out.solver = setup_info.solver;
        out.factor_time = setup_info.setup_time;
        out.reported_factor_bytes = setup_info.factor_bytes;

        opts = struct();
        opts.restart = cfg.restart;
        opts.maxit = cfg.maxit;
        opts.tol = cfg.tol;
        opts.reorthogonalize = cfg.reorthogonalize;
        opts.verbose = false;

        [x, info] = shifted_laplacian_gmres(@(v) A * v, Msolve, rhs, [], opts);
        out.solve_time = info.total_time;
        out.total_time = out.factor_time + out.solve_time;
        out.iter_count = info.iter_count;
        out.relres = norm(rhs - A * x) / max(norm(rhs), eps);
        out.flag = info.flag;
        out.stop_reason = info.stop_reason;
        clear Msolve
    catch ME
        out.stop_reason = ['error: ' ME.message];
        warning('Shifted-Laplacian beta %.3f failed: %s', beta, ME.message);
    end
end

function beta = choose_beta(shifted)
    converged = [shifted.flag] == 0;
    if any(converged)
        candidates = find(converged);
        [~, idx] = min([shifted(candidates).total_time]);
        beta = shifted(candidates(idx)).beta;
    else
        [~, idx] = min([shifted.relres]);
        beta = shifted(idx).beta;
    end
end

function tbl = make_results_table(cases)
    nrows = sum(arrayfun(@(c) 1 + numel(c.shifted), cases));
    scale = zeros(nrows, 1);
    nphys = zeros(nrows, 1);
    omega = zeros(nrows, 1);
    N = zeros(nrows, 1);
    nnz_A = zeros(nrows, 1);
    ppw = zeros(nrows, 1);
    method = strings(nrows, 1);
    beta = NaN(nrows, 1);
    assembly_s = zeros(nrows, 1);
    factor_s = zeros(nrows, 1);
    solve_s = zeros(nrows, 1);
    total_s = zeros(nrows, 1);
    iterations = zeros(nrows, 1);
    relres = zeros(nrows, 1);
    flag = zeros(nrows, 1);

    row = 0;
    for icase = 1:numel(cases)
        c = cases(icase);
        row = row + 1;
        [scale(row), nphys(row), omega(row), N(row), nnz_A(row), ppw(row), assembly_s(row)] = ...
            common_values(c);
        method(row) = "direct";
        factor_s(row) = c.direct.factor_time;
        solve_s(row) = c.direct.solve_time_median;
        total_s(row) = c.direct.total_time;
        relres(row) = c.direct.relres;

        for j = 1:numel(c.shifted)
            row = row + 1;
            [scale(row), nphys(row), omega(row), N(row), nnz_A(row), ppw(row), assembly_s(row)] = ...
                common_values(c);
            method(row) = "shifted_laplacian_" + string(c.shifted(j).solver);
            beta(row) = c.shifted(j).beta;
            factor_s(row) = c.shifted(j).factor_time;
            solve_s(row) = c.shifted(j).solve_time;
            total_s(row) = c.shifted(j).total_time;
            iterations(row) = c.shifted(j).iter_count;
            relres(row) = c.shifted(j).relres;
            flag(row) = c.shifted(j).flag;
        end
    end

    tbl = table(scale, nphys, omega, N, nnz_A, ppw, method, beta, assembly_s, ...
        factor_s, solve_s, total_s, iterations, relres, flag);
end

function [scale, nphys, omega, N, nnz_A, ppw, assembly_s] = common_values(c)
    scale = c.scale;
    nphys = c.nphys;
    omega = c.omega;
    N = c.N;
    nnz_A = c.nnz_A;
    ppw = c.ppw_shortest;
    assembly_s = c.assembly_time;
end

function plot_results(tbl, fig_file)
    direct = tbl.method == "direct";
    shifted = startsWith(tbl.method, "shifted_laplacian_") & tbl.flag == 0;
    shifted_selected = false(height(tbl), 1);
    shifted_sizes = unique(tbl.N(shifted));
    for k = 1:numel(shifted_sizes)
        rows = find(shifted & tbl.N == shifted_sizes(k));
        [~, best] = min(tbl.total_s(rows));
        shifted_selected(rows(best)) = true;
    end

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 760, 520]);
    ax = axes(fig);
    loglog(ax, tbl.N(direct), tbl.total_s(direct), '-o', 'LineWidth', 1.5, ...
        'MarkerSize', 7, 'DisplayName', 'Sparse direct');
    hold(ax, 'on');
    if any(shifted_selected)
        loglog(ax, tbl.N(shifted_selected), tbl.total_s(shifted_selected), '-s', 'LineWidth', 1.5, ...
            'MarkerSize', 7, 'DisplayName', 'Shifted Laplacian + GMRES');
    end
    xlabel(ax, 'total PML-domain unknowns');
    ylabel(ax, 'factorization + solve time (s)');
    title(ax, 'Half-space fixed-PPW scaling pilot');
    grid(ax, 'on');
    legend(ax, 'Location', 'northwest');
    exportgraphics(fig, fig_file, 'Resolution', 180);
    close(fig);
end

function out = empty_case()
    out = struct('scale', NaN, 'nphys', NaN, 'omega', NaN, 'npml', NaN, ...
        'N', NaN, 'nnz_A', NaN, 'ppw_shortest', NaN, 'assembly_time', NaN, ...
        'direct', struct(), 'shifted', struct([]));
end

function out = empty_shifted()
    out = struct('beta', NaN, 'solver', '', 'factor_time', NaN, 'solve_time', NaN, ...
        'total_time', NaN, 'iter_count', 0, 'relres', Inf, 'flag', 1, ...
        'stop_reason', '', 'reported_factor_bytes', NaN);
end
