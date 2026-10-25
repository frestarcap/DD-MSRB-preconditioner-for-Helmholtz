function results = waveholtz_piecewise_scattering_standalone(cfg)
    % Standalone WaveHoltz "scattering-like" test for left-source / right-speed parameterization.
    %
    % Scenario:
    %   - Source is on the left half of the physical domain.
    %   - Wave speed is piecewise:
    %         c(x)=1.0                 for x <= interface_x
    %         c(x)=c_right (parameter) for x >  interface_x
    %   - Solve total field via WaveHoltz, and optionally compare against:
    %         (1) homogeneous reference c_right=1.0  -> scattering-like field
    %         (2) direct Helmholtz solve on same grid.
    %
    % Usage:
    %   results = waveholtz_piecewise_scattering_standalone();
    %   results = waveholtz_piecewise_scattering_standalone(struct( ...
    %       'c_right_values', [0.80 0.86 0.92], 'do_plots', true));

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);
    validate_c_values(cfg.c_right_values);

    prob = build_problem(cfg);
    rhs = build_rhs(prob, cfg);

    fprintf('=== waveholtz_piecewise_scattering_standalone ===\n');
    fprintf('Nphys=%d x %d, npml=%d, total interior=%d x %d\n', ...
        cfg.Nphys_x, cfg.Nphys_y, cfg.npml, prob.nx, prob.ny);
    fprintf('omega(helm)=%.6f, omega(ts)=%.6f, dt=%.3e, nt=%d\n', prob.omega, prob.omega_ts, prob.dt, prob.nt);
    fprintf('material model = %s\n', cfg.material_model);
    if strcmp(cfg.material_model, 'halfspace')
        fprintf('interface x = %.4f\n', cfg.interface_x);
    else
        fprintf('parameterized material cells = %d / %d\n', prob.material_meta.num_mask_nodes, prob.Ntot);
        fprintf('pc rows=%d, lattice a=%.4f, radius=%.4f, columns=%d, rod shape=%s\n', ...
            prob.material_meta.num_rows, prob.material_meta.lattice_a, ...
            prob.material_meta.radius, prob.material_meta.num_cols, prob.material_meta.rod_shape);
    end
    fprintf('source type = %s\n', cfg.source_type);
    fprintf('source phys grid = (%d,%d), x=%.4f, y=%.4f\n', ...
        cfg.source_col_phys, cfg.source_row_phys, prob.x_phys(cfg.source_col_phys), prob.y_phys(cfg.source_row_phys));
    fprintf('c_right values = %s\n', mat2str(cfg.c_right_values, 4));

    % Optional homogeneous reference for scattering-like field:
    % uscat(c_right) = utotal(c_right) - utotal(c_right = 1).
    incident = struct();
    if cfg.compute_incident_reference
        fprintf('\nSolving homogeneous reference (c_right = 1.0) ...\n');
        [x_inc, u_inc, b_inc, info_inc] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, 1.0, zeros(prob.nfull, 1));
        incident.x = x_inc;
        incident.u = u_inc;
        incident.b = b_inc;
        incident.info = info_inc;
        incident.u_phys = interior_to_physical(u_inc, prob);
        fprintf('incident solve: iter=%3d, relres=%.3e, time=%.3fs\n', ...
            info_inc.iter_count, info_inc.relres, info_inc.solve_time);
    else
        incident = [];
    end

    nc = numel(cfg.c_right_values);
    results = struct();
    results.cfg = cfg;
    results.problem = rmfield(prob, {'pml_helm', 'pml_wave'});
    results.rhs = rhs;
    results.incident = incident;
    results.c_right_values = cfg.c_right_values(:);
    results.x_wave = zeros(prob.nfull, nc);
    results.u_wave = zeros(prob.Ntot, nc);
    results.u_wave_phys = zeros(prob.nx_phys, prob.ny_phys, nc);
    results.u_helm = zeros(prob.Ntot, nc);
    results.u_helm_phys = zeros(prob.nx_phys, prob.ny_phys, nc);
    results.rel_wave_vs_helm_phys = zeros(nc, 1);
    results.rel_vs_first_phys = zeros(nc, 1);
    results.rel_scat_vs_total_phys = zeros(nc, 1);
    empty_recovery = struct('method', '', 'helm_relres', NaN, ...
        'candidate_labels', {{}}, 'candidate_helm_relres', []);
    results.info = repmat(struct('flag', 0, 'relres', 0, 'iter', [0, 0], ...
        'iter_count', 0, 'resvec', [], 'solve_time', 0, 'recovery_time', 0, ...
        'recovery', empty_recovery, 'solver_extra', struct()), nc, 1);

    fprintf('\n=== Solving parameter sweep ===\n');
    for k = 1:nc
        c_right = cfg.c_right_values(k);
        alpha = (1 / c_right^2) - 1;
        u_helm = (prob.A_base + alpha * prob.A_right) \ rhs;
        [x_sol, u_wave, ~, info, rel_wh_vs_helm] = solve_waveholtz_aligned_with_helm( ...
            prob, cfg, rhs, c_right, u_helm);

        u_wave_phys = interior_to_physical(u_wave, prob);
        u_helm_phys = interior_to_physical(u_helm, prob);

        results.x_wave(:, k) = x_sol;
        results.u_wave(:, k) = u_wave;
        results.u_wave_phys(:, :, k) = u_wave_phys;
        results.u_helm(:, k) = u_helm;
        results.u_helm_phys(:, :, k) = u_helm_phys;
        results.rel_wave_vs_helm_phys(k) = rel_wh_vs_helm;
        results.info(k) = info;

        if ~isempty(incident)
            u_scat = u_wave_phys - incident.u_phys;
            results.rel_scat_vs_total_phys(k) = norm(u_scat(:)) / max(norm(u_wave_phys(:)), 1e-16);
        else
            results.rel_scat_vs_total_phys(k) = NaN;
        end

        fprintf('case %2d/%2d: c_right=%.4f, iter=%3d, relres=%.3e, rel(Wave-Helm,phys)=%.3e\n', ...
            k, nc, c_right, info.iter_count, info.relres, rel_wh_vs_helm);
    end

    u_ref = results.u_wave_phys(:, :, 1);
    for k = 1:nc
        uk = results.u_wave_phys(:, :, k);
        results.rel_vs_first_phys(k) = norm(uk(:) - u_ref(:)) / max(norm(u_ref(:)), 1e-16);
    end

    fprintf('\n=== Difference summary (physical domain) ===\n');
    for k = 1:nc
        fprintf('c_right=%.4f: rel_vs_first=%.3e, rel_wave_vs_helm=%.3e', ...
            cfg.c_right_values(k), results.rel_vs_first_phys(k), results.rel_wave_vs_helm_phys(k));
        if ~isempty(incident)
            fprintf(', rel_scat_vs_total=%.3e', results.rel_scat_vs_total_phys(k));
        end
        fprintf('\n');
    end

    if cfg.do_plots
        make_plots(results, prob);
    end
end

function [x_sol, u_wave, b_vec, info, rel_wh_vs_helm] = solve_waveholtz_aligned_with_helm(prob, cfg, rhs, c_right, u_helm)
    switch cfg.wh_initial_guess
        case 'zero'
            x0 = zeros(prob.nfull, 1);
        case 'helm_lift'
            x0 = lift_frequency_solution_to_real_state(prob, u_helm, c_right);
            fprintf('  WaveHoltz initial guess: Helm-lifted state.\n');
    end
    total_iter = 0;
    total_solve_time = 0;
    resvec_all = [];
    max_rounds = 1 + cfg.wh_auto_refine_against_helm * cfg.wh_auto_refine_max_rounds;
    cfg_round = cfg;

    for round_idx = 1:max_rounds
        [x_sol, u_wave, b_vec, info_round] = solve_waveholtz_fullstate_solution(prob, cfg_round, rhs, c_right, x0);
        total_iter = total_iter + info_round.iter_count;
        total_solve_time = total_solve_time + info_round.solve_time;
        resvec_all = append_resvec(resvec_all, info_round.resvec);
        rel_wh_vs_helm = compute_phys_relerr(u_wave, u_helm, prob);

        info = info_round;
        info.iter_count = total_iter;
        info.solve_time = total_solve_time;
        info.resvec = resvec_all;

        if ~cfg.wh_auto_refine_against_helm || rel_wh_vs_helm <= cfg.wh_helm_align_tol || round_idx == max_rounds
            return;
        end

        fprintf(['  WaveHoltz/Helm mismatch %.3e > %.3e; continuing GMRES ' ...
            'from current state (round %d/%d).\n'], ...
            rel_wh_vs_helm, cfg.wh_helm_align_tol, round_idx + 1, max_rounds);
        x0 = x_sol;
        cfg_round.wh_gmres_maxit = max(1, ceil(cfg_round.wh_gmres_maxit * cfg.wh_auto_refine_maxit_factor));
    end
end

function resvec_all = append_resvec(resvec_all, resvec_new)
    if isempty(resvec_all)
        resvec_all = resvec_new(:);
        return;
    end
    if isempty(resvec_new)
        return;
    end
    resvec_all = [resvec_all; resvec_new(2:end)];
end

function relerr = compute_phys_relerr(u_a, u_b, prob)
    ua = interior_to_physical(u_a, prob);
    ub = interior_to_physical(u_b, prob);
    relerr = norm(ua(:) - ub(:)) / max(norm(ub(:)), 1e-16);
end

function x_real = lift_frequency_solution_to_real_state(prob, uhat, c_right)
    xhat = recover_fullstate_frequency_from_u(prob, uhat, c_right);
    x_real = real(xhat);
end

function xhat_full = recover_fullstate_frequency_from_u(prob, uhat, c_right)
    [sigma_x_u, sigma_y_u, sigma_x_vx, sigma_y_vy, theta_coeff] = build_discrete_lift_grids(prob, c_right);
    lam = exp(-1i * prob.omega_ts * prob.dt);

    u_full = zeros(prob.nx + 2, prob.ny + 2);
    u_full(2:end-1, 2:end-1) = reshape(uhat, prob.nx, prob.ny);

    du_dx = (u_full(2:end, :) - u_full(1:end-1, :)) / prob.hx;
    du_dy = (u_full(:, 2:end) - u_full(:, 1:end-1)) / prob.hy;

    denom_vx = lam * (1 + 0.5 * prob.dt * sigma_x_vx) - (1 - 0.5 * prob.dt * sigma_x_vx);
    denom_vy = lam * (1 + 0.5 * prob.dt * sigma_y_vy) - (1 - 0.5 * prob.dt * sigma_y_vy);
    vx_hat = (prob.dt * du_dx(:, 2:end-1)) ./ denom_vx;
    vy_hat = (prob.dt * du_dy(2:end-1, :)) ./ denom_vy;

    vx_full = zeros(prob.nx + 1, prob.ny + 2);
    vy_full = zeros(prob.nx + 2, prob.ny + 1);
    vx_full(:, 2:end-1) = vx_hat;
    vy_full(2:end-1, :) = vy_hat;

    grad_vx = (vx_full(2:end, :) - vx_full(1:end-1, :)) / prob.hx;
    grad_vy = (vy_full(:, 2:end) - vy_full(:, 1:end-1)) / prob.hy;

    avg_factor = 0.5 * prob.dt * (1 + lam) / (lam - 1);
    theta_factor = prob.dt / (lam - 1);
    psi1_hat = avg_factor * (sigma_y_u .* grad_vx(:, 2:end-1));
    psi2_hat = avg_factor * (sigma_x_u .* grad_vy(2:end-1, :));
    theta_hat = theta_factor * (theta_coeff .* reshape(uhat, prob.nx, prob.ny));

    xhat_full = zeros(prob.nfull, 1);
    xhat_full(1:prob.nu) = uhat;
    xhat_full(prob.nu + 1:prob.nu + prob.nvx) = reshape(vx_hat, prob.nvx, 1);
    xhat_full(prob.nu + prob.nvx + 1:prob.nu + prob.nv) = reshape(vy_hat, prob.nvy, 1);
    xhat_full(prob.nu + prob.nv + 1:prob.nu + prob.nv + prob.nu) = reshape(psi1_hat, prob.nu, 1);
    xhat_full(prob.nu + prob.nv + prob.nu + 1:prob.nu + prob.nv + 2 * prob.nu) = reshape(psi2_hat, prob.nu, 1);
    xhat_full(prob.nu + prob.nv + 2 * prob.nu + 1:end) = reshape(theta_hat, prob.nu, 1);
end

function [sigma_x_u, sigma_y_u, sigma_x_vx, sigma_y_vy, theta_coeff] = build_discrete_lift_grids(prob, c_right)
    sx_full = zeros(prob.nx + 2, prob.ny + 2);
    sy_full = zeros(prob.nx + 2, prob.ny + 2);
    for i = 1:(prob.nx + 2), sx_full(i, :) = prob.pml_wave.sigma_x(i); end
    for j = 1:(prob.ny + 2), sy_full(:, j) = prob.pml_wave.sigma_y(j); end
    sigma_x_u = sx_full(2:end-1, 2:end-1);
    sigma_y_u = sy_full(2:end-1, 2:end-1);
    sigma_x_vx = 0.5 * (sx_full(1:end-1, 2:end-1) + sx_full(2:end, 2:end-1));
    sigma_y_vy = 0.5 * (sy_full(2:end-1, 1:end-1) + sy_full(2:end-1, 2:end));
    c_sq = build_piecewise_c_squared(prob, c_right);
    c_sq_full = ones(prob.nx + 2, prob.ny + 2);
    c_sq_full(2:end-1, 2:end-1) = c_sq;
    theta_coeff = (sx_full(2:end-1, 2:end-1) .* sy_full(2:end-1, 2:end-1)) .* ...
        (prob.c_w ^ 2) ./ c_sq_full(2:end-1, 2:end-1);
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'L', 1.0);
    cfg = set_default(cfg, 'Lx', []);
    cfg = set_default(cfg, 'Ly', []);
    cfg = set_default(cfg, 'omega', 53.5);
    cfg = set_default(cfg, 'Nphys', 125);
    cfg = set_default(cfg, 'Nphys_x', []);
    cfg = set_default(cfg, 'Nphys_y', []);
    cfg = set_default(cfg, 'ispml', 1);
    cfg = set_default(cfg, 'npml', []);
    cfg = set_default(cfg, 'sigma_max', 80);
    cfg = set_default(cfg, 'CFL', 0.5);
    cfg = set_default(cfg, 'filter_shift', 0.25);
    cfg = set_default(cfg, 'filter_periods', 1);
    cfg = set_default(cfg, 'interface_x', []);
    cfg = set_default(cfg, 'material_model', 'pc');
    cfg.material_model = lower(strtrim(cfg.material_model));
    cfg = set_default(cfg, 'pc_num_rows', 3);
    cfg = set_default(cfg, 'pc_lattice_a', []);
    cfg = set_default(cfg, 'pc_radius', []);
    cfg = set_default(cfg, 'pc_rod_shape', 'square');
    cfg.pc_rod_shape = lower(strtrim(cfg.pc_rod_shape));
    cfg = set_default(cfg, 'pc_square_halfwidth', []);
    cfg = set_default(cfg, 'pc_x_margin', []);

    cfg = set_default(cfg, 'source_col_phys', []);
    cfg = set_default(cfg, 'source_row_phys', []);
    cfg = set_default(cfg, 'source_type', 'point');
    cfg = set_default(cfg, 'point_amplitude', 1.0);
    cfg = set_default(cfg, 'point_source_profile', 'box');
    cfg = set_default(cfg, 'point_support_halfwidth', 0);
    cfg = set_default(cfg, 'point_box_radius_x', []);
    cfg = set_default(cfg, 'point_box_radius_y', []);
    cfg = set_default(cfg, 'gaussian_amplitude', 1.0);
    cfg = set_default(cfg, 'gaussian_sigma', []);

    cfg = set_default(cfg, 'c_right_values', [0.40]);
    cfg = set_default(cfg, 'compute_incident_reference', false);

    cfg = set_default(cfg, 'wh_tol', 1e-12);
    cfg = set_default(cfg, 'wh_solver', 'custom_gmres');
    cfg.wh_solver = lower(strtrim(cfg.wh_solver));
    cfg = set_default(cfg, 'wh_initial_guess', 'zero');
    cfg.wh_initial_guess = lower(strtrim(cfg.wh_initial_guess));
    cfg = set_default(cfg, 'wh_gmres_restart', 20);
    cfg = set_default(cfg, 'wh_gmres_maxit', 40);
    cfg = set_default(cfg, 'wh_verbose_gmres', false);
    cfg = set_default(cfg, 'wh_gmres_reorth', false);
    cfg = set_default(cfg, 'wh_recovery_mode', 'best_helm');
    cfg.wh_recovery_mode = lower(strtrim(cfg.wh_recovery_mode));
    cfg = set_default(cfg, 'wh_auto_refine_against_helm', true);
    cfg = set_default(cfg, 'wh_helm_align_tol', 1e-3);
    cfg = set_default(cfg, 'wh_auto_refine_max_rounds', 3);
    cfg = set_default(cfg, 'wh_auto_refine_maxit_factor', 2);
    cfg = set_default(cfg, 'fixed_point_relax', 1.0);
    cfg = set_default(cfg, 'dmd_relax', 1.0);
    cfg = set_default(cfg, 'dmd_max_snaps', 40);
    cfg = set_default(cfg, 'dmd_epsSVD', 1e-10);
    cfg = set_default(cfg, 'dmd_svd_rel_tol', 1e-6);
    cfg = set_default(cfg, 'dmd_max_rank', 30);
    cfg = set_default(cfg, 'dmd_min_snaps_for_predict', 16);
    cfg = set_default(cfg, 'dmd_min_pred_interval', 20);
    cfg = set_default(cfg, 'dmd_cond_ratio_tol', 1e-3);
    cfg = set_default(cfg, 'dmd_rho_tol', 0.999);
    cfg = set_default(cfg, 'dmd_predict_burst', 1);
    cfg = set_default(cfg, 'dmd_accept_growth', 1.0);
    cfg = set_default(cfg, 'dmd_start_plain_res', inf);
    cfg = set_default(cfg, 'dmd_window_strategy', 'rolling');
    cfg = set_default(cfg, 'do_plots', true);

    if isempty(cfg.Lx)
        cfg.Lx = cfg.L;
    end
    if isempty(cfg.Ly)
        cfg.Ly = cfg.L;
    end
    if isempty(cfg.Nphys_x)
        cfg.Nphys_x = cfg.Nphys;
    end
    if isempty(cfg.Nphys_y)
        cfg.Nphys_y = cfg.Nphys;
    end

    if isempty(cfg.npml)
        cfg.npml = cfg.ispml * ceil(0.25 * min(cfg.Nphys_x, cfg.Nphys_y));
    end
    if isempty(cfg.interface_x)
        cfg.interface_x = 0.5 * cfg.Lx;
    end
    if isempty(cfg.pc_lattice_a)
        cfg.pc_lattice_a = cfg.Ly / (2 * cfg.pc_num_rows + 3);
    end
    if isempty(cfg.pc_radius)
        cfg.pc_radius = 0.25 * cfg.pc_lattice_a;
    end
    if isempty(cfg.pc_square_halfwidth)
        cfg.pc_square_halfwidth = 0.5 * sqrt(pi) * cfg.pc_radius;
    end
    if isempty(cfg.pc_x_margin)
        cfg.pc_x_margin = cfg.pc_lattice_a;
    end
    if isempty(cfg.source_col_phys)
        if strcmp(cfg.material_model, 'halfspace')
            cfg.source_col_phys = max(2, round(0.20 * cfg.Nphys_x));
        else
            cfg.source_col_phys = round((cfg.Nphys_x + 1) / 2);
        end
    end
    if isempty(cfg.source_row_phys)
        cfg.source_row_phys = round((cfg.Nphys_y + 1) / 2);
    end
    if isempty(cfg.gaussian_sigma)
        cfg.gaussian_sigma = 3 * min(cfg.Lx / (cfg.Nphys_x + 1), cfg.Ly / (cfg.Nphys_y + 1));
    end
    if isempty(cfg.point_box_radius_x)
        cfg.point_box_radius_x = 2 * cfg.Lx / (cfg.Nphys_x + 1);
    end
    if isempty(cfg.point_box_radius_y)
        cfg.point_box_radius_y = 2 * cfg.Ly / (cfg.Nphys_y + 1);
    end

    cfg.c_right_values = unique(cfg.c_right_values(:).');

    if cfg.source_col_phys < 1 || cfg.source_col_phys > cfg.Nphys_x
        error('source_col_phys=%d is outside [1,%d].', cfg.source_col_phys, cfg.Nphys_x);
    end
    if strcmp(cfg.material_model, 'halfspace') && cfg.source_col_phys > floor(cfg.Nphys_x / 2)
        error('source_col_phys=%d must lie on the left half of the physical domain.', cfg.source_col_phys);
    end
    if cfg.source_row_phys < 1 || cfg.source_row_phys > cfg.Nphys_y
        error('source_row_phys=%d is outside [1,%d].', cfg.source_row_phys, cfg.Nphys_y);
    end
    valid_source_types = {'point', 'gaussian'};
    if ~any(strcmp(cfg.source_type, valid_source_types))
        error('source_type=%s must be one of: %s.', cfg.source_type, strjoin(valid_source_types, ', '));
    end
    valid_point_profiles = {'box', 'cell'};
    if ~any(strcmp(cfg.point_source_profile, valid_point_profiles))
        error('point_source_profile=%s must be one of: %s.', cfg.point_source_profile, strjoin(valid_point_profiles, ', '));
    end
    valid_material_models = {'halfspace', 'photonic_crystal', 'pc', 'double_layer', 'two_layer'};
    if ~any(strcmp(cfg.material_model, valid_material_models))
        error('material_model=%s must be one of: %s.', cfg.material_model, strjoin(valid_material_models, ', '));
    end
    if ~(isscalar(cfg.pc_num_rows) && cfg.pc_num_rows >= 1 && abs(cfg.pc_num_rows - round(cfg.pc_num_rows)) < 1e-12)
        error('pc_num_rows must be a positive integer.');
    end
    if ~(isscalar(cfg.pc_lattice_a) && isfinite(cfg.pc_lattice_a) && cfg.pc_lattice_a > 0)
        error('pc_lattice_a must be a positive finite scalar.');
    end
    if ~(isscalar(cfg.pc_radius) && isfinite(cfg.pc_radius) && cfg.pc_radius > 0)
        error('pc_radius must be a positive finite scalar.');
    end
    valid_rod_shapes = {'circle', 'square'};
    if ~any(strcmp(cfg.pc_rod_shape, valid_rod_shapes))
        error('pc_rod_shape=%s must be one of: %s.', cfg.pc_rod_shape, strjoin(valid_rod_shapes, ', '));
    end
    valid_dmd_window_strategies = {'rolling', 'restart'};
    cfg.dmd_window_strategy = lower(strtrim(cfg.dmd_window_strategy));
    if ~any(strcmp(cfg.dmd_window_strategy, valid_dmd_window_strategies))
        error('dmd_window_strategy=%s must be one of: %s.', cfg.dmd_window_strategy, strjoin(valid_dmd_window_strategies, ', '));
    end
    if ~(isscalar(cfg.pc_square_halfwidth) && isfinite(cfg.pc_square_halfwidth) && cfg.pc_square_halfwidth > 0)
        error('pc_square_halfwidth must be a positive finite scalar.');
    end
    if ~(isscalar(cfg.Lx) && isfinite(cfg.Lx) && cfg.Lx > 0)
        error('Lx must be a positive finite scalar.');
    end
    if ~(isscalar(cfg.Ly) && isfinite(cfg.Ly) && cfg.Ly > 0)
        error('Ly must be a positive finite scalar.');
    end
    if ~(isscalar(cfg.Nphys_x) && cfg.Nphys_x >= 1 && abs(cfg.Nphys_x - round(cfg.Nphys_x)) < 1e-12)
        error('Nphys_x must be a positive integer.');
    end
    if ~(isscalar(cfg.Nphys_y) && cfg.Nphys_y >= 1 && abs(cfg.Nphys_y - round(cfg.Nphys_y)) < 1e-12)
        error('Nphys_y must be a positive integer.');
    end
    valid_recovery_modes = {'timeproj', 'state', 'best_helm'};
    if ~any(strcmp(cfg.wh_recovery_mode, valid_recovery_modes))
        error('wh_recovery_mode=%s must be one of: %s.', cfg.wh_recovery_mode, strjoin(valid_recovery_modes, ', '));
    end
    valid_wh_solvers = {'custom_gmres', 'matlab_gmres', 'bicgstab', ...
        'fixed_point', 'plain_fixed_point', ...
        'auto_dmd_restarted_v2', 'autodmd_restarted_v2', 'autodmd'};
    if ~any(strcmp(cfg.wh_solver, valid_wh_solvers))
        error('wh_solver=%s must be one of: %s.', cfg.wh_solver, strjoin(valid_wh_solvers, ', '));
    end
    valid_initial_guesses = {'zero', 'helm_lift'};
    if ~any(strcmp(cfg.wh_initial_guess, valid_initial_guesses))
        error('wh_initial_guess=%s must be one of: %s.', cfg.wh_initial_guess, strjoin(valid_initial_guesses, ', '));
    end
    if ~(isscalar(cfg.wh_helm_align_tol) && isfinite(cfg.wh_helm_align_tol) && cfg.wh_helm_align_tol > 0)
        error('wh_helm_align_tol must be a positive finite scalar.');
    end
    if ~(isscalar(cfg.wh_auto_refine_max_rounds) && cfg.wh_auto_refine_max_rounds >= 0)
        error('wh_auto_refine_max_rounds must be a nonnegative scalar.');
    end
    if ~(isscalar(cfg.wh_auto_refine_maxit_factor) && cfg.wh_auto_refine_maxit_factor >= 1)
        error('wh_auto_refine_maxit_factor must be at least 1.');
    end
    if ~(isscalar(cfg.filter_periods) && isfinite(cfg.filter_periods) && cfg.filter_periods >= 1 && ...
            abs(cfg.filter_periods - round(cfg.filter_periods)) < 1e-12)
        error('filter_periods must be a positive integer.');
    end
end

function validate_c_values(vals)
    if isempty(vals)
        error('c_right_values cannot be empty.');
    end
    if ~isnumeric(vals) || ~isreal(vals)
        error('c_right_values must be a real numeric array.');
    end
    if any(~isfinite(vals(:)))
        error('c_right_values contains non-finite values.');
    end
    if any(vals(:) <= 0)
        error('c_right_values must be positive.');
    end
end

function prob = build_problem(cfg)
    prob = struct();
    prob.L = cfg.L;
    prob.Lx = cfg.Lx;
    prob.Ly = cfg.Ly;
    prob.omega = cfg.omega;
    prob.filter_periods = cfg.filter_periods;
    prob.nx_phys = cfg.Nphys_x;
    prob.ny_phys = cfg.Nphys_y;
    prob.hx = cfg.Lx / (prob.nx_phys + 1);
    prob.hy = cfg.Ly / (prob.ny_phys + 1);
    prob.npml = cfg.npml;
    prob.nx = prob.nx_phys + 2 * prob.npml;
    prob.ny = prob.ny_phys + 2 * prob.npml;
    prob.Ntot = prob.nx * prob.ny;
    prob.I = (prob.npml + 1):(prob.npml + prob.nx_phys);
    prob.J = (prob.npml + 1):(prob.npml + prob.ny_phys);

    c_max = max([1.0; cfg.c_right_values(:)]);
    dt0 = cfg.CFL * min(prob.hx, prob.hy) / max(c_max, 1e-14);
    [prob.omega_ts, prob.c_w, prob.nt, prob.dt] = match_omega_and_period(prob.omega, dt0);
    prob.T_ts = prob.nt * prob.dt;

    Lpml_x = prob.npml * prob.hx;
    Lpml_y = prob.npml * prob.hy;
    prob.x = linspace(-Lpml_x, cfg.Lx + Lpml_x, prob.nx + 2);
    prob.y = linspace(-Lpml_y, cfg.Ly + Lpml_y, prob.ny + 2);
    prob.x_int = prob.x(2:end-1).';
    prob.y_int = prob.y(2:end-1).';
    prob.x_phys = prob.x(prob.I + 1).';
    prob.y_phys = prob.y(prob.J + 1).';

    prob.pml_helm = build_pml_outer(prob.nx, prob.ny, prob.omega, cfg.npml, cfg.sigma_max, cfg.ispml);
    prob.pml_wave = prob.pml_helm;
    prob.pml_wave.sigma_x = prob.pml_helm.sigma_x / prob.c_w;
    prob.pml_wave.sigma_y = prob.pml_helm.sigma_y / prob.c_w;

    [prob.nu, prob.nvx, prob.nvy, prob.nv, prob.nfull] = state_sizes(prob.nx, prob.ny);

    [prob.chi_right_grid, prob.material_meta] = build_parameter_mask(prob, cfg);
    prob.chi_right = prob.chi_right_grid(:);
    prob.material_model = cfg.material_model;

    prob.K_pml = build_pml_stiffness(prob.nx, prob.ny, prob.hx, prob.hy, prob.pml_helm);
    prob.A_base = prob.K_pml + (prob.omega ^ 2) * speye(prob.Ntot);
    prob.A_right = spdiags((prob.omega ^ 2) * double(prob.chi_right), 0, prob.Ntot, prob.Ntot);
end

function [chi_grid, meta] = build_parameter_mask(prob, cfg)
    switch cfg.material_model
        case 'halfspace'
            chi_grid = repmat(prob.x_int > cfg.interface_x, 1, prob.ny);
            meta = struct();
            meta.model = cfg.material_model;
            meta.num_mask_nodes = nnz(chi_grid);
            meta.num_rows = 0;
            meta.num_cols = 0;
            meta.lattice_a = NaN;
            meta.radius = NaN;
            return;

        case {'photonic_crystal', 'pc', 'double_layer', 'two_layer'}
            [chi_grid, meta] = build_photonic_crystal_mask(prob, cfg);
            return;
    end

    error('Unsupported material_model=%s.', cfg.material_model);
end

function [chi_grid, meta] = build_photonic_crystal_mask(prob, cfg)
    a = cfg.pc_lattice_a;
    radius = cfg.pc_radius;
    square_halfwidth = cfg.pc_square_halfwidth;
    x_margin = cfg.pc_x_margin;
    y_mid = 0.5 * cfg.Ly;

    x_centers = x_margin:a:(cfg.Lx - x_margin);
    y_offsets = (1:cfg.pc_num_rows) * a;
    y_centers = [y_mid - fliplr(y_offsets), y_mid + y_offsets];
    y_centers = y_centers(y_centers > radius & y_centers < cfg.Ly - radius);

    [X, Y] = ndgrid(prob.x_int, prob.y_int);
    phys_mask = (X >= 0) & (X <= cfg.Lx) & (Y >= 0) & (Y <= cfg.Ly);
    chi_grid = false(prob.nx, prob.ny);
    for ix = 1:numel(x_centers)
        for iy = 1:numel(y_centers)
            switch cfg.pc_rod_shape
                case 'circle'
                    cell_mask = (X - x_centers(ix)).^2 + (Y - y_centers(iy)).^2 <= radius^2;
                case 'square'
                    cell_mask = abs(X - x_centers(ix)) <= square_halfwidth & ...
                        abs(Y - y_centers(iy)) <= square_halfwidth;
            end
            chi_grid = chi_grid | cell_mask;
        end
    end
    chi_grid = chi_grid & phys_mask;

    meta = struct();
    meta.model = cfg.material_model;
    meta.num_mask_nodes = nnz(chi_grid);
    meta.num_rows = cfg.pc_num_rows;
    meta.num_cols = numel(x_centers);
    meta.lattice_a = a;
    meta.radius = radius;
    meta.rod_shape = cfg.pc_rod_shape;
    meta.square_halfwidth = square_halfwidth;
    meta.x_centers = x_centers(:);
    meta.y_centers = y_centers(:);
end

function [x_sol, uhat, b_vec, info] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, c_right, x0)
    if nargin < 5 || isempty(x0)
        x0 = zeros(prob.nfull, 1);
    end
    [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right);

    tsolve = tic;
    [x_sol, flag, relres, iter, resvec, iter_count, solver_extra] = solve_waveholtz_linear_system(Afun, b_vec, cfg, x0);
    solve_time = toc(tsolve);

    trec = tic;
    [uhat, recovery_info] = recover_u_phasor(prob, cfg, x_sol, rhs, c_right);
    recovery_time = toc(trec);

    info = struct();
    info.flag = flag;
    info.relres = relres;
    info.iter = iter;
    info.iter_count = iter_count;
    info.resvec = resvec;
    info.solve_time = solve_time;
    info.recovery_time = recovery_time;
    info.recovery = recovery_info;
    info.solver_extra = solver_extra;
end

function [x_sol, flag, relres, iter, resvec, iter_count, solver_extra] = solve_waveholtz_linear_system(Afun, b_vec, cfg, x0)
    solver_extra = struct();
    switch cfg.wh_solver
        case 'custom_gmres'
            [x_sol, flag, relres, iter, resvec] = gmres_restarted_verbose( ...
                Afun, b_vec, cfg.wh_gmres_restart, cfg.wh_tol, cfg.wh_gmres_maxit, ...
                x0, cfg.wh_verbose_gmres, cfg.wh_gmres_reorth);
            iter_count = iteration_count(iter, cfg.wh_gmres_restart);

        case 'matlab_gmres'
            [x_sol, flag, relres, iter, resvec] = gmres( ...
                Afun, b_vec, cfg.wh_gmres_restart, cfg.wh_tol, cfg.wh_gmres_maxit, [], [], x0);
            iter_count = iteration_count(iter, cfg.wh_gmres_restart);

        case 'bicgstab'
            maxit = cfg.wh_gmres_restart * cfg.wh_gmres_maxit;
            [x_sol, flag, relres, iter, resvec] = bicgstab(Afun, b_vec, cfg.wh_tol, maxit, [], [], x0);
            iter_count = ceil(iter);

        case {'auto_dmd_restarted_v2', 'autodmd_restarted_v2', 'autodmd'}
            [x_sol, flag, relres, iter, resvec, iter_count, solver_extra] = solve_waveholtz_dmd_restarted(Afun, b_vec, cfg, x0);

        case {'fixed_point', 'plain_fixed_point'}
            [x_sol, flag, relres, iter, resvec, iter_count, solver_extra] = solve_waveholtz_fixed_point(Afun, b_vec, cfg, x0);
    end
end

function [x, flag, relres, iter, resvec, iter_count, stats] = solve_waveholtz_fixed_point(Afun, b, cfg, x0)
    if nargin < 4 || isempty(x0)
        x0 = zeros(size(b));
    end

    nb = norm(b);
    if nb == 0
        x = zeros(size(b));
        flag = 0;
        relres = 0;
        iter = [0, 0];
        resvec = 0;
        iter_count = 0;
        stats = struct('resvec_rel', 0);
        return;
    end

    max_iter = max(1, cfg.wh_gmres_restart * cfg.wh_gmres_maxit);
    x = x0;
    resvec = zeros(max_iter + 1, 1);
    resvec(1) = norm(b - Afun(x));
    flag = 1;
    iter_count = 0;

    for k = 1:max_iter
        r = b - Afun(x);
        if any(~isfinite(r))
            flag = 3;
            break;
        end
        x_new = x + cfg.fixed_point_relax * r;
        if any(~isfinite(x_new))
            flag = 3;
            break;
        end
        res_new = norm(b - Afun(x_new));
        if ~isfinite(res_new)
            flag = 3;
            break;
        end
        x = x_new;
        iter_count = k;
        resvec(k + 1) = res_new;
        relres = resvec(k + 1) / nb;
        if cfg.wh_verbose_gmres && (k <= 5 || mod(k, 20) == 0 || relres < cfg.wh_tol)
            fprintf('fixed-point iter %4d: ||r||/||b|| = %.3e\n', k, relres);
        end
        if relres <= cfg.wh_tol
            flag = 0;
            break;
        end
    end

    iter = [0, iter_count];
    relres = resvec(iter_count + 1) / nb;
    resvec = resvec(1:iter_count + 1);
    stats = struct('resvec_rel', resvec / nb);
end

function [x, flag, relres, iter, resvec, iter_count, stats] = solve_waveholtz_dmd_restarted(Afun, b, cfg, x0)
    % Restarted Auto-DMD v2 for the affine WaveHoltz fixed point:
    %   x_{k+1} = Pi*x_k + b = x_k - Afun(x_k) + b.
    % DMD is applied to update snapshots delta_k = x_{k+1}-x_k.
    if nargin < 4 || isempty(x0)
        x0 = zeros(size(b));
    end

    nb = norm(b);
    if nb == 0
        x = zeros(size(b));
        flag = 0;
        relres = 0;
        iter = [0, 0];
        resvec = 0;
        iter_count = 0;
        stats = empty_dmd_stats();
        return;
    end

    max_iter = max(1, cfg.wh_gmres_restart * cfg.wh_gmres_maxit);
    x = x0;
    resvec = zeros(max_iter + 1, 1);
    resvec(1) = norm(b - Afun(x));
    flag = 1;
    iter_count = 0;

    U = [];
    S = [];
    V = [];
    DeltaSnaps = zeros(numel(b), cfg.dmd_max_snaps);
    snap_count = 0;
    last_pred_iter = -1e12;
    burst_remaining = 0;

    stats = empty_dmd_stats();

    if cfg.wh_verbose_gmres
        fprintf('Auto-DMD init: ||r0||/||b|| = %.3e\n', resvec(1) / nb);
    end
    if resvec(1) / nb <= cfg.wh_tol
        flag = 0;
        relres = resvec(1) / nb;
        iter = [0, 0];
        resvec = resvec(1);
        return;
    end

    for k = 1:max_iter
        r = b - Afun(x);
        if any(~isfinite(r))
            flag = 3;
            break;
        end
        x_raw = x + cfg.dmd_relax * r;
        if any(~isfinite(x_raw))
            flag = 3;
            break;
        end
        delta = x_raw - x;
        plain_step = norm(delta) / max(norm(x_raw), 1e-16);

        window_truncated = false;
        if snap_count < cfg.dmd_max_snaps
            snap_count = snap_count + 1;
            DeltaSnaps(:, snap_count) = delta;
        else
            if strcmp(cfg.dmd_window_strategy, 'rolling')
                DeltaSnaps(:, 1:end-1) = DeltaSnaps(:, 2:end);
                DeltaSnaps(:, end) = delta;
                window_truncated = true;
            else
                snap_count = 1;
                DeltaSnaps(:, 1) = delta;
                U = [];
                S = [];
                V = [];
            end
            stats.num_window_truncations = stats.num_window_truncations + 1;
            stats.num_restarts = stats.num_window_truncations;
        end

        x_new = x_raw;
        predicted = false;

        if snap_count >= 2 && plain_step <= cfg.dmd_start_plain_res
            if isempty(U)
                [U, S, V] = dmd_snapshot_svd(DeltaSnaps(:, 1:snap_count-1), cfg.dmd_epsSVD, cfg.dmd_max_rank);
            elseif window_truncated
                [U, S, V] = dmd_snapshot_svd(DeltaSnaps(:, 1:snap_count-1), cfg.dmd_epsSVD, cfg.dmd_max_rank);
            elseif snap_count == 2
                [U, S, V] = dmd_init_svd(DeltaSnaps(:, 1), cfg.dmd_epsSVD);
                [U, S, V] = dmd_truncate_svd(U, S, V, cfg.dmd_epsSVD, cfg.dmd_max_rank);
            else
                [U, S, V] = dmd_incremental_svd(DeltaSnaps(:, snap_count - 1), cfg.dmd_epsSVD, U, S, V);
                [U, S, V] = dmd_truncate_svd(U, S, V, cfg.dmd_epsSVD, cfg.dmd_max_rank);
            end

            if ~isempty(U)
                sigmas_full = diag(S);
                if ~isempty(sigmas_full) && sigmas_full(1) > 0
                    rel_sig = sigmas_full / sigmas_full(1);
                    r_eff = nnz(rel_sig >= cfg.dmd_svd_rel_tol);
                    r_eff = min([r_eff, cfg.dmd_max_rank, numel(sigmas_full), snap_count - 1]);
                else
                    r_eff = 0;
                end

                if r_eff >= 2 && snap_count >= cfg.dmd_min_snaps_for_predict && ...
                        (k - last_pred_iter) >= cfg.dmd_min_pred_interval
                    U_eff = U(:, 1:r_eff);
                    S_eff = S(1:r_eff, 1:r_eff);
                    V_eff = V(:, 1:r_eff);
                    Y_pos = DeltaSnaps(:, 2:snap_count);
                    A_tilde = (U_eff') * Y_pos * V_eff / S_eff;
                    lam = eig(A_tilde);
                    rho = max(abs(lam));

                    sigmas_eff = diag(S_eff);
                    max_sig = max(sigmas_eff);
                    min_sig = min(sigmas_eff(sigmas_eff > 1e-17));
                    if isempty(min_sig)
                        min_sig = 0;
                    end
                    cond_ratio = min_sig / max(max_sig, 1e-16);

                    if cfg.wh_verbose_gmres && (k <= 10 || mod(k, 20) == 0)
                        fprintf('  DMD meta iter %4d: m=%d, r=%d, rho=%.5f, cond_ratio=%.3e\n', ...
                            k, snap_count, r_eff, rho, cond_ratio);
                    end

                    trigger_prediction = cond_ratio < cfg.dmd_cond_ratio_tol && rho < cfg.dmd_rho_tol;
                    if trigger_prediction
                        burst_remaining = max(burst_remaining, cfg.dmd_predict_burst);
                    end

                    can_predict = burst_remaining > 0 && cond_ratio < cfg.dmd_cond_ratio_tol && rho < cfg.dmd_rho_tol;
                    if can_predict
                        delta_z = (eye(r_eff) - A_tilde) \ ((U_eff') * delta);
                        x_pred = x + U_eff * delta_z;
                        if all(isfinite(x_pred))
                            pred_res = norm(b - Afun(x_pred)) / nb;
                            plain_res = norm(b - Afun(x_raw)) / nb;
                            if pred_res <= cfg.dmd_accept_growth * plain_res
                                x_new = x_pred;
                                predicted = true;
                                last_pred_iter = k;
                                stats.num_predictions = stats.num_predictions + 1;
                                burst_remaining = max(0, burst_remaining - 1);
                                stats.singular_vals = sigmas_eff;
                                stats.last_rank = r_eff;
                                stats.last_rho = rho;
                                stats.last_cond_ratio = cond_ratio;
                                stats.last_pred_res = pred_res;
                                stats.last_plain_res = plain_res;
                            else
                                stats.num_rejected = stats.num_rejected + 1;
                                burst_remaining = 0;
                            end
                        else
                            stats.num_rejected = stats.num_rejected + 1;
                            burst_remaining = 0;
                        end
                    end
                end
            end
        end

        if any(~isfinite(x_new))
            flag = 3;
            break;
        end
        res_new = norm(b - Afun(x_new));
        if ~isfinite(res_new)
            flag = 3;
            break;
        end
        x = x_new;
        iter_count = k;
        resvec(k + 1) = res_new;
        relres = resvec(k + 1) / nb;

        if cfg.wh_verbose_gmres && (k <= 5 || mod(k, 20) == 0 || relres < cfg.wh_tol)
            fprintf('Auto-DMD iter %4d: ||r||/||b|| = %.3e\n', k, relres);
        end
        if relres <= cfg.wh_tol
            flag = 0;
            break;
        end
    end

    iter = [0, iter_count];
    relres = resvec(iter_count + 1) / nb;
    resvec = resvec(1:iter_count + 1);
    stats.resvec_rel = resvec / nb;
end

function stats = empty_dmd_stats()
    stats = struct();
    stats.num_restarts = 0;
    stats.num_window_truncations = 0;
    stats.num_predictions = 0;
    stats.num_rejected = 0;
    stats.singular_vals = [];
    stats.last_rank = 0;
    stats.last_rho = NaN;
    stats.last_cond_ratio = NaN;
    stats.last_pred_res = NaN;
    stats.last_plain_res = NaN;
    stats.resvec_rel = [];
end

function [W, sigma, V] = dmd_init_svd(u, epsSVD)
    u_norm = norm(u);
    if u_norm > epsSVD
        W = u / u_norm;
        sigma = diag(u_norm);
        V = 1;
    else
        W = [];
        sigma = [];
        V = [];
    end
end

function [U, S, V] = dmd_snapshot_svd(Y, epsSVD, max_rank)
    if isempty(Y)
        U = [];
        S = [];
        V = [];
        return;
    end

    col_norms = vecnorm(Y);
    active = col_norms > epsSVD;
    if ~any(active)
        U = [];
        S = [];
        V = [];
        return;
    end

    [U, S, V] = svd(Y, 'econ');
    [U, S, V] = dmd_truncate_svd(U, S, V, epsSVD, max_rank);
end

function [U, S, V] = dmd_truncate_svd(U, S, V, epsSVD, max_rank)
    if isempty(S)
        U = [];
        S = [];
        V = [];
        return;
    end

    sigmas = diag(S);
    keep = find(sigmas > epsSVD);
    if max_rank > 0 && numel(keep) > max_rank
        keep = keep(1:max_rank);
    end

    if isempty(keep)
        U = [];
        S = [];
        V = [];
        return;
    end

    U = U(:, keep);
    S = diag(sigmas(keep));
    V = V(:, keep);
end

function [U_r, S_r, V_r] = dmd_incremental_svd(u, epsSVD, U, S, V)
    if isempty(U)
        [U_r, S_r, V_r] = dmd_init_svd(u, epsSVD);
        return;
    end

    p = U' * u;
    r = u - U * p;
    r_norm = norm(r);

    if r_norm > epsSVD
        r_hat = r / r_norm;
        K = [S, p; zeros(1, size(S, 2)), r_norm];
        [Uk, Sk, Vk] = svd(K, 'econ');
        U_r = [U, r_hat] * Uk;
        S_r = Sk;
        V_r = blkdiag(V, 1) * Vk;
    else
        K = [S, p];
        [Uk, Sk, Vk] = svd(K, 'econ');
        U_r = U * Uk;
        S_r = Sk;
        V_r = blkdiag(V, 1) * Vk;
    end
end

function [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right)
    b_vec = apply_pi_piecewise(prob, zeros(prob.nfull, 1), rhs, true, c_right, cfg.filter_shift, cfg.filter_periods);
    Afun = build_waveholtz_apply_only(prob, cfg, c_right);
end

function Afun = build_waveholtz_apply_only(prob, cfg, c_right)
    zero_rhs = zeros(prob.nu, 1);
    Afun = @(x) x - apply_pi_piecewise(prob, x, zero_rhs, false, c_right, cfg.filter_shift, cfg.filter_periods);
end

function [uhat, info] = recover_u_phasor(prob, cfg, x_state, rhs, c_right)
    switch cfg.wh_recovery_mode
        case 'timeproj'
            [u_plus, u_minus] = recover_u_phasor_timeproj_candidates(prob, x_state, rhs, c_right);
            [uhat, info] = choose_by_helm_residual(prob, c_right, rhs, {u_plus, u_minus}, {'timeproj+', 'timeproj-'});
        case 'state'
            [u_plus, u_minus] = recover_u_phasor_state_candidates(prob, x_state);
            [uhat, info] = choose_by_helm_residual(prob, c_right, rhs, {u_plus, u_minus}, {'state+', 'state-'});
        case 'best_helm'
            [ut_plus, ut_minus] = recover_u_phasor_timeproj_candidates(prob, x_state, rhs, c_right);
            [us_plus, us_minus] = recover_u_phasor_state_candidates(prob, x_state);
            [uhat, info] = choose_by_helm_residual(prob, c_right, rhs, ...
                {ut_plus, ut_minus, us_plus, us_minus}, ...
                {'timeproj+', 'timeproj-', 'state+', 'state-'});
    end
end

function [u_plus, u_minus] = recover_u_phasor_timeproj_candidates(prob, x_state, rhs, c_right)
    num_periods = get_filter_periods_from_prob(prob);
    Wconv = solve_wave_one_period_piecewise(prob, x_state, rhs, c_right, num_periods);
    nt_total = prob.nt * num_periods;
    T_total = prob.T_ts * num_periods;
    t_grid = (0:nt_total).' * prob.dt;
    q_cos = (2.0 / T_total) * trapz(t_grid, bsxfun(@times, cos(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
    q_sin = (2.0 / T_total) * trapz(t_grid, bsxfun(@times, sin(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
    u_plus = q_cos + 1i * q_sin;
    u_minus = q_cos - 1i * q_sin;
end

function [u_plus, u_minus] = recover_u_phasor_state_candidates(prob, x_state)
    u0_int = x_state(1:prob.nu);
    v0_int = x_state(prob.nu + 1:prob.nu + prob.nv);

    vx0 = zeros(prob.nx + 1, prob.ny + 2);
    vy0 = zeros(prob.nx + 2, prob.ny + 1);
    vx0(:, 2:end-1) = reshape(v0_int(1:prob.nvx), prob.nx + 1, prob.ny);
    vy0(2:end-1, :) = reshape(v0_int(prob.nvx + 1:end), prob.nx, prob.ny + 1);

    div_v0 = zeros(prob.nx + 2, prob.ny + 2);
    div_v0(2:end-1, :) = div_v0(2:end-1, :) + (vx0(2:end, :) - vx0(1:end-1, :)) / prob.hx;
    div_v0(:, 2:end-1) = div_v0(:, 2:end-1) + (vy0(:, 2:end) - vy0(:, 1:end-1)) / prob.hy;

    u0 = reshape(u0_int, prob.nx, prob.ny);
    uim = (1 / prob.omega) * div_v0(2:end-1, 2:end-1);
    u_plus = u0(:) + 1i * uim(:);
    u_minus = u0(:) - 1i * uim(:);
end

function [uhat, info] = choose_by_helm_residual(prob, c_right, rhs, candidates, labels)
    alpha = (1 / c_right^2) - 1;
    Ahelm = prob.A_base + alpha * prob.A_right;
    nb = max(norm(rhs), 1e-16);
    relres = zeros(numel(candidates), 1);
    for k = 1:numel(candidates)
        relres(k) = norm(Ahelm * candidates{k} - rhs) / nb;
    end
    [best_res, best_idx] = min(relres);
    uhat = candidates{best_idx};
    info = struct();
    info.method = labels{best_idx};
    info.helm_relres = best_res;
    info.candidate_labels = labels(:);
    info.candidate_helm_relres = relres;
end

function W = solve_wave_one_period_piecewise(prob, x0, rhs, c_right, num_periods)
    if nargin < 5 || isempty(num_periods)
        num_periods = 1;
    end
    nt_total = prob.nt * num_periods;

    sx_full = zeros(prob.nx + 2, prob.ny + 2);
    sy_full = zeros(prob.nx + 2, prob.ny + 2);
    for i = 1:(prob.nx + 2), sx_full(i, :) = prob.pml_wave.sigma_x(i); end
    for j = 1:(prob.ny + 2), sy_full(:, j) = prob.pml_wave.sigma_y(j); end

    sigma_x_vx = 0.5 * (sx_full(1:end-1, :) + sx_full(2:end, :));
    sigma_y_vy = 0.5 * (sy_full(:, 1:end-1) + sy_full(:, 2:end));

    if numel(x0) ~= prob.nfull
        error('x0 length mismatch: expected %d, got %d', prob.nfull, numel(x0));
    end

    c_sq = build_piecewise_c_squared(prob, c_right);
    c_sq_full = ones(prob.nx + 2, prob.ny + 2);
    c_sq_full(2:end-1, 2:end-1) = c_sq;

    u = zeros(prob.nx + 2, prob.ny + 2);
    u(2:end-1, 2:end-1) = reshape(x0(1:prob.nu), prob.nx, prob.ny);

    vx = zeros(prob.nx + 1, prob.ny + 2);
    vy = zeros(prob.nx + 2, prob.ny + 1);
    vx(:, 2:end-1) = reshape(x0(prob.nu + 1:prob.nu + prob.nvx), prob.nx + 1, prob.ny);
    vy(2:end-1, :) = reshape(x0(prob.nu + prob.nvx + 1:prob.nu + prob.nv), prob.nx, prob.ny + 1);

    psi1 = zeros(prob.nx + 2, prob.ny + 2);
    psi2 = zeros(prob.nx + 2, prob.ny + 2);
    theta = zeros(prob.nx + 2, prob.ny + 2);
    psi1(2:end-1, 2:end-1) = reshape(x0(prob.nu + prob.nv + 1:prob.nu + prob.nv + prob.nu), prob.nx, prob.ny);
    psi2(2:end-1, 2:end-1) = reshape(x0(prob.nu + prob.nv + prob.nu + 1:prob.nu + prob.nv + 2 * prob.nu), prob.nx, prob.ny);
    theta(2:end-1, 2:end-1) = reshape(x0(prob.nu + prob.nv + 2 * prob.nu + 1:end), prob.nx, prob.ny);

    rhs_2d = reshape(rhs, prob.nx, prob.ny);
    sigu = sx_full + sy_full;
    theta_coeff = (sx_full .* sy_full) .* (prob.c_w ^ 2) ./ c_sq_full;

    W = zeros(nt_total + 1, prob.nfull);
    v0_vec = [reshape(vx(:, 2:end-1), prob.nvx, 1); reshape(vy(2:end-1, :), prob.nvy, 1)];
    W(1, :) = [reshape(u(2:end-1, 2:end-1), prob.nu, 1); v0_vec; ...
               reshape(psi1(2:end-1, 2:end-1), prob.nu, 1); ...
               reshape(psi2(2:end-1, 2:end-1), prob.nu, 1); ...
               reshape(theta(2:end-1, 2:end-1), prob.nu, 1)].';

    for n = 1:nt_total
        t = (n - 1) * prob.dt;
        t_half = t + 0.5 * prob.dt;

        du_dx = (u(2:end, :) - u(1:end-1, :)) / prob.hx;
        du_dy = (u(:, 2:end) - u(:, 1:end-1)) / prob.hy;
        vx_new = ((1 - 0.5 * prob.dt * sigma_x_vx) .* vx + prob.dt * du_dx) ./ (1 + 0.5 * prob.dt * sigma_x_vx);
        vy_new = ((1 - 0.5 * prob.dt * sigma_y_vy) .* vy + prob.dt * du_dy) ./ (1 + 0.5 * prob.dt * sigma_y_vy);

        vx_avg = 0.5 * (vx + vx_new);
        vy_avg = 0.5 * (vy + vy_new);

        grad_vx = (vx_avg(2:end, :) - vx_avg(1:end-1, :)) / prob.hx;
        tmp1 = zeros(prob.nx + 2, prob.ny + 2);
        tmp1(2:end-1, :) = grad_vx;

        grad_vy = (vy_avg(:, 2:end) - vy_avg(:, 1:end-1)) / prob.hy;
        tmp2 = zeros(prob.nx + 2, prob.ny + 2);
        tmp2(:, 2:end-1) = grad_vy;

        psi1 = psi1 + prob.dt * (sy_full .* tmp1);
        psi2 = psi2 + prob.dt * (sx_full .* tmp2);
        theta = theta + prob.dt * (theta_coeff .* u);

        div_v = zeros(prob.nx + 2, prob.ny + 2);
        div_v(2:end-1, :) = div_v(2:end-1, :) + (vx_new(2:end, :) - vx_new(1:end-1, :)) / prob.hx;
        div_v(:, 2:end-1) = div_v(:, 2:end-1) + (vy_new(:, 2:end) - vy_new(:, 1:end-1)) / prob.hy;

        forcing = -(1.0 / prob.omega) * rhs_2d .* sin(prob.omega_ts * t_half);
        rhs_u = div_v(2:end-1, 2:end-1) ...
              - theta(2:end-1, 2:end-1) ...
              + psi1(2:end-1, 2:end-1) ...
              + psi2(2:end-1, 2:end-1) ...
              + forcing;

        num_u = (1 - 0.5 * prob.dt * sigu(2:end-1, 2:end-1)) .* u(2:end-1, 2:end-1) + prob.dt * (c_sq .* rhs_u);
        den_u = (1 + 0.5 * prob.dt * sigu(2:end-1, 2:end-1));
        u(2:end-1, 2:end-1) = num_u ./ den_u;

        vx = vx_new;
        vy = vy_new;

        v_vec = [reshape(vx(:, 2:end-1), prob.nvx, 1); reshape(vy(2:end-1, :), prob.nvy, 1)];
        u_vec = reshape(u(2:end-1, 2:end-1), prob.nu, 1);
        psi1_vec = reshape(psi1(2:end-1, 2:end-1), prob.nu, 1);
        psi2_vec = reshape(psi2(2:end-1, 2:end-1), prob.nu, 1);
        theta_vec = reshape(theta(2:end-1, 2:end-1), prob.nu, 1);
        W(n + 1, :) = [u_vec; v_vec; psi1_vec; psi2_vec; theta_vec].';
    end
end

function c_sq = build_piecewise_c_squared(prob, c_right)
    c_sq = ones(prob.nx, prob.ny);
    c_sq(prob.chi_right_grid) = c_right ^ 2;
end

function Pi = apply_pi_piecewise(prob, x0, rhs, source_on, c_right, shift, num_periods)
    if nargin < 7 || isempty(num_periods)
        num_periods = 1;
    end
    if source_on
        rhs_use = rhs;
    else
        rhs_use = zeros(prob.nu, 1);
    end
    W = solve_wave_one_period_piecewise(prob, x0, rhs_use, c_right, num_periods);
    nt_total = prob.nt * num_periods;
    T_total = prob.T_ts * num_periods;
    t_grid = (0:nt_total).' * prob.dt;
    weights = cos(prob.omega_ts * t_grid) - shift;
    Pi = (2.0 / max(T_total, 1e-16)) * trapz(t_grid, bsxfun(@times, weights, W), 1).';
end

function num_periods = get_filter_periods_from_prob(prob)
    if isfield(prob, 'filter_periods') && ~isempty(prob.filter_periods)
        num_periods = prob.filter_periods;
    else
        num_periods = 1;
    end
end

function rhs = build_rhs(prob, cfg)
    i_tot = prob.I(cfg.source_col_phys);
    j_tot = prob.J(cfg.source_row_phys);
    if strcmp(cfg.source_type, 'point')
        if strcmp(cfg.point_source_profile, 'box')
            rhs = build_point_box_rhs(prob.x_int, prob.y_int, prob.x_phys(cfg.source_col_phys), prob.y_phys(cfg.source_row_phys), ...
                cfg.point_amplitude, cfg.point_box_radius_x, cfg.point_box_radius_y);
        else
            rhs = build_point_cell_rhs(prob.nx, prob.ny, i_tot, j_tot, cfg.point_amplitude, cfg.point_support_halfwidth);
        end
        return;
    end

    if strcmp(cfg.source_type, 'gaussian')
        rhs = build_gaussian_rhs(prob.x_int, prob.y_int, prob.x_phys(cfg.source_col_phys), prob.y_phys(cfg.source_row_phys), ...
            cfg.gaussian_amplitude, cfg.gaussian_sigma);
        rhs_grid = reshape(rhs, prob.nx, prob.ny);
        physical_mask = false(prob.nx, prob.ny);
        physical_mask(prob.I, prob.J) = true;
        rhs_grid(~physical_mask) = 0;
        rhs = rhs_grid(:);
        return;
    end

    error('Unsupported source_type=%s.', cfg.source_type);
end

function rhs = build_point_cell_rhs(nx, ny, i_tot, j_tot, amplitude, halfwidth)
    rhs_grid = zeros(nx, ny);
    i1 = max(1, i_tot - halfwidth);
    i2 = min(nx, i_tot + halfwidth);
    j1 = max(1, j_tot - halfwidth);
    j2 = min(ny, j_tot + halfwidth);
    rhs_grid(i1:i2, j1:j2) = amplitude;
    rhs = rhs_grid(:);
end

function rhs = build_point_box_rhs(x_int, y_int, x0, y0, amplitude, radius_x, radius_y)
    [X, Y] = ndgrid(x_int, y_int);
    rhs_grid = zeros(numel(x_int), numel(y_int));
    x_mask = abs(X - x0) < radius_x;
    y_mask = abs(Y - y0) < radius_y;
    rhs_grid(x_mask & y_mask) = amplitude;
    rhs = rhs_grid(:);
end

function rhs = build_gaussian_rhs(x_int, y_int, x0, y0, amplitude, sigma)
    [X, Y] = ndgrid(x_int, y_int);
    r2 = (X - x0) .^ 2 + (Y - y0) .^ 2;
    rhs_grid = amplitude * exp(-0.5 * r2 / max(sigma ^ 2, eps));
    rhs = rhs_grid(:);
end

function [nu, nvx, nvy, nv, nfull] = state_sizes(nx, ny)
    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    nfull = nu + nv + 3 * nu;
end

function K = build_pml_stiffness(nx, ny, hx, hy, pml)
    Ntot = nx * ny;
    K = spalloc(Ntot, Ntot, 5 * Ntot);
    idx = @(i, j) i + (j - 1) * nx;

    for i = 1:nx
        for j = 1:ny
            row = idx(i, j);

            a_x = 1.0 ./ pml.s_x(i + 1);
            a_p = 1.0 ./ pml.s_x_half(i + 1);
            a_m = 1.0 ./ pml.s_x_half(i);
            K(row, row) = K(row, row) + (-(a_p + a_m) * a_x) / (hx * hx);
            if i + 1 <= nx
                K(row, idx(i + 1, j)) = K(row, idx(i + 1, j)) + (a_x * a_p) / (hx * hx);
            end
            if i - 1 >= 1
                K(row, idx(i - 1, j)) = K(row, idx(i - 1, j)) + (a_x * a_m) / (hx * hx);
            end

            b_y = 1.0 ./ pml.s_y(j + 1);
            b_p = 1.0 ./ pml.s_y_half(j + 1);
            b_m = 1.0 ./ pml.s_y_half(j);
            K(row, row) = K(row, row) + (-(b_p + b_m) * b_y) / (hy * hy);
            if j + 1 <= ny
                K(row, idx(i, j + 1)) = K(row, idx(i, j + 1)) + (b_y * b_p) / (hy * hy);
            end
            if j - 1 >= 1
                K(row, idx(i, j - 1)) = K(row, idx(i, j - 1)) + (b_y * b_m) / (hy * hy);
            end
        end
    end
end

function pml = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml)
    sigma_x = zeros(nx + 2, 1);
    sigma_y = zeros(ny + 2, 1);

    for i = 1:npml
        if ispml == 1
            val = sigma_max * ((npml - (i - 1)) / npml) ^ 2;
        else
            val = 0;
        end
        sigma_x(i) = val;
        sigma_x(end + 1 - i) = val;
        sigma_y(i) = val;
        sigma_y(end + 1 - i) = val;
    end

    s_x = 1.0 + 1i * sigma_x / omega;
    s_y = 1.0 + 1i * sigma_y / omega;
    s_x_half = 0.5 * (s_x(1:end-1) + s_x(2:end));
    s_y_half = 0.5 * (s_y(1:end-1) + s_y(2:end));

    pml = struct('sigma_x', sigma_x, 'sigma_y', sigma_y, ...
                 's_x', s_x, 's_y', s_y, ...
                 's_x_half', s_x_half, 's_y_half', s_y_half);
end

function [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt_init)
    dt = dt_init;
    if omega * dt / 2 >= 0.98
        dt = 0.98 * 2 / omega;
    end
    for it = 1:6
        omega_ts = (2 / dt) * asin(omega * dt / 2);
        nt = max(10, round(2 * pi / (omega_ts * dt)));
        dt = 2 * pi / (omega_ts * nt);
    end
    omega_ts = (2 / dt) * asin(omega * dt / 2);
    c_w = cos(omega_ts * dt / 2);
end

function [x, flag, relres, iter, resvec] = gmres_restarted_verbose(Afun, b, restart, tol, maxit_outer, x0, verbose, reorth)
    if nargin < 8
        reorth = false;
    end
    if nargin < 7
        verbose = true;
    end
    if nargin < 6 || isempty(x0)
        x0 = zeros(size(b));
    end

    x = x0;
    nb = norm(b);
    if nb == 0
        flag = 0;
        relres = 0;
        iter = [0, 0];
        resvec = 0;
        return;
    end

    r = b - Afun(x);
    beta = norm(r);
    resvec = beta;

    if verbose
        fprintf('GMRES init: ||r0||/||b|| = %.3e\n', beta / nb);
    end
    if beta / nb <= tol
        flag = 0;
        relres = beta / nb;
        iter = [0, 0];
        return;
    end

    flag = 1;
    iter = [0, 0];

    for outer = 1:maxit_outer
        V = zeros(numel(b), restart + 1);
        H = zeros(restart + 1, restart);
        cs = zeros(restart, 1);
        sn = zeros(restart, 1);
        e1 = zeros(restart + 1, 1);
        e1(1) = 1;

        V(:, 1) = r / beta;
        g = beta * e1;
        inner_last = 0;
        early_restart = false;

        for j = 1:restart
            inner_last = j;
            w = Afun(V(:, j));
            for i = 1:j
                H(i, j) = V(:, i)' * w;
                w = w - H(i, j) * V(:, i);
            end
            if reorth
                for i = 1:j
                    hcorr = V(:, i)' * w;
                    H(i, j) = H(i, j) + hcorr;
                    w = w - hcorr * V(:, i);
                end
            end
            H(j + 1, j) = norm(w);
            if H(j + 1, j) ~= 0
                V(:, j + 1) = w / H(j + 1, j);
            else
                V(:, j + 1) = zeros(size(w));
            end

            for i = 1:(j - 1)
                temp = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
                H(i + 1, j) = -sn(i) * H(i, j) + cs(i) * H(i + 1, j);
                H(i, j) = temp;
            end

            [cs(j), sn(j)] = givens_rotation(H(j, j), H(j + 1, j));
            temp = cs(j) * H(j, j) + sn(j) * H(j + 1, j);
            H(j + 1, j) = 0;
            H(j, j) = temp;

            temp = cs(j) * g(j) + sn(j) * g(j + 1);
            g(j + 1) = -sn(j) * g(j) + cs(j) * g(j + 1);
            g(j) = temp;

            resnorm = abs(g(j + 1));
            resvec(end + 1, 1) = resnorm;
            if verbose
                fprintf('  GMRES iter %4d: ||r||/||b|| = %.3e\n', (outer - 1) * restart + j, resnorm / nb);
            end

            if resnorm / nb <= tol
                y = solve_projected_gmres(H(1:j, 1:j), g(1:j));
                x = x + V(:, 1:j) * y;
                r = b - Afun(x);
                beta = norm(r);
                relres = beta / nb;
                iter = [outer, j];
                if relres <= tol
                    flag = 0;
                    return;
                end
                early_restart = true;
                break;
            end
        end

        if early_restart
            continue;
        end

        y = solve_projected_gmres(H(1:inner_last, 1:inner_last), g(1:inner_last));
        x = x + V(:, 1:inner_last) * y;
        r = b - Afun(x);
        beta = norm(r);
        iter = [outer, inner_last];
        if beta / nb <= tol
            flag = 0;
            relres = beta / nb;
            return;
        end
    end

    relres = beta / nb;
end

function y = solve_projected_gmres(Hsmall, gsmall)
    [U, S, V] = svd(Hsmall, 'econ');
    s = diag(S);
    if isempty(s)
        y = zeros(size(Hsmall, 2), 1);
        return;
    end
    tol = max(size(Hsmall)) * eps(max(s));
    keep = s > tol;
    if ~any(keep)
        y = zeros(size(Hsmall, 2), 1);
        return;
    end
    y = V(:, keep) * ((U(:, keep)' * gsmall) ./ s(keep));
end

function it = iteration_count(iter_pair, restart)
    if isequal(iter_pair, [0, 0])
        it = 0;
    else
        it = (iter_pair(1) - 1) * restart + iter_pair(2);
    end
end

function [c, s] = givens_rotation(a, b)
    if b == 0
        c = 1;
        s = 0;
    elseif abs(b) > abs(a)
        t = a / b;
        s = 1 / sqrt(1 + t ^ 2);
        c = s * t;
    else
        t = b / a;
        c = 1 / sqrt(1 + t ^ 2);
        s = c * t;
    end
end

function Uphys = interior_to_physical(u_int, prob)
    U = zeros(prob.nx + 2, prob.ny + 2);
    U(2:end-1, 2:end-1) = reshape(u_int, prob.nx, prob.ny);
    Uphys = U(prob.I + 1, prob.J + 1);
end

function make_plots(results, prob)
    cvals = results.c_right_values(:).';
    nc = numel(cvals);

    make_wave_helm_comparison_plots(results, prob, cvals);

    figure('Name', 'WaveHoltz total field (physical)');
    tlo = tiledlayout(1, nc, 'Padding', 'compact', 'TileSpacing', 'compact');
    for k = 1:nc
        nexttile;
        plot_physical_field(prob, real(results.u_wave_phys(:, :, k)), true);
        title(sprintf('Re(u), c_{right}=%.3f', cvals(k)));
    end
    title(tlo, 'Total field: WaveHoltz (physical domain)');

    if ~isempty(results.incident)
        figure('Name', 'Scattering-like field (physical)');
        tlo = tiledlayout(1, nc, 'Padding', 'compact', 'TileSpacing', 'compact');
        for k = 1:nc
            nexttile;
            Uscat = results.u_wave_phys(:, :, k) - results.incident.u_phys;
            plot_physical_field(prob, real(Uscat), true);
            title(sprintf('Re(u-u_{inc}), c_{right}=%.3f', cvals(k)));
        end
        title(tlo, 'Scattering-like field using homogeneous c=1 reference');
    end

    if nc >= 2
        Uref = results.u_wave_phys(:, :, 1);
        figure('Name', 'Difference vs first c_right');
        tlo = tiledlayout(1, nc - 1, 'Padding', 'compact', 'TileSpacing', 'compact');
        for k = 2:nc
            nexttile;
            plot_physical_field(prob, real(results.u_wave_phys(:, :, k) - Uref), true);
            title(sprintf('Re(u_{%.3f}-u_{%.3f})', cvals(k), cvals(1)));
        end
        title(tlo, 'Difference maps in physical domain');
    end
end

function make_wave_helm_comparison_plots(results, prob, cvals)
    nc = numel(cvals);
    if nc == 1
        figure('Name', 'WaveHoltz vs Helmholtz comparison (physical)');
        subplot(2, 2, 1);
        plot_physical_field(prob, real(results.u_helm_phys(:, :, 1)), true);
        title(sprintf('Helmholtz, c=%.3f', cvals(1)));

        subplot(2, 2, 2);
        plot_physical_field(prob, real(results.u_wave_phys(:, :, 1)), true);
        title(sprintf('WaveHoltz, c=%.3f', cvals(1)));

        subplot(2, 2, 3);
        plot_physical_field(prob, real(results.u_wave_phys(:, :, 1) - results.u_helm_phys(:, :, 1)), true);
        title(sprintf('Wave - Helm, rel=%.2e', results.rel_wave_vs_helm_phys(1)));

        subplot(2, 2, 4);
        plot_residual_history(results.info(1).resvec);
        title('WaveHoltz GMRES history');
        return;
    end

    figure('Name', 'WaveHoltz vs Helmholtz comparison (physical)');
    tlo = tiledlayout(nc, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    for k = 1:nc
        nexttile;
        plot_physical_field(prob, real(results.u_helm_phys(:, :, k)), true);
        title(sprintf('Helm, c=%.3f', cvals(k)));

        nexttile;
        plot_physical_field(prob, real(results.u_wave_phys(:, :, k)), true);
        title(sprintf('WH, c=%.3f', cvals(k)));

        nexttile;
        plot_physical_field(prob, real(results.u_wave_phys(:, :, k) - results.u_helm_phys(:, :, k)), true);
        title(sprintf('WH-Helm, rel=%.1e', results.rel_wave_vs_helm_phys(k)));
    end
    title(tlo, 'WaveHoltz / Helmholtz / Difference');
end

function plot_physical_field(prob, field_phys, overlay_material)
    imagesc(prob.x_phys, prob.y_phys, field_phys.');
    set(gca, 'YDir', 'normal');
    axis equal tight;
    colorbar;
    set_balanced_color_limits(field_phys);
    xlabel('x');
    ylabel('y');
    if overlay_material
        overlay_material_mask(prob);
    end
end

function overlay_material_mask(prob)
    if ~isfield(prob, 'chi_right_grid') || isempty(prob.chi_right_grid)
        return;
    end
    chi_phys = double(prob.chi_right_grid(prob.I, prob.J));
    if ~any(chi_phys(:))
        return;
    end
    hold on;
    contour(prob.x_phys, prob.y_phys, chi_phys.', [0.5, 0.5], ...
        'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.45);
    hold off;
end

function set_balanced_color_limits(field_phys)
    vals = real(field_phys(:));
    vals = vals(isfinite(vals));
    if isempty(vals)
        return;
    end

    vmax = max(abs(vals));
    if vmax <= 0
        return;
    end

    caxis([-vmax, vmax]);
end

function plot_residual_history(resvec)
    if isempty(resvec)
        text(0.5, 0.5, 'no residual history', 'HorizontalAlignment', 'center');
        axis off;
        return;
    end
    relhist = abs(resvec(:)) / max(abs(resvec(1)), 1e-16);
    semilogy(relhist, 'LineWidth', 1.5);
    grid on;
    xlabel('GMRES iteration');
    ylabel('relative residual');
end

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
