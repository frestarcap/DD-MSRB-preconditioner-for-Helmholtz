function results = waveholtz_2d_pml_v5_recovery_consistency_check(cfg)
    % Standalone consistency check for v5 recovery/lift logic.
    %
    % It compares:
    %   1) x_final -> uhat_timeproj against the direct Helmholtz phasor
    %   2) uhat -> xhat_full against the time-projected full-state phasor
    %   3) real(xhat_full) against the converged real WaveHoltz state

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults_local(cfg);
    prob = build_problem_local(cfg);
    rhs = build_rhs_local(prob, cfg);
    params = cfg.check_params(:);
    ns = numel(params);

    fprintf('=== WaveHoltz v5 Recovery Consistency Check ===\n');
    fprintf('Nphys=%d, npml=%d, total interior=%d x %d\n', cfg.Nphys, cfg.npml, prob.nx, prob.ny);
    fprintf('omega(helm)=%.6f, omega(ts)=%.6f, dt=%.3e, nt=%d\n', prob.omega, prob.omega_ts, prob.dt, prob.nt);
    fprintf('check params = %s\n', mat2str(params.', 4));

    results = struct();
    results.cfg = cfg;
    results.problem = prob;
    results.rhs = rhs;
    results.params = params;
    results.x_final = zeros(prob.nfull, ns);
    results.u_timeproj = zeros(prob.Ntot, ns);
    results.u_helm = zeros(prob.Ntot, ns);
    results.xhat_timeproj = zeros(prob.nfull, ns);
    results.xhat_from_u_timeproj = zeros(prob.nfull, ns);
    results.xhat_from_u_helm = zeros(prob.nfull, ns);
    results.xlift_from_u_timeproj = zeros(prob.nfull, ns);
    results.xlift_from_u_helm = zeros(prob.nfull, ns);
    results.rel_u_timeproj_vs_helm = zeros(ns, 1);
    results.rel_xhat_from_u_timeproj_vs_timeproj = zeros(ns, 1);
    results.rel_xhat_from_u_helm_vs_timeproj = zeros(ns, 1);
    results.rel_xlift_from_u_timeproj_vs_xfinal = zeros(ns, 1);
    results.rel_xlift_from_u_helm_vs_xfinal = zeros(ns, 1);
    results.rel_real_xhat_timeproj_vs_xfinal = zeros(ns, 1);
    results.rel_ublock_timeproj_vs_u_timeproj = zeros(ns, 1);
    results.rel_ublock_timeproj_vs_u_helm = zeros(ns, 1);
    results.solve_time = zeros(ns, 1);
    results.recover_time = zeros(ns, 1);
    results.iter = zeros(ns, 1);
    results.relres = zeros(ns, 1);

    for k = 1:ns
        c_right = params(k);
        [x_final, u_timeproj, ~, info] = solve_waveholtz_fullstate_solution_local(prob, cfg, rhs, c_right, zeros(prob.nfull, 1));
        u_helm = build_helmholtz_operator_local(prob, c_right) \ rhs;
        xhat_timeproj = recover_fullstate_phasor_timeproj_local(prob, x_final, rhs, c_right);
        xhat_from_u_timeproj = recover_fullstate_frequency_from_u_local(prob, u_timeproj, c_right);
        xhat_from_u_helm = recover_fullstate_frequency_from_u_local(prob, u_helm, c_right);
        xlift_from_u_timeproj = real(xhat_from_u_timeproj);
        xlift_from_u_helm = real(xhat_from_u_helm);

        results.x_final(:, k) = x_final;
        results.u_timeproj(:, k) = u_timeproj;
        results.u_helm(:, k) = u_helm;
        results.xhat_timeproj(:, k) = xhat_timeproj;
        results.xhat_from_u_timeproj(:, k) = xhat_from_u_timeproj;
        results.xhat_from_u_helm(:, k) = xhat_from_u_helm;
        results.xlift_from_u_timeproj(:, k) = xlift_from_u_timeproj;
        results.xlift_from_u_helm(:, k) = xlift_from_u_helm;
        results.rel_u_timeproj_vs_helm(k) = relerr(u_timeproj, u_helm);
        results.rel_xhat_from_u_timeproj_vs_timeproj(k) = relerr(xhat_from_u_timeproj, xhat_timeproj);
        results.rel_xhat_from_u_helm_vs_timeproj(k) = relerr(xhat_from_u_helm, xhat_timeproj);
        results.rel_xlift_from_u_timeproj_vs_xfinal(k) = relerr(xlift_from_u_timeproj, x_final);
        results.rel_xlift_from_u_helm_vs_xfinal(k) = relerr(xlift_from_u_helm, x_final);
        results.rel_real_xhat_timeproj_vs_xfinal(k) = relerr(real(xhat_timeproj), x_final);
        results.rel_ublock_timeproj_vs_u_timeproj(k) = relerr(xhat_timeproj(1:prob.nu), u_timeproj);
        results.rel_ublock_timeproj_vs_u_helm(k) = relerr(xhat_timeproj(1:prob.nu), u_helm);
        results.solve_time(k) = info.solve_time;
        results.recover_time(k) = info.recovery_time;
        results.iter(k) = info.iter_count;
        results.relres(k) = info.relres;

        fprintf(['param %2d/%2d: c_right=%.4f, iter=%3d, relres=%.3e, ' ...
            'u(timeproj)-u(helm)=%.3e, xhat(u_timeproj)-xhat(timeproj)=%.3e, ' ...
            'xhat(u_helm)-xhat(timeproj)=%.3e, real(xhat(u_timeproj))-x=%.3e, real(xhat(timeproj))-x=%.3e\n'], ...
            k, ns, c_right, info.iter_count, info.relres, ...
            results.rel_u_timeproj_vs_helm(k), ...
            results.rel_xhat_from_u_timeproj_vs_timeproj(k), ...
            results.rel_xhat_from_u_helm_vs_timeproj(k), ...
            results.rel_xlift_from_u_timeproj_vs_xfinal(k), ...
            results.rel_real_xhat_timeproj_vs_xfinal(k));
    end

    results.summary = struct();
    results.summary.rel_u_timeproj_vs_helm = stats3(results.rel_u_timeproj_vs_helm);
    results.summary.rel_xhat_from_u_timeproj_vs_timeproj = stats3(results.rel_xhat_from_u_timeproj_vs_timeproj);
    results.summary.rel_xhat_from_u_helm_vs_timeproj = stats3(results.rel_xhat_from_u_helm_vs_timeproj);
    results.summary.rel_xlift_from_u_timeproj_vs_xfinal = stats3(results.rel_xlift_from_u_timeproj_vs_xfinal);
    results.summary.rel_xlift_from_u_helm_vs_xfinal = stats3(results.rel_xlift_from_u_helm_vs_xfinal);
    results.summary.rel_real_xhat_timeproj_vs_xfinal = stats3(results.rel_real_xhat_timeproj_vs_xfinal);
    results.summary.rel_ublock_timeproj_vs_u_timeproj = stats3(results.rel_ublock_timeproj_vs_u_timeproj);
    results.summary.rel_ublock_timeproj_vs_u_helm = stats3(results.rel_ublock_timeproj_vs_u_helm);

    fprintf('\n=== Consistency Summary ===\n');
    print_stats_line('u(timeproj) vs u(helm)', results.summary.rel_u_timeproj_vs_helm);
    print_stats_line('xhat(u_timeproj) vs xhat(timeproj)', results.summary.rel_xhat_from_u_timeproj_vs_timeproj);
    print_stats_line('xhat(u_helm) vs xhat(timeproj)', results.summary.rel_xhat_from_u_helm_vs_timeproj);
    print_stats_line('real(xhat(u_timeproj)) vs x_final', results.summary.rel_xlift_from_u_timeproj_vs_xfinal);
    print_stats_line('real(xhat(u_helm)) vs x_final', results.summary.rel_xlift_from_u_helm_vs_xfinal);
    print_stats_line('real(xhat(timeproj)) vs x_final', results.summary.rel_real_xhat_timeproj_vs_xfinal);
    print_stats_line('u-block(xhat_timeproj) vs u_timeproj', results.summary.rel_ublock_timeproj_vs_u_timeproj);
    print_stats_line('u-block(xhat_timeproj) vs u_helm', results.summary.rel_ublock_timeproj_vs_u_helm);
end

function print_stats_line(label, s)
    fprintf('%-36s : %.3e / %.3e / %.3e\n', label, s(1), s(2), s(3));
end

function cfg = apply_defaults_local(cfg)
    cfg = set_default_local(cfg, 'L', 1.0);
    cfg = set_default_local(cfg, 'omega', 33.5);
    cfg = set_default_local(cfg, 'Nphys', 125);
    cfg = set_default_local(cfg, 'ispml', 1);
    cfg = set_default_local(cfg, 'npml', []);
    cfg = set_default_local(cfg, 'sigma_max', 80);
    cfg = set_default_local(cfg, 'CFL', 0.5);
    cfg = set_default_local(cfg, 'filter_shift', 0.25);
    cfg = set_default_local(cfg, 'interface_x', []);
    cfg = set_default_local(cfg, 'source_col_phys', []);
    cfg = set_default_local(cfg, 'source_row_phys', []);
    cfg = set_default_local(cfg, 'source_type', 'point');
    cfg = set_default_local(cfg, 'point_amplitude', 1.0);
    cfg = set_default_local(cfg, 'point_source_profile', 'box');
    cfg = set_default_local(cfg, 'point_support_halfwidth', 0);
    cfg = set_default_local(cfg, 'point_box_radius_x', []);
    cfg = set_default_local(cfg, 'point_box_radius_y', []);
    cfg = set_default_local(cfg, 'gaussian_amplitude', 1.0);
    cfg = set_default_local(cfg, 'gaussian_sigma', []);
    cfg = set_default_local(cfg, 'train_params', linspace(0.8, 0.95, 30));
    cfg = set_default_local(cfg, 'test_params', 0.875);
    cfg = set_default_local(cfg, 'check_params', []);
    cfg = set_default_local(cfg, 'wh_tol', 1e-12);
    cfg = set_default_local(cfg, 'wh_gmres_restart', 20);
    cfg = set_default_local(cfg, 'wh_gmres_maxit', 40);
    cfg = set_default_local(cfg, 'wh_verbose_gmres', false);

    if isempty(cfg.npml)
        cfg.npml = cfg.ispml * ceil(0.25 * cfg.Nphys);
    end
    if isempty(cfg.interface_x)
        cfg.interface_x = 0.5 * cfg.L;
    end
    if isempty(cfg.source_col_phys)
        cfg.source_col_phys = max(2, round(0.20 * cfg.Nphys));
    end
    if isempty(cfg.source_row_phys)
        cfg.source_row_phys = round((cfg.Nphys + 1) / 2);
    end
    if isempty(cfg.gaussian_sigma)
        cfg.gaussian_sigma = 3 * min(cfg.L / (cfg.Nphys + 1), cfg.L / (cfg.Nphys + 1));
    end
    if isempty(cfg.point_box_radius_x)
        cfg.point_box_radius_x = 2 * cfg.L / (cfg.Nphys + 1);
    end
    if isempty(cfg.point_box_radius_y)
        cfg.point_box_radius_y = 2 * cfg.L / (cfg.Nphys + 1);
    end
    if isempty(cfg.check_params)
        cfg.check_params = cfg.test_params;
    end
    cfg.train_params = unique(cfg.train_params(:).');
    cfg.test_params = unique(cfg.test_params(:).');
    cfg.check_params = unique(cfg.check_params(:).');
end

function prob = build_problem_local(cfg)
    prob = struct();
    prob.L = cfg.L;
    prob.omega = cfg.omega;
    prob.nx_phys = cfg.Nphys;
    prob.ny_phys = cfg.Nphys;
    prob.hx = cfg.L / (prob.nx_phys + 1);
    prob.hy = cfg.L / (prob.ny_phys + 1);
    prob.npml = cfg.npml;
    prob.nx = prob.nx_phys + 2 * prob.npml;
    prob.ny = prob.ny_phys + 2 * prob.npml;
    prob.Ntot = prob.nx * prob.ny;
    prob.I = (prob.npml + 1):(prob.npml + prob.nx_phys);
    prob.J = (prob.npml + 1):(prob.npml + prob.ny_phys);

    prob.c_max = max([1.0; cfg.train_params(:); cfg.test_params(:); cfg.check_params(:)]);
    dt0 = cfg.CFL * min(prob.hx, prob.hy) / max(prob.c_max, 1e-14);
    [prob.omega_ts, prob.c_w, prob.nt, prob.dt] = match_omega_and_period_local(prob.omega, dt0);
    prob.T_ts = prob.nt * prob.dt;

    Lpml_x = prob.npml * prob.hx;
    Lpml_y = prob.npml * prob.hy;
    prob.x = linspace(-Lpml_x, cfg.L + Lpml_x, prob.nx + 2);
    prob.y = linspace(-Lpml_y, cfg.L + Lpml_y, prob.ny + 2);
    prob.x_int = prob.x(2:end-1).';
    prob.y_int = prob.y(2:end-1).';
    prob.x_phys = prob.x(prob.I + 1).';
    prob.y_phys = prob.y(prob.J + 1).';

    prob.pml_helm = build_pml_outer_local(prob.nx, prob.ny, prob.omega, cfg.npml, cfg.sigma_max, cfg.ispml);
    prob.pml_wave = prob.pml_helm;
    prob.pml_wave.sigma_x = prob.pml_helm.sigma_x / prob.c_w;
    prob.pml_wave.sigma_y = prob.pml_helm.sigma_y / prob.c_w;
    prob.pml_wave.c_w_ref = prob.c_w;
    prob.pml_wave.sigma_scaled_for_wave = true;

    [prob.nu, prob.nvx, prob.nvy, prob.nv, prob.nfull] = state_sizes_local(prob.nx, prob.ny);
    prob.chi_right_grid = repmat(prob.x_int > cfg.interface_x, 1, prob.ny);
    prob.chi_right = prob.chi_right_grid(:);
    prob.K_pml = build_pml_stiffness_local(prob.nx, prob.ny, prob.hx, prob.hy, prob.pml_helm);
    prob.A_base = prob.K_pml + (prob.omega ^ 2) * speye(prob.Ntot);
    prob.A_right = spdiags((prob.omega ^ 2) * double(prob.chi_right), 0, prob.Ntot, prob.Ntot);
end

function rhs = build_rhs_local(prob, cfg)
    i_tot = prob.I(cfg.source_col_phys);
    j_tot = prob.J(cfg.source_row_phys);
    if strcmp(cfg.source_type, 'point')
        if strcmp(cfg.point_source_profile, 'box')
            rhs = build_point_box_rhs_local(prob.x_int, prob.y_int, prob.x_phys(cfg.source_col_phys), prob.y_phys(cfg.source_row_phys), ...
                cfg.point_amplitude, cfg.point_box_radius_x, cfg.point_box_radius_y);
        else
            rhs = build_point_cell_rhs_local(prob.nx, prob.ny, i_tot, j_tot, cfg.point_amplitude, cfg.point_support_halfwidth);
        end
        return;
    end

    rhs = build_gaussian_rhs_local(prob.x_int, prob.y_int, prob.x_phys(cfg.source_col_phys), prob.y_phys(cfg.source_row_phys), ...
        cfg.gaussian_amplitude, cfg.gaussian_sigma);
end

function [x_sol, uhat, b_vec, info] = solve_waveholtz_fullstate_solution_local(prob, cfg, rhs, c_right, x0)
    [b_vec, Afun] = build_waveholtz_operator_local(prob, cfg, rhs, c_right);
    tsolve = tic;
    [x_sol, flag, relres, iter, resvec] = gmres_restarted_verbose_local( ...
        Afun, b_vec, cfg.wh_gmres_restart, cfg.wh_tol, cfg.wh_gmres_maxit, x0, cfg.wh_verbose_gmres);
    solve_time = toc(tsolve);

    trec = tic;
    uhat = recover_u_phasor_timeproj_local(prob, x_sol, rhs, c_right);
    recovery_time = toc(trec);

    info = struct();
    info.flag = flag;
    info.relres = relres;
    info.iter = iter;
    info.iter_count = iteration_count_local(iter, cfg.wh_gmres_restart);
    info.resvec = resvec;
    info.solve_time = solve_time;
    info.recovery_time = recovery_time;
end

function [b_vec, Afun] = build_waveholtz_operator_local(prob, cfg, rhs, c_right)
    b_vec = apply_pi_piecewise_local(prob, zeros(prob.nfull, 1), rhs, true, c_right, cfg.filter_shift);
    Afun = @(x) x - apply_pi_piecewise_local(prob, x, zeros(prob.nu, 1), false, c_right, cfg.filter_shift);
end

function uhat = recover_u_phasor_timeproj_local(prob, x_state, rhs, c_right)
    Wconv = solve_wave_one_period_piecewise_local(prob, x_state, rhs, c_right);
    t_grid = (0:prob.nt).' * prob.dt;
    q_cos = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, cos(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
    q_sin = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, sin(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
    uhat = q_cos + 1i * q_sin;
end

function xhat = recover_fullstate_phasor_timeproj_local(prob, x_state, rhs, c_right)
    Wconv = solve_wave_one_period_piecewise_local(prob, x_state, rhs, c_right);
    t_grid = (0:prob.nt).' * prob.dt;
    q_cos = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, cos(prob.omega_ts * t_grid), Wconv), 1).';
    q_sin = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, sin(prob.omega_ts * t_grid), Wconv), 1).';
    xhat = q_cos + 1i * q_sin;
end

function xhat_full = recover_fullstate_frequency_from_u_local(prob, uhat, c_right)
    [sigma_x_u, sigma_y_u, sigma_x_vx, sigma_y_vy, theta_coeff] = build_discrete_lift_grids_local(prob, c_right);
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

function [sigma_x_u, sigma_y_u, sigma_x_vx, sigma_y_vy, theta_coeff] = build_discrete_lift_grids_local(prob, c_right)
    sx_full = zeros(prob.nx + 2, prob.ny + 2);
    sy_full = zeros(prob.nx + 2, prob.ny + 2);
    for i = 1:(prob.nx + 2), sx_full(i, :) = prob.pml_wave.sigma_x(i); end
    for j = 1:(prob.ny + 2), sy_full(:, j) = prob.pml_wave.sigma_y(j); end
    sigma_x_u = sx_full(2:end-1, 2:end-1);
    sigma_y_u = sy_full(2:end-1, 2:end-1);
    sigma_x_vx = 0.5 * (sx_full(1:end-1, 2:end-1) + sx_full(2:end, 2:end-1));
    sigma_y_vy = 0.5 * (sy_full(2:end-1, 1:end-1) + sy_full(2:end-1, 2:end));
    c_sq = build_piecewise_c_squared_local(prob, c_right);
    c_sq_full = ones(prob.nx + 2, prob.ny + 2);
    c_sq_full(2:end-1, 2:end-1) = c_sq;
    theta_coeff = (sx_full(2:end-1, 2:end-1) .* sy_full(2:end-1, 2:end-1)) .* (prob.c_w ^ 2) ./ c_sq_full(2:end-1, 2:end-1);
end

function Ahelm = build_helmholtz_operator_local(prob, c_right)
    alpha = (1 / c_right^2) - 1;
    Ahelm = prob.A_base + alpha * prob.A_right;
end

function c_sq = build_piecewise_c_squared_local(prob, c_right)
    c_sq = ones(prob.nx, prob.ny);
    c_sq(prob.chi_right_grid) = c_right ^ 2;
end

function Pi = apply_pi_piecewise_local(prob, x0, rhs, source_on, c_right, shift)
    if source_on
        rhs_use = rhs;
    else
        rhs_use = zeros(prob.nu, 1);
    end
    W = solve_wave_one_period_piecewise_local(prob, x0, rhs_use, c_right);
    t_grid = (0:prob.nt).' * prob.dt;
    weights = cos(prob.omega_ts * t_grid) - shift;
    Pi = (2.0 / max(prob.T_ts, 1e-16)) * trapz(t_grid, bsxfun(@times, weights, W), 1).';
end

function W = solve_wave_one_period_piecewise_local(prob, x0, rhs, c_right)
    sx_full = zeros(prob.nx + 2, prob.ny + 2);
    sy_full = zeros(prob.nx + 2, prob.ny + 2);
    for i = 1:(prob.nx + 2), sx_full(i, :) = prob.pml_wave.sigma_x(i); end
    for j = 1:(prob.ny + 2), sy_full(:, j) = prob.pml_wave.sigma_y(j); end

    sigma_x_vx = 0.5 * (sx_full(1:end-1, :) + sx_full(2:end, :));
    sigma_y_vy = 0.5 * (sy_full(:, 1:end-1) + sy_full(:, 2:end));

    c_sq = build_piecewise_c_squared_local(prob, c_right);
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

    W = zeros(prob.nt + 1, prob.nfull);
    v0_vec = [reshape(vx(:, 2:end-1), prob.nvx, 1); reshape(vy(2:end-1, :), prob.nvy, 1)];
    W(1, :) = [reshape(u(2:end-1, 2:end-1), prob.nu, 1); v0_vec; ...
               reshape(psi1(2:end-1, 2:end-1), prob.nu, 1); ...
               reshape(psi2(2:end-1, 2:end-1), prob.nu, 1); ...
               reshape(theta(2:end-1, 2:end-1), prob.nu, 1)].';

    for n = 1:prob.nt
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

function rhs = build_point_cell_rhs_local(nx, ny, i_tot, j_tot, amplitude, halfwidth)
    rhs_grid = zeros(nx, ny);
    i1 = max(1, i_tot - halfwidth);
    i2 = min(nx, i_tot + halfwidth);
    j1 = max(1, j_tot - halfwidth);
    j2 = min(ny, j_tot + halfwidth);
    rhs_grid(i1:i2, j1:j2) = amplitude;
    rhs = rhs_grid(:);
end

function rhs = build_point_box_rhs_local(x_int, y_int, x0, y0, amplitude, radius_x, radius_y)
    [X, Y] = ndgrid(x_int, y_int);
    rhs_grid = zeros(numel(x_int), numel(y_int));
    rhs_grid(abs(X - x0) < radius_x & abs(Y - y0) < radius_y) = amplitude;
    rhs = rhs_grid(:);
end

function rhs = build_gaussian_rhs_local(x_int, y_int, x0, y0, amplitude, sigma)
    [X, Y] = ndgrid(x_int, y_int);
    r2 = (X - x0) .^ 2 + (Y - y0) .^ 2;
    rhs_grid = amplitude * exp(-0.5 * r2 / max(sigma ^ 2, eps));
    rhs = rhs_grid(:);
end

function [nu, nvx, nvy, nv, nfull] = state_sizes_local(nx, ny)
    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    nfull = nu + nv + 3 * nu;
end

function K = build_pml_stiffness_local(nx, ny, hx, hy, pml)
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
            if i + 1 <= nx, K(row, idx(i + 1, j)) = K(row, idx(i + 1, j)) + (a_x * a_p) / (hx * hx); end
            if i - 1 >= 1, K(row, idx(i - 1, j)) = K(row, idx(i - 1, j)) + (a_x * a_m) / (hx * hx); end

            b_y = 1.0 ./ pml.s_y(j + 1);
            b_p = 1.0 ./ pml.s_y_half(j + 1);
            b_m = 1.0 ./ pml.s_y_half(j);
            K(row, row) = K(row, row) + (-(b_p + b_m) * b_y) / (hy * hy);
            if j + 1 <= ny, K(row, idx(i, j + 1)) = K(row, idx(i, j + 1)) + (b_y * b_p) / (hy * hy); end
            if j - 1 >= 1, K(row, idx(i, j - 1)) = K(row, idx(i, j - 1)) + (b_y * b_m) / (hy * hy); end
        end
    end
end

function pml = build_pml_outer_local(nx, ny, omega, npml, sigma_max, ispml)
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
    pml = struct('sigma_x', sigma_x, 'sigma_y', sigma_y, 's_x', s_x, 's_y', s_y, ...
        's_x_half', s_x_half, 's_y_half', s_y_half);
end

function [omega_ts, c_w, nt, dt] = match_omega_and_period_local(omega, dt_init)
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

function [x, flag, relres, iter, resvec] = gmres_restarted_verbose_local(Afun, b, restart, tol, maxit_outer, x0, verbose)
    if nargin < 7, verbose = false; end
    if nargin < 6 || isempty(x0), x0 = zeros(size(b)); end
    x = x0;
    nb = norm(b);
    if nb == 0
        flag = 0; relres = 0; iter = [0, 0]; resvec = 0; return;
    end

    r = b - Afun(x);
    beta = norm(r);
    resvec = beta;
    if verbose, fprintf('GMRES init: ||r0||/||b|| = %.3e\n', beta / nb); end
    if beta / nb <= tol
        flag = 0; relres = beta / nb; iter = [0, 0]; return;
    end

    flag = 1;
    iter = [0, 0];
    for outer = 1:maxit_outer
        V = zeros(numel(b), restart + 1);
        H = zeros(restart + 1, restart);
        cs = zeros(restart, 1);
        sn = zeros(restart, 1);
        g = zeros(restart + 1, 1);
        V(:, 1) = r / beta;
        g(1) = beta;

        for j = 1:restart
            w = Afun(V(:, j));
            for i = 1:j
                H(i, j) = V(:, i)' * w;
                w = w - H(i, j) * V(:, i);
            end
            H(j + 1, j) = norm(w);
            if H(j + 1, j) ~= 0
                V(:, j + 1) = w / H(j + 1, j);
            end

            for i = 1:(j - 1)
                temp = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
                H(i + 1, j) = -conj(sn(i)) * H(i, j) + cs(i) * H(i + 1, j);
                H(i, j) = temp;
            end
            [cs(j), sn(j)] = planerot_local(H(j:j + 1, j));
            H(j, j) = cs(j) * H(j, j) + sn(j) * H(j + 1, j);
            H(j + 1, j) = 0;
            g(j + 1) = -conj(sn(j)) * g(j);
            g(j) = cs(j) * g(j);
            resvec(end + 1, 1) = abs(g(j + 1)); %#ok<AGROW>

            if abs(g(j + 1)) / nb <= tol
                y = H(1:j, 1:j) \ g(1:j);
                x = x + V(:, 1:j) * y;
                relres = abs(g(j + 1)) / nb;
                iter = [outer - 1, j];
                flag = 0;
                return;
            end
        end

        y = H(1:restart, 1:restart) \ g(1:restart);
        x = x + V(:, 1:restart) * y;
        r = b - Afun(x);
        beta = norm(r);
        resvec(end + 1, 1) = beta; %#ok<AGROW>
        iter = [outer, restart];
        if beta / nb <= tol
            relres = beta / nb;
            flag = 0;
            return;
        end
    end
    relres = beta / nb;
end

function [c, s] = planerot_local(x)
    if x(2) == 0
        c = 1;
        s = 0;
    elseif abs(x(2)) > abs(x(1))
        t = x(1) / x(2);
        s = 1 / sqrt(1 + abs(t) ^ 2);
        c = s * t;
    else
        t = x(2) / x(1);
        c = 1 / sqrt(1 + abs(t) ^ 2);
        s = c * t;
    end
end

function count = iteration_count_local(iter, restart)
    if numel(iter) < 2
        count = iter(1);
    else
        count = iter(1) * restart + iter(2);
    end
end

function val = relerr(a, b)
    val = norm(a - b) / max(norm(b), 1e-16);
end

function s = stats3(v)
    s = [min(v(:)), mean(v(:)), max(v(:))];
end

function cfg = set_default_local(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
