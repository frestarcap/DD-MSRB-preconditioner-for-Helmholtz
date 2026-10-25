function results = waveholtz_2d_pml_scattering_autodmd(shape_ratio, cfg)
    % ============================================================
    %  Exterior scattering by a sound-soft elliptical obstacle
    %  Compare Helmholtz reference, plain/GMRES WaveHoltz, and
    %  restarted Auto-DMD accelerated WaveHoltz.
    % ============================================================

    rng(0);
    if nargin < 1 || isempty(shape_ratio)
        shape_ratio = 1.0; % a/b = 1 -> circle
    end
    if nargin < 2 || isempty(cfg)
        cfg = struct();
    end
    validateattributes(shape_ratio, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, ...
        mfilename, 'shape_ratio');

    cfg = set_default(cfg, 'L', 1.0);
    cfg = set_default(cfg, 'c', 1.0);
    cfg = set_default(cfg, 'omega', 18.0);
    cfg = set_default(cfg, 'max_iter', 400);
    cfg = set_default(cfg, 'tol', 1e-13);
    cfg = set_default(cfg, 'Nphys', 120);
    cfg = set_default(cfg, 'ispml', 1);
    cfg = set_default(cfg, 'npml', 30);
    cfg = set_default(cfg, 'sigma_max', 80);
    cfg = set_default(cfg, 'eta_boundary_min', 0.05);
    cfg = set_default(cfg, 'CFL', 0.20);
    cfg = set_default(cfg, 'relax', 0.15);
    cfg = set_default(cfg, 'show_plots', true);
    cfg = set_default(cfg, 'verbose_iter', true);
    cfg = set_default(cfg, 'reference_mode', 'gmres'); % 'plain' | 'gmres'

    cfg = set_default(cfg, 'run_gmres', true);
    cfg = set_default(cfg, 'gmres_tol', 1e-13);
    cfg = set_default(cfg, 'gmres_restart', 30);
    cfg = set_default(cfg, 'gmres_maxit', 60);
    cfg = set_default(cfg, 'gmres_verbose', false);

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
    cfg = set_default(cfg, 'dmd_window_strategy', 'rolling'); % 'rolling' | 'restart'

    reference_mode = lower(string(cfg.reference_mode));
    if reference_mode ~= "plain" && reference_mode ~= "gmres"
        error('cfg.reference_mode must be ''plain'' or ''gmres''.');
    end
    if reference_mode == "gmres"
        cfg.run_gmres = true;
    end

    L = cfg.L;
    c = cfg.c;
    omega = cfg.omega;
    max_iter = cfg.max_iter;
    tol = cfg.tol;

    % ---- Discretization on physical domain ----
    Nphys = cfg.Nphys;
    nx_phys = Nphys;
    ny_phys = Nphys;
    hx = L / (nx_phys + 1);
    hy = L / (ny_phys + 1);

    % ---- PML outside physical domain ----
    ispml = cfg.ispml;
    npml = cfg.npml;
    sigma_max = cfg.sigma_max;

    % ---- Scatterer geometry ----
    center = [L/2, L/2];
    radius_ref = 0.12 * L; % reference circle radius (same area baseline)
    semi_a = radius_ref * sqrt(shape_ratio);
    semi_b = radius_ref / sqrt(shape_ratio);
    eta_boundary_min = cfg.eta_boundary_min;

    % ---- Time step + frequency mapping ----
    dt0 = cfg.CFL * min(hx, hy) / c;
    [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt0);
    T_ts = nt * dt;

    fprintf('=== Time stepping match ===\n');
    fprintf('omega(helm)    = %.6f\n', omega);
    fprintf('omega(ts)      = %.6f\n', omega_ts);
    fprintf('c_w            = %.6f\n', c_w);
    fprintf('dt             = %.3e, nt = %d, T_ts = %.6f\n', dt, nt, T_ts);
    fprintf('shape ratio a/b= %.4f, semi axes a=%.4f, b=%.4f\n', shape_ratio, semi_a, semi_b);

    % ---- Total computational interior unknowns ----
    nx = nx_phys + 2*npml;
    ny = ny_phys + 2*npml;

    % ---- Total domain coordinates ----
    Lpml_x = npml * hx;
    Lpml_y = npml * hy;
    x = linspace(-Lpml_x, L + Lpml_x, nx + 2);
    y = linspace(-Lpml_y, L + Lpml_y, ny + 2);
    [X, ~] = ndgrid(x, y);

    % ---- Physical interior indices in total interior grid ----
    I = (npml + 1):(npml + nx_phys);
    J = (npml + 1):(npml + ny_phys);

    % ---- PML data ----
    pml_helm = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml);
    pml_wave = pml_helm;
    pml_wave.sigma_x = pml_helm.sigma_x / c_w;
    pml_wave.sigma_y = pml_helm.sigma_y / c_w;

    % ---- Geometry / ghost-point preprocessing ----
    geom = build_scatterer_geometry(nx, ny, x, y, center, semi_a, semi_b, eta_boundary_min);
    fprintf('active nodes     = %d\n', nnz(geom.active(2:end-1,2:end-1)));
    fprintf('internal nodes   = %d\n', nnz(geom.internal(2:end-1,2:end-1)));
    fprintf('ghost_x nodes    = %d\n', nnz(geom.ghost_x));
    fprintf('ghost_y nodes    = %d\n', nnz(geom.ghost_y));
    fprintf('boundary ghosts x/y = %d / %d\n', geom.boundary_count_x, geom.boundary_count_y);

    % ---- Incident plane wave ----
    inc = build_incident_wave(X, omega, c);

    % ---- Helmholtz reference system ----
    [A_scatt, b_scatt] = helmholtz_scattering_system(nx, ny, hx, hy, omega, pml_helm, geom, inc);
    t_helm = tic;
    u_helm_vec = A_scatt \ b_scatt;
    time_helm = toc(t_helm);
    u_helm = zeros(nx + 2, ny + 2);
    u_helm(2:end-1, 2:end-1) = reshape(u_helm_vec, nx, ny);
    u_helm(geom.internal) = 0;

    % ---- Plain WaveHoltz solve ----
    t_plain = tic;
    [u_wave, hist_plain, rec_plain] = waveholtz_iteration_scattering_plain( ...
        nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, pml_wave, ...
        geom, inc, max_iter, tol, A_scatt, b_scatt, cfg.relax, cfg.verbose_iter);
    time_plain = toc(t_plain);
    u_wave(geom.internal) = 0;

    % ---- GMRES WaveHoltz solve (optional baseline) ----
    u_gmres = [];
    hist_gmres = [];
    rec_gmres = NaN;
    time_gmres = NaN;
    gmres_info = struct('flag', NaN, 'relres', NaN, 'iter', [NaN, NaN], 'iter_count', NaN, 'resvec_rel', []);
    if cfg.run_gmres
        t_gmres = tic;
        [u_gmres, hist_gmres, rec_gmres, gmres_info] = waveholtz_iteration_scattering_gmres( ...
            nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, pml_wave, geom, inc, ...
            A_scatt, b_scatt, cfg.relax, cfg.gmres_tol, cfg.gmres_restart, cfg.gmres_maxit, cfg.gmres_verbose);
        time_gmres = toc(t_gmres);
        u_gmres(geom.internal) = 0;
    end

    % ---- Auto-DMD WaveHoltz solve ----
    dmd_cfg = struct();
    dmd_cfg.max_snaps = cfg.dmd_max_snaps;
    dmd_cfg.epsSVD = cfg.dmd_epsSVD;
    dmd_cfg.svd_rel_tol = cfg.dmd_svd_rel_tol;
    dmd_cfg.max_rank = cfg.dmd_max_rank;
    dmd_cfg.min_snaps_for_predict = cfg.dmd_min_snaps_for_predict;
    dmd_cfg.min_pred_interval = cfg.dmd_min_pred_interval;
    dmd_cfg.cond_ratio_tol = cfg.dmd_cond_ratio_tol;
    dmd_cfg.rho_tol = cfg.dmd_rho_tol;
    dmd_cfg.predict_burst = cfg.dmd_predict_burst;
    dmd_cfg.accept_growth = cfg.dmd_accept_growth;
    dmd_cfg.start_plain_res = cfg.dmd_start_plain_res;
    dmd_cfg.window_strategy = lower(string(cfg.dmd_window_strategy));
    dmd_cfg.verbose_iter = cfg.verbose_iter;
    if dmd_cfg.window_strategy ~= "rolling" && dmd_cfg.window_strategy ~= "restart"
        error('cfg.dmd_window_strategy must be ''rolling'' or ''restart''.');
    end

    t_dmd = tic;
    [u_dmd, hist_dmd, rec_dmd, dmd_stats] = waveholtz_iteration_scattering_dmd_restarted( ...
        nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, pml_wave, ...
        geom, inc, max_iter, tol, A_scatt, b_scatt, cfg.relax, dmd_cfg);
    time_dmd = toc(t_dmd);
    u_dmd(geom.internal) = 0;

    % ---- Physical-domain diagnostics ----
    active_phys = geom.active(I + 1, J + 1);
    Uh_phys = u_helm(I + 1, J + 1);
    Uw_phys = u_wave(I + 1, J + 1);
    Ud_phys = u_dmd(I + 1, J + 1);
    if cfg.run_gmres
        Ug_phys = u_gmres(I + 1, J + 1);
    else
        Ug_phys = [];
    end
    Uinc_phys = inc.u_inc_hat(I + 1, J + 1);

    rel_plain_vs_helm = norm(Uw_phys(active_phys) - Uh_phys(active_phys)) / max(norm(Uh_phys(active_phys)), 1e-15);
    rel_dmd_vs_helm = norm(Ud_phys(active_phys) - Uh_phys(active_phys)) / max(norm(Uh_phys(active_phys)), 1e-15);
    rel_dmd_vs_plain = norm(Ud_phys(active_phys) - Uw_phys(active_phys)) / max(norm(Uw_phys(active_phys)), 1e-15);
    if cfg.run_gmres
        rel_gmres_vs_helm = norm(Ug_phys(active_phys) - Uh_phys(active_phys)) / max(norm(Uh_phys(active_phys)), 1e-15);
        rel_dmd_vs_gmres = norm(Ud_phys(active_phys) - Ug_phys(active_phys)) / max(norm(Ug_phys(active_phys)), 1e-15);
    else
        rel_gmres_vs_helm = NaN;
        rel_dmd_vs_gmres = NaN;
    end

    Utot_helm = Uh_phys + Uinc_phys;
    Utot_wave = Uw_phys + Uinc_phys;
    Utot_dmd = Ud_phys + Uinc_phys;
    if cfg.run_gmres
        Utot_gmres = Ug_phys + Uinc_phys;
    end
    total_err_plain = norm(Utot_wave(active_phys) - Utot_helm(active_phys)) / max(norm(Utot_helm(active_phys)), 1e-15);
    total_err_dmd = norm(Utot_dmd(active_phys) - Utot_helm(active_phys)) / max(norm(Utot_helm(active_phys)), 1e-15);
    if cfg.run_gmres
        total_err_gmres = norm(Utot_gmres(active_phys) - Utot_helm(active_phys)) / max(norm(Utot_helm(active_phys)), 1e-15);
    else
        total_err_gmres = NaN;
    end

    it_plain = numel(hist_plain);
    it_dmd = numel(hist_dmd);
    if reference_mode == "gmres" && cfg.run_gmres
        reference_label = 'gmres';
        reference_iter = max(1, gmres_info.iter_count);
        reference_time = time_gmres;
        Uref_phys = Ug_phys;
    else
        reference_label = 'plain';
        reference_iter = max(1, it_plain);
        reference_time = time_plain;
        Uref_phys = Uw_phys;
    end
    speedup_iter = reference_iter / max(1, it_dmd);
    speedup_time = reference_time / max(1e-15, time_dmd);
    rel_dmd_vs_ref = norm(Ud_phys(active_phys) - Uref_phys(active_phys)) / max(norm(Uref_phys(active_phys)), 1e-15);
    speedup_time_vs_plain = time_plain / max(1e-15, time_dmd);
    if cfg.run_gmres
        speedup_time_vs_gmres = time_gmres / max(1e-15, time_dmd);
        dmd_beats_gmres = time_dmd < time_gmres;
    else
        speedup_time_vs_gmres = NaN;
        dmd_beats_gmres = false;
    end

    fprintf('\n===== Summary: scattering + Auto-DMD =====\n');
    fprintf('Reference mode: %s\n', reference_label);
    fprintf('Helmholtz runtime: %.3f s\n', time_helm);
    fprintf('Plain WaveHoltz:   it=%d, time=%.3f s, rec-res=%.3e\n', it_plain, time_plain, rec_plain);
    if cfg.run_gmres
        fprintf('GMRES WaveHoltz:   it=%d, time=%.3f s, rec-res=%.3e, flag=%d, relres=%.3e\n', ...
            gmres_info.iter_count, time_gmres, rec_gmres, gmres_info.flag, gmres_info.relres);
    end
    fprintf('DMD WaveHoltz:     it=%d, time=%.3f s, rec-res=%.3e\n', it_dmd, time_dmd, rec_dmd);
    fprintf('DMD stats: predictions=%d, window-trunc=%d, rejected=%d\n', ...
        dmd_stats.num_predictions, dmd_stats.num_window_truncations, dmd_stats.num_rejected);
    fprintf('Speedup vs %s (iters): %.3fx\n', reference_label, speedup_iter);
    fprintf('Speedup vs %s (time):  %.3fx\n', reference_label, speedup_time);
    if ~strcmp(reference_label, 'plain')
        fprintf('Speedup vs plain (time): %.3fx\n', speedup_time_vs_plain);
    end
    if cfg.run_gmres && ~strcmp(reference_label, 'gmres')
        fprintf('Speedup vs gmres (time): %.3fx\n', speedup_time_vs_gmres);
    end
    if cfg.run_gmres
        if dmd_beats_gmres
            fprintf('DMD beats GMRES (time): yes\n');
        else
            fprintf('DMD beats GMRES (time): no\n');
        end
    end
    fprintf('RelErr plain vs Helm (scatt): %.3e\n', rel_plain_vs_helm);
    if cfg.run_gmres
        fprintf('RelErr gmres vs Helm (scatt): %.3e\n', rel_gmres_vs_helm);
    end
    fprintf('RelErr DMD   vs Helm (scatt): %.3e\n', rel_dmd_vs_helm);
    fprintf('RelErr DMD   vs plain(scatt): %.3e\n', rel_dmd_vs_plain);
    if cfg.run_gmres
        fprintf('RelErr DMD   vs gmres(scatt): %.3e\n', rel_dmd_vs_gmres);
    end
    if ~(strcmp(reference_label, 'plain') || strcmp(reference_label, 'gmres'))
        fprintf('RelErr DMD   vs %s(scatt): %.3e\n', reference_label, rel_dmd_vs_ref);
    end
    fprintf('RelErr plain vs Helm (total): %.3e\n', total_err_plain);
    if cfg.run_gmres
        fprintf('RelErr gmres vs Helm (total): %.3e\n', total_err_gmres);
    end
    fprintf('RelErr DMD   vs Helm (total): %.3e\n', total_err_dmd);

    if cfg.show_plots
        helm_plot = masked_field(real(u_helm), geom.active);
        wave_plot = masked_field(real(u_wave), geom.active);
        dmd_plot = masked_field(real(u_dmd), geom.active);
        diff_plain_plot = masked_field(real(u_wave - u_helm), geom.active);
        diff_dmd_plot = masked_field(real(u_dmd - u_helm), geom.active);

        figure('Name', 'Scattering: Helmholtz vs plain vs Auto-DMD', 'Position', [120 80 1500 780]);
        subplot(2,3,1);
        plot_masked_image(x, y, helm_plot, geom.active, sprintf('Helmholtz (a/b=%.3f)', shape_ratio));
        subplot(2,3,2);
        plot_masked_image(x, y, wave_plot, geom.active, sprintf('WaveHoltz plain | it=%d', it_plain));
        subplot(2,3,3);
        plot_masked_image(x, y, dmd_plot, geom.active, sprintf('WaveHoltz + DMD | it=%d', it_dmd));
        subplot(2,3,4);
        plot_masked_image(x, y, diff_plain_plot, geom.active, sprintf('plain - helm | rel=%.2e', rel_plain_vs_helm));
        subplot(2,3,5);
        plot_masked_image(x, y, diff_dmd_plot, geom.active, sprintf('dmd - helm | rel=%.2e', rel_dmd_vs_helm));
        subplot(2,3,6);
        semilogy(hist_plain, 'LineWidth', 1.5); hold on;
        semilogy(hist_dmd, 'LineWidth', 1.5);
        if cfg.run_gmres && ~isempty(hist_gmres)
            semilogy(hist_gmres, '--', 'LineWidth', 1.3);
        end
        hold off;
        grid on;
        xlabel('iteration');
        ylabel('rel-res');
        if cfg.run_gmres && ~isempty(hist_gmres)
            legend('plain', 'dmd', 'gmres', 'Location', 'northeast');
        else
            legend('plain', 'dmd', 'Location', 'northeast');
        end
        title(sprintf('Convergence | ref=%s | speedup(it)=%.2fx, speedup(time)=%.2fx', ...
            reference_label, speedup_iter, speedup_time));

        if ~isempty(dmd_stats.singular_vals)
            figure('Name', 'Auto-DMD singular values (latest trigger window)', 'Position', [180 140 760 360]);
            semilogy(dmd_stats.singular_vals, '-o', 'LineWidth', 1.2, 'MarkerSize', 4);
            grid on;
            xlabel('index');
            ylabel('singular value');
            title('Singular values of DMD update window');
        end
    end

    results = struct();
    results.shape_ratio = shape_ratio;
    results.cfg = cfg;
    results.hist_plain = hist_plain;
    results.hist_gmres = hist_gmres;
    results.hist_dmd = hist_dmd;
    results.u_helm = u_helm;
    results.u_plain = u_wave;
    results.u_gmres = u_gmres;
    results.u_dmd = u_dmd;
    results.rec_plain = rec_plain;
    results.rec_gmres = rec_gmres;
    results.rec_dmd = rec_dmd;
    results.reference_mode = char(reference_label);
    results.reference_iter = reference_iter;
    results.reference_time = reference_time;
    results.speedup_iter = speedup_iter;
    results.speedup_time = speedup_time;
    results.speedup_time_vs_plain = speedup_time_vs_plain;
    results.speedup_time_vs_gmres = speedup_time_vs_gmres;
    results.dmd_beats_gmres_time = dmd_beats_gmres;
    results.rel_plain_vs_helm = rel_plain_vs_helm;
    results.rel_gmres_vs_helm = rel_gmres_vs_helm;
    results.rel_dmd_vs_helm = rel_dmd_vs_helm;
    results.rel_dmd_vs_plain = rel_dmd_vs_plain;
    results.rel_dmd_vs_gmres = rel_dmd_vs_gmres;
    results.rel_dmd_vs_reference = rel_dmd_vs_ref;
    results.total_err_plain = total_err_plain;
    results.total_err_gmres = total_err_gmres;
    results.total_err_dmd = total_err_dmd;
    results.time_helm = time_helm;
    results.time_plain = time_plain;
    results.time_gmres = time_gmres;
    results.time_dmd = time_dmd;
    results.gmres_info = gmres_info;
    results.dmd_stats = dmd_stats;
end


function out = masked_field(field, active_mask)
    out = field;
    out(~active_mask) = NaN;
end


function plot_masked_image(x, y, field, active_mask, ttl)
    h = imagesc(x, y, field.');
    set(h, 'AlphaData', active_mask.');
    axis equal tight;
    set(gca, 'YDir', 'normal', 'Color', [0.85, 0.85, 0.85]);
    title(ttl);
    colorbar;
end


function inc = build_incident_wave(X, omega, c)
    k = omega / c;
    u_inc_hat = exp(1i * k * X);
    inc = struct('k', k, 'u_inc_hat', u_inc_hat);
end


function geom = build_scatterer_geometry(nx, ny, x, y, center, semi_a, semi_b, eta_boundary_min)
    [X, Y] = ndgrid(x, y);
    interior = false(nx + 2, ny + 2);
    interior(2:end-1, 2:end-1) = true;

    ellipse_level = ((X - center(1)).^2) / (semi_a^2) + ((Y - center(2)).^2) / (semi_b^2);
    internal = interior & (ellipse_level <= 1.0);
    active = interior & ~internal;

    geom = struct();
    geom.center = center;
    geom.semi_a = semi_a;
    geom.semi_b = semi_b;
    geom.X = X;
    geom.Y = Y;
    geom.active = active;
    geom.internal = internal;
    geom.ghost_x = false(nx + 2, ny + 2);
    geom.ghost_y = false(nx + 2, ny + 2);
    geom.boundary_count_x = 0;
    geom.boundary_count_y = 0;

    zero_field = zeros(nx + 2, ny + 2);
    geom.gx_c2 = zero_field;
    geom.gx_c1 = zero_field;
    geom.gx_cg = zero_field;
    geom.gx_owner = zeros(nx + 2, ny + 2);
    geom.gx_u2 = zeros(nx + 2, ny + 2);
    geom.gx_gamma_x = zero_field;
    geom.gx_gamma_y = zero_field;

    geom.gy_c2 = zero_field;
    geom.gy_c1 = zero_field;
    geom.gy_cg = zero_field;
    geom.gy_owner = zeros(nx + 2, ny + 2);
    geom.gy_u2 = zeros(nx + 2, ny + 2);
    geom.gy_gamma_x = zero_field;
    geom.gy_gamma_y = zero_field;

    dims = [nx + 2, ny + 2];

    for i = 3:(nx)
        for j = 2:(ny + 1)
            if ~internal(i, j)
                continue;
            end

            left_active = active(i - 1, j);
            right_active = active(i + 1, j);
            if left_active && right_active
                error('Ambiguous x-ghost at (%d,%d): obstacle too small for single-valued x ghost.', i, j);
            elseif left_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0
                    error('Invalid x-ghost geometry at (%d,%d).', i, j);
                end
                x_gamma = center(1) - semi_a * sqrt(root);
                eta = abs(x_gamma - x(i - 1)) / (x(i) - x(i - 1));
                if ~(eta > 0 && eta < 1)
                    error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_x = geom.boundary_count_x + 1; end
                geom.ghost_x(i, j) = true;
                geom.gx_c2(i, j) = coeffs(1);
                geom.gx_c1(i, j) = coeffs(2);
                geom.gx_cg(i, j) = coeffs(3);
                geom.gx_owner(i, j) = sub2ind(dims, i - 1, j);
                geom.gx_u2(i, j) = sub2ind(dims, i - 2, j);
                geom.gx_gamma_x(i, j) = x_gamma;
                geom.gx_gamma_y(i, j) = y(j);
            elseif right_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0
                    error('Invalid x-ghost geometry at (%d,%d).', i, j);
                end
                x_gamma = center(1) + semi_a * sqrt(root);
                eta = abs(x(i + 1) - x_gamma) / (x(i + 1) - x(i));
                if ~(eta > 0 && eta < 1)
                    error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_x = geom.boundary_count_x + 1; end
                geom.ghost_x(i, j) = true;
                geom.gx_c2(i, j) = coeffs(1);
                geom.gx_c1(i, j) = coeffs(2);
                geom.gx_cg(i, j) = coeffs(3);
                geom.gx_owner(i, j) = sub2ind(dims, i + 1, j);
                geom.gx_u2(i, j) = sub2ind(dims, i + 2, j);
                geom.gx_gamma_x(i, j) = x_gamma;
                geom.gx_gamma_y(i, j) = y(j);
            end
        end
    end

    for i = 2:(nx + 1)
        for j = 3:(ny)
            if ~internal(i, j)
                continue;
            end

            down_active = active(i, j - 1);
            up_active = active(i, j + 1);
            if down_active && up_active
                error('Ambiguous y-ghost at (%d,%d): obstacle too small for single-valued y ghost.', i, j);
            elseif down_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0
                    error('Invalid y-ghost geometry at (%d,%d).', i, j);
                end
                y_gamma = center(2) - semi_b * sqrt(root);
                eta = abs(y_gamma - y(j - 1)) / (y(j) - y(j - 1));
                if ~(eta > 0 && eta < 1)
                    error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_y = geom.boundary_count_y + 1; end
                geom.ghost_y(i, j) = true;
                geom.gy_c2(i, j) = coeffs(1);
                geom.gy_c1(i, j) = coeffs(2);
                geom.gy_cg(i, j) = coeffs(3);
                geom.gy_owner(i, j) = sub2ind(dims, i, j - 1);
                geom.gy_u2(i, j) = sub2ind(dims, i, j - 2);
                geom.gy_gamma_x(i, j) = x(i);
                geom.gy_gamma_y(i, j) = y_gamma;
            elseif up_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0
                    error('Invalid y-ghost geometry at (%d,%d).', i, j);
                end
                y_gamma = center(2) + semi_b * sqrt(root);
                eta = abs(y(j + 1) - y_gamma) / (y(j + 1) - y(j));
                if ~(eta > 0 && eta < 1)
                    error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_y = geom.boundary_count_y + 1; end
                geom.ghost_y(i, j) = true;
                geom.gy_c2(i, j) = coeffs(1);
                geom.gy_c1(i, j) = coeffs(2);
                geom.gy_cg(i, j) = coeffs(3);
                geom.gy_owner(i, j) = sub2ind(dims, i, j + 1);
                geom.gy_u2(i, j) = sub2ind(dims, i, j + 2);
                geom.gy_gamma_x(i, j) = x(i);
                geom.gy_gamma_y(i, j) = y_gamma;
            end
        end
    end

    active_left = false(nx + 2, ny + 2);
    active_right = false(nx + 2, ny + 2);
    active_down = false(nx + 2, ny + 2);
    active_up = false(nx + 2, ny + 2);

    active_left(2:end, :) = active(1:end-1, :);
    active_right(1:end-1, :) = active(2:end, :);
    active_down(:, 2:end) = active(:, 1:end-1);
    active_up(:, 1:end-1) = active(:, 2:end);

    x_adjacent_internal = internal & (active_left | active_right);
    y_adjacent_internal = internal & (active_down | active_up);

    if any(x_adjacent_internal(:) & ~geom.ghost_x(:))
        error('Some x-adjacent internal nodes were not assigned ghost data.');
    end

    if any(y_adjacent_internal(:) & ~geom.ghost_y(:))
        error('Some y-adjacent internal nodes were not assigned ghost data.');
    end
end


function [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min)
    if eta <= eta_boundary_min
        coeffs = [0, 0, 1];
        boundary_mode = true;
        return;
    end
    c2 = (1 - eta) / (1 + eta);
    c1 = 2 * (1 - eta) / eta;
    cg = 2 / (eta * (1 + eta));
    coeffs = [c2, c1, cg];
    boundary_mode = false;
end


function pml = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml)
    sigma_x = zeros(nx + 2, 1);
    sigma_y = zeros(ny + 2, 1);

    for i = 1:npml
        if ispml == 1
            val = sigma_max * ((npml - (i - 1)) / npml)^2;
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


function [A_scatt, rhs] = helmholtz_scattering_system(nx, ny, hx, hy, omega, pml, geom, inc)
    Ntot = nx * ny;
    A_scatt = spalloc(Ntot, Ntot, 7 * Ntot);
    rhs = zeros(Ntot, 1);
    idx = @(ii, jj) ii + (jj - 1) * nx;

    dims_full = [nx + 2, ny + 2];

    for ii = 1:nx
        i = ii + 1;
        for jj = 1:ny
            j = jj + 1;
            row = idx(ii, jj);

            if geom.internal(i, j)
                A_scatt(row, row) = 1;
                continue;
            end

            a_x = 1.0 ./ pml.s_x(i);
            a_p = 1.0 ./ pml.s_x_half(i);
            a_m = 1.0 ./ pml.s_x_half(i - 1);
            b_y = 1.0 ./ pml.s_y(j);
            b_p = 1.0 ./ pml.s_y_half(j);
            b_m = 1.0 ./ pml.s_y_half(j - 1);

            A_scatt(row, row) = A_scatt(row, row) ...
                - a_x * (a_p + a_m) / (hx * hx) ...
                - b_y * (b_p + b_m) / (hy * hy) ...
                + omega^2;

            % +x neighbor
            coef = a_x * a_p / (hx * hx);
            if geom.active(i + 1, j)
                A_scatt(row, idx(ii + 1, jj)) = A_scatt(row, idx(ii + 1, jj)) + coef;
            elseif geom.internal(i + 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i + 1, j, idx, dims_full, geom, inc);
            end

            % -x neighbor
            coef = a_x * a_m / (hx * hx);
            if geom.active(i - 1, j)
                A_scatt(row, idx(ii - 1, jj)) = A_scatt(row, idx(ii - 1, jj)) + coef;
            elseif geom.internal(i - 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i - 1, j, idx, dims_full, geom, inc);
            end

            % +y neighbor
            coef = b_y * b_p / (hy * hy);
            if geom.active(i, j + 1)
                A_scatt(row, idx(ii, jj + 1)) = A_scatt(row, idx(ii, jj + 1)) + coef;
            elseif geom.internal(i, j + 1)
                [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, i, j + 1, idx, dims_full, geom, inc);
            end

            % -y neighbor
            coef = b_y * b_m / (hy * hy);
            if geom.active(i, j - 1)
                A_scatt(row, idx(ii, jj - 1)) = A_scatt(row, idx(ii, jj - 1)) + coef;
            elseif geom.internal(i, j - 1)
                [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, i, j - 1, idx, dims_full, geom, inc);
            end
        end
    end
end


function [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, gi, gj, idx, dims_full, geom, inc)
    owner_lin = geom.gx_owner(gi, gj);
    if owner_lin == 0
        error('Missing x-ghost coefficients at (%d,%d).', gi, gj);
    end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    [u2i, u2j] = ind2sub(dims_full, geom.gx_u2(gi, gj));
    if ~geom.active(oi, oj)
        error('x-ghost owner at (%d,%d) is not active.', oi, oj);
    end

    c2 = geom.gx_c2(gi, gj);
    c1 = geom.gx_c1(gi, gj);
    cg = geom.gx_cg(gi, gj);
    u_gamma_hat = -exp(1i * inc.k * geom.gx_gamma_x(gi, gj));

    A_scatt(row, idx(oi - 1, oj - 1)) = A_scatt(row, idx(oi - 1, oj - 1)) - coef * c1;
    A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * c2;
    rhs(row) = rhs(row) - coef * cg * u_gamma_hat;
end


function [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, gi, gj, idx, dims_full, geom, inc)
    owner_lin = geom.gy_owner(gi, gj);
    if owner_lin == 0
        error('Missing y-ghost coefficients at (%d,%d).', gi, gj);
    end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    [u2i, u2j] = ind2sub(dims_full, geom.gy_u2(gi, gj));
    if ~geom.active(oi, oj)
        error('y-ghost owner at (%d,%d) is not active.', oi, oj);
    end

    c2 = geom.gy_c2(gi, gj);
    c1 = geom.gy_c1(gi, gj);
    cg = geom.gy_cg(gi, gj);
    u_gamma_hat = -exp(1i * inc.k * geom.gy_gamma_x(gi, gj));

    A_scatt(row, idx(oi - 1, oj - 1)) = A_scatt(row, idx(oi - 1, oj - 1)) - coef * c1;
    A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * c2;
    rhs(row) = rhs(row) - coef * cg * u_gamma_hat;
end


function W = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x0, geom, inc)
    sx_full = zeros(nx + 2, ny + 2);
    sy_full = zeros(nx + 2, ny + 2);
    sx = pml.sigma_x;
    sy = pml.sigma_y;
    for i = 1:(nx + 2)
        sx_full(i, :) = sx(i);
    end
    for j = 1:(ny + 2)
        sy_full(:, j) = sy(j);
    end

    sigma_x_vx = 0.5 * (sx_full(1:end-1, :) + sx_full(2:end, :));
    sigma_y_vy = 0.5 * (sy_full(:, 1:end-1) + sy_full(:, 2:end));

    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    naux = 3 * nu;
    nstate = nu + nv + naux;
    if numel(x0) ~= nstate
        error('x0 length mismatch: expect %d, got %d', nstate, numel(x0));
    end

    u0_int = x0(1:nu);
    v0_int = x0(nu + 1:nu + nv);
    psi1_int = x0(nu + nv + 1 : nu + nv + nu);
    psi2_int = x0(nu + nv + nu + 1 : nu + nv + 2*nu);
    theta_int = x0(nu + nv + 2*nu + 1 : end);

    u = zeros(nx + 2, ny + 2);
    u(2:end-1, 2:end-1) = reshape(u0_int, nx, ny);
    u(geom.internal) = 0;

    vx = zeros(nx + 1, ny + 2);
    vy = zeros(nx + 2, ny + 1);
    vx_int = reshape(v0_int(1:nvx), nx + 1, ny);
    vy_int = reshape(v0_int(nvx + 1:end), nx, ny + 1);
    vx(:, 2:end-1) = vx_int;
    vy(2:end-1, :) = vy_int;

    psi1 = zeros(nx + 2, ny + 2);
    psi2 = zeros(nx + 2, ny + 2);
    theta = zeros(nx + 2, ny + 2);
    psi1(2:end-1, 2:end-1) = reshape(psi1_int, nx, ny);
    psi2(2:end-1, 2:end-1) = reshape(psi2_int, nx, ny);
    theta(2:end-1, 2:end-1) = reshape(theta_int, nx, ny);
    psi1(geom.internal) = 0;
    psi2(geom.internal) = 0;
    theta(geom.internal) = 0;

    c_w = cos(omega_ts * dt / 2);
    sigu = sx_full + sy_full;

    v0_vec = [reshape(vx(:, 2:end-1), nvx, 1); reshape(vy(2:end-1, :), nvy, 1)];
    W = zeros(nt + 1, nstate);
    u0_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
    W(1, :) = [u0_vec; v0_vec; psi1_int; psi2_int; theta_int].';

    for n = 1:nt
        t = (n - 1) * dt;

        [u_x, u_y] = fill_directional_ghosts(u, geom, inc, omega_ts, t);

        du_dx = (u_x(2:end, :) - u_x(1:end-1, :)) / hx;
        du_dy = (u_y(:, 2:end) - u_y(:, 1:end-1)) / hy;
        vx_new = ((1 - 0.5 * dt * sigma_x_vx) .* vx + dt * c^2 * du_dx) ./ (1 + 0.5 * dt * sigma_x_vx);
        vy_new = ((1 - 0.5 * dt * sigma_y_vy) .* vy + dt * c^2 * du_dy) ./ (1 + 0.5 * dt * sigma_y_vy);

        vx_avg = 0.5 * (vx + vx_new);
        vy_avg = 0.5 * (vy + vy_new);

        grad_vx = (vx_avg(2:end, :) - vx_avg(1:end-1, :)) / hx;
        tmp1 = zeros(nx + 2, ny + 2);
        tmp1(2:end-1, :) = grad_vx;

        grad_vy = (vy_avg(:, 2:end) - vy_avg(:, 1:end-1)) / hy;
        tmp2 = zeros(nx + 2, ny + 2);
        tmp2(:, 2:end-1) = grad_vy;

        psi1 = psi1 + dt * (sy_full .* tmp1);
        psi2 = psi2 + dt * (sx_full .* tmp2);
        theta = theta + dt * ((sx_full .* sy_full) .* (c_w^2) .* u);

        div_v = zeros(nx + 2, ny + 2);
        div_v(2:end-1, :) = div_v(2:end-1, :) + (vx_new(2:end, :) - vx_new(1:end-1, :)) / hx;
        div_v(:, 2:end-1) = div_v(:, 2:end-1) + (vy_new(:, 2:end) - vy_new(:, 1:end-1)) / hy;

        rhs_u = ...
              div_v(2:end-1, 2:end-1) ...
            - theta(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) ...
            + psi2(2:end-1, 2:end-1);

        u_int = u(2:end-1, 2:end-1);
        num_u = (1 - 0.5 * dt * sigu(2:end-1, 2:end-1)) .* u_int + dt * rhs_u;
        den_u = (1 + 0.5 * dt * sigu(2:end-1, 2:end-1));
        u_int_new = num_u ./ den_u;
        active_int = geom.active(2:end-1, 2:end-1);
        u_int(~active_int) = 0;
        u_int(active_int) = u_int_new(active_int);
        u(2:end-1, 2:end-1) = u_int;
        u(geom.internal) = 0;

        psi1(geom.internal) = 0;
        psi2(geom.internal) = 0;
        theta(geom.internal) = 0;

        vx = vx_new;
        vy = vy_new;

        v_vec = [reshape(vx(:, 2:end-1), nvx, 1); reshape(vy(2:end-1, :), nvy, 1)];
        u_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
        psi1_vec = reshape(psi1(2:end-1, 2:end-1), nu, 1);
        psi2_vec = reshape(psi2(2:end-1, 2:end-1), nu, 1);
        theta_vec = reshape(theta(2:end-1, 2:end-1), nu, 1);
        W(n + 1, :) = [u_vec; v_vec; psi1_vec; psi2_vec; theta_vec].';
    end
end


function [u_x, u_y] = fill_directional_ghosts(u, geom, inc, omega_ts, t)
    u_x = u;
    u_y = u;

    if any(geom.ghost_x(:))
        mask = geom.ghost_x;
        owner_vals = u(geom.gx_owner(mask));
        u2_vals = u(geom.gx_u2(mask));
        u_gamma_hat = -exp(1i * inc.k * geom.gx_gamma_x(mask));
        u_gamma_t = real(u_gamma_hat .* exp(-1i * omega_ts * t));
        ghost_vals = geom.gx_c2(mask) .* u2_vals - geom.gx_c1(mask) .* owner_vals + geom.gx_cg(mask) .* u_gamma_t;
        u_x(mask) = ghost_vals;
    end

    if any(geom.ghost_y(:))
        mask = geom.ghost_y;
        owner_vals = u(geom.gy_owner(mask));
        u2_vals = u(geom.gy_u2(mask));
        u_gamma_hat = -exp(1i * inc.k * geom.gy_gamma_x(mask));
        u_gamma_t = real(u_gamma_hat .* exp(-1i * omega_ts * t));
        ghost_vals = geom.gy_c2(mask) .* u2_vals - geom.gy_c1(mask) .* owner_vals + geom.gy_cg(mask) .* u_gamma_t;
        u_y(mask) = ghost_vals;
    end
end


function [u_full, hist, rec_res] = waveholtz_iteration_scattering_plain(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, geom, inc, max_iter, tol, A, b, relax, verbose_iter)
    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;

    x = zeros(nu + nv + 3 * nu, 1);
    hist = zeros(max_iter, 1);
    t_grid = (0:nt).' * dt;
    weights = cos(omega_ts * t_grid) - 0.25;

    for k = 1:max_iter
        W = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x, geom, inc);
        if any(~isfinite(W(:)))
            error('Non-finite values encountered during one-period time stepping at iteration %d.', k);
        end

        integrand = bsxfun(@times, weights, W);
        integral = trapz(t_grid, integrand, 1);
        x_raw = (2.0 / T_ts) * integral.';
        x_new = (1 - relax) * x + relax * x_raw;
        if any(~isfinite(x_new))
            error('Non-finite fixed-point state at iteration %d.', k);
        end

        res = norm(x_new - x) / max(1e-16, norm(x_new));
        hist(k) = res;
        x = x_new;

        if verbose_iter && (k <= 5 || mod(k, 20) == 0 || res < tol)
            fprintf('plain iter %3d: rel-res = %.3e\n', k, res);
        end
        if res < tol
            hist = hist(1:k);
            break;
        end
    end

    u_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x, geom, inc);
    u_full(geom.internal) = 0;

    rec_res = norm(A * reshape(u_full(2:end-1, 2:end-1), nu, 1) - b) / max(norm(b), 1e-15);
    if verbose_iter
        fprintf('Plain recovered phasor residual: %.3e\n', rec_res);
    end
end


function [u_full, hist_rel, rec_res, info] = waveholtz_iteration_scattering_gmres( ...
    nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, geom, inc, ...
    A, b, relax, gmres_tol, gmres_restart, gmres_maxit, gmres_verbose)

    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    nstate = nu + nv + 3 * nu;

    x_zero = zeros(nstate, 1);
    F0 = apply_waveholtz_map_scattering(nx, ny, hx, hy, c, omega_ts, T_ts, dt, nt, pml, x_zero, geom, inc, relax);
    if any(~isfinite(F0))
        error('Non-finite F0 when constructing GMRES WaveHoltz operator.');
    end

    % Affine fixed-point map F(x) = Sx + F0, solve (I-S)x = F0.
    Afun = @(x) x - (apply_waveholtz_map_scattering(nx, ny, hx, hy, c, omega_ts, T_ts, dt, nt, pml, x, geom, inc, relax) - F0);
    [x_sol, flag, relres, iter, resvec] = gmres(Afun, F0, gmres_restart, gmres_tol, gmres_maxit, [], [], x_zero);

    denom = max(norm(F0), 1e-16);
    hist_rel = resvec / denom;
    if isempty(iter)
        iter_count = 0;
        iter = [0, 0];
    else
        iter_count = iter(1) * gmres_restart + iter(2);
    end

    u_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x_sol, geom, inc);
    u_full(geom.internal) = 0;
    rec_res = norm(A * reshape(u_full(2:end-1, 2:end-1), nu, 1) - b) / max(norm(b), 1e-15);

    info = struct();
    info.flag = flag;
    info.relres = relres;
    info.iter = iter;
    info.iter_count = iter_count;
    info.resvec_rel = hist_rel;

    if gmres_verbose
        fprintf('GMRES solve: flag=%d, relres=%.3e, iter=[%d %d], count=%d\n', ...
            flag, relres, iter(1), iter(2), iter_count);
        fprintf('GMRES recovered phasor residual: %.3e\n', rec_res);
    end
end


function x_new = apply_waveholtz_map_scattering(nx, ny, hx, hy, c, omega_ts, T_ts, dt, nt, pml, x, geom, inc, relax)
    W = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x, geom, inc);
    if any(~isfinite(W(:)))
        error('Non-finite values in one-period stepping while applying WaveHoltz map.');
    end

    t_grid = (0:nt).' * dt;
    weights = cos(omega_ts * t_grid) - 0.25;
    integrand = bsxfun(@times, weights, W);
    integral = trapz(t_grid, integrand, 1);
    x_raw = (2.0 / T_ts) * integral.';
    x_new = (1 - relax) * x + relax * x_raw;

    if any(~isfinite(x_new))
        error('Non-finite fixed-point state while applying WaveHoltz map.');
    end
end


function [u_full, hist, rec_res, stats] = waveholtz_iteration_scattering_dmd_restarted( ...
    nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, geom, inc, max_iter, tol, A, b, relax, dmd_cfg)

    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    nstate = nu + nv + 3 * nu;

    x = zeros(nstate, 1);
    hist = zeros(max_iter, 1);
    t_grid = (0:nt).' * dt;
    weights = cos(omega_ts * t_grid) - 0.25;

    U = [];
    S = [];
    V = [];
    DeltaSnaps = zeros(nstate, dmd_cfg.max_snaps);
    snap_count = 0;

    num_window_truncations = 0;
    num_predictions = 0;
    num_rejected = 0;
    singular_vals = [];
    last_pred_iter = -1e12;
    burst_remaining = 0;

    for k = 1:max_iter
        W = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x, geom, inc);
        if any(~isfinite(W(:)))
            error('Non-finite values encountered during one-period time stepping at iteration %d.', k);
        end

        integrand = bsxfun(@times, weights, W);
        integral = trapz(t_grid, integrand, 1);
        x_raw = (2.0 / T_ts) * integral.';
        x_new = (1 - relax) * x + relax * x_raw;
        if any(~isfinite(x_new))
            error('Non-finite fixed-point state at iteration %d.', k);
        end

        delta = x_new - x;
        x_plain = x_new;
        plain_res = norm(delta) / max(1e-16, norm(x_plain));
        iter_res = plain_res;

        window_truncated = false;
        if snap_count < dmd_cfg.max_snaps
            snap_count = snap_count + 1;
            DeltaSnaps(:, snap_count) = delta;
        else
            if dmd_cfg.window_strategy == "rolling"
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
            num_window_truncations = num_window_truncations + 1;
        end

        if snap_count >= 2 && plain_res <= dmd_cfg.start_plain_res
            if isempty(U)
                [U, S, V] = snapshotSVD(DeltaSnaps(:, 1:snap_count-1), dmd_cfg.epsSVD, dmd_cfg.max_rank);
            elseif window_truncated
                [U, S, V] = snapshotSVD(DeltaSnaps(:, 1:snap_count-1), dmd_cfg.epsSVD, dmd_cfg.max_rank);
            elseif snap_count == 2
                [U, S, V] = initSVD(DeltaSnaps(:, 1), dmd_cfg.epsSVD);
                [U, S, V] = truncateSVD(U, S, V, dmd_cfg.epsSVD, dmd_cfg.max_rank);
            else
                [U, S, V] = incrementalSVD(DeltaSnaps(:, snap_count - 1), dmd_cfg.epsSVD, U, S, V);
                [U, S, V] = truncateSVD(U, S, V, dmd_cfg.epsSVD, dmd_cfg.max_rank);
            end

            if ~isempty(U)
                sigmas_full = diag(S);
                if ~isempty(sigmas_full) && sigmas_full(1) > 0
                    rel_sig = sigmas_full / sigmas_full(1);
                    r_eff = nnz(rel_sig >= dmd_cfg.svd_rel_tol);
                    r_eff = min([r_eff, dmd_cfg.max_rank, numel(sigmas_full), snap_count - 1]);
                else
                    r_eff = 0;
                end

                if r_eff >= 2
                    U_eff = U(:, 1:r_eff);
                    S_eff = S(1:r_eff, 1:r_eff);
                    V_eff = V(:, 1:r_eff);

                    if snap_count >= dmd_cfg.min_snaps_for_predict && (k - last_pred_iter) >= dmd_cfg.min_pred_interval
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
                        cond_ratio = min_sig / max(1e-16, max_sig);

                        if dmd_cfg.verbose_iter && (k <= 10 || mod(k, 20) == 0)
                            fprintf('dmd-meta iter %3d: m=%d, r=%d, rho=%.5f, cond_ratio=%.3e\n', ...
                                k, snap_count, r_eff, rho, cond_ratio);
                        end

                        trigger_prediction = cond_ratio < dmd_cfg.cond_ratio_tol && rho < dmd_cfg.rho_tol;
                        if trigger_prediction
                            burst_remaining = max(burst_remaining, dmd_cfg.predict_burst);
                        end

                        can_predict = burst_remaining > 0 && cond_ratio < dmd_cfg.cond_ratio_tol && rho < dmd_cfg.rho_tol;
                        if can_predict
                            rz = size(A_tilde, 1);
                            delta_z = (eye(rz) - A_tilde) \ ((U_eff') * delta);
                            x_pred = x + U_eff * delta_z;

                            if all(isfinite(x_pred))
                                W_check = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x_pred, geom, inc);
                                integrand_check = bsxfun(@times, weights, W_check);
                                integral_check = trapz(t_grid, integrand_check, 1);
                                x_check_raw = (2.0 / T_ts) * integral_check.';
                                x_check = (1 - relax) * x_pred + relax * x_check_raw;
                                pred_res = norm(x_check - x_pred) / max(1e-16, norm(x_check));

                                if pred_res <= dmd_cfg.accept_growth * plain_res
                                    x_new = x_pred;
                                    iter_res = pred_res;
                                    num_predictions = num_predictions + 1;
                                    last_pred_iter = k;
                                    burst_remaining = max(0, burst_remaining - 1);
                                    singular_vals = sigmas_eff;
                                    if dmd_cfg.verbose_iter
                                        fprintf('dmd   iter %3d: prediction accepted, pred-res=%.3e, plain-res=%.3e, burst remaining %d\n', ...
                                            k, pred_res, plain_res, burst_remaining);
                                    end
                                else
                                    num_rejected = num_rejected + 1;
                                    burst_remaining = 0;
                                    x_new = x_plain;
                                    if dmd_cfg.verbose_iter
                                        fprintf('dmd   iter %3d: prediction rejected, pred-res=%.3e, plain-res=%.3e\n', ...
                                            k, pred_res, plain_res);
                                    end
                                end
                            else
                                num_rejected = num_rejected + 1;
                                burst_remaining = 0;
                            end
                        end
                    end
                end
            end
        end

        res = iter_res;
        hist(k) = res;
        x = x_new;

        if dmd_cfg.verbose_iter && (k <= 5 || mod(k, 20) == 0 || res < tol)
            fprintf('dmd   iter %3d: rel-res = %.3e\n', k, res);
        end
        if res < tol
            hist = hist(1:k);
            break;
        end
    end

    u_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x, geom, inc);
    u_full(geom.internal) = 0;
    rec_res = norm(A * reshape(u_full(2:end-1, 2:end-1), nu, 1) - b) / max(norm(b), 1e-15);
    if dmd_cfg.verbose_iter
        fprintf('DMD recovered phasor residual: %.3e\n', rec_res);
    end

    stats = struct();
    stats.num_restarts = num_window_truncations;
    stats.num_window_truncations = num_window_truncations;
    stats.num_predictions = num_predictions;
    stats.num_rejected = num_rejected;
    stats.singular_vals = singular_vals;
end


function uhat_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, ~, T_ts, dt, nt, pml, x_state, geom, inc)
    Wconv = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x_state, geom, inc);
    nu = nx * ny;
    t_grid = (0:nt).' * dt;
    q_cos = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, cos(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';
    q_sin = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, sin(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';
    uhat = q_cos + 1i * q_sin;

    uhat_full = zeros(nx + 2, ny + 2);
    uhat_full(2:end-1, 2:end-1) = reshape(uhat, nx, ny);
    uhat_full(geom.internal) = 0;
end


function [W, sigma, V] = initSVD(u, epsSVD)
    if norm(u) > epsSVD
        sigma = norm(u);
        W = u / sigma;
        V = 1;
        sigma = diag(sigma);
    else
        sigma = [];
        W = [];
        V = [];
    end
end


function [U, S, V] = snapshotSVD(Y, epsSVD, max_rank)
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
    [U, S, V] = truncateSVD(U, S, V, epsSVD, max_rank);
end


function [U, S, V] = truncateSVD(U, S, V, epsSVD, max_rank)
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


function [U_r, S_r, V_r] = incrementalSVD(u, epsSVD, U, S, V)
    if isempty(U)
        [U_r, S_r, V_r] = initSVD(u, epsSVD);
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


function out = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
    out = s;
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
