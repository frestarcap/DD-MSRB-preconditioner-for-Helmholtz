function results = run_waveholtz_3d_forward_timing(candidate, user_cfg)
%RUN_WAVEHOLTZ_3D_FORWARD_TIMING Time a matrix-free 3D WaveHoltz period.
%
% This is a feasibility probe for the 40% DD benchmark configurations. It
% uses a staggered split-field acoustic PML and streams the WaveHoltz filter
% without storing the time history. The publication 3D results continue to
% use shifted-Laplacian completion until the split-field discretization has
% completed a dedicated accuracy study.

    if nargin < 1 || isempty(candidate)
        candidate = 'lens';
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end
    cfg = merge_structs(default_cfg(), user_cfg);
    spec = helmholtz_3d_40pct_config(candidate);
    problem_cfg = merge_structs(spec.problem_cfg, cfg.problem_cfg);
    problem = helmholtz_3d_problem_interface(problem_cfg);
    mu = cfg.mu;
    if isempty(mu)
        mu = problem_cfg.test_params(1);
    end

    [A, rhs] = problem.assemble(mu);
    rhs = rhs / max(norm(rhs), eps);
    pre = prepare_waveholtz_period(problem, mu, rhs, cfg);
    x0 = zeros(pre.layout.nstate, 1);

    fprintf('\n=== 3D WaveHoltz forward timing: %s ===\n', spec.name);
    fprintf('grid=%s, N=%d, state=%d, omega=%.3f, mu=%.6f\n', ...
        mat2str(problem.n), problem.N, pre.layout.nstate, ...
        problem_cfg.omega, mu);
    fprintf('dt=%.4e, steps/period=%d, state storage=%.2f MiB\n', ...
        pre.dt, pre.nt, pre.layout.nstate * 8 / 2^20);

    [b_wave, first_forced_time] = apply_waveholtz_period(pre, x0, true, false);
    forced_times = zeros(cfg.timing_repeats, 1);
    for k = 1:cfg.timing_repeats
        [b_wave, forced_times(k)] = apply_waveholtz_period(pre, x0, true, false);
    end
    forced_period_time = median(forced_times);

    Afun = @(x) x - apply_waveholtz_period(pre, x, false, false);
    [probe_residual, probe_times] = gmres_probe( ...
        Afun, b_wave, cfg.gmres_probe_steps, cfg.gmres_reorthogonalize);
    homogeneous_period_time = median(probe_times);
    estimated_applies = estimate_gmres_applies(probe_residual, cfg.waveholtz_tol);

    shifted = benchmark_shifted_laplacian(problem, A, rhs, mu, cfg);
    scenarios = build_cost_scenarios(spec, homogeneous_period_time, ...
        shifted.total_time, cfg.assumed_waveholtz_applies);

    full = struct('ran', false, 'flag', NaN, 'iterations', NaN, ...
        'state_relres', NaN, 'helmholtz_relres', NaN, ...
        'scaled_helmholtz_relres', NaN, 'response_scale', NaN, ...
        'physical_helmholtz_relres', NaN, 'pml_helmholtz_relres', NaN, ...
        'rhs_period_s', NaN, 'solve_time_s', NaN, 'recovery_period_s', NaN, ...
        'total_time_s', NaN, 'speedup_vs_shifted', NaN);
    if cfg.run_full_waveholtz_solve
        solve_timer = tic;
        [x_wave, flag, relres, iter] = gmres(Afun, b_wave, ...
            cfg.full_gmres_restart, cfg.waveholtz_tol, cfg.full_gmres_maxit);
        full.solve_time_s = toc(solve_timer);
        [~, full.recovery_period_s, phasor] = apply_waveholtz_period( ...
            pre, x_wave, true, true);
        full.rhs_period_s = first_forced_time;
        full.total_time_s = full.rhs_period_s + full.solve_time_s + ...
            full.recovery_period_s;
        full.speedup_vs_shifted = shifted.total_time / full.total_time_s;
        full.ran = true;
        full.flag = flag;
        full.iterations = iteration_count(iter, cfg.full_gmres_restart);
        full.state_relres = relres;
        helmholtz_action = A * phasor;
        full.helmholtz_relres = norm(helmholtz_action - rhs) / max(norm(rhs), eps);
        full.response_scale = (rhs' * helmholtz_action) / max(rhs' * rhs, eps);
        full.scaled_helmholtz_relres = norm( ...
            helmholtz_action / full.response_scale - rhs) / max(norm(rhs), eps);
        partition = problem.partition(mu);
        physical = partition.labels(:) ~= 3;
        pml = ~physical;
        residual = helmholtz_action - rhs;
        full.physical_helmholtz_relres = norm(residual(physical)) / ...
            max(norm(rhs), eps);
        full.pml_helmholtz_relres = norm(residual(pml)) / max(norm(rhs), eps);
        fprintf(['full WaveHoltz: %d GMRES iterations, state residual %.3e, ' ...
            'Helmholtz residual %.3e (scaled %.3e), %.2fs, %.2fx speedup\n'], ...
            full.iterations, full.state_relres, full.helmholtz_relres, ...
            full.scaled_helmholtz_relres, full.total_time_s, ...
            full.speedup_vs_shifted);
    end

    fprintf('forced period: first %.3fs, median warm %.3fs\n', ...
        first_forced_time, forced_period_time);
    fprintf('homogeneous period: median %.3fs\n', homogeneous_period_time);
    fprintf('shifted-MG solve: %.3fs, %d iterations, residual %.3e\n', ...
        shifted.total_time, shifted.iterations, shifted.relative_residual);
    fprintf('short GMRES probe residuals: %s\n', mat2str(probe_residual.', 4));
    if isfinite(estimated_applies)
        fprintf('geometric extrapolation to %.1e: about %d period applies\n', ...
            cfg.waveholtz_tol, estimated_applies);
    else
        fprintf('short probe does not support a finite geometric extrapolation\n');
    end
    disp(scenarios);

    results = struct();
    results.created_at = datetime('now');
    results.candidate = spec.name;
    results.problem_cfg = problem_cfg;
    results.cfg = cfg;
    results.mu = mu;
    results.grid_size = problem.n;
    results.N = problem.N;
    results.state_dofs = pre.layout.nstate;
    results.state_MiB = pre.layout.nstate * 8 / 2^20;
    results.estimated_GMRES_basis_MiB = ...
        (cfg.full_gmres_restart + 1) * results.state_MiB;
    results.dt = pre.dt;
    results.steps_per_period = pre.nt;
    results.first_forced_period_s = first_forced_time;
    results.forced_period_s = forced_period_time;
    results.homogeneous_period_s = homogeneous_period_time;
    results.probe_residual = probe_residual;
    results.probe_period_times_s = probe_times;
    results.estimated_waveholtz_applies = estimated_applies;
    results.shifted = shifted;
    results.cost_scenarios = scenarios;
    results.full_waveholtz = full;
    if full.ran
        parallel_batches = spec.driver_cfg.num_layers * ...
            ceil(spec.driver_cfg.train_count / spec.driver_cfg.parallel_workers);
        results.estimated_waveholtz_snapshot_wall_s = ...
            parallel_batches * full.total_time_s;
        results.estimated_waveholtz_ROM_offline_s = ...
            results.estimated_waveholtz_snapshot_wall_s + ...
            spec.expected.non_snapshot_rom_overhead_s;
        results.measured_shifted_ROM_offline_s = ...
            spec.expected.shifted_snapshot_rom_offline_s;
        results.estimated_ROM_offline_speedup = ...
            results.measured_shifted_ROM_offline_s / ...
            results.estimated_waveholtz_ROM_offline_s;
        fprintf(['estimated WaveHoltz ROM offline wall time: %.1fs ' ...
            '(%.1fs snapshot solves + %.1fs measured non-snapshot work), ' ...
            '%.2fx projected speedup\n'], ...
            results.estimated_waveholtz_ROM_offline_s, ...
            results.estimated_waveholtz_snapshot_wall_s, ...
            spec.expected.non_snapshot_rom_overhead_s, ...
            results.estimated_ROM_offline_speedup);
    else
        results.estimated_waveholtz_snapshot_wall_s = NaN;
        results.estimated_waveholtz_ROM_offline_s = NaN;
        results.measured_shifted_ROM_offline_s = ...
            spec.expected.shifted_snapshot_rom_offline_s;
        results.estimated_ROM_offline_speedup = NaN;
    end

    if cfg.save_results
        out_dir = helm_output_path('outputs', 'waveholtz_3d_timing');
        if ~exist(out_dir, 'dir')
            mkdir(out_dir);
        end
        tag = sprintf('%s_%s', spec.name, cfg.output_tag);
        save(fullfile(out_dir, [tag '.mat']), 'results', '-v7.3');
        writetable(scenarios, fullfile(out_dir, [tag '_cost_scenarios.csv']));
    end
end

function cfg = default_cfg()
    cfg = struct();
    cfg.problem_cfg = struct();
    cfg.mu = [];
    cfg.cfl = 0.45;
    cfg.filter_shift = 0.25;
    cfg.timing_repeats = 2;
    cfg.gmres_probe_steps = 4;
    cfg.gmres_reorthogonalize = true;
    cfg.waveholtz_tol = 1e-6;
    cfg.assumed_waveholtz_applies = [10, 25, 50, 100];
    cfg.run_full_waveholtz_solve = false;
    cfg.full_gmres_restart = 20;
    cfg.full_gmres_maxit = 10;
    cfg.shift_beta = 1.2;
    cfg.gmres_tol = 1e-6;
    cfg.gmres_restart = 40;
    cfg.gmres_maxit = 20;
    cfg.mg_max_levels = 6;
    cfg.mg_coarse_max_dofs = 3000;
    cfg.mg_pre_sweeps = 2;
    cfg.mg_post_sweeps = 2;
    cfg.mg_jacobi_weight = 0.6;
    cfg.mg_cycles_per_apply = 1;
    cfg.save_results = true;
    cfg.output_tag = 'large_forward_probe';
end

function pre = prepare_waveholtz_period(problem, mu, rhs, cfg)
    mass_diagonal = problem.mass_diagonal(mu);
    c_squared = reshape(problem.cfg.omega^2 ./ mass_diagonal, problem.n);
    max_speed = sqrt(max(c_squared(:)));
    dt_initial = cfg.cfl * min(problem.spacing) / max_speed;
    [omega_ts, c_w, nt, dt] = match_omega_and_period( ...
        problem.cfg.omega, dt_initial);

    sigma_x = problem.pml.sigma_x / c_w;
    sigma_y = problem.pml.sigma_y / c_w;
    sigma_z = problem.pml.sigma_z / c_w;
    sx_node = reshape(sigma_x(2:end - 1), [], 1, 1);
    sy_node = reshape(sigma_y(2:end - 1), 1, [], 1);
    sz_node = reshape(sigma_z(2:end - 1), 1, 1, []);
    sx_face = reshape(0.5 * (sigma_x(1:end - 1) + sigma_x(2:end)), [], 1, 1);
    sy_face = reshape(0.5 * (sigma_y(1:end - 1) + sigma_y(2:end)), 1, [], 1);
    sz_face = reshape(0.5 * (sigma_z(1:end - 1) + sigma_z(2:end)), 1, 1, []);

    pre = struct();
    pre.n = problem.n;
    pre.spacing = problem.spacing;
    pre.omega = problem.cfg.omega;
    pre.omega_ts = omega_ts;
    pre.c_w = c_w;
    pre.nt = nt;
    pre.dt = dt;
    pre.period = nt * dt;
    pre.filter_shift = cfg.filter_shift;
    pre.c_squared = c_squared;
    pre.force_grid = reshape(real(rhs), problem.n) / problem.cfg.omega;
    pre.px_decay = cn_decay(sx_node, dt);
    pre.py_decay = cn_decay(sy_node, dt);
    pre.pz_decay = cn_decay(sz_node, dt);
    pre.px_drive = cn_drive(sx_node, dt);
    pre.py_drive = cn_drive(sy_node, dt);
    pre.pz_drive = cn_drive(sz_node, dt);
    pre.vx_decay = cn_decay(sx_face, dt);
    pre.vy_decay = cn_decay(sy_face, dt);
    pre.vz_decay = cn_decay(sz_face, dt);
    pre.vx_drive = cn_drive(sx_face, dt);
    pre.vy_drive = cn_drive(sy_face, dt);
    pre.vz_drive = cn_drive(sz_face, dt);
    pre.layout = state_layout(problem.n);
end

function [Pi, elapsed, phasor] = apply_waveholtz_period(pre, x0, source_on, recover)
    timer = tic;
    layout = pre.layout;
    n = pre.n;
    hx = pre.spacing(1);
    hy = pre.spacing(2);
    hz = pre.spacing(3);

    px = reshape(x0(layout.px), n);
    py = reshape(x0(layout.py), n);
    pz = reshape(x0(layout.pz), n);
    vx = reshape(x0(layout.vx), n(1) + 1, n(2), n(3));
    vy = reshape(x0(layout.vy), n(1), n(2) + 1, n(3));
    vz = reshape(x0(layout.vz), n(1), n(2), n(3) + 1);

    accumulator = 0.5 * (1 - pre.filter_shift) * x0;
    if recover
        p_total = px + py + pz;
        cos_accumulator = 0.5 * p_total;
        sin_accumulator = zeros(n);
    else
        cos_accumulator = [];
        sin_accumulator = [];
    end

    grad_x = zeros(n(1) + 1, n(2), n(3));
    grad_y = zeros(n(1), n(2) + 1, n(3));
    grad_z = zeros(n(1), n(2), n(3) + 1);
    for step = 1:pre.nt
        p_total = px + py + pz;
        grad_x(1, :, :) = p_total(1, :, :) / hx;
        grad_x(2:n(1), :, :) = diff(p_total, 1, 1) / hx;
        grad_x(end, :, :) = -p_total(end, :, :) / hx;
        grad_y(:, 1, :) = p_total(:, 1, :) / hy;
        grad_y(:, 2:n(2), :) = diff(p_total, 1, 2) / hy;
        grad_y(:, end, :) = -p_total(:, end, :) / hy;
        grad_z(:, :, 1) = p_total(:, :, 1) / hz;
        grad_z(:, :, 2:n(3)) = diff(p_total, 1, 3) / hz;
        grad_z(:, :, end) = -p_total(:, :, end) / hz;

        vx = pre.vx_decay .* vx + pre.vx_drive .* grad_x;
        vy = pre.vy_decay .* vy + pre.vy_drive .* grad_y;
        vz = pre.vz_decay .* vz + pre.vz_drive .* grad_z;

        div_x = diff(vx, 1, 1) / hx;
        div_y = diff(vy, 1, 2) / hy;
        div_z = diff(vz, 1, 3) / hz;
        t_half = (step - 0.5) * pre.dt;
        if source_on
            forcing = -(sin(pre.omega_ts * t_half) / 3) * pre.force_grid;
        else
            forcing = 0;
        end
        px = pre.px_decay .* px + pre.px_drive .* ...
            (pre.c_squared .* (div_x + forcing));
        py = pre.py_decay .* py + pre.py_drive .* ...
            (pre.c_squared .* (div_y + forcing));
        pz = pre.pz_decay .* pz + pre.pz_drive .* ...
            (pre.c_squared .* (div_z + forcing));

        endpoint_weight = 1;
        if step == pre.nt
            endpoint_weight = 0.5;
        end
        t_now = step * pre.dt;
        weight = endpoint_weight * ...
            (cos(pre.omega_ts * t_now) - pre.filter_shift);
        accumulator(layout.px) = accumulator(layout.px) + weight * px(:);
        accumulator(layout.py) = accumulator(layout.py) + weight * py(:);
        accumulator(layout.pz) = accumulator(layout.pz) + weight * pz(:);
        accumulator(layout.vx) = accumulator(layout.vx) + weight * vx(:);
        accumulator(layout.vy) = accumulator(layout.vy) + weight * vy(:);
        accumulator(layout.vz) = accumulator(layout.vz) + weight * vz(:);

        if recover
            p_total = px + py + pz;
            cos_accumulator = cos_accumulator + ...
                endpoint_weight * cos(pre.omega_ts * t_now) * p_total;
            sin_accumulator = sin_accumulator + ...
                endpoint_weight * sin(pre.omega_ts * t_now) * p_total;
        end
    end

    scale = 2 * pre.dt / pre.period;
    Pi = scale * accumulator;
    if recover
        phasor = scale * (cos_accumulator + 1i * sin_accumulator);
        phasor = phasor(:);
    else
        phasor = [];
    end
    elapsed = toc(timer);
end

function shifted = benchmark_shifted_laplacian(problem, A, rhs, mu, cfg)
    Ashift = problem.assemble_shifted(mu, cfg.shift_beta);
    options = struct('solver', 'mg', 'grid_size', problem.n, ...
        'mg_max_levels', cfg.mg_max_levels, ...
        'mg_coarse_max_dofs', cfg.mg_coarse_max_dofs, ...
        'mg_pre_sweeps', cfg.mg_pre_sweeps, ...
        'mg_post_sweeps', cfg.mg_post_sweeps, ...
        'mg_jacobi_weight', cfg.mg_jacobi_weight, ...
        'mg_cycles_per_apply', cfg.mg_cycles_per_apply);
    [Msolve, setup] = build_shifted_laplacian_preconditioner(Ashift, options);
    gmres_options = struct('restart', cfg.gmres_restart, ...
        'maxit', cfg.gmres_maxit, 'tol', cfg.gmres_tol, ...
        'reorthogonalize', cfg.gmres_reorthogonalize, 'verbose', false);
    [~, solve] = shifted_laplacian_gmres(@(v) A * v, ...
        Msolve, rhs, [], gmres_options);
    shifted = struct('setup_time', setup.setup_time, ...
        'solve_time', solve.total_time, ...
        'total_time', setup.setup_time + solve.total_time, ...
        'iterations', solve.iter_count, ...
        'relative_residual', solve.relres_hist(end));
end

function [residuals, apply_times] = gmres_probe(Afun, rhs, steps, reorthogonalize)
    beta = norm(rhs);
    if beta <= eps || steps == 0
        residuals = zeros(0, 1);
        apply_times = zeros(0, 1);
        return;
    end
    Q = zeros(numel(rhs), steps + 1);
    H = zeros(steps + 1, steps);
    Q(:, 1) = rhs / beta;
    residuals = ones(steps, 1);
    apply_times = zeros(steps, 1);
    reduced_rhs = zeros(steps + 1, 1);
    reduced_rhs(1) = beta;
    for j = 1:steps
        timer = tic;
        w = Afun(Q(:, j));
        apply_times(j) = toc(timer);
        for i = 1:j
            H(i, j) = Q(:, i)' * w;
            w = w - H(i, j) * Q(:, i);
        end
        if reorthogonalize
            for i = 1:j
                correction = Q(:, i)' * w;
                H(i, j) = H(i, j) + correction;
                w = w - correction * Q(:, i);
            end
        end
        H(j + 1, j) = norm(w);
        if H(j + 1, j) > eps
            Q(:, j + 1) = w / H(j + 1, j);
        end
        coefficients = H(1:j + 1, 1:j) \ reduced_rhs(1:j + 1);
        residuals(j) = norm(reduced_rhs(1:j + 1) - ...
            H(1:j + 1, 1:j) * coefficients) / beta;
    end
end

function scenarios = build_cost_scenarios(spec, period_time, shifted_time, applies)
    applies = applies(:);
    solve_periods = applies + 2;
    waveholtz_solve_s = solve_periods * period_time;
    parallel_batches = spec.driver_cfg.num_layers * ...
        ceil(spec.driver_cfg.train_count / spec.driver_cfg.parallel_workers);
    estimated_rom_offline_s = parallel_batches * waveholtz_solve_s;
    ratio_vs_shifted = waveholtz_solve_s / shifted_time;
    scenarios = table(applies, solve_periods, waveholtz_solve_s, ...
        ratio_vs_shifted, estimated_rom_offline_s, ...
        'VariableNames', {'assumed_GMRES_applies', 'periods_per_solve', ...
        'estimated_waveholtz_solve_s', 'ratio_vs_one_shifted_solve', ...
        'estimated_12_snapshot_ROM_wall_s'});
end

function count = estimate_gmres_applies(residuals, tolerance)
    if isempty(residuals) || residuals(end) <= 0 || residuals(end) >= 1
        count = Inf;
        return;
    end
    contraction = residuals(end)^(1 / numel(residuals));
    if contraction <= 0 || contraction >= 1
        count = Inf;
    else
        count = max(numel(residuals), ceil(log(tolerance) / log(contraction)));
    end
end

function layout = state_layout(n)
    nu = prod(n);
    counts = [nu, nu, nu, ...
        (n(1) + 1) * n(2) * n(3), ...
        n(1) * (n(2) + 1) * n(3), ...
        n(1) * n(2) * (n(3) + 1)];
    edges = [0, cumsum(counts)];
    names = {'px', 'py', 'pz', 'vx', 'vy', 'vz'};
    layout = struct();
    for k = 1:numel(names)
        layout.(names{k}) = (edges(k) + 1:edges(k + 1)).';
    end
    layout.nstate = edges(end);
end

function [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt_initial)
    dt = dt_initial;
    if omega * dt / 2 >= 0.98
        dt = 0.98 * 2 / omega;
    end
    for k = 1:6
        omega_ts = (2 / dt) * asin(omega * dt / 2);
        nt = max(10, round(2 * pi / (omega_ts * dt)));
        dt = 2 * pi / (omega_ts * nt);
    end
    omega_ts = (2 / dt) * asin(omega * dt / 2);
    c_w = cos(omega_ts * dt / 2);
end

function value = cn_decay(sigma, dt)
    value = (1 - 0.5 * dt * sigma) ./ (1 + 0.5 * dt * sigma);
end

function value = cn_drive(sigma, dt)
    value = dt ./ (1 + 0.5 * dt * sigma);
end

function count = iteration_count(iter, restart)
    if numel(iter) == 2
        count = max(0, (iter(1) - 1) * restart + iter(2));
    else
        count = iter;
    end
end

function out = merge_structs(base, override)
    out = base;
    names = fieldnames(override);
    for k = 1:numel(names)
        out.(names{k}) = override.(names{k});
    end
end
