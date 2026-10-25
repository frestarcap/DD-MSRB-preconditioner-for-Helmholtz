function results = helm_waveholtz_correction_rom_deepseekbroken(cfg)
    % Standalone single-file script/function for:
    %   trajectory-aware layered Helmholtz FGMRES ROM
    %   + online WaveHoltz / Helmholtz completion.
    %
    % Usage:
    %   results = helm_waveholtz_correction_rom_deepseekbroken();
    %   results = helm_waveholtz_correction_rom_deepseekbroken(cfg);
    %
    % This file is self-contained: all required helpers live below in the
    % same .m file, so it does not rely on external project interfaces.
    %
    % Algorithmic split:
    %   1) Offline references come from high-accuracy WaveHoltz solves.
    %   2) Layered ROM spaces are trained from the corresponding
    %      trajectory-aware Helmholtz FGMRES vectors in the frequency-domain
    %      variable u(mu).
    %   3) Online evaluation starts from zero, performs a few layered
    %      reduced Helmholtz FGMRES steps in u, lifts once to a WaveHoltz
    %      state x when needed, and then lets WaveHoltz fixed-point /
    %      Richardson finish the solve.
    %
    % The target PDE is
    %   Delta u + omega^2 / c(x)^2 * u = f,
    % with c(x)=1 on the left half and c(x)=c_right on the right half.
    % WaveHoltz is used in standard variable-speed form:
    %   (1 / c(x)^2) w_tt = Delta w - f(x) cos(omega t).

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);
    prob = build_problem(cfg);

    fprintf('=== helm_waveholtz_correction_rom ===\n');
    fprintf('Nphys=%d, npml=%d, total interior=%d x %d\n', cfg.Nphys, cfg.npml, prob.nx, prob.ny);
    fprintf('omega(helm)=%.6f, omega(ts)=%.6f, dt=%.3e, nt=%d\n', prob.omega, prob.omega_ts, prob.dt, prob.nt);
    fprintf('interface x = %.4f\n', cfg.interface_x);
    fprintf('source type = %s\n', cfg.source_type);
    fprintf('ROM check target = %s\n', cfg.rom_check_target);
    fprintf('Helmholtz layered FGMRES ROM steps = %d\n', cfg.corr_num_layers);
    fprintf('online completion mode = %s\n', cfg.online_completion_mode);
    fprintf('run WaveHoltz baselines = %d\n', cfg.run_wave_baselines);
    if ~isempty(cfg.corr_pod_singular_value_tols)
        fprintf('layered Helmholtz svd tol = %s\n', mat2str(cfg.corr_pod_singular_value_tols, 4));
    end
    fprintf('Helmholtz GMRES track tol = %.1e\n', cfg.helm_gmres_track_tol);
    fprintf('online fixed-point tol = %.1e, maxit = %d\n', cfg.fp_tol, cfg.fp_maxit);
    if ~isempty(cfg.fp_step_tol)
        fprintf('online fixed-point step tol = %.1e\n', cfg.fp_step_tol);
    end
    fprintf('left-source phys grid = (%d,%d), x=%.4f, y=%.4f\n', ...
        cfg.source_col_phys, cfg.source_row_phys, prob.x_phys(cfg.source_col_phys), prob.y_phys(cfg.source_row_phys));
    fprintf('train params = %s\n', mat2str(cfg.train_params, 4));
    fprintf('test  params = %s\n', mat2str(cfg.test_params, 4));

    rhs = build_rhs(prob, cfg);

    fprintf('\n=== Offline stage 1: WaveHoltz reference solves ===\n');
    train_ref = solve_reference_waveholtz_set(prob, cfg, rhs, cfg.train_params, 'train');

    fprintf('\n=== Offline stage 2: trajectory-aware Helmholtz layered FGMRES ROM ===\n');
    corr_rom = train_correction_rom_layers(prob, cfg, rhs, train_ref);
    print_correction_rom_summary(corr_rom);

    fprintf('\n=== Online evaluation on unseen parameters ===\n');
    test = evaluate_correction_rom_online(prob, cfg, rhs, corr_rom);

    summary = summarize_correction_results(train_ref, corr_rom, test, prob, cfg);
    print_correction_summary(summary);

    if cfg.do_plots
        make_correction_plots(cfg, prob, rhs, corr_rom, test, summary);
    end

    results = struct();
    results.cfg = cfg;
    results.problem = rmfield(prob, {'pml_helm', 'pml_wave'});
    results.rhs = rhs;
    results.train_reference = train_ref;
    results.correction_rom = corr_rom;
    results.test = test;
    results.summary = summary;
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'L', 1.0);
    cfg = set_default(cfg, 'omega', 53.5);
    cfg = set_default(cfg, 'Nphys', 125);
    cfg = set_default(cfg, 'ispml', 1);
    cfg = set_default(cfg, 'npml', []);
    cfg = set_default(cfg, 'sigma_max', 80);
    cfg = set_default(cfg, 'CFL', 0.5);
    cfg = set_default(cfg, 'filter_shift', 0.25);
    cfg = set_default(cfg, 'interface_x', []);

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

    cfg = set_default(cfg, 'train_params', []);
    cfg = set_default(cfg, 'test_params', []);
    cfg = set_default(cfg, 'pod_energy_target', 1e-9);
    cfg = set_default(cfg, 'pod_singular_value_tol', 1e-9);
    cfg = set_default(cfg, 'pod_min_rank', 1);
    cfg = set_default(cfg, 'pod_max_rank', []);
    cfg = set_default(cfg, 'rom_check_target', 'wave');

    cfg = set_default(cfg, 'corr_num_layers', 5);
    cfg = set_default(cfg, 'corr_pod_energy_target', 0.999);
    cfg = set_default(cfg, 'corr_pod_singular_value_tol', 1e-10);
    cfg = set_default(cfg, 'corr_pod_singular_value_tols', logspace(-5,-9,5));
    cfg = set_default(cfg, 'corr_pod_min_rank', 1);
    cfg = set_default(cfg, 'corr_pod_max_rank', []);
    cfg = set_default(cfg, 'helm_gmres_track_tol', 1e-14);
    cfg = set_default(cfg, 'online_completion_mode', 'helmholtz');
    cfg = set_default(cfg, 'run_wave_baselines', true);
    cfg = set_default(cfg, 'helm_completion_restart', 20);
    cfg = set_default(cfg, 'helm_completion_maxit', 40);
    cfg = set_default(cfg, 'helm_completion_tol', []);
    cfg = set_default(cfg, 'helm_baseline_iters', 60);
    cfg = set_default(cfg, 'helm_baseline_tol', 0);
    cfg = set_default(cfg, 'wh_max_iter', 1200);
    cfg = set_default(cfg, 'wh_tol', 1e-14);
    cfg = set_default(cfg, 'wh_gmres_restart', 20);
    cfg = set_default(cfg, 'wh_gmres_maxit', 40);
    cfg = set_default(cfg, 'wh_verbose_gmres', false);
    cfg = set_default(cfg, 'fp_tol', 1e-14);
    cfg = set_default(cfg, 'fp_step_tol', 1e-14);
    cfg = set_default(cfg, 'fp_maxit', 200);
    cfg = set_default(cfg, 'train_params',linspace(0.8, 0.95, 24));

    cfg = set_default(cfg, 'do_reference_check', true);
    cfg = set_default(cfg, 'plot_test_index', []);
    cfg = set_default(cfg, 'do_plots', true);
    cfg = set_default(cfg, 'parallel_train', true);
    cfg = set_default(cfg, 'parallel_train_min_cases', 2);
    cfg = set_default(cfg, 'parallel_test', true);
    cfg = set_default(cfg, 'parallel_test_min_cases', 2);

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
    if isempty(cfg.train_params)
        cfg.train_params = linspace(0.8, 0.95, 30);
    end
    if isempty(cfg.test_params)
        if numel(cfg.train_params) >= 2
            cfg.test_params = 0.5 * (cfg.train_params(1:end-1) + cfg.train_params(2:end));
        else
            cfg.test_params = cfg.train_params .* 1.01;
        end
    end

    cfg.train_params = unique(cfg.train_params(:).');
    cfg.test_params = unique(cfg.test_params(:).');

    validate_param_array(cfg.train_params, 'train_params');
    validate_param_array(cfg.test_params, 'test_params');
    if any(abs(cfg.test_params(:) - cfg.train_params(:).') < 1e-12)
        error('test_params must be unseen and cannot overlap train_params.');
    end
    if cfg.source_col_phys < 1 || cfg.source_col_phys > floor(cfg.Nphys / 2)
        error('source_col_phys=%d must lie on the left half of the physical domain.', cfg.source_col_phys);
    end
    if cfg.source_row_phys < 1 || cfg.source_row_phys > cfg.Nphys
        error('source_row_phys=%d is outside [1,%d].', cfg.source_row_phys, cfg.Nphys);
    end
    valid_source_types = {'point', 'gaussian'};
    if ~any(strcmp(cfg.source_type, valid_source_types))
        error('source_type=%s must be one of: %s.', cfg.source_type, strjoin(valid_source_types, ', '));
    end
    valid_point_profiles = {'box', 'cell'};
    if ~any(strcmp(cfg.point_source_profile, valid_point_profiles))
        error('point_source_profile=%s must be one of: %s.', cfg.point_source_profile, strjoin(valid_point_profiles, ', '));
    end
    valid_check_targets = {'wave', 'helm'};
    if ~any(strcmp(cfg.rom_check_target, valid_check_targets))
        error('rom_check_target=%s must be one of: %s.', cfg.rom_check_target, strjoin(valid_check_targets, ', '));
    end
    valid_completion_modes = {'waveholtz', 'helmholtz'};
    if ~any(strcmp(cfg.online_completion_mode, valid_completion_modes))
        error('online_completion_mode=%s must be one of: %s.', cfg.online_completion_mode, strjoin(valid_completion_modes, ', '));
    end
    if isempty(cfg.run_wave_baselines)
        cfg.run_wave_baselines = true;
    end
    if isempty(cfg.helm_completion_tol)
        cfg.helm_completion_tol = cfg.helm_gmres_track_tol;
    end
    if strcmp(cfg.rom_check_target, 'helm')
        cfg.do_reference_check = true;
    end
    if strcmp(cfg.online_completion_mode, 'helmholtz')
        cfg.do_reference_check = true;
    end
end

function validate_param_array(vals, name)
    if isempty(vals)
        error('%s cannot be empty.', name);
    end
    if ~isnumeric(vals) || ~isreal(vals)
        error('%s must be a real numeric array.', name);
    end
    if any(~isfinite(vals(:)))
        error('%s contains non-finite values.', name);
    end
    if any(vals(:) <= 0)
        error('%s must contain positive speeds.', name);
    end
end

function prob = build_problem(cfg)
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

    prob.c_max = max([1.0; cfg.train_params(:); cfg.test_params(:)]);
    dt0 = cfg.CFL * min(prob.hx, prob.hy) / max(prob.c_max, 1e-14);
    [prob.omega_ts, prob.c_w, prob.nt, prob.dt] = match_omega_and_period(prob.omega, dt0);
    prob.T_ts = prob.nt * prob.dt;

    Lpml_x = prob.npml * prob.hx;
    Lpml_y = prob.npml * prob.hy;
    prob.x = linspace(-Lpml_x, cfg.L + Lpml_x, prob.nx + 2);
    prob.y = linspace(-Lpml_y, cfg.L + Lpml_y, prob.ny + 2);
    prob.x_int = prob.x(2:end-1).';
    prob.y_int = prob.y(2:end-1).';
    prob.x_phys = prob.x(prob.I + 1).';
    prob.y_phys = prob.y(prob.J + 1).';

    prob.pml_helm = build_pml_outer(prob.nx, prob.ny, prob.omega, cfg.npml, cfg.sigma_max, cfg.ispml);
    prob.pml_wave = prob.pml_helm;
    prob.pml_wave.sigma_x = prob.pml_helm.sigma_x / prob.c_w;
    prob.pml_wave.sigma_y = prob.pml_helm.sigma_y / prob.c_w;
    prob.pml_wave.c_w_ref = prob.c_w;
    prob.pml_wave.sigma_scaled_for_wave = true;

    [prob.nu, prob.nvx, prob.nvy, prob.nv, prob.nfull] = state_sizes(prob.nx, prob.ny);

    prob.chi_right_grid = repmat(prob.x_int > cfg.interface_x, 1, prob.ny);
    prob.chi_right = prob.chi_right_grid(:);

    prob.K_pml = build_pml_stiffness(prob.nx, prob.ny, prob.hx, prob.hy, prob.pml_helm);
    prob.A_base = prob.K_pml + (prob.omega ^ 2) * speye(prob.Ntot);
    prob.A_right = spdiags((prob.omega ^ 2) * double(prob.chi_right), ...
        0, prob.Ntot, prob.Ntot);
end

function dataset = solve_reference_waveholtz_set(prob, cfg, rhs, params, label)
    ns = numel(params);
    dataset = struct();
    dataset.params = params(:);
    dataset.x_ref_snapshots = zeros(prob.nfull, ns);
    dataset.u_ref_snapshots = zeros(prob.Ntot, ns);
    dataset.b_vecs = zeros(prob.nfull, ns);
    dataset.solve_time = zeros(ns, 1);
    dataset.recovery_time = zeros(ns, 1);
    dataset.iter = zeros(ns, 1);
    dataset.relres = zeros(ns, 1);
    dataset.flag = zeros(ns, 1);

    use_parallel = should_parallelize_cases(cfg.parallel_train, ns, cfg.parallel_train_min_cases);
    fprintf('%s parallel = %d (%d cases)\n', label, use_parallel, ns);
    case_results = cell(ns, 1);
    if use_parallel
        parfor k = 1:ns
            case_results{k} = solve_one_reference_case(prob, cfg, rhs, params(k));
        end
    else
        for k = 1:ns
            case_results{k} = solve_one_reference_case(prob, cfg, rhs, params(k));
        end
    end

    for k = 1:ns
        case_result = case_results{k};
        dataset.x_ref_snapshots(:, k) = case_result.x_ref;
        dataset.u_ref_snapshots(:, k) = case_result.u_ref;
        dataset.b_vecs(:, k) = case_result.b_vec;
        dataset.solve_time(k) = case_result.info.solve_time;
        dataset.recovery_time(k) = case_result.info.recovery_time;
        dataset.iter(k) = case_result.info.iter_count;
        dataset.relres(k) = case_result.info.relres;
        dataset.flag(k) = case_result.info.flag;
        fprintf('%s\n', strrep(case_result.log_line, 'case ??:', sprintf('%s %2d/%2d:', label, k, ns)));
    end
end

function case_result = solve_one_reference_case(prob, cfg, rhs, c_right)
    [x_ref, u_ref, b_vec, info] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, c_right, zeros(prob.nfull, 1));
    case_result = struct();
    case_result.x_ref = x_ref;
    case_result.u_ref = u_ref;
    case_result.b_vec = b_vec;
    case_result.info = info;
    case_result.log_line = sprintf('case ??: c_right=%.4f, solve=%.3fs, recover=%.3fs, iter=%3d, relres=%.3e', ...
        c_right, info.solve_time, info.recovery_time, info.iter_count, info.relres);
end

function [x_sol, uhat, b_vec, info] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, c_right, x0)
    if nargin < 5 || isempty(x0)
        x0 = zeros(prob.nfull, 1);
    end
    [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right);

    tsolve = tic;
    [x_sol, flag, relres, iter, resvec] = gmres_restarted_verbose( ...
        Afun, b_vec, cfg.wh_gmres_restart, cfg.wh_tol, cfg.wh_gmres_maxit, x0, cfg.wh_verbose_gmres);
    solve_time = toc(tsolve);

    trec = tic;
    uhat = recover_u_phasor_timeproj(prob, x_sol, rhs, c_right);
    recovery_time = toc(trec);

    info = struct();
    info.flag = flag;
    info.relres = relres;
    info.iter = iter;
    info.iter_count = iteration_count(iter, cfg.wh_gmres_restart);
    info.resvec = resvec;
    info.solve_time = solve_time;
    info.recovery_time = recovery_time;
end

function [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right)
    b_vec = apply_pi_piecewise(prob, zeros(prob.nfull, 1), rhs, true, c_right, cfg.filter_shift);
    Afun = build_waveholtz_apply_only(prob, cfg, c_right);
end

function Afun = build_waveholtz_apply_only(prob, cfg, c_right)
    zero_rhs = zeros(prob.nu, 1);
    Afun = @(x) x - apply_pi_piecewise(prob, x, zero_rhs, false, c_right, cfg.filter_shift);
end

function uhat = recover_u_phasor_timeproj(prob, x_state, rhs, c_right)
    Wconv = solve_wave_one_period_piecewise(prob, x_state, rhs, c_right);
    t_grid = (0:prob.nt).' * prob.dt;
    q_cos = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, cos(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
    q_sin = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, sin(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
    uhat = q_cos + 1i * q_sin;
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

function x_real = lift_frequency_solution_to_real_state(prob, uhat, c_right)
    xhat = recover_fullstate_frequency_from_u(prob, uhat, c_right);
    x_real = real(xhat);
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
    theta_coeff = (sx_full(2:end-1, 2:end-1) .* sy_full(2:end-1, 2:end-1)) .* (prob.c_w ^ 2) ./ c_sq_full(2:end-1, 2:end-1);
end


function corr_rom = train_correction_rom_layers(prob, cfg, rhs, train_ref)
    ns = numel(train_ref.params);
    nlayers = cfg.corr_num_layers;
    layer_svd_tols = expand_layer_parameter(cfg.corr_pod_singular_value_tols, cfg.corr_pod_singular_value_tol, nlayers, 'corr_pod_singular_value_tols');
    corr_rom = struct();
    corr_rom.num_layers = nlayers;
    corr_rom.layers = cell(nlayers, 1);
    corr_rom.x_init_train = zeros(prob.nfull, ns);
    corr_rom.u_init_train = zeros(prob.Ntot, ns);
    corr_rom.u_train_after_layer = cell(nlayers, 1);
    corr_rom.x_train_after_layer = cell(nlayers, 1);

    x_curr = zeros(prob.nfull, ns);
    u_init = zeros(prob.Ntot, ns);
    corr_rom.x_init_train = x_curr;
    corr_rom.u_init_train = u_init;

    train_state = initialize_trajectory_aware_training_state(prob, rhs, train_ref.params, u_init, train_ref.u_ref_snapshots);
    for j = 1:nlayers
        eta_snapshots = zeros(prob.Ntot, ns);
        if j == 1
            for k = 1:ns
                eta_snapshots(:, k) = train_state.eta_hist{k}(:, 1);
            end
        else
            prev_layer = corr_rom.layers{j - 1};
            for k = 1:ns
                [train_state, eta_curr] = advance_trajectory_aware_training_sample( ...
                    prob, train_state, prev_layer, train_ref.params(k), k, j);
                eta_snapshots(:, k) = eta_curr;
            end
        end

        pod = build_pod_basis_custom(eta_snapshots, cfg.corr_pod_energy_target, layer_svd_tols(j), cfg.corr_pod_min_rank, cfg.corr_pod_max_rank);
        layer = struct();
        layer.index = j;
        layer.singular_value_tol = layer_svd_tols(j);
        layer.eta_snapshots = eta_snapshots;
        layer.pod = pod;
        layer.basis_u = pod.basis;
        layer.Ar_base = layer.basis_u' * (prob.A_base * layer.basis_u);
        layer.Ar_right = layer.basis_u' * (prob.A_right * layer.basis_u);
        layer.beta0 = train_state.beta0;
        layer.q_snapshot_norm = zeros(ns, 1);
        layer.train_state_error_after = zeros(ns, 1);
        layer.train_freq_error_after = zeros(ns, 1);
        layer.train_fgmres_relres = zeros(ns, 1);
        x_after = zeros(prob.nfull, ns);
        u_after = zeros(prob.Ntot, ns);

        corr_rom.layers{j} = layer;
        prefix_layers = corr_rom.layers(1:j);
        prefix_rom = struct('num_layers', j, 'layers', {prefix_layers});
        use_parallel = should_parallelize_cases(cfg.parallel_train, ns, cfg.parallel_train_min_cases);
        case_results = cell(ns, 1);
        if use_parallel
            parfor k = 1:ns
                c_right = train_ref.params(k);
                [u_local, info_train] = layered_fgmres_precorrection(prob, cfg, rhs, c_right, u_init(:, k), prefix_rom);
                x_local = lift_frequency_solution_to_real_state(prob, u_local, c_right);
                freq_err = norm(u_local - train_ref.u_ref_snapshots(:, k)) / max(norm(train_ref.u_ref_snapshots(:, k)), 1e-16);
                state_err = norm(x_local - train_ref.x_ref_snapshots(:, k)) / max(norm(train_ref.x_ref_snapshots(:, k)), 1e-16);
                Ahelm = build_helmholtz_operator(prob, c_right);
                relres_local = norm(rhs - Ahelm * u_local) / max(norm(rhs), 1e-16);
                if info_train.iter_count < j
                    relres_local = info_train.relres_final;
                end
                case_results{k} = struct( ...
                    'u_after', u_local, ...
                    'x_after', x_local, ...
                    'freq_err', freq_err, ...
                    'state_err', state_err, ...
                    'relres', relres_local, ...
                    'q_norm', norm(train_state.q_curr(:, k)));
            end
        else
            for k = 1:ns
                c_right = train_ref.params(k);
                [u_local, info_train] = layered_fgmres_precorrection(prob, cfg, rhs, c_right, u_init(:, k), prefix_rom);
                x_local = lift_frequency_solution_to_real_state(prob, u_local, c_right);
                freq_err = norm(u_local - train_ref.u_ref_snapshots(:, k)) / max(norm(train_ref.u_ref_snapshots(:, k)), 1e-16);
                state_err = norm(x_local - train_ref.x_ref_snapshots(:, k)) / max(norm(train_ref.x_ref_snapshots(:, k)), 1e-16);
                Ahelm = build_helmholtz_operator(prob, c_right);
                relres_local = norm(rhs - Ahelm * u_local) / max(norm(rhs), 1e-16);
                if info_train.iter_count < j
                    relres_local = info_train.relres_final;
                end
                case_results{k} = struct( ...
                    'u_after', u_local, ...
                    'x_after', x_local, ...
                    'freq_err', freq_err, ...
                    'state_err', state_err, ...
                    'relres', relres_local, ...
                    'q_norm', norm(train_state.q_curr(:, k)));
            end
        end
        for k = 1:ns
            u_after(:, k) = case_results{k}.u_after;
            x_after(:, k) = case_results{k}.x_after;
            layer.train_freq_error_after(k) = case_results{k}.freq_err;
            layer.train_state_error_after(k) = case_results{k}.state_err;
            layer.train_fgmres_relres(k) = case_results{k}.relres;
            layer.q_snapshot_norm(k) = case_results{k}.q_norm;
        end

        corr_rom.layers{j} = layer;
        corr_rom.u_train_after_layer{j} = u_after;
        corr_rom.x_train_after_layer{j} = x_after;
        fprintf(['layer %2d/%2d: svd tol = %.1e, eta snapshot size = %d x %d, basis rank = %d, ' ...
            'train relres mean = %.3e, train freq err mean = %.3e, train state err mean = %.3e\n'], ...
            j, nlayers, layer.singular_value_tol, size(eta_snapshots, 1), size(eta_snapshots, 2), pod.rank_used, ...
            mean(layer.train_fgmres_relres), mean(layer.train_freq_error_after), mean(layer.train_state_error_after));
    end
end

function print_correction_rom_summary(corr_rom)
    if corr_rom.num_layers == 0
        fprintf('no layered Helmholtz ROM steps requested.\n');
        return;
    end
    for j = 1:corr_rom.num_layers
        pod = corr_rom.layers{j}.pod;
        fprintf('layered FGMRES ROM layer %2d: tol = %.1e, rank = %d, r90/r95/r99 = %d / %d / %d, train relres mean = %.3e\n', ...
            j, corr_rom.layers{j}.singular_value_tol, pod.rank_used, pod.r90, pod.r95, pod.r99, mean(corr_rom.layers{j}.train_fgmres_relres));
    end
end

function test = evaluate_correction_rom_online(prob, cfg, rhs, corr_rom)
    ns = numel(cfg.test_params);
    nlayers = corr_rom.num_layers;
    test = initialize_online_test_struct(prob, ns, nlayers, cfg.do_reference_check);
    test.params = cfg.test_params(:);
    use_parallel = should_parallelize_online_eval(cfg, ns);
    fprintf('online test parallel = %d (%d cases)\n', use_parallel, ns);

    case_results = cell(ns, 1);
    if use_parallel
        parfor k = 1:ns
            case_results{k} = evaluate_one_online_case(prob, cfg, rhs, corr_rom, cfg.test_params(k));
        end
    else
        for k = 1:ns
            case_results{k} = evaluate_one_online_case(prob, cfg, rhs, corr_rom, cfg.test_params(k));
        end
    end

    for k = 1:ns
        test = store_online_case_result(test, case_results{k}, k, nlayers, cfg);
        fprintf('%s\n', strrep(case_results{k}.log_line, 'test ??:', sprintf('test %2d/%2d:', k, ns)));
    end
end

function test = initialize_online_test_struct(prob, ns, nlayers, do_reference_check)
    test = struct();
    test.params = zeros(ns, 1);
    test.wave_u_ref = zeros(prob.Ntot, ns);
    test.wave_x_ref = zeros(prob.nfull, ns);
    test.wave_time = zeros(ns, 1);
    test.wave_iter = zeros(ns, 1);
    test.plain_fp_u = zeros(prob.Ntot, ns);
    test.plain_fp_x = zeros(prob.nfull, ns);
    test.plain_fp_time = zeros(ns, 1);
    test.plain_fp_iter = zeros(ns, 1);
    test.plain_fp_relres_final = zeros(ns, 1);
    test.plain_fp_op_apply = zeros(ns, 1);
    test.plain_fp_relres_hist = cell(ns, 1);
    test.plain_fp_step_hist = cell(ns, 1);
    test.plain_fp_stop_reason = cell(ns, 1);
    test.online_final_u = zeros(prob.Ntot, ns);
    test.online_final_x = zeros(prob.nfull, ns);
    test.online_precorr_u = zeros(prob.Ntot, ns);
    test.online_precorr_x = zeros(prob.nfull, ns);
    test.online_precorr_time = zeros(ns, 1);
    test.online_lift_time = zeros(ns, 1);
    test.online_precorr_matvec = zeros(ns, 1);
    test.online_precorr_relres_final = zeros(ns, 1);
    test.online_fp_time = zeros(ns, 1);
    test.online_total_time = zeros(ns, 1);
    test.online_fp_iter = zeros(ns, 1);
    test.online_fp_relres_final = zeros(ns, 1);
    test.online_fp_op_apply = zeros(ns, 1);
    test.online_fp_relres_hist = cell(ns, 1);
    test.online_fp_step_hist = cell(ns, 1);
    test.online_fp_stop_reason = cell(ns, 1);
    test.online_combined_relres_hist = cell(ns, 1);
    test.online_combined_transition_idx = zeros(ns, 1);
    test.helm_baseline_time = zeros(ns, 1);
    test.helm_baseline_iter = zeros(ns, 1);
    test.helm_baseline_op_apply = zeros(ns, 1);
    test.helm_baseline_relres_final = zeros(ns, 1);
    test.helm_baseline_relres_hist = cell(ns, 1);
    test.corr_u_layers = cell(nlayers, 1);
    test.corr_x_layers = cell(nlayers, 1);
    test.corr_time_layers = zeros(nlayers, ns);
    for j = 1:nlayers
        test.corr_u_layers{j} = zeros(prob.Ntot, ns);
        test.corr_x_layers{j} = zeros(prob.nfull, ns);
    end
    if do_reference_check
        test.helm_u_ref = zeros(prob.Ntot, ns);
        test.helm_time = zeros(ns, 1);
    else
        test.helm_u_ref = [];
        test.helm_time = [];
    end

    test.plain_fp_err_wave_full = zeros(ns, 1);
    test.plain_fp_err_wave_phys = zeros(ns, 1);
    test.plain_fp_err_helm_full = zeros(ns, 1);
    test.plain_fp_err_helm_phys = zeros(ns, 1);
    test.layer_err_wave_full = zeros(nlayers, ns);
    test.layer_err_wave_phys = zeros(nlayers, ns);
    test.layer_err_helm_full = zeros(nlayers, ns);
    test.layer_err_helm_phys = zeros(nlayers, ns);
    test.online_final_err_wave_full = zeros(ns, 1);
    test.online_final_err_wave_phys = zeros(ns, 1);
    test.online_final_err_helm_full = zeros(ns, 1);
    test.online_final_err_helm_phys = zeros(ns, 1);
end

function tf = should_parallelize_online_eval(cfg, ns)
    tf = should_parallelize_cases(cfg.parallel_test, ns, cfg.parallel_test_min_cases);
end

function tf = should_parallelize_cases(enabled, ns, min_cases)
    tf = enabled && ns >= min_cases;
    if ~tf
        return;
    end
    tf = license('test', 'Distrib_Computing_Toolbox') && exist('parpool', 'file') == 2;
end

function case_result = evaluate_one_online_case(prob, cfg, rhs, corr_rom, c_right)
    nlayers = corr_rom.num_layers;
    case_result = struct();

    [x_ref, u_ref, b_vec, info_ref] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, c_right, zeros(prob.nfull, 1));
    case_result.wave_x_ref = x_ref;
    case_result.wave_u_ref = u_ref;
    case_result.wave_time = info_ref.solve_time + info_ref.recovery_time;
    case_result.wave_iter = info_ref.iter_count;

    if cfg.do_reference_check
        alpha = (1 / c_right^2) - 1;
        tref = tic;
        case_result.helm_u_ref = (prob.A_base + alpha * prob.A_right) \ rhs;
        case_result.helm_time = toc(tref);
    else
        case_result.helm_u_ref = [];
        case_result.helm_time = [];
    end

    [~, info_helm_baseline] = run_plain_helmholtz_gmres( ...
        prob, rhs, c_right, zeros(prob.Ntot, 1), ...
        cfg.helm_baseline_iters, 1, cfg.helm_baseline_tol, false);
    case_result.helm_baseline_time = info_helm_baseline.total_time;
    case_result.helm_baseline_iter = info_helm_baseline.iter_count;
    case_result.helm_baseline_op_apply = info_helm_baseline.op_apply_count;
    case_result.helm_baseline_relres_final = info_helm_baseline.relres_final;
    case_result.helm_baseline_relres_hist = info_helm_baseline.relres_hist;

    u_init_online = zeros(prob.Ntot, 1);

    need_wave_operator = cfg.run_wave_baselines || strcmp(cfg.online_completion_mode, 'waveholtz');
    if need_wave_operator
        Afun = build_waveholtz_apply_only(prob, cfg, c_right);
    else
        Afun = [];
    end

    if cfg.run_wave_baselines
        [x_plain, info_plain] = run_online_fixedpoint(prob, cfg, rhs, c_right, b_vec, Afun, zeros(prob.nfull, 1));
        case_result.plain_fp_x = x_plain;
        case_result.plain_fp_u = info_plain.u_final;
        case_result.plain_fp_time = info_plain.total_time;
        case_result.plain_fp_iter = info_plain.iter_count;
        case_result.plain_fp_relres_final = info_plain.relres_final;
        case_result.plain_fp_op_apply = info_plain.op_apply_count;
        case_result.plain_fp_relres_hist = info_plain.relres_hist;
        case_result.plain_fp_step_hist = info_plain.step_hist;
        case_result.plain_fp_stop_reason = info_plain.stop_reason;
        [case_result.plain_fp_err_wave_full, case_result.plain_fp_err_wave_phys] = compute_solution_errors(info_plain.u_final, u_ref, prob);

        if cfg.do_reference_check
            [case_result.plain_fp_err_helm_full, case_result.plain_fp_err_helm_phys] = compute_solution_errors(info_plain.u_final, case_result.helm_u_ref, prob);
        else
            case_result.plain_fp_err_helm_full = NaN;
            case_result.plain_fp_err_helm_phys = NaN;
        end
    else
        case_result.plain_fp_x = zeros(prob.nfull, 1);
        case_result.plain_fp_u = zeros(prob.Ntot, 1);
        case_result.plain_fp_time = NaN;
        case_result.plain_fp_iter = NaN;
        case_result.plain_fp_relres_final = NaN;
        case_result.plain_fp_op_apply = NaN;
        case_result.plain_fp_relres_hist = [];
        case_result.plain_fp_step_hist = [];
        case_result.plain_fp_stop_reason = '';
        case_result.plain_fp_err_wave_full = NaN;
        case_result.plain_fp_err_wave_phys = NaN;
        case_result.plain_fp_err_helm_full = NaN;
        case_result.plain_fp_err_helm_phys = NaN;
    end

    [u_precorr, info_precorr] = run_helmholtz_precorrection(prob, cfg, rhs, c_right, u_init_online, corr_rom);
    if strcmp(cfg.online_completion_mode, 'helmholtz')
        x_precorr = zeros(prob.nfull, 1);
        lift_time = 0;
        [u_online, info_online] = run_plain_helmholtz_gmres( ...
            prob, rhs, c_right, u_precorr, ...
            cfg.helm_completion_restart, cfg.helm_completion_maxit, cfg.helm_completion_tol, false);
        x_online = zeros(prob.nfull, 1);
        online_u_final = u_online;
        online_relres_hist = stitch_online_relres_hist(info_precorr.relres_hist, info_online.relres_hist);
    else
        tlift = tic;
        x_precorr = lift_frequency_solution_to_real_state(prob, u_precorr, c_right);
        lift_time = toc(tlift);
        [x_online, info_online] = run_online_fixedpoint(prob, cfg, rhs, c_right, b_vec, Afun, x_precorr);
        online_u_final = info_online.u_final;
        online_relres_hist = stitch_online_relres_hist(info_precorr.relres_hist, info_online.relres_hist);
    end

    case_result.online_precorr_u = u_precorr;
    case_result.online_precorr_x = x_precorr;
    case_result.online_precorr_time = info_precorr.total_time;
    case_result.online_lift_time = lift_time;
    case_result.online_precorr_matvec = info_precorr.matvec_count;
    case_result.online_precorr_relres_final = info_precorr.relres_final;
    case_result.online_final_x = x_online;
    case_result.online_final_u = online_u_final;
    case_result.online_fp_time = info_online.total_time;
    case_result.online_total_time = info_precorr.total_time + lift_time + info_online.total_time;
    case_result.online_fp_iter = info_online.iter_count;
    case_result.online_fp_relres_final = info_online.relres_final;
    case_result.online_fp_op_apply = info_online.op_apply_count;
    case_result.online_fp_relres_hist = info_online.relres_hist;
    case_result.online_fp_step_hist = info_online.step_hist;
    case_result.online_fp_stop_reason = info_online.stop_reason;
    case_result.online_combined_relres_hist = online_relres_hist;
    case_result.online_combined_transition_idx = numel(info_precorr.relres_hist);
    [case_result.online_final_err_wave_full, case_result.online_final_err_wave_phys] = compute_solution_errors(online_u_final, u_ref, prob);
    if cfg.do_reference_check
        [case_result.online_final_err_helm_full, case_result.online_final_err_helm_phys] = compute_solution_errors(online_u_final, case_result.helm_u_ref, prob);
    else
        case_result.online_final_err_helm_full = NaN;
        case_result.online_final_err_helm_phys = NaN;
    end

    case_result.corr_time_layers = zeros(nlayers, 1);
    case_result.corr_u_layers = cell(nlayers, 1);
    case_result.corr_x_layers = cell(nlayers, 1);
    case_result.layer_err_wave_full = zeros(nlayers, 1);
    case_result.layer_err_wave_phys = zeros(nlayers, 1);
    case_result.layer_err_helm_full = zeros(nlayers, 1);
    case_result.layer_err_helm_phys = zeros(nlayers, 1);
    for j = 1:nlayers
        case_result.corr_time_layers(j) = info_precorr.layer_time(j);
        if ~isempty(info_precorr.layer_u{j})
            case_result.corr_u_layers{j} = info_precorr.layer_u{j};
            case_result.corr_x_layers{j} = lift_frequency_solution_to_real_state(prob, info_precorr.layer_u{j}, c_right);
        else
            case_result.corr_u_layers{j} = u_precorr;
            case_result.corr_x_layers{j} = x_precorr;
        end
        [case_result.layer_err_wave_full(j), case_result.layer_err_wave_phys(j)] = ...
            compute_solution_errors(case_result.corr_u_layers{j}, u_ref, prob);
        if cfg.do_reference_check
            [case_result.layer_err_helm_full(j), case_result.layer_err_helm_phys(j)] = ...
                compute_solution_errors(case_result.corr_u_layers{j}, case_result.helm_u_ref, prob);
        else
            case_result.layer_err_helm_full(j) = NaN;
            case_result.layer_err_helm_phys(j) = NaN;
        end
    end

    if strcmp(cfg.rom_check_target, 'helm') && cfg.do_reference_check
        final_full = case_result.online_final_err_helm_full;
        final_phys = case_result.online_final_err_helm_phys;
    else
        final_full = case_result.online_final_err_wave_full;
        final_phys = case_result.online_final_err_wave_phys;
    end

    if strcmp(cfg.online_completion_mode, 'helmholtz') && ~cfg.run_wave_baselines
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref-gmres=%.3fs (%3d it), ' ...
            'helm-FGMRES=%.3fs + plain-helm=%.3fs (%3d it, %.2e), helm100=%.3fs (%3d it, %.2e), ' ...
            'final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.online_precorr_time, case_result.online_fp_time, case_result.online_fp_iter, case_result.online_fp_relres_final, ...
            case_result.helm_baseline_time, case_result.helm_baseline_iter, case_result.helm_baseline_relres_final, ...
            final_full, final_phys);
    elseif strcmp(cfg.online_completion_mode, 'helmholtz')
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref-gmres=%.3fs (%3d it), ' ...
            'plain-fp=%.3fs (%3d it, %.2e, %3d apply), ' ...
            'helm-FGMRES=%.3fs + plain-helm=%.3fs (%3d it, %.2e), helm100=%.3fs (%3d it, %.2e), ' ...
            'final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.plain_fp_time, case_result.plain_fp_iter, case_result.plain_fp_relres_final, case_result.plain_fp_op_apply, ...
            case_result.online_precorr_time, case_result.online_fp_time, case_result.online_fp_iter, case_result.online_fp_relres_final, ...
            case_result.helm_baseline_time, case_result.helm_baseline_iter, case_result.helm_baseline_relres_final, ...
            final_full, final_phys);
    else
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref-gmres=%.3fs (%3d it), ' ...
            'plain-fp=%.3fs (%3d it, %.2e, %3d apply), helm-FGMRES=%.3fs + lift=%.3fs + WH=%.3fs (%3d it, %.2e, %3d apply), ' ...
            'helm100=%.3fs (%3d it, %.2e), final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.plain_fp_time, case_result.plain_fp_iter, case_result.plain_fp_relres_final, case_result.plain_fp_op_apply, ...
            case_result.online_precorr_time, case_result.online_lift_time, case_result.online_fp_time, case_result.online_fp_iter, case_result.online_fp_relres_final, case_result.online_fp_op_apply, ...
            case_result.helm_baseline_time, case_result.helm_baseline_iter, case_result.helm_baseline_relres_final, ...
            final_full, final_phys);
    end
end

function test = store_online_case_result(test, case_result, k, nlayers, cfg)
    test.wave_x_ref(:, k) = case_result.wave_x_ref;
    test.wave_u_ref(:, k) = case_result.wave_u_ref;
    test.wave_time(k) = case_result.wave_time;
    test.wave_iter(k) = case_result.wave_iter;

    if cfg.do_reference_check
        test.helm_u_ref(:, k) = case_result.helm_u_ref;
        test.helm_time(k) = case_result.helm_time;
    end

    test.plain_fp_x(:, k) = case_result.plain_fp_x;
    test.plain_fp_u(:, k) = case_result.plain_fp_u;
    test.plain_fp_time(k) = case_result.plain_fp_time;
    test.plain_fp_iter(k) = case_result.plain_fp_iter;
    test.plain_fp_relres_final(k) = case_result.plain_fp_relres_final;
    test.plain_fp_op_apply(k) = case_result.plain_fp_op_apply;
    test.plain_fp_relres_hist{k} = case_result.plain_fp_relres_hist;
    test.plain_fp_step_hist{k} = case_result.plain_fp_step_hist;
    test.plain_fp_stop_reason{k} = case_result.plain_fp_stop_reason;

    test.online_precorr_u(:, k) = case_result.online_precorr_u;
    test.online_precorr_x(:, k) = case_result.online_precorr_x;
    test.online_precorr_time(k) = case_result.online_precorr_time;
    test.online_lift_time(k) = case_result.online_lift_time;
    test.online_precorr_matvec(k) = case_result.online_precorr_matvec;
    test.online_precorr_relres_final(k) = case_result.online_precorr_relres_final;
    test.online_final_x(:, k) = case_result.online_final_x;
    test.online_final_u(:, k) = case_result.online_final_u;
    test.online_fp_time(k) = case_result.online_fp_time;
    test.online_total_time(k) = case_result.online_total_time;
    test.online_fp_iter(k) = case_result.online_fp_iter;
    test.online_fp_relres_final(k) = case_result.online_fp_relres_final;
    test.online_fp_op_apply(k) = case_result.online_fp_op_apply;
    test.online_fp_relres_hist{k} = case_result.online_fp_relres_hist;
    test.online_fp_step_hist{k} = case_result.online_fp_step_hist;
    test.online_fp_stop_reason{k} = case_result.online_fp_stop_reason;
    test.online_combined_relres_hist{k} = case_result.online_combined_relres_hist;
    test.online_combined_transition_idx(k) = case_result.online_combined_transition_idx;

    test.helm_baseline_time(k) = case_result.helm_baseline_time;
    test.helm_baseline_iter(k) = case_result.helm_baseline_iter;
    test.helm_baseline_op_apply(k) = case_result.helm_baseline_op_apply;
    test.helm_baseline_relres_final(k) = case_result.helm_baseline_relres_final;
    test.helm_baseline_relres_hist{k} = case_result.helm_baseline_relres_hist;

    test.plain_fp_err_wave_full(k) = case_result.plain_fp_err_wave_full;
    test.plain_fp_err_wave_phys(k) = case_result.plain_fp_err_wave_phys;
    test.plain_fp_err_helm_full(k) = case_result.plain_fp_err_helm_full;
    test.plain_fp_err_helm_phys(k) = case_result.plain_fp_err_helm_phys;
    test.online_final_err_wave_full(k) = case_result.online_final_err_wave_full;
    test.online_final_err_wave_phys(k) = case_result.online_final_err_wave_phys;
    test.online_final_err_helm_full(k) = case_result.online_final_err_helm_full;
    test.online_final_err_helm_phys(k) = case_result.online_final_err_helm_phys;

    for j = 1:nlayers
        test.corr_time_layers(j, k) = case_result.corr_time_layers(j);
        test.corr_u_layers{j}(:, k) = case_result.corr_u_layers{j};
        test.corr_x_layers{j}(:, k) = case_result.corr_x_layers{j};
        test.layer_err_wave_full(j, k) = case_result.layer_err_wave_full(j);
        test.layer_err_wave_phys(j, k) = case_result.layer_err_wave_phys(j);
        test.layer_err_helm_full(j, k) = case_result.layer_err_helm_full(j);
        test.layer_err_helm_phys(j, k) = case_result.layer_err_helm_phys(j);
    end

end

function [u_curr, info] = run_helmholtz_precorrection(prob, cfg, rhs, c_right, u0, corr_rom)
    nlayers = corr_rom.num_layers;
    u_curr = u0;
    layer_u = cell(nlayers, 1);
    layer_time = zeros(nlayers, 1);
    layer_relres = zeros(nlayers, 1);
    relres_hist = [];
    iter_count = 0;
    tprec = tic;

    if nlayers > 0
        [u_curr, fgmres_info] = layered_fgmres_precorrection(prob, cfg, rhs, c_right, u0, corr_rom);
        layer_u = fgmres_info.layer_u;
        layer_time = fgmres_info.layer_time;
        layer_relres = fgmres_info.layer_relres;
        relres_hist = fgmres_info.relres_hist;
        iter_count = fgmres_info.iter_count;
    else
        Ahelm = build_helmholtz_operator(prob, c_right);
        relres_hist = norm(rhs - Ahelm * u_curr) / max(norm(rhs), 1e-16);
    end

    info = struct();
    info.total_time = toc(tprec);
    info.layer_u = layer_u;
    info.layer_time = layer_time;
    info.layer_relres = layer_relres;
    info.relres_hist = relres_hist;
    info.iter_count = iter_count;
    if nlayers == 0
        info.relres_final = relres_hist(end);
        info.matvec_count = 1;
    else
        info.relres_final = layer_relres(max(iter_count, 1));
        info.matvec_count = fgmres_info.matvec_count;
    end
end

function [u_sol, info] = run_plain_helmholtz_gmres(prob, rhs, c_right, u0, restart, maxit_outer, tol, verbose)
    if nargin < 8
        verbose = false;
    end
    Ahelm = build_helmholtz_operator(prob, c_right);
    Afun = @(x) Ahelm * x;
    nb = max(norm(rhs), 1e-16);
    tsolve = tic;
    [u_sol, flag, relres, iter, resvec] = gmres_restarted_verbose(Afun, rhs, restart, tol, maxit_outer, u0, verbose);
    total_time = toc(tsolve);

    info = struct();
    info.total_time = total_time;
    info.iter_count = iteration_count(iter, restart);
    info.relres_final = relres;
    info.relres_hist = resvec / nb;
    info.op_apply_count = info.iter_count + 1;
    info.step_hist = zeros(0, 1);
    if flag == 0
        info.stop_reason = 'residual';
    else
        info.stop_reason = 'maxit';
    end
    info.u_final = u_sol;
end

function [x_curr, info] = run_online_fixedpoint(prob, cfg, rhs, c_right, b_vec, Afun, x0)
    normb = max(norm(b_vec), 1e-16);
    x_curr = x0;
    iter_count = 0;
    op_apply_count = 0;
    relres_hist = zeros(cfg.fp_maxit + 1, 1);
    step_hist = zeros(cfg.fp_maxit, 1);
    stop_reason = 'maxit';
    tfull = tic;
    for it = 1:cfg.fp_maxit
        Ax_curr = Afun(x_curr);
        op_apply_count = op_apply_count + 1;
        r = b_vec - Ax_curr;
        relres = norm(r) / normb;
        relres_hist(it) = relres;
        if relres <= cfg.fp_tol
            stop_reason = 'residual';
            break;
        end
        iter_count = it;
        x_next = x_curr + r;
        step_hist(it) = norm(x_next - x_curr) / max(norm(x_next), 1e-16);
        x_curr = x_next;
        if ~isempty(cfg.fp_step_tol) && step_hist(it) <= cfg.fp_step_tol
            stop_reason = 'step';
            break;
        end
    end
    total_time = toc(tfull);
    r_final = b_vec - Afun(x_curr);
    op_apply_count = op_apply_count + 1;
    relres_final = norm(r_final) / normb;
    relres_hist(iter_count + 1) = relres_final;
    u_final = recover_u_phasor_timeproj(prob, x_curr, rhs, c_right);

    info = struct();
    info.iter_count = iter_count;
    info.op_apply_count = op_apply_count;
    info.relres_final = relres_final;
    info.relres_hist = relres_hist(1:iter_count+1);
    info.step_hist = step_hist(1:iter_count);
    info.u_final = u_final;
    info.total_time = total_time;
    info.fp_time = total_time;
    info.stop_reason = stop_reason;
end

function Ahelm = build_helmholtz_operator(prob, c_right)
    alpha = (1 / c_right^2) - 1;
    Ahelm = prob.A_base + alpha * prob.A_right;
end

function train_state = initialize_trajectory_aware_training_state(prob, rhs, params, u_init, u_ref)
    ns = numel(params);
    train_state = struct();
    train_state.u_init = u_init;
    train_state.beta0 = zeros(ns, 1);
    train_state.q_curr = zeros(prob.Ntot, ns);
    train_state.Q_hist = cell(ns, 1);
    train_state.H_hist = cell(ns, 1);
    train_state.eta_hist = cell(ns, 1);

    for k = 1:ns
        Ahelm = build_helmholtz_operator(prob, params(k));
        r0 = rhs - Ahelm * u_init(:, k);
        beta0 = norm(r0);
        train_state.beta0(k) = beta0;
        q1 = zeros(prob.Ntot, 1);
        eta1 = zeros(prob.Ntot, 1);
        if beta0 > 1e-15
            q1 = r0 / beta0;
            eta1 = (u_ref(:, k) - u_init(:, k)) / beta0;
        end
        train_state.q_curr(:, k) = q1;
        train_state.Q_hist{k} = q1;
        train_state.H_hist{k} = zeros(1, 0);
        train_state.eta_hist{k} = eta1;
    end
end

function [train_state, eta_curr] = advance_trajectory_aware_training_sample(prob, train_state, prev_layer, c_right, param_idx, layer_idx)
    Ahelm = build_helmholtz_operator(prob, c_right);
    q_curr = train_state.q_curr(:, param_idx);
    prev_step = layer_idx - 1;
    if train_state.beta0(param_idx) <= 1e-15 || norm(q_curr) <= 1e-15
        Q_hist = train_state.Q_hist{param_idx};
        H_hist = train_state.H_hist{param_idx};
        eta_hist = train_state.eta_hist{param_idx};
        if size(Q_hist, 2) < layer_idx
            Q_hist(:, layer_idx) = zeros(prob.Ntot, 1);
        end
        H_hist(layer_idx, prev_step) = 0;
        eta_hist(:, layer_idx) = zeros(prob.Ntot, 1);
        train_state.Q_hist{param_idx} = Q_hist;
        train_state.H_hist{param_idx} = H_hist;
        train_state.eta_hist{param_idx} = eta_hist;
        train_state.q_curr(:, param_idx) = zeros(prob.Ntot, 1);
        eta_curr = zeros(prob.Ntot, 1);
        return;
    end

    z_rom = apply_layer_inverse_to_rhs(prob, prev_layer, c_right, q_curr);
    Q_hist = train_state.Q_hist{param_idx};
    H_hist = train_state.H_hist{param_idx};
    eta_hist = train_state.eta_hist{param_idx};

    w = Ahelm * z_rom;
    for i = 1:prev_step
        H_hist(i, prev_step) = Q_hist(:, i)' * w;
        w = w - H_hist(i, prev_step) * Q_hist(:, i);
    end
    H_hist(layer_idx, prev_step) = norm(w);
    if H_hist(layer_idx, prev_step) > 1e-15
        Q_hist(:, layer_idx) = w / H_hist(layer_idx, prev_step);
        train_state.q_curr(:, param_idx) = Q_hist(:, layer_idx);
    else
        Q_hist(:, layer_idx) = zeros(prob.Ntot, 1);
        train_state.q_curr(:, param_idx) = zeros(prob.Ntot, 1);
    end

    eta_num = z_rom;
    for i = 1:prev_step
        eta_num = eta_num - H_hist(i, prev_step) * eta_hist(:, i);
    end
    if H_hist(layer_idx, prev_step) > 1e-15
        eta_curr = eta_num / H_hist(layer_idx, prev_step);
    else
        eta_curr = zeros(prob.Ntot, 1);
    end
    eta_hist(:, layer_idx) = eta_curr;

    train_state.Q_hist{param_idx} = Q_hist;
    train_state.H_hist{param_idx} = H_hist;
    train_state.eta_hist{param_idx} = eta_hist;
end

function z = apply_layer_inverse_to_rhs(prob, layer, c_right, rhs_vec)
    if isempty(layer.basis_u)
        z = zeros(size(rhs_vec));
        return;
    end
    alpha = (1 / c_right^2) - 1;
    Ar = layer.Ar_base + alpha * layer.Ar_right;
    coeff = Ar \ (layer.basis_u' * rhs_vec);
    z = layer.basis_u * coeff;
end

function [u_curr, info] = layered_fgmres_precorrection(prob, cfg, rhs, c_right, u0, corr_rom)
    nlayers = corr_rom.num_layers;
    Ahelm = build_helmholtz_operator(prob, c_right);
    nb = max(norm(rhs), 1e-16);
    r0 = rhs - Ahelm * u0;
    beta0 = norm(r0);

    info = struct();
    info.layer_u = cell(nlayers, 1);
    info.layer_time = zeros(nlayers, 1);
    info.layer_relres = zeros(nlayers, 1);
    info.relres_hist = zeros(nlayers + 1, 1);
    info.matvec_count = 1;
    info.iter_count = 0;
    info.relres_final = beta0 / nb;

    if beta0 <= cfg.helm_gmres_track_tol
        u_curr = u0;
        info.relres_hist(1) = beta0 / nb;
        info.relres_final = beta0 / nb;
        return;
    end

    Q = zeros(prob.Ntot, nlayers + 1);
    Z = zeros(prob.Ntot, nlayers);
    H = zeros(nlayers + 1, nlayers);
    Q(:, 1) = r0 / beta0;
    info.relres_hist(1) = beta0 / nb;
    u_curr = u0;

    for j = 1:nlayers
        tlayer = tic;
        z = apply_layer_inverse_to_rhs(prob, corr_rom.layers{j}, c_right, Q(:, j));
        Z(:, j) = z;
        w = Ahelm * z;
        info.matvec_count = info.matvec_count + 1;
        for i = 1:j
            H(i, j) = Q(:, i)' * w;
            w = w - H(i, j) * Q(:, i);
        end
        H(j + 1, j) = norm(w);
        if H(j + 1, j) > 1e-15
            Q(:, j + 1) = w / H(j + 1, j);
        end

        rhs_ls = zeros(j + 1, 1);
        rhs_ls(1) = beta0;
        y = H(1:j + 1, 1:j) \ rhs_ls;
        u_curr = u0 + Z(:, 1:j) * y;
        r_true = rhs - Ahelm * u_curr;
        info.matvec_count = info.matvec_count + 1;
        info.layer_relres(j) = norm(r_true) / nb;
        info.relres_hist(j + 1) = info.layer_relres(j);
        info.layer_u{j} = u_curr;
        info.layer_time(j) = toc(tlayer);
        info.iter_count = j;

        if info.layer_relres(j) <= cfg.helm_gmres_track_tol
            break;
        end
    end

    info.relres_hist = info.relres_hist(1:info.iter_count + 1);
    if info.iter_count > 0
        info.relres_final = info.layer_relres(info.iter_count);
    else
        info.relres_final = beta0 / nb;
    end
end

function [rel_full, rel_phys] = compute_solution_errors(u_approx, u_ref, prob)
    rel_full = norm(u_approx - u_ref) / max(norm(u_ref), 1e-16);
    rel_phys = cropped_relerr(u_approx, u_ref, prob);
end

function summary = summarize_correction_results(train_ref, corr_rom, test, prob, cfg)
    summary = struct();
    summary.online_completion_mode = cfg.online_completion_mode;
    summary.train_total_time = sum(train_ref.solve_time + train_ref.recovery_time);
    summary.train_mean_time = mean(train_ref.solve_time + train_ref.recovery_time);
    summary.train_mean_iter = mean(train_ref.iter);
    summary.corr_ranks = cellfun(@(layer) layer.pod.rank_used, corr_rom.layers);
    summary.corr_num_layers = corr_rom.num_layers;
    [summary.delta_c_stats, summary.adjacent_phys_stats, summary.endpoint_phys] = ...
        summarize_training_parameter_variation_from_u(train_ref.params, train_ref.u_ref_snapshots, prob);
    [summary.train_lift_vs_ref_state, summary.train_lift_vs_ref_u_component] = ...
        summarize_training_lift_error(prob, train_ref);

    summary.plain_fp_wave_full = stats3(test.plain_fp_err_wave_full);
    summary.plain_fp_wave_phys = stats3(test.plain_fp_err_wave_phys);
    summary.final_wave_full = stats3(test.online_final_err_wave_full);
    summary.final_wave_phys = stats3(test.online_final_err_wave_phys);

    if ~isempty(test.helm_u_ref)
        summary.plain_fp_helm_full = stats3(test.plain_fp_err_helm_full);
        summary.plain_fp_helm_phys = stats3(test.plain_fp_err_helm_phys);
        summary.final_helm_full = stats3(test.online_final_err_helm_full);
        summary.final_helm_phys = stats3(test.online_final_err_helm_phys);
        summary.wave_vs_helm_full = stats3(vec_solution_compare(test.wave_u_ref, test.helm_u_ref, prob, 'full'));
        summary.wave_vs_helm_phys = stats3(vec_solution_compare(test.wave_u_ref, test.helm_u_ref, prob, 'phys'));
    else
        summary.plain_fp_helm_full = [];
        summary.plain_fp_helm_phys = [];
        summary.final_helm_full = [];
        summary.final_helm_phys = [];
        summary.wave_vs_helm_full = [];
        summary.wave_vs_helm_phys = [];
    end
    summary.plain_fp_time = stats3(test.plain_fp_time);
    summary.plain_fp_iter = stats3(test.plain_fp_iter);
    summary.plain_fp_op_apply = stats3(test.plain_fp_op_apply);
    summary.plain_fp_relres = stats3(test.plain_fp_relres_final);
    summary.online_precorr_time = stats3(test.online_precorr_time);
    summary.online_lift_time = stats3(test.online_lift_time);
    summary.online_precorr_matvec = stats3(test.online_precorr_matvec);
    summary.online_precorr_relres = stats3(test.online_precorr_relres_final);
    summary.online_fp_time = stats3(test.online_fp_time);
    summary.online_total_time = stats3(test.online_total_time);
    summary.online_fp_iter = stats3(test.online_fp_iter);
    summary.online_fp_op_apply = stats3(test.online_fp_op_apply);
    summary.online_fp_relres = stats3(test.online_fp_relres_final);
    summary.helm_baseline_time = stats3(test.helm_baseline_time);
    summary.helm_baseline_iter = stats3(test.helm_baseline_iter);
    summary.helm_baseline_op_apply = stats3(test.helm_baseline_op_apply);
    summary.helm_baseline_relres = stats3(test.helm_baseline_relres_final);
    summary.train_gmres_relres_by_layer = zeros(corr_rom.num_layers, 1);
    for j = 1:corr_rom.num_layers
        summary.train_gmres_relres_by_layer(j) = mean(corr_rom.layers{j}.train_fgmres_relres);
    end
    if isempty(summary.plain_fp_time)
        summary.time_speedup_online_vs_plain_mean = NaN;
    else
        summary.time_speedup_online_vs_plain_mean = summary.plain_fp_time(2) / max(summary.online_total_time(2), 1e-16);
    end
    summary.time_speedup_online_vs_helm100_mean = summary.helm_baseline_time(2) / max(summary.online_total_time(2), 1e-16);
    if strcmp(cfg.online_completion_mode, 'helmholtz')
        summary.iter_reduction_online_vs_plain_mean = summary.helm_baseline_iter(2) - summary.online_fp_iter(2);
        summary.iter_ratio_online_vs_plain_mean = summary.online_fp_iter(2) / max(summary.helm_baseline_iter(2), 1e-16);
        summary.op_apply_ratio_online_vs_plain_mean = summary.online_fp_op_apply(2) / max(summary.helm_baseline_op_apply(2), 1e-16);
    else
        summary.iter_reduction_online_vs_plain_mean = summary.plain_fp_iter(2) - summary.online_fp_iter(2);
        summary.iter_ratio_online_vs_plain_mean = summary.online_fp_iter(2) / max(summary.plain_fp_iter(2), 1e-16);
        summary.op_apply_ratio_online_vs_plain_mean = summary.online_fp_op_apply(2) / max(summary.plain_fp_op_apply(2), 1e-16);
    end

    summary.primary_target = upper(cfg.rom_check_target);
    if strcmp(cfg.rom_check_target, 'helm') && ~isempty(test.helm_u_ref)
        final_primary_curve = test.online_final_err_helm_phys;
    else
        final_primary_curve = test.online_final_err_wave_phys;
    end
    [~, summary.representative_test_index] = max(final_primary_curve);
    summary.representative_c = test.params(summary.representative_test_index);
end

function print_correction_summary(summary)
    if strcmp(summary.online_completion_mode, 'helmholtz')
        completion_label = 'Helmholtz completion';
        compare_label = 'helm100';
    else
        completion_label = 'WaveHoltz completion';
        compare_label = 'plain';
    end
    fprintf('\n=== Summary ===\n');
    fprintf('train total / mean time   : %.3fs / %.3fs\n', summary.train_total_time, summary.train_mean_time);
    fprintf('mean train iters          : %.2f\n', summary.train_mean_iter);
    fprintf('layered Helmholtz ROM steps: %d\n', summary.corr_num_layers);
    if ~isempty(summary.corr_ranks)
        fprintf('layered ROM ranks         : %s\n', mat2str(summary.corr_ranks));
    end
    if ~isempty(summary.train_gmres_relres_by_layer)
        fprintf('train layered relres mean : %s\n', mat2str(summary.train_gmres_relres_by_layer, 4));
    end
    if ~isempty(summary.delta_c_stats)
        fprintf('adjacent train delta c    : %.3e / %.3e / %.3e\n', summary.delta_c_stats(1), summary.delta_c_stats(2), summary.delta_c_stats(3));
        fprintf('adjacent train phys diff  : %.3e / %.3e / %.3e\n', summary.adjacent_phys_stats(1), summary.adjacent_phys_stats(2), summary.adjacent_phys_stats(3));
        fprintf('endpoint train phys diff  : %.3e\n', summary.endpoint_phys);
    end
    if ~isempty(summary.train_lift_vs_ref_state)
        fprintf('train lift(u_ref)->x vs x_ref state relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.train_lift_vs_ref_state(1), summary.train_lift_vs_ref_state(2), summary.train_lift_vs_ref_state(3));
        fprintf('train lift u-block vs Re(u_ref) relerr min/mean/max   = %.3e / %.3e / %.3e\n', ...
            summary.train_lift_vs_ref_u_component(1), summary.train_lift_vs_ref_u_component(2), summary.train_lift_vs_ref_u_component(3));
    end
    if ~isempty(summary.plain_fp_wave_full)
        fprintf('plain FP-vs-WH full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.plain_fp_wave_full(1), summary.plain_fp_wave_full(2), summary.plain_fp_wave_full(3));
        fprintf('plain FP-vs-WH phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.plain_fp_wave_phys(1), summary.plain_fp_wave_phys(2), summary.plain_fp_wave_phys(3));
    end
    fprintf('final layered-ROM-vs-WH full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.final_wave_full(1), summary.final_wave_full(2), summary.final_wave_full(3));
    fprintf('final layered-ROM-vs-WH phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.final_wave_phys(1), summary.final_wave_phys(2), summary.final_wave_phys(3));
    if ~isempty(summary.final_helm_full)
        if ~isempty(summary.plain_fp_helm_full)
            fprintf('plain FP-vs-Helm full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
                summary.plain_fp_helm_full(1), summary.plain_fp_helm_full(2), summary.plain_fp_helm_full(3));
            fprintf('plain FP-vs-Helm phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
                summary.plain_fp_helm_phys(1), summary.plain_fp_helm_phys(2), summary.plain_fp_helm_phys(3));
        end
        fprintf('final layered-ROM-vs-Helm full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.final_helm_full(1), summary.final_helm_full(2), summary.final_helm_full(3));
        fprintf('final layered-ROM-vs-Helm phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.final_helm_phys(1), summary.final_helm_phys(2), summary.final_helm_phys(3));
        fprintf('WH-vs-Helm full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.wave_vs_helm_full(1), summary.wave_vs_helm_full(2), summary.wave_vs_helm_full(3));
        fprintf('WH-vs-Helm phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.wave_vs_helm_phys(1), summary.wave_vs_helm_phys(2), summary.wave_vs_helm_phys(3));
    end
    if ~isempty(summary.plain_fp_time)
        fprintf('plain WaveHoltz time min/mean/max: %.3fs / %.3fs / %.3fs\n', ...
            summary.plain_fp_time(1), summary.plain_fp_time(2), summary.plain_fp_time(3));
        fprintf('plain WaveHoltz iter min/mean/max: %.0f / %.2f / %.0f\n', ...
            summary.plain_fp_iter(1), summary.plain_fp_iter(2), summary.plain_fp_iter(3));
        fprintf('plain WaveHoltz apply min/mean/max: %.0f / %.2f / %.0f\n', ...
            summary.plain_fp_op_apply(1), summary.plain_fp_op_apply(2), summary.plain_fp_op_apply(3));
        fprintf('plain WaveHoltz relres min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.plain_fp_relres(1), summary.plain_fp_relres(2), summary.plain_fp_relres(3));
    end
    fprintf('online layered-ROM time min/mean/max: %.3fs / %.3fs / %.3fs\n', ...
        summary.online_precorr_time(1), summary.online_precorr_time(2), summary.online_precorr_time(3));
    fprintf('online lift time min/mean/max: %.3fs / %.3fs / %.3fs\n', ...
        summary.online_lift_time(1), summary.online_lift_time(2), summary.online_lift_time(3));
    fprintf('online layered-ROM Helm matvec min/mean/max: %.0f / %.2f / %.0f\n', ...
        summary.online_precorr_matvec(1), summary.online_precorr_matvec(2), summary.online_precorr_matvec(3));
    fprintf('online layered-ROM Helm relres min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.online_precorr_relres(1), summary.online_precorr_relres(2), summary.online_precorr_relres(3));
    fprintf('online total time min/mean/max: %.3fs / %.3fs / %.3fs\n', ...
        summary.online_total_time(1), summary.online_total_time(2), summary.online_total_time(3));
    fprintf('%s iter min/mean/max: %.0f / %.2f / %.0f\n', completion_label, ...
        summary.online_fp_iter(1), summary.online_fp_iter(2), summary.online_fp_iter(3));
    fprintf('%s apply min/mean/max: %.0f / %.2f / %.0f\n', completion_label, ...
        summary.online_fp_op_apply(1), summary.online_fp_op_apply(2), summary.online_fp_op_apply(3));
    fprintf('%s relres min/mean/max = %.3e / %.3e / %.3e\n', completion_label, ...
        summary.online_fp_relres(1), summary.online_fp_relres(2), summary.online_fp_relres(3));
    fprintf('plain Helm(100) time min/mean/max: %.3fs / %.3fs / %.3fs\n', ...
        summary.helm_baseline_time(1), summary.helm_baseline_time(2), summary.helm_baseline_time(3));
    fprintf('plain Helm(100) iter min/mean/max: %.0f / %.2f / %.0f\n', ...
        summary.helm_baseline_iter(1), summary.helm_baseline_iter(2), summary.helm_baseline_iter(3));
    fprintf('plain Helm(100) relres min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.helm_baseline_relres(1), summary.helm_baseline_relres(2), summary.helm_baseline_relres(3));
    if isfinite(summary.time_speedup_online_vs_plain_mean)
        fprintf('mean time speedup WaveHoltz/online = %.3fx\n', summary.time_speedup_online_vs_plain_mean);
    end
    fprintf('mean time speedup helm100/online = %.3fx\n', summary.time_speedup_online_vs_helm100_mean);
    fprintf('mean iter change %s-online = %.2f (ratio %.3f)\n', compare_label, ...
        summary.iter_reduction_online_vs_plain_mean, summary.iter_ratio_online_vs_plain_mean);
    fprintf('mean apply ratio online/%s = %.3f\n', compare_label, summary.op_apply_ratio_online_vs_plain_mean);
    fprintf('ROM check target          : %s\n', summary.primary_target);
    fprintf('representative c_right    : %.4f\n', summary.representative_c);
end

function make_correction_plots(cfg, prob, rhs, corr_rom, test, summary)
    figure('Name', 'Correction-ROM Errors', 'Position', [100 100 1200 420]);
    ax1 = subplot(1, 3, 1);
    if corr_rom.num_layers > 0
        corr_sv = clamp_positive_curve(corr_rom.layers{1}.pod.singular_values, 1e-16);
        plot(ax1, 1:numel(corr_sv), corr_sv, 'o-', 'LineWidth', 1.4);
    end
    set(ax1, 'YScale', 'log');
    grid on;
    xlabel('mode index');
    ylabel('singular value');
    title('Layer 1 singular values');

    ax2 = subplot(1, 3, 2);
    hold(ax2, 'on');
    for j = 1:corr_rom.num_layers
        corr_sv = clamp_positive_curve(corr_rom.layers{j}.pod.singular_values, 1e-16);
        plot(ax2, 1:numel(corr_sv), corr_sv, 'o-', 'LineWidth', 1.2, 'DisplayName', sprintf('layer %d', j));
    end
    hold(ax2, 'off');
    set(ax2, 'YScale', 'log');
    grid on;
    xlabel('mode index');
    ylabel('singular value');
    title('Layered Helmholtz ROM singular values');
    if corr_rom.num_layers > 0
        legend('Location', 'best');
    end

    ax3 = subplot(1, 3, 3);
    hold(ax3, 'on');
    rep_idx = summary.representative_test_index;
    plot_floor = 1e-16;

    if ~isempty(test.online_fp_step_hist{rep_idx})
        plot(1:numel(test.online_fp_step_hist{rep_idx}), max(test.online_fp_step_hist{rep_idx}, plot_floor), ...
            '--', 'Color', [1 0 0], 'LineWidth', 1.0, 'DisplayName', 'corr step');
    end
    online_combined_hist = test.online_combined_relres_hist{rep_idx};
    plot(1:numel(online_combined_hist), max(online_combined_hist, plot_floor), ...
        '-o', 'Color', [1 0 0], 'LineWidth', 1.2, 'MarkerSize', 7, 'DisplayName', 'online residual');
    if ~isempty(test.helm_baseline_relres_hist{rep_idx})
        plot(1:numel(test.helm_baseline_relres_hist{rep_idx}), max(test.helm_baseline_relres_hist{rep_idx}, plot_floor), ...
            '-o', 'Color', [0.65 0 0.75], 'LineWidth', 1.2, 'MarkerSize', 5, 'DisplayName', 'plain Helm(100)');
    end
    transition_idx = test.online_combined_transition_idx(rep_idx);
    if transition_idx >= 1 && transition_idx <= numel(online_combined_hist)
        plot(transition_idx, max(online_combined_hist(transition_idx), plot_floor), ...
            's', 'Color', [0.85 0.1 0.1], 'MarkerFaceColor', 'w', 'MarkerSize', 8, 'DisplayName', 'lift transition');
    end
    if ~isempty(test.plain_fp_step_hist{rep_idx})
        plot(1:numel(test.plain_fp_step_hist{rep_idx}), max(test.plain_fp_step_hist{rep_idx}, plot_floor), ...
            'k--', 'LineWidth', 1.4, 'DisplayName', 'plain WH step');
    end
    if ~isempty(test.plain_fp_relres_hist{rep_idx})
        plot(1:numel(test.plain_fp_relres_hist{rep_idx}), max(test.plain_fp_relres_hist{rep_idx}, plot_floor), ...
            'k-o', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'w', 'DisplayName', 'plain WH residual');
    end
    set(ax3, 'YScale', 'log');
    ylim(ax3, [plot_floor, 2]);
    hold(ax3, 'off');
    grid(ax3, 'on');
    xlabel('iteration');
    ylabel('relative residual / step (log scale)');
    title(sprintf('Convergence curves at c=%.4f', summary.representative_c));
    legend('Location', 'best');

    if strcmp(cfg.rom_check_target, 'helm') && ~isempty(test.helm_u_ref)
        target_u = test.helm_u_ref(:, rep_idx);
        target_label = 'u_{Helm}';
    else
        target_u = test.wave_u_ref(:, rep_idx);
        target_label = 'u_{WH}';
    end
    corr_u = test.online_final_u(:, rep_idx);

    target_grid = embed_interior_field(target_u, prob.nx, prob.ny);
    corr_grid = embed_interior_field(corr_u, prob.nx, prob.ny);
    err_grid = abs(corr_grid - target_grid);
    rhs_grid = reshape(rhs, prob.nx, prob.ny);

    plot_solution_components(prob, sprintf('Representative Solutions (Real), c=%.4f', summary.representative_c), @real, ...
        target_grid, corr_grid, err_grid, rhs_grid, target_label);
    plot_solution_components(prob, sprintf('Representative Solutions (Imag), c=%.4f', summary.representative_c), @imag, ...
        target_grid, corr_grid, err_grid, rhs_grid, target_label);
end

function combined = stitch_online_relres_hist(helm_hist, wave_hist)
    helm_hist = helm_hist(:);
    wave_hist = wave_hist(:);
    if isempty(helm_hist)
        combined = wave_hist;
        return;
    end
    if isempty(wave_hist)
        combined = helm_hist;
        return;
    end
    combined = [helm_hist; wave_hist(2:end)];
end

function plot_solution_components(prob, fig_name, component_fun, target_grid, corr_grid, err_grid, rhs_grid, target_label)
    figure('Name', fig_name, 'Position', [100 560 1320 460]);
    subplot(1, 4, 1);
    imagesc(prob.x_phys, prob.y_phys, component_fun(target_grid(prob.I + 1, prob.J + 1)).');
    set(gca, 'YDir', 'normal'); axis equal tight; colorbar;
    xlabel('x'); ylabel('y'); title(sprintf('%s(%s)', func2str(component_fun), target_label));

    subplot(1, 4, 2);
    imagesc(prob.x_phys, prob.y_phys, component_fun(corr_grid(prob.I + 1, prob.J + 1)).');
    set(gca, 'YDir', 'normal'); axis equal tight; colorbar;
    xlabel('x'); ylabel('y'); title(sprintf('%s(u_{corr})', func2str(component_fun)));

    subplot(1, 4, 3);
    imagesc(prob.x_phys, prob.y_phys, err_grid(prob.I + 1, prob.J + 1).');
    set(gca, 'YDir', 'normal'); axis equal tight; colorbar;
    xlabel('x'); ylabel('y'); title(sprintf('|u_{corr} - %s|', target_label));

    subplot(1, 4, 4);
    imagesc(prob.x_int, prob.y_int, abs(rhs_grid.'));
    set(gca, 'YDir', 'normal'); axis equal tight; colorbar;
    xlabel('x'); ylabel('y'); title(sprintf('Left %s source', 'input'));
end

function pod = build_pod_basis_custom(snapshot_matrix, energy_target, singular_value_tol, min_rank, max_rank)
    [U, S, V] = svd(snapshot_matrix, 'econ');
    singular_values = diag(S);
    if isempty(singular_value_tol)
        zero_tol = 1e-15;
    else
        zero_tol = max([1e-15, abs(singular_value_tol)]);
    end
    if isempty(singular_values) || norm(snapshot_matrix, 'fro') <= zero_tol
        pod = struct();
        pod.left_modes = U;
        pod.right_modes = V;
        pod.singular_values = singular_values;
        pod.energy_ratio = zeros(size(singular_values));
        pod.cum_energy = zeros(size(singular_values));
        pod.rank_sigma_tol = 0;
        pod.rank_used = 0;
        pod.basis = zeros(size(snapshot_matrix, 1), 0);
        pod.r90 = 0;
        pod.r95 = 0;
        pod.r99 = 0;
        pod.singular_value_tol = singular_value_tol;
        return;
    end
    energy_raw = singular_values .^ 2;
    if sum(energy_raw) > 0
        energy_ratio = energy_raw / sum(energy_raw);
    else
        energy_ratio = zeros(size(energy_raw));
    end
    cum_energy = cumsum(energy_ratio);

    rank_energy = find(cum_energy >= energy_target, 1, 'first');
    if isempty(rank_energy)
        rank_energy = numel(singular_values);
    end
    if isempty(singular_value_tol)
        rank_sigma = rank_energy;
    else
        rank_sigma = find(singular_values >= singular_value_tol, 1, 'last');
        if isempty(rank_sigma)
            rank_sigma = 1;
        end
    end
    rank_use = max(rank_sigma, min_rank);
    if ~isempty(max_rank)
        rank_use = min(rank_use, max_rank);
    end
    rank_use = min(rank_use, size(U, 2));

    pod = struct();
    pod.left_modes = U;
    pod.right_modes = V;
    pod.singular_values = singular_values;
    pod.energy_ratio = energy_ratio;
    pod.cum_energy = cum_energy;
    pod.rank_sigma_tol = rank_sigma;
    pod.rank_used = rank_use;
    pod.basis = U(:, 1:rank_use);
    pod.r90 = first_mode_above(cum_energy, 0.90);
    pod.r95 = first_mode_above(cum_energy, 0.95);
    pod.r99 = first_mode_above(cum_energy, 0.99);
    pod.singular_value_tol = singular_value_tol;
end

function vals = vec_solution_compare(Ua, Ub, prob, part)
    ns = size(Ua, 2);
    vals = zeros(ns, 1);
    for k = 1:ns
        if strcmp(part, 'phys')
            vals(k) = cropped_relerr(Ua(:, k), Ub(:, k), prob);
        else
            vals(k) = norm(Ua(:, k) - Ub(:, k)) / max(norm(Ub(:, k)), 1e-16);
        end
    end
end

function s = stats3(v)
    if isempty(v)
        s = [];
    else
        v = v(isfinite(v));
        if isempty(v)
            s = [];
        else
            s = [min(v(:)), mean(v(:)), max(v(:))];
        end
    end
end

function [delta_c_stats, adjacent_phys_stats, span_phys] = summarize_training_parameter_variation_from_u(params, u_snapshots, prob)
    if numel(params) < 2 || size(u_snapshots, 2) < 2
        delta_c_stats = [];
        adjacent_phys_stats = [];
        span_phys = [];
        return;
    end
    delta_c = diff(params(:));
    adjacent_phys = zeros(numel(delta_c), 1);
    for k = 1:numel(delta_c)
        adjacent_phys(k) = cropped_relerr(u_snapshots(:, k + 1), u_snapshots(:, k), prob);
    end
    delta_c_stats = [min(delta_c), mean(delta_c), max(delta_c)];
    adjacent_phys_stats = [min(adjacent_phys), mean(adjacent_phys), max(adjacent_phys)];
    span_phys = cropped_relerr(u_snapshots(:, end), u_snapshots(:, 1), prob);
end

function values = expand_layer_parameter(layer_values, fallback_value, nlayers, field_name)
    if isempty(layer_values)
        values = fallback_value * ones(nlayers, 1);
        return;
    end

    values = layer_values(:);
    if numel(values) ~= nlayers
        error('%s must have exactly corr_num_layers=%d entries.', field_name, nlayers);
    end
end

function [state_stats, ublock_stats] = summarize_training_lift_error(prob, train_ref)
    ns = numel(train_ref.params);
    if ns == 0
        state_stats = [];
        ublock_stats = [];
        return;
    end

    state_err = zeros(ns, 1);
    ublock_err = zeros(ns, 1);
    for k = 1:ns
        x_lift = lift_frequency_solution_to_real_state(prob, train_ref.u_ref_snapshots(:, k), train_ref.params(k));
        x_ref = train_ref.x_ref_snapshots(:, k);
        state_err(k) = norm(x_lift - x_ref) / max(norm(x_ref), 1e-16);
        ublock_err(k) = norm(x_lift(1:prob.nu) - real(train_ref.u_ref_snapshots(:, k))) / ...
            max(norm(real(train_ref.u_ref_snapshots(:, k))), 1e-16);
    end

    state_stats = stats3(state_err);
    ublock_stats = stats3(ublock_err);
end

function W = solve_wave_one_period_piecewise(prob, x0, rhs, c_right)
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

function c_sq = build_piecewise_c_squared(prob, c_right)
    c_sq = ones(prob.nx, prob.ny);
    c_sq(prob.chi_right_grid) = c_right ^ 2;
end

function Pi = apply_pi_piecewise(prob, x0, rhs, source_on, c_right, shift)
    if source_on
        rhs_use = rhs;
    else
        rhs_use = zeros(prob.nu, 1);
    end
    W = solve_wave_one_period_piecewise(prob, x0, rhs_use, c_right);
    t_grid = (0:prob.nt).' * prob.dt;
    weights = cos(prob.omega_ts * t_grid) - shift;
    Pi = (2.0 / max(prob.T_ts, 1e-16)) * trapz(t_grid, bsxfun(@times, weights, W), 1).';
end

function ridx = first_mode_above(cum_energy, threshold)
    if isempty(cum_energy)
        ridx = 0;
        return;
    end
    ridx = find(cum_energy >= threshold, 1, 'first');
    if isempty(ridx)
        ridx = numel(cum_energy);
    end
end

function vals = clamp_positive_curve(vals, floor_val)
    if isempty(vals)
        vals = floor_val;
        return;
    end
    vals = vals(:);
    vals(abs(vals) < floor_val) = floor_val;
end

function relerr = cropped_relerr(u_rom, u_fom, prob)
    rom_grid = embed_interior_field(u_rom, prob.nx, prob.ny);
    fom_grid = embed_interior_field(u_fom, prob.nx, prob.ny);
    rom_phys = rom_grid(prob.I + 1, prob.J + 1);
    fom_phys = fom_grid(prob.I + 1, prob.J + 1);
    relerr = norm(rom_phys(:) - fom_phys(:)) / max(norm(fom_phys(:)), 1e-16);
end

function field = embed_interior_field(u_int, nx, ny)
    field = zeros(nx + 2, ny + 2);
    field(2:end-1, 2:end-1) = reshape(u_int, nx, ny);
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

function [x, flag, relres, iter, resvec] = gmres_restarted_verbose(Afun, b, restart, tol, maxit_outer, x0, verbose)
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

        for j = 1:restart
            inner_last = j;
            w = Afun(V(:, j));
            for i = 1:j
                H(i, j) = V(:, i)' * w;
                w = w - H(i, j) * V(:, i);
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
                y = H(1:j, 1:j) \ g(1:j);
                x = x + V(:, 1:j) * y;
                r = b - Afun(x);
                beta = norm(r);
                flag = 0;
                relres = beta / nb;
                iter = [outer, j];
                return;
            end
        end

        y = H(1:inner_last, 1:inner_last) \ g(1:inner_last);
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

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
