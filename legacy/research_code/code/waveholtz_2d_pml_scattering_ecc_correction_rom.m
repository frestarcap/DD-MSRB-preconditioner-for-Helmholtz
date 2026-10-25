function results = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg)
    % ============================================================
    %  Layered correction-ROM for log(a/b)-parametric scattering.
    %
    %  Workflow aligned with helm_waveholtz_correction_rom:
    %    1) Offline stage 1: full-order reference solves (frequency domain)
    %    2) Offline stage 2: direct ROM init space
    %    3) Offline stage 3: trajectory-aware layered correction ROM
    %    4) Online: layered pre-correction + completion switch
    %       completion mode = helmholtz | waveholtz | shifted_laplacian
    %
    %  Key difference from truncation-based ROM:
    %    - POD bases are built in full numerical rank (no energy truncation,
    %      no user singular-value truncation), up to numerical rank only.
    % ============================================================

    run_clock = tic;
    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);

    setup_clock = tic;
    domain = build_domain(cfg);
    [x, y, X] = build_coordinates(domain);

    dt0 = cfg.CFL * min(domain.hx, domain.hy) / cfg.c;
    [omega_ts, c_w, nt, dt] = match_omega_and_period(cfg.omega, dt0);
    T_ts = nt * dt;

    pml_helm = build_pml_outer(domain.nx, domain.ny, cfg.omega, cfg.npml, cfg.sigma_max, cfg.ispml);
    pml_wave = pml_helm;
    pml_wave.sigma_x = pml_helm.sigma_x / c_w;
    pml_wave.sigma_y = pml_helm.sigma_y / c_w;
    pml_wave = prepare_wave_pml_2d(pml_wave, domain.nx, domain.ny, dt);

    inc = build_incident_wave(X, cfg.omega, cfg.c);

    prob = struct();
    prob.domain = domain;
    prob.nx = domain.nx;
    prob.ny = domain.ny;
    prob.Ntot = domain.nu;
    prob.x = x;
    prob.y = y;
    prob.x_int = x(2:end-1);
    prob.y_int = y(2:end-1);
    prob.I = (cfg.npml + 1):(cfg.npml + domain.nx_phys);
    prob.J = (cfg.npml + 1):(cfg.npml + domain.ny_phys);
    prob.x_phys = x(prob.I + 1);
    prob.y_phys = y(prob.J + 1);
    prob.X = X;
    prob.pml_helm = pml_helm;
    prob.pml_wave = pml_wave;
    prob.inc = inc;
    prob.omega_ts = omega_ts;
    prob.c_w = c_w;
    prob.nt = nt;
    prob.dt = dt;
    prob.T_ts = T_ts;
    prob.stream_wave_projection = cfg.stream_wave_projection;

    cfg = adjust_train_ratios_for_supported_geometry(cfg, prob);
    if cfg.ratio_test_auto_generated
        cfg.ratio_test = generate_ratio_test_params(cfg.ratio_train, cfg.ratio_test_strategy, ...
            cfg.ratio_test_rng_seed, cfg.ratio_test_log_offset);
    end
    cfg = adjust_test_ratios_for_supported_geometry(cfg, prob);
    setup_time = toc(setup_clock);

    if isfield(cfg,'deployment_file') && ~isempty(cfg.deployment_file)
        saved=load(cfg.deployment_file,'model'); model=saved.model; clear saved;
        results=struct('true_residual',[],'completion_iterations',[]);
        for ratio=cfg.ratio_test(:).'
            sys=build_scattering_system_for_ratio(ratio,cfg,prob);
            [u,precorr]=layered_fgmres_precorrection(sys,cfg,zeros(size(sys.b)),model);
            if isfield(cfg,'deployment_audit') && cfg.deployment_audit
                results.precorr_u=u;
                results.precorr=precorr;
            end
            [u,completion]=run_baseline_completion_ecc(prob,cfg,sys,u);
            rho=norm(sys.b-sys.A*u)/norm(sys.b);
            assert(rho<=1e-8,'Deployment true residual failed.');
            results.true_residual(end+1)=rho;
            results.completion_iterations(end+1)=completion.iter_count;
            if isfield(cfg,'deployment_audit') && cfg.deployment_audit
                assert(isscalar(cfg.ratio_test),'Audit one query at a time.');
                results.final_u=u;
                results.completion=completion;
                results.ratio=ratio;
            end
        end
        return;
    end

    if cfg.setup_only
        t_setup = tic;
        sys = build_scattering_system_for_ratio(cfg.ratio_test(1), cfg, prob);
        results = struct('cfg', cfg, 'problem', rmfield(prob, {'X'}), ...
            'system', sys, 'assembly_time', toc(t_setup));
        results.timing = struct('setup', setup_time, 'offline', 0, ...
            'online', 0, 'total', toc(run_clock));
        return;
    end

    fprintf('=== waveholtz_2d_pml_scattering_ecc_correction_rom ===\n');
    fprintf('Nphys=%d, npml=%d, total interior=%d x %d\n', cfg.Nphys, cfg.npml, domain.nx, domain.ny);
    fprintf('omega(helm)=%.6f, omega(ts)=%.6f, dt=%.3e, nt=%d\n', cfg.omega, omega_ts, dt, nt);
    fprintf('ratio_train(a/b)=%s\n', mat2str(cfg.ratio_train, 4));
    fprintf('ratio_test (a/b)=%s\n', mat2str(cfg.ratio_test, 4));
    fprintf('ROM all-rank numeric rtol = %.1e\n', cfg.pod_rank_rtol);
    fprintf('use direct ROM init = %d\n', cfg.use_direct_rom_init);
    fprintf('layered Helmholtz FGMRES ROM steps = %d\n', cfg.corr_num_layers);
    fprintf('normalize correction snapshots = %d\n', cfg.corr_normalize_snapshots);
    fprintf('reference mode = %s\n', cfg.reference_mode);
    fprintf('online completion mode = %s\n', cfg.online_completion_mode);
    if strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        fprintf('shifted-Laplacian beta = %.3g, restart=%d, maxit=%d, tol=%.1e\n', ...
            cfg.shifted_laplacian_beta, cfg.shifted_laplacian_restart, ...
            cfg.shifted_laplacian_maxit, cfg.shifted_laplacian_tol);
    end
    fprintf('WaveHoltz lift mode = %s\n', cfg.wh_lift_mode);
    fprintf('run completion baselines = %d\n', cfg.run_wave_baselines);
    fprintf('Helmholtz GMRES track tol = %.1e\n', cfg.helm_gmres_track_tol);
    fprintf('direct basis style = %s, corr basis style = %s\n', cfg.direct_basis_style, cfg.corr_basis_style);
    if strcmp(cfg.direct_basis_style, 'block3')
        fprintf('direct block3 trunc rtol[near far pml]=[%g %g %g], cap=[%g %g %g]\n', ...
            cfg.direct_block3_rtol_near, cfg.direct_block3_rtol_far, cfg.direct_block3_rtol_pml, ...
            cfg.direct_block3_rank_cap_near, cfg.direct_block3_rank_cap_far, cfg.direct_block3_rank_cap_pml);
    end
    fprintf('corr   truncation rule = %s, layer tolerances=%s\n', ...
        cfg.corr_pod_truncation_rule, mat2str(cfg.corr_rank_rtols, 4));
    if strcmp(cfg.corr_basis_style, 'block3')
        fprintf('corr   block3 trunc rtol near=%s, far=%s, pml=%s\n', ...
            mat2str(cfg.corr_block3_rtol_near, 4), ...
            mat2str(cfg.corr_block3_rtol_far, 4), ...
            mat2str(cfg.corr_block3_rtol_pml, 4));
        fprintf('corr   block3 tail tol near=%s, far=%s, pml=%s, cap=[%g %g %g]\n', ...
            mat2str(cfg.corr_block3_tail_tol_near, 4), ...
            mat2str(cfg.corr_block3_tail_tol_far, 4), ...
            mat2str(cfg.corr_block3_tail_tol_pml, 4), ...
            cfg.corr_block3_rank_cap_near, cfg.corr_block3_rank_cap_far, cfg.corr_block3_rank_cap_pml);
        if ~isempty(cfg.corr_block3_global_tail_tols)
            fprintf('corr   block3 global tail budgets=%s, strategy=%s\n', ...
                mat2str(cfg.corr_block3_global_tail_tols, 4), cfg.corr_block3_global_tail_strategy);
        end
    end

    offline_clock = tic;
    fprintf('\n=== Offline stage 1: %s reference solves ===\n', reference_mode_label_ecc(cfg.reference_mode));
    train_ref = solve_reference_set(prob, cfg, cfg.ratio_train, 'train');
    block3_partition = [];
    if strcmp(cfg.direct_basis_style, 'block3') || strcmp(cfg.corr_basis_style, 'block3')
        block3_partition = build_scattering_block3_partition(prob, cfg, train_ref);
    end

    fprintf('\n=== Offline stage 2: direct ROM init space ===\n');
    direct_rom = build_direct_rom(prob, cfg, train_ref, block3_partition);
    print_direct_rom_summary(direct_rom);

    fprintf('\n=== Offline stage 3: layered correction ROM ===\n');
    corr_rom = train_correction_rom_layers(prob, cfg, train_ref, direct_rom, block3_partition);
    print_correction_rom_summary(corr_rom);
    offline_time = toc(offline_clock);

    fprintf('\n=== Online evaluation on unseen shape ratios ===\n');
    online_clock = tic;
    test = evaluate_correction_rom_online(prob, cfg, train_ref, direct_rom, corr_rom);
    online_time = toc(online_clock);

    summary = summarize_correction_results(prob, train_ref, direct_rom, corr_rom, test, cfg);
    print_correction_summary(summary);

    if cfg.save_benchmark
        save_benchmark_summary_ecc(cfg, summary);
    end

    if cfg.do_plots
        make_correction_plots(cfg, prob, direct_rom, corr_rom, test, summary);
    end

    results = struct();
    results.cfg = cfg;
    results.problem = rmfield(prob, {'X'});
    results.train_reference = train_ref;
    results.direct_rom = direct_rom;
    results.correction_rom = corr_rom;
    results.test = test;
    results.summary = summary;
    results.timing = struct('setup', setup_time, 'offline', offline_time, ...
        'online', online_time, 'total', toc(run_clock));
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'L', 1.0);
    cfg = set_default(cfg, 'c', 1.0);
    cfg = set_default(cfg, 'omega', 18.0);

    cfg = set_default(cfg, 'Nphys', 85);
    cfg = set_default(cfg, 'ispml', 1);
    cfg = set_default(cfg, 'npml', 30);
    cfg = set_default(cfg, 'sigma_max', 80);

    cfg = set_default(cfg, 'center', [0.5, 0.5]);
    cfg = set_default(cfg, 'radius_ref', 0.12);
    cfg = set_default(cfg, 'eta_boundary_min', 0.05);

    cfg = set_default(cfg, 'ratio_min', 9/10);
    cfg = set_default(cfg, 'ratio_max', 10/9);
    cfg = set_default(cfg, 'ratio_ntrain', 12);
    cfg = set_default(cfg, 'ratio_train', []);
    cfg = set_default(cfg, 'ratio_test', []);
    cfg = set_default(cfg, 'ratio_test_strategy', 'random_interval');
    if ~isfield(cfg, 'ratio_test_rng_seed')
        cfg.ratio_test_rng_seed = 1;
    end
    cfg = set_default(cfg, 'ratio_test_log_offset', 1/3);

    cfg = set_default(cfg, 'use_direct_rom_init', false);
    cfg = set_default(cfg, 'pod_rank_rtol', 1e-13);
    cfg = set_default(cfg, 'direct_basis_style', 'block3'); % global | block3

    cfg = normalize_ecc_tolerance_aliases(cfg);
    cfg = set_default(cfg, 'corr_num_layers', 3);
    cfg = set_default(cfg, 'corr_rank_rtol', []);
    cfg = set_default(cfg, 'corr_rank_rtols', [1e-4, 1e-3, 1e-2]);
    cfg = set_default(cfg, 'corr_pod_truncation_rule', 'relative_frobenius_tail');
    cfg = set_default(cfg, 'corr_basis_style', 'block3'); % global | block3
    cfg = set_default(cfg, 'corr_normalize_snapshots', false);
    cfg = set_default(cfg, 'block3_center', cfg.center);
    cfg = set_default(cfg, 'block3_r_near', 2.0 * cfg.radius_ref);
    cfg = set_default(cfg, 'block3_use_active_union', true);
    % Blockwise truncation.  For correction ROMs, use the same tail tolerance
    % strategy in all blocks unless the caller explicitly imposes a memory cap.
    cfg = set_default(cfg, 'direct_block3_rtol_near', cfg.pod_rank_rtol);
    cfg = set_default(cfg, 'direct_block3_rtol_far', 1e-2);
    cfg = set_default(cfg, 'direct_block3_rtol_pml', 1e-2);
    cfg = set_default(cfg, 'direct_block3_rank_cap_near', Inf);
    cfg = set_default(cfg, 'direct_block3_rank_cap_far', Inf);
    cfg = set_default(cfg, 'direct_block3_rank_cap_pml', Inf);
    cfg = set_default(cfg, 'corr_block3_rtol_near', []);
    cfg = set_default(cfg, 'corr_block3_rtol_far', []);
    cfg = set_default(cfg, 'corr_block3_rtol_pml', []);
    cfg = set_default(cfg, 'corr_block3_tail_tol_near', []);
    cfg = set_default(cfg, 'corr_block3_tail_tol_far', []);
    cfg = set_default(cfg, 'corr_block3_tail_tol_pml', []);
    cfg = set_default(cfg, 'corr_block3_global_tail_tols', []);
    cfg = set_default(cfg, 'corr_block3_global_tail_strategy', 'marginal_storage');
    cfg = set_default(cfg, 'corr_block3_rank_cap_near', Inf);
    cfg = set_default(cfg, 'corr_block3_rank_cap_far', Inf);
    cfg = set_default(cfg, 'corr_block3_rank_cap_pml', Inf);

    cfg = set_default(cfg, 'reference_mode', 'waveholtz'); % waveholtz | helmholtz
    cfg = set_default(cfg, 'online_completion_mode', 'waveholtz'); % helmholtz | waveholtz | shifted_laplacian
    cfg = set_default(cfg, 'run_wave_baselines', true);
    cfg = set_default(cfg, 'rom_check_target', 'wave'); % helm | wave

    cfg = set_default(cfg, 'helm_gmres_track_tol', 1e-10);
    cfg = set_default(cfg, 'helm_completion_restart', 30);
    cfg = set_default(cfg, 'helm_completion_maxit', 60);
    cfg = set_default(cfg, 'helm_completion_tol', []);
    cfg = set_default(cfg, 'helm_baseline_iters', 60);
    cfg = set_default(cfg, 'helm_baseline_tol', 1e-12);
    cfg = set_default(cfg, 'shifted_laplacian_beta', 0.20);
    cfg = set_default(cfg, 'shifted_laplacian_shift_scale', 'omega2'); % omega2 | identity
    cfg = set_default(cfg, 'shifted_laplacian_restart', cfg.helm_completion_restart);
    cfg = set_default(cfg, 'shifted_laplacian_maxit', 5);
    cfg = set_default(cfg, 'shifted_laplacian_tol', []);
    cfg = set_default(cfg, 'shifted_laplacian_reorthogonalize', true);
    cfg = set_default(cfg, 'shifted_laplacian_verbose', false);
    cfg = set_default(cfg, 'shifted_laplacian_solver', 'lu');
    cfg = set_default(cfg, 'shifted_laplacian_ilu_type', 'ilutp');
    cfg = set_default(cfg, 'shifted_laplacian_ilu_droptol', 1e-2);
    cfg = set_default(cfg, 'shifted_laplacian_ilu_thresh', 1.0);
    cfg = set_default(cfg, 'shifted_laplacian_ilu_milu', 'off');

    cfg = set_default(cfg, 'CFL', 0.20);
    cfg = set_default(cfg, 'filter_shift', 0.25);
    cfg = set_default(cfg, 'wh_max_iter', 400);
    cfg = set_default(cfg, 'wh_tol', 1e-8);
    cfg = set_default(cfg, 'wh_relax', 0.15);
    cfg = set_default(cfg, 'wh_relax_min', 0.03);
    cfg = set_default(cfg, 'wh_rec_tol', Inf);
    cfg = set_default(cfg, 'wh_max_passes', 1);
    cfg = set_default(cfg, 'wh_pass_relax_decay', 0.5);
    cfg = set_default(cfg, 'wh_pass_iter', []);
    cfg = set_default(cfg, 'wh_verbose_iter', false);
    cfg = set_default(cfg, 'wh_lift_mode', 'frequency'); % frequency | u_real_only

    cfg = set_default(cfg, 'do_plots', true);
    cfg = set_default(cfg, 'save_plots', true);
    cfg = set_default(cfg, 'figure_root', helm_output_path('outputs', 'figures'));
    cfg = set_default(cfg, 'figure_subdir', 'ecc_correction_rom');
    cfg = set_default(cfg, 'figure_format', 'png');
    cfg = set_default(cfg, 'plot_obstacle_white', true);
    cfg = set_default(cfg, 'plot_y_limits', []);
    cfg = set_default(cfg, 'save_benchmark', true);
    cfg = set_default(cfg, 'benchmark_root', helm_output_path('outputs', 'benchmarks'));
    cfg = set_default(cfg, 'benchmark_name', 'ecc_correction_rom');
    cfg = set_default(cfg, 'plot_test_index', 1);

    cfg = set_default(cfg, 'parallel_train', true);
    cfg = set_default(cfg, 'parallel_train_min_cases', 2);
    cfg = set_default(cfg, 'parallel_test', true);
    cfg = set_default(cfg, 'parallel_test_min_cases', 2);
    cfg = set_default(cfg, 'vectorized_scattering_assembly', true);
    cfg = set_default(cfg, 'stream_wave_projection', true);
    cfg = set_default(cfg, 'setup_only', false);

    if cfg.ratio_min <= 0 || cfg.ratio_max <= 0 || cfg.ratio_min >= cfg.ratio_max
        error('Require 0 < ratio_min < ratio_max.');
    end
    if isempty(cfg.ratio_train)
        log_train = linspace(log(cfg.ratio_min), log(cfg.ratio_max), cfg.ratio_ntrain);
        cfg.ratio_train = exp(log_train);
    end
    cfg.ratio_test_auto_generated = isempty(cfg.ratio_test);
    if isempty(cfg.ratio_test)
        if numel(cfg.ratio_train) >= 2
            cfg.ratio_test = generate_ratio_test_params(cfg.ratio_train, cfg.ratio_test_strategy, ...
                cfg.ratio_test_rng_seed, cfg.ratio_test_log_offset);
        else
            cfg.ratio_test = cfg.ratio_train * 1.01;
        end
    end

    cfg.ratio_train = unique(cfg.ratio_train(:).');
    cfg.ratio_test = unique(cfg.ratio_test(:).');
    validate_shape_ratios(cfg.ratio_train, 'ratio_train');
    validate_shape_ratios(cfg.ratio_test, 'ratio_test');
    if any(abs(cfg.ratio_test(:) - cfg.ratio_train(:).') < 1e-12)
        error('ratio_test must be unseen and cannot overlap ratio_train.');
    end
    cfg.reference_mode = normalize_reference_mode_ecc(cfg.reference_mode);
    cfg.online_completion_mode = normalize_online_completion_mode_ecc(cfg.online_completion_mode);
    cfg.shifted_laplacian_shift_scale = normalize_shifted_laplacian_shift_scale_ecc(cfg.shifted_laplacian_shift_scale);
    if ~any(strcmp(cfg.reference_mode, {'helmholtz', 'waveholtz'}))
        error('reference_mode must be helmholtz or waveholtz.');
    end
    if ~any(strcmp(cfg.online_completion_mode, {'helmholtz', 'waveholtz', 'shifted_laplacian'}))
        error('online_completion_mode must be helmholtz, waveholtz, or shifted_laplacian.');
    end
    if ~any(strcmp(cfg.wh_lift_mode, {'frequency', 'u_real_only'}))
        error('wh_lift_mode must be frequency or u_real_only.');
    end
    if ~any(strcmp(cfg.rom_check_target, {'helm', 'wave'}))
        error('rom_check_target must be helm or wave.');
    end
    cfg.corr_pod_truncation_rule = normalize_pod_truncation_rule(cfg.corr_pod_truncation_rule);
    if cfg.corr_num_layers < 0 || floor(cfg.corr_num_layers) ~= cfg.corr_num_layers
        error('corr_num_layers must be a nonnegative integer.');
    end
    cfg.corr_rank_rtols = expand_layer_rtols(cfg.corr_rank_rtols, cfg.corr_rank_rtol, ...
        cfg.corr_num_layers, 'corr_rank_rtols');
    if isempty(cfg.corr_rank_rtol) && ~isempty(cfg.corr_rank_rtols)
        cfg.corr_rank_rtol = cfg.corr_rank_rtols(1);
    end
    cfg.corr_block3_rtol_near = expand_layer_rtols(cfg.corr_block3_rtol_near, cfg.corr_rank_rtols, ...
        cfg.corr_num_layers, 'corr_block3_rtol_near');
    cfg.corr_block3_rtol_far = expand_layer_rtols(cfg.corr_block3_rtol_far, cfg.corr_rank_rtols, ...
        cfg.corr_num_layers, 'corr_block3_rtol_far');
    cfg.corr_block3_rtol_pml = expand_layer_rtols(cfg.corr_block3_rtol_pml, cfg.corr_rank_rtols, ...
        cfg.corr_num_layers, 'corr_block3_rtol_pml');
    if strcmp(cfg.corr_pod_truncation_rule, 'relative_frobenius_tail') && isempty(cfg.corr_block3_global_tail_tols)
        if isempty(cfg.corr_block3_tail_tol_near)
            cfg.corr_block3_tail_tol_near = cfg.corr_rank_rtols;
        end
        if isempty(cfg.corr_block3_tail_tol_far)
            cfg.corr_block3_tail_tol_far = cfg.corr_rank_rtols;
        end
        if isempty(cfg.corr_block3_tail_tol_pml)
            cfg.corr_block3_tail_tol_pml = cfg.corr_rank_rtols;
        end
    end
    cfg.corr_block3_tail_tol_near = expand_optional_layer_values(cfg.corr_block3_tail_tol_near, ...
        cfg.corr_num_layers, 'corr_block3_tail_tol_near');
    cfg.corr_block3_tail_tol_far = expand_optional_layer_values(cfg.corr_block3_tail_tol_far, ...
        cfg.corr_num_layers, 'corr_block3_tail_tol_far');
    cfg.corr_block3_tail_tol_pml = expand_optional_layer_values(cfg.corr_block3_tail_tol_pml, ...
        cfg.corr_num_layers, 'corr_block3_tail_tol_pml');
    cfg.corr_block3_global_tail_tols = expand_optional_layer_values(cfg.corr_block3_global_tail_tols, ...
        cfg.corr_num_layers, 'corr_block3_global_tail_tols');
    valid_basis_styles = {'global', 'block3'};
    if ~any(strcmp(cfg.direct_basis_style, valid_basis_styles))
        error('direct_basis_style must be global or block3.');
    end
    if ~any(strcmp(cfg.corr_basis_style, valid_basis_styles))
        error('corr_basis_style must be global or block3.');
    end
    if numel(cfg.block3_center) ~= 2 || any(~isfinite(cfg.block3_center))
        error('block3_center must be a finite 2-vector.');
    end
    if ~(isfinite(cfg.block3_r_near) && cfg.block3_r_near > 0)
        error('block3_r_near must be a positive finite scalar.');
    end
    validate_block3_trunc_pair(cfg.direct_block3_rtol_near, cfg.direct_block3_rtol_far, cfg.direct_block3_rtol_pml, ...
        cfg.direct_block3_rank_cap_near, cfg.direct_block3_rank_cap_far, cfg.direct_block3_rank_cap_pml, 'direct');
    validate_block3_trunc_pair(cfg.corr_block3_rtol_near, cfg.corr_block3_rtol_far, cfg.corr_block3_rtol_pml, ...
        cfg.corr_block3_rank_cap_near, cfg.corr_block3_rank_cap_far, cfg.corr_block3_rank_cap_pml, 'corr');
    validate_optional_layer_positive(cfg.corr_block3_tail_tol_near, 'corr_block3_tail_tol_near');
    validate_optional_layer_positive(cfg.corr_block3_tail_tol_far, 'corr_block3_tail_tol_far');
    validate_optional_layer_positive(cfg.corr_block3_tail_tol_pml, 'corr_block3_tail_tol_pml');
    validate_optional_layer_positive(cfg.corr_block3_global_tail_tols, 'corr_block3_global_tail_tols');
    if isempty(cfg.helm_completion_tol)
        cfg.helm_completion_tol = cfg.helm_gmres_track_tol;
    end
    if isempty(cfg.shifted_laplacian_tol)
        cfg.shifted_laplacian_tol = cfg.helm_completion_tol;
    end
    if ~(isfinite(cfg.shifted_laplacian_beta) && cfg.shifted_laplacian_beta > 0)
        error('shifted_laplacian_beta must be a positive finite scalar.');
    end
end

function ratio_test = generate_ratio_test_params(ratio_train, strategy, rng_seed, log_offset)
    ratio_train = ratio_train(:).';
    if numel(ratio_train) < 2
        ratio_test = ratio_train * 1.01;
        return;
    end

    strategy = lower(strtrim(strategy));
    log_train = log(ratio_train(:));
    log_lo = log_train(1:end-1);
    log_hi = log_train(2:end);

    switch strategy
        case {'random_interval', 'random', 'rand'}
            if isempty(rng_seed)
                theta = rand(size(log_lo));
            else
                old_rng = rng;
                rng(rng_seed, 'twister');
                theta = rand(size(log_lo));
                rng(old_rng);
            end
            log_test = log_lo + theta .* (log_hi - log_lo);

        case {'midpoint', 'middle', 'mid'}
            log_test = 0.5 * (log_lo + log_hi);

        case {'log_offset', 'offset'}
            if ~(isfinite(log_offset) && log_offset > 0 && log_offset < 1)
                error('ratio_test_log_offset must be between 0 and 1.');
            end
            log_test = log_lo + log_offset * (log_hi - log_lo);

        otherwise
            error('ratio_test_strategy=%s must be random_interval, midpoint, or log_offset.', strategy);
    end

    ratio_test = exp(log_test).';
end

function validate_block3_trunc_pair(rn, rf, rp, cn, cf, cp, tag)
    rtols = [rn, rf, rp];
    caps = [cn, cf, cp];
    if any(~isfinite(rtols)) || any(rtols <= 0)
        error('%s block3 rtol values must be positive finite.', tag);
    end
    bad_cap = ~(isinf(caps) | (isfinite(caps) & caps >= 0));
    if any(bad_cap)
        error('%s block3 rank caps must be nonnegative finite or inf.', tag);
    end
end

function validate_optional_layer_positive(values, field_name)
    if isempty(values)
        return;
    end
    if any(~isfinite(values(:))) || any(values(:) <= 0)
        error('%s must be empty or contain positive finite values.', field_name);
    end
end

function rule = normalize_pod_truncation_rule(rule)
    rule = lower(strtrim(string(rule)));
    switch rule
        case {"relative_frobenius_tail", "frobenius_tail", "tail", "relative_tail"}
            rule = 'relative_frobenius_tail';
        case {"relative_singular_value", "relative_svd", "svd", "relative"}
            rule = 'relative_singular_value';
        otherwise
            error('corr_pod_truncation_rule=%s must be relative_frobenius_tail or relative_singular_value.', rule);
    end
end

function values = expand_optional_layer_values(values, nlayers, field_name)
    if isempty(values)
        return;
    end
    values = values(:).';
    if isscalar(values)
        values = repmat(values, 1, nlayers);
    elseif numel(values) ~= nlayers
        error('%s must be scalar or have %d entries.', field_name, nlayers);
    end
end

function vals = expand_layer_rtols(values, fallback_values, nlayers, field_name)
    if nlayers == 0
        vals = [];
        return;
    end
    if isempty(values)
        if isempty(fallback_values)
            values = 10 .^ linspace(-9, -6, nlayers);
        else
            values = fallback_values;
        end
    end
    values = values(:).';
    values = convert_log10_exponent_rtols(values, field_name);
    if isscalar(values)
        vals = repmat(values, 1, nlayers);
    elseif numel(values) == nlayers
        vals = values;
    else
        error('%s must be scalar or have corr_num_layers=%d entries.', field_name, nlayers);
    end
    if any(vals <= 0)
        error(['%s entries must be positive finite tolerances, or all-negative log10 exponents ' ...
            'such as linspace(-6,-9,3).'], field_name);
    end
    if any(~isfinite(vals))
        error('%s entries must be positive finite tolerances.', field_name);
    end
end

function vals = convert_log10_exponent_rtols(vals, field_name)
    if isempty(vals)
        return;
    end
    finite_mask = isfinite(vals);
    if all(finite_mask(:)) && all(vals(:) < 0)
        fprintf('%s interpreted as log10 exponents; using tolerances %s\n', ...
            field_name, mat2str(10 .^ vals, 4));
        vals = 10 .^ vals;
        return;
    end
    if any(vals(:) <= 0)
        error(['%s must contain either positive tolerances or all-negative log10 exponents. ' ...
            'Examples: logspace(-6,-9,3), 10.^linspace(-6,-9,3), or linspace(-6,-9,3).'], field_name);
    end
end

function cfg = normalize_ecc_tolerance_aliases(cfg)
    cfg = move_alias_field(cfg, 'corr_rankrtol', 'corr_rank_rtol');
    cfg = move_alias_field(cfg, 'corr_rankrtols', 'corr_rank_rtols');
end

function cfg = move_alias_field(cfg, alias_name, canonical_name)
    if ~isfield(cfg, alias_name) || isempty(cfg.(alias_name))
        return;
    end
    if isfield(cfg, canonical_name) && ~isempty(cfg.(canonical_name))
        error('Use either cfg.%s or cfg.%s, not both.', canonical_name, alias_name);
    end
    cfg.(canonical_name) = cfg.(alias_name);
end

function out = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
    out = s;
end

function mode = normalize_reference_mode_ecc(mode)
    mode = lower(strtrim(char(string(mode))));
    mode = strrep(mode, '-', '_');
    switch mode
        case {'helm', 'helmholtz', 'helm_lift', 'direct_helmholtz'}
            mode = 'helmholtz';
        case {'wave', 'waveholtz', 'wh'}
            mode = 'waveholtz';
        otherwise
            error('reference_mode=%s must be helmholtz or waveholtz.', mode);
    end
end

function mode = normalize_online_completion_mode_ecc(mode)
    mode = lower(strtrim(char(string(mode))));
    mode = strrep(mode, '-', '_');
    mode = strrep(mode, ' ', '_');
    switch mode
        case {'helm', 'helmholtz', 'plain_helmholtz', 'helm_gmres'}
            mode = 'helmholtz';
        case {'wave', 'waveholtz', 'wh'}
            mode = 'waveholtz';
        case {'shifted_laplacian', 'shifted_lap', 'sl', 'shifted'}
            mode = 'shifted_laplacian';
        otherwise
            error('online_completion_mode=%s must be helmholtz, waveholtz, or shifted_laplacian.', mode);
    end
end

function label = reference_mode_label_ecc(mode)
    if strcmp(mode, 'helmholtz')
        label = 'Helm';
    else
        label = 'WH';
    end
end

function msg = format_reference_log_ecc(mode, info)
    if strcmp(mode, 'helmholtz')
        msg = sprintf('%.3fs (%3d it, rel %.2e)', info.total_time, info.iter_count, info.relres_final);
    else
        msg = sprintf('%.3fs (%3d it, rec %.2e, step %.2e)', ...
            info.total_time, info.iter_count, info.relres_final, info.step_relres_final);
    end
end

function label = completion_log_label_ecc(mode)
    switch mode
        case 'helmholtz'
            label = 'plain-helm';
        case 'shifted_laplacian'
            label = 'shifted-Lap';
        otherwise
            label = 'WH';
    end
end

function label = plain_completion_log_label_ecc(mode)
    switch mode
        case 'helmholtz'
            label = 'plain-helm';
        case 'shifted_laplacian'
            label = 'plain-shifted';
        otherwise
            label = 'plain-fp';
    end
end

function label = direct_completion_log_label_ecc(mode)
    switch mode
        case 'helmholtz'
            label = 'direct-helm';
        case 'shifted_laplacian'
            label = 'direct-shifted';
        otherwise
            label = 'direct-fp';
    end
end

function label = completion_display_label_ecc(mode)
    switch mode
        case 'helmholtz'
            label = 'Helmholtz completion';
        case 'shifted_laplacian'
            label = 'shifted-Laplacian completion';
        otherwise
            label = 'WaveHoltz completion';
    end
end

function label = plain_completion_display_label_ecc(mode)
    switch mode
        case 'helmholtz'
            label = 'plain Helmholtz';
        case 'shifted_laplacian'
            label = 'plain shifted-Laplacian';
        otherwise
            label = 'plain WaveHoltz';
    end
end

function scale = normalize_shifted_laplacian_shift_scale_ecc(scale)
    scale = lower(strtrim(char(string(scale))));
    scale = strrep(scale, '-', '_');
    switch scale
        case {'omega2', 'omega_2', 'k2', 'k_2', 'physical', 'helmholtz'}
            scale = 'omega2';
        case {'identity', 'i', 'unit', 'none'}
            scale = 'identity';
        otherwise
            error('shifted_laplacian_shift_scale must be omega2 or identity.');
    end
end

function validate_shape_ratios(r_list, name)
    if any(~isfinite(r_list(:))) || any(r_list(:) <= 0)
        error('%s must satisfy ratio > 0.', name);
    end
end

function cfg = adjust_train_ratios_for_supported_geometry(cfg, prob)
    if isempty(cfg.ratio_train)
        return;
    end

    ratio_old = cfg.ratio_train(:).';
    ratio_new = ratio_old;
    changed = false(size(ratio_old));

    for k = 1:numel(ratio_old)
        if is_supported_scattering_ratio(ratio_new(k), cfg, prob)
            continue;
        end

        candidate = find_nearby_supported_train_ratio(ratio_old(k), ratio_new, k, cfg, prob);
        if isempty(candidate)
            error(['ratio_train %.16g generates a scatterer geometry with an ambiguous ghost stencil. ' ...
                'Choose a nearby a/b value or reduce eta_boundary_min.'], ratio_old(k));
        end

        ratio_new(k) = candidate;
        changed(k) = true;
    end

    if any(changed)
        fprintf('Adjusted unsupported near-degenerate ratio_train values for ghost-stencil geometry:\n');
        for k = find(changed)
            fprintf('  a/b %.16g -> %.16g\n', ratio_old(k), ratio_new(k));
        end
    end

    cfg.ratio_train = unique(ratio_new(:).');
    if numel(cfg.ratio_train) ~= numel(ratio_old)
        error('Adjusted ratio_train lost a sample through duplication; choose a different ratio_train.');
    end
end

function candidate = find_nearby_supported_train_ratio(ratio_old, ratio_current, idx, cfg, prob)
    candidate = [];
    log_lo = log(cfg.ratio_min);
    log_hi = log(cfg.ratio_max);
    log_r0 = log(ratio_old);
    deltas = [1e-4, -1e-4, 3e-4, -3e-4, 1e-3, -1e-3, 3e-3, -3e-3, 1e-2, -1e-2];

    for delta = deltas
        log_try = log_r0 + delta;
        if log_try < log_lo || log_try > log_hi
            continue;
        end
        r_try = exp(log_try);
        if ratio_collides_with_existing_train(r_try, ratio_current, idx)
            continue;
        end
        if is_supported_scattering_ratio(r_try, cfg, prob)
            candidate = r_try;
            return;
        end
    end
end

function tf = ratio_collides_with_existing_train(r_try, ratio_current, idx)
    other = ratio_current(:).';
    other(idx) = [];
    tf = any(abs(r_try - other) < 1e-12);
end

function cfg = adjust_test_ratios_for_supported_geometry(cfg, prob)
    if isempty(cfg.ratio_test)
        return;
    end

    ratio_old = cfg.ratio_test(:).';
    ratio_new = ratio_old;
    changed = false(size(ratio_old));
    train_logs = sort(log(cfg.ratio_train(:)));
    offset_candidates = [cfg.ratio_test_log_offset, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 1/3, 2/3, 0.50];
    offset_candidates = unique(offset_candidates(isfinite(offset_candidates) & offset_candidates > 0 & offset_candidates < 1), 'stable');

    for k = 1:numel(ratio_old)
        if is_supported_scattering_ratio(ratio_old(k), cfg, prob)
            continue;
        end

        candidate = [];
        lr = log(ratio_old(k));
        if numel(train_logs) >= 2
            left_idx = find(train_logs <= lr, 1, 'last');
            if isempty(left_idx)
                left_idx = 1;
            elseif left_idx >= numel(train_logs)
                left_idx = numel(train_logs) - 1;
            end
            lo = train_logs(left_idx);
            hi = train_logs(left_idx + 1);
            for off = offset_candidates
                r_try = exp(lo + off * (hi - lo));
                if any(abs(r_try - cfg.ratio_train(:).') < 1e-12)
                    continue;
                end
                if is_supported_scattering_ratio(r_try, cfg, prob)
                    candidate = r_try;
                    break;
                end
            end
        end

        if isempty(candidate)
            direction = sign(log(ratio_old(k)));
            if direction == 0
                direction = 1;
            end
            for scale = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
                r_try = ratio_old(k) * exp(direction * scale);
                if any(abs(r_try - cfg.ratio_train(:).') < 1e-12)
                    continue;
                end
                if is_supported_scattering_ratio(r_try, cfg, prob)
                    candidate = r_try;
                    break;
                end
            end
        end

        if isempty(candidate)
            error(['ratio_test %.16g generates a scatterer geometry with an ambiguous ghost stencil. ' ...
                'Choose a nearby unseen a/b value or reduce eta_boundary_min.'], ratio_old(k));
        end

        ratio_new(k) = candidate;
        changed(k) = true;
    end

    if any(changed)
        fprintf('Adjusted unsupported near-degenerate ratio_test values for ghost-stencil geometry:\n');
        for k = find(changed)
            fprintf('  a/b %.16g -> %.16g\n', ratio_old(k), ratio_new(k));
        end
    end

    cfg.ratio_test = unique(ratio_new(:).');
    overlap_mat = abs(cfg.ratio_test(:) - cfg.ratio_train(:).') < 1e-12;
    if any(overlap_mat(:))
        error('Adjusted ratio_test overlaps ratio_train; choose a different ratio_test_log_offset.');
    end
end

function tf = is_supported_scattering_ratio(ratio_ab, cfg, prob)
    domain = prob.domain;
    semi_a = cfg.radius_ref * sqrt(ratio_ab);
    semi_b = cfg.radius_ref / sqrt(ratio_ab);
    try
        build_scatterer_geometry(domain.nx, domain.ny, prob.x, prob.y, cfg.center, semi_a, semi_b, cfg.eta_boundary_min);
        tf = true;
    catch ME
        if contains(ME.message, 'Ambiguous') || contains(ME.message, 'eta out of range') || contains(ME.message, 'Invalid')
            tf = false;
        else
            rethrow(ME);
        end
    end
end

function domain = build_domain(cfg)
    domain = struct();
    domain.nx_phys = cfg.Nphys;
    domain.ny_phys = cfg.Nphys;
    domain.hx = cfg.L / (domain.nx_phys + 1);
    domain.hy = cfg.L / (domain.ny_phys + 1);
    domain.nx = domain.nx_phys + 2 * cfg.npml;
    domain.ny = domain.ny_phys + 2 * cfg.npml;
    domain.nu = domain.nx * domain.ny;
end

function [x, y, X] = build_coordinates(domain)
    Lpml_x = (domain.nx - domain.nx_phys) * 0.5 * domain.hx;
    Lpml_y = (domain.ny - domain.ny_phys) * 0.5 * domain.hy;
    x = linspace(-Lpml_x, 1 + Lpml_x, domain.nx + 2);
    y = linspace(-Lpml_y, 1 + Lpml_y, domain.ny + 2);
    [X, ~] = ndgrid(x, y);
end

function dataset = solve_reference_set(prob, cfg, ratio_list, label)
    ns = numel(ratio_list);
    nu = prob.domain.nu;
    dataset = struct();
    dataset.ratio = ratio_list(:);
    dataset.log_ratio = log(ratio_list(:));
    dataset.u_ref = zeros(nu, ns);
    dataset.systems = cell(ns, 1);
    dataset.solve_time = zeros(ns, 1);
    dataset.iter = zeros(ns, 1);
    dataset.relres = zeros(ns, 1);
    dataset.log_line = cell(ns, 1);

    use_parallel = should_parallelize_cases(cfg.parallel_train, ns, cfg.parallel_train_min_cases);
    if strcmp(label, 'test')
        use_parallel = should_parallelize_cases(cfg.parallel_test, ns, cfg.parallel_test_min_cases);
    end
    fprintf('%s parallel = %d (%d cases)\n', label, use_parallel, ns);

    case_results = cell(ns, 1);
    if use_parallel
        parfor k = 1:ns
            case_results{k} = solve_reference_case(prob, cfg, ratio_list(k));
        end
    else
        for k = 1:ns
            case_results{k} = solve_reference_case(prob, cfg, ratio_list(k));
        end
    end

    for k = 1:ns
        case_result = case_results{k};
        dataset.u_ref(:, k) = case_result.u_ref;
        dataset.systems{k} = case_result.sys;
        dataset.solve_time(k) = case_result.solve_time;
        dataset.iter(k) = case_result.iter;
        dataset.relres(k) = case_result.relres;
        dataset.log_line{k} = case_result.log_line;
        fprintf('%s\n', strrep(case_result.log_line, 'case ??:', sprintf('%s %2d/%2d:', label, k, ns)));
    end
end

function case_result = solve_reference_case(prob, cfg, ratio_ab)
    sys = build_scattering_system_for_ratio(ratio_ab, cfg, prob);
    if strcmp(cfg.reference_mode, 'helmholtz')
        [u_ref, info_ref] = run_direct_helmholtz_reference_ecc(sys);
        ref_label = 'Helm';
        detail = sprintf('relres=%.2e', info_ref.relres_final);
    else
        [u_ref, info_ref] = run_waveholtz_completion(prob, cfg, sys, zeros(size(sys.b)));
        ref_label = 'WH';
        detail = sprintf('rec-res=%.2e, step=%.2e', info_ref.relres_final, info_ref.step_relres_final);
    end
    solve_time = info_ref.total_time;

    case_result = struct();
    case_result.sys = sys;
    case_result.u_ref = u_ref;
    case_result.solve_time = solve_time;
    case_result.iter = info_ref.iter_count;
    case_result.relres = info_ref.relres_final;
    case_result.log_line = sprintf('case ??: r=a/b=%.4f, ref-%s=%.3fs (%3d it), %s, active=%d, internal=%d', ...
        ratio_ab, ref_label, solve_time, info_ref.iter_count, detail, ...
        nnz(sys.geom.active(2:end-1, 2:end-1)), nnz(sys.geom.internal(2:end-1, 2:end-1)));
end

function block3_partition = build_scattering_block3_partition(prob, cfg, train_ref)
    nu = prob.domain.nu;
    active_mask = true(nu, 1);
    if cfg.block3_use_active_union
        active_mask = false(nu, 1);
        for k = 1:numel(train_ref.systems)
            active_mask(train_ref.systems{k}.active_vec_idx) = true;
        end
    end
    grid = struct();
    grid.nx = prob.domain.nx;
    grid.ny = prob.domain.ny;
    grid.x = prob.x;
    grid.y = prob.y;
    grid.npml = cfg.npml;
    grid.center = cfg.block3_center;
    grid.r_near = cfg.block3_r_near;
    grid.active_mask = active_mask;

    block3_partition = rom_reduced_space_utils('build_block3_partition', grid);
    fprintf('block3 partition: near=%d, far=%d, pml=%d\n', ...
        block3_partition.count_near, block3_partition.count_far, block3_partition.count_pml);
end

function direct_rom = build_direct_rom(~, cfg, train_ref, block3_partition)
    pod = build_pod_basis_fullrank(train_ref.u_ref, cfg.pod_rank_rtol);
    V = pod.basis;
    basis_space = rom_reduced_space_utils('make_dense_space', V);

    if strcmp(cfg.direct_basis_style, 'block3')
        if isempty(block3_partition)
            error('direct_basis_style=block3 but block3_partition is empty.');
        end
        basis_space = rom_reduced_space_utils('build_block3_space', train_ref.u_ref, block3_partition, ...
            struct( ...
                'rank_rtol_near', cfg.direct_block3_rtol_near, ...
                'rank_rtol_far', cfg.direct_block3_rtol_far, ...
                'rank_rtol_pml', cfg.direct_block3_rtol_pml, ...
                'rank_cap_near', cfg.direct_block3_rank_cap_near, ...
                'rank_cap_far', cfg.direct_block3_rank_cap_far, ...
                'rank_cap_pml', cfg.direct_block3_rank_cap_pml));
    end

    ns = numel(train_ref.ratio);
    nu = size(train_ref.u_ref, 1);
    train_u_init = zeros(nu, ns);
    train_direct_time = zeros(ns, 1);

    if cfg.use_direct_rom_init && ~rom_reduced_space_utils('is_empty', basis_space)
        for k = 1:ns
            A = train_ref.systems{k}.A;
            b = train_ref.systems{k}.b;
            tsolve = tic;
            train_u_init(:, k) = solve_galerkin(A, b, basis_space);
            train_direct_time(k) = toc(tsolve);
        end
    end

    direct_rom = struct();
    direct_rom.pod = pod;
    direct_rom.basis = V;
    direct_rom.basis_space = basis_space;
    direct_rom.train_u_init = train_u_init;
    direct_rom.train_direct_time = train_direct_time;
end

function print_direct_rom_summary(direct_rom)
    fprintf('direct snapshot matrix size = %d x %d\n', ...
        size(direct_rom.pod.left_modes, 1), size(direct_rom.pod.right_modes, 1));
    fprintf('direct basis rank used      = %d\n', direct_rom.pod.rank_used);
    if isfield(direct_rom, 'basis_space')
        space = direct_rom.basis_space;
        fprintf('direct reduced dim (active) = %d\n', rom_reduced_space_utils('dimension', space));
        if is_block3_space(space) && isfield(space, 'block_meta')
            bm = space.block_meta;
            fprintf('direct block ranks [near far pml] = [%d %d %d]\n', bm.near.rank, bm.far.rank, bm.pml.rank);
            fprintf('direct block dofs  [near far pml] = [%d %d %d]\n', bm.near.nrow, bm.far.nrow, bm.pml.nrow);
            storage = block_storage_stats_from_space(space, direct_rom.pod.rank_used);
            fprintf('direct storage dense/block = %.3g / %.3g, saving = %.3fx (%.1f%% less)\n', ...
                storage.dense_storage, storage.block_storage, ...
                storage.saving_ratio, 100 * storage.saved_fraction);
        end
    end
    fprintf('direct energy ranks r90/r95/r99 = %d / %d / %d\n', ...
        direct_rom.pod.r90, direct_rom.pod.r95, direct_rom.pod.r99);
end

function corr_rom = train_correction_rom_layers(prob, cfg, train_ref, direct_rom, block3_partition)
    ns = numel(train_ref.ratio);
    nlayers = cfg.corr_num_layers;
    nu = prob.domain.nu;

    corr_rom = struct();
    corr_rom.num_layers = nlayers;
    corr_rom.layers = cell(nlayers, 1);
    corr_rom.u_init_train = zeros(nu, ns);
    corr_rom.u_train_after_layer = cell(nlayers, 1);

    if cfg.use_direct_rom_init
        u_init = direct_rom.train_u_init;
    else
        u_init = zeros(nu, ns);
    end
    corr_rom.u_init_train = u_init;

    train_state = initialize_trajectory_aware_training_state(train_ref, u_init);

    for j = 1:nlayers
        eta_snapshots = zeros(nu, ns);
        if j == 1
            for k = 1:ns
                eta_snapshots(:, k) = train_state.eta_hist{k}(:, 1);
            end
        else
            prev_layer = corr_rom.layers{j - 1};
            for k = 1:ns
                [train_state, eta_curr] = advance_trajectory_aware_training_sample( ...
                    train_state, prev_layer, train_ref.systems{k}, k, j);
                eta_snapshots(:, k) = eta_curr;
            end
        end

        layer_rank_rtol = cfg.corr_rank_rtols(j);
        [basis_snapshots, snapshot_column_norms] = prepare_correction_snapshots_for_basis(eta_snapshots, cfg);
        pod = build_pod_basis_fullrank(basis_snapshots, layer_rank_rtol, cfg.corr_pod_truncation_rule);
        layer = struct();
        layer.index = j;
        layer.rank_rtol = layer_rank_rtol;
        layer.pod = pod;
        layer.basis_snapshots_normalized = cfg.corr_normalize_snapshots;
        layer.snapshot_column_norms = snapshot_column_norms;
        if strcmp(cfg.corr_basis_style, 'block3')
            if isempty(block3_partition)
                error('corr_basis_style=block3 but block3_partition is empty.');
            end
            block_rtol_near = cfg.corr_block3_rtol_near(j);
            block_rtol_far = cfg.corr_block3_rtol_far(j);
            block_rtol_pml = cfg.corr_block3_rtol_pml(j);
            layer.block3_rank_rtols = [block_rtol_near, block_rtol_far, block_rtol_pml];
            trunc_cfg = struct( ...
                'rank_rtol_near', block_rtol_near, ...
                'rank_rtol_far', block_rtol_far, ...
                'rank_rtol_pml', block_rtol_pml, ...
                'rank_cap_near', cfg.corr_block3_rank_cap_near, ...
                'rank_cap_far', cfg.corr_block3_rank_cap_far, ...
                'rank_cap_pml', cfg.corr_block3_rank_cap_pml);
            if ~isempty(cfg.corr_block3_tail_tol_near)
                trunc_cfg.rank_tail_tol_near = cfg.corr_block3_tail_tol_near(j);
            end
            if ~isempty(cfg.corr_block3_tail_tol_far)
                trunc_cfg.rank_tail_tol_far = cfg.corr_block3_tail_tol_far(j);
            end
            if ~isempty(cfg.corr_block3_tail_tol_pml)
                trunc_cfg.rank_tail_tol_pml = cfg.corr_block3_tail_tol_pml(j);
            end
            if ~isempty(cfg.corr_block3_global_tail_tols)
                trunc_cfg.rank_global_tail_tol = cfg.corr_block3_global_tail_tols(j);
                trunc_cfg.rank_global_tail_strategy = cfg.corr_block3_global_tail_strategy;
            end
            layer.basis_space = rom_reduced_space_utils('build_block3_space', basis_snapshots, block3_partition, trunc_cfg);
        else
            layer.block3_rank_rtols = [];
            layer.basis_space = rom_reduced_space_utils('make_dense_space', pod.basis);
        end
        layer.basis_u = pod.basis;
        layer.space_dim = rom_reduced_space_utils('dimension', layer.basis_space);
        layer.eta_snapshots = eta_snapshots;
        layer.train_fgmres_relres = zeros(ns, 1);
        layer.train_freq_error_after = zeros(ns, 1);
        layer.train_time = zeros(ns, 1);

        prefix_layers = corr_rom.layers;
        prefix_layers{j} = layer;
        prefix_rom = struct('num_layers', j, 'layers', {prefix_layers(1:j)});

        u_after = zeros(nu, ns);
        for k = 1:ns
            [u_local, info_train] = layered_fgmres_precorrection(train_ref.systems{k}, cfg, u_init(:, k), prefix_rom);
            u_after(:, k) = u_local;
            layer.train_fgmres_relres(k) = info_train.relres_final;
            layer.train_freq_error_after(k) = norm(u_local - train_ref.u_ref(:, k)) / max(norm(train_ref.u_ref(:, k)), 1e-16);
            layer.train_time(k) = info_train.total_time;
        end

        layer.trajectory_diagnostics = train_state.diagnostics(:,j);
        corr_rom.layers{j} = layer;
        corr_rom.u_train_after_layer{j} = u_after;

        fprintf(['layer %2d/%2d: error snapshot size = %d x %d, rtol = %.2e, basis rank = %d, ' ...
            'train relres mean = %.3e, train freq err mean = %.3e\n'], ...
            j, nlayers, size(eta_snapshots, 1), size(eta_snapshots, 2), layer_rank_rtol, layer.space_dim, ...
            mean(layer.train_fgmres_relres), mean(layer.train_freq_error_after));
    end
end

function print_correction_rom_summary(corr_rom)
    if corr_rom.num_layers == 0
        fprintf('no layered Helmholtz ROM steps requested.\n');
        return;
    end
    for j = 1:corr_rom.num_layers
        pod = corr_rom.layers{j}.pod;
        fprintf('layered FGMRES ROM layer %2d: rtol = %.2e, rank = %d, r90/r95/r99 = %d / %d / %d, train relres mean = %.3e\n', ...
            j, corr_rom.layers{j}.rank_rtol, corr_rom.layers{j}.space_dim, pod.r90, pod.r95, pod.r99, mean(corr_rom.layers{j}.train_fgmres_relres));
        layer_space = get_layer_basis_space(corr_rom.layers{j});
        if is_block3_space(layer_space) && isfield(layer_space, 'block_meta')
            bm = layer_space.block_meta;
            storage = block_storage_stats_from_space(layer_space, corr_rom.layers{j}.pod.rank_used);
            if isfield(bm.near, 'rank_global_tail_tol')
                fprintf(['  block global tail budget = %.2e, achieved = %.3e, local tails = [%s], ' ...
                    'ranks = [%d %d %d], dofs = [%d %d %d], storage dense/block = %.3g / %.3g, ' ...
                    'saving = %.3fx (%.1f%% less)\n'], ...
                    bm.near.rank_global_tail_tol, bm.near.global_relative_tail, ...
                    sprintf('%.2e %.2e %.2e', bm.near.local_relative_tail, bm.far.local_relative_tail, bm.pml.local_relative_tail), ...
                    bm.near.rank, bm.far.rank, bm.pml.rank, bm.near.nrow, bm.far.nrow, bm.pml.nrow, ...
                    storage.dense_storage, storage.block_storage, ...
                    storage.saving_ratio, 100 * storage.saved_fraction);
            else
                fprintf(['  block tail tolerances [near far pml] = [%s], ranks = [%d %d %d], dofs = [%d %d %d], ' ...
                    'storage dense/block = %.3g / %.3g, saving = %.3fx (%.1f%% less)\n'], ...
                sprintf('%.1e %.1e %.1e', bm.near.rank_tail_tol, bm.far.rank_tail_tol, bm.pml.rank_tail_tol), ...
                bm.near.rank, bm.far.rank, bm.pml.rank, bm.near.nrow, bm.far.nrow, bm.pml.nrow, ...
                storage.dense_storage, storage.block_storage, ...
                storage.saving_ratio, 100 * storage.saved_fraction);
            end
        end
    end
end

function [basis_snapshots, column_norms] = prepare_correction_snapshots_for_basis(eta_snapshots, cfg)
    column_norms = vecnorm(eta_snapshots, 2, 1);
    basis_snapshots = eta_snapshots;
    if ~cfg.corr_normalize_snapshots
        return;
    end
    safe_norms = column_norms;
    safe_norms(safe_norms <= 1e-15 | ~isfinite(safe_norms)) = 1;
    basis_snapshots = eta_snapshots ./ safe_norms;
end

function train_state = initialize_trajectory_aware_training_state(train_ref, u_init)
    ns = numel(train_ref.ratio);
    nu = size(u_init, 1);
    train_state = struct();
    train_state.beta0 = zeros(ns, 1);
    train_state.q_curr = zeros(nu, ns);
    train_state.Q_hist = cell(ns, 1);
    train_state.H_hist = cell(ns, 1);
    train_state.eta_hist = cell(ns, 1);
    train_state.diagnostics = cell(ns,1);

    for k = 1:ns
        A = train_ref.systems{k}.A;
        b = train_ref.systems{k}.b;
        r0 = b - A * u_init(:, k);
        beta0 = norm(r0);
        train_state.beta0(k) = beta0;

        q1 = zeros(nu, 1);
        eta1 = zeros(nu, 1);
        if beta0 > 1e-15
            q1 = r0 / beta0;
            eta1 = (train_ref.u_ref(:, k) - u_init(:, k)) / beta0;
        end

        train_state.q_curr(:, k) = q1;
        train_state.Q_hist{k} = q1;
        train_state.H_hist{k} = zeros(1, 0);
        train_state.eta_hist{k} = eta1;
    end
end

function [train_state, eta_curr] = advance_trajectory_aware_training_sample(train_state, prev_layer, sys, param_idx, layer_idx)
    A = sys.A;
    q_curr = train_state.q_curr(:, param_idx);
    prev_step = layer_idx - 1;

    if train_state.beta0(param_idx) <= 1e-15 || norm(q_curr) <= 1e-15
        train_state.diagnostics{param_idx,layer_idx} = struct('breakdown',true);
        Q_hist = train_state.Q_hist{param_idx};
        H_hist = train_state.H_hist{param_idx};
        eta_hist = train_state.eta_hist{param_idx};
        if size(Q_hist, 2) < layer_idx
            Q_hist(:, layer_idx) = zeros(size(Q_hist, 1), 1);
        end
        H_hist(layer_idx, prev_step) = 0;
        eta_hist(:, layer_idx) = zeros(size(eta_hist, 1), 1);
        train_state.Q_hist{param_idx} = Q_hist;
        train_state.H_hist{param_idx} = H_hist;
        train_state.eta_hist{param_idx} = eta_hist;
        train_state.q_curr(:, param_idx) = zeros(size(q_curr));
        eta_curr = zeros(size(q_curr));
        return;
    end

    z_rom = apply_layer_inverse_to_rhs(prev_layer, A, q_curr);
    Q_hist = train_state.Q_hist{param_idx};
    H_hist = train_state.H_hist{param_idx};
    eta_hist = train_state.eta_hist{param_idx};

    [qnext,eta_curr,h,hn,diagnostic] = rom_verified_snapshot_step( ...
        A,z_rom,Q_hist(:,1:prev_step),eta_hist(:,1:prev_step));
    H_hist(1:prev_step,prev_step)=h;
    H_hist(layer_idx,prev_step)=hn;
    Q_hist(:,layer_idx)=qnext;
    train_state.q_curr(:,param_idx)=qnext;
    train_state.diagnostics{param_idx,layer_idx}=diagnostic;
    eta_hist(:, layer_idx) = eta_curr;

    train_state.Q_hist{param_idx} = Q_hist;
    train_state.H_hist{param_idx} = H_hist;
    train_state.eta_hist{param_idx} = eta_hist;
end

function z = apply_layer_inverse_to_rhs(layer, A, rhs_vec)
    basis_space = get_layer_basis_space(layer);
    if rom_reduced_space_utils('is_empty', basis_space)
        z = zeros(size(rhs_vec));
        return;
    end
    Ar = rom_reduced_space_utils('assemble_reduced_matrix', A, basis_space);
    coeff = solve_small_system(Ar, rom_reduced_space_utils('project', basis_space, rhs_vec));
    z = rom_reduced_space_utils('lift', basis_space, coeff);
end

function [u_curr, info] = layered_fgmres_precorrection(sys, cfg, u0, corr_rom)
    nlayers = corr_rom.num_layers;
    A = sys.A;
    b = sys.b;
    nb = max(norm(b), 1e-16);

    r0 = b - A * u0;
    beta0 = norm(r0);

    info = struct();
    info.layer_u = cell(nlayers, 1);
    info.layer_time = zeros(nlayers, 1);
    info.layer_relres = zeros(nlayers, 1);
    info.relres_hist = zeros(nlayers + 1, 1);
    info.matvec_count = 1;
    info.iter_count = 0;
    info.relres_final = beta0 / nb;
    info.total_time = 0;

    tprec = tic;
    if beta0 <= cfg.helm_gmres_track_tol || nlayers == 0
        u_curr = u0;
        info.relres_hist(1) = beta0 / nb;
        info.relres_final = beta0 / nb;
        info.total_time = toc(tprec);
        return;
    end

    nu = size(A, 1);
    Q = zeros(nu, nlayers + 1);
    Z = zeros(nu, nlayers);
    H = zeros(nlayers + 1, nlayers);

    Q(:, 1) = r0 / beta0;
    info.relres_hist(1) = beta0 / nb;
    u_curr = u0;

    for j = 1:nlayers
        tlayer = tic;
        z = apply_layer_inverse_to_rhs(corr_rom.layers{j}, A, Q(:, j));
        Z(:, j) = z;

        w = A * z;
        info.matvec_count = info.matvec_count + 1;

        [Q(:,j+1),H(1:j,j),H(j+1,j),breakdown] = rom_arnoldi_step(w,Q(:,1:j));

        rhs_ls = zeros(j + 1, 1);
        rhs_ls(1) = beta0;
        y = solve_small_least_squares(H(1:j + 1, 1:j), rhs_ls);
        u_curr = u0 + Z(:, 1:j) * y;

        r_true = b - A * u_curr;
        info.matvec_count = info.matvec_count + 1;
        info.layer_relres(j) = norm(r_true) / nb;
        info.relres_hist(j + 1) = info.layer_relres(j);
        info.layer_u{j} = u_curr;
        info.layer_time(j) = toc(tlayer);
        info.iter_count = j;

        if info.layer_relres(j) <= cfg.helm_gmres_track_tol || breakdown
            break;
        end
    end

    info.relres_hist = info.relres_hist(1:info.iter_count + 1);
    if info.iter_count > 0
        info.relres_final = info.layer_relres(info.iter_count);
    else
        info.relres_final = beta0 / nb;
    end
    info.total_time = toc(tprec);
end

function [u_sol, info] = run_plain_helmholtz_gmres(sys, u0, restart, maxit_outer, tol, verbose)
    if nargin < 6
        verbose = false;
    end
    A = sys.A;
    b = sys.b;
    nb = max(norm(b), 1e-16);
    Afun = @(x) A * x;

    tol_use = sanitize_gmres_tol(tol);

    tsolve = tic;
    [u_sol, flag, relres, iter, resvec] = gmres(Afun, b, restart, tol_use, maxit_outer, [], [], u0);
    total_time = toc(tsolve);

    info = struct();
    info.total_time = total_time;
    info.iter_count = iteration_count(iter, restart);
    info.relres_final = relres;
    info.relres_hist = resvec / nb;
    info.op_apply_count = info.iter_count + 1;
    info.requested_tol = tol;
    info.used_tol = tol_use;
    info.step_hist = zeros(0, 1);
    if flag == 0
        info.stop_reason = 'residual';
    else
        info.stop_reason = 'maxit';
    end
    info.u_final = u_sol;

    if verbose
        fprintf('gmres: it=%d, relres=%.3e\n', info.iter_count, info.relres_final);
    end
end

function [u_ref, info] = run_direct_helmholtz_reference_ecc(sys)
    A = sys.A;
    b = sys.b;
    tref = tic;
    u_ref = A \ b;
    total_time = toc(tref);
    nb = max(norm(b), 1e-16);

    info = struct();
    info.total_time = total_time;
    info.wave_time = NaN;
    info.lift_time = 0;
    info.iter_count = 0;
    info.relres_final = norm(b - A * u_ref) / nb;
    info.relres_hist = info.relres_final;
    info.op_apply_count = 1;
    info.step_hist = zeros(0, 1);
    info.step_relres_final = NaN;
    info.stop_reason = 'direct';
    info.u_final = u_ref;
end

function [u_sol, info] = run_shifted_laplacian_completion_ecc(sys, cfg, u0)
    A = sys.A;
    b = sys.b;
    Afun = @(x) A * x;
    if norm(b-A*u0)/max(norm(b),1e-16) <= cfg.shifted_laplacian_tol
        % An accepted ROM field needs no shifted matrix or factorization.
        % GMRES still performs its normal initial true-residual verification.
        Msolve = @(v)v;
        setup_info = struct('setup_time',0,'factor_bytes',0, ...
            'solver',cfg.shifted_laplacian_solver);
    else
        [Msolve, setup_info] = build_shifted_laplacian_solver_ecc(sys, cfg);
    end

    opts = struct();
    opts.restart = cfg.shifted_laplacian_restart;
    opts.maxit = cfg.shifted_laplacian_maxit;
    opts.tol = cfg.shifted_laplacian_tol;
    opts.verbose = cfg.shifted_laplacian_verbose;
    opts.reorthogonalize = cfg.shifted_laplacian_reorthogonalize;

    [u_sol, info] = shifted_laplacian_gmres(Afun, Msolve, b, u0, opts);
    info.solve_time = info.total_time;
    info.preconditioner_setup_time = setup_info.setup_time;
    info.preconditioner_factor_bytes = setup_info.factor_bytes;
    info.preconditioner_solver = setup_info.solver;
    info.total_time = setup_info.setup_time + info.solve_time;
    info.wave_time = NaN;
    info.lift_time = 0;
    info.lift_mode = 'none';
    info.step_hist = zeros(0, 1);
    info.step_relres_final = NaN;
    info.shifted_laplacian_beta = cfg.shifted_laplacian_beta;
    info.shifted_laplacian_shift_scale = cfg.shifted_laplacian_shift_scale;
    info.shifted_laplacian_shift_strength = shifted_laplacian_shift_strength_ecc(cfg);
    info.u_final = u_sol;
end

function [Msolve, info] = build_shifted_laplacian_solver_ecc(sys, cfg)
    n = size(sys.A, 1);
    shift_strength = shifted_laplacian_shift_strength_ecc(cfg);
    Ashift = sys.A + 1i * cfg.shifted_laplacian_beta * shift_strength * speye(n);
    opts = struct('solver', cfg.shifted_laplacian_solver, ...
        'ilu_type', cfg.shifted_laplacian_ilu_type, ...
        'ilu_droptol', cfg.shifted_laplacian_ilu_droptol, ...
        'ilu_thresh', cfg.shifted_laplacian_ilu_thresh, ...
        'ilu_milu', cfg.shifted_laplacian_ilu_milu);
    [Msolve, info] = build_shifted_laplacian_preconditioner(Ashift, opts);
end

function shift_strength = shifted_laplacian_shift_strength_ecc(cfg)
    switch cfg.shifted_laplacian_shift_scale
        case 'omega2'
            shift_strength = cfg.omega ^ 2;
        otherwise
            shift_strength = 1;
    end
end

function [u_sol, info] = run_baseline_completion_ecc(prob, cfg, sys, u0)
    switch cfg.online_completion_mode
        case 'helmholtz'
            [u_sol, info] = run_plain_helmholtz_gmres( ...
                sys, u0, cfg.helm_completion_restart, cfg.helm_completion_maxit, cfg.helm_completion_tol, false);
        case 'shifted_laplacian'
            [u_sol, info] = run_shifted_laplacian_completion_ecc(sys, cfg, u0);
        otherwise
            [u_sol, info] = run_waveholtz_completion(prob, cfg, sys, u0);
    end
end

function tol_use = sanitize_gmres_tol(tol)
    % MATLAB gmres warns when tol is too small (or non-positive). Clamp to a
    % practical lower bound while preserving user intent for tight solves.
    if isempty(tol) || ~isfinite(tol) || tol <= 0
        tol_use = 1e-12;
        return;
    end
    tol_use = min(max(tol, 1e-14), 1 - 1e-12);
end

function [u_sol, info] = run_waveholtz_completion(prob, cfg, sys, u_init)
    domain = prob.domain;
    nu = domain.nu;

    tlift = tic;
    if strcmp(cfg.wh_lift_mode, 'frequency')
        x0 = lift_frequency_solution_to_real_state_scattering( ...
            domain.nx, domain.ny, domain.hx, domain.hy, cfg.c, ...
            prob.omega_ts, prob.dt, prob.pml_wave, sys.geom, prob.inc, u_init);
    else
        x0 = zeros(4 * nu + (domain.nx + 1) * domain.ny + domain.nx * (domain.ny + 1), 1);
        x0(1:nu) = real(u_init);
    end
    lift_time = toc(tlift);

    if isempty(cfg.wh_pass_iter)
        pass_iter = cfg.wh_max_iter;
    else
        pass_iter = cfg.wh_pass_iter;
    end

    tw = tic;
    x_pass = x0;
    relax_pass = cfg.wh_relax;
    step_hist_all = [];
    rec_hist_all = [];
    rec_res = Inf;
    u_full = zeros(domain.nx + 2, domain.ny + 2);
    for ipass = 1:max(cfg.wh_max_passes, 1)
        [u_full, step_hist, rec_hist, rec_res, x_pass] = waveholtz_iteration_scattering_with_init( ...
            domain.nx, domain.ny, domain.hx, domain.hy, cfg.c, ...
            prob.omega_ts, prob.T_ts, prob.dt, prob.nt, prob.pml_wave, ...
            sys.geom, prob.inc, pass_iter, cfg.wh_tol, cfg.filter_shift, relax_pass, sys.A, sys.b, cfg.wh_verbose_iter, x_pass, cfg.stream_wave_projection);
        step_hist_all = [step_hist_all; step_hist(:)]; %#ok<AGROW>
        if isempty(rec_hist_all)
            rec_hist_all = rec_hist(:);
        else
            rec_hist_all = [rec_hist_all; rec_hist(2:end)]; %#ok<AGROW>
        end
        if rec_res <= cfg.wh_rec_tol
            break;
        end
        relax_pass = max(cfg.wh_relax_min, relax_pass * cfg.wh_pass_relax_decay);
    end
    wave_time = toc(tw);
    total_time = lift_time + wave_time;

    u_sol = reshape(u_full(2:end-1, 2:end-1), nu, 1);

    info = struct();
    info.total_time = total_time;
    info.wave_time = wave_time;
    info.lift_time = lift_time;
    info.lift_mode = cfg.wh_lift_mode;
    info.iter_count = numel(step_hist_all);
    info.relres_final = rec_res;
    info.relres_hist = rec_hist_all(:);
    info.op_apply_count = numel(step_hist_all);
    info.step_hist = step_hist_all(:);
    if isempty(step_hist_all)
        info.step_relres_final = NaN;
    else
        info.step_relres_final = step_hist_all(end);
    end
    if info.step_relres_final <= cfg.wh_tol
        info.stop_reason = 'fixed_point_step';
    elseif isfinite(cfg.wh_rec_tol) && rec_res <= cfg.wh_rec_tol
        info.stop_reason = 'recovered_phasor_residual';
    else
        info.stop_reason = 'maxit';
        if isfinite(cfg.wh_rec_tol) && ~cfg.wh_verbose_iter
            warning('WaveHoltz rec_res %.3e > wh_rec_tol %.3e after %d iterations.', rec_res, cfg.wh_rec_tol, info.iter_count);
        end
    end
    info.rec_tol = cfg.wh_rec_tol;
    info.u_final = u_sol;
end

function test = evaluate_correction_rom_online(prob, cfg, ~, direct_rom, corr_rom)
    ns = numel(cfg.ratio_test);
    nlayers = corr_rom.num_layers;
    nu = prob.domain.nu;

    test = initialize_online_test_struct(ns, nlayers, nu);
    test.ratio = cfg.ratio_test(:);

    use_parallel = should_parallelize_cases(cfg.parallel_test, ns, cfg.parallel_test_min_cases);
    fprintf('online test parallel = %d (%d cases)\n', use_parallel, ns);

    case_results = cell(ns, 1);
    if use_parallel
        parfor k = 1:ns
            case_results{k} = evaluate_one_online_case(prob, cfg, direct_rom, corr_rom, cfg.ratio_test(k));
        end
    else
        for k = 1:ns
            case_results{k} = evaluate_one_online_case(prob, cfg, direct_rom, corr_rom, cfg.ratio_test(k));
        end
    end

    for k = 1:ns
        test = store_online_case_result(test, case_results{k}, k, nlayers);
        fprintf('%s\n', strrep(case_results{k}.log_line, 'test ??:', sprintf('test %2d/%2d:', k, ns)));
    end
end

function test = initialize_online_test_struct(ns, nlayers, nu)
    test = struct();
    test.ratio = zeros(ns, 1);
    test.systems = cell(ns, 1);
    test.wave_u_ref = zeros(nu, ns);
    test.wave_time = zeros(ns, 1);
    test.wave_iter = zeros(ns, 1);
    test.helm_u_ref = zeros(nu, ns);
    test.helm_time = zeros(ns, 1);

    test.direct_u = zeros(nu, ns);
    test.direct_time = zeros(ns, 1);
    test.direct_op_apply = zeros(ns, 1);

    test.direct_fp_u = zeros(nu, ns);
    test.direct_fp_time = zeros(ns, 1);
    test.direct_fp_total_time = zeros(ns, 1);
    test.direct_fp_iter = zeros(ns, 1);
    test.direct_fp_relres_final = zeros(ns, 1);
    test.direct_fp_op_apply = zeros(ns, 1);
    test.direct_fp_relres_hist = cell(ns, 1);
    test.direct_fp_step_hist = cell(ns, 1);
    test.direct_fp_stop_reason = cell(ns, 1);

    test.plain_fp_u = zeros(nu, ns);
    test.plain_fp_time = zeros(ns, 1);
    test.plain_fp_iter = zeros(ns, 1);
    test.plain_fp_relres_final = zeros(ns, 1);
    test.plain_fp_op_apply = zeros(ns, 1);
    test.plain_fp_relres_hist = cell(ns, 1);
    test.plain_fp_step_hist = cell(ns, 1);
    test.plain_fp_stop_reason = cell(ns, 1);

    test.online_precorr_u = zeros(nu, ns);
    test.online_precorr_time = zeros(ns, 1);
    test.online_lift_time = zeros(ns, 1);
    test.online_precorr_matvec = zeros(ns, 1);
    test.online_precorr_relres_final = zeros(ns, 1);
    test.online_precorr_relres_hist = cell(ns, 1);

    test.online_final_u = zeros(nu, ns);
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

    test.direct_err_wave_full = zeros(ns, 1);
    test.direct_fp_err_wave_full = zeros(ns, 1);
    test.plain_fp_err_wave_full = zeros(ns, 1);
    test.online_final_err_wave_full = zeros(ns, 1);

    test.direct_err_wave_phys = zeros(ns, 1);
    test.direct_fp_err_wave_phys = zeros(ns, 1);
    test.plain_fp_err_wave_phys = zeros(ns, 1);
    test.online_final_err_wave_phys = zeros(ns, 1);

    test.direct_err_helm_full = zeros(ns, 1);
    test.direct_fp_err_helm_full = zeros(ns, 1);
    test.plain_fp_err_helm_full = zeros(ns, 1);
    test.online_final_err_helm_full = zeros(ns, 1);

    test.direct_err_helm_phys = zeros(ns, 1);
    test.direct_fp_err_helm_phys = zeros(ns, 1);
    test.plain_fp_err_helm_phys = zeros(ns, 1);
    test.online_final_err_helm_phys = zeros(ns, 1);

    test.corr_u_layers = cell(nlayers, 1);
    test.corr_time_layers = zeros(nlayers, ns);
    test.layer_err_helm_full = zeros(nlayers, ns);
    test.layer_err_helm_phys = zeros(nlayers, ns);
    for j = 1:nlayers
        test.corr_u_layers{j} = zeros(nu, ns);
    end
end

function case_result = evaluate_one_online_case(prob, cfg, direct_rom, corr_rom, ratio_ab)
    sys = build_scattering_system_for_ratio(ratio_ab, cfg, prob);

    [u_helm_ref, info_helm_ref] = run_direct_helmholtz_reference_ecc(sys);
    helm_time = info_helm_ref.total_time;
    if strcmp(cfg.reference_mode, 'waveholtz')
        [u_wave_ref, info_wave_ref] = run_waveholtz_completion(prob, cfg, sys, zeros(size(sys.b)));
        u_ref = u_wave_ref;
    else
        u_wave_ref = u_helm_ref;
        info_wave_ref = info_helm_ref;
        u_ref = u_helm_ref;
    end
    ref_label = reference_mode_label_ecc(cfg.reference_mode);
    ref_log = format_reference_log_ecc(cfg.reference_mode, info_wave_ref);

    [u_helm_base, info_helm_baseline] = run_plain_helmholtz_gmres( ...
        sys, zeros(size(sys.b)), cfg.helm_baseline_iters, 1, cfg.helm_baseline_tol, false);

    direct_space = get_direct_basis_space(direct_rom);
    if cfg.use_direct_rom_init && ~rom_reduced_space_utils('is_empty', direct_space)
        tdir = tic;
        u_direct = solve_galerkin(sys.A, sys.b, direct_space);
        direct_time = toc(tdir);
        direct_apply_count = rom_reduced_space_utils('dimension', direct_space);
    else
        u_direct = zeros(size(sys.b));
        direct_time = 0;
        direct_apply_count = 0;
    end

    [direct_err_full, direct_err_phys] = compute_solution_errors(u_direct, u_helm_ref, sys);
    [direct_err_wave_full, direct_err_wave_phys] = compute_solution_errors(u_direct, u_wave_ref, sys);

    if cfg.run_wave_baselines
        [u_direct_fp, info_direct_fp] = run_baseline_completion_ecc(prob, cfg, sys, u_direct);
        [u_plain_fp, info_plain_fp] = run_baseline_completion_ecc(prob, cfg, sys, zeros(size(sys.b)));
        [direct_fp_err_full, direct_fp_err_phys] = compute_solution_errors(u_direct_fp, u_helm_ref, sys);
        [plain_fp_err_full, plain_fp_err_phys] = compute_solution_errors(u_plain_fp, u_helm_ref, sys);
        [direct_fp_err_wave_full, direct_fp_err_wave_phys] = compute_solution_errors(u_direct_fp, u_wave_ref, sys);
        [plain_fp_err_wave_full, plain_fp_err_wave_phys] = compute_solution_errors(u_plain_fp, u_wave_ref, sys);
    else
        u_direct_fp = zeros(size(sys.b));
        info_direct_fp = make_empty_online_info();
        u_plain_fp = zeros(size(sys.b));
        info_plain_fp = make_empty_online_info();
        direct_fp_err_full = NaN;
        direct_fp_err_phys = NaN;
        plain_fp_err_full = NaN;
        plain_fp_err_phys = NaN;
        direct_fp_err_wave_full = NaN;
        direct_fp_err_wave_phys = NaN;
        plain_fp_err_wave_full = NaN;
        plain_fp_err_wave_phys = NaN;
    end

    if cfg.use_direct_rom_init
        u_init_online = u_direct;
    else
        u_init_online = zeros(size(sys.b));
    end

    [u_precorr, info_precorr] = layered_fgmres_precorrection(sys, cfg, u_init_online, corr_rom);

    [u_online, info_online] = run_baseline_completion_ecc(prob, cfg, sys, u_precorr);
    online_relres_hist = stitch_online_relres_hist(info_precorr.relres_hist, info_online.relres_hist);
    if strcmp(cfg.online_completion_mode, 'waveholtz')
        online_lift_time_case = info_online.lift_time;
        online_completion_time_case = info_online.wave_time;
    else
        online_lift_time_case = 0;
        online_completion_time_case = info_online.total_time;
    end

    [online_err_full, online_err_phys] = compute_solution_errors(u_online, u_helm_ref, sys);
    [online_err_wave_full, online_err_wave_phys] = compute_solution_errors(u_online, u_wave_ref, sys);

    nlayers = corr_rom.num_layers;
    corr_time_layers = zeros(nlayers, 1);
    corr_u_layers = cell(nlayers, 1);
    layer_err_full = zeros(nlayers, 1);
    layer_err_phys = zeros(nlayers, 1);
    for j = 1:nlayers
        corr_time_layers(j) = info_precorr.layer_time(j);
        if ~isempty(info_precorr.layer_u{j})
            corr_u_layers{j} = info_precorr.layer_u{j};
        else
            corr_u_layers{j} = u_precorr;
        end
        [layer_err_full(j), layer_err_phys(j)] = compute_solution_errors(corr_u_layers{j}, u_ref, sys);
    end

    if strcmp(cfg.rom_check_target, 'wave')
        final_full = online_err_wave_full;
        final_phys = online_err_wave_phys;
    else
        final_full = online_err_full;
        final_phys = online_err_phys;
    end

    completion_short_label = completion_log_label_ecc(cfg.online_completion_mode);
    plain_label = plain_completion_log_label_ecc(cfg.online_completion_mode);
    direct_label = direct_completion_log_label_ecc(cfg.online_completion_mode);
    if ~cfg.run_wave_baselines
        log_line = sprintf(['test ??: r=a/b=%.4f, ref-%s=%s, helm-FGMRES=%.3fs + %s=%.3fs (%3d it, %.2e), ' ...
            'helm100=%.3fs (%3d it, %.2e), final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            ratio_ab, ref_label, ref_log, info_precorr.total_time, completion_short_label, ...
            info_online.total_time, info_online.iter_count, info_online.relres_final, ...
            info_helm_baseline.total_time, info_helm_baseline.iter_count, info_helm_baseline.relres_final, ...
            final_full, final_phys);
    elseif strcmp(cfg.online_completion_mode, 'helmholtz')
        log_line = sprintf(['test ??: r=a/b=%.4f, ref-%s=%s, ' ...
            '%s=%.3fs (%3d it, %.2e, %3d apply), %s=%.3fs total (%3d it, %.2e, %3d apply), ' ...
            'helm-FGMRES=%.3fs + plain-helm=%.3fs (%3d it, %.2e), helm100=%.3fs (%3d it, %.2e), ' ...
            'final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            ratio_ab, ref_label, ref_log, plain_label, ...
            info_plain_fp.total_time, info_plain_fp.iter_count, info_plain_fp.relres_final, info_plain_fp.op_apply_count, ...
            direct_label, ...
            direct_time + info_direct_fp.total_time, info_direct_fp.iter_count, info_direct_fp.relres_final, direct_apply_count + info_direct_fp.op_apply_count, ...
            info_precorr.total_time, info_online.total_time, info_online.iter_count, info_online.relres_final, ...
            info_helm_baseline.total_time, info_helm_baseline.iter_count, info_helm_baseline.relres_final, ...
            final_full, final_phys);
    elseif strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        log_line = sprintf(['test ??: r=a/b=%.4f, ref-%s=%s, ' ...
            '%s=%.3fs (%3d it, %.2e, %3d apply), %s=%.3fs total (%3d it, %.2e, %3d apply), ' ...
            'helm-FGMRES=%.3fs + shifted-Lap=%.3fs (%3d it, %.2e, %3d apply), ' ...
            'helm100=%.3fs (%3d it, %.2e), final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            ratio_ab, ref_label, ref_log, plain_label, ...
            info_plain_fp.total_time, info_plain_fp.iter_count, info_plain_fp.relres_final, info_plain_fp.op_apply_count, ...
            direct_label, ...
            direct_time + info_direct_fp.total_time, info_direct_fp.iter_count, info_direct_fp.relres_final, direct_apply_count + info_direct_fp.op_apply_count, ...
            info_precorr.total_time, online_completion_time_case, info_online.iter_count, info_online.relres_final, direct_apply_count + info_online.op_apply_count, ...
            info_helm_baseline.total_time, info_helm_baseline.iter_count, info_helm_baseline.relres_final, ...
            final_full, final_phys);
    else
        log_line = sprintf(['test ??: r=a/b=%.4f, ref-%s=%s, ' ...
            '%s=%.3fs (%3d it, %.2e, %3d apply), %s=%.3fs total (%3d it, %.2e, %3d apply), ' ...
            'helm-FGMRES=%.3fs + lift=%.3fs + WH=%.3fs (%3d it, %.2e, %3d apply), ' ...
            'helm100=%.3fs (%3d it, %.2e), final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            ratio_ab, ref_label, ref_log, plain_label, ...
            info_plain_fp.total_time, info_plain_fp.iter_count, info_plain_fp.relres_final, info_plain_fp.op_apply_count, ...
            direct_label, ...
            direct_time + info_direct_fp.total_time, info_direct_fp.iter_count, info_direct_fp.relres_final, direct_apply_count + info_direct_fp.op_apply_count, ...
            info_precorr.total_time, online_lift_time_case, online_completion_time_case, info_online.iter_count, info_online.relres_final, direct_apply_count + info_online.op_apply_count, ...
            info_helm_baseline.total_time, info_helm_baseline.iter_count, info_helm_baseline.relres_final, ...
            final_full, final_phys);
    end

    case_result = struct();
    case_result.ratio = ratio_ab;
    case_result.sys = sys;
    case_result.wave_u_ref = u_wave_ref;
    case_result.wave_time = info_wave_ref.total_time;
    case_result.wave_iter = info_wave_ref.iter_count;
    case_result.helm_u_ref = u_helm_ref;
    case_result.helm_time = helm_time;

    case_result.direct_u = u_direct;
    case_result.direct_time = direct_time;
    case_result.direct_op_apply = direct_apply_count;

    case_result.direct_fp_u = u_direct_fp;
    case_result.direct_fp_time = info_direct_fp.total_time;
    case_result.direct_fp_total_time = direct_time + info_direct_fp.total_time;
    case_result.direct_fp_iter = info_direct_fp.iter_count;
    case_result.direct_fp_relres_final = info_direct_fp.relres_final;
    case_result.direct_fp_op_apply = direct_apply_count + info_direct_fp.op_apply_count;
    case_result.direct_fp_relres_hist = info_direct_fp.relres_hist;
    case_result.direct_fp_step_hist = info_direct_fp.step_hist;
    case_result.direct_fp_stop_reason = info_direct_fp.stop_reason;

    case_result.plain_fp_u = u_plain_fp;
    case_result.plain_fp_time = info_plain_fp.total_time;
    case_result.plain_fp_iter = info_plain_fp.iter_count;
    case_result.plain_fp_relres_final = info_plain_fp.relres_final;
    case_result.plain_fp_op_apply = info_plain_fp.op_apply_count;
    case_result.plain_fp_relres_hist = info_plain_fp.relres_hist;
    case_result.plain_fp_step_hist = info_plain_fp.step_hist;
    case_result.plain_fp_stop_reason = info_plain_fp.stop_reason;

    case_result.online_precorr_u = u_precorr;
    case_result.online_precorr_time = info_precorr.total_time;
    case_result.online_lift_time = online_lift_time_case;
    case_result.online_precorr_matvec = info_precorr.matvec_count;
    case_result.online_precorr_relres_final = info_precorr.relres_final;
    case_result.online_precorr_relres_hist = info_precorr.relres_hist;

    case_result.online_final_u = u_online;
    case_result.online_fp_time = online_completion_time_case;
    case_result.online_total_time = info_precorr.total_time + info_online.total_time;
    case_result.online_fp_iter = info_online.iter_count;
    case_result.online_fp_relres_final = info_online.relres_final;
    case_result.online_fp_op_apply = direct_apply_count + info_online.op_apply_count;
    case_result.online_fp_relres_hist = info_online.relres_hist;
    case_result.online_fp_step_hist = info_online.step_hist;
    case_result.online_fp_stop_reason = info_online.stop_reason;
    case_result.online_combined_relres_hist = online_relres_hist;
    case_result.online_combined_transition_idx = numel(info_precorr.relres_hist);

    case_result.helm_baseline_time = info_helm_baseline.total_time;
    case_result.helm_baseline_iter = info_helm_baseline.iter_count;
    case_result.helm_baseline_op_apply = info_helm_baseline.op_apply_count;
    case_result.helm_baseline_relres_final = info_helm_baseline.relres_final;
    case_result.helm_baseline_relres_hist = info_helm_baseline.relres_hist;
    case_result.helm_baseline_u = u_helm_base;

    case_result.direct_err_wave_full = direct_err_wave_full;
    case_result.direct_fp_err_wave_full = direct_fp_err_wave_full;
    case_result.plain_fp_err_wave_full = plain_fp_err_wave_full;
    case_result.online_final_err_wave_full = online_err_wave_full;

    case_result.direct_err_wave_phys = direct_err_wave_phys;
    case_result.direct_fp_err_wave_phys = direct_fp_err_wave_phys;
    case_result.plain_fp_err_wave_phys = plain_fp_err_wave_phys;
    case_result.online_final_err_wave_phys = online_err_wave_phys;

    case_result.direct_err_helm_full = direct_err_full;
    case_result.direct_fp_err_helm_full = direct_fp_err_full;
    case_result.plain_fp_err_helm_full = plain_fp_err_full;
    case_result.online_final_err_helm_full = online_err_full;

    case_result.direct_err_helm_phys = direct_err_phys;
    case_result.direct_fp_err_helm_phys = direct_fp_err_phys;
    case_result.plain_fp_err_helm_phys = plain_fp_err_phys;
    case_result.online_final_err_helm_phys = online_err_phys;

    case_result.corr_time_layers = corr_time_layers;
    case_result.corr_u_layers = corr_u_layers;
    case_result.layer_err_helm_full = layer_err_full;
    case_result.layer_err_helm_phys = layer_err_phys;

    case_result.log_line = log_line;
end

function info = make_empty_online_info()
    info = struct();
    info.total_time = NaN;
    info.wave_time = NaN;
    info.lift_time = NaN;
    info.iter_count = NaN;
    info.relres_final = NaN;
    info.relres_hist = [];
    info.op_apply_count = NaN;
    info.step_hist = [];
    info.stop_reason = '';
    info.u_final = [];
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

function test = store_online_case_result(test, case_result, k, nlayers)
    test.systems{k} = case_result.sys;
    test.wave_u_ref(:, k) = case_result.wave_u_ref;
    test.wave_time(k) = case_result.wave_time;
    test.wave_iter(k) = case_result.wave_iter;
    test.helm_u_ref(:, k) = case_result.helm_u_ref;
    test.helm_time(k) = case_result.helm_time;

    test.direct_u(:, k) = case_result.direct_u;
    test.direct_time(k) = case_result.direct_time;
    test.direct_op_apply(k) = case_result.direct_op_apply;

    test.direct_fp_u(:, k) = case_result.direct_fp_u;
    test.direct_fp_time(k) = case_result.direct_fp_time;
    test.direct_fp_total_time(k) = case_result.direct_fp_total_time;
    test.direct_fp_iter(k) = case_result.direct_fp_iter;
    test.direct_fp_relres_final(k) = case_result.direct_fp_relres_final;
    test.direct_fp_op_apply(k) = case_result.direct_fp_op_apply;
    test.direct_fp_relres_hist{k} = case_result.direct_fp_relres_hist;
    test.direct_fp_step_hist{k} = case_result.direct_fp_step_hist;
    test.direct_fp_stop_reason{k} = case_result.direct_fp_stop_reason;

    test.plain_fp_u(:, k) = case_result.plain_fp_u;
    test.plain_fp_time(k) = case_result.plain_fp_time;
    test.plain_fp_iter(k) = case_result.plain_fp_iter;
    test.plain_fp_relres_final(k) = case_result.plain_fp_relres_final;
    test.plain_fp_op_apply(k) = case_result.plain_fp_op_apply;
    test.plain_fp_relres_hist{k} = case_result.plain_fp_relres_hist;
    test.plain_fp_step_hist{k} = case_result.plain_fp_step_hist;
    test.plain_fp_stop_reason{k} = case_result.plain_fp_stop_reason;

    test.online_precorr_u(:, k) = case_result.online_precorr_u;
    test.online_precorr_time(k) = case_result.online_precorr_time;
    test.online_lift_time(k) = case_result.online_lift_time;
    test.online_precorr_matvec(k) = case_result.online_precorr_matvec;
    test.online_precorr_relres_final(k) = case_result.online_precorr_relres_final;
    test.online_precorr_relres_hist{k} = case_result.online_precorr_relres_hist;

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

    test.direct_err_wave_full(k) = case_result.direct_err_wave_full;
    test.direct_fp_err_wave_full(k) = case_result.direct_fp_err_wave_full;
    test.plain_fp_err_wave_full(k) = case_result.plain_fp_err_wave_full;
    test.online_final_err_wave_full(k) = case_result.online_final_err_wave_full;

    test.direct_err_wave_phys(k) = case_result.direct_err_wave_phys;
    test.direct_fp_err_wave_phys(k) = case_result.direct_fp_err_wave_phys;
    test.plain_fp_err_wave_phys(k) = case_result.plain_fp_err_wave_phys;
    test.online_final_err_wave_phys(k) = case_result.online_final_err_wave_phys;

    test.direct_err_helm_full(k) = case_result.direct_err_helm_full;
    test.direct_fp_err_helm_full(k) = case_result.direct_fp_err_helm_full;
    test.plain_fp_err_helm_full(k) = case_result.plain_fp_err_helm_full;
    test.online_final_err_helm_full(k) = case_result.online_final_err_helm_full;

    test.direct_err_helm_phys(k) = case_result.direct_err_helm_phys;
    test.direct_fp_err_helm_phys(k) = case_result.direct_fp_err_helm_phys;
    test.plain_fp_err_helm_phys(k) = case_result.plain_fp_err_helm_phys;
    test.online_final_err_helm_phys(k) = case_result.online_final_err_helm_phys;

    for j = 1:nlayers
        test.corr_time_layers(j, k) = case_result.corr_time_layers(j);
        test.corr_u_layers{j}(:, k) = case_result.corr_u_layers{j};
        test.layer_err_helm_full(j, k) = case_result.layer_err_helm_full(j);
        test.layer_err_helm_phys(j, k) = case_result.layer_err_helm_phys(j);
    end
end

function summary = summarize_correction_results(~, train_ref, direct_rom, corr_rom, test, cfg)
    summary = struct();
    summary.reference_mode = cfg.reference_mode;
    summary.online_completion_mode = cfg.online_completion_mode;
    if strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        summary.shifted_laplacian_beta = cfg.shifted_laplacian_beta;
        summary.shifted_laplacian_shift_scale = cfg.shifted_laplacian_shift_scale;
        summary.shifted_laplacian_shift_strength = shifted_laplacian_shift_strength_ecc(cfg);
        summary.shifted_laplacian_restart = cfg.shifted_laplacian_restart;
        summary.shifted_laplacian_maxit = cfg.shifted_laplacian_maxit;
        summary.shifted_laplacian_tol = cfg.shifted_laplacian_tol;
    end
    summary.train_total_time = sum(train_ref.solve_time);
    summary.train_mean_time = mean(train_ref.solve_time);
    if isfield(train_ref, 'iter')
        summary.train_mean_iter = mean(train_ref.iter);
    else
        summary.train_mean_iter = NaN;
    end
    summary.direct_rank = direct_rom.pod.rank_used;
    direct_space = get_direct_basis_space(direct_rom);
    summary.direct_reduced_dim = rom_reduced_space_utils('dimension', direct_space);
    summary.direct_block_ranks = [];
    summary.direct_block_dofs = [];
    summary.direct_storage_dense = NaN;
    summary.direct_storage_block = NaN;
    summary.direct_storage_saving_ratio = NaN;
    summary.direct_storage_saved_fraction = NaN;
    if is_block3_space(direct_space) && isfield(direct_space, 'block_meta')
        bm = direct_space.block_meta;
        summary.direct_block_ranks = [bm.near.rank, bm.far.rank, bm.pml.rank];
        storage = block_storage_stats_from_space(direct_space, direct_rom.pod.rank_used);
        summary.direct_block_dofs = storage.dofs;
        summary.direct_storage_dense = storage.dense_storage;
        summary.direct_storage_block = storage.block_storage;
        summary.direct_storage_saving_ratio = storage.saving_ratio;
        summary.direct_storage_saved_fraction = storage.saved_fraction;
    end
    summary.corr_ranks = cellfun(@(layer) layer.pod.rank_used, corr_rom.layers);
    summary.corr_space_dims = zeros(corr_rom.num_layers, 1);
    summary.corr_block_ranks = nan(corr_rom.num_layers, 3);
    summary.corr_block_dofs = nan(corr_rom.num_layers, 3);
    summary.corr_storage_dense = nan(corr_rom.num_layers, 1);
    summary.corr_storage_block = nan(corr_rom.num_layers, 1);
    summary.corr_storage_saving_ratio = nan(corr_rom.num_layers, 1);
    summary.corr_storage_saved_fraction = nan(corr_rom.num_layers, 1);
    for j = 1:corr_rom.num_layers
        layer_space = get_layer_basis_space(corr_rom.layers{j});
        summary.corr_space_dims(j) = rom_reduced_space_utils('dimension', layer_space);
        if is_block3_space(layer_space) && isfield(layer_space, 'block_meta')
            bm = layer_space.block_meta;
            summary.corr_block_ranks(j, :) = [bm.near.rank, bm.far.rank, bm.pml.rank];
            storage = block_storage_stats_from_space(layer_space, corr_rom.layers{j}.pod.rank_used);
            summary.corr_block_dofs(j, :) = storage.dofs;
            summary.corr_storage_dense(j) = storage.dense_storage;
            summary.corr_storage_block(j) = storage.block_storage;
            summary.corr_storage_saving_ratio(j) = storage.saving_ratio;
            summary.corr_storage_saved_fraction(j) = storage.saved_fraction;
        end
    end
    summary.corr_num_layers = corr_rom.num_layers;
    [summary.delta_c_stats, summary.adjacent_full_stats, summary.endpoint_full] = ...
        summarize_training_parameter_variation_from_u_scattering(log(train_ref.ratio), train_ref.u_ref);
    summary.adjacent_phys_stats = summary.adjacent_full_stats;
    summary.endpoint_phys = summary.endpoint_full;
    summary.train_lift_vs_ref_state = [];
    summary.train_lift_vs_ref_u_component = [];
    [nearest_full, nearest_phys, nearest_ratio, nearest_log_gap] = nearest_train_solution_errors_ecc(train_ref, test);
    summary.nearest_train_full = stats3(nearest_full);
    summary.nearest_train_phys = stats3(nearest_phys);
    summary.nearest_train_ratio = nearest_ratio;
    summary.nearest_train_log_gap = stats3(nearest_log_gap);

    summary.direct_wave_full = stats3(test.direct_err_wave_full);
    summary.direct_wave_phys = stats3(test.direct_err_wave_phys);
    summary.direct_fp_wave_full = stats3(test.direct_fp_err_wave_full);
    summary.direct_fp_wave_phys = stats3(test.direct_fp_err_wave_phys);
    summary.plain_fp_wave_full = stats3(test.plain_fp_err_wave_full);
    summary.plain_fp_wave_phys = stats3(test.plain_fp_err_wave_phys);
    summary.final_wave_full = stats3(test.online_final_err_wave_full);
    summary.final_wave_phys = stats3(test.online_final_err_wave_phys);

    summary.direct_helm_full = stats3(test.direct_err_helm_full);
    summary.direct_helm_phys = stats3(test.direct_err_helm_phys);
    summary.direct_fp_helm_full = stats3(test.direct_fp_err_helm_full);
    summary.direct_fp_helm_phys = stats3(test.direct_fp_err_helm_phys);
    summary.plain_fp_helm_full = stats3(test.plain_fp_err_helm_full);
    summary.plain_fp_helm_phys = stats3(test.plain_fp_err_helm_phys);
    summary.final_helm_full = stats3(test.online_final_err_helm_full);
    summary.final_helm_phys = stats3(test.online_final_err_helm_phys);

    summary.direct_fp_time = stats3(test.direct_fp_time);
    summary.direct_fp_total_time = stats3(test.direct_fp_total_time);
    summary.direct_fp_iter = stats3(test.direct_fp_iter);
    summary.direct_fp_op_apply = stats3(test.direct_fp_op_apply);
    summary.direct_fp_relres = stats3(test.direct_fp_relres_final);

    summary.plain_fp_time = stats3(test.plain_fp_time);
    summary.plain_fp_iter = stats3(test.plain_fp_iter);
    summary.plain_fp_op_apply = stats3(test.plain_fp_op_apply);
    summary.plain_fp_relres = stats3(test.plain_fp_relres_final);

    summary.direct_op_apply = stats3(test.direct_op_apply);
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
    summary.wave_vs_helm_full = stats3(vec_solution_compare_scattering(test.wave_u_ref, test.helm_u_ref, test, 'full'));
    summary.wave_vs_helm_phys = stats3(vec_solution_compare_scattering(test.wave_u_ref, test.helm_u_ref, test, 'phys'));

    summary.train_gmres_relres_by_layer = zeros(corr_rom.num_layers, 1);
    for j = 1:corr_rom.num_layers
        summary.train_gmres_relres_by_layer(j) = mean(corr_rom.layers{j}.train_fgmres_relres);
    end

    if isempty(summary.plain_fp_time) || ~isfinite(summary.plain_fp_time(2))
        summary.time_speedup_online_vs_plain_mean = NaN;
    else
        summary.time_speedup_online_vs_plain_mean = summary.plain_fp_time(2) / max(summary.online_total_time(2), 1e-16);
    end
    if isempty(summary.direct_fp_total_time) || ~isfinite(summary.direct_fp_total_time(2))
        summary.time_speedup_online_vs_direct_wave_mean = NaN;
    else
        summary.time_speedup_online_vs_direct_wave_mean = summary.direct_fp_total_time(2) / max(summary.online_total_time(2), 1e-16);
    end
    summary.time_speedup_online_vs_helm100_mean = summary.helm_baseline_time(2) / max(summary.online_total_time(2), 1e-16);

    if strcmp(cfg.online_completion_mode, 'helmholtz')
        summary.iter_reduction_online_vs_plain_mean = summary.helm_baseline_iter(2) - summary.online_fp_iter(2);
        summary.iter_ratio_online_vs_plain_mean = summary.online_fp_iter(2) / max(summary.helm_baseline_iter(2), 1e-16);
        summary.op_apply_ratio_online_vs_plain_mean = summary.online_fp_op_apply(2) / max(summary.helm_baseline_op_apply(2), 1e-16);
    elseif ~isempty(summary.plain_fp_iter) && ~isempty(summary.plain_fp_op_apply)
        summary.iter_reduction_online_vs_plain_mean = summary.plain_fp_iter(2) - summary.online_fp_iter(2);
        summary.iter_ratio_online_vs_plain_mean = summary.online_fp_iter(2) / max(summary.plain_fp_iter(2), 1e-16);
        summary.op_apply_ratio_online_vs_plain_mean = summary.online_fp_op_apply(2) / max(summary.plain_fp_op_apply(2), 1e-16);
    else
        summary.iter_reduction_online_vs_plain_mean = NaN;
        summary.iter_ratio_online_vs_plain_mean = NaN;
        summary.op_apply_ratio_online_vs_plain_mean = NaN;
    end

    summary.primary_target = upper(cfg.rom_check_target);
    summary.primary_error_part = 'FULL';
    if strcmp(cfg.rom_check_target, 'wave')
        final_primary_curve = test.online_final_err_wave_full;
    else
        final_primary_curve = test.online_final_err_helm_full;
    end
    if ~isempty(cfg.plot_test_index)
        if ~(isscalar(cfg.plot_test_index) && cfg.plot_test_index >= 1 && ...
                cfg.plot_test_index <= numel(final_primary_curve) && ...
                abs(cfg.plot_test_index - round(cfg.plot_test_index)) < 1e-12)
            error('plot_test_index must be empty or an integer in [1,%d].', numel(final_primary_curve));
        end
        summary.representative_test_index = round(cfg.plot_test_index);
    else
        [~, summary.representative_test_index] = max(final_primary_curve);
    end
    summary.representative_ratio = test.ratio(summary.representative_test_index);
end

function print_correction_summary(summary)
    completion_label = completion_display_label_ecc(summary.online_completion_mode);
    plain_label = plain_completion_display_label_ecc(summary.online_completion_mode);
    if strcmp(summary.reference_mode, 'helmholtz')
        wave_label = 'reference';
        wave_helm_label = 'reference-vs-Helm';
    else
        wave_label = 'WH';
        wave_helm_label = 'WH-vs-Helm';
    end
    if strcmp(summary.online_completion_mode, 'helmholtz')
        compare_label = 'helm100';
    else
        compare_label = plain_label;
    end

    fprintf('\n=== Summary ===\n');
    fprintf('reference mode            : %s\n', summary.reference_mode);
    fprintf('completion mode           : %s\n', summary.online_completion_mode);
    if isfield(summary, 'shifted_laplacian_beta')
        fprintf('shifted-Laplacian beta/scale/strength: %.3g / %s / %.3g\n', ...
            summary.shifted_laplacian_beta, summary.shifted_laplacian_shift_scale, ...
            summary.shifted_laplacian_shift_strength);
        fprintf('shifted-Laplacian restart/maxit/tol: %d / %d / %.1e\n', ...
            summary.shifted_laplacian_restart, ...
            summary.shifted_laplacian_maxit, summary.shifted_laplacian_tol);
    end
    fprintf('train total / mean time   : %.3fs / %.3fs\n', summary.train_total_time, summary.train_mean_time);
    fprintf('mean train iters          : %.2f\n', summary.train_mean_iter);
    fprintf('direct ROM rank           : %d\n', summary.direct_rank);
    fprintf('direct reduced dim        : %d\n', summary.direct_reduced_dim);
    if ~isempty(summary.direct_block_ranks)
        fprintf('direct block ranks [near far pml] : %s\n', mat2str(summary.direct_block_ranks));
        fprintf('direct block dofs  [near far pml] : %s\n', mat2str(summary.direct_block_dofs));
        fprintf('direct storage saving     : %.3fx (%.1f%% less)\n', ...
            summary.direct_storage_saving_ratio, 100 * summary.direct_storage_saved_fraction);
    end
    fprintf('layered Helmholtz ROM steps: %d\n', summary.corr_num_layers);
    if ~isempty(summary.corr_ranks)
        fprintf('layered ROM ranks         : %s\n', mat2str(summary.corr_ranks));
    end
    if ~isempty(summary.corr_space_dims)
        fprintf('layered reduced dims      : %s\n', mat2str(summary.corr_space_dims.'));
    end
    if ~isempty(summary.corr_block_ranks) && any(isfinite(summary.corr_block_ranks(:)))
        for j = 1:size(summary.corr_block_ranks, 1)
            if all(isfinite(summary.corr_block_ranks(j, :)))
                fprintf(['  layer %d block ranks [near far pml] : %s, dofs = %s, ' ...
                    'storage saving = %.3fx (%.1f%% less)\n'], ...
                    j, mat2str(summary.corr_block_ranks(j, :)), ...
                    mat2str(summary.corr_block_dofs(j, :)), ...
                    summary.corr_storage_saving_ratio(j), ...
                    100 * summary.corr_storage_saved_fraction(j));
            end
        end
    end
    if ~isempty(summary.train_gmres_relres_by_layer)
        fprintf('train layered relres mean : %s\n', mat2str(summary.train_gmres_relres_by_layer, 4));
    end
    if ~isempty(summary.delta_c_stats)
        fprintf('adjacent train delta log(a/b): %.3e / %.3e / %.3e\n', summary.delta_c_stats(1), summary.delta_c_stats(2), summary.delta_c_stats(3));
        fprintf('adjacent train full diff  : %.3e / %.3e / %.3e\n', summary.adjacent_full_stats(1), summary.adjacent_full_stats(2), summary.adjacent_full_stats(3));
        fprintf('endpoint train full diff  : %.3e\n', summary.endpoint_full);
    end
    if ~isempty(summary.nearest_train_full)
        fprintf('nearest train-vs-test full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.nearest_train_full(1), summary.nearest_train_full(2), summary.nearest_train_full(3));
        fprintf('nearest train-vs-test phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.nearest_train_phys(1), summary.nearest_train_phys(2), summary.nearest_train_phys(3));
        fprintf('nearest train log(a/b) gap min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.nearest_train_log_gap(1), summary.nearest_train_log_gap(2), summary.nearest_train_log_gap(3));
    end
    if ~isempty(summary.plain_fp_wave_full)
        fprintf('plain completion-vs-%s full relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_label, ...
            summary.plain_fp_wave_full(1), summary.plain_fp_wave_full(2), summary.plain_fp_wave_full(3));
        fprintf('plain completion-vs-%s phys relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_label, ...
            summary.plain_fp_wave_phys(1), summary.plain_fp_wave_phys(2), summary.plain_fp_wave_phys(3));
    end
    if ~isempty(summary.direct_fp_wave_full)
        fprintf('direct+completion-vs-%s full relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_label, ...
            summary.direct_fp_wave_full(1), summary.direct_fp_wave_full(2), summary.direct_fp_wave_full(3));
        fprintf('direct+completion-vs-%s phys relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_label, ...
            summary.direct_fp_wave_phys(1), summary.direct_fp_wave_phys(2), summary.direct_fp_wave_phys(3));
    end
    fprintf('final layered-ROM-vs-%s full relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_label, ...
        summary.final_wave_full(1), summary.final_wave_full(2), summary.final_wave_full(3));
    fprintf('final layered-ROM-vs-%s phys relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_label, ...
        summary.final_wave_phys(1), summary.final_wave_phys(2), summary.final_wave_phys(3));

    fprintf('final layered-ROM-vs-Helm full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.final_helm_full(1), summary.final_helm_full(2), summary.final_helm_full(3));
    fprintf('final layered-ROM-vs-Helm phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.final_helm_phys(1), summary.final_helm_phys(2), summary.final_helm_phys(3));
    if ~isempty(summary.plain_fp_helm_full)
        fprintf('plain FP-vs-Helm full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.plain_fp_helm_full(1), summary.plain_fp_helm_full(2), summary.plain_fp_helm_full(3));
        fprintf('plain FP-vs-Helm phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.plain_fp_helm_phys(1), summary.plain_fp_helm_phys(2), summary.plain_fp_helm_phys(3));
    end
    if ~isempty(summary.direct_fp_helm_full)
        fprintf('direct+FP-vs-Helm full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.direct_fp_helm_full(1), summary.direct_fp_helm_full(2), summary.direct_fp_helm_full(3));
        fprintf('direct+FP-vs-Helm phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.direct_fp_helm_phys(1), summary.direct_fp_helm_phys(2), summary.direct_fp_helm_phys(3));
    end
    if ~isempty(summary.wave_vs_helm_full)
        fprintf('%s full relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_helm_label, ...
            summary.wave_vs_helm_full(1), summary.wave_vs_helm_full(2), summary.wave_vs_helm_full(3));
        fprintf('%s phys relerr min/mean/max = %.3e / %.3e / %.3e\n', wave_helm_label, ...
            summary.wave_vs_helm_phys(1), summary.wave_vs_helm_phys(2), summary.wave_vs_helm_phys(3));
    end

    if ~isempty(summary.plain_fp_time)
        fprintf('%s time min/mean/max: %.3fs / %.3fs / %.3fs\n', plain_label, ...
            summary.plain_fp_time(1), summary.plain_fp_time(2), summary.plain_fp_time(3));
        fprintf('%s iter min/mean/max: %.0f / %.2f / %.0f\n', plain_label, ...
            summary.plain_fp_iter(1), summary.plain_fp_iter(2), summary.plain_fp_iter(3));
        fprintf('%s apply min/mean/max: %.0f / %.2f / %.0f\n', plain_label, ...
            summary.plain_fp_op_apply(1), summary.plain_fp_op_apply(2), summary.plain_fp_op_apply(3));
        fprintf('%s relres min/mean/max = %.3e / %.3e / %.3e\n', plain_label, ...
            summary.plain_fp_relres(1), summary.plain_fp_relres(2), summary.plain_fp_relres(3));
    end
    if ~isempty(summary.direct_fp_total_time)
        fprintf('direct+%s total time min/mean/max: %.3fs / %.3fs / %.3fs\n', plain_label, ...
            summary.direct_fp_total_time(1), summary.direct_fp_total_time(2), summary.direct_fp_total_time(3));
        fprintf('direct+%s iter min/mean/max: %.0f / %.2f / %.0f\n', plain_label, ...
            summary.direct_fp_iter(1), summary.direct_fp_iter(2), summary.direct_fp_iter(3));
        fprintf('direct+%s apply min/mean/max: %.0f / %.2f / %.0f\n', plain_label, ...
            summary.direct_fp_op_apply(1), summary.direct_fp_op_apply(2), summary.direct_fp_op_apply(3));
        fprintf('direct+%s relres min/mean/max = %.3e / %.3e / %.3e\n', plain_label, ...
            summary.direct_fp_relres(1), summary.direct_fp_relres(2), summary.direct_fp_relres(3));
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
    fprintf('direct ROM apply min/mean/max: %.0f / %.2f / %.0f\n', ...
        summary.direct_op_apply(1), summary.direct_op_apply(2), summary.direct_op_apply(3));

    if isfinite(summary.time_speedup_online_vs_plain_mean)
        fprintf('mean time speedup %s/online = %.3fx\n', plain_label, summary.time_speedup_online_vs_plain_mean);
    end
    if isfinite(summary.time_speedup_online_vs_direct_wave_mean)
        fprintf('mean time speedup direct+%s/online = %.3fx\n', plain_label, summary.time_speedup_online_vs_direct_wave_mean);
    end
    fprintf('mean time speedup helm100/online = %.3fx\n', summary.time_speedup_online_vs_helm100_mean);
    fprintf('mean iter change %s-online = %.2f (ratio %.3f)\n', compare_label, ...
        summary.iter_reduction_online_vs_plain_mean, summary.iter_ratio_online_vs_plain_mean);
    fprintf('mean apply ratio online/%s = %.3f\n', compare_label, summary.op_apply_ratio_online_vs_plain_mean);
    fprintf('ROM check target          : %s\n', summary.primary_target);
    fprintf('primary error field       : %s\n', summary.primary_error_part);
    fprintf('representative ratio a/b    = %.4f\n', summary.representative_ratio);
end

function save_benchmark_summary_ecc(cfg, summary)
    out_dir = cfg.benchmark_root;
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    name = benchmark_file_stem_ecc(cfg, 'ecc_correction_rom');
    benchmark_table = struct2table(benchmark_row_from_summary_ecc(summary));
    csv_file = fullfile(out_dir, [name '_benchmark.csv']);
    mat_file = fullfile(out_dir, [name '_benchmark.mat']);
    writetable(benchmark_table, csv_file);
    save(mat_file, 'benchmark_table', 'summary', 'cfg');
    fprintf('saved benchmark: %s\n', csv_file);
    fprintf('saved benchmark: %s\n', mat_file);
end

function row = benchmark_row_from_summary_ecc(summary)
    row = struct();
    row.experiment = {'ecc_correction_rom'};
    row.method = {'domain-decomposed trajectory-aware'};
    row.parameter_ratio = summary_parameter_ecc(summary);
    row.train_total_time = safe_scalar_ecc(summary, 'train_total_time');
    row.train_mean_time = safe_scalar_ecc(summary, 'train_mean_time');
    row.train_mean_iter = safe_scalar_ecc(summary, 'train_mean_iter');
    row.direct_rank = safe_scalar_ecc(summary, 'direct_rank');
    row.direct_reduced_dim = safe_scalar_ecc(summary, 'direct_reduced_dim');
    row.direct_storage_saving_ratio = safe_scalar_ecc(summary, 'direct_storage_saving_ratio');
    row.direct_storage_saved_fraction = safe_scalar_ecc(summary, 'direct_storage_saved_fraction');
    row.mean_correction_rank = safe_mean_ecc(summary, 'corr_ranks');
    row.max_correction_rank = safe_max_ecc(summary, 'corr_ranks');
    row.plain_waveholtz_time_mean = safe_stat_ecc(summary, 'plain_fp_time', 2);
    row.plain_waveholtz_iter_mean = safe_stat_ecc(summary, 'plain_fp_iter', 2);
    row.plain_waveholtz_relres_mean = safe_stat_ecc(summary, 'plain_fp_relres', 2);
    row.direct_waveholtz_total_time_mean = safe_stat_ecc(summary, 'direct_fp_total_time', 2);
    row.direct_waveholtz_iter_mean = safe_stat_ecc(summary, 'direct_fp_iter', 2);
    row.direct_waveholtz_relres_mean = safe_stat_ecc(summary, 'direct_fp_relres', 2);
    row.plain_helm_gmres_time_mean = safe_stat_ecc(summary, 'helm_baseline_time', 2);
    row.plain_helm_gmres_iter_mean = safe_stat_ecc(summary, 'helm_baseline_iter', 2);
    row.plain_helm_gmres_relres_mean = safe_stat_ecc(summary, 'helm_baseline_relres', 2);
    row.precorrection_time_mean = safe_stat_ecc(summary, 'online_precorr_time', 2);
    row.lift_time_mean = safe_stat_ecc(summary, 'online_lift_time', 2);
    row.completion_time_mean = safe_stat_ecc(summary, 'online_fp_time', 2);
    row.online_total_time_mean = safe_stat_ecc(summary, 'online_total_time', 2);
    row.completion_iter_mean = safe_stat_ecc(summary, 'online_fp_iter', 2);
    row.completion_relres_mean = safe_stat_ecc(summary, 'online_fp_relres', 2);
    row.speedup_vs_plain_waveholtz = safe_scalar_ecc(summary, 'time_speedup_online_vs_plain_mean');
    row.speedup_vs_direct_waveholtz = safe_scalar_ecc(summary, 'time_speedup_online_vs_direct_wave_mean');
    row.speedup_vs_plain_helm_gmres = safe_scalar_ecc(summary, 'time_speedup_online_vs_helm100_mean');
    row.iter_ratio_vs_baseline = safe_scalar_ecc(summary, 'iter_ratio_online_vs_plain_mean');
    row.op_apply_ratio_vs_baseline = safe_scalar_ecc(summary, 'op_apply_ratio_online_vs_plain_mean');
    row.final_wave_full_error_mean = safe_stat_ecc(summary, 'final_wave_full', 2);
    row.final_helm_full_error_mean = safe_stat_ecc(summary, 'final_helm_full', 2);
    row.wave_vs_helm_full_error_mean = safe_stat_ecc(summary, 'wave_vs_helm_full', 2);
end

function val = summary_parameter_ecc(summary)
    if isfield(summary, 'representative_ratio')
        val = summary.representative_ratio;
    else
        val = NaN;
    end
end

function val = safe_stat_ecc(summary, field_name, idx)
    val = NaN;
    if isfield(summary, field_name)
        data = summary.(field_name);
        if numel(data) >= idx
            val = data(idx);
        end
    end
end

function val = safe_scalar_ecc(summary, field_name)
    val = NaN;
    if isfield(summary, field_name) && ~isempty(summary.(field_name))
        data = summary.(field_name);
        if isscalar(data)
            val = data;
        end
    end
end

function val = safe_mean_ecc(summary, field_name)
    val = NaN;
    if isfield(summary, field_name) && ~isempty(summary.(field_name))
        data = summary.(field_name);
        data = data(isfinite(data));
        if ~isempty(data)
            val = mean(data);
        end
    end
end

function val = safe_max_ecc(summary, field_name)
    val = NaN;
    if isfield(summary, field_name) && ~isempty(summary.(field_name))
        data = summary.(field_name);
        data = data(isfinite(data));
        if ~isempty(data)
            val = max(data);
        end
    end
end

function stem = benchmark_file_stem_ecc(cfg, fallback)
    if isfield(cfg, 'benchmark_name') && ~isempty(cfg.benchmark_name)
        stem = sanitize_filename(char(cfg.benchmark_name));
    elseif isfield(cfg, 'figure_subdir') && ~isempty(cfg.figure_subdir)
        stem = sanitize_filename(char(cfg.figure_subdir));
    else
        stem = sanitize_filename(fallback);
    end
end

function make_correction_plots(cfg, prob, direct_rom, corr_rom, test, summary)
    fig_dir = prepare_figure_dir(cfg);
    rep_idx = summary.representative_test_index;
    tag = sanitize_filename(sprintf('ratio_%0.4f', summary.representative_ratio));

    direct_sv = direct_rom.pod.singular_values;
    if cfg.use_direct_rom_init && has_finite_data(direct_sv)
        fig = figure('Name', 'ECC direct ROM singular values', 'Position', [100 100 620 460]);
        ax = axes(fig);
        semilogy(ax, 1:numel(direct_sv), clamp_positive_curve(direct_sv, 1e-16), ...
            'o-', 'LineWidth', 1.7, 'MarkerSize', 6, 'DisplayName', 'direct ROM');
        style_publication_axes(ax);
        set(ax, 'YScale', 'log');
        xlabel(ax, 'mode index');
        ylabel(ax, 'singular value');
        title(ax, 'Scattering direct ROM singular values');
        legend(ax, 'Location', 'best');
        save_publication_figure(fig, fig_dir, ['ecc_direct_rom_singular_values_' tag], cfg);
    end

    if corr_rom.num_layers > 0
        fig = figure('Name', 'ECC correction ROM singular values', 'Position', [100 100 680 480]);
        ax = axes(fig);
        hold(ax, 'on');
        for j = 1:corr_rom.num_layers
            corr_sv = corr_rom.layers{j}.pod.singular_values;
            if has_finite_data(corr_sv)
                semilogy(ax, 1:numel(corr_sv), clamp_positive_curve(corr_sv, 1e-16), ...
                    'o-', 'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', sprintf('layer %d', j));
            end
        end
        hold(ax, 'off');
        style_publication_axes(ax);
        set(ax, 'YScale', 'log');
        xlabel(ax, 'mode index');
        ylabel(ax, 'singular value');
        title(ax, 'Scattering correction singular-value decay');
        legend(ax, 'Location', 'best');
        save_publication_figure(fig, fig_dir, ['ecc_correction_rom_singular_values_' tag], cfg);
    end

    plot_full_field_error_curves_ecc(cfg, test, summary, fig_dir, tag);
    plot_joint_residual_decay_ecc(test, summary, fig_dir, tag, cfg);

    if strcmp(cfg.rom_check_target, 'wave')
        target_u = test.wave_u_ref(:, rep_idx);
    else
        target_u = test.helm_u_ref(:, rep_idx);
    end
    direct_u = test.direct_u(:, rep_idx);
    corr_u = test.online_final_u(:, rep_idx);

    target_grid = embed_interior_field(target_u, prob.nx, prob.ny);
    direct_grid = embed_interior_field(direct_u, prob.nx, prob.ny);
    corr_grid = embed_interior_field(corr_u, prob.nx, prob.ny);
    err_grid = abs(corr_grid - target_grid);
    rhs_grid = embed_interior_field(test.systems{rep_idx}.b, prob.nx, prob.ny);
    obstacle_mask = [];
    if cfg.plot_obstacle_white && isfield(test.systems{rep_idx}, 'geom') && ...
            isfield(test.systems{rep_idx}.geom, 'internal')
        obstacle_mask = test.systems{rep_idx}.geom.internal;
    end

    prefix = ['ecc_' tag];
    plot_full_field_component_set(prob, cfg, fig_dir, prefix, 'real', @real, ...
        target_grid, direct_grid, corr_grid, err_grid, rhs_grid, cfg.use_direct_rom_init, obstacle_mask);
    plot_full_field_component_set(prob, cfg, fig_dir, prefix, 'imag', @imag, ...
        target_grid, direct_grid, corr_grid, err_grid, rhs_grid, cfg.use_direct_rom_init, obstacle_mask);

    if is_block3_space(get_direct_basis_space(direct_rom)) || any(cellfun(@(ly) is_block3_space(get_layer_basis_space(ly)), corr_rom.layers))
        plot_blockwise_singular_decay(direct_rom, corr_rom, fig_dir, cfg);
    end
end

function plot_full_field_error_curves_ecc(cfg, test, summary, fig_dir, tag)
    if strcmp(cfg.rom_check_target, 'wave')
        direct_err = test.direct_err_wave_full;
        direct_fp_err = test.direct_fp_err_wave_full;
        final_err = test.online_final_err_wave_full;
    else
        direct_err = test.direct_err_helm_full;
        direct_fp_err = test.direct_fp_err_helm_full;
        final_err = test.online_final_err_helm_full;
    end

    fig = figure('Name', 'ECC full-field relative errors', 'Position', [120 120 700 480]);
    ax = axes(fig);
    hold(ax, 'on');
    x = test.ratio(:);
    if cfg.use_direct_rom_init && has_finite_data(direct_err)
        semilogy(ax, x, clamp_positive_curve(direct_err, 1e-16), 'o-', ...
            'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'direct ROM');
    end
    if has_finite_data(direct_fp_err)
        semilogy(ax, x, clamp_positive_curve(direct_fp_err, 1e-16), 's-', ...
            'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'direct ROM + completion');
    end
    semilogy(ax, x, clamp_positive_curve(final_err, 1e-16), 'd-', ...
        'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'trajectory correction + completion');
    hold(ax, 'off');
    style_publication_axes(ax);
    set(ax, 'YScale', 'log');
    xlabel(ax, 'test shape ratio a/b');
    ylabel(ax, 'relative full-field error');
    title(ax, 'Scattering full-field error');
    legend(ax, 'Location', 'best');
    save_publication_figure(fig, fig_dir, ['ecc_full_field_error_curves_' tag], cfg);
end

function plot_joint_residual_decay_ecc(test, summary, fig_dir, tag, cfg)
    rep_idx = summary.representative_test_index;
    plot_floor = 1e-16;
    fig = figure('Name', 'ECC joint residual decay', 'Position', [140 140 760 500]);
    ax = axes(fig);
    hold(ax, 'on');

    online_combined_hist = test.online_combined_relres_hist{rep_idx};
    if has_finite_data(online_combined_hist)
        semilogy(ax, 0:(numel(online_combined_hist) - 1), sanitize_log_curve(online_combined_hist, plot_floor), ...
            '-o', 'Color', [0.85 0.05 0.05], 'LineWidth', 1.8, 'MarkerSize', 6, ...
            'DisplayName', 'TA-ROM + completion');
    end
    transition_idx = test.online_combined_transition_idx(rep_idx);
    if transition_idx >= 1 && transition_idx <= numel(online_combined_hist)
        semilogy(ax, transition_idx - 1, sanitize_log_curve(online_combined_hist(transition_idx), plot_floor), ...
            's', 'Color', [0.85 0.1 0.1], 'MarkerFaceColor', 'w', 'MarkerSize', 9, ...
            'DisplayName', 'completion transition');
    end
    if has_finite_data(test.direct_fp_relres_hist{rep_idx})
        hist_direct = test.direct_fp_relres_hist{rep_idx};
        semilogy(ax, 0:(numel(hist_direct) - 1), sanitize_log_curve(hist_direct, plot_floor), ...
            '-o', 'Color', [0 0.35 1], 'LineWidth', 1.3, 'MarkerSize', 5, ...
            'DisplayName', 'direct ROM + completion');
    end
    if has_finite_data(test.plain_fp_relres_hist{rep_idx})
        hist_plain = test.plain_fp_relres_hist{rep_idx};
        semilogy(ax, 0:(numel(hist_plain) - 1), sanitize_log_curve(hist_plain, plot_floor), ...
            'k-o', 'LineWidth', 1.3, 'MarkerSize', 5, ...
            'DisplayName', plain_completion_display_label_ecc(cfg.online_completion_mode));
    end
    if has_finite_data(test.helm_baseline_relres_hist{rep_idx})
        hist_helm = test.helm_baseline_relres_hist{rep_idx};
        semilogy(ax, 0:(numel(hist_helm) - 1), sanitize_log_curve(hist_helm, plot_floor), ...
            '-o', 'Color', [0.55 0 0.65], 'LineWidth', 1.3, 'MarkerSize', 5, ...
            'DisplayName', 'plain Helmholtz GMRES');
    end
    hold(ax, 'off');
    style_publication_axes(ax);
    set(ax, 'YScale', 'log');
    ylim(ax, [plot_floor, 2]);
    xlabel(ax, 'online step index');
    ylabel(ax, 'relative residual');
    title(ax, sprintf('Scattering residual decay, a/b=%.4f', summary.representative_ratio));
    legend(ax, 'Location', 'best');
    save_publication_figure(fig, fig_dir, ['ecc_joint_combined_residual_decay_full_field_' tag], cfg);
end

function plot_full_field_component_set(prob, cfg, fig_dir, prefix, component_name, component_fun, ...
        target_grid, direct_grid, corr_grid, err_grid, rhs_grid, include_direct, obstacle_mask)
    target_data = component_fun(target_grid);
    corr_data = component_fun(corr_grid);
    direct_data = component_fun(direct_grid);

    plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_reference_full_field'], ...
        target_data, sprintf('Reference %s part', component_name), component_name, obstacle_mask);
    if include_direct && has_finite_data(direct_data) && norm(direct_data(:)) > 0
        plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_direct_rom_full_field'], ...
            direct_data, sprintf('Direct ROM %s part', component_name), component_name, obstacle_mask);
    end
    plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_joint_correction_full_field'], ...
        corr_data, sprintf('TA-ROM %s part', component_name), component_name, obstacle_mask);
    plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_joint_abs_diff_full_field'], ...
        err_grid, sprintf('Absolute error, %s part', component_name), 'absolute difference', obstacle_mask);
    if strcmp(component_name, 'real')
        plot_full_field_image(prob, cfg, fig_dir, [prefix '_source_full_field'], ...
            abs(rhs_grid), 'Input source magnitude', '|b|', obstacle_mask);
    end
end

function plot_full_field_image(prob, cfg, fig_dir, file_base, data_grid, ttl, cbar_label, obstacle_mask)
    if nargin < 8
        obstacle_mask = [];
    end
    fig = figure('Name', ttl, 'Position', [160 160 620 520]);
    ax = axes(fig);
    cdata = data_grid(2:end-1, 2:end-1).';
    alpha_data = true(size(cdata));
    if ~isempty(obstacle_mask)
        interior_mask = obstacle_mask(2:end-1, 2:end-1).';
        cdata(interior_mask) = NaN;
        alpha_data(interior_mask) = false;
    end
    h = imagesc(ax, prob.x_int, prob.y_int, cdata);
    if ~isempty(obstacle_mask)
        set(h, 'AlphaData', double(alpha_data));
        set(ax, 'Color', 'w');
    end
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal');
    axis(ax, 'tight');
    if isfield(cfg, 'plot_y_limits') && ~isempty(cfg.plot_y_limits)
        ylim(ax, cfg.plot_y_limits);
    end
    style_publication_axes(ax);
    grid(ax, 'off');
    xlabel(ax, 'x');
    ylabel(ax, 'y');
    title(ax, ttl, 'Interpreter', 'none');
    if any(strcmp(cbar_label, {'real','imag'}))
        colormap(ax, helm_solution_colormap(clim(ax)));
    end
    cb = colorbar(ax);
    ylabel(cb, cbar_label, 'Interpreter', 'none');
    save_publication_figure(fig, fig_dir, file_base, cfg);
end

function fig_dir = prepare_figure_dir(cfg)
    fig_dir = fullfile(cfg.figure_root, cfg.figure_subdir);
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end
end

function save_publication_figure(fig, fig_dir, file_base, cfg)
    if ~cfg.save_plots
        return;
    end
    set(fig, 'Color', 'w');
    file_base = sanitize_filename(file_base);
    out_file = fullfile(fig_dir, [file_base '.' char(cfg.figure_format)]);
    try
        exportgraphics(fig, out_file, 'Resolution', 300);
    catch
        saveas(fig, out_file);
    end
    fprintf('saved figure: %s\n', out_file);
end

function style_publication_axes(ax)
    set(ax, 'FontSize', 12, 'LineWidth', 1.0, 'Box', 'on');
    grid(ax, 'on');
end

function y = sanitize_log_curve(curve, floor_value)
    y = curve(:).';
    bad = ~isfinite(y) | y <= 0;
    y(bad) = NaN;
    y = max(y, floor_value);
end

function tf = has_finite_data(vals)
    tf = ~isempty(vals) && any(isfinite(vals(:)));
end

function name = sanitize_filename(name)
    name = lower(regexprep(name, '[^a-zA-Z0-9]+', '_'));
    name = regexprep(name, '^_+|_+$', '');
end

function plot_blockwise_singular_decay(direct_rom, corr_rom, fig_dir, cfg)
    direct_space = get_direct_basis_space(direct_rom);
    if is_block3_space(direct_space)
        plot_block3_partition_schematic(direct_space, fig_dir, cfg);
        fig = figure('Name', 'ECC direct blockwise singular decay', 'Position', [120, 120, 660, 480]);
        ax = axes(fig);
        plot_one_block3_space(ax, direct_space, 'Direct blockwise singular value decay');
        save_publication_figure(fig, fig_dir, 'ecc_direct_blockwise_singular_decay', cfg);
    else
        for j = 1:corr_rom.num_layers
            layer_space = get_layer_basis_space(corr_rom.layers{j});
            if is_block3_space(layer_space)
                plot_block3_partition_schematic(layer_space, fig_dir, cfg);
                break;
            end
        end
    end

    for j = 1:corr_rom.num_layers
        layer_space = get_layer_basis_space(corr_rom.layers{j});
        if is_block3_space(layer_space)
            fig = figure('Name', sprintf('ECC layer %d blockwise singular decay', j), 'Position', [120, 120, 660, 480]);
            ax = axes(fig);
            plot_one_block3_space(ax, layer_space, sprintf('Layer %d blockwise singular value decay', j));
            save_publication_figure(fig, fig_dir, sprintf('ecc_layer_%02d_blockwise_singular_decay', j), cfg);
        end
    end
end

function plot_block3_partition_schematic(space, fig_dir, cfg)
    if ~is_block3_space(space) || ~isfield(space, 'partition')
        return;
    end
    part = space.partition;
    labels = zeros(part.n, 1);
    labels(part.idx_far) = 1;
    labels(part.idx_near) = 2;
    labels(part.idx_pml) = 3;
    label_grid = reshape(labels, part.nx, part.ny).';

    fig = figure('Name', 'ECC near-far-PML partition', 'Position', [120, 120, 620, 500]);
    ax = axes(fig);
    imagesc(ax, label_grid);
    axis(ax, 'image');
    set(ax, 'YDir', 'normal');
    colors = dd_partition_colormap(3);
    colormap(ax, [0.90 0.90 0.90; colors([2 1 3], :)]);
    caxis(ax, [0 3]);
    cb = colorbar(ax, 'Ticks', [0.375, 1.125, 1.875, 2.625], ...
        'TickLabels', {'inactive', 'far', 'near', 'PML'});
    ylabel(cb, 'block');
    xlabel(ax, 'x grid index');
    ylabel(ax, 'y grid index');
    title(ax, 'ECC domain-decomposed ROM partition');
    box(ax, 'on');
    save_publication_figure(fig, fig_dir, 'ecc_domain_decomposed_partition', cfg);
end

function plot_one_block3_space(ax, space, ttl)
    if ~is_block3_space(space) || ~isfield(space, 'block_meta')
        return;
    end

    hold(ax, 'on');
    plot_one_block_curve(ax, space.block_meta.near.singular_values, '-', [0.85 0.20 0.15], ...
        sprintf('near (r=%d,n=%d)', space.block_meta.near.rank, space.block_meta.near.nrow));
    plot_one_block_curve(ax, space.block_meta.far.singular_values, '--', [0.20 0.45 0.95], ...
        sprintf('far (r=%d,n=%d)', space.block_meta.far.rank, space.block_meta.far.nrow));
    plot_one_block_curve(ax, space.block_meta.pml.singular_values, ':', [0.15 0.65 0.25], ...
        sprintf('pml (r=%d,n=%d)', space.block_meta.pml.rank, space.block_meta.pml.nrow));
    hold(ax, 'off');

    style_publication_axes(ax);
    set(ax, 'YScale', 'log');
    xlabel(ax, 'mode index');
    ylabel(ax, '\sigma_i/\sigma_1');
    title(ax, ttl);
    legend(ax, 'Location', 'best', 'Interpreter', 'none');
end

function plot_one_block_curve(ax, sv, line_style, color, label_text)
    if isempty(sv)
        return;
    end
    denom = max(abs(sv(1)), 1e-30);
    yy = clamp_positive_curve(abs(sv(:)) / denom, 1e-16);
    semilogy(ax, 1:numel(yy), yy, line_style, 'Color', color, 'LineWidth', 1.5, 'DisplayName', label_text);
end

function tf = is_block3_space(space)
    tf = isstruct(space) && isfield(space, 'type') && strcmp(space.type, 'block3');
end

function stats = block_storage_stats_from_space(space, dense_rank)
    stats = [];
    if ~is_block3_space(space) || ~isfield(space, 'block_meta')
        return;
    end

    bm = space.block_meta;
    ranks = [bm.near.rank, bm.far.rank, bm.pml.rank];
    dofs = [bm.near.nrow, bm.far.nrow, bm.pml.nrow];
    dense_storage = dense_rank * sum(dofs);
    block_storage = sum(ranks .* dofs);
    if dense_storage <= 0 || block_storage <= 0
        saving_ratio = NaN;
        saved_fraction = NaN;
    else
        saving_ratio = dense_storage / block_storage;
        saved_fraction = 1 - block_storage / dense_storage;
    end

    stats = struct();
    stats.ranks = ranks;
    stats.dofs = dofs;
    stats.dense_storage = dense_storage;
    stats.block_storage = block_storage;
    stats.saving_ratio = saving_ratio;
    stats.saved_fraction = saved_fraction;
end

function [delta_stats, adjacent_stats, span] = summarize_training_parameter_variation_from_u_scattering(params, u_snapshots)
    if numel(params) < 2 || size(u_snapshots, 2) < 2
        delta_stats = [];
        adjacent_stats = [];
        span = [];
        return;
    end
    delta = diff(params(:));
    adjacent = zeros(numel(delta), 1);
    for k = 1:numel(delta)
        adjacent(k) = norm(u_snapshots(:, k + 1) - u_snapshots(:, k)) / max(norm(u_snapshots(:, k)), 1e-16);
    end
    delta_stats = [min(delta), mean(delta), max(delta)];
    adjacent_stats = [min(adjacent), mean(adjacent), max(adjacent)];
    span = norm(u_snapshots(:, end) - u_snapshots(:, 1)) / max(norm(u_snapshots(:, 1)), 1e-16);
end

function vals = vec_solution_compare_scattering(Ua, Ub, test, part)
    ns = size(Ua, 2);
    vals = zeros(ns, 1);
    for k = 1:ns
        if strcmp(part, 'phys')
            idx = test.systems{k}.active_vec_idx;
            vals(k) = norm(Ua(idx, k) - Ub(idx, k)) / max(norm(Ub(idx, k)), 1e-16);
        else
            vals(k) = norm(Ua(:, k) - Ub(:, k)) / max(norm(Ub(:, k)), 1e-16);
        end
    end
end

function [nearest_full, nearest_phys, nearest_ratio, nearest_log_gap] = nearest_train_solution_errors_ecc(train_ref, test)
    ns = numel(test.ratio);
    nearest_full = zeros(ns, 1);
    nearest_phys = zeros(ns, 1);
    nearest_ratio = zeros(ns, 1);
    nearest_log_gap = zeros(ns, 1);

    train_log = log(train_ref.ratio(:));
    for k = 1:ns
        [nearest_log_gap(k), idx] = min(abs(log(test.ratio(k)) - train_log));
        nearest_ratio(k) = train_ref.ratio(idx);
        u_near = train_ref.u_ref(:, idx);
        u_ref = test.helm_u_ref(:, k);
        nearest_full(k) = norm(u_near - u_ref) / max(norm(u_ref), 1e-16);

        active_idx = test.systems{k}.active_vec_idx;
        if isempty(active_idx)
            nearest_phys(k) = nearest_full(k);
        else
            nearest_phys(k) = norm(u_near(active_idx) - u_ref(active_idx)) / ...
                max(norm(u_ref(active_idx)), 1e-16);
        end
    end
end

function vals = clamp_positive_curve(vals, floor_val)
    vals = vals(:);
    vals(~isfinite(vals) | vals <= floor_val) = floor_val;
end

function grid = embed_interior_field(u_int, nx, ny)
    grid = zeros(nx + 2, ny + 2);
    grid(2:end-1, 2:end-1) = reshape(u_int, nx, ny);
end

function [rel_full, rel_phys] = compute_solution_errors(u_approx, u_ref, sys)
    rel_full = norm(u_approx - u_ref) / max(norm(u_ref), 1e-16);
    idx = sys.active_vec_idx;
    if isempty(idx)
        rel_phys = rel_full;
    else
        d = u_approx(idx) - u_ref(idx);
        r = u_ref(idx);
        rel_phys = norm(d) / max(norm(r), 1e-16);
    end
end

function u = solve_galerkin(A, b, basis_or_space)
    basis_space = normalize_basis_space(basis_or_space);
    if rom_reduced_space_utils('is_empty', basis_space)
        u = zeros(size(b));
        return;
    end
    Ar = rom_reduced_space_utils('assemble_reduced_matrix', A, basis_space);
    br = rom_reduced_space_utils('project', basis_space, b);
    alpha = solve_small_system(Ar, br);
    u = rom_reduced_space_utils('lift', basis_space, alpha);
end

function basis_space = normalize_basis_space(basis_or_space)
    if isempty(basis_or_space)
        basis_space = rom_reduced_space_utils('make_dense_space', zeros(0, 0));
        return;
    end
    if isnumeric(basis_or_space)
        basis_space = rom_reduced_space_utils('make_dense_space', basis_or_space);
        return;
    end
    basis_space = basis_or_space;
end

function basis_space = get_direct_basis_space(direct_rom)
    if isfield(direct_rom, 'basis_space') && ~isempty(direct_rom.basis_space)
        basis_space = direct_rom.basis_space;
        return;
    end
    basis_space = rom_reduced_space_utils('make_dense_space', direct_rom.basis);
end

function basis_space = get_layer_basis_space(layer)
    if isfield(layer, 'basis_space') && ~isempty(layer.basis_space)
        basis_space = layer.basis_space;
        return;
    end
    basis_space = rom_reduced_space_utils('make_dense_space', layer.basis_u);
end

function x = solve_small_system(M, rhs)
    if isempty(M)
        x = zeros(size(rhs));
        return;
    end
    Mf = full(M);
    if rcond(Mf) < 1e-12
        x = pinv(Mf) * rhs;
    else
        x = Mf \ rhs;
    end
end

function x = solve_small_least_squares(M, rhs)
    Mf = full(M);
    if isempty(Mf)
        x = zeros(size(rhs));
        return;
    end
    x = lsqminnorm(Mf, rhs);
end

function pod = build_pod_basis_fullrank(snapshot_matrix, rank_rtol, truncation_rule)
    if nargin < 3 || isempty(truncation_rule)
        truncation_rule = 'relative_singular_value';
    end
    truncation_rule = normalize_pod_truncation_rule(truncation_rule);
    [U, S, V] = svd(snapshot_matrix, 'econ');
    s = diag(S);

    pod = struct();
    pod.left_modes = U;
    pod.right_modes = V;
    pod.singular_values = s;
    pod.rank_used = 0;
    pod.rank_numeric = 0;
    pod.r90 = 0;
    pod.r95 = 0;
    pod.r99 = 0;
    pod.rank_tail_tol = [];
    pod.rank_tail = [];
    pod.singular_value_tol_mode = truncation_rule;
    pod.basis = zeros(size(snapshot_matrix, 1), 0);

    if isempty(s)
        return;
    end

    switch truncation_rule
        case 'relative_frobenius_tail'
            rnum = local_rank_from_relative_frobenius_tail(s, rank_rtol);
            pod.rank_tail_tol = rank_rtol;
            pod.rank_tail = rnum;
        case 'relative_singular_value'
            denom = max(s(1), 1e-30);
            srel = s / denom;
            rnum = nnz(srel >= rank_rtol);
            if rnum == 0
                rnum = 1;
            end
    end

    e2 = cumsum(s.^2) / max(sum(s.^2), 1e-30);
    pod.r90 = first_mode_above(e2, 0.90);
    pod.r95 = first_mode_above(e2, 0.95);
    pod.r99 = first_mode_above(e2, 0.99);

    pod.rank_numeric = rnum;
    pod.rank_used = rnum;
    pod.basis = U(:, 1:rnum);
end

function r = local_rank_from_relative_frobenius_tail(singular_values, tail_tol)
    total = norm(singular_values);
    if total <= 1e-30
        r = 0;
        return;
    end
    tail_energy = flipud(cumsum(flipud(abs(singular_values(:)).^2)));
    tail_after = [tail_energy(2:end); 0];
    rel_tail = sqrt(tail_after) / total;
    r = find(rel_tail <= tail_tol, 1, 'first');
    if isempty(r)
        r = numel(singular_values);
    end
end

function ridx = first_mode_above(cum_energy, threshold)
    ridx = find(cum_energy >= threshold, 1, 'first');
    if isempty(ridx)
        ridx = numel(cum_energy);
    end
end

function s = stats3(v)
    v = v(:);
    v = v(isfinite(v));
    if isempty(v)
        s = [];
    else
        s = [min(v), mean(v), max(v)];
    end
end

function tf = should_parallelize_cases(enabled, ns, min_cases)
    tf = enabled && ns >= min_cases;
    if ~tf
        return;
    end
    tf = license('test', 'Distrib_Computing_Toolbox') && exist('parpool', 'file') == 2;
end

function it = iteration_count(iter_pair, restart)
    if isempty(iter_pair)
        it = 0;
        return;
    end
    it = (iter_pair(1) - 1) * restart + iter_pair(2);
end

function sys = build_scattering_system_for_ratio(ratio_ab, cfg, prob)
    domain = prob.domain;
    semi_a = cfg.radius_ref * sqrt(ratio_ab);
    semi_b = cfg.radius_ref / sqrt(ratio_ab);
    geom = build_scatterer_geometry(domain.nx, domain.ny, prob.x, prob.y, cfg.center, semi_a, semi_b, cfg.eta_boundary_min);
    if cfg.vectorized_scattering_assembly
        [A, b] = helmholtz_scattering_system(domain.nx, domain.ny, domain.hx, ...
            domain.hy, cfg.omega, prob.pml_helm, geom, prob.inc);
    else
        [A, b] = helmholtz_scattering_system_legacy(domain.nx, domain.ny, ...
            domain.hx, domain.hy, cfg.omega, prob.pml_helm, geom, prob.inc);
    end
    active_vec = reshape(geom.active(2:end-1, 2:end-1), domain.nu, 1);

    sys = struct();
    sys.A = A;
    sys.b = b;
    sys.geom = geom;
    sys.ratio_ab = ratio_ab;
    sys.log_ratio = log(ratio_ab);
    sys.semi_a = semi_a;
    sys.semi_b = semi_b;
    sys.active_vec_idx = find(active_vec);
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
    semi_long = max(semi_a, semi_b);
    semi_short = min(semi_a, semi_b);
    geom.ecc = sqrt(max(0, 1 - (semi_short / semi_long)^2));
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

    left_active = false(size(active));
    left_active(2:end, :) = active(1:end - 1, :);
    right_active = false(size(active));
    right_active(1:end - 1, :) = active(2:end, :);
    x_search = false(size(active));
    x_search(3:nx, 2:ny + 1) = true;
    [x_candidates_i, x_candidates_j] = find(internal & x_search ...
        & (left_active | right_active));

    for k = 1:numel(x_candidates_i)
            i = x_candidates_i(k);
            j = x_candidates_j(k);

            left_is_active = active(i - 1, j);
            right_is_active = active(i + 1, j);
            if left_is_active && right_is_active
                error('Ambiguous x-ghost at (%d,%d).', i, j);
            elseif left_is_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0, error('Invalid x-ghost geometry at (%d,%d).', i, j); end
                x_gamma = center(1) - semi_a * sqrt(root);
                eta = abs(x_gamma - x(i - 1)) / (x(i) - x(i - 1));
                if ~(eta > 0 && eta < 1), error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

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
            elseif right_is_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0, error('Invalid x-ghost geometry at (%d,%d).', i, j); end
                x_gamma = center(1) + semi_a * sqrt(root);
                eta = abs(x(i + 1) - x_gamma) / (x(i + 1) - x(i));
                if ~(eta > 0 && eta < 1), error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

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

    down_active = false(size(active));
    down_active(:, 2:end) = active(:, 1:end - 1);
    up_active = false(size(active));
    up_active(:, 1:end - 1) = active(:, 2:end);
    y_search = false(size(active));
    y_search(2:nx + 1, 3:ny) = true;
    [y_candidates_i, y_candidates_j] = find(internal & y_search ...
        & (down_active | up_active));

    for k = 1:numel(y_candidates_i)
            i = y_candidates_i(k);
            j = y_candidates_j(k);

            down_is_active = active(i, j - 1);
            up_is_active = active(i, j + 1);
            if down_is_active && up_is_active
                error('Ambiguous y-ghost at (%d,%d).', i, j);
            elseif down_is_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0, error('Invalid y-ghost geometry at (%d,%d).', i, j); end
                y_gamma = center(2) - semi_b * sqrt(root);
                eta = abs(y_gamma - y(j - 1)) / (y(j) - y(j - 1));
                if ~(eta > 0 && eta < 1), error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

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
            elseif up_is_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0, error('Invalid y-ghost geometry at (%d,%d).', i, j); end
                y_gamma = center(2) + semi_b * sqrt(root);
                eta = abs(y(j + 1) - y_gamma) / (y(j + 1) - y(j));
                if ~(eta > 0 && eta < 1), error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

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
    A_base = pml_stiffness_2d(nx, ny, hx, hy, pml) + omega^2 * speye(Ntot);
    active = reshape(geom.active(2:end - 1, 2:end - 1), Ntot, 1);
    internal = ~active;

    A_scatt = spdiags(double(active), 0, Ntot, Ntot) * A_base ...
        + spdiags(double(internal), 0, Ntot, Ntot);
    [delta_x, rhs_x] = ghost_matrix_updates(A_base, geom, inc, 'x', nx, ny);
    [delta_y, rhs_y] = ghost_matrix_updates(A_base, geom, inc, 'y', nx, ny);
    A_scatt = A_scatt + delta_x + delta_y;
    rhs = rhs_x + rhs_y;
end

function [delta, rhs] = ghost_matrix_updates(A_base, geom, inc, direction, nx, ny)
    Ntot = nx * ny;
    dims_full = [nx + 2, ny + 2];
    switch direction
        case 'x'
            mask = geom.ghost_x;
            owner_full = geom.gx_owner(mask);
            u2_full = geom.gx_u2(mask);
            c1 = geom.gx_c1(mask);
            c2 = geom.gx_c2(mask);
            cg = geom.gx_cg(mask);
            gamma_x = geom.gx_gamma_x(mask);
        case 'y'
            mask = geom.ghost_y;
            owner_full = geom.gy_owner(mask);
            u2_full = geom.gy_u2(mask);
            c1 = geom.gy_c1(mask);
            c2 = geom.gy_c2(mask);
            cg = geom.gy_cg(mask);
            gamma_x = geom.gy_gamma_x(mask);
        otherwise
            error('Unknown ghost direction %s.', direction);
    end

    if isempty(owner_full)
        delta = sparse(Ntot, Ntot);
        rhs = zeros(Ntot, 1);
        return;
    end

    ghost_full = find(mask);
    rows = full_to_interior_indices(owner_full, dims_full, nx, ny);
    ghost_cols = full_to_interior_indices(ghost_full, dims_full, nx, ny);
    u2_cols = full_to_interior_indices(u2_full, dims_full, nx, ny);
    coef = full(A_base(sub2ind([Ntot, Ntot], rows, ghost_cols)));

    keep_c1 = c1 ~= 0;
    keep_c2 = c2 ~= 0;
    delta_rows = [rows; rows(keep_c1); rows(keep_c2)];
    delta_cols = [ghost_cols; rows(keep_c1); u2_cols(keep_c2)];
    delta_vals = [-coef; -coef(keep_c1) .* c1(keep_c1); ...
        coef(keep_c2) .* c2(keep_c2)];
    delta = sparse(delta_rows, delta_cols, delta_vals, Ntot, Ntot);

    u_gamma_hat = -exp(1i * inc.k * gamma_x);
    rhs = full(sparse(rows, ones(size(rows)), -coef .* cg .* u_gamma_hat, Ntot, 1));
end

function idx = full_to_interior_indices(full_idx, dims_full, nx, ny)
    [ii, jj] = ind2sub(dims_full, full_idx);
    if any(ii < 2 | ii > nx + 1 | jj < 2 | jj > ny + 1)
        error('Ghost stencil reaches outside the stored interior grid.');
    end
    idx = (ii - 1) + (jj - 2) * nx;
end

function [A_scatt, rhs] = helmholtz_scattering_system_legacy(nx, ny, hx, hy, omega, pml, geom, inc)
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

            coef = a_x * a_p / (hx * hx);
            if geom.active(i + 1, j)
                A_scatt(row, idx(ii + 1, jj)) = A_scatt(row, idx(ii + 1, jj)) + coef;
            elseif geom.internal(i + 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i + 1, j, idx, dims_full, geom, inc);
            end

            coef = a_x * a_m / (hx * hx);
            if geom.active(i - 1, j)
                A_scatt(row, idx(ii - 1, jj)) = A_scatt(row, idx(ii - 1, jj)) + coef;
            elseif geom.internal(i - 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i - 1, j, idx, dims_full, geom, inc);
            end

            coef = b_y * b_p / (hy * hy);
            if geom.active(i, j + 1)
                A_scatt(row, idx(ii, jj + 1)) = A_scatt(row, idx(ii, jj + 1)) + coef;
            elseif geom.internal(i, j + 1)
                [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, i, j + 1, idx, dims_full, geom, inc);
            end

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
    if owner_lin == 0, error('Missing x-ghost coefficients at (%d,%d).', gi, gj); end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    [u2i, u2j] = ind2sub(dims_full, geom.gx_u2(gi, gj));
    c2 = geom.gx_c2(gi, gj);
    c1 = geom.gx_c1(gi, gj);
    cg = geom.gx_cg(gi, gj);
    u_gamma_hat = -exp(1i * inc.k * geom.gx_gamma_x(gi, gj));

    if abs(c1) > 0
        A_scatt(row, idx(oi - 1, oj - 1)) = A_scatt(row, idx(oi - 1, oj - 1)) - coef * c1;
    end
    if abs(c2) > 0
        A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * c2;
    end
    rhs(row) = rhs(row) - coef * cg * u_gamma_hat;
end

function [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, gi, gj, idx, dims_full, geom, inc)
    owner_lin = geom.gy_owner(gi, gj);
    if owner_lin == 0, error('Missing y-ghost coefficients at (%d,%d).', gi, gj); end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    [u2i, u2j] = ind2sub(dims_full, geom.gy_u2(gi, gj));
    c2 = geom.gy_c2(gi, gj);
    c1 = geom.gy_c1(gi, gj);
    cg = geom.gy_cg(gi, gj);
    u_gamma_hat = -exp(1i * inc.k * geom.gy_gamma_x(gi, gj));

    if abs(c1) > 0
        A_scatt(row, idx(oi - 1, oj - 1)) = A_scatt(row, idx(oi - 1, oj - 1)) - coef * c1;
    end
    if abs(c2) > 0
        A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * c2;
    end
    rhs(row) = rhs(row) - coef * cg * u_gamma_hat;
end

function [W, Pi, uhat] = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x0, geom, inc, projection_shift, store_history)
    if nargin < 13
        projection_shift = [];
    end
    if nargin < 14
        store_history = true;
    end
    need_pi = ~isempty(projection_shift);
    need_uhat = nargout >= 3;
    sx_full = pml.sx_full;
    sy_full = pml.sy_full;

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
    psi2_int = x0(nu + nv + nu + 1 : nu + nv + 2 * nu);
    theta_int = x0(nu + nv + 2 * nu + 1 : end);

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

    u0_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
    v0_vec = [reshape(vx(:, 2:end-1), nvx, 1); reshape(vy(2:end-1, :), nvy, 1)];
    initial_state = [u0_vec; v0_vec; psi1_int; psi2_int; theta_int];
    if store_history
        W = zeros(nt + 1, nstate);
        W(1, :) = initial_state.';
    else
        W = [];
    end
    if need_pi
        pi_acc = 0.5 * (1 - projection_shift) * initial_state;
    else
        pi_acc = [];
    end
    if need_uhat
        q_cos_acc = 0.5 * u0_vec;
        q_sin_acc = zeros(nu, 1);
    else
        q_cos_acc = [];
        q_sin_acc = [];
    end

    tmp1 = zeros(nx + 2, ny + 2);
    tmp2 = zeros(nx + 2, ny + 2);
    div_v = zeros(nx + 2, ny + 2);

    for n = 1:nt
        t = (n - 1) * dt;

        [u_x, u_y] = fill_directional_ghosts(u, geom, inc, omega_ts, t);

        du_dx = (u_x(2:end, :) - u_x(1:end-1, :)) / hx;
        du_dy = (u_y(:, 2:end) - u_y(:, 1:end-1)) / hy;
        vx_new = pml.vx_decay .* vx + (c^2) * pml.vx_drive .* du_dx;
        vy_new = pml.vy_decay .* vy + (c^2) * pml.vy_drive .* du_dy;

        vx_avg = 0.5 * (vx + vx_new);
        vy_avg = 0.5 * (vy + vy_new);

        grad_vx = (vx_avg(2:end, :) - vx_avg(1:end-1, :)) / hx;
        tmp1(:) = 0;
        tmp1(2:end-1, :) = grad_vx;

        grad_vy = (vy_avg(:, 2:end) - vy_avg(:, 1:end-1)) / hy;
        tmp2(:) = 0;
        tmp2(:, 2:end-1) = grad_vy;

        psi1 = psi1 + dt * (sy_full .* tmp1);
        psi2 = psi2 + dt * (sx_full .* tmp2);
        theta = theta + dt * (pml.sigma_xy .* (c_w^2) .* u);

        div_v(:) = 0;
        div_v(2:end-1, :) = div_v(2:end-1, :) + (vx_new(2:end, :) - vx_new(1:end-1, :)) / hx;
        div_v(:, 2:end-1) = div_v(:, 2:end-1) + (vy_new(:, 2:end) - vy_new(:, 1:end-1)) / hy;

        rhs_u = div_v(2:end-1, 2:end-1) - theta(2:end-1, 2:end-1) + psi1(2:end-1, 2:end-1) + psi2(2:end-1, 2:end-1);

        u_int = u(2:end-1, 2:end-1);
        u_int_new = pml.u_decay .* u_int + pml.u_drive .* rhs_u;
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

        u_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
        vx_vec = reshape(vx(:, 2:end-1), nvx, 1);
        vy_vec = reshape(vy(2:end-1, :), nvy, 1);
        psi1_vec = reshape(psi1(2:end-1, 2:end-1), nu, 1);
        psi2_vec = reshape(psi2(2:end-1, 2:end-1), nu, 1);
        theta_vec = reshape(theta(2:end-1, 2:end-1), nu, 1);

        if store_history
            W(n + 1, 1:nu) = u_vec.';
            W(n + 1, nu + (1:nvx)) = vx_vec.';
            W(n + 1, nu + nvx + (1:nvy)) = vy_vec.';
            W(n + 1, nu + nv + (1:nu)) = psi1_vec.';
            W(n + 1, nu + nv + nu + (1:nu)) = psi2_vec.';
            W(n + 1, nu + nv + 2 * nu + (1:nu)) = theta_vec.';
        end

        endpoint_weight = 1.0;
        if n == nt
            endpoint_weight = 0.5;
        end
        t_now = n * dt;
        if need_pi
            weight = endpoint_weight * (cos(omega_ts * t_now) - projection_shift);
            pi_acc(1:nu) = pi_acc(1:nu) + weight * u_vec;
            pi_acc(nu + (1:nvx)) = pi_acc(nu + (1:nvx)) + weight * vx_vec;
            pi_acc(nu + nvx + (1:nvy)) = pi_acc(nu + nvx + (1:nvy)) + weight * vy_vec;
            pi_acc(nu + nv + (1:nu)) = pi_acc(nu + nv + (1:nu)) + weight * psi1_vec;
            pi_acc(nu + nv + nu + (1:nu)) = pi_acc(nu + nv + nu + (1:nu)) + weight * psi2_vec;
            pi_acc(nu + nv + 2 * nu + (1:nu)) = pi_acc(nu + nv + 2 * nu + (1:nu)) + weight * theta_vec;
        end
        if need_uhat
            q_cos_acc = q_cos_acc + endpoint_weight * cos(omega_ts * t_now) * u_vec;
            q_sin_acc = q_sin_acc + endpoint_weight * sin(omega_ts * t_now) * u_vec;
        end
    end

    projection_scale = 2.0 * dt / max(nt * dt, 1e-16);
    Pi = projection_scale * pi_acc;
    uhat = projection_scale * (q_cos_acc + 1i * q_sin_acc);
end

function x_real = lift_frequency_solution_to_real_state_scattering(nx, ny, hx, hy, c, omega_ts, dt, pml, geom, inc, uhat)
    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    nstate = nu + nv + 3 * nu;

    if isempty(uhat) || norm(uhat) <= 0
        x_real = zeros(nstate, 1);
        return;
    end

    u = zeros(nx + 2, ny + 2);
    u(2:end-1, 2:end-1) = reshape(uhat, nx, ny);
    u(geom.internal) = 0;

    [u_x, u_y] = fill_directional_ghosts_phasor(u, geom, inc);
    du_dx = (u_x(2:end, :) - u_x(1:end-1, :)) / hx;
    du_dy = (u_y(:, 2:end) - u_y(:, 1:end-1)) / hy;

    sx_full = pml.sx_full;
    sy_full = pml.sy_full;
    sigma_x_u = sx_full(2:end-1, 2:end-1);
    sigma_y_u = sy_full(2:end-1, 2:end-1);
    sigma_x_vx = 0.5 * (sx_full(1:end-1, 2:end-1) + sx_full(2:end, 2:end-1));
    sigma_y_vy = 0.5 * (sy_full(2:end-1, 1:end-1) + sy_full(2:end-1, 2:end));

    lam = exp(-1i * omega_ts * dt);
    denom_vx = lam * (1 + 0.5 * dt * sigma_x_vx) - (1 - 0.5 * dt * sigma_x_vx);
    denom_vy = lam * (1 + 0.5 * dt * sigma_y_vy) - (1 - 0.5 * dt * sigma_y_vy);
    vx_hat = (dt * c^2 * du_dx(:, 2:end-1)) ./ denom_vx;
    vy_hat = (dt * c^2 * du_dy(2:end-1, :)) ./ denom_vy;

    vx_full = zeros(nx + 1, ny + 2);
    vy_full = zeros(nx + 2, ny + 1);
    vx_full(:, 2:end-1) = vx_hat;
    vy_full(2:end-1, :) = vy_hat;

    grad_vx = (vx_full(2:end, :) - vx_full(1:end-1, :)) / hx;
    grad_vy = (vy_full(:, 2:end) - vy_full(:, 1:end-1)) / hy;

    avg_factor = 0.5 * dt * (1 + lam) / (lam - 1);
    theta_factor = dt / (lam - 1);
    psi1 = avg_factor * (sigma_y_u .* grad_vx(:, 2:end-1));
    psi2 = avg_factor * (sigma_x_u .* grad_vy(2:end-1, :));
    theta = theta_factor * ((sigma_x_u .* sigma_y_u) .* (cos(omega_ts * dt / 2)^2) .* reshape(uhat, nx, ny));

    active_int = geom.active(2:end-1, 2:end-1);
    u_int = reshape(uhat, nx, ny);
    u_int(~active_int) = 0;
    psi1(~active_int) = 0;
    psi2(~active_int) = 0;
    theta(~active_int) = 0;

    xhat = zeros(nstate, 1);
    xhat(1:nu) = u_int(:);
    xhat(nu + 1:nu + nvx) = reshape(vx_hat, nvx, 1);
    xhat(nu + nvx + 1:nu + nv) = reshape(vy_hat, nvy, 1);
    xhat(nu + nv + 1:nu + nv + nu) = psi1(:);
    xhat(nu + nv + nu + 1:nu + nv + 2 * nu) = psi2(:);
    xhat(nu + nv + 2 * nu + 1:end) = theta(:);
    x_real = real(xhat);
end

function [u_x, u_y] = fill_directional_ghosts_phasor(u, geom, inc)
    u_x = u;
    u_y = u;

    if any(geom.ghost_x(:))
        mask = geom.ghost_x;
        owner_vals = u(geom.gx_owner(mask));
        u2_vals = u(geom.gx_u2(mask));
        u_gamma_hat = -exp(1i * inc.k * geom.gx_gamma_x(mask));
        ghost_vals = geom.gx_c2(mask) .* u2_vals - geom.gx_c1(mask) .* owner_vals + geom.gx_cg(mask) .* u_gamma_hat;
        u_x(mask) = ghost_vals;
    end

    if any(geom.ghost_y(:))
        mask = geom.ghost_y;
        owner_vals = u(geom.gy_owner(mask));
        u2_vals = u(geom.gy_u2(mask));
        u_gamma_hat = -exp(1i * inc.k * geom.gy_gamma_x(mask));
        ghost_vals = geom.gy_c2(mask) .* u2_vals - geom.gy_c1(mask) .* owner_vals + geom.gy_cg(mask) .* u_gamma_hat;
        u_y(mask) = ghost_vals;
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

function [u_full, step_hist, rec_hist, rec_res, x_final] = waveholtz_iteration_scattering_with_init(nx, ny, hx, hy, c, omega_ts, T_ts, dt, nt, pml, geom, inc, max_iter, tol, filter_shift, relax, A, b, verbose_iter, x0, stream_projection)
    % x0 is the 20th input; when supplied it is the lifted WaveHoltz state.
    if nargin < 20 || isempty(x0)
        nu = nx * ny;
        nvx = (nx + 1) * ny;
        nvy = nx * (ny + 1);
        x = zeros(nu + nvx + nvy + 3 * nu, 1);
    else
        x = x0;
    end
    if nargin < 21
        stream_projection = true;
    end

    step_hist = zeros(max_iter, 1);
    rec_hist = zeros(max_iter + 1, 1);
    nb = max(norm(b), 1e-15);
    step_count = 0;

    for k = 1:max_iter
        if stream_projection
            [~, x_raw, u_rec_vec] = solve_wave_one_period_scattering( ...
                nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x, geom, inc, filter_shift, false);
            if any(~isfinite(x_raw)) || any(~isfinite(u_rec_vec))
                error('Non-finite values encountered during one-period time stepping at iteration %d.', k);
            end
            rec_hist(k) = norm(A * u_rec_vec - b) / nb;
        else
            W = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x, geom, inc);
            if any(~isfinite(W(:)))
                error('Non-finite values encountered during one-period time stepping at iteration %d.', k);
            end
            u_rec = recover_u_phasor_from_history_scattering(nx, ny, omega_ts, T_ts, dt, nt, W, geom);
            rec_hist(k) = norm(A * reshape(u_rec(2:end-1, 2:end-1), nx * ny, 1) - b) / nb;

            t_grid = (0:nt).' * dt;
            weights = cos(omega_ts * t_grid) - filter_shift;
            integrand = bsxfun(@times, weights, W);
            integral = trapz(t_grid, integrand, 1);
            x_raw = (2.0 / T_ts) * integral.';
        end
        x_new = (1 - relax) * x + relax * x_raw;
        if any(~isfinite(x_new))
            error('Non-finite fixed-point state at iteration %d.', k);
        end

        res = norm(x_new - x) / max(1e-16, norm(x_new));
        step_hist(k) = res;
        x = x_new;
        step_count = k;

        if verbose_iter && (k <= 5 || mod(k, 20) == 0 || res < tol)
            fprintf('iter %3d: step = %.3e, rec-res = %.3e\n', k, res, rec_hist(k));
        end
        if res < tol
            break;
        end
    end
    step_hist = step_hist(1:step_count);

    u_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, T_ts, dt, nt, pml, x, geom, inc, stream_projection);
    u_full(geom.internal) = 0;

    nu = nx * ny;
    rec_res = norm(A * reshape(u_full(2:end-1, 2:end-1), nu, 1) - b) / nb;
    rec_hist(step_count + 1) = rec_res;
    rec_hist = rec_hist(1:step_count + 1);
    if verbose_iter
        fprintf('Recovered phasor residual: %.3e\n', rec_res);
    end
    x_final = x;
end

function uhat_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, T_ts, dt, nt, pml, x_state, geom, inc, stream_projection)
    if stream_projection
        [~, ~, uhat] = solve_wave_one_period_scattering( ...
            nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x_state, geom, inc, [], false);
        uhat_full = zeros(nx + 2, ny + 2);
        uhat_full(2:end-1, 2:end-1) = reshape(uhat, nx, ny);
        uhat_full(geom.internal) = 0;
    else
        Wconv = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x_state, geom, inc);
        uhat_full = recover_u_phasor_from_history_scattering(nx, ny, omega_ts, T_ts, dt, nt, Wconv, geom);
    end
end

function uhat_full = recover_u_phasor_from_history_scattering(nx, ny, omega_ts, T_ts, dt, nt, Wconv, geom)
    nu = nx * ny;
    t_grid = (0:nt).' * dt;
    q_cos = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, cos(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';
    q_sin = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, sin(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';
    uhat = q_cos + 1i * q_sin;

    uhat_full = zeros(nx + 2, ny + 2);
    uhat_full(2:end-1, 2:end-1) = reshape(uhat, nx, ny);
    uhat_full(geom.internal) = 0;
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
