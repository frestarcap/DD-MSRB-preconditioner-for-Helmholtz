function results = helm_waveholtz_correction_rom(cfg, frozen)
    % Standalone single-file script/function for:
    %   Direct WaveHoltz POD-ROM initial guess
    %   + trajectory-aware layered Helmholtz FGMRES ROM.
    %
    % Usage:
    %   results = helm_waveholtz_correction_rom();
    %   results = helm_waveholtz_correction_rom(cfg);
    %
    % This file is self-contained: all required helpers live below in the
    % same .m file, so it does not rely on external project interfaces.
    %
    % Algorithmic split:
    %   1) Offline references come from high-accuracy WaveHoltz solves.
    %   2) Layered ROM spaces are trained from the corresponding
    %      trajectory-aware Helmholtz FGMRES vectors in the frequency-domain
    %      variable u(mu).
    %   3) Online evaluation first performs a few layered reduced
    %      Helmholtz FGMRES steps in u, lifts once to a WaveHoltz state x,
    %      and then lets WaveHoltz fixed-point / Richardson finish the
    %      solve.
    %
    % The target PDE is
    %   Delta u + omega^2 / c(x)^2 * u = f,
    % with c(x)=1 on the left half and c(x)=c_right on the right half.
    % WaveHoltz is used in standard variable-speed form:
    %   (1 / c(x)^2) w_tt = Delta w - f(x) cos(omega t).

    run_clock = tic;
    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);
    setup_clock = tic;
    prob = build_problem(cfg);
    rhs = build_rhs(prob, cfg);
    setup_time = toc(setup_clock);

    if isfield(cfg,'frozen_online_file') && ~isempty(cfg.frozen_online_file)
        % Deploy a frozen prefix without reference solves or training metadata.
        saved=load(cfg.frozen_online_file,'model','A_base','A_right','rhs');
        assert(isequaln(saved.A_base,prob.A_base) && ...
            isequaln(saved.A_right,prob.A_right) && isequaln(saved.rhs,rhs), ...
            'Frozen online model and requested Helmholtz problem differ.');
        assert(~isempty(cfg.online_true_helm_tol), ...
            'Frozen online evaluation requires a true residual target.');
        model=saved.model; clear saved;
        rows=struct([]);
        for mu=cfg.test_params(:).'
            timer=tic;
            [u,pre]=run_helmholtz_precorrection(prob,cfg,rhs,mu,zeros(size(rhs)),model);
            iterations=0; native=[]; native_final=NaN;
            if pre.relres_final>cfg.online_true_helm_tol
                [bwh,Afun]=build_waveholtz_operator(prob,cfg,rhs,mu);
                x=lift_frequency_solution_to_real_state(prob,u,mu);
                [~,completion]=run_true_verified_wave_completion(prob,cfg,rhs,mu,bwh,Afun,x);
                u=completion.u_final; iterations=completion.iter_count;
                native=completion.relres_hist; native_final=completion.relres_final;
            end
            A=build_helmholtz_operator(prob,mu); rho=norm(rhs-A*u)/norm(rhs);
            elapsed=toc(timer);
            assert(rho<=cfg.online_true_helm_tol,'Frozen online true residual failed.');
            one=struct('parameter',mu,'post_ROM_residual',pre.relres_final, ...
                'completion_iterations',iterations,'total_s',elapsed, ...
                'true_residual',rho,'native_final',native_final, ...
                'native_history',native,'ROM_history',pre.relres_hist,'u',u);
            if isempty(rows), rows=one; else, rows(end+1)=one; end %#ok<AGROW>
        end
        results=struct('cfg',cfg,'rows',rows,'setup_s',setup_time);
        return;
    end

    if isfield(cfg,'deployment_file') && ~isempty(cfg.deployment_file)
        saved=load(cfg.deployment_file,'model'); model=saved.model; clear saved;
        results=struct('true_residual',[],'completion_iterations',[]);
        for mu=cfg.test_params(:).'
            [u,pre]=run_helmholtz_precorrection(prob,cfg,rhs,mu,zeros(size(rhs)),model);
            iterations=0;
            if isempty(cfg.online_helm_accept_tol) || pre.relres_final>cfg.online_helm_accept_tol
                [bwh,Afun]=build_waveholtz_operator(prob,cfg,rhs,mu);
                x=lift_frequency_solution_to_real_state(prob,u,mu);
                [~,completion]=run_online_wave_completion(prob,cfg,rhs,mu,bwh,Afun,x,false);
                u=completion.u_final; iterations=completion.iter_count;
            end
            A=build_helmholtz_operator(prob,mu);
            rho=norm(rhs-A*u)/norm(rhs);
            assert(rho<=1e-12,'Deployment true residual failed.');
            results.true_residual(end+1)=rho;
            results.completion_iterations(end+1)=iterations;
        end
        return;
    end

    fprintf('=== helm_waveholtz_correction_rom ===\n');
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
    fprintf('ROM check target = %s\n', cfg.rom_check_target);
    fprintf('use direct ROM init = %d\n', cfg.use_direct_rom_init);
    fprintf('Helmholtz layered FGMRES ROM steps = %d\n', cfg.corr_num_layers);
    fprintf('online completion mode = %s\n', cfg.online_completion_mode);
    fprintf('run WaveHoltz baselines = %d\n', cfg.run_wave_baselines);
    if cfg.run_wave_baselines && ~isempty(cfg.wave_baseline_max_iter)
        fprintf('WaveHoltz baseline max iter = %d\n', cfg.wave_baseline_max_iter);
    end
    fprintf('corr basis style = %s\n', cfg.corr_basis_style);
    if ~isempty(cfg.corr_pod_singular_value_tols)
        fprintf('layered Helmholtz POD truncation = %s, tol = %s\n', ...
            cfg.corr_pod_truncation_rule, mat2str(cfg.corr_pod_singular_value_tols, 4));
    end
    fprintf('Helmholtz GMRES track tol = %.1e\n', cfg.helm_gmres_track_tol);
    fprintf('online fixed-point tol = %.1e, maxit = %d\n', cfg.fp_tol, cfg.fp_maxit);
    if ~isempty(cfg.fp_step_tol)
        fprintf('online fixed-point step tol = %.1e\n', cfg.fp_step_tol);
    end
    fprintf('source phys grid = (%d,%d), x=%.4f, y=%.4f\n', ...
        cfg.source_col_phys, cfg.source_row_phys, prob.x_phys(cfg.source_col_phys), prob.y_phys(cfg.source_row_phys));
    fprintf('train params = %s\n', mat2str(cfg.train_params, 4));
    fprintf('test  params = %s\n', mat2str(cfg.test_params, 4));

    if cfg.setup_only
        results = struct();
        results.cfg = cfg;
        results.problem = rmfield(prob, {'pml_helm', 'pml_wave'});
        results.rhs = rhs;
        results.timing = struct('setup', setup_time, 'offline', 0, ...
            'online', 0, 'total', toc(run_clock));
        return;
    end

    offline_clock = tic;
    if nargin >= 2 && ~isempty(frozen)
        assert(isequaln(prob.A_base,frozen.problem.A_base) && ...
            isequaln(prob.A_right,frozen.problem.A_right) && isequaln(rhs,frozen.rhs), ...
            'Frozen ROM and requested full-order problem differ.');
        train_ref = frozen.train_reference;
        direct_rom = frozen.direct_rom;
        corr_rom = frozen.correction_rom;
        assert(corr_rom.num_layers == cfg.corr_num_layers);
        offline_time = 0;
    else
    if strcmp(cfg.reference_mode, 'helm_lift')
        fprintf('\n=== Offline stage 1: Helm references lifted to WaveHoltz state ===\n');
    else
        fprintf('\n=== Offline stage 1: WaveHoltz reference solves ===\n');
    end
    train_ref = solve_reference_waveholtz_set(prob, cfg, rhs, cfg.train_params, 'train');
    block3_partition = [];
    if strcmp(cfg.corr_basis_style, 'block3')
        block3_partition = build_helm_u_block3_partition(prob, cfg);
    end

    fprintf('\n=== Offline stage 2: direct full-state WaveHoltz ROM ===\n');
    direct_rom = build_direct_fullstate_rom(prob, cfg, rhs, train_ref);
    print_direct_fullstate_rom_summary(direct_rom);

    fprintf('\n=== Offline stage 3: trajectory-aware Helmholtz layered FGMRES ROM ===\n');
    corr_rom = train_correction_rom_layers(prob, cfg, rhs, train_ref, direct_rom, block3_partition);
    print_correction_rom_summary(corr_rom);
    offline_time = toc(offline_clock);
    end

    fprintf('\n=== Online evaluation on unseen parameters ===\n');
    online_clock = tic;
    test = evaluate_correction_rom_online(prob, cfg, rhs, direct_rom, corr_rom);
    online_time = toc(online_clock);

    summary = summarize_correction_results(train_ref, direct_rom, corr_rom, test, prob, cfg);
    print_correction_summary(summary);

    nr_comparison = [];
    if cfg.run_nr_rom_comparison && corr_rom.num_layers > 1
        fprintf('\n=== Equal-rank non-replayed layered correction ROM comparison ===\n');
        nr_corr_rom = train_nonreplayed_correction_rom_layers(prob, cfg, rhs, train_ref, direct_rom, corr_rom);
        cfg_nr = cfg;
        cfg_nr.run_nr_rom_comparison = false;
        cfg_nr.run_wave_baselines = false;
        cfg_nr.parallel_test = cfg.parallel_test;
        nr_test = evaluate_correction_rom_online(prob, cfg_nr, rhs, direct_rom, nr_corr_rom);
        nr_summary = summarize_correction_results(train_ref, direct_rom, nr_corr_rom, nr_test, prob, cfg_nr);
        nr_comparison = struct();
        nr_comparison.correction_rom = nr_corr_rom;
        nr_comparison.test = nr_test;
        nr_comparison.summary = nr_summary;
    end

    if cfg.save_benchmark
        save_benchmark_summary_helm(cfg, summary, nr_comparison);
    end

    if cfg.do_plots
        make_correction_plots(cfg, prob, rhs, direct_rom, corr_rom, test, summary);
        if ~isempty(nr_comparison)
            plot_nr_vs_trajectory_residual_decay_helm(cfg, test, summary, nr_comparison.test, nr_comparison.summary);
        end
    end

    results = struct();
    results.cfg = cfg;
    results.problem = rmfield(prob, {'pml_helm', 'pml_wave'});
    results.rhs = rhs;
    results.train_reference = train_ref;
    results.direct_rom = direct_rom;
    results.correction_rom = corr_rom;
    results.test = test;
    results.residual_metadata = struct('precorrection','norm(b-A*u)/norm(b)', ...
        'completion',cfg.online_completion_mode, ...
        'true_helm_history_recorded',cfg.record_true_helm_history, ...
        'true_completion_field','online_true_helm_relres_hist', ...
        'combined_is_true_helmholtz',~strcmp(cfg.online_completion_mode,'waveholtz'), ...
        'waveholtz_combined_convention','legacy endpoint-scaled display only; use separate stage histories');
    results.summary = summary;
    results.nr_comparison = nr_comparison;
    results.timing = struct('setup', setup_time, 'offline', offline_time, ...
        'online', online_time, 'total', toc(run_clock));
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'L', 1.0);
    cfg = set_default(cfg, 'Lx', cfg.L);
    cfg = set_default(cfg, 'Ly', cfg.L);
    cfg = set_default(cfg, 'omega', 53.5);
    cfg = set_default(cfg, 'Nphys', 125);
    cfg = set_default(cfg, 'Nphys_x', cfg.Nphys);
    cfg = set_default(cfg, 'Nphys_y', cfg.Nphys);
    cfg = set_default(cfg, 'ispml', 1);
    cfg = set_default(cfg, 'npml', []);
    cfg = set_default(cfg, 'sigma_max', 80);
    cfg = set_default(cfg, 'CFL', 0.5);
    cfg = set_default(cfg, 'filter_shift', 0.25);
    cfg = set_default(cfg, 'interface_x', []);
    cfg = set_default(cfg, 'material_model', 'halfspace');
    cfg.material_model = lower(strtrim(cfg.material_model));
    cfg = set_default(cfg, 'pc_num_rows', 3);
    cfg = set_default(cfg, 'pc_lattice_a', []);
    cfg = set_default(cfg, 'pc_radius', []);
    cfg = set_default(cfg, 'pc_rod_shape', 'circle');
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

    cfg = set_default(cfg, 'train_params', []);
    cfg = set_default(cfg, 'test_params', []);
    cfg = set_default(cfg, 'test_param_strategy', 'random_interval');
    if ~isfield(cfg, 'test_param_rng_seed')
        cfg.test_param_rng_seed = 1;
    end
    cfg = set_default(cfg, 'pod_energy_target', 1e-9);
    cfg = set_default(cfg, 'pod_singular_value_tol', 1e-9);
    cfg = set_default(cfg, 'pod_min_rank', 1);
    cfg = set_default(cfg, 'pod_max_rank', []);
    cfg = set_default(cfg, 'rom_check_target', 'wave');
    cfg = set_default(cfg, 'reference_mode', 'waveholtz');
    cfg.reference_mode = lower(strtrim(cfg.reference_mode));
    cfg = set_default(cfg, 'use_direct_rom_init', false);
    cfg = set_default(cfg, 'build_direct_rom_when_unused', true);

    cfg = set_default(cfg, 'corr_num_layers', 3);
    cfg = set_default(cfg, 'corr_pod_energy_target', 0.999);
    cfg = set_default(cfg, 'corr_pod_singular_value_tol', 1e-10);
    cfg = set_default(cfg, 'corr_pod_truncation_rule', 'relative_frobenius_tail');
    if isfield(cfg, 'corr_rank_rtols') && ~isempty(cfg.corr_rank_rtols) && ...
            (~isfield(cfg, 'corr_pod_singular_value_tols') || isempty(cfg.corr_pod_singular_value_tols))
        cfg.corr_pod_singular_value_tols = cfg.corr_rank_rtols;
    end
    cfg = set_default(cfg, 'corr_pod_singular_value_tols', [1e-4, 1e-3, 1e-2]);
    cfg = set_default(cfg, 'corr_pod_min_rank', 1);
    cfg = set_default(cfg, 'corr_pod_max_rank', []);
    cfg = set_default(cfg, 'corr_normalize_snapshots', false);
    cfg = set_default(cfg, 'corr_basis_style', 'block3'); % global | block3
    cfg = set_default(cfg, 'block3_partition_mode', 'pc_waveguide');
    cfg = set_default(cfg, 'block3_center', [0.5 * cfg.L, 0.5 * cfg.L]);
    cfg = set_default(cfg, 'block3_r_near', 0.25 * cfg.L);
    cfg = set_default(cfg, 'block3_waveguide_halfwidth', []);
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
    cfg = set_default(cfg, 'helm_gmres_track_tol', 1e-14);
    cfg = set_default(cfg, 'online_completion_mode', 'waveholtz');
    cfg = set_default(cfg, 'online_wave_solver', 'fixedpoint');
    cfg.online_wave_solver = lower(strtrim(cfg.online_wave_solver));
    cfg = set_default(cfg, 'record_true_helm_history', false);
    if cfg.record_true_helm_history && ~strcmp(cfg.online_wave_solver,'fixedpoint')
        error('True history recording currently requires fixedpoint completion.');
    end
    cfg = set_default(cfg, 'run_wave_baselines', true);
    cfg = set_default(cfg, 'wave_baseline_max_iter', []);
    cfg = set_default(cfg, 'helm_completion_restart', 20);
    cfg = set_default(cfg, 'helm_completion_maxit', 40);
    cfg = set_default(cfg, 'helm_completion_tol', []);
    cfg = set_default(cfg, 'shifted_laplacian_beta', 0.2);
    cfg = set_default(cfg, 'shifted_laplacian_restart', []);
    cfg = set_default(cfg, 'shifted_laplacian_maxit', []);
    cfg = set_default(cfg, 'shifted_laplacian_tol', []);
    cfg = set_default(cfg, 'shifted_laplacian_reorth', true);
    cfg = set_default(cfg, 'shifted_laplacian_verbose', false);
    cfg = set_default(cfg, 'shifted_laplacian_solver', 'lu');
    cfg = set_default(cfg, 'shifted_laplacian_ilu_type', 'ilutp');
    cfg = set_default(cfg, 'shifted_laplacian_ilu_droptol', 1e-2);
    cfg = set_default(cfg, 'shifted_laplacian_ilu_thresh', 1.0);
    cfg = set_default(cfg, 'shifted_laplacian_ilu_milu', 'off');
    cfg = set_default(cfg, 'helm_baseline_iters', 60);
    cfg = set_default(cfg, 'helm_baseline_tol', 0);
    cfg = set_default(cfg, 'wh_max_iter', 1200);
    cfg = set_default(cfg, 'wh_tol', 1e-14);
    cfg = set_default(cfg, 'wh_gmres_restart', 20);
    cfg = set_default(cfg, 'wh_gmres_maxit', 40);
    cfg = set_default(cfg, 'wh_gmres_reorth', false);
    cfg = set_default(cfg, 'wh_verbose_gmres', false);
    cfg = set_default(cfg, 'wh_initial_guess', 'zero');
    cfg.wh_initial_guess = lower(strtrim(cfg.wh_initial_guess));
    cfg = set_default(cfg, 'fp_tol', 1e-14);
    cfg = set_default(cfg, 'online_true_helm_tol', []);
    if ~isempty(cfg.online_true_helm_tol)
        validateattributes(cfg.online_true_helm_tol,{'numeric'}, ...
            {'scalar','real','finite','positive','<',1});
    end
    cfg = set_default(cfg, 'online_helm_accept_tol', []);
    if ~isempty(cfg.online_helm_accept_tol)
        validateattributes(cfg.online_helm_accept_tol, {'numeric'}, ...
            {'scalar','real','finite','positive','<',1});
    end
    if ~isempty(cfg.online_true_helm_tol) && ~isempty(cfg.online_helm_accept_tol)
        assert(cfg.online_helm_accept_tol<=cfg.online_true_helm_tol, ...
            'Early Helmholtz acceptance must be at least as strict as final acceptance.');
    end
    cfg = set_default(cfg, 'fp_step_tol', []);
    cfg = set_default(cfg, 'fp_maxit', 200);
    cfg = set_default(cfg, 'train_params',linspace(0.8, 0.95, 24));

    cfg = set_default(cfg, 'do_reference_check', true);
    cfg = set_default(cfg, 'plot_test_index', []);
    cfg = set_default(cfg, 'run_nr_rom_comparison', false);
    cfg = set_default(cfg, 'do_plots', true);
    cfg = set_default(cfg, 'save_plots', true);
    cfg = set_default(cfg, 'figure_root', helm_output_path('outputs', 'figures'));
    cfg = set_default(cfg, 'figure_subdir', '');
    cfg = set_default(cfg, 'figure_format', 'png');
    cfg = set_default(cfg, 'plot_y_limits', []);
    cfg = set_default(cfg, 'save_benchmark', true);
    cfg = set_default(cfg, 'benchmark_root', helm_output_path('outputs', 'benchmarks'));
    cfg = set_default(cfg, 'benchmark_name', '');
    cfg = set_default(cfg, 'parallel_train', true);
    cfg = set_default(cfg, 'parallel_train_min_cases', 2);
    cfg = set_default(cfg, 'parallel_test', true);
    cfg = set_default(cfg, 'parallel_test_min_cases', 2);
    cfg = set_default(cfg, 'stream_wave_projection', true);
    if cfg.record_true_helm_history && ~cfg.stream_wave_projection
        error('True history recording requires stream_wave_projection=true.');
    end
    cfg = set_default(cfg, 'setup_only', false);

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
    if isempty(cfg.figure_subdir)
        if strcmp(cfg.material_model, 'photonic_crystal')
            cfg.figure_subdir = 'photonic_crystal_correction_rom';
        else
            cfg.figure_subdir = 'helm_correction_rom';
        end
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
    if isempty(cfg.train_params)
        cfg.train_params = linspace(0.6, 0.9, 30);
    end
    if isempty(cfg.test_params)
        cfg.test_params = generate_interval_test_params(cfg.train_params, cfg.test_param_strategy, cfg.test_param_rng_seed);
    end

    cfg.train_params = unique(cfg.train_params(:).');
    cfg.test_params = unique(cfg.test_params(:).');

    validate_param_array(cfg.train_params, 'train_params');
    validate_param_array(cfg.test_params, 'test_params');
    if any(abs(cfg.test_params(:) - cfg.train_params(:).') < 1e-12)
        error('test_params must be unseen and cannot overlap train_params.');
    end
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
    valid_check_targets = {'wave', 'helm'};
    if ~any(strcmp(cfg.rom_check_target, valid_check_targets))
        error('rom_check_target=%s must be one of: %s.', cfg.rom_check_target, strjoin(valid_check_targets, ', '));
    end
    valid_reference_modes = {'waveholtz', 'helm_lift'};
    if ~any(strcmp(cfg.reference_mode, valid_reference_modes))
        error('reference_mode=%s must be one of: %s.', cfg.reference_mode, strjoin(valid_reference_modes, ', '));
    end
    valid_completion_modes = {'waveholtz', 'helmholtz', 'shifted_laplacian'};
    if ~any(strcmp(cfg.online_completion_mode, valid_completion_modes))
        error('online_completion_mode=%s must be one of: %s.', cfg.online_completion_mode, strjoin(valid_completion_modes, ', '));
    end
    valid_wave_solvers = {'fixedpoint', 'gmres'};
    if ~any(strcmp(cfg.online_wave_solver, valid_wave_solvers))
        error('online_wave_solver=%s must be one of: %s.', cfg.online_wave_solver, strjoin(valid_wave_solvers, ', '));
    end
    valid_initial_guesses = {'zero', 'helm_lift'};
    if ~any(strcmp(cfg.wh_initial_guess, valid_initial_guesses))
        error('wh_initial_guess=%s must be one of: %s.', cfg.wh_initial_guess, strjoin(valid_initial_guesses, ', '));
    end
    valid_basis_styles = {'global', 'block3'};
    if ~any(strcmp(cfg.corr_basis_style, valid_basis_styles))
        error('corr_basis_style=%s must be one of: %s.', cfg.corr_basis_style, strjoin(valid_basis_styles, ', '));
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
    if ~(isscalar(cfg.pc_square_halfwidth) && isfinite(cfg.pc_square_halfwidth) && cfg.pc_square_halfwidth > 0)
        error('pc_square_halfwidth must be a positive finite scalar.');
    end
    if numel(cfg.block3_center) ~= 2 || any(~isfinite(cfg.block3_center))
        error('block3_center must be a finite 2-vector.');
    end
    if ~(isfinite(cfg.block3_r_near) && cfg.block3_r_near > 0)
        error('block3_r_near must be a positive finite scalar.');
    end
    cfg.block3_partition_mode = lower(strtrim(cfg.block3_partition_mode));
    valid_partition_modes = {'radial', 'pc_waveguide', 'waveguide', ...
        'halfspace_material', 'halfspace_source_interface', 'halfspace_refined', ...
        'pc_physical_directional_pml', 'pc_channel_lr_pml', ...
        'pc_core_cladding_lr_tb_pml', 'pc_refined'};
    if ~any(strcmp(cfg.block3_partition_mode, valid_partition_modes))
        error('block3_partition_mode=%s must be one of: %s.', cfg.block3_partition_mode, strjoin(valid_partition_modes, ', '));
    end
    if isempty(cfg.block3_waveguide_halfwidth)
        cfg.block3_waveguide_halfwidth = cfg.block3_r_near;
    end
    if ~(isfinite(cfg.block3_waveguide_halfwidth) && cfg.block3_waveguide_halfwidth > 0)
        error('block3_waveguide_halfwidth must be a positive finite scalar.');
    end
    cfg.corr_block3_rank_cap_near = expand_optional_layer_values(cfg.corr_block3_rank_cap_near, ...
        cfg.corr_num_layers, 'corr_block3_rank_cap_near');
    cfg.corr_block3_rank_cap_far = expand_optional_layer_values(cfg.corr_block3_rank_cap_far, ...
        cfg.corr_num_layers, 'corr_block3_rank_cap_far');
    cfg.corr_block3_rank_cap_pml = expand_optional_layer_values(cfg.corr_block3_rank_cap_pml, ...
        cfg.corr_num_layers, 'corr_block3_rank_cap_pml');
    validate_block3_rank_cap(cfg.corr_block3_rank_cap_near, 'corr_block3_rank_cap_near');
    validate_block3_rank_cap(cfg.corr_block3_rank_cap_far, 'corr_block3_rank_cap_far');
    validate_block3_rank_cap(cfg.corr_block3_rank_cap_pml, 'corr_block3_rank_cap_pml');
    cfg.corr_block3_rtol_near = expand_optional_layer_values(cfg.corr_block3_rtol_near, ...
        cfg.corr_num_layers, 'corr_block3_rtol_near');
    cfg.corr_block3_rtol_far = expand_optional_layer_values(cfg.corr_block3_rtol_far, ...
        cfg.corr_num_layers, 'corr_block3_rtol_far');
    cfg.corr_block3_rtol_pml = expand_optional_layer_values(cfg.corr_block3_rtol_pml, ...
        cfg.corr_num_layers, 'corr_block3_rtol_pml');
    cfg.corr_block3_tail_tol_near = expand_optional_layer_values(cfg.corr_block3_tail_tol_near, ...
        cfg.corr_num_layers, 'corr_block3_tail_tol_near');
    cfg.corr_block3_tail_tol_far = expand_optional_layer_values(cfg.corr_block3_tail_tol_far, ...
        cfg.corr_num_layers, 'corr_block3_tail_tol_far');
    cfg.corr_block3_tail_tol_pml = expand_optional_layer_values(cfg.corr_block3_tail_tol_pml, ...
        cfg.corr_num_layers, 'corr_block3_tail_tol_pml');
    cfg.corr_block3_global_tail_tols = expand_optional_layer_values(cfg.corr_block3_global_tail_tols, ...
        cfg.corr_num_layers, 'corr_block3_global_tail_tols');
    validate_optional_positive(cfg.corr_block3_rtol_near, 'corr_block3_rtol_near');
    validate_optional_positive(cfg.corr_block3_rtol_far, 'corr_block3_rtol_far');
    validate_optional_positive(cfg.corr_block3_rtol_pml, 'corr_block3_rtol_pml');
    validate_optional_positive(cfg.corr_block3_tail_tol_near, 'corr_block3_tail_tol_near');
    validate_optional_positive(cfg.corr_block3_tail_tol_far, 'corr_block3_tail_tol_far');
    validate_optional_positive(cfg.corr_block3_tail_tol_pml, 'corr_block3_tail_tol_pml');
    validate_optional_positive(cfg.corr_block3_global_tail_tols, 'corr_block3_global_tail_tols');
    if isempty(cfg.run_wave_baselines)
        cfg.run_wave_baselines = true;
    end
    if ~isempty(cfg.wave_baseline_max_iter) && ...
            ~(isscalar(cfg.wave_baseline_max_iter) && isfinite(cfg.wave_baseline_max_iter) && ...
            cfg.wave_baseline_max_iter >= 1 && abs(cfg.wave_baseline_max_iter - round(cfg.wave_baseline_max_iter)) < 1e-12)
        error('wave_baseline_max_iter must be empty or a positive integer.');
    end
    if isempty(cfg.helm_completion_tol)
        cfg.helm_completion_tol = cfg.helm_gmres_track_tol;
    end
    if isempty(cfg.shifted_laplacian_restart)
        cfg.shifted_laplacian_restart = cfg.helm_completion_restart;
    end
    if isempty(cfg.shifted_laplacian_maxit)
        cfg.shifted_laplacian_maxit = cfg.helm_completion_maxit;
    end
    if isempty(cfg.shifted_laplacian_tol) && strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        cfg.shifted_laplacian_tol = 1e-15;
    elseif isempty(cfg.shifted_laplacian_tol)
        cfg.shifted_laplacian_tol = cfg.helm_completion_tol;
    end
    if ~(isscalar(cfg.shifted_laplacian_beta) && isfinite(cfg.shifted_laplacian_beta) && cfg.shifted_laplacian_beta > 0)
        error('shifted_laplacian_beta must be a positive finite scalar.');
    end
    if strcmp(cfg.rom_check_target, 'helm')
        cfg.do_reference_check = true;
    end
    if any(strcmp(cfg.online_completion_mode, {'helmholtz', 'shifted_laplacian'}))
        cfg.do_reference_check = true;
    end
    cfg.corr_pod_truncation_rule = normalize_pod_truncation_rule(cfg.corr_pod_truncation_rule);
end

function test_params = generate_interval_test_params(train_params, strategy, rng_seed)
    train_params = train_params(:).';
    if numel(train_params) < 2
        test_params = train_params .* 1.01;
        return;
    end

    strategy = lower(strtrim(strategy));
    lo = train_params(1:end-1);
    hi = train_params(2:end);

    switch strategy
        case {'random_interval', 'random', 'rand'}
            if isempty(rng_seed)
                test_params = lo + rand(size(lo)) .* (hi - lo);
            else
                old_rng = rng;
                rng(rng_seed, 'twister');
                test_params = lo + rand(size(lo)) .* (hi - lo);
                rng(old_rng);
            end

        case {'midpoint', 'middle', 'mid'}
            test_params = 0.5 * (lo + hi);

        otherwise
            error('test_param_strategy=%s must be random_interval or midpoint.', strategy);
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

function validate_block3_rank_cap(value, name)
    if ~(isnumeric(value) && isreal(value) && ...
            all(isinf(value(:)) | (isfinite(value(:)) & value(:) >= 0)))
        error('%s must contain nonnegative finite values or Inf.', name);
    end
end

function validate_optional_positive(value, name)
    if isempty(value)
        return;
    end
    if ~(isnumeric(value) && isreal(value) && all(isfinite(value(:))) && all(value(:) > 0))
        error('%s must be empty or contain positive finite values.', name);
    end
end

function rule = normalize_pod_truncation_rule(rule)
    rule = lower(strtrim(char(rule)));
    switch rule
        case {'relative_frobenius_tail', 'frobenius_tail', 'tail', 'relative_tail'}
            rule = 'relative_frobenius_tail';
        case {'relative_singular_value', 'relative_svd', 'svd', 'relative'}
            rule = 'relative_singular_value';
        otherwise
            error('corr_pod_truncation_rule=%s must be relative_frobenius_tail or relative_singular_value.', rule);
    end
end

function trunc_cfg = make_corr_block3_trunc_cfg(cfg, layer_svd_tol, layer_index)
    rtol = max(layer_svd_tol, 1e-15);
    trunc_cfg = struct();
    trunc_cfg.rank_rtol = rtol;
    if strcmp(cfg.corr_pod_truncation_rule, 'relative_frobenius_tail') && ...
            isempty(cfg.corr_block3_global_tail_tols)
        trunc_cfg.rank_tail_tol = rtol;
    end
    trunc_cfg.rank_cap_near = layer_value(cfg.corr_block3_rank_cap_near, layer_index);
    trunc_cfg.rank_cap_far = layer_value(cfg.corr_block3_rank_cap_far, layer_index);
    trunc_cfg.rank_cap_pml = layer_value(cfg.corr_block3_rank_cap_pml, layer_index);

    if ~isempty(cfg.corr_block3_rtol_near)
        trunc_cfg.rank_rtol_near = layer_value(cfg.corr_block3_rtol_near, layer_index);
    end
    if ~isempty(cfg.corr_block3_rtol_far)
        trunc_cfg.rank_rtol_far = layer_value(cfg.corr_block3_rtol_far, layer_index);
    end
    if ~isempty(cfg.corr_block3_rtol_pml)
        trunc_cfg.rank_rtol_pml = layer_value(cfg.corr_block3_rtol_pml, layer_index);
    end
    if ~isempty(cfg.corr_block3_tail_tol_near)
        trunc_cfg.rank_tail_tol_near = layer_value(cfg.corr_block3_tail_tol_near, layer_index);
    end
    if ~isempty(cfg.corr_block3_tail_tol_far)
        trunc_cfg.rank_tail_tol_far = layer_value(cfg.corr_block3_tail_tol_far, layer_index);
    end
    if ~isempty(cfg.corr_block3_tail_tol_pml)
        trunc_cfg.rank_tail_tol_pml = layer_value(cfg.corr_block3_tail_tol_pml, layer_index);
    end
    if ~isempty(cfg.corr_block3_global_tail_tols)
        trunc_cfg.rank_global_tail_tol = layer_value(cfg.corr_block3_global_tail_tols, layer_index);
        trunc_cfg.rank_global_tail_strategy = cfg.corr_block3_global_tail_strategy;
    end
    if isfield(cfg, 'corr_blockn_rtol') && ~isempty(cfg.corr_blockn_rtol)
        trunc_cfg.rank_rtols = layer_vector_value(cfg.corr_blockn_rtol, layer_index);
    end
    if isfield(cfg, 'corr_blockn_tail_tols') && ~isempty(cfg.corr_blockn_tail_tols)
        trunc_cfg.rank_tail_tols = layer_vector_value(cfg.corr_blockn_tail_tols, layer_index);
    end
    if isfield(cfg, 'corr_blockn_rank_caps') && ~isempty(cfg.corr_blockn_rank_caps)
        trunc_cfg.rank_caps = layer_vector_value(cfg.corr_blockn_rank_caps, layer_index);
    end
end

function prob = build_problem(cfg)
    prob = struct();
    prob.L = cfg.L;
    prob.Lx = cfg.Lx;
    prob.Ly = cfg.Ly;
    prob.omega = cfg.omega;
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

    prob.c_max = max([1.0; cfg.train_params(:); cfg.test_params(:)]);
    dt0 = cfg.CFL * min(prob.hx, prob.hy) / max(prob.c_max, 1e-14);
    [prob.omega_ts, prob.c_w, prob.nt, prob.dt] = match_omega_and_period(prob.omega, dt0);
    prob.T_ts = prob.nt * prob.dt;
    prob.stream_wave_projection = cfg.stream_wave_projection;

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
    prob.pml_wave.c_w_ref = prob.c_w;
    prob.pml_wave.sigma_scaled_for_wave = true;
    prob.pml_wave = prepare_wave_pml_2d(prob.pml_wave, prob.nx, prob.ny, prob.dt);

    [prob.nu, prob.nvx, prob.nvy, prob.nv, prob.nfull] = state_sizes(prob.nx, prob.ny);

    [prob.chi_right_grid, prob.material_meta] = build_parameter_mask(prob, cfg);
    prob.chi_right = prob.chi_right_grid(:);
    prob.material_model = cfg.material_model;

    prob.K_pml = build_pml_stiffness(prob.nx, prob.ny, prob.hx, prob.hy, prob.pml_helm);
    prob.A_base = prob.K_pml + (prob.omega ^ 2) * speye(prob.Ntot);
    prob.A_right = spdiags((prob.omega ^ 2) * double(prob.chi_right), ...
        0, prob.Ntot, prob.Ntot);
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

    chi_grid = false(prob.nx, prob.ny);
    for ix = 1:numel(x_centers)
        for iy = 1:numel(y_centers)
            switch cfg.pc_rod_shape
                case 'circle'
                    x_idx = find(abs(prob.x_int - x_centers(ix)) <= radius);
                    y_idx = find(abs(prob.y_int - y_centers(iy)) <= radius);
                    [dx, dy] = ndgrid(prob.x_int(x_idx) - x_centers(ix), ...
                        prob.y_int(y_idx) - y_centers(iy));
                    cell_mask = dx.^2 + dy.^2 <= radius^2;
                case 'square'
                    x_idx = find(abs(prob.x_int - x_centers(ix)) <= square_halfwidth);
                    y_idx = find(abs(prob.y_int - y_centers(iy)) <= square_halfwidth);
                    cell_mask = true(numel(x_idx), numel(y_idx));
            end
            chi_grid(x_idx, y_idx) = chi_grid(x_idx, y_idx) | cell_mask;
        end
    end
    phys_x = prob.x_int >= 0 & prob.x_int <= cfg.Lx;
    phys_y = prob.y_int >= 0 & prob.y_int <= cfg.Ly;
    chi_grid = chi_grid & (phys_x & phys_y.');

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

function block3_partition = build_helm_u_block3_partition(prob, cfg)
    switch cfg.block3_partition_mode
        case 'pc_refined'
            block3_partition = build_pc_refined_partition(prob,cfg);
            return;
        case 'halfspace_refined'
            block3_partition = build_halfspace_refined_partition(prob, cfg);
            fprintf('Half-space refined DD: %d blocks, sizes=%s\n', ...
                numel(block3_partition.blocks), ...
                mat2str([block3_partition.blocks.count]));
            return;
        case 'halfspace_material'
            block3_partition = build_halfspace_material_block3_partition(prob, cfg);
            fprintf(['block3 partition (u-space, halfspace material): left=%d, ' ...
                'right=%d, pml=%d\n'], block3_partition.count_near, ...
                block3_partition.count_far, block3_partition.count_pml);
            return;
        case 'halfspace_source_interface'
            block3_partition = build_halfspace_source_interface_block3_partition(prob, cfg);
            fprintf(['block3 partition (u-space, halfspace source/interface): active=%d, ' ...
                'rest=%d, pml=%d\n'], block3_partition.count_near, ...
                block3_partition.count_far, block3_partition.count_pml);
            return;
        case 'pc_physical_directional_pml'
            block3_partition = build_pc_directional_pml_block3_partition(prob, cfg, false);
            fprintf(['block3 partition (u-space, pc physical/directional PML): phys=%d, ' ...
                'lr-pml=%d, tb-corner-pml=%d\n'], block3_partition.count_near, ...
                block3_partition.count_far, block3_partition.count_pml);
            return;
        case 'pc_channel_lr_pml'
            block3_partition = build_pc_directional_pml_block3_partition(prob, cfg, true);
            fprintf(['block3 partition (u-space, pc channel/directional PML): channel=%d, ' ...
                'cladding+tb-pml=%d, lr-pml=%d\n'], block3_partition.count_near, ...
                block3_partition.count_far, block3_partition.count_pml);
            return;
        case 'pc_core_cladding_lr_tb_pml'
            block3_partition = build_pc_core_cladding_lr_tb_pml_partition(prob, cfg);
            counts = [block3_partition.blocks.count];
            fprintf(['block4 partition (u-space, pc core/cladding/directional PML): ' ...
                'core=%d, cladding=%d, lr-pml=%d, tb-corner-pml=%d\n'], counts);
            return;
    end

    if any(strcmp(cfg.material_model, {'photonic_crystal', 'pc', 'double_layer', 'two_layer'})) && ...
            any(strcmp(cfg.block3_partition_mode, {'pc_waveguide', 'waveguide'}))
        block3_partition = build_pc_waveguide_block3_partition(prob, cfg);
        fprintf(['block3 partition (u-space, pc waveguide): near=%d, ' ...
            'far=%d, pml=%d, halfwidth=%.3f\n'], ...
            block3_partition.count_near, block3_partition.count_far, ...
            block3_partition.count_pml, cfg.block3_waveguide_halfwidth);
        return;
    end

    grid = struct();
    grid.nx = prob.nx;
    grid.ny = prob.ny;
    grid.x = prob.x;
    grid.y = prob.y;
    grid.npml = prob.npml;
    grid.center = cfg.block3_center;
    grid.r_near = cfg.block3_r_near;
    block3_partition = rom_reduced_space_utils('build_block3_partition', grid);
    fprintf('block3 partition (u-space): near=%d, far=%d, pml=%d\n', ...
        block3_partition.count_near, block3_partition.count_far, block3_partition.count_pml);
end

function part = build_halfspace_material_block3_partition(prob, cfg)
    [X, ~] = ndgrid(prob.x_int, prob.y_int);
    phys_mask = false(prob.nx, prob.ny);
    phys_mask(prob.I, prob.J) = true;
    pml_mask = ~phys_mask;
    near_mask = phys_mask & (X < cfg.interface_x);
    far_mask = phys_mask & ~near_mask;
    part = make_block3_partition_from_masks(near_mask, far_mask, pml_mask, 'halfspace_material');
end

function part = build_halfspace_source_interface_block3_partition(prob, cfg)
    [X, Y] = ndgrid(prob.x_int, prob.y_int);
    phys_mask = false(prob.nx, prob.ny);
    phys_mask(prob.I, prob.J) = true;
    pml_mask = ~phys_mask;
    source_x = prob.x_phys(cfg.source_col_phys);
    source_y = prob.y_phys(cfg.source_row_phys);
    source_radius = 4 * max(prob.hx, prob.hy);
    interface_halfwidth = max(0.10 * cfg.Lx, 8 * prob.hx);
    source_mask = hypot(X - source_x, Y - source_y) <= source_radius;
    interface_mask = abs(X - cfg.interface_x) <= interface_halfwidth;
    near_mask = phys_mask & (source_mask | interface_mask);
    far_mask = phys_mask & ~near_mask;
    part = make_block3_partition_from_masks(near_mask, far_mask, pml_mask, 'halfspace_source_interface');
end

function part = build_pc_waveguide_block3_partition(prob, cfg)
    [~, Y] = ndgrid(prob.x_int, prob.y_int);
    phys_mask = false(prob.nx, prob.ny);
    phys_mask(prob.I, prob.J) = true;
    near_mask = phys_mask & (abs(Y - 0.5 * cfg.Ly) <= cfg.block3_waveguide_halfwidth);
    pml_mask = ~phys_mask;
    far_mask = phys_mask & ~near_mask;

    part = struct();
    part.n = prob.nx * prob.ny;
    part.nx = prob.nx;
    part.ny = prob.ny;
    part.idx_near = find(near_mask(:));
    part.idx_far = find(far_mask(:));
    part.idx_pml = find(pml_mask(:));
    part.count_near = numel(part.idx_near);
    part.count_far = numel(part.idx_far);
    part.count_pml = numel(part.idx_pml);
    part.mode = 'pc_waveguide';
end

function part = build_pc_directional_pml_block3_partition(prob, cfg, split_channel)
    [~, Y] = ndgrid(prob.x_int, prob.y_int);
    phys_mask = false(prob.nx, prob.ny);
    phys_mask(prob.I, prob.J) = true;
    ii = repmat((1:prob.nx).', 1, prob.ny);
    jj = repmat(1:prob.ny, prob.nx, 1);
    pml_mask = ~phys_mask;
    lr_pml_mask = pml_mask & ((ii <= prob.npml) | (ii > prob.nx - prob.npml));
    tb_corner_pml_mask = pml_mask & ~lr_pml_mask;

    if split_channel
        channel_mask = phys_mask & (abs(Y - 0.5 * cfg.Ly) <= cfg.block3_waveguide_halfwidth);
        near_mask = channel_mask;
        far_mask = (phys_mask & ~channel_mask) | tb_corner_pml_mask;
        pml_block_mask = lr_pml_mask;
        mode = 'pc_channel_lr_pml';
    else
        near_mask = phys_mask;
        far_mask = lr_pml_mask;
        pml_block_mask = tb_corner_pml_mask;
        mode = 'pc_physical_directional_pml';
    end

    part = make_block3_partition_from_masks(near_mask, far_mask, pml_block_mask, mode);
end

function part = build_pc_core_cladding_lr_tb_pml_partition(prob, cfg)
    [~, Y] = ndgrid(prob.x_int, prob.y_int);
    phys_mask = false(prob.nx, prob.ny);
    phys_mask(prob.I, prob.J) = true;
    ii = repmat((1:prob.nx).', 1, prob.ny);
    jj = repmat(1:prob.ny, prob.nx, 1);
    pml_mask = ~phys_mask;

    core_mask = phys_mask & (abs(Y - 0.5 * cfg.Ly) <= cfg.block3_waveguide_halfwidth);
    cladding_mask = phys_mask & ~core_mask;
    lr_pml_mask = pml_mask & ((ii <= prob.npml) | (ii > prob.nx - prob.npml)) & ...
        (jj > prob.npml) & (jj <= prob.ny - prob.npml);
    tb_corner_pml_mask = pml_mask & ~lr_pml_mask;

    masks = {core_mask, cladding_mask, lr_pml_mask, tb_corner_pml_mask};
    names = {'core', 'cladding', 'lr_pml', 'tb_corner_pml'};
    part = make_blockn_partition_from_masks(masks, names, 'pc_core_cladding_lr_tb_pml');
end

function part = make_block3_partition_from_masks(near_mask, far_mask, pml_mask, mode)
    if any((near_mask(:) & far_mask(:)) | (near_mask(:) & pml_mask(:)) | (far_mask(:) & pml_mask(:)))
        error('Block3 masks for mode %s overlap.', mode);
    end
    covered = near_mask | far_mask | pml_mask;
    if ~all(covered(:))
        error('Block3 masks for mode %s do not cover the full grid.', mode);
    end

    part = struct();
    [part.nx, part.ny] = size(near_mask);
    part.n = part.nx * part.ny;
    part.idx_near = find(near_mask(:));
    part.idx_far = find(far_mask(:));
    part.idx_pml = find(pml_mask(:));
    part.count_near = numel(part.idx_near);
    part.count_far = numel(part.idx_far);
    part.count_pml = numel(part.idx_pml);
    part.mode = mode;
end

function part = make_blockn_partition_from_masks(masks, names, mode)
    if numel(masks) ~= numel(names)
        error('BlockN masks and names must have the same length.');
    end
    base_size = size(masks{1});
    covered = false(base_size);
    for b = 1:numel(masks)
        if ~isequal(size(masks{b}), base_size)
            error('BlockN mask sizes are inconsistent for mode %s.', mode);
        end
        if any(covered(:) & masks{b}(:))
            error('BlockN masks for mode %s overlap.', mode);
        end
        covered = covered | masks{b};
    end
    if ~all(covered(:))
        error('BlockN masks for mode %s do not cover the full grid.', mode);
    end

    part = struct();
    [part.nx, part.ny] = size(masks{1});
    part.n = part.nx * part.ny;
    part.mode = mode;
    part.blocks = repmat(struct('name', '', 'idx', [], 'count', 0), numel(masks), 1);
    for b = 1:numel(masks)
        part.blocks(b).name = names{b};
        part.blocks(b).idx = find(masks{b}(:));
        part.blocks(b).count = numel(part.blocks(b).idx);
    end
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
    [x_ref, u_ref, b_vec, info] = solve_reference_solution(prob, cfg, rhs, c_right);
    case_result = struct();
    case_result.x_ref = x_ref;
    case_result.u_ref = u_ref;
    case_result.b_vec = b_vec;
    case_result.info = info;
    case_result.log_line = sprintf('case ??: c_right=%.4f, solve=%.3fs, recover=%.3fs, iter=%3d, relres=%.3e', ...
        c_right, info.solve_time, info.recovery_time, info.iter_count, info.relres);
end

function [x_ref, u_ref, b_vec, info] = solve_reference_solution(prob, cfg, rhs, c_right)
    switch cfg.reference_mode
        case 'waveholtz'
            [x0, ~] = build_waveholtz_initial_guess(prob, cfg, rhs, c_right);
            [x_ref, u_ref, b_vec, info] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, c_right, x0);

        case 'helm_lift'
            tref = tic;
            Ahelm = build_helmholtz_operator(prob, c_right);
            u_ref = Ahelm \ rhs;
            x_ref = lift_frequency_solution_to_real_state(prob, u_ref, c_right);
            b_vec = zeros(prob.nfull, 1);
            ref_time = toc(tref);
            info = struct();
            info.flag = 0;
            info.relres = 0;
            info.iter = [0, 0];
            info.iter_count = 0;
            info.resvec = 0;
            info.solve_time = ref_time;
            info.recovery_time = 0;
    end
end

function [x0, u_helm] = build_waveholtz_initial_guess(prob, cfg, rhs, c_right)
    u_helm = [];
    switch cfg.wh_initial_guess
        case 'zero'
            x0 = zeros(prob.nfull, 1);
        case 'helm_lift'
            Ahelm = build_helmholtz_operator(prob, c_right);
            u_helm = Ahelm \ rhs;
            x0 = lift_frequency_solution_to_real_state(prob, u_helm, c_right);
    end
end

function [x_sol, uhat, b_vec, info] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, c_right, x0)
    if nargin < 5 || isempty(x0)
        x0 = zeros(prob.nfull, 1);
    end
    [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right);

    tsolve = tic;
    [x_sol, flag, relres, iter, resvec] = gmres_restarted_verbose( ...
        Afun, b_vec, cfg.wh_gmres_restart, cfg.wh_tol, cfg.wh_gmres_maxit, ...
        x0, cfg.wh_verbose_gmres, cfg.wh_gmres_reorth);
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
    if prob.stream_wave_projection
        [~, ~, uhat] = solve_wave_one_period_piecewise(prob, x_state, rhs, c_right, [], false);
    else
        Wconv = solve_wave_one_period_piecewise(prob, x_state, rhs, c_right);
        t_grid = (0:prob.nt).' * prob.dt;
        q_cos = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, cos(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
        q_sin = (2.0 / prob.T_ts) * trapz(t_grid, bsxfun(@times, sin(prob.omega_ts * t_grid), Wconv(:, 1:prob.nu)), 1).';
        uhat = q_cos + 1i * q_sin;
    end
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
    sx_full = prob.pml_wave.sx_full;
    sy_full = prob.pml_wave.sy_full;
    sigma_x_u = sx_full(2:end-1, 2:end-1);
    sigma_y_u = sy_full(2:end-1, 2:end-1);
    sigma_x_vx = 0.5 * (sx_full(1:end-1, 2:end-1) + sx_full(2:end, 2:end-1));
    sigma_y_vy = 0.5 * (sy_full(2:end-1, 1:end-1) + sy_full(2:end-1, 2:end));
    c_sq = build_piecewise_c_squared(prob, c_right);
    c_sq_full = ones(prob.nx + 2, prob.ny + 2);
    c_sq_full(2:end-1, 2:end-1) = c_sq;
    theta_coeff = (sx_full(2:end-1, 2:end-1) .* sy_full(2:end-1, 2:end-1)) .* (prob.c_w ^ 2) ./ c_sq_full(2:end-1, 2:end-1);
end

function direct_rom = build_direct_fullstate_rom(prob, cfg, rhs, train_ref)
    if ~cfg.use_direct_rom_init && ~cfg.build_direct_rom_when_unused
        ns = numel(train_ref.params);
        pod = struct('basis', zeros(prob.nfull, 0), ...
            'left_modes', zeros(prob.nfull, 0), ...
            'right_modes', zeros(ns, 0), ...
            'singular_values', zeros(0, 1), ...
            'rank_used', 0, 'rank_sigma_tol', [], ...
            'singular_value_tol', cfg.pod_singular_value_tol, ...
            'r90', 0, 'r95', 0, 'r99', 0);
    else
        pod = build_pod_basis_custom(train_ref.x_ref_snapshots, cfg.pod_energy_target, ...
            cfg.pod_singular_value_tol, cfg.pod_min_rank, cfg.pod_max_rank);
    end
    V = pod.basis;
    direct_rom = struct();
    direct_rom.pod = pod;
    direct_rom.basis = V;
    direct_rom.train_x_init = zeros(prob.nfull, numel(train_ref.params));
    direct_rom.train_u_init = zeros(prob.Ntot, numel(train_ref.params));
    direct_rom.train_direct_time = zeros(numel(train_ref.params), 1);
    if ~cfg.use_direct_rom_init
        return;
    end
    ns = numel(train_ref.params);
    train_params = train_ref.params;
    use_parallel = should_parallelize_cases(cfg.parallel_train, ns, cfg.parallel_train_min_cases);
    basis_rom = struct('basis', V);
    case_results = cell(ns, 1);
    if use_parallel
        parfor k = 1:ns
            [x_direct, u_direct, solve_time] = solve_direct_fullstate_rom_at_param(prob, cfg, rhs, basis_rom, train_params(k));
            case_results{k} = struct('x_direct', x_direct, 'u_direct', u_direct, 'solve_time', solve_time);
        end
    else
        for k = 1:ns
            [x_direct, u_direct, solve_time] = solve_direct_fullstate_rom_at_param(prob, cfg, rhs, basis_rom, train_params(k));
            case_results{k} = struct('x_direct', x_direct, 'u_direct', u_direct, 'solve_time', solve_time);
        end
    end
    for k = 1:ns
        direct_rom.train_x_init(:, k) = case_results{k}.x_direct;
        direct_rom.train_u_init(:, k) = case_results{k}.u_direct;
        direct_rom.train_direct_time(k) = case_results{k}.solve_time;
    end
end

function [x_direct, u_direct, solve_time, op_apply_count] = solve_direct_fullstate_rom_at_param(prob, cfg, rhs, direct_rom, c_right)
    [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right);
    tsolve = tic;
    Ar = assemble_reduced_waveholtz_operator(Afun, direct_rom.basis);
    br = direct_rom.basis' * b_vec;
    coeff = Ar \ br;
    x_direct = direct_rom.basis * coeff;
    solve_time = toc(tsolve);
    op_apply_count = size(direct_rom.basis, 2);
    u_direct = recover_u_phasor_timeproj(prob, x_direct, rhs, c_right);
end

function print_direct_fullstate_rom_summary(direct_rom)
    fprintf('direct snapshot matrix size = %d x %d\n', size(direct_rom.pod.left_modes, 1), size(direct_rom.pod.right_modes, 1));
    fprintf('direct basis rank used      = %d\n', direct_rom.pod.rank_used);
    if ~isempty(direct_rom.pod.rank_sigma_tol)
        fprintf('direct relative singular value cutoff = %.3e, cutoff rank = %d\n', ...
            direct_rom.pod.singular_value_tol, direct_rom.pod.rank_sigma_tol);
    end
    fprintf('direct energy ranks r90/r95/r99 = %d / %d / %d\n', direct_rom.pod.r90, direct_rom.pod.r95, direct_rom.pod.r99);
end

function corr_rom = train_correction_rom_layers(prob, cfg, rhs, train_ref, direct_rom, block3_partition)
    ns = numel(train_ref.params);
    train_params = train_ref.params;
    u_ref_snapshots = train_ref.u_ref_snapshots;
    x_ref_snapshots = train_ref.x_ref_snapshots;
    nlayers = cfg.corr_num_layers;
    layer_svd_tols = expand_layer_parameter(cfg.corr_pod_singular_value_tols, cfg.corr_pod_singular_value_tol, nlayers, 'corr_pod_singular_value_tols');
    corr_rom = struct();
    corr_rom.num_layers = nlayers;
    corr_rom.layers = cell(nlayers, 1);
    corr_rom.x_init_train = zeros(prob.nfull, ns);
    corr_rom.u_init_train = direct_rom.train_u_init;
    corr_rom.u_train_after_layer = cell(nlayers, 1);
    corr_rom.x_train_after_layer = cell(nlayers, 1);

    if cfg.use_direct_rom_init
        x_curr = direct_rom.train_x_init;
        u_init = direct_rom.train_u_init;
    else
        x_curr = zeros(prob.nfull, ns);
        u_init = zeros(prob.Ntot, ns);
        corr_rom.u_init_train = zeros(prob.Ntot, ns);
    end
    corr_rom.x_init_train = x_curr;
    corr_rom.u_init_train = u_init;

    train_state = initialize_trajectory_aware_training_state(prob, rhs, train_params, u_init, u_ref_snapshots);
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

        [basis_snapshots, snapshot_column_norms] = prepare_correction_snapshots_for_basis(eta_snapshots, cfg);
        pod = build_pod_basis_custom(basis_snapshots, cfg.corr_pod_energy_target, layer_svd_tols(j), ...
            cfg.corr_pod_min_rank, cfg.corr_pod_max_rank, cfg.corr_pod_truncation_rule);
        layer = struct();
        layer.index = j;
        layer.singular_value_tol = layer_svd_tols(j);
        layer.eta_snapshots = eta_snapshots;
        layer.basis_snapshots_normalized = cfg.corr_normalize_snapshots;
        layer.snapshot_column_norms = snapshot_column_norms;
        layer.pod = pod;
        if strcmp(cfg.corr_basis_style, 'block3')
            if isempty(block3_partition)
                error('corr_basis_style=block3 but block3_partition is empty.');
            end
            layer.basis_space = rom_reduced_space_utils('build_block3_space', basis_snapshots, block3_partition, ...
                make_corr_block3_trunc_cfg(cfg, layer_svd_tols(j), j));
        else
            layer.basis_space = rom_reduced_space_utils('make_dense_space', pod.basis);
        end
        layer.basis_u = pod.basis;
        layer.space_dim = rom_reduced_space_utils('dimension', layer.basis_space);
        layer.Ar_base = rom_reduced_space_utils('assemble_reduced_matrix', prob.A_base, layer.basis_space);
        layer.Ar_right = rom_reduced_space_utils('assemble_reduced_matrix', prob.A_right, layer.basis_space);
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
        q_curr = train_state.q_curr;
        use_parallel = should_parallelize_cases(cfg.parallel_train, ns, cfg.parallel_train_min_cases);
        case_results = cell(ns, 1);
        if use_parallel
            parfor k = 1:ns
                c_right = train_params(k);
                [u_local, info_train] = layered_fgmres_precorrection(prob, cfg, rhs, c_right, u_init(:, k), prefix_rom);
                x_local = lift_frequency_solution_to_real_state(prob, u_local, c_right);
                freq_err = norm(u_local - u_ref_snapshots(:, k)) / max(norm(u_ref_snapshots(:, k)), 1e-16);
                state_err = norm(x_local - x_ref_snapshots(:, k)) / max(norm(x_ref_snapshots(:, k)), 1e-16);
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
                    'q_norm', norm(q_curr(:, k)));
            end
        else
            for k = 1:ns
                c_right = train_params(k);
                [u_local, info_train] = layered_fgmres_precorrection(prob, cfg, rhs, c_right, u_init(:, k), prefix_rom);
                x_local = lift_frequency_solution_to_real_state(prob, u_local, c_right);
                freq_err = norm(u_local - u_ref_snapshots(:, k)) / max(norm(u_ref_snapshots(:, k)), 1e-16);
                state_err = norm(x_local - x_ref_snapshots(:, k)) / max(norm(x_ref_snapshots(:, k)), 1e-16);
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
                    'q_norm', norm(q_curr(:, k)));
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

        layer.trajectory_diagnostics = train_state.diagnostics(:,j);
        corr_rom.layers{j} = layer;
        corr_rom.u_train_after_layer{j} = u_after;
        corr_rom.x_train_after_layer{j} = x_after;
        fprintf(['layer %2d/%2d: %s tol = %.1e, error snapshot size = %d x %d, basis rank = %d, ' ...
            'train relres mean = %.3e, train freq err mean = %.3e, train state err mean = %.3e\n'], ...
            j, nlayers, pod.singular_value_tol_mode, layer.singular_value_tol, size(eta_snapshots, 1), size(eta_snapshots, 2), layer.space_dim, ...
            mean(layer.train_fgmres_relres), mean(layer.train_freq_error_after), mean(layer.train_state_error_after));
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

function print_correction_rom_summary(corr_rom)
    if corr_rom.num_layers == 0
        fprintf('no layered Helmholtz ROM steps requested.\n');
        return;
    end
    for j = 1:corr_rom.num_layers
        pod = corr_rom.layers{j}.pod;
        fprintf('layered FGMRES ROM layer %2d: %s tol = %.1e, rank = %d, r90/r95/r99 = %d / %d / %d, train relres mean = %.3e\n', ...
            j, pod.singular_value_tol_mode, corr_rom.layers{j}.singular_value_tol, corr_rom.layers{j}.space_dim, pod.r90, pod.r95, pod.r99, mean(corr_rom.layers{j}.train_fgmres_relres));
        storage = block_storage_stats_from_layer(corr_rom.layers{j});
        if ~isempty(storage)
            fprintf(['  block ranks [%s] = %s, dofs = %s, ' ...
                'storage dense/block = %.3g / %.3g, saving = %.3fx (%.1f%% less)\n'], ...
                strjoin(storage.labels, ' '), mat2str(storage.ranks), mat2str(storage.dofs), ...
                storage.dense_storage, storage.block_storage, ...
                storage.saving_ratio, 100 * storage.saved_fraction);
            if isfinite(storage.global_tail_tol)
                fprintf('  block global tail budget = %.2e, achieved = %.3e, local tails = [%s]\n', ...
                    storage.global_tail_tol, storage.global_tail, ...
                    sprintf('%.2e ', storage.local_tails));
            end
        end
    end
end

function nr_corr_rom = train_nonreplayed_correction_rom_layers(prob, cfg, rhs, train_ref, direct_rom, ta_corr_rom)
    if ~strcmp(cfg.corr_basis_style, 'global')
        error('The equal-rank NR-ROM comparison currently requires corr_basis_style=global.');
    end

    ns = numel(train_ref.params);
    train_params = train_ref.params;
    nlayers = ta_corr_rom.num_layers;
    if cfg.use_direct_rom_init
        u_init = direct_rom.train_u_init;
    else
        u_init = zeros(prob.Ntot, ns);
    end

    nr_corr_rom = struct();
    nr_corr_rom.num_layers = nlayers;
    nr_corr_rom.layers = cell(nlayers, 1);
    nr_corr_rom.x_init_train = ta_corr_rom.x_init_train;
    nr_corr_rom.u_init_train = u_init;
    nr_corr_rom.u_train_after_layer = cell(nlayers, 1);
    nr_corr_rom.x_train_after_layer = cell(nlayers, 1);
    nr_corr_rom.training_mode = 'non-replayed-plain-arnoldi';

    Q_hist = cell(ns, 1);
    q_curr = zeros(prob.Ntot, ns);
    for k = 1:ns
        Ahelm = build_helmholtz_operator(prob, train_params(k));
        r0 = rhs - Ahelm * u_init(:, k);
        if norm(r0) > 1e-15
            q_curr(:, k) = r0 / norm(r0);
        end
        Q_hist{k} = q_curr(:, k);
    end

    for j = 1:nlayers
        eta_snapshots = zeros(prob.Ntot, ns);
        q_snapshots = q_curr;
        next_q = zeros(prob.Ntot, ns);
        use_parallel = should_parallelize_cases(cfg.parallel_train, ns, cfg.parallel_train_min_cases);
        if use_parallel
            parfor k = 1:ns
                Ahelm = build_helmholtz_operator(prob, train_params(k));
                if norm(q_snapshots(:, k)) > 1e-15
                    eta_snapshots(:, k) = Ahelm \ q_snapshots(:, k);
                end
            end
        else
            for k = 1:ns
                Ahelm = build_helmholtz_operator(prob, train_params(k));
                if norm(q_snapshots(:, k)) > 1e-15
                    eta_snapshots(:, k) = Ahelm \ q_snapshots(:, k);
                end
            end
        end

        target_rank = layer_effective_space_dim(ta_corr_rom.layers{j});
        [basis_snapshots, snapshot_column_norms] = prepare_correction_snapshots_for_basis(eta_snapshots, cfg);
        pod = build_pod_basis_custom(basis_snapshots, cfg.corr_pod_energy_target, [], ...
            target_rank, target_rank, cfg.corr_pod_truncation_rule);
        layer = struct();
        layer.index = j;
        layer.singular_value_tol = [];
        layer.eta_snapshots = eta_snapshots;
        layer.q_snapshots = q_snapshots;
        layer.basis_snapshots_normalized = cfg.corr_normalize_snapshots;
        layer.snapshot_column_norms = snapshot_column_norms;
        layer.pod = pod;
        layer.basis_space = rom_reduced_space_utils('make_dense_space', pod.basis);
        layer.basis_u = pod.basis;
        layer.space_dim = rom_reduced_space_utils('dimension', layer.basis_space);
        layer.Ar_base = rom_reduced_space_utils('assemble_reduced_matrix', prob.A_base, layer.basis_space);
        layer.Ar_right = rom_reduced_space_utils('assemble_reduced_matrix', prob.A_right, layer.basis_space);
        layer.train_fgmres_relres = NaN(ns, 1);
        layer.train_freq_error_after = NaN(ns, 1);
        layer.train_state_error_after = NaN(ns, 1);
        layer.q_snapshot_norm = vecnorm(q_snapshots, 2, 1).';
        nr_corr_rom.layers{j} = layer;

        for k = 1:ns
            Ahelm = build_helmholtz_operator(prob, train_ref.params(k));
            w = Ahelm * q_snapshots(:, k);
            Qk = Q_hist{k};
            for i = 1:size(Qk, 2)
                w = w - (Qk(:, i)' * w) * Qk(:, i);
            end
            if norm(w) > 1e-15
                next_q(:, k) = w / norm(w);
            end
            Q_hist{k}(:, end + 1) = next_q(:, k);
        end
        q_curr = next_q;
        fprintf('NR-ROM layer %2d/%2d: equal rank = %d, snapshot size = %d x %d\n', ...
            j, nlayers, layer.space_dim, size(eta_snapshots, 1), size(eta_snapshots, 2));
    end
end

function test = evaluate_correction_rom_online(prob, cfg, rhs, direct_rom, corr_rom)
    ns = numel(cfg.test_params);
    nlayers = corr_rom.num_layers;
    test = initialize_online_test_struct(prob, ns, nlayers, cfg.do_reference_check);
    test.params = cfg.test_params(:);
    test.online_true_helm_relres_hist = cell(ns,1);
    test.plain_true_helm_relres_hist = cell(ns,1);
    use_parallel = should_parallelize_online_eval(cfg, ns);
    fprintf('online test parallel = %d (%d cases)\n', use_parallel, ns);

    case_results = cell(ns, 1);
    if use_parallel
        parfor k = 1:ns
            case_results{k} = evaluate_one_online_case(prob, cfg, rhs, direct_rom, corr_rom, cfg.test_params(k));
        end
    else
        for k = 1:ns
            case_results{k} = evaluate_one_online_case(prob, cfg, rhs, direct_rom, corr_rom, cfg.test_params(k));
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
    test.direct_u = zeros(prob.Ntot, ns);
    test.direct_x = zeros(prob.nfull, ns);
    test.direct_time = zeros(ns, 1);
    test.direct_op_apply = zeros(ns, 1);
    test.direct_fp_u = zeros(prob.Ntot, ns);
    test.direct_fp_x = zeros(prob.nfull, ns);
    test.direct_fp_time = zeros(ns, 1);
    test.direct_fp_total_time = zeros(ns, 1);
    test.direct_fp_iter = zeros(ns, 1);
    test.direct_fp_relres_final = zeros(ns, 1);
    test.direct_fp_op_apply = zeros(ns, 1);
    test.direct_fp_relres_hist = cell(ns, 1);
    test.direct_fp_step_hist = cell(ns, 1);
    test.direct_fp_stop_reason = cell(ns, 1);
    test.plain_fp_u = zeros(prob.Ntot, ns);
    test.plain_fp_x = zeros(prob.nfull, ns);
    test.plain_fp_time = zeros(ns, 1);
    test.plain_preconditioner_setup_time = NaN(ns, 1);
    test.plain_completion_solve_time = NaN(ns, 1);
    test.plain_preconditioner_factor_bytes = NaN(ns, 1);
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
    test.online_preconditioner_setup_time = NaN(ns, 1);
    test.online_completion_solve_time = NaN(ns, 1);
    test.online_preconditioner_factor_bytes = NaN(ns, 1);
    test.online_total_time = zeros(ns, 1);
    test.online_fp_iter = zeros(ns, 1);
    test.online_fp_relres_final = zeros(ns, 1);
    test.online_fp_op_apply = zeros(ns, 1);
    test.online_fp_relres_hist = cell(ns, 1);
    test.online_fp_step_hist = cell(ns, 1);
    test.online_fp_stop_reason = cell(ns, 1);
    test.online_precorr_relres_hist = cell(ns, 1);
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

    test.direct_err_wave_full = zeros(ns, 1);
    test.direct_err_wave_phys = zeros(ns, 1);
    test.direct_err_helm_full = zeros(ns, 1);
    test.direct_err_helm_phys = zeros(ns, 1);
    test.direct_fp_err_wave_full = zeros(ns, 1);
    test.direct_fp_err_wave_phys = zeros(ns, 1);
    test.direct_fp_err_helm_full = zeros(ns, 1);
    test.direct_fp_err_helm_phys = zeros(ns, 1);
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

function case_result = evaluate_one_online_case(prob, cfg, rhs, direct_rom, corr_rom, c_right)
    nlayers = corr_rom.num_layers;
    case_result = struct();

    [x_ref, u_ref, b_vec, info_ref] = solve_reference_solution(prob, cfg, rhs, c_right);
    case_result.wave_x_ref = x_ref;
    case_result.wave_u_ref = u_ref;
    case_result.wave_time = info_ref.solve_time + info_ref.recovery_time;
    case_result.wave_iter = info_ref.iter_count;

    if cfg.do_reference_check
        if strcmp(cfg.reference_mode, 'helm_lift')
            case_result.helm_u_ref = u_ref;
            case_result.helm_time = 0;
        else
            alpha = (1 / c_right^2) - 1;
            tref = tic;
            case_result.helm_u_ref = (prob.A_base + alpha * prob.A_right) \ rhs;
            case_result.helm_time = toc(tref);
        end
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

    if cfg.use_direct_rom_init
        tdir = tic;
        [x_direct, u_direct, ~, direct_apply_count] = solve_direct_fullstate_rom_at_param(prob, cfg, rhs, direct_rom, c_right);
        case_result.direct_time = toc(tdir);
        case_result.direct_op_apply = direct_apply_count;
    else
        x_direct = zeros(prob.nfull, 1);
        u_direct = zeros(prob.Ntot, 1);
        direct_apply_count = 0;
        case_result.direct_time = 0;
        case_result.direct_op_apply = 0;
    end
    case_result.direct_x = x_direct;
    case_result.direct_u = u_direct;
    [case_result.direct_err_wave_full, case_result.direct_err_wave_phys] = compute_solution_errors(u_direct, u_ref, prob);
    if cfg.do_reference_check
        [case_result.direct_err_helm_full, case_result.direct_err_helm_phys] = compute_solution_errors(u_direct, case_result.helm_u_ref, prob);
    else
        case_result.direct_err_helm_full = NaN;
        case_result.direct_err_helm_phys = NaN;
    end

    if cfg.use_direct_rom_init
        x_init_online = x_direct;
        u_init_online = u_direct;
    else
        x_init_online = zeros(prob.nfull, 1);
        u_init_online = zeros(prob.Ntot, 1);
    end

    need_wave_operator = (strcmp(cfg.online_completion_mode, 'waveholtz') && ...
        isempty(cfg.online_helm_accept_tol)) || ...
        (cfg.run_wave_baselines && ~strcmp(cfg.online_completion_mode, 'shifted_laplacian'));
    if need_wave_operator
        [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right);
    else
        Afun = [];
    end

    if cfg.run_wave_baselines && cfg.use_direct_rom_init
        [x_direct_fp, info_direct_fp] = run_online_baseline_completion( ...
            prob, cfg, rhs, c_right, b_vec, Afun, x_init_online, u_init_online, true);
        case_result.direct_fp_x = x_direct_fp;
        case_result.direct_fp_u = info_direct_fp.u_final;
        case_result.direct_fp_time = info_direct_fp.total_time;
        case_result.direct_fp_total_time = case_result.direct_time + info_direct_fp.total_time;
        case_result.direct_fp_iter = info_direct_fp.iter_count;
        case_result.direct_fp_relres_final = info_direct_fp.relres_final;
        case_result.direct_fp_op_apply = direct_apply_count + info_direct_fp.op_apply_count;
        case_result.direct_fp_relres_hist = info_direct_fp.relres_hist;
        case_result.direct_fp_step_hist = info_direct_fp.step_hist;
        case_result.direct_fp_stop_reason = info_direct_fp.stop_reason;
        [case_result.direct_fp_err_wave_full, case_result.direct_fp_err_wave_phys] = compute_solution_errors(info_direct_fp.u_final, u_ref, prob);

        if cfg.do_reference_check
            [case_result.direct_fp_err_helm_full, case_result.direct_fp_err_helm_phys] = compute_solution_errors(info_direct_fp.u_final, case_result.helm_u_ref, prob);
        else
            case_result.direct_fp_err_helm_full = NaN;
            case_result.direct_fp_err_helm_phys = NaN;
        end
    else
        case_result.direct_fp_x = zeros(prob.nfull, 1);
        case_result.direct_fp_u = zeros(prob.Ntot, 1);
        case_result.direct_fp_time = NaN;
        case_result.direct_fp_total_time = NaN;
        case_result.direct_fp_iter = NaN;
        case_result.direct_fp_relres_final = NaN;
        case_result.direct_fp_op_apply = NaN;
        case_result.direct_fp_relres_hist = [];
        case_result.direct_fp_step_hist = [];
        case_result.direct_fp_stop_reason = '';
        case_result.direct_fp_err_wave_full = NaN;
        case_result.direct_fp_err_wave_phys = NaN;
        case_result.direct_fp_err_helm_full = NaN;
        case_result.direct_fp_err_helm_phys = NaN;
    end

    if cfg.run_wave_baselines
        [x_plain, info_plain] = run_online_baseline_completion( ...
            prob, cfg, rhs, c_right, b_vec, Afun, zeros(prob.nfull, 1), zeros(prob.Ntot, 1), true);
        case_result.plain_fp_x = x_plain;
        case_result.plain_fp_u = info_plain.u_final;
        case_result.plain_fp_time = info_plain.total_time;
        case_result.plain_preconditioner_setup_time = get_optional_field(info_plain, 'preconditioner_setup_time');
        case_result.plain_completion_solve_time = get_optional_field(info_plain, 'solve_time');
        case_result.plain_preconditioner_factor_bytes = get_optional_field(info_plain, 'preconditioner_factor_bytes');
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
        case_result.plain_preconditioner_setup_time = NaN;
        case_result.plain_completion_solve_time = NaN;
        case_result.plain_preconditioner_factor_bytes = NaN;
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
    if ~isempty(cfg.online_helm_accept_tol) && ...
            info_precorr.relres_final <= cfg.online_helm_accept_tol
        % No WaveHoltz evaluation is needed after frequency-domain acceptance.
        % NaN states/residuals explicitly mark quantities that were not computed.
        x_precorr = nan(prob.nfull, 1);
        x_online = x_precorr;
        lift_time = 0;
        online_u_final = u_precorr;
        info_online = struct('total_time',0,'iter_count',0,'relres_final',NaN, ...
            'op_apply_count',0,'relres_hist',[],'step_hist',[], ...
            'stop_reason','helmholtz_residual','u_final',u_precorr);
        online_relres_hist = info_precorr.relres_hist;
    elseif strcmp(cfg.online_completion_mode, 'helmholtz')
        x_precorr = zeros(prob.nfull, 1);
        lift_time = 0;
        [u_online, info_online] = run_plain_helmholtz_gmres( ...
            prob, rhs, c_right, u_precorr, ...
            cfg.helm_completion_restart, cfg.helm_completion_maxit, cfg.helm_completion_tol, false);
        x_online = zeros(prob.nfull, 1);
        online_u_final = u_online;
        online_relres_hist = [info_precorr.relres_hist(:); info_online.relres_hist(2:end)];
    elseif strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        x_precorr = zeros(prob.nfull, 1);
        lift_time = 0;
        [u_online, info_online] = run_shifted_laplacian_completion(prob, cfg, rhs, c_right, u_precorr);
        x_online = zeros(prob.nfull, 1);
        online_u_final = u_online;
        online_relres_hist = [info_precorr.relres_hist(:); info_online.relres_hist(2:end)];
    else
        wave_setup_time = 0;
        if isempty(Afun)
            tsetup = tic;
            [b_vec, Afun] = build_waveholtz_operator(prob, cfg, rhs, c_right);
            wave_setup_time = toc(tsetup);
        end
        tlift = tic;
        x_precorr = lift_frequency_solution_to_real_state(prob, u_precorr, c_right);
        lift_time = toc(tlift);
        if ~isempty(cfg.online_true_helm_tol)
            [x_online, info_online] = run_true_verified_wave_completion( ...
                prob,cfg,rhs,c_right,b_vec,Afun,x_precorr);
        elseif strcmp(cfg.online_wave_solver, 'gmres')
            [x_online, info_online] = run_online_waveholtz_gmres(prob, cfg, rhs, c_right, x_precorr);
        else
            [x_online, info_online] = run_online_fixedpoint(prob, cfg, rhs, c_right, b_vec, Afun, x_precorr);
        end
        info_online.total_time = info_online.total_time + wave_setup_time;
        online_u_final = info_online.u_final;
        online_relres_hist = stitch_online_relres_hist(info_precorr.relres_hist, info_online.relres_hist);
    end

    case_result.online_true_helm_relres_hist = [];
    if isfield(info_online,'true_helm_relres_hist')
        case_result.online_true_helm_relres_hist = info_online.true_helm_relres_hist;
    end
    case_result.plain_true_helm_relres_hist = [];
    if cfg.run_wave_baselines && isfield(info_plain,'true_helm_relres_hist')
        case_result.plain_true_helm_relres_hist = info_plain.true_helm_relres_hist;
    end
    case_result.online_precorr_relres_hist = info_precorr.relres_hist;
    case_result.online_precorr_u = u_precorr;
    case_result.online_precorr_x = x_precorr;
    case_result.online_precorr_time = info_precorr.total_time;
    case_result.online_lift_time = lift_time;
    case_result.online_precorr_matvec = info_precorr.matvec_count;
    case_result.online_precorr_relres_final = info_precorr.relres_final;
    case_result.online_final_x = x_online;
    case_result.online_final_u = online_u_final;
    case_result.online_fp_time = info_online.total_time;
    case_result.online_preconditioner_setup_time = get_optional_field(info_online, 'preconditioner_setup_time');
    case_result.online_completion_solve_time = get_optional_field(info_online, 'solve_time');
    case_result.online_preconditioner_factor_bytes = get_optional_field(info_online, 'preconditioner_factor_bytes');
    case_result.online_total_time = info_precorr.total_time + lift_time + info_online.total_time;
    case_result.online_fp_iter = info_online.iter_count;
    case_result.online_fp_relres_final = info_online.relres_final;
    case_result.online_fp_op_apply = direct_apply_count + info_online.op_apply_count;
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

    if strcmp(cfg.online_completion_mode, 'shifted_laplacian') && ~cfg.run_wave_baselines
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref=%.3fs (%3d it), ' ...
            'helm-FGMRES=%.3fs + shifted-Lap=%.3fs (%3d it, %.2e), helm100=%.3fs (%3d it, %.2e), ' ...
            'final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.online_precorr_time, case_result.online_fp_time, case_result.online_fp_iter, case_result.online_fp_relres_final, ...
            case_result.helm_baseline_time, case_result.helm_baseline_iter, case_result.helm_baseline_relres_final, ...
            final_full, final_phys);
    elseif strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref=%.3fs (%3d it), ' ...
            'plain-shifted=%.3fs (%3d it, %.2e, %3d apply), ' ...
            'helm-FGMRES=%.3fs + shifted-Lap=%.3fs (%3d it, %.2e), helm100=%.3fs (%3d it, %.2e), ' ...
            'final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.plain_fp_time, case_result.plain_fp_iter, case_result.plain_fp_relres_final, case_result.plain_fp_op_apply, ...
            case_result.online_precorr_time, case_result.online_fp_time, case_result.online_fp_iter, case_result.online_fp_relres_final, ...
            case_result.helm_baseline_time, case_result.helm_baseline_iter, case_result.helm_baseline_relres_final, ...
            final_full, final_phys);
    elseif strcmp(cfg.online_completion_mode, 'helmholtz') && ~cfg.run_wave_baselines
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref=%.3fs (%3d it), ' ...
            'helm-FGMRES=%.3fs + plain-helm=%.3fs (%3d it, %.2e), helm100=%.3fs (%3d it, %.2e), ' ...
            'final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.online_precorr_time, case_result.online_fp_time, case_result.online_fp_iter, case_result.online_fp_relres_final, ...
            case_result.helm_baseline_time, case_result.helm_baseline_iter, case_result.helm_baseline_relres_final, ...
            final_full, final_phys);
    elseif strcmp(cfg.online_completion_mode, 'helmholtz')
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref=%.3fs (%3d it), ' ...
            'plain-fp=%.3fs (%3d it, %.2e, %3d apply), direct-fp=%.3fs total (%3d it, %.2e, %3d apply), ' ...
            'helm-FGMRES=%.3fs + plain-helm=%.3fs (%3d it, %.2e), helm100=%.3fs (%3d it, %.2e), ' ...
            'final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.plain_fp_time, case_result.plain_fp_iter, case_result.plain_fp_relres_final, case_result.plain_fp_op_apply, ...
            case_result.direct_fp_total_time, case_result.direct_fp_iter, case_result.direct_fp_relres_final, case_result.direct_fp_op_apply, ...
            case_result.online_precorr_time, case_result.online_fp_time, case_result.online_fp_iter, case_result.online_fp_relres_final, ...
            case_result.helm_baseline_time, case_result.helm_baseline_iter, case_result.helm_baseline_relres_final, ...
            final_full, final_phys);
    else
        case_result.log_line = sprintf(['test ??: c_right=%.4f, ref=%.3fs (%3d it), ' ...
            'plain-fp=%.3fs (%3d it, %.2e, %3d apply), direct-fp=%.3fs total (%3d it, %.2e, %3d apply), helm-FGMRES=%.3fs + lift=%.3fs + WH=%.3fs (%3d it, %.2e, %3d apply), ' ...
            'helm100=%.3fs (%3d it, %.2e), final relerr(full)=%.3e, relerr(phys)=%.3e'], ...
            c_right, case_result.wave_time, case_result.wave_iter, ...
            case_result.plain_fp_time, case_result.plain_fp_iter, case_result.plain_fp_relres_final, case_result.plain_fp_op_apply, ...
            case_result.direct_fp_total_time, case_result.direct_fp_iter, case_result.direct_fp_relres_final, case_result.direct_fp_op_apply, ...
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

    test.direct_u(:, k) = case_result.direct_u;
    test.direct_x(:, k) = case_result.direct_x;
    test.direct_time(k) = case_result.direct_time;
    test.direct_op_apply(k) = case_result.direct_op_apply;

    test.direct_fp_x(:, k) = case_result.direct_fp_x;
    test.direct_fp_u(:, k) = case_result.direct_fp_u;
    test.direct_fp_time(k) = case_result.direct_fp_time;
    test.direct_fp_total_time(k) = case_result.direct_fp_total_time;
    test.direct_fp_iter(k) = case_result.direct_fp_iter;
    test.direct_fp_relres_final(k) = case_result.direct_fp_relres_final;
    test.direct_fp_op_apply(k) = case_result.direct_fp_op_apply;
    test.direct_fp_relres_hist{k} = case_result.direct_fp_relres_hist;
    test.direct_fp_step_hist{k} = case_result.direct_fp_step_hist;
    test.direct_fp_stop_reason{k} = case_result.direct_fp_stop_reason;

    test.plain_fp_x(:, k) = case_result.plain_fp_x;
    test.plain_fp_u(:, k) = case_result.plain_fp_u;
    test.plain_fp_time(k) = case_result.plain_fp_time;
    test.plain_preconditioner_setup_time(k) = case_result.plain_preconditioner_setup_time;
    test.plain_completion_solve_time(k) = case_result.plain_completion_solve_time;
    test.plain_preconditioner_factor_bytes(k) = case_result.plain_preconditioner_factor_bytes;
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
    test.online_preconditioner_setup_time(k) = case_result.online_preconditioner_setup_time;
    test.online_completion_solve_time(k) = case_result.online_completion_solve_time;
    test.online_preconditioner_factor_bytes(k) = case_result.online_preconditioner_factor_bytes;
    test.online_total_time(k) = case_result.online_total_time;
    test.online_fp_iter(k) = case_result.online_fp_iter;
    test.online_fp_relres_final(k) = case_result.online_fp_relres_final;
    test.online_fp_op_apply(k) = case_result.online_fp_op_apply;
    test.online_fp_relres_hist{k} = case_result.online_fp_relres_hist;
    test.online_fp_step_hist{k} = case_result.online_fp_step_hist;
    test.online_fp_stop_reason{k} = case_result.online_fp_stop_reason;
    test.online_true_helm_relres_hist{k} = case_result.online_true_helm_relres_hist;
    test.plain_true_helm_relres_hist{k} = case_result.plain_true_helm_relres_hist;
    test.online_precorr_relres_hist{k} = case_result.online_precorr_relres_hist;
    test.online_combined_relres_hist{k} = case_result.online_combined_relres_hist;
    test.online_combined_transition_idx(k) = case_result.online_combined_transition_idx;

    test.helm_baseline_time(k) = case_result.helm_baseline_time;
    test.helm_baseline_iter(k) = case_result.helm_baseline_iter;
    test.helm_baseline_op_apply(k) = case_result.helm_baseline_op_apply;
    test.helm_baseline_relres_final(k) = case_result.helm_baseline_relres_final;
    test.helm_baseline_relres_hist{k} = case_result.helm_baseline_relres_hist;

    test.direct_err_wave_full(k) = case_result.direct_err_wave_full;
    test.direct_err_wave_phys(k) = case_result.direct_err_wave_phys;
    test.direct_err_helm_full(k) = case_result.direct_err_helm_full;
    test.direct_err_helm_phys(k) = case_result.direct_err_helm_phys;
    test.direct_fp_err_wave_full(k) = case_result.direct_fp_err_wave_full;
    test.direct_fp_err_wave_phys(k) = case_result.direct_fp_err_wave_phys;
    test.direct_fp_err_helm_full(k) = case_result.direct_fp_err_helm_full;
    test.direct_fp_err_helm_phys(k) = case_result.direct_fp_err_helm_phys;
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

    if isfield(corr_rom,'fixed_pod_initial') && corr_rom.fixed_pod_initial
        Ahelm = build_helmholtz_operator(prob,c_right);
        r0 = rhs-Ahelm*u0;
        u_curr = u0 + apply_layer_inverse_to_rhs(prob,corr_rom.layers{1},c_right,r0);
        rho = norm(rhs-Ahelm*u_curr)/max(norm(rhs),eps);
        elapsed = toc(tprec);
        info = struct('total_time',elapsed,'layer_u',{{u_curr}}, ...
            'layer_time',elapsed,'layer_relres',rho, ...
            'relres_hist',[norm(r0)/max(norm(rhs),eps);rho], ...
            'iter_count',1,'relres_final',rho,'matvec_count',2);
        return;
    end
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
        info.relres_final = fgmres_info.relres_final;
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

function [x_sol, info] = run_online_baseline_completion(prob, cfg, rhs, c_right, b_vec, Afun, x0, u0, is_baseline)
    if strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        [u_sol, info] = run_shifted_laplacian_completion(prob, cfg, rhs, c_right, u0);
        x_sol = zeros(prob.nfull, 1);
        return;
    end

    [x_sol, info] = run_online_wave_completion(prob, cfg, rhs, c_right, b_vec, Afun, x0, is_baseline);
end

function [u_sol, info] = run_shifted_laplacian_completion(prob, cfg, rhs, c_right, u0)
    Ahelm = build_helmholtz_operator(prob, c_right);
    Afun = @(x) Ahelm * x;
    [Msolve, setup_info] = build_shifted_laplacian_solver(prob, cfg, c_right);

    opts = struct();
    opts.restart = cfg.shifted_laplacian_restart;
    opts.maxit = cfg.shifted_laplacian_maxit;
    opts.tol = cfg.shifted_laplacian_tol;
    opts.verbose = cfg.shifted_laplacian_verbose;
    opts.reorthogonalize = cfg.shifted_laplacian_reorth;

    [u_sol, info] = shifted_laplacian_gmres(Afun, Msolve, rhs, u0, opts);
    info.solve_time = info.total_time;
    info.preconditioner_setup_time = setup_info.setup_time;
    info.preconditioner_factor_bytes = setup_info.factor_bytes;
    info.preconditioner_solver = setup_info.solver;
    info.total_time = setup_info.setup_time + info.solve_time;
    info.fp_time = info.total_time;
    info.step_hist = zeros(0, 1);
    info.u_final = u_sol;
    info.shifted_laplacian_beta = cfg.shifted_laplacian_beta;
end

function [Msolve, info] = build_shifted_laplacian_solver(prob, cfg, c_right)
    alpha = (1 / c_right^2) - 1;
    Ahelm = build_helmholtz_operator(prob, c_right);
    mass_part = (prob.omega ^ 2) * speye(prob.Ntot) + alpha * prob.A_right;
    Ashift = Ahelm + 1i * cfg.shifted_laplacian_beta * mass_part;
    opts = struct('solver', cfg.shifted_laplacian_solver, ...
        'ilu_type', cfg.shifted_laplacian_ilu_type, ...
        'ilu_droptol', cfg.shifted_laplacian_ilu_droptol, ...
        'ilu_thresh', cfg.shifted_laplacian_ilu_thresh, ...
        'ilu_milu', cfg.shifted_laplacian_ilu_milu);
    [Msolve, info] = build_shifted_laplacian_preconditioner(Ashift, opts);
end

function [x_sol, info] = run_online_wave_completion(prob, cfg, rhs, c_right, b_vec, Afun, x0, is_baseline)
    if nargin < 8
        is_baseline = false;
    end
    run_cfg = cfg;
    if is_baseline && ~isempty(cfg.wave_baseline_max_iter)
        max_iter = round(cfg.wave_baseline_max_iter);
        if strcmp(cfg.online_wave_solver, 'gmres')
            run_cfg.wh_gmres_restart = max_iter;
            run_cfg.wh_gmres_maxit = 1;
        else
            run_cfg.fp_maxit = max_iter;
        end
    end
    if strcmp(cfg.online_wave_solver, 'gmres')
        [x_sol, info] = run_online_waveholtz_gmres(prob, run_cfg, rhs, c_right, x0);
    else
        [x_sol, info] = run_online_fixedpoint(prob, run_cfg, rhs, c_right, b_vec, Afun, x0);
    end
end

function [x,info] = run_true_verified_wave_completion(prob,cfg,rhs,mu,bwh,Afun,x)
    % Check the recovered Helmholtz field before advancing and after each
    % native solve. Native tolerance is tightened only when that check fails.
    timer=tic; A=build_helmholtz_operator(prob,mu); nb=max(norm(rhs),eps);
    u=recover_u_phasor_timeproj(prob,x,rhs,mu);
    info=struct('iter_count',0,'op_apply_count',0,'relres_final',NaN, ...
        'relres_hist',[],'step_hist',[],'u_final',u,'total_time',0, ...
        'stop_reason','recovered_helmholtz_residual','true_helm_relres_hist',[]);
    for attempt=0:8
        rho=norm(rhs-A*u)/nb;
        if rho<=cfg.online_true_helm_tol
            info.u_final=u; info.total_time=toc(timer); info.true_helm_relres=rho;
            return;
        end
        if attempt==8, error('helm:OnlineTolerance','True Helmholtz residual %.3e exceeds %.3e.',rho,cfg.online_true_helm_tol); end
        if strcmp(cfg.online_wave_solver,'gmres')
            [x,part]=run_online_waveholtz_gmres(prob,cfg,rhs,mu,x);
        else
            [x,part]=run_online_fixedpoint(prob,cfg,rhs,mu,bwh,Afun,x);
        end
        if isempty(info.relres_hist), info.relres_hist=part.relres_hist;
        else, info.relres_hist=[info.relres_hist(:);part.relres_hist(2:end)]; end
        if isfield(part,'true_helm_relres_hist')
            if isempty(info.true_helm_relres_hist)
                info.true_helm_relres_hist=part.true_helm_relres_hist;
            else
                info.true_helm_relres_hist=[info.true_helm_relres_hist(:);part.true_helm_relres_hist(2:end)];
            end
        end
        info.step_hist=[info.step_hist(:);part.step_hist(:)];
        info.iter_count=info.iter_count+part.iter_count;
        info.op_apply_count=info.op_apply_count+part.op_apply_count;
        info.relres_final=part.relres_final; u=part.u_final;
        cfg.wh_tol=cfg.wh_tol/2; cfg.fp_tol=cfg.fp_tol/2;
    end
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
    true_hist = [];
    if cfg.record_true_helm_history
        Ahelm = build_helmholtz_operator(prob,c_right);
        nbhelm = max(norm(rhs),1e-16);
        % Affine recovery R_f(x)=R_0(x)+R_f(0). The homogeneous period
        % preserves the original fixed-point operator and stopping sequence.
        u_forcing = recover_u_phasor_timeproj(prob,zeros(prob.nfull,1),rhs,c_right);
        true_hist = nan(cfg.fp_maxit+1,1);
    end
    for it = 1:cfg.fp_maxit
        if cfg.record_true_helm_history
            [~,Pi0,u_hom] = solve_wave_one_period_piecewise( ...
                prob,x_curr,zeros(size(rhs)),c_right,cfg.filter_shift,false);
            Ax_curr = x_curr-Pi0;
            true_hist(it) = norm(rhs-Ahelm*(u_hom+u_forcing))/nbhelm;
        else
            Ax_curr = Afun(x_curr);
        end
        op_apply_count = op_apply_count + 1;
        if any(~isfinite(Ax_curr))
            stop_reason = 'nonfinite-operator';
            relres_hist(it) = NaN;
            break;
        end
        r = b_vec - Ax_curr;
        relres = norm(r) / normb;
        relres_hist(it) = relres;
        if ~isfinite(relres)
            stop_reason = 'nonfinite-residual';
            break;
        end
        if relres <= cfg.fp_tol
            stop_reason = 'residual';
            break;
        end
        iter_count = it;
        x_next = x_curr + r;
        step_hist(it) = norm(x_next - x_curr) / max(norm(x_next), 1e-16);
        if any(~isfinite(x_next)) || ~isfinite(step_hist(it))
            stop_reason = 'nonfinite-step';
            break;
        end
        x_curr = x_next;
        if ~isempty(cfg.fp_step_tol) && step_hist(it) <= cfg.fp_step_tol
            stop_reason = 'step';
            break;
        end
    end
    r_final = b_vec - Afun(x_curr);
    op_apply_count = op_apply_count + 1;
    relres_final = norm(r_final) / normb;
    if ~isfinite(relres_final)
        stop_reason = 'nonfinite-final';
    end
    relres_hist(iter_count + 1) = relres_final;
    u_final = recover_u_phasor_timeproj(prob, x_curr, rhs, c_right);
    if cfg.record_true_helm_history
        true_hist(iter_count+1) = norm(rhs-Ahelm*u_final)/nbhelm;
        true_hist = true_hist(1:iter_count+1);
    end
    total_time = toc(tfull);

    info = struct();
    info.iter_count = iter_count;
    info.op_apply_count = op_apply_count;
    info.relres_final = relres_final;
    info.relres_hist = relres_hist(1:iter_count+1);
    info.true_helm_relres_hist = true_hist;
    info.step_hist = step_hist(1:iter_count);
    info.u_final = u_final;
    info.total_time = total_time;
    info.fp_time = total_time;
    info.stop_reason = stop_reason;
end

function [x_sol, info] = run_online_waveholtz_gmres(prob, cfg, rhs, c_right, x0)
    tsolve = tic;
    [x_sol, u_final, b_vec, wh_info] = solve_waveholtz_fullstate_solution(prob, cfg, rhs, c_right, x0);
    total_time = toc(tsolve);

    nb = max(norm(b_vec), 1e-16);
    info = struct();
    info.iter_count = wh_info.iter_count;
    info.op_apply_count = wh_info.iter_count + 1;
    info.relres_final = wh_info.relres;
    info.relres_hist = wh_info.resvec(:) / nb;
    info.step_hist = [];
    info.u_final = u_final;
    info.total_time = total_time;
    info.fp_time = total_time;
    if wh_info.flag == 0
        info.stop_reason = 'gmres-converged';
    else
        info.stop_reason = 'gmres-maxit';
    end
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
    train_state.diagnostics = cell(ns,1);

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
        train_state.diagnostics{param_idx,layer_idx} = struct('breakdown',true);
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

    [qnext,eta_curr,h,hn,diagnostic] = rom_verified_snapshot_step( ...
        Ahelm,z_rom,Q_hist(:,1:prev_step),eta_hist(:,1:prev_step));
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

function z = apply_layer_inverse_to_rhs(prob, layer, c_right, rhs_vec)
    basis_space = get_layer_basis_space(layer);
    if rom_reduced_space_utils('is_empty', basis_space)
        z = zeros(size(rhs_vec));
        return;
    end
    alpha = (1 / c_right^2) - 1;
    Ar = layer.Ar_base + alpha * layer.Ar_right;
    br = rom_reduced_space_utils('project', basis_space, rhs_vec);
    coeff = Ar \ br;
    z = rom_reduced_space_utils('lift', basis_space, coeff);
end

function basis_space = get_layer_basis_space(layer)
    if isfield(layer, 'basis_space') && ~isempty(layer.basis_space)
        basis_space = layer.basis_space;
        return;
    end
    basis_space = rom_reduced_space_utils('make_dense_space', layer.basis_u);
end

function tf = is_block3_space(space)
    tf = isstruct(space) && isfield(space, 'type') && strcmp(space.type, 'block3');
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

    if beta0 / nb <= cfg.helm_gmres_track_tol
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
        [Q(:,j+1),H(1:j,j),H(j+1,j),breakdown] = rom_arnoldi_step(w,Q(:,1:j));

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
end

function Ar = assemble_reduced_waveholtz_operator(Afun, basis_or_space)
    if isnumeric(basis_or_space)
        Ar = rom_reduced_space_utils('assemble_reduced_matrix', Afun, ...
            rom_reduced_space_utils('make_dense_space', basis_or_space));
    else
        Ar = rom_reduced_space_utils('assemble_reduced_matrix', Afun, basis_or_space);
    end
end

function [rel_full, rel_phys] = compute_solution_errors(u_approx, u_ref, prob)
    rel_full = norm(u_approx - u_ref) / max(norm(u_ref), 1e-16);
    rel_phys = cropped_relerr(u_approx, u_ref, prob);
end

function val = choose_primary_error_entry(test, cfg, ~, k, part)
    if strcmp(cfg.rom_check_target, 'helm') && ~isempty(test.helm_u_ref)
        if strcmp(part, 'phys')
            val = test.online_final_err_helm_phys(k);
        else
            val = test.online_final_err_helm_full(k);
        end
    else
        if strcmp(part, 'phys')
            val = test.online_final_err_wave_phys(k);
        else
            val = test.online_final_err_wave_full(k);
        end
    end
end

function summary = summarize_correction_results(train_ref, direct_rom, corr_rom, test, prob, cfg)
    summary = struct();
    summary.online_completion_mode = cfg.online_completion_mode;
    if strcmp(cfg.online_completion_mode, 'shifted_laplacian')
        summary.shifted_laplacian_beta = cfg.shifted_laplacian_beta;
    end
    summary.train_total_time = sum(train_ref.solve_time + train_ref.recovery_time);
    summary.train_mean_time = mean(train_ref.solve_time + train_ref.recovery_time);
    summary.train_mean_iter = mean(train_ref.iter);
    summary.direct_rank = direct_rom.pod.rank_used;
    summary.corr_pod_ranks = cellfun(@(layer) layer.pod.rank_used, corr_rom.layers);
    summary.corr_ranks = cellfun(@layer_effective_space_dim, corr_rom.layers);
    summary.corr_block_ranks = nan(corr_rom.num_layers, 0);
    summary.corr_block_dofs = nan(corr_rom.num_layers, 0);
    summary.corr_block_labels = {};
    summary.corr_storage_dense = nan(corr_rom.num_layers, 1);
    summary.corr_storage_block = nan(corr_rom.num_layers, 1);
    summary.corr_storage_saving_ratio = nan(corr_rom.num_layers, 1);
    summary.corr_storage_saved_fraction = nan(corr_rom.num_layers, 1);
    for j = 1:corr_rom.num_layers
        storage = block_storage_stats_from_layer(corr_rom.layers{j});
        if ~isempty(storage)
            nb = numel(storage.ranks);
            if size(summary.corr_block_ranks, 2) < nb
                old_cols = size(summary.corr_block_ranks, 2);
                summary.corr_block_ranks(:, old_cols + 1:nb) = NaN;
                summary.corr_block_dofs(:, old_cols + 1:nb) = NaN;
            end
            summary.corr_block_ranks(j, 1:nb) = storage.ranks;
            summary.corr_block_dofs(j, 1:nb) = storage.dofs;
            summary.corr_block_labels = storage.labels;
            summary.corr_storage_dense(j) = storage.dense_storage;
            summary.corr_storage_block(j) = storage.block_storage;
            summary.corr_storage_saving_ratio(j) = storage.saving_ratio;
            summary.corr_storage_saved_fraction(j) = storage.saved_fraction;
        end
    end
    summary.corr_num_layers = corr_rom.num_layers;
    [summary.delta_c_stats, summary.adjacent_full_stats, summary.endpoint_full, ...
        summary.adjacent_phys_stats, summary.endpoint_phys] = ...
        summarize_training_parameter_variation_from_u(train_ref.params, train_ref.u_ref_snapshots, prob);
    [summary.train_lift_vs_ref_state, summary.train_lift_vs_ref_u_component] = ...
        summarize_training_lift_error(prob, train_ref);

    summary.direct_wave_full = stats3(test.direct_err_wave_full);
    summary.direct_wave_phys = stats3(test.direct_err_wave_phys);
    summary.direct_fp_wave_full = stats3(test.direct_fp_err_wave_full);
    summary.direct_fp_wave_phys = stats3(test.direct_fp_err_wave_phys);
    summary.plain_fp_wave_full = stats3(test.plain_fp_err_wave_full);
    summary.plain_fp_wave_phys = stats3(test.plain_fp_err_wave_phys);
    summary.final_wave_full = stats3(test.online_final_err_wave_full);
    summary.final_wave_phys = stats3(test.online_final_err_wave_phys);
    if ~isempty(test.helm_u_ref)
        summary.direct_helm_full = stats3(test.direct_err_helm_full);
        summary.direct_helm_phys = stats3(test.direct_err_helm_phys);
        summary.direct_fp_helm_full = stats3(test.direct_fp_err_helm_full);
        summary.direct_fp_helm_phys = stats3(test.direct_fp_err_helm_phys);
        summary.plain_fp_helm_full = stats3(test.plain_fp_err_helm_full);
        summary.plain_fp_helm_phys = stats3(test.plain_fp_err_helm_phys);
        summary.final_helm_full = stats3(test.online_final_err_helm_full);
        summary.final_helm_phys = stats3(test.online_final_err_helm_phys);
        summary.wave_vs_helm_full = stats3(vec_solution_compare(test.wave_u_ref, test.helm_u_ref, prob, 'full'));
        summary.wave_vs_helm_phys = stats3(vec_solution_compare(test.wave_u_ref, test.helm_u_ref, prob, 'phys'));
    else
        summary.direct_helm_full = [];
        summary.direct_helm_phys = [];
        summary.direct_fp_helm_full = [];
        summary.direct_fp_helm_phys = [];
        summary.plain_fp_helm_full = [];
        summary.plain_fp_helm_phys = [];
        summary.final_helm_full = [];
        summary.final_helm_phys = [];
        summary.wave_vs_helm_full = [];
        summary.wave_vs_helm_phys = [];
    end
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
    summary.train_gmres_relres_by_layer = zeros(corr_rom.num_layers, 1);
    for j = 1:corr_rom.num_layers
        summary.train_gmres_relres_by_layer(j) = mean(corr_rom.layers{j}.train_fgmres_relres);
    end
    if isempty(summary.plain_fp_time)
        summary.time_speedup_online_vs_plain_mean = NaN;
    else
        summary.time_speedup_online_vs_plain_mean = summary.plain_fp_time(2) / max(summary.online_total_time(2), 1e-16);
    end
    if isempty(summary.direct_fp_total_time)
        summary.time_speedup_online_vs_direct_wave_mean = NaN;
    else
        summary.time_speedup_online_vs_direct_wave_mean = summary.direct_fp_total_time(2) / max(summary.online_total_time(2), 1e-16);
    end
    summary.time_speedup_online_vs_helm100_mean = summary.helm_baseline_time(2) / max(summary.online_total_time(2), 1e-16);
    if strcmp(cfg.online_completion_mode, 'helmholtz')
        summary.iter_reduction_online_vs_plain_mean = summary.helm_baseline_iter(2) - summary.online_fp_iter(2);
        summary.iter_ratio_online_vs_plain_mean = summary.online_fp_iter(2) / max(summary.helm_baseline_iter(2), 1e-16);
        summary.op_apply_ratio_online_vs_plain_mean = summary.online_fp_op_apply(2) / max(summary.helm_baseline_op_apply(2), 1e-16);
    elseif strcmp(cfg.online_completion_mode, 'shifted_laplacian') && ~isempty(summary.plain_fp_iter)
        summary.iter_reduction_online_vs_plain_mean = summary.plain_fp_iter(2) - summary.online_fp_iter(2);
        summary.iter_ratio_online_vs_plain_mean = summary.online_fp_iter(2) / max(summary.plain_fp_iter(2), 1e-16);
        summary.op_apply_ratio_online_vs_plain_mean = summary.online_fp_op_apply(2) / max(summary.plain_fp_op_apply(2), 1e-16);
    elseif isempty(summary.plain_fp_iter)
        summary.iter_reduction_online_vs_plain_mean = NaN;
        summary.iter_ratio_online_vs_plain_mean = NaN;
        summary.op_apply_ratio_online_vs_plain_mean = NaN;
    else
        summary.iter_reduction_online_vs_plain_mean = summary.plain_fp_iter(2) - summary.online_fp_iter(2);
        summary.iter_ratio_online_vs_plain_mean = summary.online_fp_iter(2) / max(summary.plain_fp_iter(2), 1e-16);
        summary.op_apply_ratio_online_vs_plain_mean = summary.online_fp_op_apply(2) / max(summary.plain_fp_op_apply(2), 1e-16);
    end

    summary.primary_target = upper(cfg.rom_check_target);
    summary.primary_error_part = 'FULL';
    if strcmp(cfg.rom_check_target, 'helm') && ~isempty(test.helm_u_ref)
        final_primary_curve = test.online_final_err_helm_full;
    else
        final_primary_curve = test.online_final_err_wave_full;
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
    summary.representative_c = test.params(summary.representative_test_index);
end

function k = layer_effective_space_dim(layer)
    if isfield(layer, 'space_dim') && ~isempty(layer.space_dim)
        k = layer.space_dim;
    else
        k = layer.pod.rank_used;
    end
end

function stats = block_storage_stats_from_layer(layer)
    stats = [];
    if ~isfield(layer, 'basis_space') || isempty(layer.basis_space) || ...
            ~isstruct(layer.basis_space) || ~isfield(layer.basis_space, 'type')
        return;
    end

    if strcmp(layer.basis_space.type, 'block3') && isfield(layer.basis_space, 'block_meta')
        bm = layer.basis_space.block_meta;
        labels = {'near', 'far', 'pml'};
        ranks = [bm.near.rank, bm.far.rank, bm.pml.rank];
        dofs = [bm.near.nrow, bm.far.nrow, bm.pml.nrow];
        local_tails = [get_optional_field(bm.near, 'local_relative_tail'), ...
            get_optional_field(bm.far, 'local_relative_tail'), ...
            get_optional_field(bm.pml, 'local_relative_tail')];
        global_tail = get_optional_field(bm.near, 'global_relative_tail');
        global_tail_tol = get_optional_field(bm.near, 'rank_global_tail_tol');
    elseif strcmp(layer.basis_space.type, 'blockn') && isfield(layer.basis_space, 'blocks')
        blocks = layer.basis_space.blocks;
        labels = cell(1, numel(blocks));
        ranks = zeros(1, numel(blocks));
        dofs = zeros(1, numel(blocks));
        local_tails = nan(1, numel(blocks));
        global_tail = NaN;
        global_tail_tol = NaN;
        for b = 1:numel(blocks)
            labels{b} = blocks(b).label;
            ranks(b) = blocks(b).k;
            dofs(b) = numel(blocks(b).idx);
            if isfield(blocks(b), 'meta') && ~isempty(blocks(b).meta)
                local_tails(b) = get_optional_field(blocks(b).meta, 'local_relative_tail');
                if b == 1
                    global_tail = get_optional_field(blocks(b).meta, 'global_relative_tail');
                    global_tail_tol = get_optional_field(blocks(b).meta, 'rank_global_tail_tol');
                end
            end
        end
    else
        return;
    end

    dense_rank = layer.pod.rank_used;
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
    stats.labels = labels;
    stats.ranks = ranks;
    stats.dofs = dofs;
    stats.dense_storage = dense_storage;
    stats.block_storage = block_storage;
    stats.saving_ratio = saving_ratio;
    stats.saved_fraction = saved_fraction;
    stats.local_tails = local_tails;
    stats.global_tail = global_tail;
    stats.global_tail_tol = global_tail_tol;
end

function print_correction_summary(summary)
    if strcmp(summary.online_completion_mode, 'helmholtz')
        completion_label = 'Helmholtz completion';
        compare_label = 'helm100';
    elseif strcmp(summary.online_completion_mode, 'shifted_laplacian')
        completion_label = 'shifted-Laplacian completion';
        compare_label = 'plain shifted-Laplacian';
    else
        completion_label = 'WaveHoltz completion';
        compare_label = 'plain';
    end
    fprintf('\n=== Summary ===\n');
    fprintf('train total / mean time   : %.3fs / %.3fs\n', summary.train_total_time, summary.train_mean_time);
    fprintf('mean train iters          : %.2f\n', summary.train_mean_iter);
    fprintf('direct ROM rank           : %d\n', summary.direct_rank);
    fprintf('layered Helmholtz ROM steps: %d\n', summary.corr_num_layers);
    if ~isempty(summary.corr_ranks)
        fprintf('layered ROM ranks         : %s\n', mat2str(summary.corr_ranks));
    end
    if ~isempty(summary.corr_block_ranks) && any(isfinite(summary.corr_block_ranks(:)))
        block_label = 'blocks';
        if isfield(summary, 'corr_block_labels') && ~isempty(summary.corr_block_labels)
            block_label = strjoin(summary.corr_block_labels, ' ');
        end
        for j = 1:size(summary.corr_block_ranks, 1)
            if all(isfinite(summary.corr_block_ranks(j, :)))
                fprintf(['  layer %d block ranks [%s] = %s, dofs = %s, ' ...
                    'storage saving = %.3fx (%.1f%% less)\n'], ...
                    j, block_label, mat2str(summary.corr_block_ranks(j, :)), ...
                    mat2str(summary.corr_block_dofs(j, :)), ...
                    summary.corr_storage_saving_ratio(j), ...
                    100 * summary.corr_storage_saved_fraction(j));
            end
        end
    end
    if ~isempty(summary.train_gmres_relres_by_layer)
        fprintf('train layered relres mean : %s\n', mat2str(summary.train_gmres_relres_by_layer, 4));
    end
    if isfield(summary, 'shifted_laplacian_beta')
        fprintf('shifted-Laplacian beta    : %.3f\n', summary.shifted_laplacian_beta);
    end
    if ~isempty(summary.delta_c_stats)
        fprintf('adjacent train delta c    : %.3e / %.3e / %.3e\n', summary.delta_c_stats(1), summary.delta_c_stats(2), summary.delta_c_stats(3));
        fprintf('adjacent train full diff  : %.3e / %.3e / %.3e\n', summary.adjacent_full_stats(1), summary.adjacent_full_stats(2), summary.adjacent_full_stats(3));
        fprintf('endpoint train full diff  : %.3e\n', summary.endpoint_full);
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
    if ~isempty(summary.direct_fp_wave_full)
        fprintf('direct+FP-vs-WH full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.direct_fp_wave_full(1), summary.direct_fp_wave_full(2), summary.direct_fp_wave_full(3));
        fprintf('direct+FP-vs-WH phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
            summary.direct_fp_wave_phys(1), summary.direct_fp_wave_phys(2), summary.direct_fp_wave_phys(3));
    end
    fprintf('final layered-ROM-vs-WH full relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.final_wave_full(1), summary.final_wave_full(2), summary.final_wave_full(3));
    fprintf('final layered-ROM-vs-WH phys relerr min/mean/max = %.3e / %.3e / %.3e\n', ...
        summary.final_wave_phys(1), summary.final_wave_phys(2), summary.final_wave_phys(3));
    if ~isempty(summary.direct_helm_full)
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
        plain_completion_label = 'plain WaveHoltz';
        if strcmp(summary.online_completion_mode, 'shifted_laplacian')
            plain_completion_label = 'plain shifted-Laplacian';
        end
        fprintf('%s time min/mean/max: %.3fs / %.3fs / %.3fs\n', plain_completion_label, ...
            summary.plain_fp_time(1), summary.plain_fp_time(2), summary.plain_fp_time(3));
        fprintf('%s iter min/mean/max: %.0f / %.2f / %.0f\n', plain_completion_label, ...
            summary.plain_fp_iter(1), summary.plain_fp_iter(2), summary.plain_fp_iter(3));
        fprintf('%s apply min/mean/max: %.0f / %.2f / %.0f\n', plain_completion_label, ...
            summary.plain_fp_op_apply(1), summary.plain_fp_op_apply(2), summary.plain_fp_op_apply(3));
        fprintf('%s relres min/mean/max = %.3e / %.3e / %.3e\n', plain_completion_label, ...
            summary.plain_fp_relres(1), summary.plain_fp_relres(2), summary.plain_fp_relres(3));
    end
    if ~isempty(summary.direct_fp_total_time)
        fprintf('direct WaveHoltz total time min/mean/max: %.3fs / %.3fs / %.3fs\n', ...
            summary.direct_fp_total_time(1), summary.direct_fp_total_time(2), summary.direct_fp_total_time(3));
        fprintf('direct WaveHoltz iter min/mean/max: %.0f / %.2f / %.0f\n', ...
            summary.direct_fp_iter(1), summary.direct_fp_iter(2), summary.direct_fp_iter(3));
        fprintf('direct WaveHoltz apply min/mean/max: %.0f / %.2f / %.0f\n', ...
            summary.direct_fp_op_apply(1), summary.direct_fp_op_apply(2), summary.direct_fp_op_apply(3));
        fprintf('direct WaveHoltz relres min/mean/max = %.3e / %.3e / %.3e\n', ...
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
        speedup_label = 'WaveHoltz';
        if strcmp(summary.online_completion_mode, 'shifted_laplacian')
            speedup_label = 'plain shifted-Laplacian';
        end
        fprintf('mean time speedup %s/online = %.3fx\n', speedup_label, summary.time_speedup_online_vs_plain_mean);
    end
    if isfinite(summary.time_speedup_online_vs_direct_wave_mean)
        fprintf('mean time speedup direct-WaveHoltz/online = %.3fx\n', summary.time_speedup_online_vs_direct_wave_mean);
    end
    fprintf('mean time speedup helm100/online = %.3fx\n', summary.time_speedup_online_vs_helm100_mean);
    fprintf('mean iter change %s-online = %.2f (ratio %.3f)\n', compare_label, ...
        summary.iter_reduction_online_vs_plain_mean, summary.iter_ratio_online_vs_plain_mean);
    fprintf('mean apply ratio online/%s = %.3f\n', compare_label, summary.op_apply_ratio_online_vs_plain_mean);
    fprintf('ROM check target          : %s\n', summary.primary_target);
    fprintf('primary error field       : %s\n', summary.primary_error_part);
    fprintf('representative c_right    : %.4f\n', summary.representative_c);
end

function save_benchmark_summary_helm(cfg, summary, nr_comparison)
    out_dir = cfg.benchmark_root;
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    name = benchmark_file_stem(cfg, 'helm_correction_rom');
    rows = benchmark_rows_from_summary(summary, 'trajectory-aware');
    if ~isempty(nr_comparison) && isfield(nr_comparison, 'summary')
        rows = [rows; benchmark_rows_from_summary(nr_comparison.summary, 'equal-rank-nr-rom')]; %#ok<AGROW>
    end
    benchmark_table = struct2table(rows);
    csv_file = fullfile(out_dir, [name '_benchmark.csv']);
    mat_file = fullfile(out_dir, [name '_benchmark.mat']);
    writetable(benchmark_table, csv_file);
    save(mat_file, 'benchmark_table', 'summary', 'nr_comparison', 'cfg');
    fprintf('saved benchmark: %s\n', csv_file);
    fprintf('saved benchmark: %s\n', mat_file);
end

function rows = benchmark_rows_from_summary(summary, method_label)
    row = struct();
    row.experiment = {char(summary_label(summary))};
    row.method = {method_label};
    row.parameter = summary_parameter(summary);
    row.plain_waveholtz_time_mean = safe_stat(summary, 'plain_fp_time', 2);
    row.plain_waveholtz_iter_mean = safe_stat(summary, 'plain_fp_iter', 2);
    row.plain_waveholtz_relres_mean = safe_stat(summary, 'plain_fp_relres', 2);
    row.plain_helm_gmres_time_mean = safe_stat(summary, 'helm_baseline_time', 2);
    row.plain_helm_gmres_iter_mean = safe_stat(summary, 'helm_baseline_iter', 2);
    row.plain_helm_gmres_relres_mean = safe_stat(summary, 'helm_baseline_relres', 2);
    row.precorrection_time_mean = safe_stat(summary, 'online_precorr_time', 2);
    row.lift_time_mean = safe_stat(summary, 'online_lift_time', 2);
    row.completion_time_mean = safe_stat(summary, 'online_fp_time', 2);
    row.online_total_time_mean = safe_stat(summary, 'online_total_time', 2);
    row.completion_iter_mean = safe_stat(summary, 'online_fp_iter', 2);
    row.completion_relres_mean = safe_stat(summary, 'online_fp_relres', 2);
    row.speedup_vs_plain_waveholtz = safe_scalar(summary, 'time_speedup_online_vs_plain_mean');
    row.speedup_vs_plain_helm_gmres = safe_scalar(summary, 'time_speedup_online_vs_helm100_mean');
    row.iter_ratio_vs_baseline = safe_scalar(summary, 'iter_ratio_online_vs_plain_mean');
    row.op_apply_ratio_vs_baseline = safe_scalar(summary, 'op_apply_ratio_online_vs_plain_mean');
    row.mean_correction_rank = mean(summary.corr_ranks(:), 'omitnan');
    row.max_correction_rank = max(summary.corr_ranks(:), [], 'omitnan');
    row.final_wave_full_error_mean = safe_stat(summary, 'final_wave_full', 2);
    row.final_helm_full_error_mean = safe_stat(summary, 'final_helm_full', 2);
    rows = row;
end

function label = summary_label(summary)
    if isfield(summary, 'primary_target')
        label = lower(char(summary.primary_target));
    else
        label = 'correction_rom';
    end
end

function val = summary_parameter(summary)
    if isfield(summary, 'representative_c')
        val = summary.representative_c;
    elseif isfield(summary, 'representative_ratio')
        val = summary.representative_ratio;
    else
        val = NaN;
    end
end

function val = safe_stat(summary, field_name, idx)
    val = NaN;
    if isfield(summary, field_name)
        data = summary.(field_name);
        if numel(data) >= idx
            val = data(idx);
        end
    end
end

function val = safe_scalar(summary, field_name)
    val = NaN;
    if isfield(summary, field_name) && ~isempty(summary.(field_name))
        val = summary.(field_name);
    end
end

function stem = benchmark_file_stem(cfg, fallback)
    if isfield(cfg, 'benchmark_name') && ~isempty(cfg.benchmark_name)
        stem = sanitize_filename(char(cfg.benchmark_name));
    elseif isfield(cfg, 'figure_subdir') && ~isempty(cfg.figure_subdir)
        stem = sanitize_filename(char(cfg.figure_subdir));
    else
        stem = sanitize_filename(fallback);
    end
end

function make_correction_plots(cfg, prob, rhs, direct_rom, corr_rom, test, summary)
    fig_dir = prepare_figure_dir(cfg);
    rep_idx = summary.representative_test_index;
    tag = sanitize_filename(sprintf('c_right_%0.4f', summary.representative_c));
    file_prefix = correction_figure_prefix(cfg);
    plot_label = correction_plot_label(cfg);

    direct_sv = direct_rom.pod.singular_values;
    if cfg.use_direct_rom_init && has_finite_data(direct_sv)
        fig = figure('Name', 'Direct ROM singular values', 'Position', [100 100 620 460]);
        ax = axes(fig);
        semilogy(ax, 1:numel(direct_sv), clamp_positive_curve(direct_sv, 1e-16), ...
            'o-', 'LineWidth', 1.7, 'MarkerSize', 6, 'DisplayName', 'direct ROM');
        style_publication_axes(ax);
        set(ax, 'YScale', 'log');
        xlabel(ax, 'mode index');
        ylabel(ax, 'singular value');
        title(ax, sprintf('%s direct ROM singular values', plot_label));
        legend(ax, 'Location', 'best');
        save_publication_figure(fig, fig_dir, [file_prefix '_direct_rom_singular_values_' tag], cfg);
    end

    if corr_rom.num_layers > 0
        fig = figure('Name', 'Correction ROM singular values', 'Position', [100 100 680 480]);
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
        title(ax, sprintf('%s correction singular-value decay', plot_label));
        legend(ax, 'Location', 'best');
        save_publication_figure(fig, fig_dir, [file_prefix '_correction_rom_singular_values_' tag], cfg);
    end

    plot_full_field_error_curves_helm(cfg, test, summary, fig_dir, tag, file_prefix);
    plot_joint_residual_decay_helm(test, summary, fig_dir, tag, file_prefix, cfg);

    if strcmp(cfg.rom_check_target, 'helm') && ~isempty(test.helm_u_ref)
        target_u = test.helm_u_ref(:, rep_idx);
    else
        target_u = test.wave_u_ref(:, rep_idx);
    end
    direct_u = test.direct_u(:, rep_idx);
    corr_u = test.online_final_u(:, rep_idx);

    target_grid = embed_interior_field(target_u, prob.nx, prob.ny);
    direct_grid = embed_interior_field(direct_u, prob.nx, prob.ny);
    corr_grid = embed_interior_field(corr_u, prob.nx, prob.ny);
    err_grid = abs(corr_grid - target_grid);
    rhs_grid = embed_interior_field(rhs, prob.nx, prob.ny);

    prefix = [file_prefix '_' tag];
    plot_full_field_component_set(prob, cfg, fig_dir, prefix, 'real', @real, ...
        target_grid, direct_grid, corr_grid, err_grid, rhs_grid, cfg.use_direct_rom_init);
    plot_full_field_component_set(prob, cfg, fig_dir, prefix, 'imag', @imag, ...
        target_grid, direct_grid, corr_grid, err_grid, rhs_grid, cfg.use_direct_rom_init);
end

function plot_full_field_error_curves_helm(cfg, test, summary, fig_dir, tag, file_prefix)
    if strcmp(cfg.rom_check_target, 'helm') && ~isempty(test.helm_u_ref)
        direct_err = test.direct_err_helm_full;
        direct_fp_err = test.direct_fp_err_helm_full;
        final_err = test.online_final_err_helm_full;
    else
        direct_err = test.direct_err_wave_full;
        direct_fp_err = test.direct_fp_err_wave_full;
        final_err = test.online_final_err_wave_full;
    end

    fig = figure('Name', 'Full-field relative errors', 'Position', [120 120 700 480]);
    ax = axes(fig);
    hold(ax, 'on');
    x = test.params(:);
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
    xlabel(ax, 'test parameter c_{right}');
    ylabel(ax, 'relative full-field error');
    title(ax, sprintf('%s full-field error', correction_plot_label(cfg)));
    legend(ax, 'Location', 'best');
    save_publication_figure(fig, fig_dir, [file_prefix '_full_field_error_curves_' tag], cfg);
end

function plot_joint_residual_decay_helm(test, summary, fig_dir, tag, file_prefix, cfg)
    rep_idx = summary.representative_test_index;
    plot_floor = 1e-16;
    fig = figure('Name', 'Joint residual decay', 'Position', [140 140 760 500]);
    ax = axes(fig);
    hold(ax, 'on');

    online_combined_hist = test.online_combined_relres_hist{rep_idx};
    if has_finite_data(online_combined_hist)
        online_label = 'TA-ROM + completion';
        if strcmp(cfg.online_completion_mode, 'shifted_laplacian')
            online_label = 'TA-ROM + shifted-Laplacian GMRES';
        elseif strcmp(cfg.online_completion_mode, 'waveholtz')
            online_label = 'TA-ROM + WaveHoltz completion';
        end
        h_online = semilogy(ax, 0:(numel(online_combined_hist) - 1), sanitize_log_curve(online_combined_hist, plot_floor), ...
            '-o', 'Color', [0.85 0.05 0.05], 'LineWidth', 1.8, 'MarkerSize', 6, ...
            'DisplayName', online_label);
        apply_sparse_residual_markers(h_online, online_combined_hist, cfg);
    end
    transition_idx = test.online_combined_transition_idx(rep_idx);
    if transition_idx >= 1 && transition_idx <= numel(online_combined_hist)
        transition_label = 'completion transition';
        if strcmp(cfg.online_completion_mode, 'waveholtz')
            transition_label = 'lift to WaveHoltz state';
        elseif strcmp(cfg.online_completion_mode, 'shifted_laplacian')
            transition_label = 'shifted-Laplacian completion starts';
        end
        semilogy(ax, transition_idx - 1, sanitize_log_curve(online_combined_hist(transition_idx), plot_floor), ...
            's', 'Color', [0.85 0.1 0.1], 'MarkerFaceColor', 'w', 'MarkerSize', 9, ...
            'DisplayName', transition_label);
        yl = ylim(ax);
        line(ax, [transition_idx - 1, transition_idx - 1], yl, ...
            'Color', [0.85 0.1 0.1], 'LineStyle', ':', 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end
    if has_finite_data(test.direct_fp_relres_hist{rep_idx})
        hist_direct = test.direct_fp_relres_hist{rep_idx};
        h_direct = semilogy(ax, 0:(numel(hist_direct) - 1), sanitize_log_curve(hist_direct, plot_floor), ...
            '-o', 'Color', [0 0.35 1], 'LineWidth', 1.3, 'MarkerSize', 5, ...
            'DisplayName', 'direct ROM + completion');
        apply_sparse_residual_markers(h_direct, hist_direct, cfg);
    end
    if has_finite_data(test.plain_fp_relres_hist{rep_idx})
        hist_plain = test.plain_fp_relres_hist{rep_idx};
        plain_label = 'plain WaveHoltz';
        if strcmp(cfg.online_completion_mode, 'shifted_laplacian')
            plain_label = 'plain shifted-Laplacian GMRES';
        end
        h_plain = semilogy(ax, 0:(numel(hist_plain) - 1), sanitize_log_curve(hist_plain, plot_floor), ...
            'k-o', 'LineWidth', 1.3, 'MarkerSize', 5, 'DisplayName', plain_label);
        apply_sparse_residual_markers(h_plain, hist_plain, cfg);
    end
    if has_finite_data(test.helm_baseline_relres_hist{rep_idx})
        hist_helm = test.helm_baseline_relres_hist{rep_idx};
        h_helm = semilogy(ax, 0:(numel(hist_helm) - 1), sanitize_log_curve(hist_helm, plot_floor), ...
            '-o', 'Color', [0.55 0 0.65], 'LineWidth', 1.3, 'MarkerSize', 5, ...
            'DisplayName', 'plain Helmholtz GMRES');
        apply_sparse_residual_markers(h_helm, hist_helm, cfg);
    end
    hold(ax, 'off');
    style_publication_axes(ax);
    set(ax, 'YScale', 'log');
    ylim(ax, [plot_floor, 2]);
    xlabel(ax, 'online step index');
    ylabel(ax, 'relative residual');
    title(ax, sprintf('%s residual decay, c_{right}=%.4f', correction_plot_label(cfg), summary.representative_c));
    legend(ax, 'Location', 'best');
    save_publication_figure(fig, fig_dir, [file_prefix '_joint_combined_residual_decay_full_field_' tag], cfg);
end

function apply_sparse_residual_markers(h, hist, cfg)
    %#ok<INUSD>
    % Keep every residual-history marker visible in shifted-Laplacian figures.
end

function plot_nr_vs_trajectory_residual_decay_helm(cfg, traj_test, traj_summary, nr_test, nr_summary)
    fig_dir = prepare_figure_dir(cfg);
    file_prefix = correction_figure_prefix(cfg);
    rep_idx = traj_summary.representative_test_index;
    c_rep = traj_summary.representative_c;
    nr_idx = choose_matching_nr_test_index(nr_test, c_rep, nr_summary);
    plot_floor = 1e-16;

    fig = figure('Name', 'Trajectory-aware versus non-replayed correction ROM residual decay', ...
        'Position', [140 140 760 500]);
    ax = axes(fig);
    hold(ax, 'on');

    traj_hist = traj_test.online_combined_relres_hist{rep_idx};
    if has_finite_data(traj_hist)
        semilogy(ax, 0:(numel(traj_hist) - 1), sanitize_log_curve(traj_hist, plot_floor), ...
            '-o', 'Color', [0.85 0.05 0.05], 'LineWidth', 1.8, 'MarkerSize', 6, ...
            'DisplayName', 'TA-ROM + completion');
        traj_transition = traj_test.online_combined_transition_idx(rep_idx);
        if traj_transition >= 1 && traj_transition <= numel(traj_hist)
            semilogy(ax, traj_transition - 1, sanitize_log_curve(traj_hist(traj_transition), plot_floor), ...
                's', 'Color', [0.85 0.05 0.05], 'MarkerFaceColor', 'w', 'MarkerSize', 9, ...
                'HandleVisibility', 'off');
        end
    end

    nr_hist = nr_test.online_combined_relres_hist{nr_idx};
    if has_finite_data(nr_hist)
        semilogy(ax, 0:(numel(nr_hist) - 1), sanitize_log_curve(nr_hist, plot_floor), ...
            '--d', 'Color', [0.15 0.35 0.75], 'LineWidth', 1.7, 'MarkerSize', 6, ...
            'DisplayName', 'NR-ROM + completion');
        nr_transition = nr_test.online_combined_transition_idx(nr_idx);
        if nr_transition >= 1 && nr_transition <= numel(nr_hist)
            semilogy(ax, nr_transition - 1, sanitize_log_curve(nr_hist(nr_transition), plot_floor), ...
                '^', 'Color', [0.15 0.35 0.75], 'MarkerFaceColor', 'w', 'MarkerSize', 9, ...
                'HandleVisibility', 'off');
        end
    end

    if has_finite_data(traj_test.helm_baseline_relres_hist{rep_idx})
        hist_helm = traj_test.helm_baseline_relres_hist{rep_idx};
        semilogy(ax, 0:(numel(hist_helm) - 1), sanitize_log_curve(hist_helm, plot_floor), ...
            ':', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.5, ...
            'DisplayName', 'plain Helmholtz GMRES');
    end

    hold(ax, 'off');
    style_publication_axes(ax);
    set(ax, 'YScale', 'log');
    ylim(ax, [plot_floor, 2]);
    xlabel(ax, 'online step index');
    ylabel(ax, 'relative residual');
    title(ax, sprintf('%s residual ablation, c_{right}=%.4f', correction_plot_label(cfg), c_rep));
    legend(ax, 'Location', 'best');

    tag = sanitize_filename(sprintf('c_right_%0.4f', c_rep));
    save_publication_figure(fig, fig_dir, [file_prefix '_trajectory_vs_nr_residual_decay_' tag], cfg);
end

function idx = choose_matching_nr_test_index(nr_test, c_rep, nr_summary)
    if isfield(nr_test, 'params') && ~isempty(nr_test.params)
        [~, idx] = min(abs(nr_test.params(:) - c_rep));
        return;
    end
    idx = nr_summary.representative_test_index;
end

function plot_full_field_component_set(prob, cfg, fig_dir, prefix, component_name, component_fun, ...
        target_grid, direct_grid, corr_grid, err_grid, rhs_grid, include_direct)
    target_data = component_fun(target_grid);
    corr_data = component_fun(corr_grid);
    direct_data = component_fun(direct_grid);

    plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_reference_full_field'], ...
        target_data, sprintf('Reference %s part', component_name), component_name);
    if include_direct && has_finite_data(direct_data) && norm(direct_data(:)) > 0
        plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_direct_rom_full_field'], ...
            direct_data, sprintf('Direct ROM %s part', component_name), component_name);
    end
    plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_joint_correction_full_field'], ...
        corr_data, sprintf('TA-ROM %s part', component_name), component_name);
    plot_full_field_image(prob, cfg, fig_dir, [prefix '_' component_name '_joint_abs_diff_full_field'], ...
        err_grid, sprintf('Absolute error, %s part', component_name), 'absolute difference');
    if strcmp(component_name, 'real')
        plot_full_field_image(prob, cfg, fig_dir, [prefix '_source_full_field'], ...
            abs(rhs_grid), 'Input source magnitude', '|b|');
    end
end

function plot_full_field_image(prob, cfg, fig_dir, file_base, data_grid, ttl, cbar_label)
    fig = figure('Name', ttl, 'Position', [160 160 620 520]);
    ax = axes(fig);
    imagesc(ax, prob.x_int, prob.y_int, data_grid(2:end-1, 2:end-1).');
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
    set(ax, 'FontSize', 18, 'LineWidth', 1.4, 'Box', 'on');
    grid(ax, 'on');
end

function tf = has_finite_data(vals)
    tf = ~isempty(vals) && any(isfinite(vals(:)));
end

function name = sanitize_filename(name)
    name = lower(regexprep(name, '[^a-zA-Z0-9]+', '_'));
    name = regexprep(name, '^_+|_+$', '');
end

function prefix = correction_figure_prefix(cfg)
    subdir = lower(char(cfg.figure_subdir));
    if contains(subdir, 'photonic') || strcmp(cfg.material_model, 'photonic_crystal')
        prefix = 'pc';
    elseif contains(subdir, 'helm')
        prefix = 'helm';
    else
        prefix = sanitize_filename(subdir);
        if isempty(prefix)
            prefix = 'correction_rom';
        end
    end
end

function label = correction_plot_label(cfg)
    if any(strcmp(cfg.material_model, {'photonic_crystal', 'pc', 'double_layer', 'two_layer'}))
        label = 'Photonic waveguide';
    elseif strcmp(cfg.material_model, 'halfspace')
        label = 'Half-space';
    else
        label = 'Correction ROM';
    end
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

    % Helm pre-correction and WaveHoltz completion use different residual
    % equations and normalizations.  For plotting the online trajectory as a
    % single curve, match the WaveHoltz residual scale to the Helm residual
    % at the lift transition instead of letting it restart near one.
    helm_anchor = helm_hist(end);
    wave_anchor = wave_hist(1);
    if isfinite(helm_anchor) && helm_anchor > 0 && isfinite(wave_anchor) && wave_anchor > 0
        wave_hist = wave_hist * (helm_anchor / wave_anchor);
    end
    combined = [helm_hist; wave_hist(2:end)];
end

function y = sanitize_log_curve(curve, floor_value)
    y = curve(:).';
    bad = ~isfinite(y) | y <= 0;
    y(bad) = NaN;
    y = max(y, floor_value);
end

function completion_matched = match_completion_to_precorr(precorr_hist, completion_hist)
    completion_matched = completion_hist(:);
    if isempty(precorr_hist) || isempty(completion_matched)
        return;
    end
    helm_anchor = precorr_hist(end);
    completion_anchor = completion_matched(1);
    if isfinite(helm_anchor) && helm_anchor > 0 && isfinite(completion_anchor) && completion_anchor > 0
        completion_matched = completion_matched * (helm_anchor / completion_anchor);
    end
end

function pod = build_pod_basis_custom(snapshot_matrix, energy_target, singular_value_tol, min_rank, max_rank, truncation_rule)
    if nargin < 6 || isempty(truncation_rule)
        truncation_rule = 'relative_singular_value';
    end
    truncation_rule = normalize_pod_truncation_rule(truncation_rule);
    [U, S, V] = svd(snapshot_matrix, 'econ');
    singular_values = diag(S);
    zero_tol = 1e-15;
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
        pod.singular_value_tol_mode = truncation_rule;
        pod.rank_truncation = 0;
        pod.rank_tail_tol = [];
        pod.rank_tail = [];
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
        switch truncation_rule
            case 'relative_frobenius_tail'
                rank_sigma = local_rank_from_relative_frobenius_tail(singular_values, abs(singular_value_tol));
            case 'relative_singular_value'
                rel_singular_values = singular_values / max(abs(singular_values(1)), 1e-30);
                rank_sigma = find(rel_singular_values >= abs(singular_value_tol), 1, 'last');
                if isempty(rank_sigma)
                    rank_sigma = 1;
                end
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
    pod.rank_truncation = rank_sigma;
    pod.rank_used = rank_use;
    pod.basis = U(:, 1:rank_use);
    pod.r90 = first_mode_above(cum_energy, 0.90);
    pod.r95 = first_mode_above(cum_energy, 0.95);
    pod.r99 = first_mode_above(cum_energy, 0.99);
    pod.singular_value_tol = singular_value_tol;
    pod.singular_value_tol_mode = truncation_rule;
    pod.rank_tail_tol = [];
    pod.rank_tail = [];
    if strcmp(truncation_rule, 'relative_frobenius_tail')
        pod.rank_tail_tol = singular_value_tol;
        pod.rank_tail = rank_sigma;
    end
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
            s = [NaN, NaN, NaN];
        else
            s = [min(v(:)), mean(v(:)), max(v(:))];
        end
    end
end

function [delta_c_stats, adjacent_full_stats, span_full, adjacent_phys_stats, span_phys] = summarize_training_parameter_variation_from_u(params, u_snapshots, prob)
    if numel(params) < 2 || size(u_snapshots, 2) < 2
        delta_c_stats = [];
        adjacent_full_stats = [];
        span_full = [];
        adjacent_phys_stats = [];
        span_phys = [];
        return;
    end
    delta_c = diff(params(:));
    adjacent_full = zeros(numel(delta_c), 1);
    adjacent_phys = zeros(numel(delta_c), 1);
    for k = 1:numel(delta_c)
        adjacent_full(k) = norm(u_snapshots(:, k + 1) - u_snapshots(:, k)) / ...
            max(norm(u_snapshots(:, k)), 1e-16);
        adjacent_phys(k) = cropped_relerr(u_snapshots(:, k + 1), u_snapshots(:, k), prob);
    end
    delta_c_stats = [min(delta_c), mean(delta_c), max(delta_c)];
    adjacent_full_stats = [min(adjacent_full), mean(adjacent_full), max(adjacent_full)];
    span_full = norm(u_snapshots(:, end) - u_snapshots(:, 1)) / max(norm(u_snapshots(:, 1)), 1e-16);
    adjacent_phys_stats = [min(adjacent_phys), mean(adjacent_phys), max(adjacent_phys)];
    span_phys = cropped_relerr(u_snapshots(:, end), u_snapshots(:, 1), prob);
end

function values = expand_layer_parameter(layer_values, fallback_value, nlayers, field_name)
    if nlayers == 0
        values = zeros(0, 1);
        return;
    end

    if isempty(layer_values)
        values = fallback_value * ones(nlayers, 1);
        return;
    end

    values = layer_values(:);
    if numel(values) ~= nlayers
        error('%s must have exactly corr_num_layers=%d entries.', field_name, nlayers);
    end
end

function values = expand_optional_layer_values(values, nlayers, field_name)
    if nlayers == 0
        values = [];
        return;
    end
    if isempty(values)
        return;
    end
    values = values(:).';
    if isscalar(values)
        values = repmat(values, 1, nlayers);
    elseif numel(values) ~= nlayers
        error('%s must be scalar or have corr_num_layers=%d entries.', field_name, nlayers);
    end
end

function value = layer_value(values, layer_index)
    if isempty(values)
        value = [];
    elseif isscalar(values)
        value = values;
    else
        value = values(layer_index);
    end
end

function value = layer_vector_value(values, layer_index)
    if isempty(values)
        value = [];
    elseif isvector(values)
        value = values(:).';
    elseif size(values, 1) >= layer_index
        value = values(layer_index, :);
    elseif size(values, 2) >= layer_index
        value = values(:, layer_index).';
    else
        error('Layer-vector values do not contain layer %d.', layer_index);
    end
end

function value = get_optional_field(s, field_name)
    if isstruct(s) && isfield(s, field_name) && ~isempty(s.(field_name))
        value = s.(field_name);
    else
        value = NaN;
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

function [W, Pi, uhat] = solve_wave_one_period_piecewise(prob, x0, rhs, c_right, projection_shift, store_history)
    if nargin < 5
        projection_shift = [];
    end
    if nargin < 6
        store_history = true;
    end
    need_pi = ~isempty(projection_shift);
    need_uhat = nargout >= 3;
    sx_full = prob.pml_wave.sx_full;
    sy_full = prob.pml_wave.sy_full;

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
    theta_coeff = prob.pml_wave.sigma_xy .* (prob.c_w ^ 2) ./ c_sq_full;

    if store_history
        W = zeros(prob.nt + 1, prob.nfull);
        W(1, :) = x0.';
    else
        W = [];
    end
    if need_pi
        pi_acc = 0.5 * (1 - projection_shift) * x0;
    else
        pi_acc = [];
    end
    if need_uhat
        u0 = x0(1:prob.nu);
        q_cos_acc = 0.5 * u0;
        q_sin_acc = zeros(prob.nu, 1);
    else
        q_cos_acc = [];
        q_sin_acc = [];
    end

    tmp1 = zeros(prob.nx + 2, prob.ny + 2);
    tmp2 = zeros(prob.nx + 2, prob.ny + 2);
    div_v = zeros(prob.nx + 2, prob.ny + 2);

    for n = 1:prob.nt
        t = (n - 1) * prob.dt;
        t_half = t + 0.5 * prob.dt;

        du_dx = (u(2:end, :) - u(1:end-1, :)) / prob.hx;
        du_dy = (u(:, 2:end) - u(:, 1:end-1)) / prob.hy;
        vx_new = prob.pml_wave.vx_decay .* vx + prob.pml_wave.vx_drive .* du_dx;
        vy_new = prob.pml_wave.vy_decay .* vy + prob.pml_wave.vy_drive .* du_dy;

        vx_avg = 0.5 * (vx + vx_new);
        vy_avg = 0.5 * (vy + vy_new);

        grad_vx = (vx_avg(2:end, :) - vx_avg(1:end-1, :)) / prob.hx;
        tmp1(:) = 0;
        tmp1(2:end-1, :) = grad_vx;

        grad_vy = (vy_avg(:, 2:end) - vy_avg(:, 1:end-1)) / prob.hy;
        tmp2(:) = 0;
        tmp2(:, 2:end-1) = grad_vy;

        psi1 = psi1 + prob.dt * (sy_full .* tmp1);
        psi2 = psi2 + prob.dt * (sx_full .* tmp2);
        theta = theta + prob.dt * (theta_coeff .* u);

        div_v(:) = 0;
        div_v(2:end-1, :) = div_v(2:end-1, :) + (vx_new(2:end, :) - vx_new(1:end-1, :)) / prob.hx;
        div_v(:, 2:end-1) = div_v(:, 2:end-1) + (vy_new(:, 2:end) - vy_new(:, 1:end-1)) / prob.hy;

        forcing = -(1.0 / prob.omega) * rhs_2d .* sin(prob.omega_ts * t_half);
        rhs_u = div_v(2:end-1, 2:end-1) ...
              - theta(2:end-1, 2:end-1) ...
              + psi1(2:end-1, 2:end-1) ...
              + psi2(2:end-1, 2:end-1) ...
              + forcing;

        u(2:end-1, 2:end-1) = prob.pml_wave.u_decay .* u(2:end-1, 2:end-1) ...
            + prob.pml_wave.u_drive .* (c_sq .* rhs_u);

        vx = vx_new;
        vy = vy_new;

        u_vec = reshape(u(2:end-1, 2:end-1), prob.nu, 1);
        vx_vec = reshape(vx(:, 2:end-1), prob.nvx, 1);
        vy_vec = reshape(vy(2:end-1, :), prob.nvy, 1);
        psi1_vec = reshape(psi1(2:end-1, 2:end-1), prob.nu, 1);
        psi2_vec = reshape(psi2(2:end-1, 2:end-1), prob.nu, 1);
        theta_vec = reshape(theta(2:end-1, 2:end-1), prob.nu, 1);

        if store_history
            W(n + 1, 1:prob.nu) = u_vec.';
            W(n + 1, prob.nu + (1:prob.nvx)) = vx_vec.';
            W(n + 1, prob.nu + prob.nvx + (1:prob.nvy)) = vy_vec.';
            W(n + 1, prob.nu + prob.nv + (1:prob.nu)) = psi1_vec.';
            W(n + 1, prob.nu + prob.nv + prob.nu + (1:prob.nu)) = psi2_vec.';
            W(n + 1, prob.nu + prob.nv + 2 * prob.nu + (1:prob.nu)) = theta_vec.';
        end

        endpoint_weight = 1.0;
        if n == prob.nt
            endpoint_weight = 0.5;
        end
        t_now = n * prob.dt;
        if need_pi
            weight = endpoint_weight * (cos(prob.omega_ts * t_now) - projection_shift);
            pi_acc(1:prob.nu) = pi_acc(1:prob.nu) + weight * u_vec;
            pi_acc(prob.nu + (1:prob.nvx)) = pi_acc(prob.nu + (1:prob.nvx)) + weight * vx_vec;
            pi_acc(prob.nu + prob.nvx + (1:prob.nvy)) = pi_acc(prob.nu + prob.nvx + (1:prob.nvy)) + weight * vy_vec;
            pi_acc(prob.nu + prob.nv + (1:prob.nu)) = pi_acc(prob.nu + prob.nv + (1:prob.nu)) + weight * psi1_vec;
            pi_acc(prob.nu + prob.nv + prob.nu + (1:prob.nu)) = pi_acc(prob.nu + prob.nv + prob.nu + (1:prob.nu)) + weight * psi2_vec;
            pi_acc(prob.nu + prob.nv + 2 * prob.nu + (1:prob.nu)) = pi_acc(prob.nu + prob.nv + 2 * prob.nu + (1:prob.nu)) + weight * theta_vec;
        end
        if need_uhat
            q_cos_acc = q_cos_acc + endpoint_weight * cos(prob.omega_ts * t_now) * u_vec;
            q_sin_acc = q_sin_acc + endpoint_weight * sin(prob.omega_ts * t_now) * u_vec;
        end
    end

    projection_scale = 2.0 * prob.dt / max(prob.T_ts, 1e-16);
    Pi = projection_scale * pi_acc;
    uhat = projection_scale * (q_cos_acc + 1i * q_sin_acc);
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
    if prob.stream_wave_projection
        [~, Pi] = solve_wave_one_period_piecewise(prob, x0, rhs_use, c_right, shift, false);
    else
        W = solve_wave_one_period_piecewise(prob, x0, rhs_use, c_right);
        t_grid = (0:prob.nt).' * prob.dt;
        weights = cos(prob.omega_ts * t_grid) - shift;
        Pi = (2.0 / max(prob.T_ts, 1e-16)) * trapz(t_grid, bsxfun(@times, weights, W), 1).';
    end
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
    K = pml_stiffness_2d(nx, ny, hx, hy, pml);
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
    if nargin < 7
        verbose = true;
    end
    if nargin < 8 || isempty(reorth)
        reorth = false;
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
    resvec = zeros(1 + maxit_outer * restart, 1);
    resvec(1) = beta;
    res_count = 1;

    if verbose
        fprintf('GMRES init: ||r0||/||b|| = %.3e\n', beta / nb);
    end
    if beta / nb <= tol
        flag = 0;
        relres = beta / nb;
        iter = [0, 0];
        resvec = resvec(1:res_count);
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
            if reorth
                for i = 1:j
                    h_corr = V(:, i)' * w;
                    H(i, j) = H(i, j) + h_corr;
                    w = w - h_corr * V(:, i);
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
            res_count = res_count + 1;
            resvec(res_count) = resnorm;
            if verbose
                fprintf('  GMRES iter %4d: ||r||/||b|| = %.3e\n', (outer - 1) * restart + j, resnorm / nb);
            end

            if resnorm / nb <= tol
                y = solve_projected_gmres(H(1:j, 1:j), g(1:j));
                x_candidate = x + V(:, 1:j) * y;
                r_candidate = b - Afun(x_candidate);
                beta_candidate = norm(r_candidate);
                resvec(res_count) = beta_candidate;
                iter = [outer, j];
                if beta_candidate / nb <= tol
                    x = x_candidate;
                    r = r_candidate;
                    beta = beta_candidate;
                    flag = 0;
                    relres = beta / nb;
                    resvec = resvec(1:res_count);
                    return;
                end
                x = x_candidate;
                r = r_candidate;
                beta = beta_candidate;
                break;
            end
        end

        if beta / nb <= tol
            flag = 0;
            relres = beta / nb;
            resvec = resvec(1:res_count);
            return;
        end
        if inner_last < restart
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
            resvec = resvec(1:res_count);
            return;
        end
    end

    relres = beta / nb;
    resvec = resvec(1:res_count);
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

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
