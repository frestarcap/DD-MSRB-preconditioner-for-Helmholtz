function results = photonic_crystal_waveguide_correction_rom(cfg)
%PHOTONIC_CRYSTAL_WAVEGUIDE_CORRECTION_ROM
% Correction-ROM workflow for the photonic-crystal waveguide material parameter.
%
% This is the third-step workflow:
%   fixed geometry + source, parameterized crystal wave speed c_pc,
%   selectable offline references, then layered Helmholtz correction ROM.
%
% Example:
%   r = photonic_crystal_waveguide_correction_rom(struct( ...
%       'corr_num_layers', 3, 'do_plots', true));
%
% Offline reference switch:
%   'offline_reference_solver', 'waveholtz'  -> zero-initialized WaveHoltz-GMRES
%   'offline_reference_solver', 'helmholtz'  -> frequency Helmholtz solve lifted to WaveHoltz state
%
% Default test-point policy:
%   one random point per adjacent training interval, controlled by test_param_rng_seed.
%
% Large reference geometry:
%   r = photonic_crystal_waveguide_correction_rom(struct( ...
%       'pc_num_cols', 21, 'pc_num_rows', 7, 'pc_points_per_a', 6, ...
%       'wh_gmres_restart', 500, 'wh_gmres_maxit', 3));

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end

    cfg = apply_pc_rom_defaults(cfg);
    results = helm_waveholtz_correction_rom(cfg);
    results.problem_name = 'photonic_crystal_waveguide_correction_rom';
    results.c_pc_train_params = results.cfg.train_params;
    results.c_pc_test_params = results.cfg.test_params;
end

function cfg = apply_pc_rom_defaults(cfg)
    cfg = set_default(cfg, 'pc_lattice_a', 1.0);
    cfg = set_default(cfg, 'pc_num_cols', 9);
    cfg = set_default(cfg, 'pc_num_rows', 3);
    cfg = set_default(cfg, 'Lx', cfg.pc_num_cols * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'Ly', (2 * cfg.pc_num_rows + 1) * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'L', cfg.Ly);
    cfg = set_default(cfg, 'pc_points_per_a', 20);
    cfg = set_default(cfg, 'Nphys_x', round(cfg.Lx / cfg.pc_lattice_a * cfg.pc_points_per_a));
    cfg = set_default(cfg, 'Nphys_y', round(cfg.Ly / cfg.pc_lattice_a * cfg.pc_points_per_a));
    cfg = set_default(cfg, 'Nphys', cfg.Nphys_y);
    cfg = set_default(cfg, 'npml', 2 * cfg.pc_points_per_a);
    cfg = set_default(cfg, 'pc_radius', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'pc_rod_shape', 'circle');
    cfg = set_default(cfg, 'pc_square_halfwidth', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'pc_x_margin', 0.5 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'lambda_over_a', 2.0);
    cfg = set_default(cfg, 'omega', 2 * pi / (cfg.lambda_over_a * cfg.pc_lattice_a));
    cfg = set_default(cfg, 'c_pc_ref', 1 / sqrt(11.4));
    cfg = set_default(cfg, 'source_type', 'point');
    cfg = set_default(cfg, 'point_source_profile', 'box');
    cfg = set_default(cfg, 'point_amplitude', 1.0);
    cfg = set_default(cfg, 'point_box_radius_x', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'point_box_radius_y', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'source_col_phys', round((cfg.Nphys_x + 1) / 2));
    cfg = set_default(cfg, 'source_row_phys', round((cfg.Nphys_y + 1) / 2));
    cfg = set_default(cfg, 'material_model', 'photonic_crystal');
    cfg = set_default(cfg, 'rom_check_target', 'helm');
    cfg = apply_offline_reference_solver_default(cfg);
    cfg = set_default(cfg, 'wh_tol', 1e-6);
    cfg = set_default(cfg, 'wh_gmres_restart', 120);
    cfg = set_default(cfg, 'wh_gmres_maxit', 3);
    cfg = set_default(cfg, 'wh_gmres_reorth', true);
    cfg = set_default(cfg, 'online_completion_mode', 'waveholtz');
    cfg = set_default(cfg, 'online_wave_solver', 'gmres');
    cfg = set_default(cfg, 'run_wave_baselines', true);
    cfg = set_default(cfg, 'wave_baseline_max_iter', []);
    cfg = set_default(cfg, 'corr_basis_style', 'global');
    cfg = set_default(cfg, 'block3_partition_mode', 'pc_waveguide');
    cfg = set_default(cfg, 'block3_center', [0.5 * cfg.Lx, 0.5 * cfg.Ly]);
    cfg = set_default(cfg, 'block3_r_near', 1.5 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'block3_waveguide_halfwidth', 0.75 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'corr_block3_rank_cap_near', Inf);
    cfg = set_default(cfg, 'corr_block3_rank_cap_far', Inf);
    cfg = set_default(cfg, 'corr_block3_rank_cap_pml', Inf);
    cfg = set_default(cfg, 'corr_block3_tail_tol_near', [1, 1, 1]);
    cfg = set_default(cfg, 'corr_block3_tail_tol_far', [1, 1, 1]);
    cfg = set_default(cfg, 'corr_block3_tail_tol_pml', [1, 1, 1]);
    cfg = set_default(cfg, 'corr_block3_global_tail_tols', [1e-4, 1e-3, 1e-2]);
    cfg = set_default(cfg, 'corr_block3_global_tail_strategy', 'marginal_storage');
    cfg = set_default(cfg, 'corr_num_layers', 3);
    cfg = set_default(cfg, 'figure_subdir', 'photonic_crystal_correction_rom');
    cfg = set_default(cfg, 'benchmark_name', 'photonic_crystal_correction_rom');
    cfg = set_default(cfg, 'plot_y_limits', [1, 6]);
    cfg = set_default(cfg, 'test_param_strategy', 'random_interval');
    if ~isfield(cfg, 'test_param_rng_seed')
        cfg.test_param_rng_seed = 1;
    end

    if isfield(cfg, 'c_pc_train_params') && ~isempty(cfg.c_pc_train_params)
        cfg.train_params = cfg.c_pc_train_params;
    elseif ~isfield(cfg, 'train_params') || isempty(cfg.train_params)
        cfg.train_params = cfg.c_pc_ref * linspace(0.96, 1.04, 12);
    end

    if isfield(cfg, 'c_pc_test_params') && ~isempty(cfg.c_pc_test_params)
        cfg.test_params = cfg.c_pc_test_params;
    elseif ~isfield(cfg, 'test_params') || isempty(cfg.test_params)
        cfg.test_params = generate_pc_test_params(cfg.train_params, cfg.test_param_strategy, cfg.test_param_rng_seed);
    end

    if ~isfield(cfg, 'corr_pod_singular_value_tols') || isempty(cfg.corr_pod_singular_value_tols)
        cfg.corr_pod_singular_value_tols = [1e-4, 1e-3, 1e-2];
    end
end

function test_params = generate_pc_test_params(train_params, strategy, rng_seed)
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
            if ~isempty(rng_seed)
                old_rng = rng;
                cleanup = onCleanup(@() rng(old_rng));
                rng(rng_seed, 'twister');
            end
            test_params = lo + rand(size(lo)) .* (hi - lo);

        case {'midpoint', 'middle', 'mid'}
            test_params = 0.5 * (lo + hi);

        otherwise
            error('test_param_strategy=%s must be random_interval or midpoint.', strategy);
    end
end

function cfg = apply_offline_reference_solver_default(cfg)
    if isfield(cfg, 'offline_reference_solver') && ~isempty(cfg.offline_reference_solver)
        solver = lower(strtrim(cfg.offline_reference_solver));
    elseif isfield(cfg, 'reference_mode') && ~isempty(cfg.reference_mode)
        solver = lower(strtrim(cfg.reference_mode));
    else
        solver = 'waveholtz';
    end

    switch solver
        case {'waveholtz', 'wave', 'wh'}
            cfg.offline_reference_solver = 'waveholtz';
            cfg.reference_mode = 'waveholtz';
            cfg = set_default(cfg, 'wh_initial_guess', 'zero');

        case {'helmholtz', 'helm', 'helm_lift'}
            cfg.offline_reference_solver = 'helmholtz';
            cfg.reference_mode = 'helm_lift';
            cfg = set_default(cfg, 'wh_initial_guess', 'helm_lift');

        otherwise
            error('offline_reference_solver=%s must be waveholtz or helmholtz.', solver);
    end
end

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
