function results = run_helm_ta_shifted_scaling_experiment(mode, user_cfg)
%RUN_HELM_TA_SHIFTED_SCALING_EXPERIMENT Fixed-PPW TA-ROM scaling benchmark.
%
% Compares sparse direct solves, ILU shifted-Laplacian GMRES from zero, and
% the same shifted-Laplacian completion after three TA-ROM FGMRES steps.

    if nargin < 1 || isempty(mode)
        mode = 'paper';
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end
    mode = lower(char(string(mode)));
    cfg = merge_structs(default_cfg(mode), user_cfg);

    out_dir = helm_output_path('outputs', 'helm_ta_shifted_scaling');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    cases = repmat(empty_case(), numel(cfg.scale_factors), 1);
    for k = 1:numel(cfg.scale_factors)
        scale = cfg.scale_factors(k);
        nphys = round((cfg.base_nphys + 1) * scale) - 1;
        omega = cfg.base_omega * scale;
        train_params = linspace(cfg.parameter_range(1), cfg.parameter_range(2), cfg.train_count(k));

        solver_cfg = struct();
        solver_cfg.Nphys = nphys;
        solver_cfg.omega = omega;
        solver_cfg.train_params = train_params;
        solver_cfg.test_params = cfg.test_params;
        solver_cfg.reference_mode = 'helm_lift';
        solver_cfg.rom_check_target = 'helm';
        solver_cfg.online_completion_mode = 'shifted_laplacian';
        solver_cfg.shifted_laplacian_solver = 'ilu';
        solver_cfg.shifted_laplacian_ilu_droptol = cfg.ilu_droptol;
        solver_cfg.shifted_laplacian_beta = cfg.beta;
        solver_cfg.shifted_laplacian_restart = cfg.restart;
        solver_cfg.shifted_laplacian_maxit = cfg.maxit;
        solver_cfg.shifted_laplacian_tol = cfg.tol;
        solver_cfg.corr_num_layers = cfg.num_layers;
        solver_cfg.corr_pod_singular_value_tols = cfg.layer_tolerances;
        solver_cfg.build_direct_rom_when_unused = false;
        solver_cfg.run_wave_baselines = true;
        solver_cfg.helm_baseline_iters = 1;
        solver_cfg.helm_baseline_tol = 0;
        solver_cfg.do_plots = false;
        solver_cfg.save_plots = false;
        solver_cfg.save_benchmark = false;
        solver_cfg.parallel_train = cfg.parallel_train;
        solver_cfg.parallel_test = cfg.parallel_test;

        fprintf('\n=== TA/shifted scaling %d/%d: Nphys=%d, omega=%.3f ===\n', ...
            k, numel(cfg.scale_factors), nphys, omega);
        run_result = helm_waveholtz_correction_rom(solver_cfg);
        cases(k) = summarize_case(scale, run_result);

        if cfg.save_full_results
            case_file = fullfile(out_dir, sprintf('case_N%d.mat', run_result.problem.Ntot));
            save(case_file, 'run_result', '-v7.3');
        end
        clear run_result
    end

    results = struct();
    results.mode = mode;
    results.cfg = cfg;
    results.cases = cases;
    results.table = struct2table(cases);
    results.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));

    tag = cfg.output_tag;
    mat_file = fullfile(out_dir, ['helm_ta_shifted_scaling_' tag '.mat']);
    csv_file = fullfile(out_dir, ['helm_ta_shifted_scaling_' tag '.csv']);
    save(mat_file, 'results');
    writetable(results.table, csv_file);
    fprintf('Saved %s\nSaved %s\n', mat_file, csv_file);
    disp(results.table);
end

function cfg = default_cfg(mode)
    cfg = struct();
    cfg.base_nphys = 125;
    cfg.base_omega = 53.5;
    cfg.parameter_range = [0.8, 0.95];
    cfg.test_params = [0.827, 0.873, 0.921];
    cfg.beta = 0.2;
    cfg.ilu_droptol = 1e-3;
    cfg.restart = 30;
    cfg.maxit = 20;
    cfg.tol = 1e-8;
    cfg.num_layers = 3;
    cfg.layer_tolerances = [1e-4, 1e-3, 1e-2];
    cfg.parallel_train = false;
    cfg.parallel_test = false;
    cfg.save_full_results = false;
    cfg.output_tag = mode;

    switch mode
        case 'smoke'
            cfg.scale_factors = 0.25;
            cfg.train_count = 4;
            cfg.test_params = 0.873;
            cfg.maxit = 4;
        case 'pilot'
            cfg.scale_factors = [1, 2];
            cfg.train_count = [12, 12];
        case 'paper'
            cfg.scale_factors = [1, 2, 4];
            cfg.train_count = [12, 12, 16];
        otherwise
            error('mode must be smoke, pilot, or paper.');
    end
end

function one = summarize_case(scale, r)
    one = empty_case();
    one.scale = scale;
    one.nphys = r.cfg.Nphys;
    one.omega = r.cfg.omega;
    one.N = r.problem.Ntot;
    one.ppw_shortest = 2 * pi * min(r.cfg.test_params) ...
        / (r.cfg.omega * min(r.problem.hx, r.problem.hy));
    one.train_count = numel(r.cfg.train_params);
    one.test_count = numel(r.cfg.test_params);
    one.setup_s = r.timing.setup;
    one.offline_s = r.timing.offline;
    one.direct_mean_s = mean(r.test.wave_time);
    one.plain_shifted_mean_s = mean(r.test.plain_fp_time);
    one.plain_setup_mean_s = mean(r.test.plain_preconditioner_setup_time);
    one.plain_solve_mean_s = mean(r.test.plain_completion_solve_time);
    one.plain_iterations_mean = mean(r.test.plain_fp_iter);
    one.ta_precorr_mean_s = mean(r.test.online_precorr_time);
    one.ta_precorr_relres_mean = mean(r.test.online_precorr_relres_final);
    one.ta_shifted_mean_s = mean(r.test.online_fp_time);
    one.ta_setup_mean_s = mean(r.test.online_preconditioner_setup_time);
    one.ta_solve_mean_s = mean(r.test.online_completion_solve_time);
    one.ta_iterations_mean = mean(r.test.online_fp_iter);
    one.ta_total_mean_s = mean(r.test.online_total_time);
    one.ta_final_relres_mean = mean(r.test.online_fp_relres_final);
    one.ta_full_error_mean = mean(r.test.online_final_err_helm_full);
    one.speedup_vs_shifted = one.plain_shifted_mean_s / one.ta_total_mean_s;
    one.speedup_vs_direct = one.direct_mean_s / one.ta_total_mean_s;
    one.corr_rank_max = max(r.summary.corr_ranks);
    one.corr_storage_entries = sum(r.summary.corr_storage_block, 'omitnan');
    one.prec_factor_bytes = mean(r.test.online_preconditioner_factor_bytes);
end

function one = empty_case()
    names = {'scale', 'nphys', 'omega', 'N', 'ppw_shortest', 'train_count', ...
        'test_count', 'setup_s', 'offline_s', 'direct_mean_s', ...
        'plain_shifted_mean_s', 'plain_setup_mean_s', 'plain_solve_mean_s', ...
        'plain_iterations_mean', 'ta_precorr_mean_s', 'ta_precorr_relres_mean', ...
        'ta_shifted_mean_s', 'ta_setup_mean_s', 'ta_solve_mean_s', ...
        'ta_iterations_mean', 'ta_total_mean_s', 'ta_final_relres_mean', ...
        'ta_full_error_mean', 'speedup_vs_shifted', 'speedup_vs_direct', ...
        'corr_rank_max', 'corr_storage_entries', 'prec_factor_bytes'};
    one = cell2struct(num2cell(nan(size(names))), names, 2);
end

function out = merge_structs(base, override)
    out = base;
    names = fieldnames(override);
    for k = 1:numel(names)
        out.(names{k}) = override.(names{k});
    end
    if isscalar(out.train_count)
        out.train_count = repmat(out.train_count, size(out.scale_factors));
    end
    if numel(out.train_count) ~= numel(out.scale_factors)
        error('train_count must be scalar or match scale_factors.');
    end
end
