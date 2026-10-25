function results = run_pc_four_block_partition_experiment(mode, user_cfg)
%RUN_PC_FOUR_BLOCK_PARTITION_EXPERIMENT
% Run the photonic-crystal waveguide DD-TA-ROM with a directional PML split.
%
% Blocks, in order:
%   channel core / physical cladding / left-right PML / top-bottom-corner PML
%
% Usage:
%   results = run_pc_four_block_partition_experiment("smoke");
%   results = run_pc_four_block_partition_experiment("paper");
%
% The tolerance sweep is optional and disabled by default; Experiment 1 is the
% sole paper-facing completion-tolerance trade-off.

    if nargin < 1 || isempty(mode)
        mode = "paper";
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end
    mode = lower(string(mode));
    if mode ~= "smoke" && mode ~= "medium" && mode ~= "paper"
        error('mode must be smoke, medium, or paper.');
    end

    out_dir = helm_output_path('outputs', 'helm_trajectory_evidence');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    cfg = default_four_block_cfg(mode);
    cfg = merge_structs_local(cfg, user_cfg);

    fprintf('=== PC waveguide four-block DD-TA-ROM: %s ===\n', mode);
    results = photonic_crystal_waveguide_correction_rom(cfg);

    out_file = fullfile(out_dir, ['pc_four_block_partition_' char(mode) '.mat']);
    save(out_file, 'mode', 'cfg', 'results', '-v7.3');
    fprintf('Saved four-block PC result: %s\n', out_file);
    print_four_block_summary(results);
    maybe_plot_residual_ablation(mode, out_file);

    if should_run_tradeoff(cfg, mode)
        [tradeoff_table, tol_values, sweep] = run_tolerance_tradeoff(cfg, mode, out_dir);
        results.tradeoff_table = tradeoff_table;
        results.tol_values = tol_values;
        results.tradeoff_sweep = sweep;
        save(out_file, 'tradeoff_table', 'tol_values', 'sweep', '-append');
    end
end

function tf = should_run_tradeoff(cfg, mode)
    if isfield(cfg, 'run_tradeoff') && ~isempty(cfg.run_tradeoff)
        tf = logical(cfg.run_tradeoff);
        return;
    end
    tf = false;
end

function [tbl, tol_values, sweep] = run_tolerance_tradeoff(base_cfg, mode, out_dir)
    tol_values = choose_tradeoff_tolerances(base_cfg, mode);
    sweep = repmat(empty_tradeoff_entry(), numel(tol_values), 1);

    fprintf('\n=== PC four-block WaveHoltz tolerance trade-off sweep (%s) ===\n', mode);
    for k = 1:numel(tol_values)
        tol = tol_values(k);
        fprintf('\n--- PC four-block trade-off %d/%d: completion tol=%.1e ---\n', ...
            k, numel(tol_values), tol);

        cfg = base_cfg;
        cfg.do_plots = false;
        cfg.save_plots = false;
        cfg.save_benchmark = false;
        cfg.wh_tol = tol;
        cfg.fp_tol = tol;
        r = photonic_crystal_waveguide_correction_rom(cfg);
        sweep(k) = summarize_tradeoff_run(tol, r);
    end

    tbl = struct2table(sweep);
    stem = ['pc_tol_tradeoff_' char(mode)];
    csv_file = fullfile(out_dir, [stem '.csv']);
    mat_file = fullfile(out_dir, [stem '.mat']);
    writetable(tbl, csv_file);
    save(mat_file, 'mode', 'base_cfg', 'tol_values', 'sweep', 'tbl', '-v7.3');
    fprintf('Saved PC four-block tolerance sweep: %s\n', csv_file);
    fprintf('Saved PC four-block tolerance sweep: %s\n', mat_file);
end

function tol_values = choose_tradeoff_tolerances(cfg, mode)
    if isfield(cfg, 'tradeoff_tols') && ~isempty(cfg.tradeoff_tols)
        tol_values = cfg.tradeoff_tols(:).';
        return;
    end
    switch mode
        case "smoke"
            tol_values = [1e-4, 1e-6];
        otherwise
            tol_values = [1e-4, 1e-5, 1e-6, 1e-7];
    end
end

function entry = empty_tradeoff_entry()
    entry = struct( ...
        'tol', NaN, ...
        'mean_rank', NaN, ...
        'max_rank', NaN, ...
        'mean_online_time', NaN, ...
        'mean_plain_waveholtz_time', NaN, ...
        'speedup_vs_plain_waveholtz', NaN, ...
        'mean_final_full_error', NaN, ...
        'mean_precorr_relres', NaN, ...
        'mean_final_relres', NaN);
end

function entry = summarize_tradeoff_run(tol, r)
    entry = empty_tradeoff_entry();
    entry.tol = tol;
    if isfield(r.summary, 'corr_ranks') && ~isempty(r.summary.corr_ranks)
        entry.mean_rank = mean(r.summary.corr_ranks(:), 'omitnan');
        entry.max_rank = max(r.summary.corr_ranks(:), [], 'omitnan');
    end
    entry.mean_online_time = safe_stat_local(r.summary, 'online_total_time', 2);
    entry.mean_plain_waveholtz_time = safe_stat_local(r.summary, 'plain_fp_time', 2);
    entry.speedup_vs_plain_waveholtz = r.summary.time_speedup_online_vs_plain_mean;
    entry.mean_precorr_relres = safe_stat_local(r.summary, 'online_precorr_relres', 2);
    entry.mean_final_relres = safe_stat_local(r.summary, 'online_fp_relres', 2);
    if isfield(r.cfg, 'rom_check_target') && strcmpi(r.cfg.rom_check_target, 'helm') && ...
            isfield(r.summary, 'final_helm_full') && ~isempty(r.summary.final_helm_full)
        entry.mean_final_full_error = safe_stat_local(r.summary, 'final_helm_full', 2);
    else
        entry.mean_final_full_error = safe_stat_local(r.summary, 'final_wave_full', 2);
    end
end

function val = safe_stat_local(summary, field_name, idx)
    val = NaN;
    if ~isfield(summary, field_name) || isempty(summary.(field_name))
        return;
    end
    data = summary.(field_name);
    if numel(data) >= idx
        val = data(idx);
    end
end

function cfg = default_four_block_cfg(mode)
    cfg = struct();
    cfg.figure_subdir = 'photonic_crystal_correction_rom_4block';
    cfg.benchmark_name = ['photonic_crystal_correction_rom_4block_' char(mode)];
    cfg.rom_check_target = 'helm';
    cfg.online_completion_mode = 'waveholtz';
    cfg.online_wave_solver = 'gmres';
    cfg.run_wave_baselines = true;
    cfg.corr_basis_style = 'block3';
    cfg.block3_partition_mode = 'pc_core_cladding_lr_tb_pml';
    cfg.corr_num_layers = 3;
    cfg.corr_pod_singular_value_tols = [1e-4, 1e-3, 1e-2];

    cfg.corr_block3_global_tail_tols = cfg.corr_pod_singular_value_tols;
    cfg.corr_block3_global_tail_strategy = 'dof_weighted';
    cfg.corr_blockn_tail_tols = ones(3, 4);
    cfg.corr_blockn_rank_caps = Inf(3, 4);

    cfg.do_plots = mode ~= "smoke";
    cfg.save_plots = mode ~= "smoke";
    cfg.save_benchmark = mode ~= "smoke";
    cfg.parallel_train = mode ~= "smoke";
    cfg.parallel_test = mode ~= "smoke";
    cfg.test_param_strategy = 'random_interval';
    cfg.test_param_rng_seed = 1;

    switch mode
        case "smoke"
            cfg.reference_mode = 'helm_lift';
            cfg.pc_num_cols = 5;
            cfg.pc_num_rows = 1;
            cfg.pc_points_per_a = 8;
            cfg.npml = 8;
            cfg.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.98, 1.02, 4);
            cfg.wh_gmres_restart = 40;
            cfg.wh_gmres_maxit = 1;
            cfg.helm_baseline_iters = 20;

        case "medium"
            cfg.reference_mode = 'helm_lift';
            cfg.pc_num_cols = 7;
            cfg.pc_num_rows = 2;
            cfg.pc_points_per_a = 10;
            cfg.npml = 10;
            cfg.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.98, 1.02, 6);
            cfg.test_count = 4;
            cfg.wh_gmres_restart = 60;
            cfg.wh_gmres_maxit = 2;
            cfg.helm_baseline_iters = 40;
    end
end

function print_four_block_summary(results)
    if ~isfield(results, 'summary') || isempty(results.summary)
        return;
    end
    s = results.summary;
    if isfield(s, 'corr_block_labels') && ~isempty(s.corr_block_labels)
        fprintf('Block labels: %s\n', strjoin(s.corr_block_labels, ' / '));
    end
    if isfield(s, 'corr_block_ranks') && ~isempty(s.corr_block_ranks)
        fprintf('Block ranks by layer:\n');
        disp(s.corr_block_ranks);
    end
    if isfield(s, 'corr_block_dofs') && ~isempty(s.corr_block_dofs)
        fprintf('Block dofs:\n');
        disp(s.corr_block_dofs(1, :));
    end
    if isfield(s, 'corr_storage_block') && isfield(s, 'corr_storage_dense')
        fprintf('Total block/dense entries: %.6g / %.6g\n', ...
            sum(s.corr_storage_block, 'omitnan'), sum(s.corr_storage_dense, 'omitnan'));
    end
    if isfield(s, 'online_precorr_relres')
        fprintf('Mean precorrection relative residual: %.3e\n', s.online_precorr_relres(2));
    end
    if isfield(s, 'online_total_time')
        fprintf('Mean online total time: %.3fs\n', s.online_total_time(2));
    end
end

function maybe_plot_residual_ablation(mode, dd_mat)
    if mode ~= "paper"
        return;
    end
    global_mat = helm_output_path('outputs', 'helm_trajectory_evidence', ...
        'pc_residual_ablation_paper.mat');
    if exist(global_mat, 'file') ~= 2
        fprintf('Skipped four-block residual-ablation overlay; missing %s\n', global_mat);
        return;
    end
    try
        plot_pc_four_block_residual_ablation(global_mat, dd_mat);
    catch err
        warning('Could not plot four-block residual-ablation overlay: %s', err.message);
    end
end

function a = merge_structs_local(a, b)
    if isempty(b)
        return;
    end
    names = fieldnames(b);
    for i = 1:numel(names)
        a.(names{i}) = b.(names{i});
    end
end
