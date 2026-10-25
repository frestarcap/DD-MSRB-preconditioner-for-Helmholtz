function results = run_helm_trajectory_evidence_experiment(mode, user_cfg)
%RUN_HELM_TRAJECTORY_EVIDENCE_EXPERIMENT Generate half-space trajectory evidence.
%
% Usage:
%   results = run_helm_trajectory_evidence_experiment("smoke");
%   results = run_helm_trajectory_evidence_experiment("paper");
%
% The smoke mode is for syntax and plotting checks.  The paper mode uses the
% final Half-space tail schedule, the DOF-weighted DD allocation, the equal-rank
% non-replayed layered-ROM comparison, and the completion-tolerance sweep.

    if nargin < 1 || isempty(mode)
        mode = "paper";
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end
    mode = lower(string(mode));

    out_dir = helm_output_path('outputs', 'helm_trajectory_evidence');
    fig_subdir = 'helm_correction_rom';
    if mode == "smoke"
        fig_subdir = 'helm_correction_rom_smoke';
    end
    fig_dir = helm_output_path('outputs', 'figures', fig_subdir);
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    base_cfg = default_base_cfg(mode);
    base_cfg = merge_structs(base_cfg, user_cfg);

    fprintf('=== Helm half-space trajectory evidence: %s ===\n', mode);
    fprintf('Running representative trajectory/non-replayed comparison...\n');
    rep_cfg = base_cfg;
    rep_cfg.run_nr_rom_comparison = true;
    rep_cfg.do_plots = true;
    rep_cfg.save_plots = true;
    rep_cfg.figure_subdir = fig_subdir;
    rep_results = helm_waveholtz_correction_rom(rep_cfg);

    fprintf('\nRunning representative domain-decomposed comparison...\n');
    dd_cfg = make_dd_cfg(base_cfg);
    dd_results = helm_waveholtz_correction_rom(dd_cfg);

    plot_helm_residual_ablation(rep_results, fig_dir, dd_results);

    fprintf('\nRunning residual-tolerance speedup sweep...\n');
    tol_values = choose_tolerance_values(mode, base_cfg);
    sweep = repmat(empty_sweep_entry(), numel(tol_values), 1);
    for k = 1:numel(tol_values)
        tol = tol_values(k);
        fprintf('\n--- residual tolerance sweep %d/%d: %.1e ---\n', k, numel(tol_values), tol);
        cfg = base_cfg;
        cfg.run_nr_rom_comparison = false;
        cfg.do_plots = false;
        cfg.save_plots = false;
        cfg.save_benchmark = false;
        cfg.fp_tol = tol;
        r = helm_waveholtz_correction_rom(cfg);
        sweep(k) = summarize_tol_run(tol, r);
    end

    tradeoff_table = struct2table(sweep);
    writetable(tradeoff_table, fullfile(out_dir, ['helm_tol_tradeoff_' char(mode) '.csv']));
    save(fullfile(out_dir, ['helm_trajectory_evidence_' char(mode) '.mat']), ...
        'mode', 'base_cfg', 'tol_values', 'rep_results', 'dd_results', 'sweep', 'tradeoff_table', '-v7.3');

    plot_helm_tol_tradeoff(tradeoff_table, fig_dir, mode, rep_results);

    results = struct();
    results.mode = mode;
    results.base_cfg = base_cfg;
    results.representative = rep_results;
    results.dd_results = dd_results;
    results.tolerance_sweep = sweep;
    results.tradeoff_table = tradeoff_table;
end

function cfg = make_dd_cfg(base_cfg)
    cfg = halfspace_paper_dd_config(base_cfg);
    cfg.run_nr_rom_comparison = false;
    cfg.do_plots = false;
    cfg.save_plots = false;
    cfg.save_benchmark = false;
    cfg.corr_block3_global_tail_tols = cfg.corr_pod_singular_value_tols;
    cfg.corr_block3_global_tail_strategy = 'marginal_storage';
    cfg.corr_block3_tail_tol_near = ones(1, cfg.corr_num_layers);
    cfg.corr_block3_tail_tol_far = ones(1, cfg.corr_num_layers);
    cfg.corr_block3_tail_tol_pml = ones(1, cfg.corr_num_layers);
    cfg.corr_block3_rank_cap_near = Inf;
    cfg.corr_block3_rank_cap_far = Inf;
    cfg.corr_block3_rank_cap_pml = Inf;
end

function cfg = default_base_cfg(mode)
    cfg = struct();
    cfg.material_model = 'halfspace';
    cfg.figure_subdir = 'helm_correction_rom';
    cfg.rom_check_target = 'helm';
    cfg.reference_mode = 'helm_lift';
    cfg.online_completion_mode = 'waveholtz';
    cfg.online_wave_solver = 'fixedpoint';
    cfg.use_direct_rom_init = false;
    cfg.corr_basis_style = 'global';
    cfg.corr_num_layers = 3;
    cfg.corr_pod_singular_value_tols = [1e-5, 1e-5, 1e-5];
    cfg.corr_block3_global_tail_strategy = 'marginal_storage';
    cfg.online_helm_accept_tol = 1e-12;
    cfg.fp_tol = 1e-12;
    cfg.run_wave_baselines = true;
    cfg.parallel_train = true;
    cfg.parallel_test = true;

    switch mode
        case "smoke"
            cfg.Nphys = 45;
            cfg.train_params = linspace(0.80, 0.95, 6);
            cfg.test_params = [0.8130, 0.9072];
            cfg.wh_max_iter = 80;
            cfg.fp_maxit = 40;
            cfg.wave_baseline_max_iter = 60;
            cfg.parallel_train = false;
            cfg.parallel_test = false;
        otherwise
            cfg.train_params = linspace(0.80, 0.95, 24);
            cfg.test_param_strategy = 'random_interval';
            cfg.test_param_rng_seed = 1;
            cfg.plot_test_index = [];
    end
end

function tol_values = choose_tolerance_values(mode, cfg)
    if isfield(cfg, 'tradeoff_tols') && ~isempty(cfg.tradeoff_tols)
        tol_values = cfg.tradeoff_tols(:).';
        return;
    end
    switch mode
        case "smoke"
            tol_values = [1e-10, 1e-7];
        otherwise
            tol_values = [1e-14, 1e-12, 1e-10, 1e-7];
    end
end

function entry = empty_sweep_entry()
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

function entry = summarize_tol_run(tol, r)
    entry = empty_sweep_entry();
    entry.tol = tol;
    entry.mean_rank = mean(r.summary.corr_ranks, 'omitnan');
    entry.max_rank = max(r.summary.corr_ranks, [], 'omitnan');
    entry.mean_online_time = r.summary.online_total_time(2);
    entry.mean_plain_waveholtz_time = r.summary.plain_fp_time(2);
    entry.speedup_vs_plain_waveholtz = r.summary.time_speedup_online_vs_plain_mean;
    if strcmp(r.cfg.rom_check_target, 'helm') && ~isempty(r.test.helm_u_ref)
        entry.mean_final_full_error = mean(r.test.online_final_err_helm_full, 'omitnan');
    else
        entry.mean_final_full_error = mean(r.test.online_final_err_wave_full, 'omitnan');
    end
    entry.mean_precorr_relres = r.summary.online_precorr_relres(2);
    entry.mean_final_relres = r.summary.online_fp_relres(2);
end

function plot_helm_tol_tradeoff(tbl, fig_dir, mode, rep_results)
    fig = figure('Name', 'WaveHoltz residual tolerance speedup', 'Position', [120 120 720 480]);
    ax = axes(fig);
    semilogx(ax, tbl.tol, tbl.speedup_vs_plain_waveholtz, '-o', ...
        'Color', [0.10 0.35 0.75], 'LineWidth', 1.9, 'MarkerSize', 7, ...
        'DisplayName', 'TA-ROM speedup');
    yline(ax, 1, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    set(ax, 'XDir', 'reverse');
    xlabel(ax, 'WaveHoltz completion residual tolerance');
    ylabel(ax, 'speedup over plain WaveHoltz');
    title(ax, 'Half-space WaveHoltz residual-tolerance speedup');
    grid(ax, 'on');
    box(ax, 'on');
    legend(ax, 'Location', 'best');
    exportgraphics(fig, fullfile(fig_dir, ['helm_tol_tradeoff_' char(mode) '.png']), 'Resolution', 300);

    fig2 = figure('Name', 'Helm correction singular value decay', 'Position', [140 140 700 480]);
    ax2 = axes(fig2);
    hold(ax2, 'on');
    for j = 1:rep_results.correction_rom.num_layers
        sv = rep_results.correction_rom.layers{j}.pod.singular_values;
        if has_finite_curve(sv)
            semilogy(ax2, 1:numel(sv), clamp_positive_curve(sv, 1e-16), ...
                'o-', 'LineWidth', 1.5, 'MarkerSize', 5, ...
                'DisplayName', sprintf('layer %d', j));
        end
    end
    hold(ax2, 'off');
    xlabel(ax2, 'mode index');
    ylabel(ax2, 'singular value');
    title(ax2, 'Half-space correction singular-value decay');
    grid(ax2, 'on');
    box(ax2, 'on');
    legend(ax2, 'Location', 'best');
    exportgraphics(fig2, fullfile(fig_dir, ['helm_correction_singular_decay_' char(mode) '.png']), 'Resolution', 300);
end

function plot_helm_residual_ablation(rep_results, fig_dir, dd_results)
    if nargin < 3
        dd_results = [];
    end
    traj_test = rep_results.test;
    traj_summary = rep_results.summary;
    rep_idx = traj_summary.representative_test_index;
    c_rep = traj_summary.representative_c;
    nr = rep_results.nr_comparison;
    plot_floor = 1e-16;

    fig = figure('Name', 'Half-space TA/NR residual decay', 'Position', [140 140 760 500]);
    ax = axes(fig);
    hold(ax, 'on');

    traj_hist = traj_test.online_combined_relres_hist{rep_idx};
    if has_finite_curve(traj_hist)
        semilogy(ax, 0:(numel(traj_hist) - 1), clamp_positive_curve(traj_hist, plot_floor), ...
            '-o', 'Color', [0.85 0.05 0.05], 'LineWidth', 1.8, 'MarkerSize', 6, ...
            'DisplayName', 'TA-ROM');
    end

    if ~isempty(dd_results) && isfield(dd_results, 'test')
        dd_idx = choose_matching_test_index(dd_results.test, c_rep, dd_results.summary);
        dd_hist = dd_results.test.online_combined_relres_hist{dd_idx};
        if has_finite_curve(dd_hist)
            semilogy(ax, 0:(numel(dd_hist) - 1), clamp_positive_curve(dd_hist, plot_floor), ...
                '-s', 'Color', [0.05 0.45 0.25], 'LineWidth', 1.6, 'MarkerSize', 6, ...
                'DisplayName', 'DD-TA-ROM');
        end
    end

    if ~isempty(nr) && isfield(nr, 'test')
        nr_idx = choose_matching_test_index(nr.test, c_rep, nr.summary);
        nr_hist = nr.test.online_combined_relres_hist{nr_idx};
        if has_finite_curve(nr_hist)
            semilogy(ax, 0:(numel(nr_hist) - 1), clamp_positive_curve(nr_hist, plot_floor), ...
                '--d', 'Color', [0.15 0.35 0.75], 'LineWidth', 1.6, 'MarkerSize', 6, ...
                'DisplayName', 'NR-ROM');
        end
    end

    plain_hist = traj_test.plain_fp_relres_hist{rep_idx};
    if has_finite_curve(plain_hist)
        semilogy(ax, 0:(numel(plain_hist) - 1), clamp_positive_curve(plain_hist, plot_floor), ...
            'k-o', 'LineWidth', 1.3, 'MarkerSize', 5, ...
            'DisplayName', 'Plain WaveHoltz');
    end

    helm_hist = traj_test.helm_baseline_relres_hist{rep_idx};
    if has_finite_curve(helm_hist)
        semilogy(ax, 0:(numel(helm_hist) - 1), clamp_positive_curve(helm_hist, plot_floor), ...
            ':', 'Color', [0.25 0.25 0.25], 'LineWidth', 1.5, ...
            'DisplayName', 'Plain Helmholtz GMRES');
    end

    hold(ax, 'off');
    set(ax, 'YScale', 'log', 'FontSize', 18, 'LineWidth', 1.4);
    ylim(ax, [plot_floor, 2]);
    xlabel(ax, 'online step index');
    ylabel(ax, 'relative residual');
    title(ax, sprintf('Half-space residual decay, c_{right}=%.4f', c_rep));
    grid(ax, 'on');
    box(ax, 'on');
    lgd = legend(ax, 'Location', 'best');
    lgd.FontSize = 15;
    tag = sanitize_filename_local(sprintf('c_right_%0.4f', c_rep));
    exportgraphics(fig, fullfile(fig_dir, ['helm_trajectory_vs_nr_residual_decay_' tag '.png']), 'Resolution', 300);
end

function idx = choose_matching_test_index(test, c_rep, summary)
    if isfield(test, 'params') && ~isempty(test.params)
        [~, idx] = min(abs(test.params(:) - c_rep));
        return;
    end
    idx = summary.representative_test_index;
end

function tf = has_finite_curve(x)
    tf = ~isempty(x) && any(isfinite(x(:)));
end

function y = clamp_positive_curve(x, floor_value)
    y = real(x(:));
    y(~isfinite(y) | y <= floor_value) = floor_value;
end

function stem = sanitize_filename_local(stem)
    stem = regexprep(char(stem), '[^A-Za-z0-9_\-]+', '_');
    stem = regexprep(stem, '_+', '_');
    stem = regexprep(stem, '^_|_$', '');
end

function out = merge_structs(a, b)
    out = a;
    names = fieldnames(b);
    for i = 1:numel(names)
        out.(names{i}) = b.(names{i});
    end
end
