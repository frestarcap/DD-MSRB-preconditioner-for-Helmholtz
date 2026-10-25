function results = run_pc_ecc_residual_ablation_experiment(problem_type, mode, user_cfg)
%RUN_PC_ECC_RESIDUAL_ABLATION_EXPERIMENT
% Generate residual-ablation data for the PC and scattering examples.
%
% The script runs a global trajectory-aware correction ROM and a blockwise
% domain-decomposed trajectory-aware correction ROM, then plots both against plain WaveHoltz
% and plain Helmholtz GMRES on the representative test case.

    if nargin < 1 || isempty(problem_type)
        problem_type = "both";
    end
    if nargin < 2 || isempty(mode)
        mode = "paper";
    end
    if nargin < 3 || isempty(user_cfg)
        user_cfg = struct();
    end

    problem_type = lower(string(problem_type));
    mode = lower(string(mode));

    switch problem_type
        case "both"
            results = struct();
            results.pc = run_one_problem("pc", mode, user_cfg);
            results.ecc = run_one_problem("ecc", mode, user_cfg);
        case {"pc", "photonic_crystal", "waveguide"}
            results = run_one_problem("pc", mode, user_cfg);
        case {"ecc", "scattering"}
            results = run_one_problem("ecc", mode, user_cfg);
        otherwise
            error('problem_type must be pc, ecc, or both.');
    end
end

function results = run_one_problem(problem_type, mode, user_cfg)
    out_dir = helm_output_path('outputs', 'helm_trajectory_evidence');
    ensure_dir(out_dir);

    base_cfg = default_cfg(problem_type, mode);
    base_cfg = merge_structs(base_cfg, user_cfg);
    validate_paper_source_cfg(problem_type, mode, base_cfg);
    if isfield(base_cfg, 'output_tag') && ~isempty(base_cfg.output_tag)
        assert(~isempty(regexp(base_cfg.output_tag, '^[A-Za-z0-9_-]+$', 'once')));
        out_dir = fullfile(out_dir, base_cfg.output_tag);
        ensure_dir(out_dir);
    end

    fprintf('\n=== Running %s Global TA-ROM residual ablation ===\n', upper(problem_type));
    global_cfg = configure_global(problem_type, base_cfg);
    global_results = run_cached_problem(problem_type, global_cfg, out_dir, mode, 'global');

    fprintf('\n=== Running %s DD-TA-ROM residual ablation ===\n', upper(problem_type));
    dd_cfg = configure_dd(problem_type, base_cfg);
    dd_results = run_cached_problem(problem_type, dd_cfg, out_dir, mode, 'dd');

    speedup_comparison = build_online_speedup_comparison(global_results, dd_results);
    print_online_speedup_comparison(problem_type, speedup_comparison);

    fig_dir = helm_output_path('outputs', 'figures', figure_subdir(problem_type));
    if isfield(base_cfg, 'output_tag') && ~isempty(base_cfg.output_tag)
        fig_dir = fullfile(fig_dir, base_cfg.output_tag);
    end
    ensure_dir(fig_dir);
    plot_residual_ablation(problem_type, global_results, dd_results, fig_dir);

    tradeoff_table = table();
    if should_run_tradeoff(base_cfg)
        tradeoff_table = run_tolerance_tradeoff(problem_type, mode, base_cfg, out_dir, fig_dir);
    end

    results = struct();
    results.problem_type = char(problem_type);
    results.mode = char(mode);
    results.base_cfg = base_cfg;
    results.global_results = global_results;
    results.dd_results = dd_results;
    results.speedup_comparison = speedup_comparison;
    results.tradeoff_table = tradeoff_table;

    save(fullfile(out_dir, sprintf('%s_residual_ablation_%s.mat', problem_type, mode)), ...
        'results', 'global_results', 'dd_results', 'base_cfg', 'speedup_comparison', 'tradeoff_table', '-v7.3');
    if mode == "paper" && strcmpi(base_cfg.reference_mode, 'waveholtz') && ...
            strcmpi(base_cfg.online_completion_mode, 'waveholtz')
        save(fullfile(out_dir, sprintf('%s_waveholtz_residual_ablation_%s.mat', problem_type, mode)), ...
            'results', 'global_results', 'dd_results', 'base_cfg', 'speedup_comparison', 'tradeoff_table', '-v7.3');
    end
end

function validate_paper_source_cfg(problem_type, mode, cfg)
    if mode ~= "paper"
        return;
    end
    if ~any(strcmp(problem_type, ["pc", "ecc"]))
        return;
    end
    expected_reference = 'helm_lift';
    if problem_type == "ecc"
        expected_reference = 'helmholtz';
    end
    if ~isfield(cfg, 'reference_mode') || ~strcmpi(cfg.reference_mode, expected_reference) || ...
            ~isfield(cfg, 'online_completion_mode') || ~strcmpi(cfg.online_completion_mode, 'waveholtz')
        error(['%s paper residual-ablation outputs must use %s reference data ' ...
            'and WaveHoltz completion.'], upper(problem_type), expected_reference);
    end
end

function cfg = default_cfg(problem_type, mode)
    cfg = struct();
    cfg.do_plots = false;
    cfg.save_plots = false;
    cfg.save_benchmark = false;
    cfg.run_wave_baselines = true;
    cfg.run_tradeoff = false;
    cfg.parallel_train = mode ~= "smoke";
    cfg.parallel_test = mode ~= "smoke";
    cfg.corr_block3_global_tail_strategy = 'marginal_storage';

    switch problem_type
        case "pc"
            cfg.corr_rank_rtols = [1e-4, 1e-3, 1e-2];
            cfg.corr_pod_singular_value_tols = cfg.corr_rank_rtols;
            cfg.figure_subdir = 'photonic_crystal_correction_rom';
            cfg.reference_mode = 'helm_lift';
            cfg.rom_check_target = 'helm';
            cfg.online_completion_mode = 'waveholtz';
            cfg.online_wave_solver = 'gmres';
            cfg.test_param_strategy = 'random_interval';
            cfg.test_param_rng_seed = 1;
            cfg.corr_num_layers = 3;
            cfg.block3_partition_mode = 'pc_core_cladding_lr_tb_pml';
            cfg.block3_waveguide_halfwidth = 0.75;
            if mode == "smoke"
                cfg.pc_num_cols = 5;
                cfg.pc_num_rows = 1;
                cfg.pc_points_per_a = 8;
                cfg.npml = 8;
                cfg.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.98, 1.02, 4);
                cfg.wh_gmres_restart = 40;
                cfg.wh_gmres_maxit = 1;
                cfg.helm_baseline_iters = 20;
            end

        case "ecc"
            cfg.corr_rank_rtols = [1e-3, 1e-2, 1e-1];
            cfg.corr_pod_singular_value_tols = cfg.corr_rank_rtols;
            cfg.figure_subdir = 'ecc_correction_rom';
            cfg.reference_mode = 'helmholtz';
            cfg.rom_check_target = 'wave';
            cfg.online_completion_mode = 'waveholtz';
            cfg.ratio_test_strategy = 'random_interval';
            cfg.ratio_test_rng_seed = 1;
            cfg.corr_num_layers = 3;
            if mode == "smoke"
                cfg.Nphys = 24;
                cfg.npml = 6;
                cfg.omega = 10;
                cfg.ratio_train = [0.94 1.00 1.06];
                cfg.ratio_test = 0.97;
                cfg.wh_max_iter = 40;
                cfg.helm_baseline_iters = 20;
                cfg.helm_completion_restart = 10;
                cfg.helm_completion_maxit = 4;
            end
    end
end

function tf = should_run_tradeoff(cfg)
    tf = isfield(cfg, 'run_tradeoff') && ~isempty(cfg.run_tradeoff) && logical(cfg.run_tradeoff);
end

function tbl = run_tolerance_tradeoff(problem_type, mode, base_cfg, out_dir, fig_dir)
    tol_values = choose_tradeoff_tolerances(problem_type, mode, base_cfg);
    sweep = repmat(empty_tradeoff_entry(), numel(tol_values), 1);

    fprintf('\n=== Running %s WaveHoltz tolerance trade-off sweep ===\n', upper(problem_type));
    for k = 1:numel(tol_values)
        tol = tol_values(k);
        fprintf('\n--- %s trade-off sweep %d/%d: completion tol=%.1e ---\n', ...
            upper(problem_type), k, numel(tol_values), tol);

        cfg = configure_global(problem_type, base_cfg);
        cfg.run_tradeoff = false;
        cfg.do_plots = false;
        cfg.save_plots = false;
        cfg.save_benchmark = false;
        cfg.wh_tol = tol;
        cfg.fp_tol = tol;
        r = run_problem(problem_type, cfg);
        sweep(k) = summarize_tradeoff_run(tol, r);
    end

    tbl = struct2table(sweep);
    stem = sprintf('%s_tol_tradeoff_%s', problem_type, mode);
    writetable(tbl, fullfile(out_dir, [char(stem) '.csv']));
    save(fullfile(out_dir, [char(stem) '.mat']), 'mode', 'base_cfg', 'tol_values', 'sweep', 'tbl', '-v7.3');
    plot_tolerance_tradeoff(problem_type, tbl, fig_dir, mode);
end

function tol_values = choose_tradeoff_tolerances(problem_type, mode, cfg)
    if isfield(cfg, 'tradeoff_tols') && ~isempty(cfg.tradeoff_tols)
        tol_values = cfg.tradeoff_tols(:).';
        return;
    end

    switch mode
        case "smoke"
            tol_values = [1e-4, 1e-6];
        otherwise
            switch problem_type
                case "pc"
                    tol_values = [1e-4, 1e-5, 1e-6];
                case "ecc"
                    tol_values = [1e-6, 1e-8, 1e-10];
            end
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
    entry.mean_rank = mean(r.summary.corr_ranks(:), 'omitnan');
    entry.max_rank = max(r.summary.corr_ranks(:), [], 'omitnan');
    entry.mean_online_time = safe_stat(r.summary, 'online_total_time', 2);
    entry.mean_plain_waveholtz_time = safe_stat(r.summary, 'plain_fp_time', 2);
    entry.speedup_vs_plain_waveholtz = r.summary.time_speedup_online_vs_plain_mean;
    entry.mean_precorr_relres = safe_stat(r.summary, 'online_precorr_relres', 2);
    entry.mean_final_relres = safe_stat(r.summary, 'online_fp_relres', 2);

    if strcmpi(r.cfg.rom_check_target, 'helm')
        entry.mean_final_full_error = safe_stat(r.summary, 'final_helm_full', 2);
    else
        entry.mean_final_full_error = safe_stat(r.summary, 'final_wave_full', 2);
    end
end

function val = safe_stat(summary, field_name, idx)
    val = NaN;
    if ~isfield(summary, field_name) || isempty(summary.(field_name))
        return;
    end
    data = summary.(field_name);
    if numel(data) >= idx
        val = data(idx);
    end
end

function plot_tolerance_tradeoff(problem_type, tbl, fig_dir, mode)
    switch problem_type
        case "pc"
            title_text = 'Photonic-crystal WaveHoltz tolerance speedup';
            out_name = ['pc_tol_tradeoff_' char(mode) '.png'];
            legend_location = 'best';
        case "ecc"
            title_text = 'Scattering WaveHoltz tolerance speedup';
            out_name = ['ecc_tol_tradeoff_' char(mode) '.png'];
            legend_location = 'northeast';
    end

    fig = figure('Name', title_text, 'Position', [120 120 720 480]);
    ax = axes(fig);
    semilogx(ax, tbl.tol, tbl.speedup_vs_plain_waveholtz, '-o', ...
        'Color', [0.10 0.35 0.75], 'LineWidth', 1.9, 'MarkerSize', 7, ...
        'DisplayName', 'Global TA-ROM speedup');
    yline(ax, 1, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    set(ax, 'XDir', 'reverse');
    xlabel(ax, 'WaveHoltz completion residual tolerance');
    ylabel(ax, 'speedup over plain WaveHoltz');
    title(ax, title_text);
    grid(ax, 'on');
    box(ax, 'on');
    legend(ax, 'Location', legend_location);
    exportgraphics(fig, fullfile(fig_dir, out_name), 'Resolution', 300);
end

function cfg = configure_global(problem_type, cfg)
    switch problem_type
        case "pc"
            cfg.corr_basis_style = 'global';
        case "ecc"
            cfg.direct_basis_style = 'global';
            cfg.corr_basis_style = 'global';
    end
end

function cfg = configure_dd(problem_type, cfg)
    switch problem_type
        case "pc"
            cfg.corr_basis_style = 'block3';
            cfg.block3_partition_mode = 'pc_core_cladding_lr_tb_pml';
            cfg.block3_waveguide_halfwidth = 0.75;
            cfg = set_default_field_local(cfg, 'corr_block3_rank_cap_near', Inf);
            cfg = set_default_field_local(cfg, 'corr_block3_rank_cap_far', Inf);
            cfg = set_default_field_local(cfg, 'corr_block3_rank_cap_pml', Inf);
            cfg.corr_block3_tail_tol_near = [1, 1, 1];
            cfg.corr_block3_tail_tol_far = [1, 1, 1];
            cfg.corr_block3_tail_tol_pml = [1, 1, 1];
            cfg.corr_block3_global_tail_tols = cfg.corr_rank_rtols;
            cfg = set_default_field_local(cfg, 'corr_block3_global_tail_strategy', 'marginal_storage');
            cfg.corr_blockn_tail_tols = ones(3, 4);
            cfg.corr_blockn_rank_caps = Inf(3, 4);
        case "ecc"
            cfg.direct_basis_style = 'block3';
            cfg.corr_basis_style = 'block3';
            cfg.corr_block3_tail_tol_near = [1, 1, 1];
            cfg.corr_block3_tail_tol_far = [1, 1, 1];
            cfg.corr_block3_tail_tol_pml = [1, 1, 1];
            cfg.corr_block3_global_tail_tols = cfg.corr_rank_rtols;
            cfg = set_default_field_local(cfg, 'corr_block3_global_tail_strategy', 'marginal_storage');
    end
end

function r = run_problem(problem_type, cfg)
    switch problem_type
        case "pc"
            r = photonic_crystal_waveguide_correction_rom(cfg);
        case "ecc"
            r = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
    end
end

function r = run_cached_problem(problem_type, cfg, out, mode, style)
    % Tagged experiments can resume without losing a completed representation.
    if ~isfield(cfg, 'output_tag') || isempty(cfg.output_tag)
        r = run_problem(problem_type, cfg);
        return;
    end
    file = fullfile(out, sprintf('%s_%s_%s_checkpoint.mat',problem_type,mode,style));
    if isfile(file)
        s = load(file, 'r', 'requested_cfg');
        assert(isequaln(s.requested_cfg,cfg), 'Stale checkpoint; use a new tag.');
        r = s.r;
    else
        r = run_problem(problem_type, cfg);
        requested_cfg = cfg;
        save(file,'r','requested_cfg','-v7.3');
    end
end

function tbl = build_online_speedup_comparison(global_results, dd_results)
    methods = {'Global TA-ROM'; 'DD-TA-ROM'};
    plain_mean_s = [
        safe_stat(global_results.summary, 'plain_fp_time', 2);
        safe_stat(dd_results.summary, 'plain_fp_time', 2)];
    online_mean_s = [
        safe_stat(global_results.summary, 'online_total_time', 2);
        safe_stat(dd_results.summary, 'online_total_time', 2)];
    speedup_vs_plain = [
        global_results.summary.time_speedup_online_vs_plain_mean;
        dd_results.summary.time_speedup_online_vs_plain_mean];
    completion_iter_mean = [
        safe_stat(global_results.summary, 'online_fp_iter', 2);
        safe_stat(dd_results.summary, 'online_fp_iter', 2)];
    completion_apply_mean = [
        safe_stat(global_results.summary, 'online_fp_op_apply', 2);
        safe_stat(dd_results.summary, 'online_fp_op_apply', 2)];
    precorr_time_mean_s = [
        safe_stat(global_results.summary, 'online_precorr_time', 2);
        safe_stat(dd_results.summary, 'online_precorr_time', 2)];
    final_relerr_mean = [
        primary_final_error_mean(global_results);
        primary_final_error_mean(dd_results)];

    tbl = table(methods, plain_mean_s, online_mean_s, speedup_vs_plain, ...
        completion_iter_mean, completion_apply_mean, precorr_time_mean_s, ...
        final_relerr_mean, ...
        'VariableNames', {'method', 'plain_mean_s', 'online_mean_s', ...
        'speedup_vs_plain', 'completion_iter_mean', 'completion_apply_mean', ...
        'precorr_time_mean_s', 'final_relerr_mean'});
end

function print_online_speedup_comparison(problem_type, tbl)
    fprintf('\n=== %s Global/DD online speedup comparison ===\n', upper(problem_type));
    disp(tbl);
    if height(tbl) >= 2 && all(isfinite(tbl.online_mean_s(1:2))) && tbl.online_mean_s(1) > 0
        fprintf('DD/global online time ratio = %.3fx\n', tbl.online_mean_s(2) / tbl.online_mean_s(1));
    end
    if height(tbl) >= 2 && all(isfinite(tbl.speedup_vs_plain(1:2))) && tbl.speedup_vs_plain(1) > 0
        fprintf('DD speedup retention vs global = %.1f%%\n', ...
            100 * tbl.speedup_vs_plain(2) / tbl.speedup_vs_plain(1));
    end
end

function val = primary_final_error_mean(r)
    if isfield(r.cfg, 'rom_check_target') && strcmpi(r.cfg.rom_check_target, 'helm') && ...
            isfield(r.summary, 'final_helm_full') && ~isempty(r.summary.final_helm_full)
        val = safe_stat(r.summary, 'final_helm_full', 2);
    else
        val = safe_stat(r.summary, 'final_wave_full', 2);
    end
end

function plot_residual_ablation(problem_type, global_results, dd_results, fig_dir)
    rep_idx = global_results.summary.representative_test_index;
    plot_floor = 1e-16;

    switch problem_type
        case "pc"
            param_name = 'c';
            param_value = global_results.summary.representative_c;
            tag = sanitize_filename_local(sprintf('c_right_%0.4f', param_value));
            title_text = sprintf('Waveguide residual decay, c=%.4f', param_value);
            output_name = ['pc_joint_combined_residual_decay_full_field_' tag '.png'];
            dd_idx = choose_matching_index(dd_results.test.params, param_value, dd_results.summary.representative_test_index);
        case "ecc"
            param_name = 'a/b';
            param_value = global_results.summary.representative_ratio;
            tag = sanitize_filename_local(sprintf('ratio_%0.4f', param_value));
            title_text = sprintf('Scattering residual decay, a/b=%.4f', param_value);
            output_name = ['ecc_joint_combined_residual_decay_full_field_' tag '.png'];
            dd_idx = choose_matching_index(dd_results.test.ratio, param_value, dd_results.summary.representative_test_index);
    end

    fig = figure('Name', sprintf('%s residual ablation', upper(problem_type)), ...
        'Position', [140 140 760 500]);
    ax = axes(fig);
    hold(ax, 'on');

    plot_one_curve(ax, global_results.test.online_combined_relres_hist{rep_idx}, ...
        '-o', [0.85 0.05 0.05], 'Global TA-ROM', plot_floor);
    plot_one_curve(ax, dd_results.test.online_combined_relres_hist{dd_idx}, ...
        '-s', [0.05 0.45 0.25], 'DD-TA-ROM', plot_floor);
    plot_one_curve(ax, global_results.test.plain_fp_relres_hist{rep_idx}, ...
        'k-o', [0 0 0], plain_completion_label_local(global_results.cfg), plot_floor);
    plot_one_curve(ax, global_results.test.helm_baseline_relres_hist{rep_idx}, ...
        ':', [0.25 0.25 0.25], 'Plain Helmholtz GMRES', plot_floor);

    hold(ax, 'off');
    set(ax, 'YScale', 'log');
    ylim(ax, [plot_floor, 2]);
    xlabel(ax, 'online step index');
    ylabel(ax, 'relative residual');
    title(ax, title_text);
    grid(ax, 'on');
    box(ax, 'on');
    set(ax, 'FontSize', 18, 'LineWidth', 1.1);
    legend(ax, 'Location', 'northeast', 'FontSize', 17);
    exportgraphics(fig, fullfile(fig_dir, output_name), 'Resolution', 300);

    fprintf('Saved %s residual ablation for %s=%.4f\n', problem_type, param_name, param_value);
end

function label = plain_completion_label_local(cfg)
    if ~isfield(cfg, 'online_completion_mode') || isempty(cfg.online_completion_mode)
        label = 'Plain WaveHoltz';
        return;
    end
    mode = lower(strrep(strrep(char(string(cfg.online_completion_mode)), '-', '_'), ' ', '_'));
    switch mode
        case {'helm', 'helmholtz', 'plain_helmholtz', 'helm_gmres'}
            label = 'Plain Helmholtz completion';
        case {'shifted_laplacian', 'shifted_lap', 'sl', 'shifted'}
            label = 'Plain shifted-Laplacian';
        otherwise
            label = 'Plain WaveHoltz';
    end
end

function plot_one_curve(ax, hist, style, color, display_name, plot_floor)
    if isempty(hist) || ~any(isfinite(hist(:)))
        return;
    end
    y = real(hist(:));
    y(~isfinite(y) | y <= plot_floor) = plot_floor;
    semilogy(ax, 0:(numel(y) - 1), y, style, ...
        'Color', color, 'LineWidth', 1.8, 'MarkerSize', 6, ...
        'DisplayName', display_name);
end

function idx = choose_matching_index(values, target, fallback)
    if ~isempty(values)
        [~, idx] = min(abs(values(:) - target));
    else
        idx = fallback;
    end
end

function subdir = figure_subdir(problem_type)
    switch problem_type
        case "pc"
            subdir = 'photonic_crystal_correction_rom';
        case "ecc"
            subdir = 'ecc_correction_rom';
    end
end

function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end

function out = merge_structs(a, b)
    out = a;
    if isempty(b)
        return;
    end
    names = fieldnames(b);
    for i = 1:numel(names)
        out.(names{i}) = b.(names{i});
    end
end

function cfg = set_default_field_local(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end

function stem = sanitize_filename_local(stem)
    stem = regexprep(char(stem), '[^A-Za-z0-9_\-]+', '_');
    stem = regexprep(stem, '_+', '_');
    stem = regexprep(stem, '^_|_$', '');
end
