function results = run_rom_threshold_balance_probe(problem_type, mode, user_cfg)
%RUN_ROM_THRESHOLD_BALANCE_PROBE Study ROM truncation accuracy/storage balance.
%
% This is a research-only sweep.  It does not overwrite the paper evidence
% files and stores only compact summaries, not the full trained ROM objects.
%
% Examples:
%   run_rom_threshold_balance_probe("halfspace", "smoke");
%   run_rom_threshold_balance_probe("ecc", "medium");
%   run_rom_threshold_balance_probe("pc", "paper", struct( ...
%       'schedules', [1e-4 1e-4 1e-4]));
%   run_rom_threshold_balance_probe("ecc", "paper", struct( ...
%       'schedules', [1e-3 1e-2 1e-1], ...
%       'solver_cfg', struct('corr_block3_global_tail_strategy', ...
%           'dof_weighted'), 'output_tag', 'dof_weighted'));

    if nargin < 1 || isempty(problem_type)
        problem_type = "all";
    end
    if nargin < 2 || isempty(mode)
        mode = "smoke";
    end
    if nargin < 3 || isempty(user_cfg)
        user_cfg = struct();
    end

    problem_type = normalize_problem(problem_type);
    mode = lower(string(mode));
    if ~any(mode == ["smoke", "medium", "paper"])
        error('mode must be smoke, medium, or paper.');
    end

    if problem_type == "all"
        problems = ["halfspace", "ecc", "pc"];
    else
        problems = problem_type;
    end

    out_dir = helm_output_path('outputs', 'rom_threshold_balance');
    fig_dir = helm_output_path('outputs', 'figures', 'rom_threshold_balance');
    ensure_dir(out_dir);
    ensure_dir(fig_dir);

    results = struct();
    results.mode = char(mode);
    results.problems = cellstr(problems);

    for problem = problems
        schedules = choose_schedules(problem, user_cfg);
        base_cfg = default_probe_cfg(problem, mode);
        base_cfg = merge_structs_local(base_cfg, choose_solver_overrides(problem, user_cfg));

        fprintf('\n============================================================\n');
        fprintf('ROM threshold balance probe: %s (%s)\n', upper(problem), mode);
        fprintf('Schedules:\n');
        disp(schedules);

        tbl = run_problem_sweep(problem, mode, schedules, base_cfg);
        stem = sprintf('rom_threshold_balance_%s_%s', problem, mode);
        output_tag = normalized_output_tag(user_cfg);
        if ~isempty(output_tag)
            stem = sprintf('%s_%s', stem, output_tag);
        end
        csv_file = fullfile(out_dir, [char(stem) '.csv']);
        mat_file = fullfile(out_dir, [char(stem) '.mat']);
        fig_file = fullfile(fig_dir, [char(stem) '.png']);

        writetable(tbl, csv_file);
        save(mat_file, 'problem', 'mode', 'schedules', 'base_cfg', ...
            'output_tag', 'tbl');
        plot_balance_summary(problem, mode, tbl, fig_file);

        one = struct();
        one.problem = char(problem);
        one.mode = char(mode);
        one.schedules = schedules;
        one.base_cfg = base_cfg;
        one.table = tbl;
        one.csv_file = csv_file;
        one.mat_file = mat_file;
        one.figure_file = fig_file;
        results.(char(problem)) = one;

        fprintf('\nCompact threshold balance summary (%s):\n', upper(problem));
        disp(tbl(:, {'schedule', 'global_precorr_relres', 'dd_precorr_relres', ...
            'global_online_time_s', 'dd_online_time_s', ...
            'dd_saved_vs_global_percent', 'global_ranks', 'dd_block_ranks'}));
        fprintf('Saved %s\n', csv_file);
        fprintf('Saved %s\n', fig_file);
    end
end


function problem = normalize_problem(problem)
    problem = lower(string(problem));
    switch problem
        case {"all", "both"}
            problem = "all";
        case {"halfspace", "half", "helm"}
            problem = "halfspace";
        case {"ecc", "scattering"}
            problem = "ecc";
        case {"pc", "photonic_crystal", "waveguide"}
            problem = "pc";
        otherwise
            error('problem_type must be all, halfspace, ecc, or pc.');
    end
end


function schedules = choose_schedules(problem, user_cfg)
    problem_field = [char(problem) '_schedules'];
    if isfield(user_cfg, problem_field) && ~isempty(user_cfg.(problem_field))
        schedules = user_cfg.(problem_field);
    elseif isfield(user_cfg, 'schedules') && ~isempty(user_cfg.schedules)
        schedules = user_cfg.schedules;
    else
        switch problem
            case "halfspace"
                schedules = [
                    1e-4, 1e-3, 1e-2
                    1e-5, 1e-4, 1e-3
                    1e-4, 1e-4, 1e-4
                    1e-5, 1e-5, 1e-5
                    1e-4, 1e-5, 1e-7
                ];
            case "ecc"
                schedules = [
                    1e-3, 1e-2, 1e-1
                    1e-4, 1e-3, 1e-2
                    1e-4, 1e-4, 1e-4
                    1e-5, 1e-4, 1e-3
                    1e-5, 1e-5, 1e-5
                ];
            case "pc"
                schedules = [
                    1e-4, 1e-2, 1e-1
                    1e-4, 1e-2, 1e-2
                    1e-4, 1e-3, 1e-2
                    1e-4, 1e-4, 1e-4
                    1e-5, 1e-4, 1e-3
                    1e-5, 1e-5, 1e-5
                ];
        end
    end

    if isvector(schedules) && numel(schedules) == 3
        schedules = reshape(schedules, 1, 3);
    end
    if size(schedules, 2) ~= 3 || any(~isfinite(schedules(:))) || any(schedules(:) <= 0)
        error('Each threshold schedule must be a row of three positive finite values.');
    end
end


function cfg = default_probe_cfg(problem, mode)
    cfg = struct();
    cfg.corr_num_layers = 3;
    cfg.do_plots = false;
    cfg.save_plots = false;
    cfg.save_benchmark = false;
    cfg.run_wave_baselines = true;
    cfg.parallel_train = mode == "paper";
    cfg.parallel_test = mode == "paper";

    switch problem
        case "halfspace"
            cfg.material_model = 'halfspace';
            cfg.rom_check_target = 'helm';
            cfg.reference_mode = 'helm_lift';
            cfg.online_completion_mode = 'waveholtz';
            cfg.online_wave_solver = 'fixedpoint';
            cfg.run_nr_rom_comparison = false;

            switch mode
                case "smoke"
                    cfg.Nphys = 45;
                    cfg.train_params = linspace(0.80, 0.95, 6);
                    cfg.test_params = [0.8130, 0.9072];
                    cfg.wh_max_iter = 80;
                    cfg.fp_maxit = 40;
                    cfg.wave_baseline_max_iter = 60;
                case "medium"
                    cfg.Nphys = 75;
                    cfg.train_params = linspace(0.80, 0.95, 10);
                    cfg.test_params = interval_midpoints(cfg.train_params, 5);
                    cfg.wh_max_iter = 160;
                    cfg.fp_maxit = 100;
                    cfg.wave_baseline_max_iter = 140;
                case "paper"
                    cfg.train_params = linspace(0.80, 0.95, 24);
                    cfg.test_param_strategy = 'random_interval';
                    cfg.test_param_rng_seed = 1;
            end

        case "ecc"
            cfg.reference_mode = 'waveholtz';
            cfg.rom_check_target = 'wave';
            cfg.online_completion_mode = 'waveholtz';
            cfg.ratio_test_strategy = 'midpoint';
            cfg.ratio_test_rng_seed = 1;

            switch mode
                case "smoke"
                    cfg.Nphys = 24;
                    cfg.npml = 6;
                    cfg.omega = 10;
                    cfg.ratio_train = [0.94, 1.00, 1.06];
                    cfg.ratio_test = [0.97, 1.03];
                    cfg.wh_max_iter = 40;
                    cfg.helm_baseline_iters = 20;
                case "medium"
                    cfg.Nphys = 45;
                    cfg.npml = 12;
                    cfg.omega = 14;
                    cfg.ratio_train = exp(linspace(log(0.92), log(1.08), 6));
                    cfg.ratio_test = geometric_midpoints(cfg.ratio_train);
                    cfg.wh_max_iter = 140;
                    cfg.helm_baseline_iters = 40;
                case "paper"
                    cfg.ratio_test_strategy = 'random_interval';
            end

        case "pc"
            cfg.reference_mode = 'waveholtz';
            cfg.rom_check_target = 'helm';
            cfg.online_completion_mode = 'waveholtz';
            cfg.online_wave_solver = 'gmres';
            cfg.test_param_strategy = 'midpoint';
            cfg.test_param_rng_seed = 1;

            c_ref = 1 / sqrt(11.4);
            switch mode
                case "smoke"
                    % Small PC grids do not reliably converge the WaveHoltz
                    % reference solve.  Use an exact Helmholtz lift so this
                    % probe isolates ROM truncation rather than reference error.
                    cfg.reference_mode = 'helm_lift';
                    cfg.pc_num_cols = 5;
                    cfg.pc_num_rows = 1;
                    cfg.pc_points_per_a = 8;
                    cfg.npml = 8;
                    cfg.c_pc_train_params = c_ref * linspace(0.98, 1.02, 4);
                    cfg.c_pc_test_params = interval_midpoints(cfg.c_pc_train_params, 3);
                    cfg.wh_gmres_restart = 40;
                    cfg.wh_gmres_maxit = 1;
                    cfg.helm_baseline_iters = 20;
                case "medium"
                    cfg.reference_mode = 'helm_lift';
                    cfg.pc_num_cols = 7;
                    cfg.pc_num_rows = 2;
                    cfg.pc_points_per_a = 10;
                    cfg.npml = 10;
                    cfg.c_pc_train_params = c_ref * linspace(0.98, 1.02, 6);
                    cfg.c_pc_test_params = interval_midpoints(cfg.c_pc_train_params, 5);
                    cfg.wh_gmres_restart = 60;
                    cfg.wh_gmres_maxit = 2;
                    cfg.helm_baseline_iters = 40;
                case "paper"
                    cfg.test_param_strategy = 'random_interval';
            end
    end
end


function overrides = choose_solver_overrides(problem, user_cfg)
    overrides = struct();
    if isfield(user_cfg, 'solver_cfg') && ~isempty(user_cfg.solver_cfg)
        overrides = merge_structs_local(overrides, user_cfg.solver_cfg);
    end
    field_name = [char(problem) '_cfg'];
    if isfield(user_cfg, field_name) && ~isempty(user_cfg.(field_name))
        overrides = merge_structs_local(overrides, user_cfg.(field_name));
    end
end


function tbl = run_problem_sweep(problem, mode, schedules, base_cfg)
    rows = repmat(empty_row(), size(schedules, 1), 1);

    for k = 1:size(schedules, 1)
        schedule = schedules(k, :);
        fprintf('\n------------------------------------------------------------\n');
        fprintf('%s schedule %d/%d: %s\n', upper(problem), k, size(schedules, 1), ...
            format_schedule(schedule));

        global_cfg = configure_global(problem, base_cfg, schedule);
        global_results = run_problem(problem, global_cfg);
        global_summary = summarize_single_run(problem, global_results);
        clear global_results;

        dd_cfg = configure_dd(problem, base_cfg, schedule);
        dd_results = run_problem(problem, dd_cfg);
        dd_summary = summarize_single_run(problem, dd_results);
        clear dd_results;

        rows(k) = combine_summaries(problem, mode, schedule, global_summary, dd_summary);
    end

    tbl = struct2table(rows);
end


function cfg = configure_global(problem, base_cfg, schedule)
    cfg = apply_schedule(base_cfg, schedule);
    switch problem
        case {"halfspace", "pc"}
            cfg.corr_basis_style = 'global';
        case "ecc"
            cfg.direct_basis_style = 'global';
            cfg.corr_basis_style = 'global';
    end
end


function cfg = configure_dd(problem, base_cfg, schedule)
    cfg = apply_schedule(base_cfg, schedule);
    switch problem
        case "halfspace"
            cfg.corr_basis_style = 'block3';
            cfg.block3_partition_mode = 'halfspace_material';
            cfg.corr_block3_tail_tol_near = ones(1, cfg.corr_num_layers);
            cfg.corr_block3_tail_tol_far = ones(1, cfg.corr_num_layers);
            cfg.corr_block3_tail_tol_pml = ones(1, cfg.corr_num_layers);
        case "pc"
            cfg.corr_basis_style = 'block3';
            cfg.block3_partition_mode = 'pc_waveguide';
            cfg.block3_waveguide_halfwidth = 0.75;
            cfg.corr_block3_tail_tol_near = [1, 1, 1];
            cfg.corr_block3_tail_tol_far = [1, 1, 1];
            cfg.corr_block3_tail_tol_pml = [1, 1, 1];
        case "ecc"
            cfg.direct_basis_style = 'block3';
            cfg.corr_basis_style = 'block3';
            cfg.corr_block3_tail_tol_near = [1, 1, 1];
            cfg.corr_block3_tail_tol_far = [1, 1, 1];
            cfg.corr_block3_tail_tol_pml = [1, 1, 1];
    end

    cfg.corr_block3_rank_cap_near = Inf;
    cfg.corr_block3_rank_cap_far = Inf;
    cfg.corr_block3_rank_cap_pml = Inf;
    cfg.corr_block3_global_tail_tols = schedule;
    if ~isfield(cfg, 'corr_block3_global_tail_strategy') || ...
            isempty(cfg.corr_block3_global_tail_strategy)
        cfg.corr_block3_global_tail_strategy = 'dof_weighted';
    end
end


function cfg = apply_schedule(cfg, schedule)
    cfg.corr_rank_rtols = schedule;
    cfg.corr_pod_singular_value_tols = schedule;
end


function r = run_problem(problem, cfg)
    switch problem
        case "halfspace"
            r = helm_waveholtz_correction_rom(cfg);
        case "ecc"
            r = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
        case "pc"
            r = photonic_crystal_waveguide_correction_rom(cfg);
    end
end


function out = summarize_single_run(problem, r)
    out = struct();
    out.ranks = row_vector(safe_field(r.summary, 'corr_ranks', []));
    out.block_ranks = safe_field(r.summary, 'corr_block_ranks', []);
    out.space_dims = row_vector(safe_field(r.summary, 'corr_space_dims', out.ranks));
    out.basis_entries = correction_basis_entries(r);
    out.internal_dense_entries = sum(safe_field(r.summary, 'corr_storage_dense', NaN), 'omitnan');
    out.precorr_relres = stat_mean(r.summary, 'online_precorr_relres');
    out.online_time = stat_mean(r.summary, 'online_total_time');
    out.precorr_time = stat_mean(r.summary, 'online_precorr_time');
    out.completion_iter = stat_mean(r.summary, 'online_fp_iter');
    out.plain_time = stat_mean(r.summary, 'plain_fp_time');
    out.speedup = safe_scalar(r.summary, 'time_speedup_online_vs_plain_mean');
    out.final_error = primary_final_error(problem, r.summary);
    out.achieved_tails = achieved_layer_tails(r.correction_rom);
    [out.layer_relres, out.layer_contraction] = mean_layer_residual_history(r.test, r.cfg.corr_num_layers);
end


function entries = correction_basis_entries(r)
    block_entries = safe_field(r.summary, 'corr_storage_block', []);
    if ~isempty(block_entries) && any(isfinite(block_entries(:)))
        entries = sum(block_entries(:), 'omitnan');
        return;
    end

    entries = 0;
    for j = 1:numel(r.correction_rom.layers)
        layer = r.correction_rom.layers{j};
        if isfield(layer, 'basis_u') && ~isempty(layer.basis_u)
            entries = entries + numel(layer.basis_u);
        elseif isfield(layer, 'pod') && isfield(layer.pod, 'basis')
            entries = entries + numel(layer.pod.basis);
        end
    end
end


function tails = achieved_layer_tails(corr_rom)
    tails = nan(1, corr_rom.num_layers);
    for j = 1:corr_rom.num_layers
        layer = corr_rom.layers{j};
        space = layer.basis_space;
        if isfield(space, 'blocks') && ~isempty(space.blocks) && ...
                isfield(space.blocks(1), 'meta') && isfield(space.blocks(1).meta, 'global_relative_tail')
            tails(j) = space.blocks(1).meta.global_relative_tail;
        elseif isfield(space, 'block_meta')
            names = fieldnames(space.block_meta);
            if ~isempty(names) && isfield(space.block_meta.(names{1}), 'global_relative_tail')
                tails(j) = space.block_meta.(names{1}).global_relative_tail;
            end
        elseif isfield(layer, 'pod') && isfield(layer.pod, 'singular_values')
            s = abs(layer.pod.singular_values(:));
            rank_used = layer.pod.rank_used;
            if rank_used >= numel(s)
                tails(j) = 0;
            elseif ~isempty(s)
                tails(j) = norm(s(rank_used + 1:end)) / max(norm(s), 1e-30);
            end
        end
    end
end


function [mean_relres, mean_contraction] = mean_layer_residual_history(test, nlayers)
    histories = safe_field(test, 'online_precorr_relres_hist', []);
    if isempty(histories)
        histories = safe_field(test, 'online_combined_relres_hist', {});
    end
    if isempty(histories)
        mean_relres = nan(1, nlayers);
        mean_contraction = nan(1, nlayers);
        return;
    end

    ncases = numel(histories);
    relres = nan(ncases, nlayers + 1);
    for k = 1:ncases
        h = real(histories{k}(:));
        ncopy = min(numel(h), nlayers + 1);
        if ncopy > 0
            relres(k, 1:ncopy) = h(1:ncopy);
        end
    end
    mean_relres = mean(relres(:, 2:end), 1, 'omitnan');
    ratios = relres(:, 2:end) ./ max(relres(:, 1:end-1), realmin);
    ratios(~isfinite(ratios) | ratios <= 0) = NaN;
    mean_contraction = exp(mean(log(ratios), 1, 'omitnan'));
end


function row = combine_summaries(problem, mode, schedule, g, d)
    row = empty_row();
    row.problem = string(problem);
    row.mode = string(mode);
    row.schedule = string(format_schedule(schedule));
    row.eta1 = schedule(1);
    row.eta2 = schedule(2);
    row.eta3 = schedule(3);
    row.eta_product = prod(schedule);
    row.eta_geometric_mean = prod(schedule)^(1 / 3);
    row.global_ranks = string(mat2str(g.ranks));
    row.dd_block_ranks = string(mat2str(d.block_ranks));
    row.dd_space_dims = string(mat2str(d.space_dims));
    row.global_achieved_tails = string(mat2str(g.achieved_tails, 4));
    row.dd_achieved_tails = string(mat2str(d.achieved_tails, 4));
    row.global_basis_entries = g.basis_entries;
    row.dd_basis_entries = d.basis_entries;
    row.dd_saved_vs_global_percent = 100 * (1 - d.basis_entries / max(g.basis_entries, 1));
    row.dd_saved_vs_internal_dense_percent = 100 * (1 - d.basis_entries / max(d.internal_dense_entries, 1));
    row.global_precorr_relres = g.precorr_relres;
    row.dd_precorr_relres = d.precorr_relres;
    row.global_online_time_s = g.online_time;
    row.dd_online_time_s = d.online_time;
    row.dd_over_global_time = d.online_time / max(g.online_time, realmin);
    row.global_precorr_time_s = g.precorr_time;
    row.dd_precorr_time_s = d.precorr_time;
    row.global_completion_iter = g.completion_iter;
    row.dd_completion_iter = d.completion_iter;
    row.global_plain_time_s = g.plain_time;
    row.dd_plain_time_s = d.plain_time;
    row.global_speedup = g.speedup;
    row.dd_speedup = d.speedup;
    row.global_final_error = g.final_error;
    row.dd_final_error = d.final_error;

    for j = 1:3
        row.(sprintf('global_layer%d_relres', j)) = vector_entry(g.layer_relres, j);
        row.(sprintf('dd_layer%d_relres', j)) = vector_entry(d.layer_relres, j);
        row.(sprintf('global_layer%d_contraction', j)) = vector_entry(g.layer_contraction, j);
        row.(sprintf('dd_layer%d_contraction', j)) = vector_entry(d.layer_contraction, j);
    end
end


function row = empty_row()
    row = struct( ...
        'problem', "", 'mode', "", 'schedule', "", ...
        'eta1', NaN, 'eta2', NaN, 'eta3', NaN, ...
        'eta_product', NaN, 'eta_geometric_mean', NaN, ...
        'global_ranks', "", 'dd_block_ranks', "", 'dd_space_dims', "", ...
        'global_achieved_tails', "", 'dd_achieved_tails', "", ...
        'global_basis_entries', NaN, 'dd_basis_entries', NaN, ...
        'dd_saved_vs_global_percent', NaN, 'dd_saved_vs_internal_dense_percent', NaN, ...
        'global_precorr_relres', NaN, 'dd_precorr_relres', NaN, ...
        'global_online_time_s', NaN, 'dd_online_time_s', NaN, 'dd_over_global_time', NaN, ...
        'global_precorr_time_s', NaN, 'dd_precorr_time_s', NaN, ...
        'global_completion_iter', NaN, 'dd_completion_iter', NaN, ...
        'global_plain_time_s', NaN, 'dd_plain_time_s', NaN, ...
        'global_speedup', NaN, 'dd_speedup', NaN, ...
        'global_final_error', NaN, 'dd_final_error', NaN, ...
        'global_layer1_relres', NaN, 'dd_layer1_relres', NaN, ...
        'global_layer2_relres', NaN, 'dd_layer2_relres', NaN, ...
        'global_layer3_relres', NaN, 'dd_layer3_relres', NaN, ...
        'global_layer1_contraction', NaN, 'dd_layer1_contraction', NaN, ...
        'global_layer2_contraction', NaN, 'dd_layer2_contraction', NaN, ...
        'global_layer3_contraction', NaN, 'dd_layer3_contraction', NaN);
end


function plot_balance_summary(problem, mode, tbl, out_file)
    fig = figure('Visible', 'off', 'Name', 'ROM threshold balance', ...
        'Position', [120, 120, 980, 720]);
    layout = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = 1:height(tbl);

    ax1 = nexttile(layout, 1);
    semilogy(ax1, x, tbl.global_precorr_relres, '-o', ...
        'Color', [0.82, 0.12, 0.12], 'LineWidth', 1.8, 'MarkerSize', 7, ...
        'DisplayName', 'Global ROM residual');
    hold(ax1, 'on');
    semilogy(ax1, x, tbl.dd_precorr_relres, '-s', ...
        'Color', [0.08, 0.48, 0.27], 'LineWidth', 1.8, 'MarkerSize', 7, ...
        'DisplayName', 'DD-ROM residual');
    hold(ax1, 'off');
    ylabel(ax1, 'mean pre-completion residual');
    title(ax1, sprintf('%s ROM-threshold balance (%s)', upper(problem), mode));
    grid(ax1, 'on');
    box(ax1, 'on');
    legend(ax1, 'Location', 'best');

    ax2 = nexttile(layout, 2);
    yyaxis(ax2, 'left');
    plot(ax2, x, tbl.global_online_time_s, '-o', ...
        'Color', [0.82, 0.12, 0.12], 'LineWidth', 1.8, 'MarkerSize', 7, ...
        'DisplayName', 'Global online time');
    hold(ax2, 'on');
    plot(ax2, x, tbl.dd_online_time_s, '-s', ...
        'Color', [0.08, 0.48, 0.27], 'LineWidth', 1.8, 'MarkerSize', 7, ...
        'DisplayName', 'DD online time');
    ylabel(ax2, 'mean online time (s)');
    yyaxis(ax2, 'right');
    plot(ax2, x, tbl.dd_saved_vs_global_percent, '-d', ...
        'Color', [0.15, 0.35, 0.75], 'LineWidth', 1.8, 'MarkerSize', 7, ...
        'DisplayName', 'DD storage saved');
    yline(ax2, 20, ':', '20% guide', 'Color', [0.35, 0.35, 0.35], ...
        'HandleVisibility', 'off');
    ylabel(ax2, 'DD basis entries saved vs Global (%)');
    hold(ax2, 'off');
    grid(ax2, 'on');
    box(ax2, 'on');
    legend(ax2, 'Location', 'best');

    labels = cellstr(tbl.schedule);
    set([ax1, ax2], 'XTick', x, 'XTickLabel', labels, 'FontSize', 11, 'LineWidth', 1.0);
    xtickangle(ax1, 18);
    xtickangle(ax2, 18);
    xlabel(ax2, 'layerwise ROM tail schedule');

    exportgraphics(fig, out_file, 'Resolution', 220);
    close(fig);
end


function value = primary_final_error(problem, summary)
    if any(problem == ["halfspace", "pc"])
        value = stat_mean(summary, 'final_helm_full');
    else
        value = stat_mean(summary, 'final_wave_full');
    end
end


function value = stat_mean(s, field_name)
    value = NaN;
    if ~isfield(s, field_name) || isempty(s.(field_name))
        return;
    end
    data = s.(field_name);
    if numel(data) >= 2
        value = data(2);
    else
        value = mean(data(:), 'omitnan');
    end
end


function value = safe_scalar(s, field_name)
    value = NaN;
    if isfield(s, field_name) && ~isempty(s.(field_name))
        data = s.(field_name);
        value = data(1);
    end
end


function value = safe_field(s, field_name, fallback)
    value = fallback;
    if isfield(s, field_name) && ~isempty(s.(field_name))
        value = s.(field_name);
    end
end


function value = vector_entry(v, index)
    value = NaN;
    if numel(v) >= index
        value = v(index);
    end
end


function v = row_vector(v)
    v = v(:).';
end


function label = format_schedule(schedule)
    label = sprintf('(%s,%s,%s)', format_power(schedule(1)), ...
        format_power(schedule(2)), format_power(schedule(3)));
end


function label = format_power(value)
    exponent = round(log10(value));
    if abs(value - 10^exponent) <= 1e-12 * max(value, realmin)
        label = sprintf('1e%d', exponent);
    else
        label = sprintf('%.1e', value);
    end
end


function points = interval_midpoints(train_values, max_count)
    train_values = train_values(:).';
    mids = 0.5 * (train_values(1:end-1) + train_values(2:end));
    points = subsample_points(mids, max_count);
end


function points = geometric_midpoints(train_values)
    train_values = train_values(:).';
    points = sqrt(train_values(1:end-1) .* train_values(2:end));
end


function points = subsample_points(points, max_count)
    if numel(points) <= max_count
        return;
    end
    idx = unique(round(linspace(1, numel(points), max_count)));
    points = points(idx);
end


function out = merge_structs_local(out, extra)
    if isempty(extra)
        return;
    end
    names = fieldnames(extra);
    for k = 1:numel(names)
        out.(names{k}) = extra.(names{k});
    end
end


function tag = normalized_output_tag(user_cfg)
    tag = '';
    if ~isfield(user_cfg, 'output_tag') || isempty(user_cfg.output_tag)
        return;
    end
    tag = regexprep(char(string(user_cfg.output_tag)), '[^A-Za-z0-9_\-]+', '_');
    tag = regexprep(tag, '_+', '_');
    tag = regexprep(tag, '^_|_$', '');
end


function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end
