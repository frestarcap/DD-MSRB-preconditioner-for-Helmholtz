function results = run_pc_dd_storage_exploration(mode, reference_mode, user_cfg)
%RUN_PC_DD_STORAGE_EXPLORATION Compare PC block partitions and tail allocation.
%
% Research-only entry point. Outputs are isolated under
% outputs/pc_dd_storage_exploration and never overwrite paper evidence.
%
% Examples:
%   run_pc_dd_storage_exploration("medium", "helm_lift");
%   run_pc_dd_storage_exploration("paper", "waveholtz", struct( ...
%       'variant_names', "narrow4-dof-weighted", 'save_raw', true));
%   run_pc_dd_storage_exploration("medium", "helm_lift", struct( ...
%       'include_discrete_diagnostics', true));

    if nargin < 1 || isempty(mode)
        mode = "medium";
    end
    if nargin < 2 || isempty(reference_mode)
        reference_mode = "helm_lift";
    end
    if nargin < 3 || isempty(user_cfg)
        user_cfg = struct();
    end

    mode = lower(string(mode));
    reference_mode = normalize_reference_mode(reference_mode);
    if ~any(mode == ["smoke", "medium", "paper"])
        error('mode must be smoke, medium, or paper.');
    end

    base_cfg = default_cfg(mode, reference_mode);
    if isfield(user_cfg, 'solver_cfg') && ~isempty(user_cfg.solver_cfg)
        base_cfg = merge_structs_local(base_cfg, user_cfg.solver_cfg);
    end
    include_discrete_diagnostics = isfield(user_cfg, ...
        'include_discrete_diagnostics') && ...
        logical(user_cfg.include_discrete_diagnostics);
    variants = candidate_variants(include_discrete_diagnostics);
    if isfield(user_cfg, 'variant_names') && ~isempty(user_cfg.variant_names)
        variants = select_variants(variants, string(user_cfg.variant_names));
    end
    save_raw = isfield(user_cfg, 'save_raw') && logical(user_cfg.save_raw);

    out_dir = helm_output_path('outputs', 'pc_dd_storage_exploration');
    fig_dir = helm_output_path('outputs', 'figures', 'pc_dd_storage_exploration');
    ensure_dir(out_dir);
    ensure_dir(fig_dir);
    stem = sprintf('pc_dd_storage_%s_%s', mode, reference_mode);

    rows = repmat(empty_row(), numel(variants), 1);
    compact = cell(numel(variants), 1);
    raw_files = strings(numel(variants), 1);
    global_entries = NaN;
    for k = 1:numel(variants)
        variant = variants(k);
        cfg = configure_variant(base_cfg, variant);
        fprintf('\n============================================================\n');
        fprintf('PC DD storage exploration: %s (%s, %s)\n', ...
            variant.name, mode, reference_mode);
        run_result = photonic_crystal_waveguide_correction_rom(cfg);
        compact{k} = compact_result(run_result);
        rows(k) = summarize_variant(variant, run_result);
        if variant.name == "global"
            global_entries = rows(k).basis_entries;
        end

        if save_raw
            raw_file = fullfile(out_dir, sprintf('%s_%s_full.mat', stem, variant.file_tag));
            save(raw_file, 'run_result', 'cfg', '-v7.3');
            raw_files(k) = string(raw_file);
            fprintf('Saved full result: %s\n', raw_file);
        end
        clear run_result;
    end

    if ~isfinite(global_entries)
        global_entries = reference_global_entries(mode, reference_mode);
    end
    for k = 1:numel(rows)
        if isfinite(global_entries)
            rows(k).saved_vs_global_percent = 100 * ...
                (1 - rows(k).basis_entries / max(global_entries, 1));
        else
            rows(k).saved_vs_global_percent = NaN;
        end
    end
    tbl = struct2table(rows);

    csv_file = fullfile(out_dir, [stem '.csv']);
    mat_file = fullfile(out_dir, [stem '.mat']);
    fig_file = fullfile(fig_dir, [stem '.png']);
    writetable(tbl, csv_file);
    save(mat_file, 'mode', 'reference_mode', 'base_cfg', 'variants', ...
        'tbl', 'compact', 'raw_files');
    plot_summary(tbl, mode, reference_mode, fig_file);

    results = struct();
    results.mode = mode;
    results.reference_mode = reference_mode;
    results.table = tbl;
    results.compact = compact;
    results.csv_file = csv_file;
    results.mat_file = mat_file;
    results.figure_file = fig_file;
    results.raw_files = raw_files;

    fprintf('\nPC DD storage summary:\n');
    disp(tbl(:, {'variant', 'basis_entries', 'saved_vs_global_percent', ...
        'mean_precorr_relres', 'mean_online_time_s', 'mean_completion_iter', ...
        'block_ranks'}));
    fprintf('Saved compact table: %s\n', csv_file);
    fprintf('Saved figure: %s\n', fig_file);
end


function reference_mode = normalize_reference_mode(reference_mode)
    reference_mode = lower(string(reference_mode));
    switch reference_mode
        case {"helm_lift", "helmholtz", "exact"}
            reference_mode = "helm_lift";
        case {"waveholtz", "wave", "wh"}
            reference_mode = "waveholtz";
        otherwise
            error('reference_mode must be helm_lift or waveholtz.');
    end
end


function cfg = default_cfg(mode, reference_mode)
    cfg = struct();
    cfg.reference_mode = char(reference_mode);
    cfg.rom_check_target = 'helm';
    cfg.online_completion_mode = 'waveholtz';
    cfg.online_wave_solver = 'gmres';
    cfg.run_wave_baselines = false;
    cfg.corr_num_layers = 3;
    cfg.corr_pod_singular_value_tols = [1e-4, 1e-3, 1e-2];
    cfg.corr_rank_rtols = cfg.corr_pod_singular_value_tols;
    cfg.do_plots = false;
    cfg.save_plots = false;
    cfg.save_benchmark = false;
    cfg.parallel_train = mode == "paper";
    cfg.parallel_test = mode == "paper";
    cfg.test_param_strategy = 'random_interval';
    cfg.test_param_rng_seed = 1;

    c_ref = 1 / sqrt(11.4);
    switch mode
        case "smoke"
            cfg.pc_num_cols = 5;
            cfg.pc_num_rows = 1;
            cfg.pc_points_per_a = 8;
            cfg.npml = 8;
            cfg.c_pc_train_params = c_ref * linspace(0.98, 1.02, 4);
            cfg.c_pc_test_params = interval_midpoints(cfg.c_pc_train_params);
            cfg.wh_gmres_restart = 40;
            cfg.wh_gmres_maxit = 1;
            cfg.helm_baseline_iters = 20;
        case "medium"
            cfg.pc_num_cols = 7;
            cfg.pc_num_rows = 2;
            cfg.pc_points_per_a = 10;
            cfg.npml = 10;
            cfg.c_pc_train_params = c_ref * linspace(0.98, 1.02, 6);
            cfg.c_pc_test_params = interval_midpoints(cfg.c_pc_train_params);
            cfg.wh_gmres_restart = 60;
            cfg.wh_gmres_maxit = 2;
            cfg.helm_baseline_iters = 40;
    end
end


function variants = candidate_variants(include_discrete_diagnostics)
    variants = [ ...
        make_variant("global", "global", "", "equal_absolute")
        make_variant("narrow3-equal", "block3", "pc_waveguide", "equal_absolute")
        make_variant("narrow3-dof-weighted", "block3", "pc_waveguide", "dof_weighted")
        make_variant("narrow3-greedy", "block3", "pc_waveguide", "marginal_storage")
        make_variant("narrow4-equal", "block3", "pc_core_cladding_lr_tb_pml", "equal_absolute")
        make_variant("narrow4-dof-weighted", "block3", "pc_core_cladding_lr_tb_pml", "dof_weighted")
        make_variant("narrow4-greedy", "block3", "pc_core_cladding_lr_tb_pml", "marginal_storage")
        make_variant("channel-lr-greedy", "block3", "pc_channel_lr_pml", "marginal_storage")
    ];
    if include_discrete_diagnostics
        variants = [variants, ...
            make_variant("narrow3-min-storage", "block3", ...
                "pc_waveguide", "minimum_storage"), ...
            make_variant("narrow4-min-storage", "block3", ...
                "pc_core_cladding_lr_tb_pml", "minimum_storage")];
    end
end


function variant = make_variant(name, basis_style, partition_mode, strategy)
    variant = struct();
    variant.name = string(name);
    variant.file_tag = strrep(string(name), '-', '_');
    variant.basis_style = char(basis_style);
    variant.partition_mode = char(partition_mode);
    variant.strategy = char(strategy);
end


function variants = select_variants(variants, requested)
    requested = requested(:);
    names = [variants.name].';
    missing = requested(~ismember(requested, names));
    if ~isempty(missing)
        error('Unknown variant(s): %s.', strjoin(cellstr(missing), ', '));
    end
    keep = ismember(names, requested);
    variants = variants(keep);
end


function cfg = configure_variant(base_cfg, variant)
    cfg = base_cfg;
    cfg.corr_basis_style = variant.basis_style;
    if strcmp(variant.basis_style, 'global')
        return;
    end

    cfg.block3_partition_mode = variant.partition_mode;
    cfg.block3_waveguide_halfwidth = 0.75;
    cfg.corr_block3_global_tail_tols = cfg.corr_pod_singular_value_tols;
    cfg.corr_block3_global_tail_strategy = variant.strategy;
    cfg.corr_block3_rank_cap_near = Inf;
    cfg.corr_block3_rank_cap_far = Inf;
    cfg.corr_block3_rank_cap_pml = Inf;
    cfg.corr_block3_tail_tol_near = [1, 1, 1];
    cfg.corr_block3_tail_tol_far = [1, 1, 1];
    cfg.corr_block3_tail_tol_pml = [1, 1, 1];

    if strcmp(variant.partition_mode, 'pc_core_cladding_lr_tb_pml')
        % A global budget must be free to move between all four blocks.
        % Empty values would fall back to the generic per-block layer tol.
        cfg.corr_blockn_tail_tols = ones(3, 4);
        cfg.corr_blockn_rank_caps = Inf(3, 4);
    end
end


function out = compact_result(r)
    out = struct();
    out.cfg = r.cfg;
    out.problem_size = [r.problem.nx, r.problem.ny, r.problem.Ntot];
    out.summary = r.summary;
    out.block_diagnostics = block_diagnostics(r.correction_rom);
end


function row = summarize_variant(variant, r)
    row = empty_row();
    row.variant = variant.name;
    row.partition = string(variant.partition_mode);
    row.strategy = string(variant.strategy);
    row.basis_entries = correction_basis_entries(r);
    row.internal_dense_entries = sum(r.summary.corr_storage_dense, 'omitnan');
    row.internal_saved_percent = 100 * ...
        (1 - row.basis_entries / max(row.internal_dense_entries, 1));
    row.mean_precorr_relres = summary_mean(r.summary, 'online_precorr_relres');
    row.max_precorr_relres = summary_max(r.summary, 'online_precorr_relres');
    row.mean_online_time_s = summary_mean(r.summary, 'online_total_time');
    row.mean_completion_iter = summary_mean(r.summary, 'online_fp_iter');
    row.mean_final_relres = summary_mean(r.summary, 'online_fp_relres');
    row.mean_final_error = primary_error_mean(r.summary);
    row.global_ranks = string(mat2str(r.summary.corr_ranks));
    row.block_ranks = string(mat2str(r.summary.corr_block_ranks));
    row.block_dofs = string(mat2str(r.summary.corr_block_dofs));
    diag = block_diagnostics(r.correction_rom);
    row.energy_weights = string(mat2str(diag.energy_weights, 4));
    row.local_tails = string(mat2str(diag.local_tails, 4));
    row.effective_tols = string(mat2str(diag.effective_tols, 4));
    row.achieved_global_tails = string(mat2str(diag.global_tails, 4));
end


function diag = block_diagnostics(corr_rom)
    diag = struct('energy_weights', [], 'local_tails', [], ...
        'effective_tols', [], 'global_tails', []);
    if corr_rom.num_layers == 0
        return;
    end
    max_blocks = 0;
    for j = 1:corr_rom.num_layers
        max_blocks = max(max_blocks, count_blocks(corr_rom.layers{j}.basis_space));
    end
    diag.energy_weights = nan(corr_rom.num_layers, max_blocks);
    diag.local_tails = nan(corr_rom.num_layers, max_blocks);
    diag.effective_tols = nan(corr_rom.num_layers, max_blocks);
    diag.global_tails = nan(1, corr_rom.num_layers);
    for j = 1:corr_rom.num_layers
        metas = space_metas(corr_rom.layers{j}.basis_space);
        for b = 1:numel(metas)
            meta = metas{b};
            diag.energy_weights(j, b) = optional_field(meta, 'global_energy_weight');
            diag.local_tails(j, b) = optional_field(meta, 'local_relative_tail');
            diag.effective_tols(j, b) = optional_field(meta, 'generated_rank_tail_tol');
            if b == 1
                diag.global_tails(j) = optional_field(meta, 'global_relative_tail');
            end
        end
    end
end


function count = count_blocks(space)
    if isfield(space, 'blocks') && ~isempty(space.blocks)
        count = numel(space.blocks);
    elseif isfield(space, 'block_meta')
        count = numel(fieldnames(space.block_meta));
    else
        count = 0;
    end
end


function metas = space_metas(space)
    metas = {};
    if isfield(space, 'blocks') && ~isempty(space.blocks)
        metas = arrayfun(@(block) block.meta, space.blocks, 'UniformOutput', false);
    elseif isfield(space, 'block_meta')
        names = fieldnames(space.block_meta);
        metas = cellfun(@(name) space.block_meta.(name), names, 'UniformOutput', false);
    end
end


function entries = correction_basis_entries(r)
    if isfield(r.summary, 'corr_storage_block') && ...
            any(isfinite(r.summary.corr_storage_block(:)))
        entries = sum(r.summary.corr_storage_block, 'omitnan');
    else
        entries = r.problem.Ntot * sum(r.summary.corr_ranks, 'omitnan');
    end
end


function value = summary_mean(summary, field_name)
    value = NaN;
    if isfield(summary, field_name) && numel(summary.(field_name)) >= 2
        value = summary.(field_name)(2);
    end
end


function value = summary_max(summary, field_name)
    value = NaN;
    if isfield(summary, field_name) && numel(summary.(field_name)) >= 3
        value = summary.(field_name)(3);
    end
end


function value = primary_error_mean(summary)
    value = summary_mean(summary, 'online_final_helm_full_error');
    if ~isfinite(value)
        value = summary_mean(summary, 'online_final_full_error');
    end
end


function value = optional_field(s, field_name)
    value = NaN;
    if isfield(s, field_name) && ~isempty(s.(field_name))
        value = s.(field_name);
    end
end


function entries = reference_global_entries(mode, reference_mode)
    entries = NaN;
    if mode ~= "paper"
        return;
    end
    if reference_mode == "waveholtz"
        entries = 1201200;
    elseif reference_mode == "helm_lift"
        entries = 1029600;
    end
end


function row = empty_row()
    row = struct( ...
        'variant', "", 'partition', "", 'strategy', "", ...
        'basis_entries', NaN, 'saved_vs_global_percent', NaN, ...
        'internal_dense_entries', NaN, 'internal_saved_percent', NaN, ...
        'mean_precorr_relres', NaN, 'max_precorr_relres', NaN, ...
        'mean_online_time_s', NaN, 'mean_completion_iter', NaN, ...
        'mean_final_relres', NaN, 'mean_final_error', NaN, ...
        'global_ranks', "", 'block_ranks', "", 'block_dofs', "", ...
        'energy_weights', "", 'local_tails', "", 'effective_tols', "", ...
        'achieved_global_tails', "");
end


function plot_summary(tbl, mode, reference_mode, out_file)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 720]);
    layout = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    x = 1:height(tbl);

    ax1 = nexttile(layout, 1);
    yyaxis(ax1, 'left');
    bar(ax1, x, tbl.saved_vs_global_percent, 0.65, ...
        'FaceColor', [0.18, 0.48, 0.72], 'DisplayName', 'storage saved');
    ylabel(ax1, 'basis entries saved vs Global (%)');
    yline(ax1, 20, ':', '20% guide', 'LineWidth', 1.5, ...
        'HandleVisibility', 'off');
    yyaxis(ax1, 'right');
    semilogy(ax1, x, tbl.mean_precorr_relres, '-o', ...
        'Color', [0.82, 0.12, 0.12], 'LineWidth', 1.8, ...
        'MarkerSize', 7, 'DisplayName', 'pre-completion residual');
    ylabel(ax1, 'mean pre-completion residual');
    grid(ax1, 'on');
    title(ax1, sprintf('PC DD storage exploration (%s, %s)', mode, reference_mode));

    ax2 = nexttile(layout, 2);
    yyaxis(ax2, 'left');
    plot(ax2, x, tbl.mean_online_time_s, '-s', ...
        'Color', [0.08, 0.48, 0.27], 'LineWidth', 1.8, ...
        'MarkerSize', 7, 'DisplayName', 'online time');
    ylabel(ax2, 'mean online time (s)');
    yyaxis(ax2, 'right');
    plot(ax2, x, tbl.mean_completion_iter, '-d', ...
        'Color', [0.56, 0.25, 0.65], 'LineWidth', 1.8, ...
        'MarkerSize', 7, 'DisplayName', 'completion iterations');
    ylabel(ax2, 'mean completion iterations');
    grid(ax2, 'on');
    xticks(ax2, x);
    xticklabels(ax2, cellstr(tbl.variant));
    xtickangle(ax2, 18);

    set(findall(fig, '-property', 'FontSize'), 'FontSize', 14);
    set(findall(fig, '-property', 'LineWidth'), 'LineWidth', 1.2);
    exportgraphics(fig, out_file, 'Resolution', 220);
    close(fig);
end


function points = interval_midpoints(train_values)
    train_values = train_values(:).';
    points = 0.5 * (train_values(1:end-1) + train_values(2:end));
end


function out = merge_structs_local(base, override)
    out = base;
    names = fieldnames(override);
    for k = 1:numel(names)
        out.(names{k}) = override.(names{k});
    end
end


function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end
