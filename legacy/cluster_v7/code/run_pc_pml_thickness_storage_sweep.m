function results = run_pc_pml_thickness_storage_sweep(mode, user_cfg)
%RUN_PC_PML_THICKNESS_STORAGE_SWEEP
% Compare PC waveguide DD-ROM storage savings across PML thicknesses.
%
% Example:
%   results = run_pc_pml_thickness_storage_sweep("paper");
%   results = run_pc_pml_thickness_storage_sweep("quick", struct('npml_values', [25 40]));

    if nargin < 1 || isempty(mode)
        mode = "paper";
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end

    mode = lower(string(mode));
    cfg0 = default_sweep_cfg(mode);
    cfg0 = merge_structs(cfg0, user_cfg);
    npml_values = cfg0.npml_values(:).';

    out_dir = helm_output_path('outputs', 'pc_pml_thickness_storage_sweep');
    fig_dir = fullfile(out_dir, 'figures');
    ensure_dir(out_dir);
    ensure_dir(fig_dir);

    fprintf('=== PC PML thickness storage sweep: %s ===\n', mode);
    fprintf('npml values: %s\n', mat2str(npml_values));
    fprintf('reference_mode=%s, online_completion_mode=%s\n', ...
        cfg0.reference_mode, cfg0.online_completion_mode);

    rows = repmat(empty_summary_row(), numel(npml_values), 1);
    run_summaries = cell(numel(npml_values), 1);
    full_results = cell(numel(npml_values), 1);

    for k = 1:numel(npml_values)
        npml = npml_values(k);
        cfg = rmfield_if_present(cfg0, {'npml_values', 'keep_full_results'});
        cfg.npml = npml;
        cfg.benchmark_name = sprintf('pc_pml_thickness_storage_sweep_npml_%03d', npml);
        cfg.figure_subdir = sprintf('pc_pml_thickness_storage_sweep/npml_%03d', npml);

        fprintf('\n--- PML thickness %d/%d: npml=%d ---\n', k, numel(npml_values), npml);
        r = photonic_crystal_waveguide_correction_rom(cfg);
        rows(k) = summarize_run(npml, r);
        run_summaries{k} = r.summary;
        if cfg0.keep_full_results
            full_results{k} = r;
        end
    end

    tbl = struct2table(rows);
    csv_file = fullfile(out_dir, ['pc_pml_thickness_storage_sweep_' char(mode) '.csv']);
    mat_file = fullfile(out_dir, ['pc_pml_thickness_storage_sweep_' char(mode) '.mat']);
    writetable(tbl, csv_file);
    plot_file = plot_storage_sweep(tbl, fig_dir, mode);

    results = struct();
    results.mode = char(mode);
    results.cfg = cfg0;
    results.table = tbl;
    results.run_summaries = run_summaries;
    results.full_results = full_results;
    results.csv_file = csv_file;
    results.mat_file = mat_file;
    results.plot_file = plot_file;
    save(mat_file, 'results', 'tbl', 'run_summaries', 'cfg0', '-v7.3');

    fprintf('\nSaved table: %s\n', csv_file);
    fprintf('Saved plot : %s\n', plot_file);
    fprintf('Saved MAT  : %s\n', mat_file);
end

function cfg = default_sweep_cfg(mode)
    cfg = struct();
    cfg.npml_values = [25 40 55 70];
    cfg.keep_full_results = false;

    cfg.rom_check_target = 'helm';
    cfg.reference_mode = 'helm_lift';
    cfg.offline_reference_solver = 'helmholtz';
    cfg.online_completion_mode = 'waveholtz';
    cfg.run_wave_baselines = true;
    cfg.use_direct_rom_init = false;
    cfg.corr_basis_style = 'block3';
    cfg.corr_num_layers = 3;
    cfg.corr_pod_singular_value_tols = [1e-4, 1e-3, 1e-2];
    cfg.corr_block3_rank_cap_near = Inf;
    cfg.corr_block3_rank_cap_far = Inf;
    cfg.corr_block3_rank_cap_pml = Inf;
    cfg.corr_block3_tail_tol_near = [1, 1, 1];
    cfg.corr_block3_tail_tol_far = [1, 1, 1];
    cfg.corr_block3_tail_tol_pml = [1, 1, 1];
    cfg.corr_block3_global_tail_tols = cfg.corr_pod_singular_value_tols;
    cfg.corr_block3_global_tail_strategy = 'dof_weighted';
    cfg.shifted_laplacian_beta = 0.2;
    cfg.shifted_laplacian_restart = 30;
    cfg.shifted_laplacian_maxit = 3;
    cfg.shifted_laplacian_tol = 1e-15;
    cfg.shifted_laplacian_reorth = true;
    cfg.helm_baseline_iters = 20;
    cfg.helm_baseline_tol = 0;
    cfg.do_plots = false;
    cfg.save_plots = false;
    cfg.save_benchmark = false;
    cfg.parallel_train = true;
    cfg.parallel_test = true;
    cfg.test_param_strategy = 'midpoint';
    cfg.test_param_rng_seed = [];

    switch mode
        case "smoke"
            cfg.pc_num_cols = 5;
            cfg.pc_num_rows = 1;
            cfg.pc_points_per_a = 8;
            cfg.npml_values = [6 8 10 12];
            cfg.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.98, 1.02, 4);
            cfg.c_pc_test_params = (1 / sqrt(11.4)) * [0.99 1.01];
            cfg.shifted_laplacian_tol = 1e-9;
            cfg.shifted_laplacian_maxit = 4;
            cfg.helm_baseline_iters = 10;
            cfg.parallel_train = false;
            cfg.parallel_test = false;
        case {"quick", "paper"}
            cfg.pc_num_cols = 9;
            cfg.pc_num_rows = 3;
            cfg.pc_points_per_a = 20;
            cfg.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.96, 1.04, 8);
            cfg.c_pc_test_params = (1 / sqrt(11.4)) * [0.97 1.00 1.03];
            if mode == "quick"
                cfg.npml_values = [25 40];
                cfg.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.97, 1.03, 5);
                cfg.c_pc_test_params = (1 / sqrt(11.4)) * [0.985 1.015];
                cfg.shifted_laplacian_tol = 1e-10;
                cfg.shifted_laplacian_maxit = 2;
                cfg.helm_baseline_iters = 10;
                cfg.parallel_train = false;
                cfg.parallel_test = false;
            end
        otherwise
            error('mode=%s must be smoke, quick, or paper.', mode);
    end
end

function row = empty_summary_row()
    row = struct( ...
        'npml', NaN, ...
        'npml_over_points_per_a', NaN, ...
        'nx_total', NaN, ...
        'ny_total', NaN, ...
        'total_u_dofs', NaN, ...
        'near_dofs', NaN, ...
        'far_dofs', NaN, ...
        'pml_dofs', NaN, ...
        'pml_dof_fraction', NaN, ...
        'mean_dense_rank', NaN, ...
        'mean_effective_rank', NaN, ...
        'mean_near_rank', NaN, ...
        'mean_far_rank', NaN, ...
        'mean_pml_rank', NaN, ...
        'mean_storage_saving_ratio', NaN, ...
        'mean_storage_saved_fraction', NaN, ...
        'mean_storage_saved_percent', NaN, ...
        'layer1_storage_saved_percent', NaN, ...
        'layer2_storage_saved_percent', NaN, ...
        'layer3_storage_saved_percent', NaN, ...
        'mean_final_helm_full_error', NaN, ...
        'mean_final_helm_phys_error', NaN, ...
        'mean_online_precorr_relres', NaN, ...
        'mean_completion_relres', NaN, ...
        'mean_online_total_time', NaN);
end

function row = summarize_run(npml, r)
    s = r.summary;
    row = empty_summary_row();
    row.npml = npml;
    row.npml_over_points_per_a = npml / r.cfg.pc_points_per_a;
    row.nx_total = r.problem.nx;
    row.ny_total = r.problem.ny;
    row.total_u_dofs = r.problem.Ntot;

    dofs = first_finite_row(s.corr_block_dofs);
    if ~isempty(dofs)
        row.near_dofs = dofs(1);
        row.far_dofs = dofs(2);
        row.pml_dofs = dofs(3);
        row.pml_dof_fraction = dofs(3) / max(sum(dofs), 1);
    end

    row.mean_dense_rank = mean(s.corr_pod_ranks(:), 'omitnan');
    row.mean_effective_rank = mean(s.corr_ranks(:), 'omitnan');
    row.mean_near_rank = mean(s.corr_block_ranks(:, 1), 'omitnan');
    row.mean_far_rank = mean(s.corr_block_ranks(:, 2), 'omitnan');
    row.mean_pml_rank = mean(s.corr_block_ranks(:, 3), 'omitnan');
    row.mean_storage_saving_ratio = mean(s.corr_storage_saving_ratio(:), 'omitnan');
    row.mean_storage_saved_fraction = mean(s.corr_storage_saved_fraction(:), 'omitnan');
    row.mean_storage_saved_percent = 100 * row.mean_storage_saved_fraction;

    saved_pct = 100 * s.corr_storage_saved_fraction(:);
    if numel(saved_pct) >= 1, row.layer1_storage_saved_percent = saved_pct(1); end
    if numel(saved_pct) >= 2, row.layer2_storage_saved_percent = saved_pct(2); end
    if numel(saved_pct) >= 3, row.layer3_storage_saved_percent = saved_pct(3); end

    row.mean_final_helm_full_error = safe_stat(s, 'final_helm_full', 2);
    row.mean_final_helm_phys_error = safe_stat(s, 'final_helm_phys', 2);
    row.mean_online_precorr_relres = safe_stat(s, 'online_precorr_relres', 2);
    row.mean_completion_relres = safe_stat(s, 'online_fp_relres', 2);
    row.mean_online_total_time = safe_stat(s, 'online_total_time', 2);
end

function dofs = first_finite_row(A)
    dofs = [];
    if isempty(A)
        return;
    end
    idx = find(all(isfinite(A), 2), 1, 'first');
    if ~isempty(idx)
        dofs = A(idx, :);
    end
end

function val = safe_stat(s, field_name, idx)
    val = NaN;
    if isfield(s, field_name) && numel(s.(field_name)) >= idx
        val = s.(field_name)(idx);
    end
end

function plot_file = plot_storage_sweep(tbl, fig_dir, mode)
    fig = figure('Name', 'PC PML thickness storage sweep', ...
        'Position', [120 120 760 500]);
    ax = axes(fig);
    plot(ax, tbl.npml, tbl.mean_storage_saved_percent, '-o', ...
        'Color', [0.05 0.32 0.58], 'MarkerFaceColor', [0.13 0.58 0.72], ...
        'LineWidth', 2.0, 'MarkerSize', 7, 'DisplayName', 'mean across layers');
    hold(ax, 'on');
    plot_layer_if_present(ax, tbl.npml, tbl.layer1_storage_saved_percent, [0.72 0.24 0.12], 'layer 1');
    plot_layer_if_present(ax, tbl.npml, tbl.layer2_storage_saved_percent, [0.24 0.55 0.25], 'layer 2');
    plot_layer_if_present(ax, tbl.npml, tbl.layer3_storage_saved_percent, [0.42 0.30 0.65], 'layer 3');
    hold(ax, 'off');
    xlabel(ax, 'PML thickness (grid points)');
    ylabel(ax, 'block storage saved vs dense POD (%)');
    title(ax, 'Photonic-crystal waveguide PML thickness sweep');
    grid(ax, 'on');
    box(ax, 'on');
    set(ax, 'FontSize', 16, 'LineWidth', 1.1);
    legend(ax, 'Location', 'best', 'FontSize', 13);
    plot_file = fullfile(fig_dir, ['pc_pml_thickness_storage_sweep_' char(mode) '.png']);
    exportgraphics(fig, plot_file, 'Resolution', 300);
end

function plot_layer_if_present(ax, x, y, color, name)
    if any(isfinite(y))
        plot(ax, x, y, '--', 'Color', color, 'LineWidth', 1.2, ...
            'Marker', '.', 'MarkerSize', 13, 'DisplayName', name);
    end
end

function out = merge_structs(a, b)
    out = a;
    names = fieldnames(b);
    for i = 1:numel(names)
        out.(names{i}) = b.(names{i});
    end
end

function s = rmfield_if_present(s, names)
    for i = 1:numel(names)
        if isfield(s, names{i})
            s = rmfield(s, names{i});
        end
    end
end

function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end
