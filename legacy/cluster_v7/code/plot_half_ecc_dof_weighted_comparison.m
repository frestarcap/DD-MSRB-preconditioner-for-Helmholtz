function tbl = plot_half_ecc_dof_weighted_comparison()
%PLOT_HALF_ECC_DOF_WEIGHTED_COMPARISON Compare paper-grid allocation reruns.

    out_dir = helm_output_path('outputs', 'rom_threshold_balance');
    fig_dir = helm_output_path('outputs', 'figures', 'rom_threshold_balance');
    ensure_dir(fig_dir);

    inputs = {
        'Half-space', 'equal absolute', ...
            'rom_threshold_balance_halfspace_paper_uniform_1e5.csv', 1
        'Half-space', 'DOF-weighted', ...
            'rom_threshold_balance_halfspace_paper_dof_weighted.csv', 1
        'ECC', 'equal absolute', ...
            'rom_threshold_balance_ecc_paper_helmholtz_memory_sweep.csv', 1
        'ECC', 'DOF-weighted', ...
            'rom_threshold_balance_ecc_paper_dof_weighted_helmholtz.csv', 1
    };

    rows = repmat(empty_row(), size(inputs, 1), 1);
    for k = 1:size(inputs, 1)
        source = readtable(fullfile(out_dir, inputs{k, 3}), ...
            'VariableNamingRule', 'preserve');
        source = source(inputs{k, 4}, :);
        rows(k) = comparison_row(inputs{k, 1}, inputs{k, 2}, source);
    end
    tbl = struct2table(rows);

    csv_file = fullfile(out_dir, 'half_ecc_dof_weighted_comparison.csv');
    fig_file = fullfile(fig_dir, 'half_ecc_dof_weighted_comparison.png');
    writetable(tbl, csv_file);

    storage = [tbl.saved_vs_global_percent(1), tbl.saved_vs_global_percent(2); ...
        tbl.saved_vs_global_percent(3), tbl.saved_vs_global_percent(4)];
    residual = [tbl.global_precorr_relres(1), tbl.dd_precorr_relres(1), ...
            tbl.dd_precorr_relres(2); ...
        tbl.global_precorr_relres(3), tbl.dd_precorr_relres(3), ...
            tbl.dd_precorr_relres(4)];

    fig = figure('Visible', 'off', 'Name', 'DD allocation comparison', ...
        'Position', [120, 120, 1200, 500]);
    layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
        'Padding', 'compact');

    ax1 = nexttile(layout, 1);
    bars = bar(ax1, storage, 'grouped');
    bars(1).FaceColor = [0.18, 0.43, 0.70];
    bars(2).FaceColor = [0.12, 0.55, 0.32];
    set(ax1, 'XTickLabel', {'Half-space', 'ECC'});
    ylabel(ax1, 'basis entries saved vs Global (%)');
    title(ax1, 'DD storage');
    ylim(ax1, [0, 50]);
    grid(ax1, 'on');
    box(ax1, 'on');
    legend(ax1, {'equal absolute', 'DOF-weighted'}, 'Location', 'northwest');
    add_bar_labels(ax1, bars);

    ax2 = nexttile(layout, 2);
    x = (1:2).';
    semilogy(ax2, x - 0.14, residual(:, 1), 'o', ...
        'Color', [0.20, 0.20, 0.20], 'LineWidth', 1.8, ...
        'MarkerSize', 8, 'DisplayName', 'Global');
    hold(ax2, 'on');
    semilogy(ax2, x, residual(:, 2), 's', ...
        'Color', [0.18, 0.43, 0.70], 'LineWidth', 1.8, ...
        'MarkerSize', 8, 'DisplayName', 'DD equal absolute');
    semilogy(ax2, x + 0.14, residual(:, 3), 'd', ...
        'Color', [0.12, 0.55, 0.32], 'LineWidth', 1.8, ...
        'MarkerSize', 8, 'DisplayName', 'DD DOF-weighted');
    hold(ax2, 'off');
    set(ax2, 'XTick', 1:2, 'XTickLabel', {'Half-space', 'ECC'});
    xlim(ax2, [0.55, 2.45]);
    ylabel(ax2, 'mean three-step relative residual');
    title(ax2, 'Precompletion residual');
    grid(ax2, 'on');
    box(ax2, 'on');
    legend(ax2, 'Location', 'northwest');

    set([ax1, ax2], 'FontSize', 15, 'LineWidth', 1.2);
    exportgraphics(fig, fig_file, 'Resolution', 300);
    close(fig);

    fprintf('Saved comparison table: %s\n', csv_file);
    fprintf('Saved comparison figure: %s\n', fig_file);
end


function row = comparison_row(problem, allocation, source)
    row = empty_row();
    row.problem = string(problem);
    row.allocation = string(allocation);
    row.schedule = string(source.schedule);
    row.global_ranks = string(source.global_ranks);
    row.dd_block_ranks = string(source.dd_block_ranks);
    row.global_basis_entries = source.global_basis_entries;
    row.dd_basis_entries = source.dd_basis_entries;
    row.saved_vs_global_percent = source.dd_saved_vs_global_percent;
    row.global_precorr_relres = source.global_precorr_relres;
    row.dd_precorr_relres = source.dd_precorr_relres;
    row.dd_online_time_s = source.dd_online_time_s;
    row.dd_completion_iter = source.dd_completion_iter;
    row.dd_final_error = source.dd_final_error;
end


function row = empty_row()
    row = struct('problem', "", 'allocation', "", 'schedule', "", ...
        'global_ranks', "", 'dd_block_ranks', "", ...
        'global_basis_entries', NaN, 'dd_basis_entries', NaN, ...
        'saved_vs_global_percent', NaN, 'global_precorr_relres', NaN, ...
        'dd_precorr_relres', NaN, 'dd_online_time_s', NaN, ...
        'dd_completion_iter', NaN, 'dd_final_error', NaN);
end


function add_bar_labels(ax, bars)
    for k = 1:numel(bars)
        x = bars(k).XEndPoints;
        y = bars(k).YEndPoints;
        labels = compose('%.2f%%', y);
        text(ax, x, y + 0.8, labels, 'HorizontalAlignment', 'center', ...
            'FontSize', 12);
    end
end


function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end
