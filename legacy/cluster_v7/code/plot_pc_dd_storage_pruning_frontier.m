function tbl = plot_pc_dd_storage_pruning_frontier()
%PLOT_PC_DD_STORAGE_PRUNING_FRONTIER Aggregate saved PC rank-pruning probes.

    out_dir = helm_output_path('outputs', 'pc_dd_storage_exploration');
    fig_dir = helm_output_path('outputs', 'figures', 'pc_dd_storage_exploration');
    ensure_dir(out_dir);
    ensure_dir(fig_dir);

    rows = repmat(empty_row(), 9, 1);
    rows(1) = make_row('current equal-absolute', 8.891, 1.610264145e-5, ...
        'full replay');

    strict = readtable(fullfile(out_dir, 'pc_dd_storage_paper_waveholtz.csv'));
    rows(2) = make_row('narrow4 DOF-weighted', ...
        strict.saved_vs_global_percent(1), strict.mean_precorr_relres(1), ...
        'full replay');

    probes = {
        'drop_l3_lr', 'drop L3 LR-PML mode'
        'drop_l3_tb', 'drop L3 TB-PML mode'
        'drop_l3_both', 'drop both L3 PML modes'
        'drop_l3_both_l2_lr', 'also drop L2 LR-PML mode'
        'drop_l3_both_l2_tb', 'also drop L2 TB-PML mode'
        'drop_l3_both_l2_both', 'drop both L2/L3 PML modes'
        'pml_20pct', '20% PML-pruned boundary'
    };
    for k = 1:size(probes, 1)
        saved = load(fullfile(out_dir, ...
            ['pc_saved_rank_pruning_' probes{k, 1} '.mat']), 'summary');
        rows(k + 2) = make_row(probes{k, 2}, ...
            saved.summary.saved_vs_global_percent, ...
            saved.summary.pruned_mean_relres, 'saved-ROM pruning');
    end

    tbl = struct2table(rows);
    tbl = sortrows(tbl, 'storage_saved_percent');
    csv_file = fullfile(out_dir, 'pc_dd_storage_residual_frontier.csv');
    mat_file = fullfile(out_dir, 'pc_dd_storage_residual_frontier.mat');
    fig_file = fullfile(fig_dir, 'pc_dd_storage_residual_frontier.png');
    writetable(tbl, csv_file);
    save(mat_file, 'tbl');
    draw_frontier(tbl, fig_file);
    fprintf('Saved PC DD storage frontier: %s\n', csv_file);
end


function row = make_row(label, saving, residual, evidence)
    row = empty_row();
    row.variant = string(label);
    row.storage_saved_percent = saving;
    row.mean_precorr_relres = residual;
    row.evidence = string(evidence);
end


function row = empty_row()
    row = struct('variant', "", 'storage_saved_percent', NaN, ...
        'mean_precorr_relres', NaN, 'evidence', "");
end


function draw_frontier(tbl, out_file)
    fig = figure('Visible', 'off', 'Position', [100, 100, 980, 600]);
    full_mask = tbl.evidence == "full replay";
    prune_mask = ~full_mask;
    semilogy(tbl.storage_saved_percent(prune_mask), ...
        tbl.mean_precorr_relres(prune_mask), '-o', ...
        'Color', [0.35, 0.42, 0.72], 'LineWidth', 1.8, ...
        'MarkerSize', 7, 'DisplayName', 'saved-ROM PML pruning');
    hold on;
    semilogy(tbl.storage_saved_percent(full_mask), ...
        tbl.mean_precorr_relres(full_mask), 's', ...
        'Color', [0.08, 0.48, 0.27], 'MarkerFaceColor', [0.08, 0.48, 0.27], ...
        'LineWidth', 1.8, 'MarkerSize', 9, 'DisplayName', 'full replay');
    yline(1.610264145e-5, ':', 'current DD residual', ...
        'Color', [0.3, 0.3, 0.3], 'LineWidth', 1.4, ...
        'HandleVisibility', 'off');
    xline(20, ':', '20% guide', 'Color', [0.55, 0.25, 0.20], ...
        'LineWidth', 1.4, 'HandleVisibility', 'off');
    hold off;
    grid on;
    xlabel('basis entries saved versus Global TA-ROM (%)');
    ylabel('mean three-step FGMRES relative residual');
    title('Photonic-crystal DD storage-residual frontier');
    legend('Location', 'northwest');
    set(findall(fig, '-property', 'FontSize'), 'FontSize', 15);
    exportgraphics(fig, out_file, 'Resolution', 220);
    close(fig);
end


function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end
