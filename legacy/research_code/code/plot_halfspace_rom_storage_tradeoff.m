function out_file = plot_halfspace_rom_storage_tradeoff(csv_file)
%PLOT_HALFSPACE_ROM_STORAGE_TRADEOFF Plot ROM accuracy/storage threshold sweep.

    if nargin < 1 || isempty(csv_file)
        csv_file = helm_output_path('outputs', 'rom_threshold_balance', ...
            'rom_threshold_balance_halfspace_paper_uniform_storage_tradeoff_dof_weighted.csv');
    end
    if ~isfile(csv_file)
        error('Half-space ROM threshold sweep not found: %s', csv_file);
    end

    tbl = readtable(csv_file);
    required = {'eta_geometric_mean', 'global_precorr_relres', ...
        'dd_precorr_relres', 'dd_saved_vs_global_percent'};
    if ~all(ismember(required, tbl.Properties.VariableNames))
        error('Threshold sweep table is missing required columns.');
    end

    eta = tbl.eta_geometric_mean(:);
    [eta, order] = sort(eta, 'descend');
    global_residual = tbl.global_precorr_relres(order);
    dd_residual = tbl.dd_precorr_relres(order);
    dd_saving = tbl.dd_saved_vs_global_percent(order);
    selected_eta = 1e-5;

    fig = figure('Name', 'Half-space ROM threshold storage trade-off', ...
        'Color', 'w', 'Position', [60 80 1420 610], 'Visible', 'off');
    layout = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', ...
        'Padding', 'compact');

    ax1 = nexttile(layout);
    hold(ax1, 'on');
    plot(ax1, eta, global_residual, '-o', 'Color', [0.10 0.35 0.75], ...
        'LineWidth', 2.8, 'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
        'DisplayName', 'Global TA-ROM');
    plot(ax1, eta, dd_residual, '-s', 'Color', [0.05 0.48 0.27], ...
        'LineWidth', 2.8, 'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
        'DisplayName', 'DD-TA-ROM');
    xline(ax1, selected_eta, '--', 'Color', [0.25 0.25 0.25], ...
        'LineWidth', 1.6, 'HandleVisibility', 'off');
    hold(ax1, 'off');
    style_axes(ax1, eta, true);
    xlabel(ax1, 'uniform ROM tail threshold, \eta');
    ylabel(ax1, 'mean pre-completion residual');
    title(ax1, '(a) Reduced-stage accuracy');
    lgd = legend(ax1, 'Location', 'southwest');
    lgd.FontSize = 18;
    lgd.Box = 'on';

    ax2 = nexttile(layout);
    hold(ax2, 'on');
    plot(ax2, eta, dd_saving, '-', 'Color', [0.68 0.20 0.12], ...
        'LineWidth', 2.8, 'HandleVisibility', 'off');
    plot(ax2, eta, dd_saving, 'o', 'Color', [0.68 0.20 0.12], ...
        'LineWidth', 2.8, 'MarkerSize', 10, 'MarkerFaceColor', 'w', ...
        'DisplayName', 'DD saving');
    selected = abs(log10(eta) - log10(selected_eta)) < 1e-12;
    plot(ax2, eta(selected), dd_saving(selected), 'o', ...
        'Color', [0.68 0.20 0.12], 'LineWidth', 2.8, ...
        'MarkerSize', 11, 'MarkerFaceColor', [0.96 0.72 0.20], ...
        'HandleVisibility', 'off');
    xline(ax2, selected_eta, '--', 'Color', [0.25 0.25 0.25], ...
        'LineWidth', 1.6, 'HandleVisibility', 'off');
    yline(ax2, 0, ':', 'Color', [0.35 0.35 0.35], ...
        'LineWidth', 1.4, 'HandleVisibility', 'off');
    for k = 1:numel(eta)
        text(ax2, eta(k), dd_saving(k) + 0.65, sprintf('%.1f%%', dd_saving(k)), ...
            'FontSize', 17, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    hold(ax2, 'off');
    style_axes(ax2, eta, false);
    ylim(ax2, [0, max(dd_saving) + 3.0]);
    xlabel(ax2, 'uniform ROM tail threshold, \eta');
    ylabel(ax2, 'DD basis entries saved vs Global (%)');
    title(ax2, '(b) Basis-storage reduction');

    out_name = 'helm_rom_threshold_storage_tradeoff_paper.png';
    out_dir = helm_output_path('outputs', 'figures', 'helm_correction_rom');
    ensure_dir(out_dir);
    out_file = fullfile(out_dir, out_name);
    exportgraphics(fig, out_file, 'Resolution', 300);
    fprintf('saved figure: %s\n', out_file);

    latex_dir = helm_output_path('LaTeX_DL_468198_240419', 'figures', ...
        'helm_correction_rom');
    export_if_present(fig, latex_dir, out_name);
    close(fig);
end

function style_axes(ax, eta, log_y)
    set(ax, 'FontSize', 20, 'LineWidth', 1.6, 'Box', 'on', ...
        'XScale', 'log', 'XDir', 'reverse', 'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.25, 'MinorGridAlpha', 0.15);
    if log_y
        ax.YScale = 'log';
        ax.YMinorGrid = 'on';
    end
    tick_eta = sort(eta, 'ascend');
    ax.XTick = tick_eta;
    ax.XTickLabel = compose('10^{%d}', round(log10(tick_eta)));
    xlim(ax, [min(eta) / 1.8, max(eta) * 1.8]);
    ax.Title.FontSize = 24;
    ax.Title.FontWeight = 'bold';
    ax.XLabel.FontSize = 22;
    ax.YLabel.FontSize = 22;
    try
        axtoolbar(ax, {});
    catch
    end
end

function export_if_present(fig, out_dir, out_name)
    if exist(out_dir, 'dir')
        out_file = fullfile(out_dir, out_name);
        exportgraphics(fig, out_file, 'Resolution', 300);
        fprintf('saved figure: %s\n', out_file);
    end
end

function ensure_dir(dir_name)
    if ~exist(dir_name, 'dir')
        mkdir(dir_name);
    end
end
