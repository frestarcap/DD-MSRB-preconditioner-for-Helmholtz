function out_file = plot_helm_tol_tradeoff_fig6(mode_or_file)
%PLOT_HELM_TOL_TRADEOFF_FIG6 Redraw the half-space WaveHoltz tolerance
%   speedup curve (paper Fig. 6) using the same figure/font/line pipeline as
%   PLOT_HELM_SHIFTED_LAPLACIAN_COMPLETION_FIGURE (paper Fig. 5), so that
%   Fig. 6 shares Fig. 5's fonts, tick sizes, and legend styling.
%
%   The curve content is the raw
%   speedup_vs_plain_waveholtz column from the same
%   helm_tol_tradeoff_paper.csv (or .mat) that the paper already reports.
%
% Usage:
%   plot_helm_tol_tradeoff_fig6();                    % paper by default
%   plot_helm_tol_tradeoff_fig6("paper");
%   plot_helm_tol_tradeoff_fig6("smoke");
%   plot_helm_tol_tradeoff_fig6("/path/to/helm_tol_tradeoff_paper.mat");

    if nargin < 1 || isempty(mode_or_file)
        mode_or_file = "paper";
    end

    tbl = load_tradeoff_table(mode_or_file);
    if isempty(tbl)
        error('No tolerance trade-off table found for mode/file: %s', mode_or_file);
    end

    tol   = tbl.tol(:);
    speed = tbl.speedup_vs_plain_waveholtz(:);
    [tol_sorted, order] = sort(tol, 'descend');
    speed_sorted = speed(order);

    fig_dir = helm_output_path('outputs', 'figures', 'helm_correction_rom');
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    fig = figure('Name', 'Half-space WaveHoltz tolerance speedup', ...
        'Color', 'w', 'Position', [80 80 980 720], 'Visible', 'off');
    ax = axes(fig);
    hold(ax, 'on');

    % Blue line connecting the tradeoff samples.
    plot(ax, tol_sorted, speed_sorted, '-', ...
        'Color', [0.10 0.35 0.75], 'LineWidth', 3.0, ...
        'HandleVisibility', 'off');

    % Blue hollow circles for the sampled tolerances -- same marker recipe
    % as Fig. 5 (LineWidth 3.2, MarkerSize 9, MarkerFaceColor 'w').
    plot(ax, tol_sorted, speed_sorted, 'o', ...
        'Color', [0.10 0.35 0.75], 'LineWidth', 3.2, ...
        'MarkerSize', 11, 'MarkerFaceColor', 'w', ...
        'LineStyle', 'none', ...
        'DisplayName', 'TA-ROM speedup');

    % Reference line at y=1.
    yline(ax, 1, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', 1.8, ...
        'HandleVisibility', 'off');

    hold(ax, 'off');

    % Axis styling identical to Fig. 5 (except log-log for the tradeoff).
    set(ax, 'FontSize', 25, 'LineWidth', 1.8, 'Box', 'on', ...
        'XScale', 'log', 'YScale', 'log', 'XDir', 'reverse', ...
        'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.28, 'MinorGridAlpha', 0.18);
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';

    xlim(ax, tradeoff_xlim(tol_sorted));
    ylim(ax, tradeoff_ylim(speed_sorted));

    xlabel(ax, 'residual tolerance');
    ylabel(ax, 'speedup vs plain WH');
    title(ax, 'Half-space WaveHoltz tolerance speedup');
    ax.Title.FontSize = 30;
    ax.Title.FontWeight = 'bold';
    ax.XLabel.FontSize = 28;
    ax.YLabel.FontSize = 28;

    lgd = legend(ax, 'Location', 'southwest');
    lgd.FontSize = 21;
    lgd.Box = 'on';
    lgd.LineWidth = 1.1;

    % Point-wise annotations in the ECC/PC style (bold, boxed).
    annotate_points(ax, tol_sorted, speed_sorted);

    try
        axtoolbar(ax, {});
    catch
    end

    out_name = 'helm_tol_tradeoff_paper.png';
    out_file = fullfile(fig_dir, out_name);
    exportgraphics(fig, out_file, 'Resolution', 300);
    fprintf('saved figure: %s\n', out_file);

    latex_dir = helm_output_path('LaTeX_DL_468198_240419', 'figures', 'helm_correction_rom');
    if exist(latex_dir, 'dir')
        latex_file = fullfile(latex_dir, out_name);
        exportgraphics(fig, latex_file, 'Resolution', 300);
        fprintf('saved figure: %s\n', latex_file);
    end
    close(fig);
end

function tbl = load_tradeoff_table(mode_or_file)
    tbl = table();
    mode_or_file = char(mode_or_file);
    if isfile(mode_or_file)
        candidates = {mode_or_file};
    else
        mode = lower(strtrim(mode_or_file));
        stem = ['helm_tol_tradeoff_' mode];
        base_dir = helm_output_path('outputs', 'helm_trajectory_evidence');
        candidates = {
            fullfile(base_dir, [stem '.mat']);
            fullfile(base_dir, [stem '.csv'])
        };
    end
    for k = 1:numel(candidates)
        cand = candidates{k};
        if ~isfile(cand)
            continue;
        end
        [~, ~, ext] = fileparts(cand);
        switch lower(ext)
            case '.mat'
                S = load(cand);
                if isfield(S, 'tbl'),                tbl = S.tbl;                return; end
                if isfield(S, 'tradeoff_table'),     tbl = S.tradeoff_table;     return; end
            case '.csv'
                tbl = readtable(cand);
                return;
        end
    end
end

function xl = tradeoff_xlim(tols)
    x = tols(isfinite(tols) & tols > 0);
    if isempty(x)
        xl = [1e-14, 1e-6];
        return;
    end
    xl = [10^floor(log10(min(x) * 0.8)), 10^ceil(log10(max(x) * 1.25))];
    % Note: XDir=reverse handles orientation.
end

function yl = tradeoff_ylim(speeds)
    y = speeds(isfinite(speeds) & speeds > 0);
    y = [y(:); 1];
    yl = [1, 10^ceil(log10(max(y) * 1.25))];
end

function annotate_points(ax, tols, speeds)
    for k = 1:numel(tols)
        t = tols(k);
        s = speeds(k);
        label = sprintf('%s, %.1fx', format_tol_label(t), s);
        [halign, valign, dx_log, dy_log] = annotation_offset(k, numel(tols));
        text(ax, t * 10^dx_log, s * 10^dy_log, label, ...
            'FontSize', 18, 'FontWeight', 'bold', ...
            'Color', [0.08 0.08 0.08], ...
            'HorizontalAlignment', halign, ...
            'VerticalAlignment', valign, ...
            'BackgroundColor', 'w', 'Margin', 3, 'Clipping', 'on');
    end
end

function label = format_tol_label(t)
    exponent = round(log10(t));
    if abs(t - 10^exponent) <= 10 * eps(max(1, abs(t)))
        label = sprintf('10^{%d}', exponent);
    else
        label = sprintf('%.1e', t);
    end
end

function [halign, valign, dx_log, dy_log] = annotation_offset(k, ntotal)
    % Simple deterministic offset schedule so labels don't overlap the marker.
    if k == 1
        halign = 'left';  valign = 'bottom'; dx_log = -0.25; dy_log = 0.20;
    elseif k == ntotal
        halign = 'right'; valign = 'top';    dx_log = -0.10; dy_log = -0.15;
    elseif mod(k, 2) == 0
        halign = 'left';  valign = 'top';    dx_log = 0.10; dy_log = -0.15;
    else
        halign = 'left';  valign = 'bottom'; dx_log = 0.10; dy_log = 0.15;
    end
end
