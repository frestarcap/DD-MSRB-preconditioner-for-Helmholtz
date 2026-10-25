function out_file = plot_helm_trajectory_vs_nr_residual_fig4(mode_or_file, c_target)
%PLOT_HELM_TRAJECTORY_VS_NR_RESIDUAL_FIG4 Redraw the half-space residual
%   history (paper Fig. 4) using the same figure/font/line pipeline as
%   PLOT_HELM_SHIFTED_LAPLACIAN_COMPLETION_FIGURE (paper Fig. 5), so that
%   the two figures are stylistically identical (Helvetica via the native
%   MATLAB text renderer, FontSize 25/28/30, LineWidth 3.2, MarkerSize 9,
%   plot floor 1e-15, y-axis to 1e-15).
%
% Usage:
%   plot_helm_trajectory_vs_nr_residual_fig4();                     % paper
%   plot_helm_trajectory_vs_nr_residual_fig4("paper");
%   plot_helm_trajectory_vs_nr_residual_fig4("smoke");
%   plot_helm_trajectory_vs_nr_residual_fig4("/path/to/evidence.mat");
%   plot_helm_trajectory_vs_nr_residual_fig4("paper", 0.8622);

    if nargin < 1 || isempty(mode_or_file)
        mode_or_file = "paper";
    end
    if nargin < 2
        c_target = [];
    end

    if strcmpi(mode_or_file, 'paper')
        files = plot_halfspace_dd_paper([],c_target);
        out_file = files.residual;
        return;
    end

    mat_file = resolve_evidence_results_file(mode_or_file);
    data = load(mat_file);
    if isfield(data, 'rep_results')
        rep = data.rep_results;
    elseif isfield(data, 'results') && isfield(data.results, 'representative')
        rep = data.results.representative;
    else
        error('The MAT file does not contain rep_results or results.representative: %s', mat_file);
    end
    if isfield(data, 'dd_results')
        dd = data.dd_results;
    elseif isfield(data, 'results') && isfield(data.results, 'dd_results')
        dd = data.results.dd_results;
    else
        dd = [];
    end

    test    = rep.test;
    summary = rep.summary;

    rep_idx = summary.representative_test_index;
    if ~isempty(c_target) && isfield(test, 'params') && ~isempty(test.params)
        [~, rep_idx] = min(abs(test.params(:) - c_target));
    end
    c_rep = summary.representative_c;
    if isfield(test, 'params') && ~isempty(test.params)
        c_rep = test.params(rep_idx);
    end

    plot_floor = 1e-15;

    fig_dir = helm_output_path('outputs', 'figures', 'helm_correction_rom');
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    fig = figure('Name', 'Half-space residual history', ...
        'Color', 'w', 'Position', [80 80 980 720], 'Visible', 'off');
    ax = axes(fig);
    hold(ax, 'on');

    % TA-ROM (red circles)
    ta_hist = test.online_combined_relres_hist{rep_idx};
    if has_plot_data(ta_hist)
        plot_sparse_curve(ax, ta_hist, plot_floor, ...
            '-o', 'Color', [0.85 0.05 0.05], 'LineWidth', 3.2, ...
            'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
            'DisplayName', 'TA-ROM');
    end

    % DD-TA-ROM (green squares)
    if ~isempty(dd) && isfield(dd, 'test')
        dd_idx = choose_matching_index(dd.test.params, c_rep, ...
            dd.summary.representative_test_index);
        dd_hist = dd.test.online_combined_relres_hist{dd_idx};
        if has_plot_data(dd_hist)
            plot_sparse_curve(ax, dd_hist, plot_floor, ...
                '-s', 'Color', [0.05 0.45 0.25], 'LineWidth', 3.2, ...
                'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
                'DisplayName', 'DD-TA-ROM');
        end
    end

    % NR-ROM (blue diamonds, dashed)
    if isfield(rep, 'nr_comparison') && ~isempty(rep.nr_comparison)
        nr = rep.nr_comparison;
        nr_idx = choose_matching_index(nr.test.params, c_rep, ...
            nr.summary.representative_test_index);
        nr_hist = nr.test.online_combined_relres_hist{nr_idx};
        if has_plot_data(nr_hist)
            plot_sparse_curve(ax, nr_hist, plot_floor, ...
                '--d', 'Color', [0.15 0.35 0.75], 'LineWidth', 3.0, ...
                'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
                'DisplayName', 'NR-ROM');
        end
    end

    % Plain WaveHoltz (black circles)
    plain_hist = test.plain_fp_relres_hist{rep_idx};
    if has_plot_data(plain_hist)
        plot_sparse_curve(ax, plain_hist, plot_floor, ...
            '-o', 'Color', [0.15 0.15 0.15], 'LineWidth', 3.0, ...
            'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
            'DisplayName', 'Plain WaveHoltz');
    end

    % Plain Helmholtz GMRES (purple circles) --- SAME styling as Fig. 5
    helm_hist = test.helm_baseline_relres_hist{rep_idx};
    if has_plot_data(helm_hist)
        plot_sparse_curve(ax, helm_hist, plot_floor, ...
            '-o', 'Color', [0.55 0 0.65], 'LineWidth', 2.6, ...
            'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
            'DisplayName', 'Plain Helmholtz GMRES');
    end

    hold(ax, 'off');
    set(ax, 'FontSize', 25, 'LineWidth', 1.8, 'Box', 'on', ...
        'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.28, 'MinorGridAlpha', 0.18);
    ax.XMinorGrid = 'off';
    ax.YMinorGrid = 'on';
    ylim(ax, [plot_floor, 2]);
    xlabel(ax, 'online step index');
    ylabel(ax, 'relative residual');
    title(ax, sprintf('Half-space residual decay, c_{right}=%.4f', c_rep));
    ax.Title.FontSize = 30;
    ax.Title.FontWeight = 'bold';
    ax.XLabel.FontSize = 28;
    ax.YLabel.FontSize = 28;
    lgd = legend(ax, 'Location', 'best');
    lgd.FontSize = 21;
    lgd.Box = 'on';
    lgd.LineWidth = 1.1;
    try
        axtoolbar(ax, {});
    catch
    end

    tag = sanitize_filename_local(sprintf('c_right_%0.4f', c_rep));
    out_file = fullfile(fig_dir, ['helm_trajectory_vs_nr_residual_decay_' tag '.png']);
    exportgraphics(fig, out_file, 'Resolution', 300);
    fprintf('saved figure: %s\n', out_file);

    % Also copy to LaTeX figures directory next to Fig. 5.
    latex_dir = helm_output_path('LaTeX_DL_468198_240419', 'figures', 'helm_correction_rom');
    if exist(latex_dir, 'dir')
        latex_file = fullfile(latex_dir, ['helm_trajectory_vs_nr_residual_decay_' tag '.png']);
        exportgraphics(fig, latex_file, 'Resolution', 300);
        fprintf('saved figure: %s\n', latex_file);
    end
    close(fig);
end

function mat_file = resolve_evidence_results_file(mode_or_file)
    mode_or_file = char(mode_or_file);
    if isfile(mode_or_file)
        mat_file = mode_or_file;
        return;
    end
    mode = lower(strtrim(mode_or_file));
    mat_file = helm_output_path('outputs', 'helm_trajectory_evidence', ...
        ['helm_trajectory_evidence_' mode '.mat']);
    if ~isfile(mat_file)
        error(['Results file not found: %s. Run ', ...
            'run_helm_trajectory_evidence_experiment("%s") first.'], ...
            mat_file, mode);
    end
end

function idx = choose_matching_index(params, c_rep, fallback_idx)
    if isempty(params)
        idx = fallback_idx;
        return;
    end
    [~, idx] = min(abs(params(:) - c_rep));
end

function tf = has_plot_data(vals)
    tf = ~isempty(vals) && any(isfinite(vals(:)));
end

function y = sanitize_curve(curve, floor_value)
    y = curve(:).';
    bad = ~isfinite(y) | y <= 0;
    y(bad) = NaN;
    y = max(y, floor_value);
end

function h = plot_sparse_curve(ax, hist, floor_value, line_style, varargin)
    y = sanitize_curve(hist, floor_value);
    h = semilogy(ax, 0:(numel(y) - 1), y, line_style, varargin{:});
end

function name = sanitize_filename_local(name)
    name = lower(regexprep(name, '[^a-zA-Z0-9]+', '_'));
    name = regexprep(name, '^_+|_+$', '');
end
