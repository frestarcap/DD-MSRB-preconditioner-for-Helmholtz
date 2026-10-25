function out_file = plot_helm_fig6_residual_history(mat_file, c_target)
%PLOT_HELM_FIG6_RESIDUAL_HISTORY Redraw paper Fig. 6 from saved data.
%
% Usage:
%   plot_helm_fig6_residual_history
%   plot_helm_fig6_residual_history("outputs/helm_trajectory_evidence/helm_trajectory_evidence_paper.mat", 0.8622)

    if nargin < 1 || isempty(mat_file)
        mat_file = helm_output_path('outputs', 'helm_trajectory_evidence', ...
            'helm_trajectory_evidence_paper.mat');
    end
    if nargin < 2 || isempty(c_target)
        c_target = 0.8622;
    end

    data = load(mat_file, 'rep_results');
    if ~isfield(data, 'rep_results')
        error('The MAT file does not contain rep_results: %s', mat_file);
    end
    rep_results = data.rep_results;
    traj_test = rep_results.test;
    traj_summary = rep_results.summary;
    nr = rep_results.nr_comparison;

    rep_idx = choose_matching_index(traj_test, c_target, traj_summary);
    c_rep = traj_test.params(rep_idx);
    plot_floor = 1e-16;

    fig = figure('Name', 'Half-space Fig. 6 residual history', ...
        'Position', [140 140 760 500]);
    ax = axes(fig);
    hold(ax, 'on');

    traj_hist = traj_test.online_combined_relres_hist{rep_idx};
    if has_curve(traj_hist)
        semilogy(ax, 0:(numel(traj_hist) - 1), clamp_curve(traj_hist, plot_floor), ...
            '-o', 'Color', [0.85 0.05 0.05], 'LineWidth', 1.8, ...
            'MarkerSize', 6, 'DisplayName', 'TA-ROM');
    end

    if ~isempty(nr) && isfield(nr, 'test')
        nr_idx = choose_matching_index(nr.test, c_rep, nr.summary);
        nr_hist = nr.test.online_combined_relres_hist{nr_idx};
        if has_curve(nr_hist)
            semilogy(ax, 0:(numel(nr_hist) - 1), clamp_curve(nr_hist, plot_floor), ...
                '--d', 'Color', [0.15 0.35 0.75], 'LineWidth', 1.6, ...
                'MarkerSize', 6, 'DisplayName', 'NR-ROM');
        end
    end

    plain_hist = traj_test.plain_fp_relres_hist{rep_idx};
    if has_curve(plain_hist)
        semilogy(ax, 0:(numel(plain_hist) - 1), clamp_curve(plain_hist, plot_floor), ...
            'k-o', 'LineWidth', 1.3, 'MarkerSize', 5, ...
            'DisplayName', 'Plain WaveHoltz');
    end

    helm_hist = traj_test.helm_baseline_relres_hist{rep_idx};
    if has_curve(helm_hist)
        semilogy(ax, 0:(numel(helm_hist) - 1), clamp_curve(helm_hist, plot_floor), ...
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
    set(fig, 'Color', 'w');
    try
        axtoolbar(ax, {});
    catch
    end

    tag = sanitize_filename_local(sprintf('c_right_%0.4f', c_rep));
    fig_name = ['helm_trajectory_vs_nr_residual_decay_' tag '.png'];
    out_dir = helm_output_path('outputs', 'figures', 'helm_correction_rom');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    out_file = fullfile(out_dir, fig_name);
    exportgraphics(fig, out_file, 'Resolution', 300);
    fprintf('saved figure: %s\n', out_file);

    latex_dir = fullfile(helm_project_root(), 'LaTeX_DL_468198_240419', ...
        'figures', 'helm_correction_rom');
    if exist(latex_dir, 'dir')
        latex_file = fullfile(latex_dir, fig_name);
        copyfile(out_file, latex_file);
        fprintf('copied figure: %s\n', latex_file);
    end
end

function idx = choose_matching_index(test, c_target, summary)
    if isfield(test, 'params') && ~isempty(test.params)
        [~, idx] = min(abs(test.params(:) - c_target));
    else
        idx = summary.representative_test_index;
    end
end

function tf = has_curve(x)
    tf = ~isempty(x) && any(isfinite(x(:)));
end

function y = clamp_curve(x, floor_value)
    y = x(:).';
    y(~isfinite(y) | y <= 0) = NaN;
    y = max(y, floor_value);
end

function name = sanitize_filename_local(name)
    name = lower(regexprep(name, '[^a-zA-Z0-9]+', '_'));
    name = regexprep(name, '^_+|_+$', '');
end
