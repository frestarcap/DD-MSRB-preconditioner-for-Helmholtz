function out_file = plot_helm_shifted_laplacian_completion_figure(mode_or_file, fig_subdir, c_target)
%PLOT_HELM_SHIFTED_LAPLACIAN_COMPLETION_FIGURE Redraw the solver-agnostic test.
%
% Usage:
%   plot_helm_shifted_laplacian_completion_figure("smoke")
%   plot_helm_shifted_laplacian_completion_figure("paper")
%   plot_helm_shifted_laplacian_completion_figure("/path/to/results.mat")
%   plot_helm_shifted_laplacian_completion_figure("paper", "helm_shifted_laplacian_completion", 0.8905)

    if nargin < 1 || isempty(mode_or_file)
        mode_or_file = "paper";
    end
    if nargin < 2 || isempty(fig_subdir)
        fig_subdir = 'helm_shifted_laplacian_completion';
    end
    if nargin < 3
        c_target = [];
    end

    mat_file = resolve_shifted_laplacian_results_file(mode_or_file);
    data = load(mat_file);
    if isfield(data, 'results')
        results = data.results;
    else
        error('The MAT file does not contain a results struct: %s', mat_file);
    end
    if ~strcmp(results.cfg.online_completion_mode, 'shifted_laplacian')
        error('Expected online_completion_mode=shifted_laplacian, got %s.', ...
            results.cfg.online_completion_mode);
    end
    if strcmpi(mode_or_file,'paper')
        assert(results.cfg.shifted_laplacian_tol == 1e-12 && ...
            results.cfg.online_helm_accept_tol == 1e-12 && ...
            results.cfg.helm_baseline_tol == 1e-12, ...
            'The paper figure requires the common 1e-12 stopping target.');
    end
    expected_tols = [1e-5, 1e-5, 1e-5];
    if ~isfield(results.cfg, 'corr_pod_singular_value_tols') || ...
            ~isequal(size(results.cfg.corr_pod_singular_value_tols), size(expected_tols)) || ...
            any(abs(results.cfg.corr_pod_singular_value_tols - expected_tols) > ...
                10 * eps(max(1, max(expected_tols))))
        error('The shifted-Laplacian paper figure source uses a stale ROM tail schedule.');
    end

    test = results.test;
    summary = results.summary;
    rep_idx = summary.representative_test_index;
    if ~isempty(c_target)
        [~, rep_idx] = min(abs(test.params(:) - c_target));
    end
    c_rep = summary.representative_c;
    if isfield(test, 'params') && ~isempty(test.params)
        c_rep = test.params(rep_idx);
    end
    plot_floor = 1e-15;

    fig_dir = helm_output_path('outputs', 'figures', fig_subdir);
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    fig = figure('Name', 'TA-ROM shifted-Laplacian completion', ...
        'Color', 'w', 'Position', [80 80 980 720], 'Visible', 'off');
    ax = axes(fig);
    hold(ax, 'on');

    plain_hist = test.plain_fp_relres_hist{rep_idx};
    if has_plot_data(plain_hist)
        plot_sparse_curve(ax, plain_hist, plot_floor, ...
            '-o', 'Color', [0.15 0.15 0.15], 'LineWidth', 3.0, ...
            'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
            'DisplayName', 'Plain shifted-Laplacian GMRES');
    end

    ta_hist = test.online_combined_relres_hist{rep_idx};
    ta_label = 'Global TA-ROM';
    if test.online_fp_iter(rep_idx) > 0
        ta_label = 'TA-ROM + shifted-Laplacian GMRES';
    end
    if has_plot_data(ta_hist)
        plot_sparse_curve(ax, ta_hist, plot_floor, ...
            '-s', 'Color', [0.85 0.05 0.05], 'LineWidth', 3.2, ...
            'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
            'DisplayName', ta_label);
    end

    helm_hist = test.helm_baseline_relres_hist{rep_idx};
    if has_plot_data(helm_hist)
        plot_sparse_curve(ax, helm_hist, plot_floor, ...
            '-o', 'Color', [0.55 0 0.65], 'LineWidth', 2.6, ...
            'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
            'DisplayName', 'Plain Helmholtz GMRES');
    end

    transition_idx = test.online_combined_transition_idx(rep_idx);
    if test.online_fp_iter(rep_idx) > 0 && ...
            transition_idx >= 1 && transition_idx < numel(ta_hist)
        semilogy(ax, transition_idx - 1, sanitize_curve(ta_hist(transition_idx), plot_floor), ...
            'd', 'Color', [0.85 0.05 0.05], 'MarkerFaceColor', 'w', ...
            'LineWidth', 2.2, 'MarkerSize', 12, ...
            'DisplayName', 'completion starts');
        yl = ylim(ax);
        line(ax, [transition_idx - 1, transition_idx - 1], yl, ...
            'Color', [0.85 0.05 0.05], 'LineStyle', ':', ...
            'LineWidth', 1.8, 'HandleVisibility', 'off');
    end
    yline(ax,results.cfg.shifted_laplacian_tol,':','Color',[0.4 0.4 0.4], ...
        'LineWidth',1.5,'HandleVisibility','off');

    hold(ax, 'off');
    style_halfspace_residual_axes(ax,c_rep);

    tag = sanitize_filename_local(sprintf('c_right_%0.4f', c_rep));
    out_file = fullfile(fig_dir, ['helm_shifted_laplacian_solver_agnostic_residual_' tag '.png']);
    set(fig,'PaperPositionMode','auto');
    print(fig,out_file,'-dpng','-r300');
    fprintf('saved figure: %s\n', out_file);

    joint_file = fullfile(fig_dir, ['helm_joint_combined_residual_decay_full_field_' tag '.png']);
    copyfile(out_file,joint_file);
    fprintf('saved figure: %s\n', joint_file);
    latex_dir = helm_output_path('LaTeX_DL_468198_240419','figures', ...
        'helm_shifted_laplacian_completion');
    if exist(latex_dir,'dir')
        copyfile(joint_file,fullfile(latex_dir, ...
            ['helm_joint_combined_residual_decay_full_field_' tag '.png']));
    end
    close(fig);
end

function mat_file = resolve_shifted_laplacian_results_file(mode_or_file)
    mode_or_file = char(mode_or_file);
    if isfile(mode_or_file)
        mat_file = mode_or_file;
        return;
    end
    mode = lower(strtrim(mode_or_file));
    tag = mode;
    if strcmp(mode,'paper'), tag = 'paper_tol1e12'; end
    mat_file = helm_output_path('outputs', 'helm_shifted_laplacian_completion', ...
        ['helm_shifted_laplacian_completion_' tag '.mat']);
    if ~isfile(mat_file)
        error('Results file not found: %s. Run run_helm_shifted_laplacian_completion_experiment("%s") first.', ...
            mat_file, mode);
    end
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
