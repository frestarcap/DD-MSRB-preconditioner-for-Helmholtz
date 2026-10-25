function out_file = plot_pc_four_block_residual_ablation(global_mat, dd_mat)
%PLOT_PC_FOUR_BLOCK_RESIDUAL_ABLATION
% Plot Global TA-ROM and four-block DD-TA-ROM residual histories together.

    if nargin < 1 || isempty(global_mat)
        global_mat = helm_output_path('outputs', 'helm_trajectory_evidence', ...
            'pc_residual_ablation_paper.mat');
    end
    if nargin < 2 || isempty(dd_mat)
        dd_mat = helm_output_path('outputs', 'helm_trajectory_evidence', ...
            'pc_four_block_partition_paper.mat');
    end

    G = load(global_mat, 'global_results');
    D = load(dd_mat, 'results');
    global_results = G.global_results;
    dd_results = D.results;

    fig_dir = helm_output_path('outputs', 'figures', ...
        'photonic_crystal_correction_rom_4block');
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    global_c = representative_c_value(global_results);
    dd_c = representative_c_value(dd_results);
    if isfinite(dd_c)
        param_value = dd_c;
    else
        param_value = global_c;
    end
    global_idx = choose_matching_index(global_results.test.params, param_value, ...
        global_results.summary.representative_test_index);
    dd_idx = choose_matching_index(dd_results.test.params, param_value, ...
        dd_results.summary.representative_test_index);

    plot_floor = 1e-16;
    tag = sanitize_filename_local(sprintf('c_right_%0.4f', param_value));
    fig = figure('Name', 'PC four-block residual ablation', ...
        'Position', [140 140 760 500]);
    ax = axes(fig);
    if isprop(ax, 'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end
    hold(ax, 'on');
    plot_one_curve(ax, global_results.test.online_combined_relres_hist{global_idx}, ...
        '-o', [0.85 0.05 0.05], 'Global TA-ROM', plot_floor);
    plot_one_curve(ax, dd_results.test.online_combined_relres_hist{dd_idx}, ...
        '-s', [0.05 0.45 0.25], '4-block DD-TA-ROM', plot_floor);
    plot_one_curve(ax, global_results.test.plain_fp_relres_hist{global_idx}, ...
        'k-o', [0 0 0], plain_completion_label_local(global_results.cfg), plot_floor);
    plot_one_curve(ax, global_results.test.helm_baseline_relres_hist{global_idx}, ...
        ':', [0.25 0.25 0.25], 'Plain Helmholtz GMRES', plot_floor);

    hold(ax, 'off');
    set(ax, 'YScale', 'log');
    ylim(ax, [plot_floor, 2]);
    xlabel(ax, 'online step index');
    ylabel(ax, 'relative residual');
    title(ax, sprintf('Waveguide residual decay, c=%.4f', param_value));
    grid(ax, 'on');
    box(ax, 'on');
    set(ax, 'FontSize', 18, 'LineWidth', 1.1);
    legend(ax, 'Location', 'northeast', 'FontSize', 17);

    out_file = fullfile(fig_dir, ...
        ['pc_four_block_joint_combined_residual_decay_full_field_' tag '.png']);
    exportgraphics(fig, out_file, 'Resolution', 300);
    fprintf('Saved four-block residual ablation: %s\n', out_file);
end

function c = representative_c_value(results)
    c = NaN;
    if isfield(results, 'summary') && isfield(results.summary, 'representative_c')
        c = results.summary.representative_c;
    elseif isfield(results, 'summary') && isfield(results.summary, 'representative_test_index') && ...
            isfield(results, 'test') && isfield(results.test, 'params')
        c = results.test.params(results.summary.representative_test_index);
    end
end

function plot_one_curve(ax, hist, style, color, display_name, plot_floor)
    if isempty(hist) || ~any(isfinite(hist(:)))
        return;
    end
    y = real(hist(:));
    y(~isfinite(y) | y <= plot_floor) = plot_floor;
    semilogy(ax, 0:(numel(y) - 1), y, style, ...
        'Color', color, 'LineWidth', 1.8, 'MarkerSize', 6, ...
        'DisplayName', display_name);
end

function idx = choose_matching_index(values, target, fallback)
    if ~isempty(values) && isfinite(target)
        [~, idx] = min(abs(values(:) - target));
    else
        idx = fallback;
    end
end

function label = plain_completion_label_local(cfg)
    if ~isfield(cfg, 'online_completion_mode') || isempty(cfg.online_completion_mode)
        label = 'Plain WaveHoltz';
        return;
    end
    mode = lower(strrep(strrep(char(string(cfg.online_completion_mode)), '-', '_'), ' ', '_'));
    switch mode
        case {'helm', 'helmholtz', 'plain_helmholtz', 'helm_gmres'}
            label = 'Plain Helmholtz completion';
        case {'shifted_laplacian', 'shifted_lap', 'sl', 'shifted'}
            label = 'Plain shifted-Laplacian';
        otherwise
            label = 'Plain WaveHoltz';
    end
end

function s = sanitize_filename_local(s)
    s = regexprep(s, '[^A-Za-z0-9_-]+', '_');
    s = regexprep(s, '_+', '_');
    s = regexprep(s, '^_|_$', '');
end
