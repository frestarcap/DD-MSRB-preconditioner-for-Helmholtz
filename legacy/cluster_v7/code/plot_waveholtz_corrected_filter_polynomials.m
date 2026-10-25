function data = plot_waveholtz_corrected_filter_polynomials(cfg)
%PLOT_WAVEHOLTZ_CORRECTED_FILTER_POLYNOMIALS Visualize p_{N,alpha_N}.
%   Plots P_N(cos(x)), |p_N(exp(i*phi))|, and the maximum gain versus N.

    if nargin < 1, cfg = struct(); end
    cfg = set_default(cfg, 'N_values', [4 5 6 8 12 20]);
    cfg = set_default(cfg, 'N_scan', 4:80);
    cfg = set_default(cfg, 'npoints', 20001);
    cfg = set_default(cfg, 'visible', 'off');
    cfg = set_default(cfg, 'output_name', 'waveholtz_filter_polynomials');

    outdir = helm_output_path('outputs', cfg.output_name);
    if ~exist(outdir, 'dir'), mkdir(outdir); end

    N_values = cfg.N_values(:).';
    x = linspace(0, pi, cfg.npoints);
    phi = 2*x;
    y = cos(x);
    colors = lines(numel(N_values));
    curves = repmat(struct('N', [], 'alpha', [], 'P', [], 'gain', [], ...
        'target_gain', [], 'max_gain', [], 'phase_error', []), numel(N_values), 1);

    ncols = min(3, numel(N_values));
    nrows = ceil(numel(N_values)/ncols);
    figP = figure('Visible', cfg.visible, 'Color', 'w', ...
        'Position', [80 80 430*ncols 310*nrows]);
    layoutP = tiledlayout(figP, nrows, ncols, ...
        'Padding', 'compact', 'TileSpacing', 'compact');

    figGain = figure('Visible', cfg.visible, 'Color', 'w', ...
        'Position', [100 100 980 540]);
    axGain = axes(figGain); hold(axGain, 'on');

    for k = 1:numel(N_values)
        N = N_values(k);
        [coeff, alpha] = filter_coefficients(N, 'corrected');
        z = exp(1i*phi);
        p = polyval(fliplr(coeff.'), z);
        centered = exp(-1i*N*x).*p;
        P = real(centered);

        curves(k).N = N;
        curves(k).alpha = alpha;
        curves(k).P = P;
        curves(k).gain = abs(p);
        curves(k).target_gain = polyval(fliplr(coeff.'), exp(2i*pi/N));
        curves(k).max_gain = max(max(abs(p)), abs(curves(k).target_gain));
        curves(k).phase_error = max(abs(imag(centered)));

        ax = nexttile(layoutP);
        plot(ax, fliplr(y), fliplr(P), 'Color', colors(k,:), 'LineWidth', 1.5);
        hold(ax, 'on');
        yline(ax, 1, 'k--', 'LineWidth', 0.8);
        yline(ax, -1, 'k--', 'LineWidth', 0.8);
        c = cos(pi/N);
        plot(ax, [c -c], [-1 (-1)^(N+1)], 'ko', ...
            'MarkerFaceColor', [0.95 0.35 0.15], 'MarkerSize', 5);
        xlim(ax, [-1 1]); ylim(ax, [-1.08 1.08]);
        grid(ax, 'on');
        xlabel(ax, '$y$', 'Interpreter', 'latex');
        ylabel(ax, '$P_N(y)$', 'Interpreter', 'latex');
        title(ax, sprintf('$N=%d,\\;\\alpha_N=%.6f$', N, alpha), ...
            'Interpreter', 'latex');

        plot(axGain, phi/(2*pi), abs(p), 'Color', colors(k,:), ...
            'LineWidth', 1.35, 'DisplayName', sprintf('N=%d', N));
    end

    yline(axGain, 1, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    xlim(axGain, [0 1]); ylim(axGain, [0 1.03]);
    grid(axGain, 'on');
    xlabel(axGain, '$\phi/(2\pi)$', 'Interpreter', 'latex');
    ylabel(axGain, '$|p_{N,\alpha_N}(e^{i\phi})|$', 'Interpreter', 'latex');
    title(axGain, 'Corrected discrete WaveHoltz filter on the unit circle');
    legend(axGain, 'Location', 'bestoutside');

    N_scan = cfg.N_scan(:).';
    corrected_max = zeros(size(N_scan));
    standard_max = zeros(size(N_scan));
    phi_scan = linspace(0, 2*pi, cfg.npoints);
    z_scan = exp(1i*phi_scan);
    for k = 1:numel(N_scan)
        N = N_scan(k);
        corrected_coeff = filter_coefficients(N, 'corrected');
        standard_coeff = filter_coefficients(N, 'standard');
        corrected_values = polyval(fliplr(corrected_coeff.'), z_scan);
        standard_values = polyval(fliplr(standard_coeff.'), z_scan);
        target = exp(2i*pi/N);
        corrected_max(k) = max(max(abs(corrected_values)), ...
            abs(polyval(fliplr(corrected_coeff.'), target)));
        standard_max(k) = max(max(abs(standard_values)), ...
            abs(polyval(fliplr(standard_coeff.'), target)));
    end

    figScan = figure('Visible', cfg.visible, 'Color', 'w', ...
        'Position', [120 120 900 480]);
    axScan = axes(figScan);
    plot(axScan, N_scan, corrected_max, 'o-', 'LineWidth', 1.2, ...
        'MarkerSize', 3.5, 'DisplayName', 'corrected $\alpha_N$');
    hold(axScan, 'on');
    plot(axScan, N_scan, standard_max, 's-', 'LineWidth', 1.2, ...
        'MarkerSize', 3.5, 'DisplayName', 'standard $1/4$');
    yline(axScan, 1, 'k--', 'HandleVisibility', 'off');
    grid(axScan, 'on');
    xlabel(axScan, '$N$', 'Interpreter', 'latex');
    ylabel(axScan, '$\max_{|z|=1}|p_N(z)|$', 'Interpreter', 'latex');
    title(axScan, 'Maximum unit-circle gain versus steps per period');
    legend(axScan, 'Interpreter', 'latex', 'Location', 'best');

    exportgraphics(figP, fullfile(outdir, 'centered_polynomials_by_N.png'), 'Resolution', 200);
    exportgraphics(figGain, fullfile(outdir, 'unit_circle_gain_by_N.png'), 'Resolution', 200);
    exportgraphics(figScan, fullfile(outdir, 'maximum_gain_vs_N.png'), 'Resolution', 200);

    scan_table = table(N_scan(:), corrected_max(:), standard_max(:), ...
        'VariableNames', {'N', 'corrected_max_gain', 'standard_max_gain'});
    writetable(scan_table, fullfile(outdir, 'maximum_gain_vs_N.csv'));

    data = struct('cfg', cfg, 'x', x, 'phi', phi, 'y', y, ...
        'curves', curves, 'N_scan', N_scan, ...
        'corrected_max', corrected_max, 'standard_max', standard_max, ...
        'output_dir', outdir);
    save(fullfile(outdir, 'filter_polynomial_curves.mat'), 'data');
end

function [coeff, alpha] = filter_coefficients(N, mode)
    eta = ones(N+1, 1);
    eta([1 end]) = 0.5;
    if strcmp(mode, 'corrected')
        alpha = 0.25 * (1 - tan(pi/N)^2);
    else
        alpha = 0.25;
    end
    n = (0:N).';
    coeff = (2/N) * eta .* (cos(2*pi*n/N) - alpha);
end

function s = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end
