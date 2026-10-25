function out_files = plot_halfspace_partition_probe_paper(csv_file)
%PLOT_HALFSPACE_PARTITION_PROBE_PAPER Draw paper-scale half-space partition probe.
%
% This helper does not rerun any numerical experiment.  It visualizes the saved
% half-space paper-scale partition probe table and reconstructs the three DD
% partitions used by that probe.

    if nargin < 1 || isempty(csv_file)
        csv_file = helm_output_path('outputs', 'partition_probe', 'partition_probe_paper_halfspace.csv');
    end
    if ~exist(csv_file, 'file')
        error('Cannot find probe CSV: %s', csv_file);
    end

    out_dir = helm_output_path('outputs', 'figures', 'partition_probe');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    T = readtable(csv_file, 'TextType', 'string');

    out_files = struct();
    out_files.partition_png = fullfile(out_dir, 'halfspace_paper_partition_maps.png');
    out_files.partition_pdf = fullfile(out_dir, 'halfspace_paper_partition_maps.pdf');
    out_files.metrics_png = fullfile(out_dir, 'halfspace_paper_partition_metrics.png');
    out_files.metrics_pdf = fullfile(out_dir, 'halfspace_paper_partition_metrics.pdf');

    draw_partition_maps(out_files.partition_png, out_files.partition_pdf);
    draw_metric_summary(T, out_files.metrics_png, out_files.metrics_pdf);

    fprintf('saved figure: %s\n', out_files.partition_png);
    fprintf('saved figure: %s\n', out_files.partition_pdf);
    fprintf('saved figure: %s\n', out_files.metrics_png);
    fprintf('saved figure: %s\n', out_files.metrics_pdf);
end

function draw_partition_maps(out_png, out_pdf)
    geom = halfspace_paper_geometry();
    modes = ["radial-current", "left-right-pml", "source-interface-pml"];
    titles = ["Current radial", "Left/right/PML", "Source+interface/PML"];
    rank_text = [
        "[4 6 6; 8 10 11; 22 23 23]"
        "[3 6 6; 6 11 11; 23 24 24]"
        "[4 5 6; 7 10 11; 19 23 21]"
    ];

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1320 430]);
    tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    colors = dd_partition_colormap(3);

    for i = 1:numel(modes)
        ax = nexttile(tl);
        [label_grid, names] = partition_label_grid(geom, modes(i));
        imagesc(ax, geom.x_int, geom.y_int, label_grid.');
        set(ax, 'YDir', 'normal');
        axis(ax, 'equal', 'tight');
        colormap(ax, colors);
        clim(ax, [0.5 3.5]);
        hold(ax, 'on');
        xline(ax, geom.interface_x, 'k-', 'LineWidth', 1.4);
        plot(ax, geom.source_x, geom.source_y, 'kp', ...
            'MarkerFaceColor', [0.35 0.35 0.35], 'MarkerSize', 11, 'LineWidth', 1.0);
        hold(ax, 'off');
        title(ax, titles(i), 'FontWeight', 'bold');
        xlabel(ax, 'x');
        if i == 1
            ylabel(ax, 'y');
        end
        set(ax, 'FontSize', 11, 'LineWidth', 1.0, 'Box', 'on');
        subtitle(ax, sprintf('%s ranks: %s', strjoin(names, '/'), rank_text(i)), ...
            'FontSize', 9);
    end

    export_figure(fig, out_png);
    export_figure(fig, out_pdf);
    close(fig);
end

function draw_metric_summary(T, out_png, out_pdf)
    labels = compact_variant_labels(T.variant);
    x = 1:height(T);
    colors = [
        0.20 0.20 0.20
        0.13 0.45 0.70
        0.90 0.62 0.12
        0.42 0.67 0.32
    ];

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1180 780]);
    tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax = nexttile(tl);
    plot_bars(ax, x, T.total_basis_entries / 1e6, labels, colors);
    ylabel(ax, 'basis entries (millions)');
    title(ax, 'Storage');

    ax = nexttile(tl);
    semilogy(ax, x, T.mean_precorr_relres, 'o-', 'Color', [0.1 0.1 0.1], ...
        'MarkerFaceColor', [0.85 0.05 0.05], 'LineWidth', 2.0, 'MarkerSize', 7);
    set_metric_axis(ax, labels);
    ylabel(ax, 'mean relative residual');
    title(ax, 'After TA-ROM pre-correction');

    ax = nexttile(tl);
    plot_bars(ax, x, T.mean_total_time, labels, colors);
    ylabel(ax, 'seconds');
    title(ax, 'Online time');

    ax = nexttile(tl);
    plot_bars(ax, x, T.mean_completion_iter, labels, colors);
    ylabel(ax, 'iterations');
    title(ax, 'WaveHoltz completion');

    export_figure(fig, out_png);
    export_figure(fig, out_pdf);
    close(fig);
end

function geom = halfspace_paper_geometry()
    geom.Lx = 1.0;
    geom.Ly = 1.0;
    geom.Nphys_x = 125;
    geom.Nphys_y = 125;
    geom.npml = ceil(0.25 * min(geom.Nphys_x, geom.Nphys_y));
    geom.nx = geom.Nphys_x + 2 * geom.npml;
    geom.ny = geom.Nphys_y + 2 * geom.npml;
    geom.hx = geom.Lx / (geom.Nphys_x + 1);
    geom.hy = geom.Ly / (geom.Nphys_y + 1);
    Lpml_x = geom.npml * geom.hx;
    Lpml_y = geom.npml * geom.hy;
    x = linspace(-Lpml_x, geom.Lx + Lpml_x, geom.nx + 2);
    y = linspace(-Lpml_y, geom.Ly + Lpml_y, geom.ny + 2);
    geom.x_int = x(2:end-1).';
    geom.y_int = y(2:end-1).';
    geom.I = (geom.npml + 1):(geom.npml + geom.Nphys_x);
    geom.J = (geom.npml + 1):(geom.npml + geom.Nphys_y);
    geom.interface_x = 0.5 * geom.Lx;
    geom.block3_center = [0.5 * geom.Lx, 0.5 * geom.Ly];
    geom.block3_r_near = 0.25 * geom.Lx;
    geom.source_col_phys = max(2, round(0.20 * geom.Nphys_x));
    geom.source_row_phys = round((geom.Nphys_y + 1) / 2);
    x_phys = x(geom.I + 1).';
    y_phys = y(geom.J + 1).';
    geom.source_x = x_phys(geom.source_col_phys);
    geom.source_y = y_phys(geom.source_row_phys);
end

function [label_grid, block_names] = partition_label_grid(geom, mode)
    [X, Y] = ndgrid(geom.x_int, geom.y_int);
    phys_mask = false(geom.nx, geom.ny);
    phys_mask(geom.I, geom.J) = true;
    pml_mask = ~phys_mask;

    switch mode
        case "radial-current"
            R = hypot(X - geom.block3_center(1), Y - geom.block3_center(2));
            b1 = phys_mask & (R <= geom.block3_r_near);
            b2 = phys_mask & ~b1;
            b3 = pml_mask;
            block_names = ["near", "far", "PML"];

        case "left-right-pml"
            b1 = phys_mask & (X < geom.interface_x);
            b2 = phys_mask & ~b1;
            b3 = pml_mask;
            block_names = ["left", "right", "PML"];

        case "source-interface-pml"
            source_radius = 4 * max(geom.hx, geom.hy);
            interface_halfwidth = max(0.10 * geom.Lx, 8 * geom.hx);
            source_mask = hypot(X - geom.source_x, Y - geom.source_y) <= source_radius;
            interface_mask = abs(X - geom.interface_x) <= interface_halfwidth;
            b1 = phys_mask & (source_mask | interface_mask);
            b2 = phys_mask & ~b1;
            b3 = pml_mask;
            block_names = ["active", "rest", "PML"];

        otherwise
            error('Unsupported partition mode: %s', mode);
    end

    label_grid = zeros(geom.nx, geom.ny);
    label_grid(b1) = 1;
    label_grid(b2) = 2;
    label_grid(b3) = 3;
end

function labels = compact_variant_labels(variants)
    labels = strings(size(variants));
    for i = 1:numel(variants)
        switch variants(i)
            case "global"
                labels(i) = "Global";
            case "radial-current"
                labels(i) = "Radial";
            case "left-right-pml"
                labels(i) = "Left/right/PML";
            case "source-interface-pml"
                labels(i) = "Source+interface";
            otherwise
                labels(i) = variants(i);
        end
    end
end

function plot_bars(ax, x, vals, labels, colors)
    b = bar(ax, x, vals, 0.72, 'FaceColor', 'flat', 'EdgeColor', 'none');
    for i = 1:numel(vals)
        b.CData(i, :) = colors(min(i, size(colors, 1)), :);
    end
    set_metric_axis(ax, labels);
end

function set_metric_axis(ax, labels)
    grid(ax, 'on');
    set(ax, 'XTick', 1:numel(labels), 'XTickLabel', labels, ...
        'XTickLabelRotation', 18, 'FontSize', 11, 'LineWidth', 1.0, 'Box', 'on');
    xlim(ax, [0.5, numel(labels) + 0.5]);
end

function export_figure(fig, out_file)
    try
        exportgraphics(fig, out_file, 'Resolution', 300);
    catch
        saveas(fig, out_file);
    end
end
