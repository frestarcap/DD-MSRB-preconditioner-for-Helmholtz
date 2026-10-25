function out_file = plot_pc_four_block_dd_partition(output_dir)
%PLOT_PC_FOUR_BLOCK_DD_PARTITION
% Draw the photonic-crystal four-block DD partition used in Experiment 3
% (channel core / physical cladding / left--right PML / top--bottom-corner
% PML), using the same axes and colorbar style as the ECC partition figure.
%
% The partition mask is synthesized from the manuscript parameters:
%   9-column x 7-row lattice with a = 1, 20 grid points per lattice cell
%   (Nx = 180, Ny = 140), PML thickness npml = 40, channel half-width
%   1.5 a about the mid-row y = Ly/2.
%
% Usage:
%   plot_pc_four_block_dd_partition();
%   plot_pc_four_block_dd_partition('/path/to/out_dir');

    if nargin < 1 || isempty(output_dir)
        output_dir = helm_output_path('outputs', 'figures', ...
            'photonic_crystal_correction_rom_4block');
    end
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    a = 1.0;
    pc_num_cols = 9;
    pc_num_rows = 3;
    Lx = pc_num_cols * a;
    Ly = (2 * pc_num_rows + 1) * a;
    points_per_a = 20;
    Nx = pc_num_cols * points_per_a;
    Ny = (2 * pc_num_rows + 1) * points_per_a;
    npml = 2 * points_per_a;
    channel_halfwidth = 1.5 * a;

    nx = Nx + 2 * npml;
    ny = Ny + 2 * npml;
    hx = Lx / (Nx + 1);
    hy = Ly / (Ny + 1);
    x_int = (-npml + (1:nx) - 0.5) * hx;
    y_int = (-npml + (1:ny) - 0.5) * hy;

    phys_i = (npml + 1):(npml + Nx);
    phys_j = (npml + 1):(npml + Ny);
    phys_mask = false(nx, ny);
    phys_mask(phys_i, phys_j) = true;

    ii = repmat((1:nx).', 1, ny);
    jj = repmat(1:ny, nx, 1);
    [~, Y] = ndgrid(x_int, y_int);

    core_mask     = phys_mask & (abs(Y - 0.5 * Ly) <= channel_halfwidth);
    cladding_mask = phys_mask & ~core_mask;
    pml_mask      = ~phys_mask;
    lr_pml_mask   = pml_mask & ((ii <= npml) | (ii > nx - npml)) & ...
                    (jj > npml) & (jj <= ny - npml);
    tb_corner_pml = pml_mask & ~lr_pml_mask;

    img = zeros(nx, ny);
    img(core_mask)     = 1;
    img(cladding_mask) = 2;
    img(lr_pml_mask)   = 3;
    img(tb_corner_pml) = 4;

    fig = figure('Color', 'w', 'Position', [80 80 1100 720], 'Visible', 'off');
    ax = axes(fig);
    imagesc(ax, x_int, y_int, img.');
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal');
    axis(ax, 'tight');
    colormap(ax, dd_partition_colormap(4));
    clim(ax, [1, 4]);
    set(ax, 'FontSize', 24, 'LineWidth', 1.6, 'Box', 'on');
    title(ax, 'Four-block DD partition');
    ax.Title.FontSize = 29;
    ax.Title.FontWeight = 'bold';
    xlabel(ax, 'x');
    ylabel(ax, 'y');
    ax.XLabel.FontSize = 26;
    ax.YLabel.FontSize = 26;

    hold(ax, 'on');
    y_core_upper = 0.5 * Ly + channel_halfwidth;
    y_core_lower = 0.5 * Ly - channel_halfwidth;
    plot(ax, [0, Lx], [y_core_upper, y_core_upper], 'k--', 'LineWidth', 1.1);
    plot(ax, [0, Lx], [y_core_lower, y_core_lower], 'k--', 'LineWidth', 1.1);
    plot(ax, [0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], 'k-', 'LineWidth', 1.2);
    hold(ax, 'off');

    cb = colorbar(ax, 'Ticks', [1, 2, 3, 4], ...
        'TickLabels', {'core', 'cladding', 'LR PML', 'TB/corner PML'});
    cb.FontSize = 22;
    ax.Position = [0.10 0.12 0.60 0.78];
    cb.Position = [0.72 0.12 0.022 0.78];

    out_file = fullfile(output_dir, 'pc_four_block_dd_partition.png');
    exportgraphics(fig, out_file, 'Resolution', 300);
    fprintf('Saved photonic-crystal four-block DD partition: %s\n', out_file);
    close(fig);
end
