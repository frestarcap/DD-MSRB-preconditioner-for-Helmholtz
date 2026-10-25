function files = waveguide_geometry(data_dir, out_dir)
%WAVEGUIDE_GEOMETRY Draw the 3D waveguide profile and partition renderings.
%   files = waveguide_geometry(data_dir, out_dir) reads
%   <data_dir>/exp3/waveguide_geometry.mat and writes waveguide_profile.png and
%   waveguide_partition.png to out_dir (default: build/figures).
    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(data_dir), data_dir = fullfile(here, '..', 'paper_data'); end
    if nargin < 2 || isempty(out_dir), out_dir = fullfile(here, '..', 'build', 'figures'); end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    s = load(fullfile(data_dir, 'exp3', 'waveguide_geometry.mat'), 'geometry');
    g = s.geometry;
    files.profile = fullfile(out_dir, 'waveguide_profile.png');
    files.partition = fullfile(out_dir, 'waveguide_partition.png');
    draw_profile(g, files.profile);
    draw_partition(g, files.partition);
end

function draw_profile(g, filename)
% Isosurfaces of the transverse Gaussian profile g(x,y,z) at 0.1, 0.5, 0.8.
    fig = new_figure();
    ax = axes(fig);
    hold(ax, 'on');
    coordinates = cell(1, 3);
    for d = 1:3
        coordinates{d} = linspace(0, g.domain_lengths(d), g.Nphys(d) + 2);
    end
    [X, Y, Z] = meshgrid(coordinates{:});
    profile = exp(-0.5 * (((Y - g.waveguide_center(2)) / g.waveguide_width(1)).^2 + ...
        ((Z - g.waveguide_center(3)) / g.waveguide_width(2)).^2));
    levels = [0.1, 0.5, 0.8];
    opacity = [0.16, 0.30, 0.85];
    cmap = parula(256);
    for j = 1:numel(levels)
        color = cmap(1 + round(255 * levels(j)), :);
        surface = isosurface(X, Y, Z, profile, levels(j));
        assert(~isempty(surface.vertices), 'Empty material isosurface.');
        patch(ax, surface, 'FaceColor', color, 'EdgeColor', 'none', ...
            'FaceAlpha', opacity(j), 'FaceLighting', 'gouraud');
        caps = isocaps(X, Y, Z, profile, levels(j));
        patch(ax, caps, 'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', opacity(j));
    end
    draw_box(ax, zeros(1, 3), g.domain_lengths, 'k-', 1.5);
    plot3(ax, g.source_location(1), g.source_location(2), g.source_location(3), 'p', ...
        'Color', [0.85 0.05 0.05], 'MarkerFaceColor', [0.85 0.05 0.05], ...
        'MarkerEdgeColor', 'w', 'MarkerSize', 16, 'LineWidth', 1.2);
    style_3d_axes(ax, g.domain_lengths, zeros(1, 3), g.domain_lengths);
    colormap(ax, cmap);
    clim(ax, [0, 1]);
    cb = colorbar(ax, 'Ticks', [0, 0.1, 0.5, 0.8, 1]);
    style_colorbar(cb, 'profile g(x,y,z)');
    position_3d_axes(ax, cb);
    ax.SortMethod = 'childorder';
    camlight(ax, 'headlight');
    save_figure(fig, filename);
end

function draw_partition(g, filename)
% Subdomain classes on three orthogonal cuts; dashed lines mark axial sections.
    fig = new_figure();
    ax = axes(fig);
    hold(ax, 'on');
    colors = [0.92 0.96 0.86; 0.72 0.87 0.63; 0.42 0.72 0.52; ...
        0.16 0.52 0.46; 0.05 0.31 0.37; 0.73 0.76 0.80; 0.43 0.46 0.52];
    cut = g.slice_index;
    cut(1) = g.grid_size(1) - g.pml_cells(1);
    draw_orthogonal_slices(ax, g.coordinates, g.partition_classes, cut, colors);
    center = cellfun(@(x, j) x(j), g.coordinates, num2cell(g.slice_index));
    offset = 0.02 * min(g.spacing);
    for x = g.axial_interfaces(:)'
        plot3(ax, [x x], [0 g.domain_lengths(2)], [center(3) center(3)] + offset, ...
            'k--', 'LineWidth', 1.8);
        plot3(ax, [x x], [center(2) center(2)] - offset, [0 g.domain_lengths(3)], ...
            'k--', 'LineWidth', 1.8);
    end
    lower = -g.pml_cells .* g.spacing;
    upper = g.domain_lengths - lower;
    draw_box(ax, lower, upper, ':', 1.2);
    draw_box(ax, zeros(1, 3), g.domain_lengths, 'k-', 1.5);
    style_3d_axes(ax, g.domain_lengths, lower, upper);
    colormap(ax, colors);
    clim(ax, [0.5, 7.5]);
    cb = colorbar(ax, 'Ticks', 1:7, 'TickLabels', ...
        {'band 1', 'band 2', 'band 3', 'band 4', 'band 5', 'inner PML', 'outer PML'});
    style_colorbar(cb, '');
    position_3d_axes(ax, cb);
    save_figure(fig, filename);
end

function draw_orthogonal_slices(ax, coordinates, values, indices, colors)
    x = coordinates{1}; y = coordinates{2}; z = coordinates{3};
    [Y, Z] = ndgrid(y, z);
    surface(ax, x(indices(1)) * ones(size(Y)), Y, Z, ...
        ind2rgb(double(squeeze(values(indices(1), :, :))), colors), ...
        'FaceColor', 'texturemap', 'EdgeColor', 'none');
    [X, Z] = ndgrid(x, z);
    surface(ax, X, y(indices(2)) * ones(size(X)), Z, ...
        ind2rgb(double(squeeze(values(:, indices(2), :))), colors), ...
        'FaceColor', 'texturemap', 'EdgeColor', 'none');
    [X, Y] = ndgrid(x, y);
    surface(ax, X, Y, z(indices(3)) * ones(size(X)), ...
        ind2rgb(double(values(:, :, indices(3))), colors), ...
        'FaceColor', 'texturemap', 'EdgeColor', 'none');
end

function draw_box(ax, lower, upper, line_style, width)
    corners = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1];
    corners = lower + corners .* (upper - lower);
    edges = [1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; 1 5; 2 6; 3 7; 4 8];
    for j = 1:size(edges, 1)
        points = corners(edges(j, :), :);
        plot3(ax, points(:, 1), points(:, 2), points(:, 3), line_style, ...
            'Color', [0.25 0.25 0.25], 'LineWidth', width);
    end
end

function style_3d_axes(ax, lengths, lower, upper)
    axis(ax, 'equal');
    xlim(ax, [lower(1), upper(1)]);
    ylim(ax, [lower(2), upper(2)]);
    zlim(ax, [lower(3), upper(3)]);
    view(ax, [-35, 24]);
    set(ax, 'FontSize', 24, 'LineWidth', 1.6, 'Box', 'on', ...
        'XTick', [lengths(1) / 2, lengths(1)], ...
        'YTick', [0, lengths(2) / 2, lengths(2)], ...
        'ZTick', [0, lengths(3) / 2, lengths(3)], ...
        'XTickLabelRotation', 0, 'YTickLabelRotation', 0, 'ZTickLabelRotation', 0);
    xlabel(ax, 'x', 'FontSize', 26);
    ylabel(ax, 'y', 'FontSize', 26);
    zlabel(ax, 'z', 'FontSize', 26);
end

function position_3d_axes(ax, cb)
    ax.Position = [0.14, 0.15, 0.59, 0.73];
    cb.Position = [0.79, 0.19, 0.025, 0.63];
    ax.ZLabel.Units = 'normalized';
    ax.ZLabel.Position = [-0.10, 0.5, 0];
end

function style_colorbar(cb, label)
    cb.FontSize = 22;
    if ~isempty(label)
        ylabel(cb, label, 'FontSize', 24);
    end
end

function fig = new_figure()
    fig = figure('Color', 'w', 'Position', [80 80 980 720], 'Visible', 'off');
end

function save_figure(fig, filename)
    exportgraphics(fig, filename, 'Resolution', 300);
    close(fig);
    fprintf('Wrote %s\n', filename);
end
