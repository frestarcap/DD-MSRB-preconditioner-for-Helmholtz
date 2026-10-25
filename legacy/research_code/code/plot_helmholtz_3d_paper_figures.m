function files = plot_helmholtz_3d_paper_figures(cfg)
%PLOT_HELMHOLTZ_3D_PAPER_FIGURES Redraw each 3D experiment from saved MATs.
% The first redraw extracts a compact field/partition cache from full results.
% Subsequent redraws load that cache and paper_plot_data, without the bases.

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    root = helm_project_root();
    % Publication-layout export uses the existing compact geometry cache only.
    if isfield(cfg, 'target') && strcmp(cfg.target, 'waveguide_geometry')
        cfg = set_default(cfg, 'resolution', 300);
        cfg = set_default(cfg, 'show_titles', false);
        cfg = set_default(cfg, 'output_dir', fullfile(root, 'JCP_CAS', 'figures', 'layout_20260925'));
        ensure_directory(cfg.output_dir);
        saved = load(fullfile(root, 'outputs', 'helmholtz_3d_paper_figures', ...
            'helmholtz_3d_waveguide_visual_data.mat'), 'visual');
        files.profile = fullfile(cfg.output_dir, 'waveguide_profile.png');
        files.partition = fullfile(cfg.output_dir, 'waveguide_partition.png');
        draw_profile(saved.visual, 'waveguide', files.profile, cfg);
        draw_partition(saved.visual, files.partition, cfg);
        return;
    end
    cfg = set_default(cfg, 'show_titles', true);
    cfg = set_default(cfg, 'copy_to_paper', true);
    cfg = set_default(cfg, 'resolution', 300);
    cfg = set_default(cfg, 'representative_test_index', 1);
    cfg = set_default(cfg, 'marker_count', 18);
    cfg = set_default(cfg, 'target', 'all');
    cfg.target = validatestring(cfg.target, {'all','solutions','waveguide_solution'});
    solution_only = ~strcmp(cfg.target,'all');
    cfg = set_default(cfg, 'data_dir', fullfile(root, ...
        'outputs', 'helmholtz_3d_candidates', 'large'));
    cfg = set_default(cfg, 'output_dir', fullfile(root, ...
        'outputs', 'helmholtz_3d_paper_figures'));
    validateattributes(cfg.marker_count, {'numeric'}, {'scalar', 'integer', 'positive'});
    if cfg.copy_to_paper && cfg.representative_test_index ~= 1
        error('Use copy_to_paper=false for other parameters; paper captions describe test 1.');
    end
    paper_dir = fullfile(root, 'LaTeX_DL_468198_240419', 'figures', 'helmholtz_3d');
    ensure_directory(cfg.output_dir);
    if cfg.copy_to_paper
        ensure_directory(paper_dir);
    end

    cfg=set_default(cfg,'lens_tag','jsc_20260907_lens_tight');
    definitions = {'lens', cfg.lens_tag; ...
        'waveguide', 'waveguide_dd_memory_saving_41pct_reproduction'};
    if strcmp(cfg.target,'waveguide_solution'), definitions = definitions(2,:); end
    case_files = cell(size(definitions,1), 1);
    summaries = cell(size(definitions,1), 1);
    diagnostics = cell(size(definitions,1), 1);
    for k = 1:size(definitions, 1)
        name = definitions{k, 1};
        result_file = fullfile(cfg.data_dir, sprintf( ...
            'helmholtz_3d_candidates_%s.mat', definitions{k, 2}));
        require_file(result_file);
        variables = whos('-file', result_file);
        if any(strcmp({variables.name}, 'paper_plot_data'))
            saved = load(result_file, 'paper_plot_data');
            plot_data = saved.paper_plot_data;
        else
            saved = load(result_file, 'results');
            plot_data = saved.results;
        end
        candidate = plot_data.candidates{1};
        test_index = cfg.representative_test_index;
        assert(isscalar(test_index) && test_index == fix(test_index) && ...
            test_index >= 1 && test_index <= numel(candidate.tests), ...
            'Invalid representative_test_index.');
        cache_file = fullfile(cfg.output_dir, ...
            sprintf('helmholtz_3d_%s_visual_data.mat', name));
        visual = load_visual_data(result_file, cache_file, test_index);
        prefix = fullfile(cfg.output_dir, ['helmholtz_3d_' name]);
        one = struct('name', name, 'profile', [prefix '_profile.png'], ...
            'partition', [prefix '_dd_partition.png'], ...
            'solution', [prefix '_solution.png'], ...
            'reference', [prefix '_reference.png'], ...
            'solution_difference', [prefix '_solution_difference.png'], ...
            'residual', [prefix '_residual_histories.png'], ...
            'visual_data', cache_file);
        method = 'dd';
        if strcmp(name,'waveguide'), method = 'dd_complex'; end
        draw_solution(visual, method, one.solution, cfg);
        if strcmp(cfg.target,'solutions')
            draw_solution(visual, 'reference', one.reference, cfg);
        end
        if ~solution_only
            draw_profile(visual, name, one.profile, cfg);
            draw_partition(visual, one.partition, cfg);
            draw_solution(visual, 'reference', one.reference, cfg);
            draw_solution(visual, 'difference', one.solution_difference, cfg);
            draw_residual(candidate.tests(test_index), name, one.residual, cfg);
        end
        if cfg.copy_to_paper
            keys = {'profile', 'partition', 'solution', 'residual'};
            if solution_only, keys = {'solution'}; end
            for key = keys
                [~, stem, extension] = fileparts(one.(key{1}));
                copyfile(one.(key{1}), fullfile(paper_dir, [stem extension]));
            end
        end
        case_files{k} = one;
        summaries{k} = candidate.summary_table;
        diagnostics{k} = struct('candidate', string(name), 'mu', visual.mu, ...
            'relative_solution_difference', visual.relative_solution_difference, ...
            'dd_final_residual', candidate.tests(test_index).dd.relative_residual, ...
            'reference_final_residual', candidate.tests(test_index).shifted.relative_residual, ...
            'slice_z', visual.coordinates{3}(visual.slice_index(3)));
        fprintf('%s: saved-field relative difference %.6e\n', ...
            name, visual.relative_solution_difference);
    end
    if solution_only
        files = struct('cases',{case_files},'paper_directory',paper_dir);
        return;
    end
    online_file = fullfile(cfg.output_dir, 'helmholtz_3d_online_results.csv');
    online_table = vertcat(summaries{:});
    writetable(online_table(isfinite(online_table.total_s), :), online_file);
    verification_file = fullfile(cfg.output_dir, 'helmholtz_3d_field_comparison.csv');
    writetable(struct2table(vertcat(diagnostics{:})), verification_file);
    files = struct('cases', {case_files}, 'online_table', online_file, ...
        'field_comparison', verification_file, 'paper_directory', paper_dir);
end

function visual = load_visual_data(result_file, cache_file, test_index)
    source = dir(result_file);
    provenance = struct('schema_version', 2, 'source_file', result_file, ...
        'source_bytes', source.bytes, 'source_datenum', source.datenum, ...
        'test_index', test_index);
    if isfile(cache_file)
        cached = load(cache_file, 'visual');
        if isfield(cached, 'visual') && ...
                isequaln(cached.visual.provenance, provenance)
            visual = cached.visual;
            fprintf('Loaded compact visual cache: %s\n', cache_file);
            return;
        end
    end
    saved = load(result_file, 'results');
    candidate = saved.results.candidates{1};
    if ~isfield(candidate, 'partition') || ...
            ~isfield(candidate.tests(test_index).dd, 'solution')
        error(['Full saved fields and DD indices are required for the new 3D ' ...
            'figures. Reproduce this candidate with save_models=true: %s'], result_file);
    end
    visual = struct('provenance', provenance, 'problem_cfg', candidate.problem_cfg, ...
        'coordinates', {candidate.geometry.coordinates}, ...
        'grid_size', candidate.geometry.grid_size, ...
        'pml_cells', candidate.geometry.pml_cells, ...
        'spacing', candidate.geometry.spacing, ...
        'mu', candidate.tests(test_index).mu);
    n = visual.grid_size;
    visual.slice_index = zeros(1, 3);
    for d = 1:3
        [~, visual.slice_index(d)] = min(abs(visual.coordinates{d} - ...
            candidate.problem_cfg.domain_lengths(d) / 2));
    end

    % Color classes group the actual saved blocks; no partition is regenerated.
    classes = zeros(prod(n), 1, 'uint8');
    section_x = NaN(numel(candidate.partition.blocks), 1);
    for b = 1:numel(candidate.partition.blocks)
        block = candidate.partition.blocks(b);
        band = sscanf(block.name, 'profile_band_%d_s%d');
        if ~isempty(band)
            classes(block.idx) = band(1);
            [ix, ~, ~] = ind2sub(n, block.idx);
            section_x(b) = min(ix);
        elseif endsWith(block.name, '_inner')
            classes(block.idx) = 6;
        elseif endsWith(block.name, '_outer')
            classes(block.idx) = 7;
        else
            error('Unsupported saved DD block for profile-band plots: %s', block.name);
        end
    end
    assert(all(classes > 0), 'Unclassified saved DD indices.');
    visual.partition_classes = reshape(classes, n);
    % Axial interfaces are recovered from the lowest-energy band, which spans x.
    first_band = startsWith({candidate.partition.blocks.name}, 'profile_band_1_');
    starts = unique(section_x(first_band & isfinite(section_x).'));
    visual.axial_interfaces = visual.coordinates{1}(starts(2:end)) - visual.spacing(1) / 2;
    field = candidate.tests(test_index).dd.solution;
    reference = candidate.tests(test_index).shifted.solution;
    assert(numel(field) == prod(n) && numel(reference) == prod(n));
    visual.relative_solution_difference = norm(field - reference) / norm(reference);
    iz = visual.slice_index(3);
    field = reshape(field, n);
    reference = reshape(reference, n);
    visual.solution_slice = real(field(:, :, iz));
    visual.solution_imag_slice = imag(field(:, :, iz));
    visual.reference_slice = real(reference(:, :, iz));
    visual.reference_imag_slice = imag(reference(:, :, iz));
    visual.difference_slice = abs(field(:, :, iz) - reference(:, :, iz));
    visual.solution_color_limit = max(abs([visual.solution_slice(:); ...
        visual.reference_slice(:)]));
    visual.solution_complex_color_limit = max(abs([visual.solution_slice(:); ...
        visual.solution_imag_slice(:);visual.reference_slice(:); ...
        visual.reference_imag_slice(:)]));
    save(cache_file, 'visual', '-v7.3');
    fprintf('Extracted saved fields and DD indices: %s\n', cache_file);
end

function draw_profile(v, name, filename, cfg)
    fig = new_figure();
    ax = axes(fig);
    hold(ax, 'on');
    p = v.problem_cfg;
    physical_coordinates = cell(1, 3);
    for d = 1:3
        physical_coordinates{d} = linspace(0, p.domain_lengths(d), p.Nphys(d) + 2);
    end
    [X, Y, Z] = meshgrid(physical_coordinates{:});
    if strcmp(name, 'lens')
        center = p.gaussian_center;
        width = p.gaussian_width;
        profile = exp(-0.5 * (((X - center(1)) / width(1)).^2 + ...
            ((Y - center(2)) / width(2)).^2 + ((Z - center(3)) / width(3)).^2));
    else
        center = p.waveguide_center;
        width = p.waveguide_width;
        profile = exp(-0.5 * (((Y - center(2)) / width(1)).^2 + ...
            ((Z - center(3)) / width(2)).^2));
    end
    levels = [0.1, 0.5, 0.8];
    opacity = [0.16, 0.30, 0.85];
    cmap = parula(256);
    for j = 1:numel(levels)
        color = cmap(1 + round(255 * levels(j)), :);
        surface = isosurface(X, Y, Z, profile, levels(j));
        assert(~isempty(surface.vertices), 'Empty material isosurface.');
        patch(ax, surface, 'FaceColor', color, 'EdgeColor', 'none', ...
            'FaceAlpha', opacity(j), 'FaceLighting', 'gouraud');
        if strcmp(name, 'waveguide')
            caps = isocaps(X, Y, Z, profile, levels(j));
            patch(ax, caps, 'FaceColor', color, 'EdgeColor', 'none', ...
                'FaceAlpha', opacity(j));
        end
    end
    draw_box(ax, zeros(1, 3), p.domain_lengths, 'k-', 1.5);
    plot_source(ax, p.source_location);
    style_3d_axes(ax, p.domain_lengths, zeros(1, 3), p.domain_lengths);
    if cfg.show_titles, title(ax, sprintf('Gaussian %s profile', name)); end
    colormap(ax, cmap);
    clim(ax, [0, 1]);
    cb = colorbar(ax, 'Ticks', [0, 0.1, 0.5, 0.8, 1]);
    style_colorbar(cb, 'profile g(x,y,z)');
    position_3d_axes(ax, cb);
    ax.SortMethod = 'childorder';
    camlight(ax, 'headlight');
    save_figure(fig, filename, cfg.resolution);
end

function draw_partition(v, filename, cfg)
    fig = new_figure();
    ax = axes(fig);
    hold(ax, 'on');
    colors = [0.92 0.96 0.86; 0.72 0.87 0.63; 0.42 0.72 0.52; ...
        0.16 0.52 0.46; 0.05 0.31 0.37; 0.73 0.76 0.80; 0.43 0.46 0.52];
    cut_indices = v.slice_index;
    cut_indices(1) = v.grid_size(1) - v.pml_cells(1);
    draw_orthogonal_slices(ax, v.coordinates, v.partition_classes, ...
        cut_indices, colors);
    p = v.problem_cfg;
    center = cellfun(@(x, j) x(j), v.coordinates, num2cell(v.slice_index));
    offset = 0.02 * min(v.spacing);
    for x = v.axial_interfaces
        plot3(ax, [x x], [0 p.domain_lengths(2)], [center(3) center(3)] + offset, ...
            'k--', 'LineWidth', 1.8);
        plot3(ax, [x x], [center(2) center(2)] - offset, [0 p.domain_lengths(3)], ...
            'k--', 'LineWidth', 1.8);
    end
    lower = -v.pml_cells .* v.spacing;
    upper = p.domain_lengths - lower;
    draw_box(ax, lower, upper, ':', 1.2);
    draw_box(ax, zeros(1, 3), p.domain_lengths, 'k-', 1.5);
    style_3d_axes(ax, p.domain_lengths, lower, upper);
    if cfg.show_titles, title(ax, 'DD partition'); end
    colormap(ax, colors);
    clim(ax, [0.5, 7.5]);
    cb = colorbar(ax, 'Ticks', 1:7, 'TickLabels', ...
        {'band 1', 'band 2', 'band 3', 'band 4', 'band 5', 'inner PML', 'outer PML'});
    style_colorbar(cb, '');
    position_3d_axes(ax, cb);
    save_figure(fig, filename, cfg.resolution);
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

function draw_solution(v, method, filename, cfg)
    fig = new_figure();
    if strcmp(method,'dd_complex')
        fig.Position(4) = 840;
        tiles = tiledlayout(fig,2,1,'Padding','compact','TileSpacing','compact');
        values = {v.solution_slice,v.solution_imag_slice};
        headings = {'Real part','Imaginary part'};
        color_labels = {'real(u)','imag(u)'};
        limit = v.solution_complex_color_limit;
        for j = 1:2
            ax = nexttile(tiles);
            draw_solution_panel(ax,v,values{j},headings{j},color_labels{j},[-limit limit]);
            if j == 1
                ax.XTickLabel = [];
                ax.XLabel.String = '';
            end
        end
        save_figure(fig,filename,cfg.resolution);
        return;
    end
    ax = axes(fig);
    switch method
        case 'dd'
            values = v.solution_slice;
            heading = 'DD-TA-ROM real part';
        case 'reference'
            values = v.reference_slice;
            heading = 'Shifted-MG reference real part';
        otherwise
            values = v.difference_slice;
            heading = 'Difference from shifted-MG reference';
    end
    if strcmp(method,'difference')
        limits = [0,max(values(:))];
        color_label = '|u_{DD}-u_{ref}|';
    else
        limits = [-v.solution_color_limit,v.solution_color_limit];
        color_label = 'real(u)';
    end
    draw_solution_panel(ax,v,values,heading,color_label,limits);
    save_figure(fig, filename, cfg.resolution);
end

function draw_solution_panel(ax,v,values,heading,color_label,limits)
    imagesc(ax, v.coordinates{1}, v.coordinates{2}, values.');
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal');
    axis(ax, 'tight');
    hold(ax, 'on');
    p = v.problem_cfg;
    rectangle(ax, 'Position', [0, 0, p.domain_lengths(1:2)], ...
        'EdgeColor', [0.25 0.25 0.25], 'LineStyle', '--', 'LineWidth', 1.5);
    colormap(ax, parula(256));
    if any(strcmp(color_label, {'real(u)','imag(u)'}))
        colormap(ax, helm_solution_colormap(limits));
    end
    cb = colorbar(ax);
    clim(ax,limits);
    style_colorbar(cb,color_label);
    set(ax, 'FontSize', 24, 'LineWidth', 1.6, 'Box', 'on');
    xlabel(ax, 'x', 'FontSize', 26);
    ylabel(ax, 'y', 'FontSize', 26);
    title(ax, heading, 'FontSize', 29);
end

function draw_residual(test, name, filename, cfg)
    fig = new_figure();
    ax = axes(fig);
    hold(ax, 'on');
    colors = [0.85 0.05 0.05; 0.05 0.45 0.25; 0 0 0];
    labels = {'Global TA-ROM', 'DD-TA-ROM', 'Shifted-Laplacian MG-GMRES'};
    markers = {'o', 's', 'o'};
    curves = {combined_history(test.global), combined_history(test.dd), ...
        test.shifted.residual_history(:)};
    for j = 1:numel(curves)
        curve = curves{j};
        indices = unique([1:min(3, numel(curve)), ...
            round(linspace(1, numel(curve), min(cfg.marker_count, numel(curve))))]);
        semilogy(ax, 0:numel(curve) - 1, max(curve, 1e-16), ...
            'Color', colors(j, :), 'LineWidth', 3.2, 'Marker', markers{j}, ...
            'MarkerIndices', indices, 'MarkerSize', 9, 'MarkerFaceColor', 'w', ...
            'DisplayName', labels{j});
    end
    yline(ax, 1e-6, ':', 'Color', [0.35 0.35 0.35], ...
        'LineWidth', 1.8, 'HandleVisibility', 'off');
    set(ax, 'FontSize', 25, 'LineWidth', 1.8, 'Box', 'on', ...
        'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.28, 'MinorGridAlpha', 0.18, ...
        'XMinorGrid', 'off', 'YMinorGrid', 'on');
    ylim(ax, [5e-7, 2]);
    last_step = max(cellfun(@numel, curves)) - 1;
    xlim(ax, [-0.02 * last_step, 1.04 * last_step]);
    xlabel(ax, 'online step index', 'FontSize', 28);
    ylabel(ax, 'relative residual', 'FontSize', 28);
    title(ax, sprintf('%s residual, \\mu=%.3f', ...
        [upper(name(1)) name(2:end)], test.mu), 'FontSize', 30);
    lgd = legend(ax, 'Location', 'northeast');
    set(lgd, 'FontSize', 21, 'Box', 'on', 'LineWidth', 1.1);
    save_figure(fig, filename, cfg.resolution);
end

function curve = combined_history(result)
    precorrection = result.precorrection_residual_history(:);
    completion = result.residual_history(:);
    curve = [precorrection; completion(2:end)];
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

function plot_source(ax, location)
    plot3(ax, location(1), location(2), location(3), 'p', ...
        'Color', [0.85 0.05 0.05], 'MarkerFaceColor', [0.85 0.05 0.05], ...
        'MarkerEdgeColor', 'w', 'MarkerSize', 16, 'LineWidth', 1.2);
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
    ax.Title.FontSize = 29;
    ax.Title.FontWeight = 'bold';
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

function save_figure(fig, filename, resolution)
    exportgraphics(fig, filename, 'Resolution', resolution);
    close(fig);
    fprintf('Redrew %s\n', filename);
end

function require_file(filename)
    if ~isfile(filename)
        error('Required result file not found: %s', filename);
    end
end

function ensure_directory(pathname)
    if ~exist(pathname, 'dir')
        mkdir(pathname);
    end
end

function s = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end
