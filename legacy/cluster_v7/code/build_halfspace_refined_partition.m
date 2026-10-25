function part = build_halfspace_refined_partition(prob, cfg)
%BUILD_HALFSPACE_REFINED_PARTITION Material-aware physical and layered PML DD.
% All cuts are prescribed by geometry, independently of online parameters.
    if ~strcmp(cfg.material_model, 'halfspace')
        error('Refined half-space partitions require material_model=halfspace.');
    end
    physical_mode = option(cfg, 'halfspace_dd_physical_mode', 'slabs');
    sections = option(cfg, 'halfspace_dd_sections', 1);
    transverse = option(cfg, 'halfspace_dd_transverse_sections', 1);
    pml_mode = option(cfg, 'halfspace_dd_pml_mode', 'signed_depth');
    depth_edges = option(cfg, 'halfspace_dd_depth_edges', 0.5);
    radii = option(cfg, 'halfspace_dd_radii', [0.15, 0.3, 0.5]);
    validateattributes(sections, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(transverse, {'numeric'}, {'scalar','integer','positive'});
    if any(~isfinite(depth_edges)) || any(depth_edges <= 0 | depth_edges >= 1) || ...
            any(diff(depth_edges) <= 0)
        error('PML depth edges must be strictly increasing in (0,1).');
    end
    if any(~isfinite(radii)) || any(radii <= 0) || any(diff(radii) <= 0)
        error('Source radii must be positive and strictly increasing.');
    end
    [X, Y] = ndgrid(prob.x_int, prob.y_int);
    physical = false(prob.nx, prob.ny);
    physical(prob.I, prob.J) = true;
    material = 1 + (X >= cfg.interface_x);
    switch physical_mode
        case 'slabs'
            local_x = X / cfg.interface_x;
            right = material == 2;
            local_x(right) = (X(right) - cfg.interface_x) / (cfg.Lx - cfg.interface_x);
            axial = min(sections, max(1, 1 + floor(sections * local_x)));
            lateral = min(transverse, max(1, 1 + floor(transverse * Y / cfg.Ly)));
            labels = axial + sections * (lateral - 1 + transverse * (material - 1));
        case 'source_rings'
            sx = prob.x_phys(cfg.source_col_phys);
            sy = prob.y_phys(cfg.source_row_phys);
            distance = hypot(X - sx, Y - sy);
            ring = ones(size(X));
            for radius = radii(:).'
                ring = ring + (distance > radius);
            end
            labels = ring + (numel(radii) + 1) * (material - 1);
        otherwise
            error('halfspace_dd_physical_mode must be slabs or source_rings.');
    end
    labels(~physical) = 0;
    max_physical = max(labels(:));
    if prob.npml > 0
        [ii, jj] = ndgrid(1:prob.nx, 1:prob.ny);
        depths = cat(3, max(prob.npml + 1 - ii, 0), ...
            max(ii - (prob.nx - prob.npml), 0), ...
            max(prob.npml + 1 - jj, 0), ...
            max(jj - (prob.ny - prob.npml), 0)) / prob.npml;
        [depth, face] = max(depths, [], 3);
        switch pml_mode
            case 'merged'
                pml_label = ones(size(X));
            case 'signed'
                pml_label = face;
            case {'signed_depth','material_signed_depth','material_symmetric_depth'}
                face_count = 4;
                if strcmp(pml_mode, 'material_signed_depth')
                    horizontal = face >= 3;
                    face(horizontal) = face(horizontal) + 2 * (material(horizontal)-1);
                    face_count = 6;
                elseif strcmp(pml_mode, 'material_symmetric_depth')
                    horizontal = face >= 3;
                    face(horizontal) = 2 + material(horizontal);
                end
                shell = ones(size(X));
                for edge = depth_edges(:).'
                    shell = shell + (depth > edge);
                end
                pml_label = face + face_count * (shell - 1);
            otherwise
                error('Unknown halfspace_dd_pml_mode: %s.', pml_mode);
        end
        labels(~physical) = max_physical + pml_label(~physical);
    end
    ids = unique(labels(:));
    assert(all(ids > 0), 'Partition must cover every grid point.');
    blocks = repmat(struct('name', '', 'idx', [], 'count', 0), numel(ids), 1);
    compact_labels = zeros(size(labels), 'uint16');
    for b = 1:numel(ids)
        idx = find(labels(:) == ids(b));
        prefix = 'physical';
        if ids(b) > max_physical
            prefix = 'pml';
        end
        blocks(b).name = sprintf('%s_%d', prefix, ids(b));
        blocks(b).idx = idx;
        blocks(b).count = numel(idx);
        compact_labels(idx) = b;
    end
    part = struct('n', prob.Ntot, 'nx', prob.nx, 'ny', prob.ny, ...
        'mode', 'halfspace_refined', 'blocks', blocks, 'labels', compact_labels);
    assert(sum([blocks.count]) == prob.Ntot);
end

function value = option(cfg, name, fallback)
    value = fallback;
    if isfield(cfg, name)
        value = cfg.(name);
    end
end
