function part = partition(P, mode)
%PARTITION Non-overlapping subdomains of the half-space grid (PML included).
%   part = dmsrb.halfspace.partition(P, mode) with mode
%     'material_depth' (default): the two materials of the physical domain, and
%         the PML split into left, right, and lower/upper faces per material, each
%         into an inner and an outer half of the PML depth (10 subdomains);
%     'merged':        the two materials and one PML subdomain (3);
%     'signed_depth':  the two materials and the PML split by face and depth (10);
%     'uniform4x4':    4-by-4 tiles of the extended grid (16).
%   part.blocks(b).idx lists the grid indices of subdomain b; part.labels holds
%   the subdomain number of every node.  Numbering follows the paper figures.
    if nargin < 2, mode = 'material_depth'; end
    if strcmp(mode, 'uniform4x4')
        [i, j] = ndgrid(1:P.nx, 1:P.ny);
        labels = 1 + min(3, floor(4 * (i - 1) / P.nx)) + 4 * min(3, floor(4 * (j - 1) / P.ny));
        part = from_labels(labels, 16, mode);
        return;
    end
    [X, ~] = ndgrid(P.x_int, P.y_int);
    material = 1 + (X >= P.interface_x);
    physical = false(P.nx, P.ny);
    physical(P.I, P.J) = true;
    labels = material;
    labels(~physical) = 0;
    [ii, jj] = ndgrid(1:P.nx, 1:P.ny);
    depths = cat(3, max(P.npml + 1 - ii, 0), max(ii - (P.nx - P.npml), 0), ...
        max(P.npml + 1 - jj, 0), max(jj - (P.ny - P.npml), 0)) / P.npml;
    [depth, face] = max(depths, [], 3);           % faces: left, right, lower, upper
    switch mode
        case 'merged'
            pml_label = ones(size(X));
        case {'material_depth', 'signed_depth'}
            if strcmp(mode, 'material_depth')
                horizontal = face >= 3;           % lower/upper faces merged per material
                face(horizontal) = 2 + material(horizontal);
            end
            shell = 1 + (depth > 0.5);
            pml_label = face + 4 * (shell - 1);
        otherwise
            error('Unknown half-space partition mode: %s.', mode);
    end
    labels(~physical) = 2 + pml_label(~physical);
    ids = unique(labels(:));
    compact = zeros(size(labels));
    for b = 1:numel(ids), compact(labels == ids(b)) = b; end
    part = from_labels(compact, numel(ids), mode);
end

function part = from_labels(labels, count, mode)
    blocks = repmat(struct('idx', []), count, 1);
    for b = 1:count
        blocks(b).idx = find(labels(:) == b);
    end
    assert(sum(arrayfun(@(s) numel(s.idx), blocks)) == numel(labels), 'Partition must cover the grid once.');
    part = struct('mode', mode, 'n', numel(labels), 'blocks', blocks, 'labels', uint16(labels));
end
