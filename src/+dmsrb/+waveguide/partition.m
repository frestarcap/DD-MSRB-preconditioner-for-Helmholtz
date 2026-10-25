function part = partition(P)
%PARTITION Subdomains of the waveguide grid: profile bands, axial sections, PML groups.
%   The physical domain is split into bands of the material profile g with edges
%   [0, 1e-3, 1e-2, 1e-1, 0.5, Inf], each band into five sections along the
%   waveguide axis x; the PML is split by face (x, y, z), side (low, high) and
%   depth (inner/outer half).  Empty subdomains are dropped (37 remain).
    thresholds = [1e-3, 1e-2, 1e-1, 0.5];
    sections = 5;
    [ix, iy, iz] = ndgrid(1:P.n(1), 1:P.n(2), 1:P.n(3));
    physical = ix > P.npml(1) & ix <= P.n(1) - P.npml(1) & iy > P.npml(2) & ...
        iy <= P.n(2) - P.npml(2) & iz > P.npml(3) & iz <= P.n(3) - P.npml(3);
    blocks = struct('idx', {});
    physical_idx = find(physical(:));
    edges = [0, thresholds, inf];
    g = abs(P.mass_parameter) / P.omega^2;     % the profile as recovered from the operator
    for band = 1:numel(edges) - 1
        in_band = g(physical_idx) >= edges(band) & g(physical_idx) < edges(band + 1);
        idx = physical_idx(in_band);
        if isempty(idx), continue; end
        [jx, ~, ~] = ind2sub(P.n, idx);
        section = min(floor((jx - P.npml(1) - 1) * sections / P.Nphys(1)) + 1, sections);
        for k = 1:sections
            blocks(end + 1).idx = idx(section == k); %#ok<AGROW>
        end
    end
    pml_idx = find(~physical(:));
    [jx, jy, jz] = ind2sub(P.n, pml_idx);
    coords = {jx, jy, jz};
    assigned = false(numel(pml_idx), 1);
    for d = 1:3
        low = coords{d} <= P.npml(d);
        high = coords{d} > P.n(d) - P.npml(d);
        depth = zeros(numel(pml_idx), 1);
        depth(low) = P.npml(d) - coords{d}(low) + 1;
        depth(high) = coords{d}(high) - (P.n(d) - P.npml(d));
        mask = (low | high) & ~assigned;           % faces in x, then y, then z
        assigned = assigned | mask;
        inner = max(1, ceil(0.5 * P.npml(d)));
        for side = {low, high}
            blocks(end + 1).idx = pml_idx(mask & side{1} & depth <= inner); %#ok<AGROW>
            blocks(end + 1).idx = pml_idx(mask & side{1} & depth > inner); %#ok<AGROW>
        end
    end
    blocks = blocks(arrayfun(@(b) ~isempty(b.idx), blocks));
    labels = zeros(P.n, 'uint16');
    for b = 1:numel(blocks), labels(blocks(b).idx) = b; end
    assert(all(labels(:) > 0), 'The partition must cover the grid once.');
    part = struct('mode', 'bands_sections_signed_depth', 'n', P.N, 'blocks', blocks, 'labels', labels);
end
