function part = partition(P, params)
%PARTITION Near-field, far-field and PML subdomains of the scattering grid.
%   The near field is the disk of radius r_near around the obstacle centre, the
%   far field the rest of the physical domain, and the PML the absorbing layer.
%   Nodes inside the obstacle for every training geometry are left out.
    active = false(P.N, 1);
    for mu = params(:).'
        [~, ~, idx] = dmsrb.ecc.system(P, mu);
        active(idx) = true;
    end
    active = reshape(active, P.nx, P.ny);
    [X, Y] = ndgrid(P.x(2:end-1), P.y(2:end-1));
    [ii, jj] = ndgrid(1:P.nx, 1:P.ny);
    pml = (ii <= P.npml) | (ii > P.nx - P.npml) | (jj <= P.npml) | (jj > P.ny - P.npml);
    near = ~pml & hypot(X - P.center(1), Y - P.center(2)) <= P.r_near & active;
    far = ~pml & ~near & active;
    pml = pml & active;
    labels = uint16(near + 2 * far + 3 * pml);
    part = struct('mode', 'near_far_pml', 'n', P.N, 'labels', labels, ...
        'blocks', struct('idx', {find(near(:)); find(far(:)); find(pml(:))}));
end
