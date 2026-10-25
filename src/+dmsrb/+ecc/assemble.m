function [A, b] = assemble(nx, ny, h, omega, pml, geom, k)
%ASSEMBLE Scattering system for one obstacle geometry (ghost-point boundary).
%   Rows of active nodes carry the five-point PML Helmholtz stencil, internal
%   nodes get identity rows, and the stencil entries of ghost nodes are replaced by
%   the extrapolation of the scattered field with Dirichlet data -exp(1i*k*x) of
%   the incident plane wave on the obstacle boundary.
    Ntot = nx * ny;
    A_base = dmsrb.fd.stiffness2d(nx, ny, h, h, pml) + omega^2 * speye(Ntot);
    active = reshape(geom.active(2:end - 1, 2:end - 1), Ntot, 1);
    internal = ~active;
    A = spdiags(double(active), 0, Ntot, Ntot) * A_base + spdiags(double(internal), 0, Ntot, Ntot);
    [delta_x, rhs_x] = ghost_matrix_updates(A_base, geom, k, 'x', nx, ny);
    [delta_y, rhs_y] = ghost_matrix_updates(A_base, geom, k, 'y', nx, ny);
    A = A + delta_x + delta_y;
    b = rhs_x + rhs_y;
end

function [delta, rhs] = ghost_matrix_updates(A_base, geom, k, direction, nx, ny)
    Ntot = nx * ny;
    dims_full = [nx + 2, ny + 2];
    switch direction
        case 'x'
            mask = geom.ghost_x;
            owner_full = geom.gx_owner(mask);
            u2_full = geom.gx_u2(mask);
            c1 = geom.gx_c1(mask);
            c2 = geom.gx_c2(mask);
            cg = geom.gx_cg(mask);
            gamma_x = geom.gx_gamma_x(mask);
        case 'y'
            mask = geom.ghost_y;
            owner_full = geom.gy_owner(mask);
            u2_full = geom.gy_u2(mask);
            c1 = geom.gy_c1(mask);
            c2 = geom.gy_c2(mask);
            cg = geom.gy_cg(mask);
            gamma_x = geom.gy_gamma_x(mask);
        otherwise
            error('Unknown ghost direction %s.', direction);
    end

    if isempty(owner_full)
        delta = sparse(Ntot, Ntot);
        rhs = zeros(Ntot, 1);
        return;
    end

    ghost_full = find(mask);
    rows = full_to_interior_indices(owner_full, dims_full, nx, ny);
    ghost_cols = full_to_interior_indices(ghost_full, dims_full, nx, ny);
    u2_cols = full_to_interior_indices(u2_full, dims_full, nx, ny);
    coef = full(A_base(sub2ind([Ntot, Ntot], rows, ghost_cols)));

    keep_c1 = c1 ~= 0;
    keep_c2 = c2 ~= 0;
    delta_rows = [rows; rows(keep_c1); rows(keep_c2)];
    delta_cols = [ghost_cols; rows(keep_c1); u2_cols(keep_c2)];
    delta_vals = [-coef; -coef(keep_c1) .* c1(keep_c1); ...
        coef(keep_c2) .* c2(keep_c2)];
    delta = sparse(delta_rows, delta_cols, delta_vals, Ntot, Ntot);

    u_gamma_hat = -exp(1i * k * gamma_x);
    rhs = full(sparse(rows, ones(size(rows)), -coef .* cg .* u_gamma_hat, Ntot, 1));
end

function idx = full_to_interior_indices(full_idx, dims_full, nx, ny)
    [ii, jj] = ind2sub(dims_full, full_idx);
    if any(ii < 2 | ii > nx + 1 | jj < 2 | jj > ny + 1)
        error('Ghost stencil reaches outside the stored interior grid.');
    end
    idx = (ii - 1) + (jj - 2) * nx;
end
