function geom = geometry(nx, ny, x, y, center, semi_a, semi_b, eta_boundary_min)
%GEOMETRY Grid classification and ghost-point data for an elliptic obstacle.
%   Nodes inside the ellipse ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1 are internal; an
%   internal node next to an active node along x (or y) becomes a ghost node whose
%   value is extrapolated from the boundary crossing (Dirichlet data of the total
%   field) and the two active neighbours.  eta is the relative distance from the
%   last active node to the boundary; for eta <= eta_boundary_min the boundary
%   value is imposed directly.  Geometries with a node active on both sides of a
%   ghost are rejected ('Ambiguous ...').
    [X, Y] = ndgrid(x, y);
    interior = false(nx + 2, ny + 2);
    interior(2:end-1, 2:end-1) = true;

    ellipse_level = ((X - center(1)).^2) / (semi_a^2) + ((Y - center(2)).^2) / (semi_b^2);
    internal = interior & (ellipse_level <= 1.0);
    active = interior & ~internal;

    geom = struct();
    geom.center = center;
    geom.semi_a = semi_a;
    geom.semi_b = semi_b;
    semi_long = max(semi_a, semi_b);
    semi_short = min(semi_a, semi_b);
    geom.ecc = sqrt(max(0, 1 - (semi_short / semi_long)^2));
    geom.X = X;
    geom.Y = Y;
    geom.active = active;
    geom.internal = internal;
    geom.ghost_x = false(nx + 2, ny + 2);
    geom.ghost_y = false(nx + 2, ny + 2);
    geom.boundary_count_x = 0;
    geom.boundary_count_y = 0;

    zero_field = zeros(nx + 2, ny + 2);
    geom.gx_c2 = zero_field;
    geom.gx_c1 = zero_field;
    geom.gx_cg = zero_field;
    geom.gx_owner = zeros(nx + 2, ny + 2);
    geom.gx_u2 = zeros(nx + 2, ny + 2);
    geom.gx_gamma_x = zero_field;
    geom.gx_gamma_y = zero_field;

    geom.gy_c2 = zero_field;
    geom.gy_c1 = zero_field;
    geom.gy_cg = zero_field;
    geom.gy_owner = zeros(nx + 2, ny + 2);
    geom.gy_u2 = zeros(nx + 2, ny + 2);
    geom.gy_gamma_x = zero_field;
    geom.gy_gamma_y = zero_field;

    dims = [nx + 2, ny + 2];

    left_active = false(size(active));
    left_active(2:end, :) = active(1:end - 1, :);
    right_active = false(size(active));
    right_active(1:end - 1, :) = active(2:end, :);
    x_search = false(size(active));
    x_search(3:nx, 2:ny + 1) = true;
    [x_candidates_i, x_candidates_j] = find(internal & x_search ...
        & (left_active | right_active));

    for k = 1:numel(x_candidates_i)
            i = x_candidates_i(k);
            j = x_candidates_j(k);

            left_is_active = active(i - 1, j);
            right_is_active = active(i + 1, j);
            if left_is_active && right_is_active
                error('Ambiguous x-ghost at (%d,%d).', i, j);
            elseif left_is_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0, error('Invalid x-ghost geometry at (%d,%d).', i, j); end
                x_gamma = center(1) - semi_a * sqrt(root);
                eta = abs(x_gamma - x(i - 1)) / (x(i) - x(i - 1));
                if ~(eta > 0 && eta < 1), error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_x = geom.boundary_count_x + 1; end
                geom.ghost_x(i, j) = true;
                geom.gx_c2(i, j) = coeffs(1);
                geom.gx_c1(i, j) = coeffs(2);
                geom.gx_cg(i, j) = coeffs(3);
                geom.gx_owner(i, j) = sub2ind(dims, i - 1, j);
                geom.gx_u2(i, j) = sub2ind(dims, i - 2, j);
                geom.gx_gamma_x(i, j) = x_gamma;
                geom.gx_gamma_y(i, j) = y(j);
            elseif right_is_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0, error('Invalid x-ghost geometry at (%d,%d).', i, j); end
                x_gamma = center(1) + semi_a * sqrt(root);
                eta = abs(x(i + 1) - x_gamma) / (x(i + 1) - x(i));
                if ~(eta > 0 && eta < 1), error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_x = geom.boundary_count_x + 1; end
                geom.ghost_x(i, j) = true;
                geom.gx_c2(i, j) = coeffs(1);
                geom.gx_c1(i, j) = coeffs(2);
                geom.gx_cg(i, j) = coeffs(3);
                geom.gx_owner(i, j) = sub2ind(dims, i + 1, j);
                geom.gx_u2(i, j) = sub2ind(dims, i + 2, j);
                geom.gx_gamma_x(i, j) = x_gamma;
                geom.gx_gamma_y(i, j) = y(j);
            end
    end

    down_active = false(size(active));
    down_active(:, 2:end) = active(:, 1:end - 1);
    up_active = false(size(active));
    up_active(:, 1:end - 1) = active(:, 2:end);
    y_search = false(size(active));
    y_search(2:nx + 1, 3:ny) = true;
    [y_candidates_i, y_candidates_j] = find(internal & y_search ...
        & (down_active | up_active));

    for k = 1:numel(y_candidates_i)
            i = y_candidates_i(k);
            j = y_candidates_j(k);

            down_is_active = active(i, j - 1);
            up_is_active = active(i, j + 1);
            if down_is_active && up_is_active
                error('Ambiguous y-ghost at (%d,%d).', i, j);
            elseif down_is_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0, error('Invalid y-ghost geometry at (%d,%d).', i, j); end
                y_gamma = center(2) - semi_b * sqrt(root);
                eta = abs(y_gamma - y(j - 1)) / (y(j) - y(j - 1));
                if ~(eta > 0 && eta < 1), error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_y = geom.boundary_count_y + 1; end
                geom.ghost_y(i, j) = true;
                geom.gy_c2(i, j) = coeffs(1);
                geom.gy_c1(i, j) = coeffs(2);
                geom.gy_cg(i, j) = coeffs(3);
                geom.gy_owner(i, j) = sub2ind(dims, i, j - 1);
                geom.gy_u2(i, j) = sub2ind(dims, i, j - 2);
                geom.gy_gamma_x(i, j) = x(i);
                geom.gy_gamma_y(i, j) = y_gamma;
            elseif up_is_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0, error('Invalid y-ghost geometry at (%d,%d).', i, j); end
                y_gamma = center(2) + semi_b * sqrt(root);
                eta = abs(y(j + 1) - y_gamma) / (y(j + 1) - y(j));
                if ~(eta > 0 && eta < 1), error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end

                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_y = geom.boundary_count_y + 1; end
                geom.ghost_y(i, j) = true;
                geom.gy_c2(i, j) = coeffs(1);
                geom.gy_c1(i, j) = coeffs(2);
                geom.gy_cg(i, j) = coeffs(3);
                geom.gy_owner(i, j) = sub2ind(dims, i, j + 1);
                geom.gy_u2(i, j) = sub2ind(dims, i, j + 2);
                geom.gy_gamma_x(i, j) = x(i);
                geom.gy_gamma_y(i, j) = y_gamma;
            end
    end
end

function [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min)
    if eta <= eta_boundary_min
        coeffs = [0, 0, 1];
        boundary_mode = true;
        return;
    end
    c2 = (1 - eta) / (1 + eta);
    c1 = 2 * (1 - eta) / eta;
    cg = 2 / (eta * (1 + eta));
    coeffs = [c2, c1, cg];
    boundary_mode = false;
end
