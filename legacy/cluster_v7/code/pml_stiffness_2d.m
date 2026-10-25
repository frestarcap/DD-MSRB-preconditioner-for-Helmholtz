function K = pml_stiffness_2d(nx, ny, hx, hy, pml)
%PML_STIFFNESS_2D Vectorized five-point Helmholtz/PML stiffness assembly.

    ax = 1 ./ pml.s_x(2:nx + 1);
    ax_p = 1 ./ pml.s_x_half(2:nx + 1);
    ax_m = 1 ./ pml.s_x_half(1:nx);
    by = 1 ./ pml.s_y(2:ny + 1);
    by_p = 1 ./ pml.s_y_half(2:ny + 1);
    by_m = 1 ./ pml.s_y_half(1:ny);

    diag_x = -(ax_p + ax_m) .* ax / (hx * hx);
    diag_y = -(by_p + by_m) .* by / (hy * hy);
    coef_x_p = ax .* ax_p / (hx * hx);
    coef_x_m = ax .* ax_m / (hx * hx);
    coef_y_p = by .* by_p / (hy * hy);
    coef_y_m = by .* by_m / (hy * hy);

    rows_diag = (1:(nx * ny)).';
    vals_diag = reshape(diag_x + diag_y.', [], 1);

    rows_x_p = reshape((1:nx - 1).' + (0:ny - 1) * nx, [], 1);
    rows_x_m = reshape((2:nx).' + (0:ny - 1) * nx, [], 1);
    vals_x_p = repmat(coef_x_p(1:end - 1), ny, 1);
    vals_x_m = repmat(coef_x_m(2:end), ny, 1);

    rows_y_p = reshape((1:nx).' + (0:ny - 2) * nx, [], 1);
    rows_y_m = reshape((1:nx).' + (1:ny - 1) * nx, [], 1);
    vals_y_p = reshape(repmat(coef_y_p(1:end - 1).', nx, 1), [], 1);
    vals_y_m = reshape(repmat(coef_y_m(2:end).', nx, 1), [], 1);

    rows = [rows_diag; rows_x_p; rows_x_m; rows_y_p; rows_y_m];
    cols = [rows_diag; rows_x_p + 1; rows_x_m - 1; ...
        rows_y_p + nx; rows_y_m - nx];
    vals = [vals_diag; vals_x_p; vals_x_m; vals_y_p; vals_y_m];
    K = sparse(rows, cols, vals, nx * ny, nx * ny);
end
