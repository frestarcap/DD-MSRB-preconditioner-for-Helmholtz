function waveholtz_2d_pml_scattering(shape_ratio)
    % ============================================================
    %  Exterior scattering by a sound-soft elliptical obstacle
    %  Compare Helmholtz reference and WaveHoltz iteration
    %  with directional ghost-point extrapolation on a staggered grid.
    % ============================================================

    rng(0);
    if nargin < 1 || isempty(shape_ratio)
        shape_ratio = 1.0; % a/b = 1 -> circle
    end
    validateattributes(shape_ratio, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, ...
        mfilename, 'shape_ratio');

    % ---- Physical-domain parameters ----
    L = 1.0;
    c = 1.0;
    omega = 18.0;
    max_iter = 400;
    tol = 1e-8;

    % ---- Discretization on physical domain ----
    Nphys = 240;
    nx_phys = Nphys;
    ny_phys = Nphys;
    hx = L / (nx_phys + 1);
    hy = L / (ny_phys + 1);

    % ---- PML outside physical domain ----
    ispml = 1;
    npml = 30;
    sigma_max = 80;

    % ---- Scatterer geometry ----
    center = [L/2, L/2];
    radius_ref = 0.12 * L; % reference circle radius (same area baseline)
    semi_a = radius_ref * sqrt(shape_ratio);
    semi_b = radius_ref / sqrt(shape_ratio);
    eta_boundary_min = 0.05;

    % ---- Time step + frequency mapping ----
    CFL = 0.20;
    dt0 = CFL * min(hx, hy) / c;
    [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt0);
    T_ts = nt * dt;

    fprintf('=== Time stepping match ===\n');
    fprintf('omega(helm)    = %.6f\n', omega);
    fprintf('omega(ts)      = %.6f\n', omega_ts);
    fprintf('c_w            = %.6f\n', c_w);
    fprintf('dt             = %.3e, nt = %d, T_ts = %.6f\n', dt, nt, T_ts);
    fprintf('shape ratio a/b= %.4f, semi axes a=%.4f, b=%.4f\n', shape_ratio, semi_a, semi_b);

    % ---- Total computational interior unknowns ----
    nx = nx_phys + 2*npml;
    ny = ny_phys + 2*npml;

    % ---- Total domain coordinates ----
    Lpml_x = npml * hx;
    Lpml_y = npml * hy;
    x = linspace(-Lpml_x, L + Lpml_x, nx + 2);
    y = linspace(-Lpml_y, L + Lpml_y, ny + 2);
    [X, Y] = ndgrid(x, y);

    % ---- Physical interior indices in total interior grid ----
    I = (npml + 1):(npml + nx_phys);
    J = (npml + 1):(npml + ny_phys);

    % ---- PML data ----
    pml_helm = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml);
    pml_wave = pml_helm;
    pml_wave.sigma_x = pml_helm.sigma_x / c_w;
    pml_wave.sigma_y = pml_helm.sigma_y / c_w;

    % ---- Geometry / ghost-point preprocessing ----
    geom = build_scatterer_geometry(nx, ny, x, y, center, semi_a, semi_b, eta_boundary_min);
    fprintf('active nodes     = %d\n', nnz(geom.active(2:end-1,2:end-1)));
    fprintf('internal nodes   = %d\n', nnz(geom.internal(2:end-1,2:end-1)));
    fprintf('ghost_x nodes    = %d\n', nnz(geom.ghost_x));
    fprintf('ghost_y nodes    = %d\n', nnz(geom.ghost_y));
    fprintf('boundary ghosts x/y = %d / %d\n', geom.boundary_count_x, geom.boundary_count_y);

    % ---- Incident plane wave ----
    inc = build_incident_wave(X, omega, c);

    % ---- Helmholtz reference system ----
    [A_scatt, b_scatt] = helmholtz_scattering_system(nx, ny, hx, hy, omega, pml_helm, geom, inc);
    u_helm_vec = A_scatt \ b_scatt;
    u_helm = zeros(nx + 2, ny + 2);
    u_helm(2:end-1, 2:end-1) = reshape(u_helm_vec, nx, ny);
    u_helm(geom.internal) = 0;

    % ---- WaveHoltz solve ----
    [u_wave, hist] = waveholtz_iteration_scattering( ...
        nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, pml_wave, ...
        geom, inc, max_iter, tol, A_scatt, b_scatt);
    u_wave(geom.internal) = 0;

    % ---- Physical-domain diagnostics ----
    active_phys = geom.active(I + 1, J + 1);
    Uh_phys = u_helm(I + 1, J + 1);
    Uw_phys = u_wave(I + 1, J + 1);
    Uinc_phys = inc.u_inc_hat(I + 1, J + 1);

    diff_phys = Uw_phys(active_phys) - Uh_phys(active_phys);
    ref_phys = Uh_phys(active_phys);
    rel_err = norm(diff_phys(:)) / max(norm(ref_phys(:)), 1e-15);
    rel_err_real = norm(real(diff_phys(:))) / max(norm(real(ref_phys(:))), 1e-15);
    rel_err_imag = norm(imag(diff_phys(:))) / max(norm(imag(ref_phys(:))), 1e-15);
    fprintf('Relative scattering-field error in physical domain (complex): %.5e\n', rel_err);
    fprintf('Relative scattering-field error in physical domain (real): %.5e\n', rel_err_real);
    fprintf('Relative scattering-field error in physical domain (imag): %.5e\n', rel_err_imag);

    Utot_helm = Uh_phys + Uinc_phys;
    Utot_wave = Uw_phys + Uinc_phys;
    total_err = norm(Utot_wave(active_phys) - Utot_helm(active_phys)) / max(norm(Utot_helm(active_phys)), 1e-15);
    fprintf('Relative total-field error in physical domain: %.5e\n', total_err);

    % ---- Masked plotting ----
    helm_plot = masked_field(real(u_helm), geom.active);
    wave_plot = masked_field(real(u_wave), geom.active);
    diff_plot = masked_field(real(u_wave - u_helm), geom.active);
    total_plot = masked_field(real(u_wave + inc.u_inc_hat), geom.active);

    figure;
    subplot(2,2,1);
    plot_masked_image(x, y, helm_plot, geom.active, sprintf('Helmholtz scattering (a/b=%.3f)', shape_ratio));

    subplot(2,2,2);
    plot_masked_image(x, y, wave_plot, geom.active, sprintf('WaveHoltz scattering (a/b=%.3f)', shape_ratio));

    subplot(2,2,3);
    plot_masked_image(x, y, diff_plot, geom.active, 'WaveHoltz - Helmholtz');

    subplot(2,2,4);
    plot_masked_image(x, y, total_plot, geom.active, 'WaveHoltz total field');

    figure;
    semilogy(hist, 'LineWidth', 1.5);
    grid on;
    xlabel('iteration');
    ylabel('rel-res');
    title('WaveHoltz iteration history');
end


function out = masked_field(field, active_mask)
    out = field;
    out(~active_mask) = NaN;
end


function plot_masked_image(x, y, field, active_mask, ttl)
    h = imagesc(x, y, field.');
    set(h, 'AlphaData', active_mask.');
    axis equal tight;
    set(gca, 'YDir', 'normal', 'Color', [0.85, 0.85, 0.85]);
    title(ttl);
    colorbar;
end


function inc = build_incident_wave(X, omega, c)
    k = omega / c;
    u_inc_hat = exp(1i * k * X);
    inc = struct('k', k, 'u_inc_hat', u_inc_hat);
end


function geom = build_scatterer_geometry(nx, ny, x, y, center, semi_a, semi_b, eta_boundary_min)
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

    for i = 3:(nx)
        for j = 2:(ny + 1)
            if ~internal(i, j)
                continue;
            end

            left_active = active(i - 1, j);
            right_active = active(i + 1, j);
            if left_active && right_active
                error('Ambiguous x-ghost at (%d,%d): obstacle too small for single-valued x ghost.', i, j);
            elseif left_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0
                    error('Invalid x-ghost geometry at (%d,%d).', i, j);
                end
                x_gamma = center(1) - semi_a * sqrt(root);
                eta = abs(x_gamma - x(i - 1)) / (x(i) - x(i - 1));
                if ~(eta > 0 && eta < 1)
                    error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
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
            elseif right_active
                root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
                if root <= 0
                    error('Invalid x-ghost geometry at (%d,%d).', i, j);
                end
                x_gamma = center(1) + semi_a * sqrt(root);
                eta = abs(x(i + 1) - x_gamma) / (x(i + 1) - x(i));
                if ~(eta > 0 && eta < 1)
                    error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
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
    end

    for i = 2:(nx + 1)
        for j = 3:(ny)
            if ~internal(i, j)
                continue;
            end

            down_active = active(i, j - 1);
            up_active = active(i, j + 1);
            if down_active && up_active
                error('Ambiguous y-ghost at (%d,%d): obstacle too small for single-valued y ghost.', i, j);
            elseif down_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0
                    error('Invalid y-ghost geometry at (%d,%d).', i, j);
                end
                y_gamma = center(2) - semi_b * sqrt(root);
                eta = abs(y_gamma - y(j - 1)) / (y(j) - y(j - 1));
                if ~(eta > 0 && eta < 1)
                    error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
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
            elseif up_active
                root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
                if root <= 0
                    error('Invalid y-ghost geometry at (%d,%d).', i, j);
                end
                y_gamma = center(2) + semi_b * sqrt(root);
                eta = abs(y(j + 1) - y_gamma) / (y(j + 1) - y(j));
                if ~(eta > 0 && eta < 1)
                    error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta);
                end
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

    active_left = false(nx + 2, ny + 2);
    active_right = false(nx + 2, ny + 2);
    active_down = false(nx + 2, ny + 2);
    active_up = false(nx + 2, ny + 2);

    active_left(2:end, :) = active(1:end-1, :);
    active_right(1:end-1, :) = active(2:end, :);
    active_down(:, 2:end) = active(:, 1:end-1);
    active_up(:, 1:end-1) = active(:, 2:end);

    x_adjacent_internal = internal & (active_left | active_right);
    y_adjacent_internal = internal & (active_down | active_up);

    if any(x_adjacent_internal(:) & ~geom.ghost_x(:))
        error('Some x-adjacent internal nodes were not assigned ghost data.');
    end

    if any(y_adjacent_internal(:) & ~geom.ghost_y(:))
        error('Some y-adjacent internal nodes were not assigned ghost data.');
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


function pml = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml)
    sigma_x = zeros(nx + 2, 1);
    sigma_y = zeros(ny + 2, 1);

    for i = 1:npml
        if ispml == 1
            val = sigma_max * ((npml - (i - 1)) / npml)^2;
        else
            val = 0;
        end
        sigma_x(i) = val;
        sigma_x(end + 1 - i) = val;
        sigma_y(i) = val;
        sigma_y(end + 1 - i) = val;
    end

    s_x = 1.0 + 1i * sigma_x / omega;
    s_y = 1.0 + 1i * sigma_y / omega;
    s_x_half = 0.5 * (s_x(1:end-1) + s_x(2:end));
    s_y_half = 0.5 * (s_y(1:end-1) + s_y(2:end));

    pml = struct('sigma_x', sigma_x, 'sigma_y', sigma_y, ...
                 's_x', s_x, 's_y', s_y, ...
                 's_x_half', s_x_half, 's_y_half', s_y_half);
end


function [A_scatt, rhs] = helmholtz_scattering_system(nx, ny, hx, hy, omega, pml, geom, inc)
    Ntot = nx * ny;
    A_scatt = spalloc(Ntot, Ntot, 7 * Ntot);
    rhs = zeros(Ntot, 1);
    idx = @(ii, jj) ii + (jj - 1) * nx;

    dims_full = [nx + 2, ny + 2];

    for ii = 1:nx
        i = ii + 1;
        for jj = 1:ny
            j = jj + 1;
            row = idx(ii, jj);

            if geom.internal(i, j)
                A_scatt(row, row) = 1;
                continue;
            end

            a_x = 1.0 ./ pml.s_x(i);
            a_p = 1.0 ./ pml.s_x_half(i);
            a_m = 1.0 ./ pml.s_x_half(i - 1);
            b_y = 1.0 ./ pml.s_y(j);
            b_p = 1.0 ./ pml.s_y_half(j);
            b_m = 1.0 ./ pml.s_y_half(j - 1);

            A_scatt(row, row) = A_scatt(row, row) ...
                - a_x * (a_p + a_m) / (hx * hx) ...
                - b_y * (b_p + b_m) / (hy * hy) ...
                + omega^2;

            % +x neighbor
            coef = a_x * a_p / (hx * hx);
            if geom.active(i + 1, j)
                A_scatt(row, idx(ii + 1, jj)) = A_scatt(row, idx(ii + 1, jj)) + coef;
            elseif geom.internal(i + 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i + 1, j, idx, dims_full, geom, inc);
            end

            % -x neighbor
            coef = a_x * a_m / (hx * hx);
            if geom.active(i - 1, j)
                A_scatt(row, idx(ii - 1, jj)) = A_scatt(row, idx(ii - 1, jj)) + coef;
            elseif geom.internal(i - 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i - 1, j, idx, dims_full, geom, inc);
            end

            % +y neighbor
            coef = b_y * b_p / (hy * hy);
            if geom.active(i, j + 1)
                A_scatt(row, idx(ii, jj + 1)) = A_scatt(row, idx(ii, jj + 1)) + coef;
            elseif geom.internal(i, j + 1)
                [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, i, j + 1, idx, dims_full, geom, inc);
            end

            % -y neighbor
            coef = b_y * b_m / (hy * hy);
            if geom.active(i, j - 1)
                A_scatt(row, idx(ii, jj - 1)) = A_scatt(row, idx(ii, jj - 1)) + coef;
            elseif geom.internal(i, j - 1)
                [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, i, j - 1, idx, dims_full, geom, inc);
            end
        end
    end
end


function [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, gi, gj, idx, dims_full, geom, inc)
    owner_lin = geom.gx_owner(gi, gj);
    if owner_lin == 0
        error('Missing x-ghost coefficients at (%d,%d).', gi, gj);
    end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    [u2i, u2j] = ind2sub(dims_full, geom.gx_u2(gi, gj));
    if ~geom.active(oi, oj)
        error('x-ghost owner at (%d,%d) is not active.', oi, oj);
    end

    c2 = geom.gx_c2(gi, gj);
    c1 = geom.gx_c1(gi, gj);
    cg = geom.gx_cg(gi, gj);
    u_gamma_hat = -exp(1i * inc.k * geom.gx_gamma_x(gi, gj));

    A_scatt(row, idx(oi - 1, oj - 1)) = A_scatt(row, idx(oi - 1, oj - 1)) - coef * c1;
    A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * c2;
    rhs(row) = rhs(row) - coef * cg * u_gamma_hat;
end


function [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, gi, gj, idx, dims_full, geom, inc)
    owner_lin = geom.gy_owner(gi, gj);
    if owner_lin == 0
        error('Missing y-ghost coefficients at (%d,%d).', gi, gj);
    end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    [u2i, u2j] = ind2sub(dims_full, geom.gy_u2(gi, gj));
    if ~geom.active(oi, oj)
        error('y-ghost owner at (%d,%d) is not active.', oi, oj);
    end

    c2 = geom.gy_c2(gi, gj);
    c1 = geom.gy_c1(gi, gj);
    cg = geom.gy_cg(gi, gj);
    u_gamma_hat = -exp(1i * inc.k * geom.gy_gamma_x(gi, gj));

    A_scatt(row, idx(oi - 1, oj - 1)) = A_scatt(row, idx(oi - 1, oj - 1)) - coef * c1;
    A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * c2;
    rhs(row) = rhs(row) - coef * cg * u_gamma_hat;
end


function W = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x0, geom, inc)
    sx_full = zeros(nx + 2, ny + 2);
    sy_full = zeros(nx + 2, ny + 2);
    sx = pml.sigma_x;
    sy = pml.sigma_y;
    for i = 1:(nx + 2)
        sx_full(i, :) = sx(i);
    end
    for j = 1:(ny + 2)
        sy_full(:, j) = sy(j);
    end

    sigma_x_vx = 0.5 * (sx_full(1:end-1, :) + sx_full(2:end, :));
    sigma_y_vy = 0.5 * (sy_full(:, 1:end-1) + sy_full(:, 2:end));

    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    naux = 3 * nu;
    nstate = nu + nv + naux;
    if numel(x0) ~= nstate
        error('x0 length mismatch: expect %d, got %d', nstate, numel(x0));
    end

    u0_int = x0(1:nu);
    v0_int = x0(nu + 1:nu + nv);
    psi1_int = x0(nu + nv + 1 : nu + nv + nu);
    psi2_int = x0(nu + nv + nu + 1 : nu + nv + 2*nu);
    theta_int = x0(nu + nv + 2*nu + 1 : end);

    u = zeros(nx + 2, ny + 2);
    u(2:end-1, 2:end-1) = reshape(u0_int, nx, ny);
    u(geom.internal) = 0;

    vx = zeros(nx + 1, ny + 2);
    vy = zeros(nx + 2, ny + 1);
    vx_int = reshape(v0_int(1:nvx), nx + 1, ny);
    vy_int = reshape(v0_int(nvx + 1:end), nx, ny + 1);
    vx(:, 2:end-1) = vx_int;
    vy(2:end-1, :) = vy_int;

    psi1 = zeros(nx + 2, ny + 2);
    psi2 = zeros(nx + 2, ny + 2);
    theta = zeros(nx + 2, ny + 2);
    psi1(2:end-1, 2:end-1) = reshape(psi1_int, nx, ny);
    psi2(2:end-1, 2:end-1) = reshape(psi2_int, nx, ny);
    theta(2:end-1, 2:end-1) = reshape(theta_int, nx, ny);
    psi1(geom.internal) = 0;
    psi2(geom.internal) = 0;
    theta(geom.internal) = 0;

    c_w = cos(omega_ts * dt / 2);
    sigu = sx_full + sy_full;

    v0_vec = [reshape(vx(:, 2:end-1), nvx, 1); reshape(vy(2:end-1, :), nvy, 1)];
    W = zeros(nt + 1, nstate);
    u0_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
    W(1, :) = [u0_vec; v0_vec; psi1_int; psi2_int; theta_int].';

    for n = 1:nt
        t = (n - 1) * dt;

        [u_x, u_y] = fill_directional_ghosts(u, geom, inc, omega_ts, t);

        du_dx = (u_x(2:end, :) - u_x(1:end-1, :)) / hx;
        du_dy = (u_y(:, 2:end) - u_y(:, 1:end-1)) / hy;
        vx_new = ((1 - 0.5 * dt * sigma_x_vx) .* vx + dt * c^2 * du_dx) ./ (1 + 0.5 * dt * sigma_x_vx);
        vy_new = ((1 - 0.5 * dt * sigma_y_vy) .* vy + dt * c^2 * du_dy) ./ (1 + 0.5 * dt * sigma_y_vy);

        vx_avg = 0.5 * (vx + vx_new);
        vy_avg = 0.5 * (vy + vy_new);

        grad_vx = (vx_avg(2:end, :) - vx_avg(1:end-1, :)) / hx;
        tmp1 = zeros(nx + 2, ny + 2);
        tmp1(2:end-1, :) = grad_vx;

        grad_vy = (vy_avg(:, 2:end) - vy_avg(:, 1:end-1)) / hy;
        tmp2 = zeros(nx + 2, ny + 2);
        tmp2(:, 2:end-1) = grad_vy;

        psi1 = psi1 + dt * (sy_full .* tmp1);
        psi2 = psi2 + dt * (sx_full .* tmp2);
        theta = theta + dt * ((sx_full .* sy_full) .* (c_w^2) .* u);

        div_v = zeros(nx + 2, ny + 2);
        div_v(2:end-1, :) = div_v(2:end-1, :) + (vx_new(2:end, :) - vx_new(1:end-1, :)) / hx;
        div_v(:, 2:end-1) = div_v(:, 2:end-1) + (vy_new(:, 2:end) - vy_new(:, 1:end-1)) / hy;

        rhs_u = ...
              div_v(2:end-1, 2:end-1) ...
            - theta(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) ...
            + psi2(2:end-1, 2:end-1);

        u_int = u(2:end-1, 2:end-1);
        num_u = (1 - 0.5 * dt * sigu(2:end-1, 2:end-1)) .* u_int + dt * rhs_u;
        den_u = (1 + 0.5 * dt * sigu(2:end-1, 2:end-1));
        u_int_new = num_u ./ den_u;
        active_int = geom.active(2:end-1, 2:end-1);
        u_int(~active_int) = 0;
        u_int(active_int) = u_int_new(active_int);
        u(2:end-1, 2:end-1) = u_int;
        u(geom.internal) = 0;

        psi1(geom.internal) = 0;
        psi2(geom.internal) = 0;
        theta(geom.internal) = 0;

        vx = vx_new;
        vy = vy_new;

        v_vec = [reshape(vx(:, 2:end-1), nvx, 1); reshape(vy(2:end-1, :), nvy, 1)];
        u_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
        psi1_vec = reshape(psi1(2:end-1, 2:end-1), nu, 1);
        psi2_vec = reshape(psi2(2:end-1, 2:end-1), nu, 1);
        theta_vec = reshape(theta(2:end-1, 2:end-1), nu, 1);
        W(n + 1, :) = [u_vec; v_vec; psi1_vec; psi2_vec; theta_vec].';
    end
end


function [u_x, u_y] = fill_directional_ghosts(u, geom, inc, omega_ts, t)
    u_x = u;
    u_y = u;

    if any(geom.ghost_x(:))
        mask = geom.ghost_x;
        owner_vals = u(geom.gx_owner(mask));
        u2_vals = u(geom.gx_u2(mask));
        u_gamma_hat = -exp(1i * inc.k * geom.gx_gamma_x(mask));
        u_gamma_t = real(u_gamma_hat .* exp(-1i * omega_ts * t));
        ghost_vals = geom.gx_c2(mask) .* u2_vals - geom.gx_c1(mask) .* owner_vals + geom.gx_cg(mask) .* u_gamma_t;
        u_x(mask) = ghost_vals;
    end

    if any(geom.ghost_y(:))
        mask = geom.ghost_y;
        owner_vals = u(geom.gy_owner(mask));
        u2_vals = u(geom.gy_u2(mask));
        u_gamma_hat = -exp(1i * inc.k * geom.gy_gamma_x(mask));
        u_gamma_t = real(u_gamma_hat .* exp(-1i * omega_ts * t));
        ghost_vals = geom.gy_c2(mask) .* u2_vals - geom.gy_c1(mask) .* owner_vals + geom.gy_cg(mask) .* u_gamma_t;
        u_y(mask) = ghost_vals;
    end
end


function [u_full, hist] = waveholtz_iteration_scattering(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, geom, inc, max_iter, tol, A, b)
    nu = nx * ny;
    nvx = (nx + 1) * ny;
    nvy = nx * (ny + 1);
    nv = nvx + nvy;
    relax = 0.15;

    x = zeros(nu + nv + 3 * nu, 1);
    hist = zeros(max_iter, 1);

    for k = 1:max_iter
        W = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x, geom, inc);
        if any(~isfinite(W(:)))
            error('Non-finite values encountered during one-period time stepping at iteration %d.', k);
        end

        t_grid = (0:nt).' * dt;
        weights = cos(omega_ts * t_grid) - 0.25;
        integrand = bsxfun(@times, weights, W);
        integral = trapz(t_grid, integrand, 1);
        x_raw = (2.0 / T_ts) * integral.';
        x_new = (1 - relax) * x + relax * x_raw;
        if any(~isfinite(x_new))
            error('Non-finite fixed-point state at iteration %d.', k);
        end

        res = norm(x_new - x) / max(1e-16, norm(x_new));
        hist(k) = res;
        x = x_new;

        fprintf('iter %3d: rel-res = %.3e\n', k, res);
        if res < tol
            hist = hist(1:k);
            break;
        end
    end

    u_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x, geom, inc);
    u_full(geom.internal) = 0;

    rec_res = norm(A * reshape(u_full(2:end-1, 2:end-1), nu, 1) - b) / max(norm(b), 1e-15);
    fprintf('Recovered phasor residual: %.3e\n', rec_res);
end


function uhat_full = recover_u_phasor_timeproj_scattering(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x_state, geom, inc)
    Wconv = solve_wave_one_period_scattering(nx, ny, hx, hy, c, omega_ts, dt, nt, pml, x_state, geom, inc);
    nu = nx * ny;
    t_grid = (0:nt).' * dt;
    q_cos = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, cos(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';
    q_sin = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, sin(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';
    uhat = q_cos + 1i * q_sin;

    uhat_full = zeros(nx + 2, ny + 2);
    uhat_full(2:end-1, 2:end-1) = reshape(uhat, nx, ny);
    uhat_full(geom.internal) = 0;
end


function [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt_init)
    dt = dt_init;
    if omega * dt / 2 >= 0.98
        dt = 0.98 * 2 / omega;
    end

    for it = 1:6
        omega_ts = (2 / dt) * asin(omega * dt / 2);
        nt = max(10, round(2 * pi / (omega_ts * dt)));
        dt = 2 * pi / (omega_ts * nt);
    end

    omega_ts = (2 / dt) * asin(omega * dt / 2);
    c_w = cos(omega_ts * dt / 2);
end
