function result = test_pml_stiffness_vectorization()
%TEST_PML_STIFFNESS_VECTORIZATION Compare vectorized and legacy assembly.

    nx = 47;
    ny = 39;
    hx = 1 / 31;
    hy = 1 / 29;
    omega = 37.2;
    npml = 9;
    sigma_x = damping_profile(nx, npml, 80);
    sigma_y = damping_profile(ny, npml, 80);
    pml = make_pml(sigma_x, sigma_y, omega);

    t_vec = tic;
    K_vec = pml_stiffness_2d(nx, ny, hx, hy, pml);
    vectorized_time = toc(t_vec);

    t_legacy = tic;
    K_legacy = legacy_assembly(nx, ny, hx, hy, pml);
    legacy_time = toc(t_legacy);

    abs_error = norm(K_vec - K_legacy, 'fro');
    rel_error = abs_error / max(norm(K_legacy, 'fro'), eps);
    assert(isequal(spones(K_vec), spones(K_legacy)), 'Sparsity patterns differ.');
    assert(rel_error <= 10 * eps, 'Relative matrix discrepancy is %.3e.', rel_error);

    result = struct('relative_error', rel_error, ...
        'vectorized_time', vectorized_time, 'legacy_time', legacy_time, ...
        'speedup', legacy_time / vectorized_time);
    fprintf('PML assembly relerr %.3e, vectorized %.4fs, legacy %.4fs, speedup %.1fx\n', ...
        rel_error, vectorized_time, legacy_time, result.speedup);
end

function sigma = damping_profile(n, npml, sigma_max)
    sigma = zeros(n + 2, 1);
    values = sigma_max * ((npml:-1:1).' / npml) .^ 2;
    sigma(1:npml) = values;
    sigma(end - npml + 1:end) = flip(values);
end

function pml = make_pml(sigma_x, sigma_y, omega)
    s_x = 1 + 1i * sigma_x / omega;
    s_y = 1 + 1i * sigma_y / omega;
    pml = struct('s_x', s_x, 's_y', s_y, ...
        's_x_half', 0.5 * (s_x(1:end - 1) + s_x(2:end)), ...
        's_y_half', 0.5 * (s_y(1:end - 1) + s_y(2:end)));
end

function K = legacy_assembly(nx, ny, hx, hy, pml)
    N = nx * ny;
    K = spalloc(N, N, 5 * N);
    idx = @(i, j) i + (j - 1) * nx;
    for i = 1:nx
        for j = 1:ny
            row = idx(i, j);
            ax = 1 / pml.s_x(i + 1);
            ax_p = 1 / pml.s_x_half(i + 1);
            ax_m = 1 / pml.s_x_half(i);
            by = 1 / pml.s_y(j + 1);
            by_p = 1 / pml.s_y_half(j + 1);
            by_m = 1 / pml.s_y_half(j);
            K(row, row) = -(ax_p + ax_m) * ax / hx^2 ...
                -(by_p + by_m) * by / hy^2;
            if i < nx, K(row, idx(i + 1, j)) = ax * ax_p / hx^2; end
            if i > 1, K(row, idx(i - 1, j)) = ax * ax_m / hx^2; end
            if j < ny, K(row, idx(i, j + 1)) = by * by_p / hy^2; end
            if j > 1, K(row, idx(i, j - 1)) = by * by_m / hy^2; end
        end
    end
end
