function results = run_shifted_laplacian_point_source_smoke()
%RUN_SHIFTED_LAPLACIAN_POINT_SOURCE_SMOKE Verify shifted-Laplacian GMRES.
%
% The smoke test solves a small constant-coefficient 2-D Helmholtz problem
% with a point source and compares right-preconditioned GMRES against the
% direct sparse solve.

    fprintf('=== shifted-Laplacian point-source smoke ===\n');
    nx = 45;
    ny = 45;
    Lx = 1.0;
    Ly = 1.0;
    omega = 38.0;
    hx = Lx / (nx + 1);
    hy = Ly / (ny + 1);

    A = build_constant_helmholtz_matrix(nx, ny, hx, hy, omega);
    b = zeros(nx * ny, 1);
    src_i = round(0.25 * nx);
    src_j = round(0.50 * ny);
    b(src_i + (src_j - 1) * nx) = 1 / (hx * hy);
    u_ref = A \ b;

    betas = [0.2, 0.5, 1.0];
    entries = repmat(empty_entry(), numel(betas), 1);
    for k = 1:numel(betas)
        beta = betas(k);
        M = A + 1i * beta * omega^2 * speye(size(A, 1));
        Msolve = make_sparse_solver(M);
        opts = struct('restart', 30, 'maxit', 20, 'tol', 1e-10, ...
            'verbose', false, 'reorthogonalize', true);
        [u, info] = shifted_laplacian_gmres(@(x) A * x, Msolve, b, zeros(size(b)), opts);
        relerr = norm(u - u_ref) / max(norm(u_ref), 1e-16);

        entries(k).beta = beta;
        entries(k).iter = info.iter_count;
        entries(k).relres = info.relres_final;
        entries(k).relerr = relerr;
        entries(k).time = info.total_time;
        entries(k).flag = info.flag;
        entries(k).relres_hist = info.relres_hist;

        fprintf('beta=%4.2f: iter=%3d, relres=%.3e, relerr=%.3e, time=%.3fs\n', ...
            beta, info.iter_count, info.relres_final, relerr, info.total_time);
    end

    out_dir = helm_output_path('outputs', 'shifted_laplacian_smoke');
    fig_dir = helm_output_path('outputs', 'figures', 'shifted_laplacian_smoke');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end

    plot_shifted_laplacian_smoke(entries, fig_dir);
    results = struct();
    results.nx = nx;
    results.ny = ny;
    results.omega = omega;
    results.entries = entries;
    save(fullfile(out_dir, 'shifted_laplacian_point_source_smoke.mat'), 'results');
end

function A = build_constant_helmholtz_matrix(nx, ny, hx, hy, omega)
    ex = ones(nx, 1);
    ey = ones(ny, 1);
    Tx = spdiags([ex -2 * ex ex], [-1 0 1], nx, nx) / hx^2;
    Ty = spdiags([ey -2 * ey ey], [-1 0 1], ny, ny) / hy^2;
    L = kron(speye(ny), Tx) + kron(Ty, speye(nx));
    A = L + omega^2 * speye(nx * ny);
end

function Msolve = make_sparse_solver(M)
    try
        Mdec = decomposition(M, 'lu');
        Msolve = @(r) Mdec \ r;
    catch
        Msolve = @(r) M \ r;
    end
end

function entry = empty_entry()
    entry = struct('beta', NaN, 'iter', NaN, 'relres', NaN, ...
        'relerr', NaN, 'time', NaN, 'flag', NaN, 'relres_hist', []);
end

function plot_shifted_laplacian_smoke(entries, fig_dir)
    fig = figure('Name', 'Shifted-Laplacian smoke residuals', ...
        'Position', [120 120 720 480]);
    ax = axes(fig);
    hold(ax, 'on');
    for k = 1:numel(entries)
        hist = entries(k).relres_hist;
        if isempty(hist)
            continue;
        end
        semilogy(ax, 0:(numel(hist) - 1), max(hist(:), 1e-16), ...
            'LineWidth', 1.6, 'Marker', 'o', 'MarkerSize', 4, ...
            'DisplayName', sprintf('\\beta=%.1f', entries(k).beta));
    end
    hold(ax, 'off');
    set(ax, 'YScale', 'log');
    ylim(ax, [1e-16, 2]);
    grid(ax, 'on');
    box(ax, 'on');
    xlabel(ax, 'iteration');
    ylabel(ax, 'relative Helmholtz residual');
    title(ax, 'Shifted-Laplacian point-source smoke');
    legend(ax, 'Location', 'best');
    set(fig, 'Color', 'w');
    try
        axtoolbar(ax, {});
    catch
    end
    exportgraphics(fig, fullfile(fig_dir, 'shifted_laplacian_point_source_smoke.png'), ...
        'Resolution', 300);
end
