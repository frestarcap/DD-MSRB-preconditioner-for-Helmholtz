function [u, history, info] = prefix(A, b, mu, model, tol)
%PREFIX The ROM-preconditioned FGMRES iterations from the zero initial guess.
%   [u, history] = dmsrb.prefix(A, b, mu, model, tol) applies the reduced
%   inverse actions of model.layers{1..k} as the first k flexible preconditioners
%   and stops early when the true relative residual reaches tol, the Arnoldi
%   process breaks down, or a layer is empty.  history(j+1) = norm(b - A*u_j)/norm(b), history(1) = 1.
%   Affine models solve the Hessenberg least-squares problem with backslash,
%   non-affine models with lsqminnorm.
    n = numel(b);
    depth = numel(model.layers);
    nb = norm(b);
    u = zeros(n, 1);
    history = 1;
    Q = zeros(n, depth + 1);
    Z = zeros(n, depth);
    H = zeros(depth + 1, depth);
    Q(:, 1) = b / nb;
    stop = false;
    theta = [];
    if model.affine, theta = model.theta(mu); end
    for j = 1:depth
        if model.layers{j}.space.k == 0, break; end   % converged training trajectory
        Z(:, j) = dmsrb.apply_rom(model.layers{j}, A, theta, Q(:, j));
        [Q(:, j + 1), H(1:j, j), H(j + 1, j), stop] = dmsrb.arnoldi_step(A * Z(:, j), Q(:, 1:j));
        g = zeros(j + 1, 1);
        g(1) = nb;
        if model.affine
            y = H(1:j + 1, 1:j) \ g;
        else
            y = lsqminnorm(H(1:j + 1, 1:j), g);
        end
        u = Z(:, 1:j) * y;
        history(end + 1, 1) = norm(b - A * u) / nb; %#ok<AGROW>
        if stop || history(end) <= tol, break; end
    end
    info = struct('steps', numel(history) - 1, 'breakdown', stop);
end
