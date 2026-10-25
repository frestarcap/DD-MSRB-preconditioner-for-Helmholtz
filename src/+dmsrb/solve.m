function [u, info] = solve(A, b, mu, model, sl)
%SOLVE Hybrid ROM + shifted-Laplacian solve of A u = b (Algorithm 2).
%   [u, info] = dmsrb.solve(A, b, mu, model, sl) runs the ROM-preconditioned
%   FGMRES iterations of model (none if model is empty, i.e. the SL baseline),
%   checks the true relative residual, and only if it exceeds sl.tol builds the
%   SL preconditioner sl.precond(A, mu) and continues with restarted GMRES from
%   the ROM iterate.  model = 'direct' solves with sparse LU instead.
%   info: ROM_s, setup_s, solve_s, total_s, sl_steps, rho_rom, rho, history.
    clock = tic;
    nb = norm(b);
    info = struct('ROM_s', 0, 'setup_s', 0, 'solve_s', 0, 'sl_steps', 0, ...
        'rho_rom', NaN, 'history', 1, 'flag', 0);
    if ischar(model) && strcmp(model, 'direct')
        timer = tic;
        u = A \ b;
        info.solve_s = toc(timer);
    else
        u = zeros(size(b));
        if ~isempty(model)
            timer = tic;
            [u, info.history] = dmsrb.prefix(A, b, mu, model, sl.tol);
            info.ROM_s = toc(timer);
        end
        info.rho_rom = norm(b - A * u) / nb;
        if info.rho_rom > sl.tol
            timer = tic;
            M = sl.precond(A, mu);
            info.setup_s = toc(timer);
            timer = tic;
            [u, g] = dmsrb.sl.gmres(@(v) A * v, M, b, u, sl.gmres);
            info.solve_s = toc(timer);
            info.sl_steps = g.iter_count;
            info.flag = g.flag;
            info.history = [info.history(:); g.relres_hist(2:end)];
        end
    end
    info.rho = norm(b - A * u) / nb;
    info.total_s = toc(clock);
end
