function [rows, cases] = benchmark(queries, methods, sl, repetitions)
%BENCHMARK Repeated online timings with a rotating method order.
%   [rows, cases] = dmsrb.benchmark(queries, methods, sl, repetitions) solves every
%   query with every method once untimed (warm-up) and then repetitions times.  The
%   method order is circularly shifted by the query and repetition indices, so no
%   method is always run first.  queries(q) has fields mu, A, b, reference, and
%   precond (a function A -> SL preconditioner action); methods(m) has fields name
%   and model (a trained model, [] for the SL baseline, 'direct' for sparse LU).
%   Times exclude the assembly of A and b.  rows is a table with one row per timed
%   solve; cases{q,m} keeps the solution and residual history of the warm-up run.
%   Call with one computational thread (maxNumCompThreads(1)) to reproduce the paper.
    if nargin < 4, repetitions = 5; end
    M = numel(methods);
    rows = cell(numel(queries) * M * repetitions, 1);
    cases = cell(numel(queries), M);
    n = 0;
    for q = 1:numel(queries)
        Q = queries(q);
        s = sl;
        s.precond = @(A, ~) Q.precond(A);
        for rep = 0:repetitions
            for m = circshift(1:M, [0, mod(q + rep, M)])
                [u, info] = dmsrb.solve(Q.A, Q.b, Q.mu, methods(m).model, s);
                assert(info.flag == 0 && info.rho <= sl.tol, 'dmsrb:benchmark', ...
                    '%s failed at query %d: rho = %.3e.', methods(m).name, q, info.rho);
                if rep == 0
                    cases{q, m} = struct('u', u, 'history', info.history, 'rho', info.rho);
                    continue;
                end
                n = n + 1;
                rows{n} = {string(methods(m).name), q, Q.mu, rep, info.total_s, info.ROM_s, ...
                    info.setup_s, info.solve_s, info.sl_steps, info.rho_rom, info.rho, ...
                    norm(u - Q.reference) / norm(Q.reference)};
            end
        end
        fprintf('benchmark: query %d/%d done\n', q, numel(queries));
    end
    rows = cell2table(vertcat(rows{1:n}), 'VariableNames', {'method', 'query', 'parameter', 'repeat', ...
        'total_s', 'ROM_s', 'setup_s', 'solve_s', 'sl_steps', 'rho_rom', 'true_residual', 'relative_error'});
end
