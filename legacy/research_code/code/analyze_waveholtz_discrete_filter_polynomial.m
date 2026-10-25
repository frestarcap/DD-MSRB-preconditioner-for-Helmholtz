function out = analyze_waveholtz_discrete_filter_polynomial(N_values)
%ANALYZE_WAVEHOLTZ_DISCRETE_FILTER_POLYNOMIAL Exact small-N factor study.
% Requires Symbolic Math Toolbox. This is a theory diagnostic, not a solver.
    if nargin < 1, N_values = 4:10; end
    syms y real
    out = cell(numel(N_values), 1);
    for k = 1:numel(N_values)
        N = N_values(k);
        a = sym(pi) / N;
        ca = simplify(cos(a));
        alpha = simplify(sym(1)/4 * (1 - tan(a)^2));
        U = chebyshevU(N - 1, y);
        shape = simplify((1 - alpha) * (1 - y^2) + alpha * (1 - ca^2));
        P = expand(simplifyFraction((2*y/N) * U * shape / (ca^2 - y^2)));
        remainder = expand(simplifyFraction((1 - P^2) / (y^2 - ca^2)^2));
        fprintf('\nN=%d\nalpha = %s\nP_N = %s\nR_N = %s\n', ...
            N, char(alpha), char(P), char(remainder));
        out{k} = struct('N', N, 'alpha', alpha, 'P', P, 'R', remainder, ...
            'P_factors', factor(P), 'R_factors', factor(remainder));
    end
end
