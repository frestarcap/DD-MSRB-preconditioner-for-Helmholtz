function m = sl_mass(P, mu)
%SL_MASS Diagonal of the shifted-Laplacian damping term, omega^2/c(x;mu)^2.
    m = P.omega^2 * ones(P.N, 1) + (1 / mu^2 - 1) * full(diag(P.A_right));
end
