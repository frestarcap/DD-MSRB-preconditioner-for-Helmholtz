function m = sl_mass(P, ~)
%SL_MASS Shifted-Laplacian damping omega^2 on every node (identity rows included).
    m = P.omega^2 * ones(P.N, 1);
end
