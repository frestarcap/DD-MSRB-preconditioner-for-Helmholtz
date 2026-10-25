function A = operator(P, mu)
%OPERATOR Half-space Helmholtz matrix A(mu) = A_base + (1/mu^2 - 1) A_right.
    A = P.A_base + (1 / mu^2 - 1) * P.A_right;
end
