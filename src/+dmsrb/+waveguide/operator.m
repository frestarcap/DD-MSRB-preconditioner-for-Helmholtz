function A = operator(P, mu)
%OPERATOR Waveguide Helmholtz matrix A(mu) = K_pml + diag(omega^2*(1 + mu*g)).
    A = P.K + spdiags(dmsrb.waveguide.mass(P, mu), 0, P.N, P.N);
end
