function value = transverse_roughness(P, v)
%TRANSVERSE_ROUGHNESS ||grad_perp v||^2 / (omega^2 ||v||^2) over the physical domain.
%   Forward differences in y and z between physical nodes; energies are Euclidean.
    V = reshape(v, P.n);
    V = V(P.npml(1) + (1:P.Nphys(1)), P.npml(2) + (1:P.Nphys(2)), P.npml(3) + (1:P.Nphys(3)));
    dy = diff(V, 1, 2) / P.h(2);
    dz = diff(V, 1, 3) / P.h(3);
    value = (sum(abs(dy(:)).^2) + sum(abs(dz(:)).^2)) / sum(abs(V(:)).^2) / P.omega^2;
end
