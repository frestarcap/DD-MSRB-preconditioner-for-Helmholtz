function [A, b, active] = system(P, mu)
%SYSTEM Scattering matrix, right-hand side and active nodes for aspect ratio mu.
    geom = dmsrb.ecc.geometry(P.nx, P.ny, P.x, P.y, P.center, P.radius * sqrt(mu), ...
        P.radius / sqrt(mu), P.eta_boundary_min);
    [A, b] = dmsrb.ecc.assemble(P.nx, P.ny, P.h, P.omega, P.pml, geom, P.k);
    active = find(reshape(geom.active(2:end-1, 2:end-1), P.N, 1));
end
