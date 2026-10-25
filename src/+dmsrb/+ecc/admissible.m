function ok = admissible(P, mu)
%ADMISSIBLE True if the ghost-point stencil of aspect ratio mu is unambiguous.
    try
        dmsrb.ecc.geometry(P.nx, P.ny, P.x, P.y, P.center, P.radius * sqrt(mu), ...
            P.radius / sqrt(mu), P.eta_boundary_min);
        ok = true;
    catch err
        if ~any(contains(err.message, {'Ambiguous', 'eta out of range', 'Invalid'})), rethrow(err); end
        ok = false;
    end
end
