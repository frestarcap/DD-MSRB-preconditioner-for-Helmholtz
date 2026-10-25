function [q, z, h, hn, info] = verified_snapshot(A, zrom, Q, Z)
%VERIFIED_SNAPSHOT Next Arnoldi vector and its inverse action from the recurrence.
%   With A*Z = Q and w = A*zrom = Q*h + hn*q, the inverse action of the next
%   Arnoldi vector q is z = (zrom - Z*h)/hn without a solve.  In finite precision
%   the candidate is accepted only if norm(A*z - q) <= 1e-10; otherwise z = A\q.
    [q, h, hn, stop] = dmsrb.arnoldi_step(A * zrom, Q);
    info = struct('breakdown', stop, 'direct_fallback', false, 'residual', NaN);
    z = zeros(size(zrom), 'like', zrom);
    if stop, return; end
    z = (zrom - Z * h) / hn;
    residual = norm(A * z - q);
    if ~isfinite(residual) || residual > 1e-10
        z = A \ q;
        info.direct_fallback = true;
    end
    info.residual = norm(A * z - q);
    assert(isfinite(info.residual) && info.residual <= 1e-10, ...
        'dmsrb:snapshot', 'Inverse-action snapshot failed the residual check.');
end
