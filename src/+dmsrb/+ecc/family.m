function F = family(P)
%FAMILY Parameter family of the scattering problem (non-affine: no reduced terms).
    F = struct('system', @(mu) system(P, mu));
end

function [A, b] = system(P, mu)
    [A, b] = dmsrb.ecc.system(P, mu);
end
