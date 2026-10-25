function F = family(P)
%FAMILY Parameter family of the half-space problem for dmsrb.train and dmsrb.solve.
    F = struct('system', @(mu) system(P, mu), 'terms', {P.terms}, 'theta', P.theta);
end

function [A, b] = system(P, mu)
    A = dmsrb.halfspace.operator(P, mu);
    b = P.b;
end
