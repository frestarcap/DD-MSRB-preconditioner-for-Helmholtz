function P = exp1_problem(scale)
%EXP1_PROBLEM Half-space problem at omega = 53.5*scale with fixed points per wavelength.
    if nargin < 1, scale = 1; end
    Nphys = round(126 * scale) - 1;
    P = dmsrb.halfspace.problem(struct('Nphys', Nphys, 'omega', 53.5 * scale, 'sigma_max', 80 * scale));
end
