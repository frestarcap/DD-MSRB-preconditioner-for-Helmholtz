function P = problem(opts)
%PROBLEM Plane-wave scattering by an elliptic obstacle (Experiment 2).
%   P = dmsrb.ecc.problem(opts) sets up the PML-extended grid of the unit square
%   for laplace u + omega^2 u = 0 outside the ellipse with semi-axes
%   a = radius*sqrt(mu), b = radius/sqrt(mu) centred at (1/2, 1/2); mu = a/b is the
%   geometric parameter.  The scattered field satisfies u = -exp(1i*omega*x) on the
%   obstacle.  The operator depends non-affinely on mu; dmsrb.ecc.system assembles it.
%
%   Options (defaults reproduce the paper): Nphys (85), omega (18), npml (30),
%   sigma_max (80), radius (0.12), eta_boundary_min (0.05), r_near (0.24).
    if nargin < 1, opts = struct(); end
    P = struct('name', 'ecc');
    P.Nphys = option(opts, 'Nphys', 85);
    P.omega = option(opts, 'omega', 18);
    P.npml = option(opts, 'npml', 30);
    P.sigma_max = option(opts, 'sigma_max', 80);
    P.radius = option(opts, 'radius', 0.12);
    P.center = [0.5, 0.5];
    P.eta_boundary_min = option(opts, 'eta_boundary_min', 0.05);
    P.r_near = option(opts, 'r_near', 0.24);
    P.h = 1 / (P.Nphys + 1);
    P.nx = P.Nphys + 2 * P.npml;
    P.ny = P.nx;
    P.N = P.nx * P.ny;
    P.x = linspace(-P.npml * P.h, 1 + P.npml * P.h, P.nx + 2);
    P.y = P.x;
    P.pml = dmsrb.fd.pml_profile([P.nx, P.ny], P.omega, P.npml, P.sigma_max);
    P.k = P.omega;
end

function value = option(opts, name, fallback)
    value = fallback;
    if isfield(opts, name) && ~isempty(opts.(name)), value = opts.(name); end
end
