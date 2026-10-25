function P = problem(opts)
%PROBLEM Two-material half-space Helmholtz--PML problem (Experiment 1).
%   P = dmsrb.halfspace.problem(opts) discretizes
%       laplace u + omega^2/c(x)^2 u = f  on (0,1)^2,
%   c = 1 for x < 1/2 and c = mu for x > 1/2, with second-order differences on
%   Nphys^2 interior nodes, an npml-node PML on every side and a box source.
%   The operator is affine in theta(mu) = [1, 1/mu^2 - 1]:
%       A(mu) = A_base + (1/mu^2 - 1) A_right,
%   A_base = K_pml + omega^2 I and A_right = omega^2 diag(chi_right), where the
%   right material extends through the PML.
%
%   Options (defaults reproduce the paper): Nphys (125), omega (53.5),
%   npml (ceil(Nphys/4)), sigma_max (80), interface_x (0.5).
    if nargin < 1, opts = struct(); end
    Nphys = option(opts, 'Nphys', 125);
    omega = option(opts, 'omega', 53.5);
    npml = option(opts, 'npml', ceil(0.25 * Nphys));
    sigma_max = option(opts, 'sigma_max', 80);
    interface_x = option(opts, 'interface_x', 0.5);

    P = struct('name', 'halfspace', 'omega', omega, 'Nphys', Nphys, 'npml', npml, ...
        'sigma_max', sigma_max, 'interface_x', interface_x);
    P.h = 1 / (Nphys + 1);
    P.nx = Nphys + 2 * npml;
    P.ny = P.nx;
    P.N = P.nx * P.ny;
    P.I = (npml + 1):(npml + Nphys);
    P.J = P.I;
    P.x = linspace(-npml * P.h, 1 + npml * P.h, P.nx + 2);
    P.y = P.x;
    P.x_int = P.x(2:end-1).';
    P.y_int = P.y(2:end-1).';

    pml = dmsrb.fd.pml_profile([P.nx, P.ny], omega, npml, sigma_max);
    K = dmsrb.fd.stiffness2d(P.nx, P.ny, P.h, P.h, pml);
    chi = repmat(P.x_int > interface_x, 1, P.ny);
    P.chi_right = chi(:);
    P.A_base = K + omega^2 * speye(P.N);
    P.A_right = spdiags(omega^2 * double(P.chi_right), 0, P.N, P.N);
    P.terms = {P.A_base, P.A_right};
    P.theta = @(mu) [1, 1 / mu^2 - 1];

    % Box source of half-width 2h centred at physical node (round(0.2*Nphys), (Nphys+1)/2).
    col = max(2, round(0.20 * Nphys));
    row = round((Nphys + 1) / 2);
    x0 = P.x(P.I(col) + 1);
    y0 = P.y(P.J(row) + 1);
    [X, Y] = ndgrid(P.x_int, P.y_int);
    f = zeros(P.nx, P.ny);
    f(abs(X - x0) < 2 * P.h & abs(Y - y0) < 2 * P.h) = 1;
    P.b = f(:);
    P.source_xy = [x0, y0];
end

function value = option(opts, name, fallback)
    value = fallback;
    if isfield(opts, name) && ~isempty(opts.(name)), value = opts.(name); end
end
