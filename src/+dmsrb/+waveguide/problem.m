function P = problem(omega, opts)
%PROBLEM Gaussian waveguide in the box (0,3)x(0,1)x(0,1) (Experiment 3).
%   P = dmsrb.waveguide.problem(omega) discretizes laplace u + omega^2/c^2 u = f
%   with squared slowness 1/c^2 = 1 + mu*g(y,z), g = exp(-((y-1/2)^2+(z-1/2)^2)/(2*0.14^2)),
%   by the seven-point PML scheme at fixed points per wavelength:
%   Nphys = round([144 48 48]*omega/24) - 1 and npml = max(3, round(9*omega/24)).
%   The operator is affine, A(mu) = K_pml + omega^2*I + mu*omega^2*diag(g); the source
%   is a Gaussian of width 1.5h at (0.2, 1/2, 1/2), normalized to unit norm.
%   opts may override Nphys and npml (e.g. for small tests).
    if nargin < 2, opts = struct(); end
    scale = omega / 24;
    P = struct('name', 'waveguide', 'omega', omega, 'lengths', [3, 1, 1], 'sigma_max', 80, ...
        'center', [1.5, 0.5, 0.5], 'width', [0.14, 0.14], 'source', [0.2, 0.5, 0.5]);
    P.Nphys = option(opts, 'Nphys', round([144, 48, 48] * scale) - 1);
    P.npml = option(opts, 'npml', max(3, round([9, 9, 9] * scale)));
    P.h = P.lengths ./ (P.Nphys + 1);
    P.n = P.Nphys + 2 * P.npml;
    P.N = prod(P.n);
    P.coordinates = cell(1, 3);
    for d = 1:3
        full_grid = linspace(-P.npml(d) * P.h(d), P.lengths(d) + P.npml(d) * P.h(d), P.n(d) + 2);
        P.coordinates{d} = full_grid(2:end-1).';
    end
    pml = profile3(P.n, omega, P.npml, P.sigma_max);
    P.K = dmsrb.fd.stiffness3d(P.n, P.h, pml);
    [~, y, z] = deal(P.coordinates{:});
    d1 = reshape(((y - P.center(2)) / P.width(1)).^2, 1, [], 1);
    d2 = reshape(((z - P.center(3)) / P.width(2)).^2, 1, 1, []);
    g = repmat(exp(-0.5 * (d1 + d2)), P.n(1), 1, 1);
    P.profile = g(:);
    P.mass_base = omega^2;
    P.mass_parameter = omega^2 * P.profile;
    P.theta = @(mu) [1, mu];
    x = P.coordinates{1};
    sigma2 = (1.5 * min(P.h))^2;
    dx2 = reshape((x - P.source(1)).^2, [], 1, 1);
    dy2 = reshape((y - P.source(2)).^2, 1, [], 1);
    dz2 = reshape((z - P.source(3)).^2, 1, 1, []);
    f = exp(-0.5 * (dx2 + dy2 + dz2) / sigma2);
    P.b = f(:) / max(norm(f(:)), eps);
end

function pml = profile3(n, omega, npml, sigma_max)
% As dmsrb.fd.pml_profile, with a possibly different PML width per direction.
    pml = struct('sigma', {cell(1, 3)}, 's', {cell(1, 3)}, 's_half', {cell(1, 3)});
    for d = 1:3
        sigma = zeros(n(d) + 2, 1);
        layer = (npml(d):-1:1).' / max(npml(d), 1);
        values = sigma_max * layer.^2;
        sigma(1:npml(d)) = values;
        sigma(end - npml(d) + 1:end) = flipud(values);
        s = 1 + 1i * sigma / omega;
        pml.sigma{d} = sigma;
        pml.s{d} = s;
        pml.s_half{d} = 0.5 * (s(1:end-1) + s(2:end));
    end
end

function value = option(opts, name, fallback)
    value = fallback;
    if isfield(opts, name) && ~isempty(opts.(name)), value = opts.(name); end
end
