function problem = helmholtz_3d_problem_interface(cfg)
%HELMHOLTZ_3D_PROBLEM_INTERFACE Geometry-pluggable 3D Helmholtz/PML problem.
%
% The returned callbacks are:
%   [A,b,meta] = problem.assemble(mu)
%   y = problem.apply(mu,u)
%   b = problem.rhs(mu)
%   d = problem.mass_diagonal(mu)
%   Ashift = problem.assemble_shifted(mu,beta)
%   partition = problem.partition(mu)
%   problem.affine contains an affine squared-slowness decomposition when
%   the selected built-in material model supports one.
%
% Custom material callbacks receive (x,y,z,mu), where x/y/z are interior
% coordinate vectors, and must return an [nx,ny,nz] wavespeed array. Custom
% source callbacks use the same signature and return an [nx,ny,nz] RHS.
% Custom partition callbacks return an integer label array of that size.

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);

    nphys = expand_three(cfg.Nphys, 'Nphys');
    npml = expand_three(cfg.npml, 'npml');
    lengths = expand_three(cfg.domain_lengths, 'domain_lengths');
    spacing = lengths ./ (nphys + 1);
    n = nphys + 2 * npml;

    coordinates = cell(1, 3);
    for d = 1:3
        pml_length = npml(d) * spacing(d);
        full_grid = linspace(-pml_length, lengths(d) + pml_length, n(d) + 2);
        coordinates{d} = full_grid(2:end - 1).';
    end

    pml = build_pml_3d(n, cfg.omega, npml, cfg.sigma_max, cfg.use_pml);
    K_pml = pml_stiffness_3d(n(1), n(2), n(3), ...
        spacing(1), spacing(2), spacing(3), pml);

    core = struct();
    core.cfg = cfg;
    core.nphys = nphys;
    core.npml = npml;
    core.n = n;
    core.lengths = lengths;
    core.spacing = spacing;
    core.coordinates = coordinates;
    core.pml = pml;
    core.K_pml = K_pml;
    core.N = prod(n);
    core.affine = build_affine_metadata(core);

    problem = core;
    problem.assemble = @(mu) assemble_problem(core, mu);
    problem.apply = @(mu, u) apply_problem(core, mu, u);
    problem.rhs = @(mu) reshape(source_grid(core, mu), [], 1);
    problem.mass_diagonal = @(mu) mass_diagonal(core, mu);
    problem.assemble_shifted = @(mu, beta) assemble_shifted(core, mu, beta);
    problem.partition = @(mu) build_partition(core, mu);
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'domain_lengths', [1, 1, 1]);
    cfg = set_default(cfg, 'Nphys', [31, 31, 31]);
    cfg = set_default(cfg, 'npml', []);
    cfg = set_default(cfg, 'omega', 20);
    cfg = set_default(cfg, 'sigma_max', 80);
    cfg = set_default(cfg, 'use_pml', true);
    cfg = set_default(cfg, 'material_model', 'halfspace');
    cfg = set_default(cfg, 'material_fun', []);
    cfg = set_default(cfg, 'partition_fun', []);
    cfg = set_default(cfg, 'source_fun', []);
    cfg = set_default(cfg, 'background_speed', 1.0);
    cfg = set_default(cfg, 'interface_x', 0.5 * cfg.domain_lengths(1));
    cfg = set_default(cfg, 'sphere_center', 0.5 * cfg.domain_lengths);
    cfg = set_default(cfg, 'sphere_radius', 0.2 * min(cfg.domain_lengths));
    cfg = set_default(cfg, 'gaussian_center', 0.5 * cfg.domain_lengths);
    cfg = set_default(cfg, 'gaussian_width', 0.16 * cfg.domain_lengths);
    cfg = set_default(cfg, 'waveguide_axis', 'x');
    cfg = set_default(cfg, 'waveguide_center', 0.5 * cfg.domain_lengths);
    cfg = set_default(cfg, 'waveguide_width', 0.16 * min(cfg.domain_lengths));
    cfg = set_default(cfg, 'active_profile_cutoff', 0.05);
    cfg = set_default(cfg, 'source_location', [0.2, 0.5, 0.5] .* cfg.domain_lengths);
    cfg = set_default(cfg, 'source_amplitude', 1.0);
    cfg = set_default(cfg, 'source_model', 'point');
    cfg = set_default(cfg, 'source_sigma', []);

    if isempty(cfg.npml)
        cfg.npml = ceil(0.15 * expand_three(cfg.Nphys, 'Nphys'));
    end
    cfg.material_model = lower(strtrim(char(string(cfg.material_model))));
    cfg.waveguide_axis = lower(strtrim(char(string(cfg.waveguide_axis))));
    cfg.source_model = lower(strtrim(char(string(cfg.source_model))));
    if ~ismember(cfg.waveguide_axis, {'x', 'y', 'z'})
        error('waveguide_axis must be x, y, or z.');
    end
    if ~ismember(cfg.source_model, {'point', 'gaussian'})
        error('source_model must be point or gaussian.');
    end
    if isempty(cfg.source_sigma)
        spacing = cfg.domain_lengths ./ (expand_three(cfg.Nphys, 'Nphys') + 1);
        cfg.source_sigma = 1.5 * min(spacing);
    end
end

function [A, b, meta] = assemble_problem(core, mu)
    [speed, material_mask] = material_grid(core, mu);
    mass_diag = (core.cfg.omega ^ 2) ./ (speed(:) .^ 2);
    A = core.K_pml + spdiags(mass_diag, 0, core.N, core.N);
    b = source_grid(core, mu);
    b = b(:);

    meta = struct();
    meta.parameter = mu;
    meta.min_speed = min(speed(:));
    meta.max_speed = max(speed(:));
    meta.material_mask = material_mask;
    meta.mass_diagonal = mass_diag;
end

function y = apply_problem(core, mu, u)
    if numel(u) ~= core.N
        error('Expected a vector of length %d.', core.N);
    end
    speed = material_grid(core, mu);
    y = core.K_pml * u + ((core.cfg.omega ^ 2) ./ (speed(:) .^ 2)) .* u;
end

function diagonal = mass_diagonal(core, mu)
    speed = material_grid(core, mu);
    diagonal = (core.cfg.omega ^ 2) ./ (speed(:) .^ 2);
end

function Ashift = assemble_shifted(core, mu, beta)
    if ~(isscalar(beta) && isfinite(beta) && beta > 0)
        error('beta must be a positive finite scalar.');
    end
    [A, ~, meta] = assemble_problem(core, mu);
    Ashift = A + 1i * beta * spdiags(meta.mass_diagonal, 0, core.N, core.N);
end

function [speed, material_mask] = material_grid(core, mu)
    cfg = core.cfg;
    x = core.coordinates{1};
    y = core.coordinates{2};
    z = core.coordinates{3};
    dims = core.n;

    if ~isempty(cfg.material_fun)
        speed = cfg.material_fun(x, y, z, mu);
        validate_grid(speed, dims, 'material_fun');
        material_mask = abs(speed - cfg.background_speed) > 10 * eps(cfg.background_speed);
        return;
    end

    speed = cfg.background_speed * ones(dims);
    switch cfg.material_model
        case 'halfspace'
            material_mask = repmat(reshape(x > cfg.interface_x, [], 1, 1), 1, dims(2), dims(3));
        case 'affine_halfspace'
            material_mask = repmat(reshape(x > cfg.interface_x, [], 1, 1), 1, dims(2), dims(3));
            squared_slowness = 1 / cfg.background_speed^2 + mu * double(material_mask);
            validate_squared_slowness(squared_slowness, mu, cfg.material_model);
            speed = 1 ./ sqrt(squared_slowness);
            return;
        case {'sphere', 'spherical_inclusion'}
            dx2 = reshape((x - cfg.sphere_center(1)).^2, [], 1, 1);
            dy2 = reshape((y - cfg.sphere_center(2)).^2, 1, [], 1);
            dz2 = reshape((z - cfg.sphere_center(3)).^2, 1, 1, []);
            material_mask = dx2 + dy2 + dz2 <= cfg.sphere_radius^2;
        case {'gaussian_lens', 'lens'}
            profile = gaussian_lens_profile(core);
            squared_slowness = 1 / cfg.background_speed^2 + mu * profile;
            validate_squared_slowness(squared_slowness, mu, cfg.material_model);
            speed = 1 ./ sqrt(squared_slowness);
            material_mask = profile >= cfg.active_profile_cutoff;
            return;
        case {'gaussian_waveguide', 'waveguide'}
            profile = gaussian_waveguide_profile(core);
            squared_slowness = 1 / cfg.background_speed^2 + mu * profile;
            validate_squared_slowness(squared_slowness, mu, cfg.material_model);
            speed = 1 ./ sqrt(squared_slowness);
            material_mask = profile >= cfg.active_profile_cutoff;
            return;
        case 'homogeneous'
            material_mask = false(dims);
        otherwise
            error('Unknown material_model "%s".', cfg.material_model);
    end
    speed(material_mask) = mu;
end

function b = source_grid(core, mu)
    cfg = core.cfg;
    x = core.coordinates{1};
    y = core.coordinates{2};
    z = core.coordinates{3};
    if ~isempty(cfg.source_fun)
        b = cfg.source_fun(x, y, z, mu);
        validate_grid(b, core.n, 'source_fun');
        return;
    end

    if strcmp(cfg.source_model, 'gaussian')
        dx2 = reshape((x - cfg.source_location(1)).^2, [], 1, 1);
        dy2 = reshape((y - cfg.source_location(2)).^2, 1, [], 1);
        dz2 = reshape((z - cfg.source_location(3)).^2, 1, 1, []);
        sigma2 = max(cfg.source_sigma^2, eps);
        b = cfg.source_amplitude * exp(-0.5 * (dx2 + dy2 + dz2) / sigma2);
        return;
    end

    [~, ix] = min(abs(x - cfg.source_location(1)));
    [~, iy] = min(abs(y - cfg.source_location(2)));
    [~, iz] = min(abs(z - cfg.source_location(3)));
    b = zeros(core.n);
    b(ix, iy, iz) = cfg.source_amplitude;
end

function affine = build_affine_metadata(core)
    cfg = core.cfg;
    affine = struct('is_affine', false, 'parameterization', '', ...
        'mass_base', [], 'mass_parameter_diagonal', []);

    switch cfg.material_model
        case 'affine_halfspace'
            profile = double(repmat(reshape(core.coordinates{1} > cfg.interface_x, ...
                [], 1, 1), 1, core.n(2), core.n(3)));
        case {'gaussian_lens', 'lens'}
            profile = gaussian_lens_profile(core);
        case {'gaussian_waveguide', 'waveguide'}
            profile = gaussian_waveguide_profile(core);
        otherwise
            return;
    end

    affine.is_affine = true;
    affine.parameterization = 'squared_slowness_contrast';
    affine.mass_base = core.cfg.omega^2 / cfg.background_speed^2;
    affine.mass_parameter_diagonal = core.cfg.omega^2 * profile(:);
end

function profile = gaussian_lens_profile(core)
    center = expand_three(core.cfg.gaussian_center, 'gaussian_center');
    width = expand_three(core.cfg.gaussian_width, 'gaussian_width');
    x = core.coordinates{1};
    y = core.coordinates{2};
    z = core.coordinates{3};
    dx2 = reshape(((x - center(1)) / width(1)).^2, [], 1, 1);
    dy2 = reshape(((y - center(2)) / width(2)).^2, 1, [], 1);
    dz2 = reshape(((z - center(3)) / width(3)).^2, 1, 1, []);
    profile = exp(-0.5 * (dx2 + dy2 + dz2));
end

function profile = gaussian_waveguide_profile(core)
    cfg = core.cfg;
    center = expand_three(cfg.waveguide_center, 'waveguide_center');
    widths = cfg.waveguide_width(:).';
    if isscalar(widths)
        widths = repmat(widths, 1, 2);
    end
    if numel(widths) ~= 2 || any(~isfinite(widths)) || any(widths <= 0)
        error('waveguide_width must be a positive scalar or two-vector.');
    end

    coordinates = core.coordinates;
    switch cfg.waveguide_axis
        case 'x'
            d1 = reshape(((coordinates{2} - center(2)) / widths(1)).^2, 1, [], 1);
            d2 = reshape(((coordinates{3} - center(3)) / widths(2)).^2, 1, 1, []);
            transverse = exp(-0.5 * (d1 + d2));
            profile = repmat(transverse, core.n(1), 1, 1);
        case 'y'
            d1 = reshape(((coordinates{1} - center(1)) / widths(1)).^2, [], 1, 1);
            d2 = reshape(((coordinates{3} - center(3)) / widths(2)).^2, 1, 1, []);
            transverse = exp(-0.5 * (d1 + d2));
            profile = repmat(transverse, 1, core.n(2), 1);
        case 'z'
            d1 = reshape(((coordinates{1} - center(1)) / widths(1)).^2, [], 1, 1);
            d2 = reshape(((coordinates{2} - center(2)) / widths(2)).^2, 1, [], 1);
            transverse = exp(-0.5 * (d1 + d2));
            profile = repmat(transverse, 1, 1, core.n(3));
    end
end

function validate_squared_slowness(values, mu, model)
    if any(~isfinite(values(:))) || any(values(:) <= 0)
        error('%s with mu=%.6g produced non-positive squared slowness.', model, mu);
    end
end

function partition = build_partition(core, mu)
    cfg = core.cfg;
    if ~isempty(cfg.partition_fun)
        labels = cfg.partition_fun(core.coordinates{1}, core.coordinates{2}, ...
            core.coordinates{3}, mu);
        validate_grid(labels, core.n, 'partition_fun');
    else
        [~, material_mask] = material_grid(core, mu);
        physical = false(core.n);
        ix = core.npml(1) + (1:core.nphys(1));
        iy = core.npml(2) + (1:core.nphys(2));
        iz = core.npml(3) + (1:core.nphys(3));
        physical(ix, iy, iz) = true;
        labels = ones(core.n);
        labels(physical & material_mask) = 2;
        labels(~physical) = 3;
    end

    unique_labels = unique(labels(:)).';
    indices = cell(numel(unique_labels), 1);
    for k = 1:numel(unique_labels)
        indices{k} = find(labels(:) == unique_labels(k));
    end
    partition = struct('labels', labels, 'unique_labels', unique_labels, ...
        'indices', {indices});
end

function pml = build_pml_3d(n, omega, npml, sigma_max, use_pml)
    names = {'x', 'y', 'z'};
    pml = struct();
    for d = 1:3
        sigma = zeros(n(d) + 2, 1);
        if use_pml
            layer = (npml(d):-1:1).' / max(npml(d), 1);
            values = sigma_max * layer.^2;
            sigma(1:npml(d)) = values;
            sigma(end - npml(d) + 1:end) = flipud(values);
        end
        stretch = 1 + 1i * sigma / omega;
        pml.(['sigma_' names{d}]) = sigma;
        pml.(['s_' names{d}]) = stretch;
        pml.(['s_' names{d} '_half']) = 0.5 * (stretch(1:end - 1) + stretch(2:end));
    end
end

function values = expand_three(values, name)
    values = values(:).';
    if isscalar(values)
        values = repmat(values, 1, 3);
    end
    if numel(values) ~= 3 || any(~isfinite(values)) || any(values <= 0)
        error('%s must be a positive scalar or three-vector.', name);
    end
    if any(strcmp(name, {'Nphys', 'npml'})) && any(values ~= round(values))
        error('%s must contain integers.', name);
    end
end

function validate_grid(values, dims, callback_name)
    if ~isequal(size(values), dims)
        error('%s must return an array of size [%s].', callback_name, num2str(dims));
    end
    if any(~isfinite(values(:)))
        error('%s returned non-finite values.', callback_name);
    end
end

function s = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end
