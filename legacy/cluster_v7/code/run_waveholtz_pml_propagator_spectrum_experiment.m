function results = run_waveholtz_pml_propagator_spectrum_experiment(cfg)
%RUN_WAVEHOLTZ_PML_PROPAGATOR_SPECTRUM_EXPERIMENT Small Route-A experiment.
%   Explicitly assembles the homogeneous one-step matrix G for the same
%   staggered leapfrog--CN PML update used by the active Helm workflow, then
%   compares eig(p_tau(G)) with p_tau(eig(G)).  Outputs are local-only under
%   outputs/pml_propagator_spectrum/.

    if nargin < 1, cfg = struct(); end
    cfg = defaults(cfg);

    nx = cfg.nphys_x + 2 * cfg.npml;
    ny = cfg.nphys_y + 2 * cfg.npml;
    hx = cfg.Lx / (cfg.nphys_x + 1);
    hy = cfg.Ly / (cfg.nphys_y + 1);
    if isempty(cfg.nt_override)
        [omega_ts, c_w, nt, dt] = match_period(cfg.omega, cfg.cfl * min(hx, hy) / cfg.c);
    else
        nt = cfg.nt_override;
        dt = 2 * sin(pi / nt) / cfg.omega;
        omega_ts = 2 * pi / (nt * dt);
        c_w = cos(pi / nt);
    end
    pml = build_pml(nx, ny, cfg.npml, cfg.sigma_max, c_w);

    dims = state_dims(nx, ny);
    if strcmp(cfg.theta_scheme, 'double_average')
        dims.nstate = dims.nstate + dims.nu;
    end
    fprintf('Assembling G: nx=%d, ny=%d, state dimension=%d, nt=%d\n', ...
        nx, ny, dims.nstate, nt);
    G = zeros(dims.nstate);
    eye_state = eye(dims.nstate);
    for j = 1:dims.nstate
        G(:, j) = homogeneous_step(eye_state(:, j), nx, ny, hx, hy, cfg.c, ...
            dt, c_w, pml, cfg.theta_scheme);
    end

    lambda_G = eig(G);
    continuous_shift = cfg.filter_shift;
    discrete_shift = 0.25 * (1 - tan(pi / nt)^2);
    standard = analyze_filter(G, lambda_G, nt, dt, omega_ts, continuous_shift);
    corrected = analyze_filter(G, lambda_G, nt, dt, omega_ts, discrete_shift);
    rho_G = max(abs(lambda_G));
    target = exp(-1i * omega_ts * dt);

    fprintf('rho(G)                    = %.16g\n', rho_G);
    print_filter_summary('standard', standard);
    print_filter_summary('corrected', corrected);
    fprintf('distance to target points = %.3e\n', ...
        min([abs(lambda_G - target); abs(lambda_G - conj(target))]));

    results = struct('cfg', cfg, 'nx', nx, 'ny', ny, 'hx', hx, 'hy', hy, ...
        'omega_ts', omega_ts, 'c_w', c_w, 'nt', nt, 'dt', dt, ...
        'G', G, 'lambda_G', lambda_G, 'rho_G', rho_G, ...
        'standard', standard, 'corrected', corrected);
    if cfg.make_plot || cfg.save_outputs
        outdir = helm_output_path('outputs', 'pml_propagator_spectrum');
        if ~exist(outdir, 'dir'), mkdir(outdir); end
    end
    if cfg.make_plot
        fig = figure('Color', 'w', 'Position', [80 100 1500 450]);
        tiledlayout(fig, 1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
        ax = nexttile; plot_spectrum(ax, lambda_G, target, '\sigma(G_\tau)');
        ax = nexttile; plot_spectrum(ax, standard.mapped_lambda_G, 1, ...
            sprintf('standard shift %.6f', standard.shift));
        ax = nexttile; plot_spectrum(ax, corrected.mapped_lambda_G, 1, ...
            sprintf('discrete shift %.6f', corrected.shift));
        exportgraphics(fig, fullfile(outdir, 'propagator_filter_spectrum.png'), 'Resolution', 180);
    end
    if cfg.save_outputs
        save(fullfile(outdir, 'propagator_filter_spectrum.mat'), 'results');
    end
end

function cfg = defaults(cfg)
    cfg = set_default(cfg, 'nphys_x', 4);
    cfg = set_default(cfg, 'nphys_y', 4);
    cfg = set_default(cfg, 'npml', 2);
    cfg = set_default(cfg, 'Lx', 1);
    cfg = set_default(cfg, 'Ly', 1);
    cfg = set_default(cfg, 'c', 1);
    cfg = set_default(cfg, 'omega', 8);
    cfg = set_default(cfg, 'cfl', 0.35);
    cfg = set_default(cfg, 'sigma_max', 40);
    cfg = set_default(cfg, 'filter_shift', 0.25);
    cfg = set_default(cfg, 'nt_override', []);
    cfg = set_default(cfg, 'make_plot', true);
    cfg = set_default(cfg, 'save_outputs', true);
    cfg = set_default(cfg, 'theta_scheme', 'target_constant');
    if ~ismember(cfg.theta_scheme, {'target_constant', 'double_average'})
        error('theta_scheme must be target_constant or double_average.');
    end
end

function dims = state_dims(nx, ny)
    dims.nu = nx * ny;
    dims.nvx = (nx + 1) * ny;
    dims.nvy = nx * (ny + 1);
    dims.nv = dims.nvx + dims.nvy;
    dims.nstate = dims.nu + dims.nv + 3 * dims.nu;
end

function xnew = homogeneous_step(x, nx, ny, hx, hy, c, dt, c_w, pml, theta_scheme)
    d = state_dims(nx, ny);
    use_double_average = strcmp(theta_scheme, 'double_average');
    sx = repmat(pml.sigma_x(:), 1, ny + 2);
    sy = repmat(pml.sigma_y(:).', nx + 2, 1);
    sx_vx = 0.5 * (sx(1:end-1, :) + sx(2:end, :));
    sy_vy = 0.5 * (sy(:, 1:end-1) + sy(:, 2:end));

    u = zeros(nx + 2, ny + 2);
    u(2:end-1, 2:end-1) = reshape(x(1:d.nu), nx, ny);
    vx = zeros(nx + 1, ny + 2);
    vy = zeros(nx + 2, ny + 1);
    vx(:, 2:end-1) = reshape(x(d.nu + (1:d.nvx)), nx + 1, ny);
    vy(2:end-1, :) = reshape(x(d.nu + d.nvx + (1:d.nvy)), nx, ny + 1);
    k = d.nu + d.nv;
    psi1 = zeros(nx + 2, ny + 2); psi2 = psi1; theta = psi1;
    psi1(2:end-1, 2:end-1) = reshape(x(k + (1:d.nu)), nx, ny);
    psi2(2:end-1, 2:end-1) = reshape(x(k + d.nu + (1:d.nu)), nx, ny);
    theta(2:end-1, 2:end-1) = reshape(x(k + 2*d.nu + (1:d.nu)), nx, ny);
    if use_double_average
        u_prev = zeros(nx + 2, ny + 2);
        u_prev(2:end-1, 2:end-1) = reshape(x(d.nstate + (1:d.nu)), nx, ny);
    end

    du_dx = (u(2:end, :) - u(1:end-1, :)) / hx;
    du_dy = (u(:, 2:end) - u(:, 1:end-1)) / hy;
    vx_new = ((1 - 0.5*dt*sx_vx).*vx + dt*c^2*du_dx) ./ (1 + 0.5*dt*sx_vx);
    vy_new = ((1 - 0.5*dt*sy_vy).*vy + dt*c^2*du_dy) ./ (1 + 0.5*dt*sy_vy);
    vx_avg = 0.5 * (vx + vx_new); vy_avg = 0.5 * (vy + vy_new);
    tmp1 = zeros(nx + 2, ny + 2); tmp2 = tmp1;
    tmp1(2:end-1, :) = (vx_avg(2:end, :) - vx_avg(1:end-1, :)) / hx;
    tmp2(:, 2:end-1) = (vy_avg(:, 2:end) - vy_avg(:, 1:end-1)) / hy;
    psi1 = psi1 + dt * sy .* tmp1;
    psi2 = psi2 + dt * sx .* tmp2;
    div_v = zeros(nx + 2, ny + 2);
    div_v(2:end-1, :) = (vx_new(2:end, :) - vx_new(1:end-1, :)) / hx;
    div_v(:, 2:end-1) = div_v(:, 2:end-1) + (vy_new(:, 2:end) - vy_new(:, 1:end-1)) / hy;
    sigu = sx + sy;
    uint_old = u(2:end-1, 2:end-1);
    if use_double_average
        % Route C: theta_t = sigma_x*sigma_y*mu_t^2(u).  The occurrence
        % of u^{n+1} is eliminated pointwise together with the u update.
        Ktheta = (sx(2:end-1, 2:end-1) .* sy(2:end-1, 2:end-1));
        theta_old = theta(2:end-1, 2:end-1);
        uprev_int = u_prev(2:end-1, 2:end-1);
        rhs_without_theta = div_v(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) + psi2(2:end-1, 2:end-1);
        coupling = 0.25 * dt^2 * Ktheta;
        uint = ((1 - 0.5*dt*sigu(2:end-1, 2:end-1)).*uint_old ...
            + dt*(rhs_without_theta - theta_old) ...
            - coupling.*(2*uint_old + uprev_int)) ...
            ./ (1 + 0.5*dt*sigu(2:end-1, 2:end-1) + coupling);
        theta(2:end-1, 2:end-1) = theta_old ...
            + dt*Ktheta.*(uint + 2*uint_old + uprev_int)/4;
    else
        theta = theta + dt * (sx .* sy) * c_w^2 .* u;
        rhs_u = div_v(2:end-1, 2:end-1) - theta(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) + psi2(2:end-1, 2:end-1);
        uint = ((1 - 0.5*dt*sigu(2:end-1, 2:end-1)).*uint_old ...
            + dt*rhs_u) ./ (1 + 0.5*dt*sigu(2:end-1, 2:end-1));
    end
    vxint = vx_new(:, 2:end-1); vyint = vy_new(2:end-1, :);
    psi1int = psi1(2:end-1, 2:end-1); psi2int = psi2(2:end-1, 2:end-1);
    thetaint = theta(2:end-1, 2:end-1);
    xnew = [uint(:); vxint(:); vyint(:); psi1int(:); psi2int(:); thetaint(:)];
    if use_double_average
        xnew = [xnew; uint_old(:)];
    end
end

function coeff = filter_coefficients(nt, dt, omega_ts, shift)
    t = (0:nt).' * dt;
    trap = ones(nt + 1, 1); trap([1 end]) = 0.5;
    coeff = (2 / (nt * dt)) * dt * trap .* (cos(omega_ts * t) - shift);
end

function data = analyze_filter(G, lambda_G, nt, dt, omega_ts, shift)
    coeff = filter_coefficients(nt, dt, omega_ts, shift);
    mapped = polyval(fliplr(coeff.'), lambda_G);
    S = coeff(end) * eye(size(G));
    for n = (numel(coeff) - 1):-1:1
        S = G * S + coeff(n) * eye(size(G));
    end
    lambda_S = eig(S);
    phi = linspace(0, 2*pi, 200001).';
    circle_values = polyval(fliplr(coeff.'), exp(-1i * phi));
    target_phase = omega_ts * dt;
    target = exp(-1i * target_phase);
    p_target = polyval(fliplr(coeff.'), target);
    dp_target = polyval(polyder(fliplr(coeff.')), target);
    phase_derivative = -1i * target * dp_target;
    modulus_derivative = real(phase_derivative * conj(p_target)) / abs(p_target);
    [circle_max, max_idx] = max(abs(circle_values));
    circle_max = max(circle_max, abs(p_target));
    data = struct('shift', shift, 'filter_coefficients', coeff, 'S', S, ...
        'lambda_S', lambda_S, 'mapped_lambda_G', mapped, ...
        'rho_S', max(abs(lambda_S)), 'norm_S', norm(S, 2), ...
        'max_filter_on_unit_circle', circle_max, ...
        'unit_circle_max_phase', phi(max_idx), 'target_gain', p_target, ...
        'target_modulus_phase_derivative', modulus_derivative, ...
        'spectral_mapping_set_error', spectral_set_distance(lambda_S, mapped));
end

function print_filter_summary(label, data)
    fprintf('\n%s filter: shift = %.16g\n', label, data.shift);
    fprintf('  rho(p_tau(G))             = %.16g\n', data.rho_S);
    fprintf('  max |p_tau(lambda(G))|    = %.16g\n', max(abs(data.mapped_lambda_G)));
    fprintf('  ||p_tau(G)||_2            = %.16g\n', data.norm_S);
    fprintf('  max unit-circle |p_tau|   = %.16g\n', data.max_filter_on_unit_circle);
    fprintf('  target gain               = %.16g %+.3ei\n', ...
        real(data.target_gain), imag(data.target_gain));
    fprintf('  d|p(e^{-i phi})|/dphi     = %.3e\n', data.target_modulus_phase_derivative);
    fprintf('  spectral mapping set error= %.3e\n', data.spectral_mapping_set_error);
end

function [omega_ts, c_w, nt, dt] = match_period(omega, dt)
    for k = 1:8
        omega_ts = (2 / dt) * asin(omega * dt / 2);
        nt = max(10, round(2*pi / (omega_ts*dt)));
        dt = 2*pi / (omega_ts*nt);
    end
    omega_ts = (2 / dt) * asin(omega * dt / 2);
    c_w = cos(omega_ts * dt / 2);
end

function pml = build_pml(nx, ny, npml, sigma_max, c_w)
    sx = zeros(nx + 2, 1); sy = zeros(ny + 2, 1);
    for i = 1:(npml + 1)
        val = sigma_max * ((npml - (i - 1)) / npml)^2 / c_w;
        sx(i) = val; sx(end + 1 - i) = val;
        sy(i) = val; sy(end + 1 - i) = val;
    end
    pml = struct('sigma_x', sx, 'sigma_y', sy);
end

function err = spectral_set_distance(a, b)
    err = max([max(arrayfun(@(z) min(abs(z - b)), a)); ...
               max(arrayfun(@(z) min(abs(z - a)), b))]);
end

function plot_spectrum(ax, z, marked, ttl)
    th = linspace(0, 2*pi, 500);
    plot(ax, cos(th), sin(th), 'k:', 'LineWidth', 1); hold(ax, 'on');
    plot(ax, real(z), imag(z), '.', 'MarkerSize', 10);
    plot(ax, real(marked), imag(marked), 'rx', 'LineWidth', 1.5, 'MarkerSize', 9);
    axis(ax, 'equal'); grid(ax, 'on'); xlabel(ax, 'Re'); ylabel(ax, 'Im'); title(ax, ttl);
end

function s = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name)), s.(name) = value; end
end
