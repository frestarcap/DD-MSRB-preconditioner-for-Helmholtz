function results = waveholtz_2d_pml_v5(cfg)
    %WAVEHOLTZ_2D_PML_V5 Five-source WaveHoltz/PML consistency experiment.
    %
    % cfg.theta_scheme:
    %   'target_constant' (default) uses c_w^2*u^n in the theta update.
    %   'double_average' uses Route C,
    %       (u^{n+1}+2*u^n+u^{n-1})/4,
    %   and augments the fixed-point state with u^{n-1}.
    % cfg.filter_shift_mode is 'standard' (1/4) or 'corrected'.
    %
    % Saves MAT/CSV/settings/summary data under outputs/pml_v5_consistency
    % and paper figures under outputs/figures/pml_v5_consistency.

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);
    rng(cfg.random_seed);

    L = cfg.L;
    c = cfg.c;
    omega = cfg.omega;
    max_iter = cfg.max_iter;
    tol = cfg.tol;

    Nphys = cfg.Nphys;
    nx_phys = Nphys; ny_phys = Nphys;
    hx = L/(nx_phys+1);
    hy = L/(ny_phys+1);

    % ---- PML outside physical domain ----
    ispml = 1;
    npml = cfg.npml;
    sigma_max = cfg.sigma_max;

    % ---- Time step + frequency mapping ----
    CFL = cfg.CFL;
    dt0 = CFL*min(hx,hy)/c;
    [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt0);
    T_ts = nt*dt;

    fprintf('=== Time stepping match ===\n');
    fprintf('omega(helm)    = %.6f\n', omega);
    fprintf('omega(ts)      = %.6f\n', omega_ts);
    fprintf('c_w            = %.6f\n', c_w);
    fprintf('dt             = %.3e, nt = %d, T_ts = %.6f\n', dt, nt, T_ts);

    % ---- Total computational interior unknowns ----
    nx = nx_phys + 2*npml;
    ny = ny_phys + 2*npml;

    % ---- Total domain coordinates (physical [0,L] padded by PML) ----
    Lpml_x = npml*hx;
    Lpml_y = npml*hy;
    x = linspace(-Lpml_x, L+Lpml_x, nx+2);
    y = linspace(-Lpml_y, L+Lpml_y, ny+2);
    [X,Y] = meshgrid(x,y);

    % ---- Physical interior indices in total interior grid ----
    I = (npml+1):(npml+nx_phys);
    J = (npml+1):(npml+ny_phys);

    % ---- Helmholtz-side PML (target sigma) ----
    pml_helm = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml);
    A = helmholtz_with_pml(nx, ny, hx, hy, omega, pml_helm);

    % ---- Time-stepping-side PML (sigma_ts * c_w = sigma_target) ----
    pml_wave = pml_helm;
    pml_wave.sigma_x = pml_helm.sigma_x /c_w;
    pml_wave.sigma_y = pml_helm.sigma_y /c_w;

    % ---- Test sources ----
    source_list = string(cfg.source_list);
    ns = numel(source_list);
    physical_mask = false(nx, ny);
    physical_mask(I, J) = true;
    pml_mask = ~physical_mask;

    results = struct();
    results.cfg = cfg;
    results.source_list = source_list;
    results.grid = struct('nx_phys', nx_phys, 'ny_phys', ny_phys, 'nx', nx, 'ny', ny, ...
        'hx', hx, 'hy', hy, 'I', I, 'J', J, 'x', x, 'y', y);
    results.matching = struct('omega_helm', omega, 'omega_ts', omega_ts, ...
        'c_w', c_w, 'dt', dt, 'nt', nt, 'T_ts', T_ts);
    results.rhs = zeros(nx * ny, ns);
    results.u_helm = zeros(nx * ny, ns);
    results.u_wave = zeros(nx * ny, ns);
    results.relerr_full = zeros(ns, 1);
    results.relerr_phys = zeros(ns, 1);
    results.relerr_pml = zeros(ns, 1);
    results.helm_residual = zeros(ns, 1);
    results.wave_residual = zeros(ns, 1);
    results.iterations = zeros(ns, 1);
    results.hist = cell(ns, 1);
    results.helm_solve_time = zeros(ns, 1);
    results.wave_solve_time = zeros(ns, 1);
    results.err_helm_exact = NaN(ns, 1);
    results.err_wave_exact = NaN(ns, 1);

    for sid = 1:ns
        source_type = source_list(sid);
        fprintf('\n===== Source: %s =====\n', source_type);

        [f_int, u_exact] = build_rhs_case(source_type, A, nx, ny, hx, hy, L, X, Y, I, J);

        % Helmholtz solve (total domain)
        thelm = tic;
        u_helm_vec = A \ f_int;
        results.helm_solve_time(sid) = toc(thelm);
        u_helm = zeros(nx+2, ny+2);
        u_helm(2:end-1,2:end-1) = reshape(u_helm_vec, nx, ny);

        % WaveHoltz solve (time stepping at omega_ts, target omega in forcing scale)
        twave = tic;
        [u_wave, hist] = waveholtz_iteration( ...
            nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, f_int, ...
            pml_wave, max_iter, tol, A, cfg.theta_scheme, cfg.filter_shift_mode);
        results.wave_solve_time(sid) = toc(twave);

        u_wave_vec = reshape(u_wave(2:end-1, 2:end-1), nx * ny, 1);
        results.rhs(:, sid) = f_int;
        results.u_helm(:, sid) = u_helm_vec;
        results.u_wave(:, sid) = u_wave_vec;
        reference_norm = max(norm(u_helm_vec), 1e-15);
        results.relerr_full(sid) = norm(u_wave_vec - u_helm_vec) / reference_norm;
        results.relerr_phys(sid) = norm(u_wave_vec(physical_mask(:)) - u_helm_vec(physical_mask(:))) / reference_norm;
        results.relerr_pml(sid) = norm(u_wave_vec(pml_mask(:)) - u_helm_vec(pml_mask(:))) / reference_norm;
        results.helm_residual(sid) = norm(A * u_helm_vec - f_int) / max(norm(f_int), 1e-15);
        results.wave_residual(sid) = norm(A * u_wave_vec - f_int) / max(norm(f_int), 1e-15);
        results.iterations(sid) = numel(hist);
        results.hist{sid} = hist;
        fprintf('full/physical/PML phasor errors: %.3e / %.3e / %.3e\n', ...
            results.relerr_full(sid), results.relerr_phys(sid), results.relerr_pml(sid));
        fprintf('WaveHoltz Helmholtz residual: %.3e\n', results.wave_residual(sid));

        if ~isempty(u_exact)
            exact_vec = reshape(u_exact(2:end-1, 2:end-1), nx * ny, 1);
            results.err_helm_exact(sid) = relerr_local(u_helm_vec(physical_mask(:)), exact_vec(physical_mask(:)));
            results.err_wave_exact(sid) = relerr_local(u_wave_vec(physical_mask(:)), exact_vec(physical_mask(:)));
        end
    end

    results.summary = struct();
    results.summary.relerr_full = stats3_local(results.relerr_full);
    results.summary.relerr_phys = stats3_local(results.relerr_phys);
    results.summary.relerr_pml = stats3_local(results.relerr_pml);
    results.summary.wave_residual = stats3_local(results.wave_residual);
    results.summary.iterations = stats3_local(results.iterations);
    results.summary.wave_solve_time = stats3_local(results.wave_solve_time);

    if strlength(cfg.output_name) == 0
        if strcmp(cfg.theta_scheme, 'double_average')
            output_name = 'pml_v5_consistency_route_c';
        else
            output_name = 'pml_v5_consistency';
        end
    else
        output_name = char(cfg.output_name);
    end
    out_dir = helm_output_path('outputs', output_name);
    fig_dir = helm_output_path('outputs', 'figures', output_name);
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
    results.output_dir = out_dir;
    results.figure_dir = fig_dir;
    results_table = table(source_list(:), results.iterations, results.relerr_full, ...
        results.relerr_phys, results.relerr_pml, results.wave_residual, ...
        results.helm_residual, results.wave_solve_time, results.helm_solve_time, ...
        results.err_helm_exact, results.err_wave_exact, ...
        'VariableNames', {'source', 'iterations', 'phasor_error_full', ...
        'phasor_error_physical', 'phasor_error_pml', 'waveholtz_helmholtz_residual', ...
        'direct_helmholtz_residual', 'waveholtz_time', 'helmholtz_time', ...
        'helmholtz_exact_error', 'waveholtz_exact_error'});
    results.results_table = results_table;
    writetable(results_table, fullfile(out_dir, 'pml_v5_consistency.csv'));
    save(fullfile(out_dir, 'pml_v5_consistency.mat'), 'results', '-v7.3');
    write_v5_settings(results, out_dir);
    write_v5_summary(results, out_dir);
    make_v5_paper_figures(results, fig_dir, out_dir);
end


% -------------------------------------------------------------------------
% Build PML coefficients on TOTAL domain: sigma nonzero only in OUTER layers
% NOTE: stretching uses omega_stretch (we pass omega_eff here)
function pml = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml)
    sigma_x = zeros(nx+2,1);
    sigma_y = zeros(ny+2,1);

    for i = 1:npml
        if ispml == 1
            val = sigma_max*((npml - (i-1))/npml)^2;
        else
            val = 0;
        end
        sigma_x(i)       = val;
        sigma_x(end+1-i) = val;
        sigma_y(i)       = val;
        sigma_y(end+1-i) = val;
    end

    % complex stretching factors
    s_x = 1.0 + 1i*sigma_x/omega;
    s_y = 1.0 + 1i*sigma_y/omega;

    % half-step averages along axes
    s_x_half = 0.5*(s_x(1:end-1) + s_x(2:end));
    s_y_half = 0.5*(s_y(1:end-1) + s_y(2:end));

    pml = struct('sigma_x',sigma_x, 'sigma_y',sigma_y, ...
                 's_x',s_x, 's_y',s_y, ...
                 's_x_half',s_x_half, 's_y_half',s_y_half);
end


% -------------------------------------------------------------------------
% Helmholtz with PML (variable-coefficient discretization on interior unknowns)
% A ≈ (Δ_PML + omega_helm^2 I)
function A_helm_pml = helmholtz_with_pml(nx, ny, hx, hy, omega, pml)
    Ntot = nx*ny;
    A = spalloc(Ntot, Ntot, 5*Ntot);
    idx = @(i,j) i + (j-1)*nx;

    for i = 1:nx
        for j = 1:ny
            row = idx(i,j);

            % x-direction coefficients
            a_x = 1.0 ./ pml.s_x(i+1);
            a_p = 1.0 ./ pml.s_x_half(i+1);
            a_m = 1.0 ./ pml.s_x_half(i);

            A(row,row) = A(row,row) + (-(a_p + a_m)*a_x)/(hx*hx);
            if i+1 <= nx, A(row, idx(i+1,j)) = A(row, idx(i+1,j)) + (a_x*a_p)/(hx*hx); end
            if i-1 >= 1,  A(row, idx(i-1,j)) = A(row, idx(i-1,j)) + (a_x*a_m)/(hx*hx); end

            % y-direction coefficients
            b_y = 1.0 ./ pml.s_y(j+1);
            b_p = 1.0 ./ pml.s_y_half(j+1);
            b_m = 1.0 ./ pml.s_y_half(j);

            A(row,row) = A(row,row) + (-(b_p + b_m)*b_y)/(hy*hy);
            if j+1 <= ny, A(row, idx(i,j+1)) = A(row, idx(i,j+1)) + (b_y*b_p)/(hy*hy); end
            if j-1 >= 1,  A(row, idx(i,j-1)) = A(row, idx(i,j-1)) + (b_y*b_m)/(hy*hy); end
        end
    end

    A_helm_pml = A + (omega^2)*speye(Ntot);
end

% -------------------------------------------------------------------------
% Time stepping one period with staggered velocities and PML auxiliaries
% Outputs full-state snapshots. Route C appends the history variable u^{n-1}.
function W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, ...
        dt, nt, pml, x0, f_int, theta_scheme)

    % Expand sigma to full 2D nodes
    sx_full = zeros(nx+2, ny+2);
    sy_full = zeros(nx+2, ny+2);
    sx = pml.sigma_x; sy = pml.sigma_y;
    for i = 1:(nx+2), sx_full(i,:) = sx(i); end
    for j = 1:(ny+2), sy_full(:,j) = sy(j); end

    % Half-step damping at faces
    sigma_x_vx = 0.5*(sx_full(1:end-1,:) + sx_full(2:end,:)); % (nx+1, ny+2)
    sigma_y_vy = 0.5*(sy_full(:,1:end-1) + sy_full(:,2:end)); % (nx+2, ny+1)

    nu = nx*ny;
    nvx = (nx+1) * ny;
    nvy = nx * (ny+1);
    nv = nvx + nvy;
    naux = 3*nu;
    use_double_average = strcmp(theta_scheme, 'double_average');
    nstate_base = nu + nv + naux;
    nstate = nstate_base + use_double_average * nu;
    if numel(x0) ~= nstate
        error('x0 length mismatch: expect %d, got %d', nstate, numel(x0));
    end

    u0_int = x0(1:nu);
    v0_int = x0(nu+1:nu+nv);
    psi1_int = x0(nu+nv+1 : nu+nv+nu);
    psi2_int = x0(nu+nv+nu+1 : nu+nv+2*nu);
    theta_int = x0(nu+nv+2*nu+1 : nstate_base);
    if use_double_average
        u_prev_int = x0(nstate_base+1 : nstate);
    else
        u_prev_int = [];
    end

    % Displacement u on nodes with Dirichlet 0 boundary
    u = zeros(nx+2, ny+2);
    u(2:end-1, 2:end-1) = reshape(u0_int, nx, ny);

    % Staggered velocities (vx, vy): v0 stores [vx(:,2:end-1); vy(2:end-1,:)]
    vx = zeros(nx+1, ny+2);
    vy = zeros(nx+2, ny+1);
    vx_int = reshape(v0_int(1:nvx), nx+1, ny);
    vy_int = reshape(v0_int(nvx+1:end), nx, ny+1);
    vx(:, 2:end-1) = vx_int;
    vy(2:end-1, :) = vy_int;

    % PML auxiliary variables (node-centered, interior stored in state)
    psi1  = zeros(nx+2, ny+2);
    psi2  = zeros(nx+2, ny+2);
    theta = zeros(nx+2, ny+2);
    psi1(2:end-1,2:end-1) = reshape(psi1_int, nx, ny);
    psi2(2:end-1,2:end-1) = reshape(psi2_int, nx, ny);
    theta(2:end-1,2:end-1) = reshape(theta_int, nx, ny);
    if use_double_average
        u_prev = zeros(nx+2, ny+2);
        u_prev(2:end-1,2:end-1) = reshape(u_prev_int, nx, ny);
    end

    f_2d = reshape(f_int, nx, ny);

    % Half-step scheme constants from derivation
    c_w = cos(omega_ts*dt/2);
    sigu = sx_full + sy_full;


    % Save v^0 (integer-time value) as the provided initial velocity
    v0_vec = [reshape(vx(:,2:end-1), nvx, 1); reshape(vy(2:end-1,:), nvy, 1)];

    % Snapshot matrix: [u; v; psi1; psi2; theta]
    W = zeros(nt+1, nstate);

    % n=0 snapshot: [u^0; v^{-1/2}; psi^{-1/2}; theta^{-1/2}]
    u0_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
    snapshot0 = [u0_vec; v0_vec; psi1_int; psi2_int; theta_int];
    if use_double_average
        snapshot0 = [snapshot0; u_prev_int];
    end
    W(1,:) = snapshot0.';

    for n = 1:nt
        t = (n-1)*dt;
        t_half = t + 0.5*dt;

        % (2) Velocity update: v^{n-1/2} -> v^{n+1/2}  (CN in sigma)
        du_dx = (u(2:end,:) - u(1:end-1,:))/hx;
        du_dy = (u(:,2:end) - u(:,1:end-1))/hy;
        vx_new = ( (1 - 0.5*dt*sigma_x_vx).*vx + dt*c^2*du_dx ) ./ (1 + 0.5*dt*sigma_x_vx);
        vy_new = ( (1 - 0.5*dt*sigma_y_vy).*vy + dt*c^2*du_dy ) ./ (1 + 0.5*dt*sigma_y_vy);

        % Midpoint velocity for auxiliary equations
        vx_avg = 0.5*(vx + vx_new);
        vy_avg = 0.5*(vy + vy_new);

        % (3) Psi updates at half step
        grad_vx = (vx_avg(2:end,:) - vx_avg(1:end-1,:))/hx;    % (nx, ny+2)
        tmp1 = zeros(nx+2, ny+2); tmp1(2:end-1,:) = grad_vx;

        grad_vy = (vy_avg(:,2:end) - vy_avg(:,1:end-1))/hy;    % (nx+2, ny)
        tmp2 = zeros(nx+2, ny+2); tmp2(:,2:end-1) = grad_vy;

        psi1 = psi1 + dt * ( sy_full .* tmp1 );
        psi2 = psi2 + dt * ( sx_full .* tmp2 );

        % Divergence of v^{n+1/2}
        div_v = zeros(nx+2, ny+2);
        div_v(2:end-1,:) = div_v(2:end-1,:) + (vx_new(2:end,:) - vx_new(1:end-1,:))/hx;
        div_v(:,2:end-1) = div_v(:,2:end-1) + (vy_new(:,2:end) - vy_new(:,1:end-1))/hy;

        % Forcing (physical omega)
        forcing = -(1.0/omega_helm) * f_2d .* sin(omega_ts * t_half);

        % (4) Theta and u update. Route C uses
        % theta_t = sigma_x*sigma_y*(u^{n+1}+2u^n+u^{n-1})/4.
        % Its u^{n+1} coupling is diagonal and eliminated pointwise.
        u_old = u(2:end-1,2:end-1);
        sigu_int = sigu(2:end-1,2:end-1);
        rhs_without_theta = ...
              div_v(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) ...
            + psi2(2:end-1, 2:end-1) ...
            + forcing;
        if use_double_average
            Ktheta = sx_full(2:end-1,2:end-1) .* sy_full(2:end-1,2:end-1);
            theta_old = theta(2:end-1,2:end-1);
            uprev_old = u_prev(2:end-1,2:end-1);
            coupling = 0.25 * dt^2 * Ktheta;
            num_u = (1 - 0.5*dt*sigu_int).*u_old ...
                + dt*(rhs_without_theta - theta_old) ...
                - coupling.*(2*u_old + uprev_old);
            den_u = 1 + 0.5*dt*sigu_int + coupling;
            u_new = num_u ./ den_u;
            theta(2:end-1,2:end-1) = theta_old ...
                + dt*Ktheta.*(u_new + 2*u_old + uprev_old)/4;
            u_prev(2:end-1,2:end-1) = u_old;
        else
            theta = theta + dt * ((sx_full .* sy_full) .* (c_w^2) .* u);
            rhs_u = rhs_without_theta - theta(2:end-1,2:end-1);
            num_u = (1 - 0.5*dt*sigu_int).*u_old + dt*rhs_u;
            den_u = 1 + 0.5*dt*sigu_int;
            u_new = num_u ./ den_u;
        end
        u(2:end-1,2:end-1) = u_new;

        % Shift half-step velocities
        vx = vx_new;
        vy = vy_new;

        v_vec = [reshape(vx(:,2:end-1), nvx, 1); reshape(vy(2:end-1,:), nvy, 1)];
        u_vec = reshape(u(2:end-1, 2:end-1), nu, 1);

        psi1_vec = reshape(psi1(2:end-1,2:end-1), nu, 1);
        psi2_vec = reshape(psi2(2:end-1,2:end-1), nu, 1);
        theta_vec = reshape(theta(2:end-1,2:end-1), nu, 1);

        snapshot = [u_vec; v_vec; psi1_vec; psi2_vec; theta_vec];
        if use_double_average
            uprev_vec = reshape(u_prev(2:end-1,2:end-1), nu, 1);
            snapshot = [snapshot; uprev_vec];
        end
        W(n+1,:) = snapshot.';
    end
end

% -------------------------------------------------------------------------
% WaveHoltz fixed-point iteration using one-period time stepping
% Now iterates on x = [u0; v0] and filters both u and v together.
function [u_full, hist] = waveholtz_iteration(nx, ny, hx, hy, c, omega_ts, ...
        omega_helm, T_ts, dt, nt, f_int, pml, max_iter, tol, A, ...
        theta_scheme, filter_shift_mode)

    nu = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nv = nvx + nvy;

    use_double_average = strcmp(theta_scheme, 'double_average');
    x = zeros(nu + nv + 3*nu + use_double_average*nu, 1);
    hist = zeros(max_iter,1);

    for k = 1:max_iter

        W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, ...
            dt, nt, pml, x, f_int, theta_scheme);

        if strcmp(filter_shift_mode, 'corrected')
            filter_shift = 0.25 * (1 - tan(pi/nt)^2);
        else
            filter_shift = 0.25;
        end
        t_grid = (0:nt).' * dt;            % nt+1 points, includes t=T
        weights = cos(omega_ts*t_grid) - filter_shift;
        
        integrand = bsxfun(@times, weights, W);   % apply weights row-wise
        integral = trapz(t_grid, integrand, 1);
        
        x_new = (2.0/T_ts) * integral.';      % now includes both u and v

        
        res = norm(x_new - x) / max(1e-16, norm(x_new));
        hist(k) = res;
        x = x_new;

        fprintf('iter %3d: rel-res = %.3e\n', k, res);
        if res < tol
            hist = hist(1:k);
            break;
        end
    end

    % Recover the complex phasor by projecting the converged periodic orbit.
    % This uses the same matched temporal frequency as the WaveHoltz filter
    % and remains valid in the PML, where a velocity-divergence shortcut does
    % not reconstruct the damped auxiliary-variable contribution.
    Wconv = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, ...
        dt, nt, pml, x, f_int, theta_scheme);
    t_grid = (0:nt).' * dt;
    q_cos = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, cos(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';
    q_sin = (2.0 / T_ts) * trapz(t_grid, bsxfun(@times, sin(omega_ts * t_grid), Wconv(:, 1:nu)), 1).';

    u_full = zeros(nx+2, ny+2);
    u_full(2:end-1, 2:end-1) = reshape(q_cos + 1i * q_sin, nx, ny);

end

% -------------------------------------------------------------------------
% RHS builder for each source case
function [f_int, u_exact] = build_rhs_case(source_type, A, nx, ny, hx, hy, L, X, Y, I, J)
    u_exact = [];

    if source_type == "sin"
        u_exact = zeros(nx+2, ny+2);
        Xp = X(I+1, J+1);
        Yp = Y(I+1, J+1);
        u_exact(I+1, J+1) = sin(pi*Xp/L) .* sin(pi*Yp/L);
        u_vec = reshape(u_exact(2:end-1, 2:end-1), nx*ny, 1);
        f_int = A * u_vec;

    elseif source_type == "point"
        f_cont = zeros(nx+2, ny+2);
        x_mask = abs(X - L/2) < 2*hx;
        y_mask = abs(Y - L/2) < 2*hy;
        f_cont(x_mask & y_mask) = 1.0;
        f_cont(1,:) = 0; f_cont(end,:) = 0;
        f_cont(:,1) = 0; f_cont(:,end) = 0;
        f_int = reshape(f_cont(2:end-1, 2:end-1), nx*ny, 1);

    elseif source_type == "polynomial"
        u_exact = zeros(nx+2, ny+2);
        Xp = X(I+1, J+1);
        Yp = Y(I+1, J+1);
        Xn = (Xp - L/2) / (L/2);
        Yn = (Yp - L/2) / (L/2);
        poly = 1.0 ...
             + 0.30*Xn + 0.20*Yn ...
             + 0.10*Xn.^2 - 0.15*Xn.*Yn + 0.05*Yn.^2 ...
             + 0.02*Xn.^3 - 0.01*Yn.^3;
        wx = (Xp - X(I(1)+1,J(1)+1)) .* (X(I(end)+1,J(1)+1) - Xp);
        wy = (Yp - Y(I(1)+1,J(1)+1)) .* (Y(I(1)+1,J(end)+1) - Yp);
        win = (wx .* wy);
        win = win / max(win(:));
        u_exact(I+1, J+1) = poly .* win;
        u_vec = reshape(u_exact(2:end-1, 2:end-1), nx*ny, 1);
        f_int = A * u_vec;

    elseif source_type == "gauss"
        u_init = zeros(nx+2, ny+2);
        Xp = X(I+1, J+1);
        Yp = Y(I+1, J+1);
        sigma = 0.15*L;
        r2 = (Xp - L/2).^2 + (Yp - L/2).^2;
        u_init(I+1, J+1) = exp(-r2/(2*sigma^2));
        u_vec = reshape(u_init(2:end-1, 2:end-1), nx*ny, 1);
        f_int = u_vec;

    elseif source_type == "random"
        % Keep random field in physical domain only (no source in PML)
        u_exact = zeros(nx+2, ny+2);
        u_exact(I+1, J+1) = rand(numel(I), numel(J));
        u_vec = reshape(u_exact(2:end-1, 2:end-1), nx*ny, 1);
        f_int = A * u_vec;

    else
        error('Unknown source_type: %s', source_type);
    end
end

% -------------------------------------------------------------------------
% Match leapfrog discrete frequency to target Helmholtz omega:
%   (2/dt)sin(omega_ts*dt/2) = omega
% and enforce exact one-period sampling for omega_ts.
function [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt_init)
    dt = dt_init;
    if omega*dt/2 >= 0.98
        dt = 0.98*2/omega;
    end

    for it = 1:6
        omega_ts = (2/dt) * asin(omega*dt/2);
        nt = max(10, round(2*pi/(omega_ts*dt)));
        dt = 2*pi/(omega_ts*nt);
    end

    omega_ts = (2/dt) * asin(omega*dt/2);
    c_w = cos(omega_ts*dt/2);
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'L', 1.0);
    cfg = set_default(cfg, 'c', 1.0);
    cfg = set_default(cfg, 'omega', 12);
    cfg = set_default(cfg, 'Nphys', 45);
    cfg = set_default(cfg, 'npml', 12);
    cfg = set_default(cfg, 'sigma_max', 80);
    cfg = set_default(cfg, 'CFL', 0.5);
    cfg = set_default(cfg, 'max_iter', 600);
    cfg = set_default(cfg, 'tol', 1e-10);
    cfg = set_default(cfg, 'random_seed', 0);
    cfg = set_default(cfg, 'source_list', ["sin", "point", "polynomial", "gauss", "random"]);
    cfg = set_default(cfg, 'representative_source', "point");
    cfg = set_default(cfg, 'theta_scheme', 'target_constant');
    cfg = set_default(cfg, 'filter_shift_mode', 'standard');
    cfg = set_default(cfg, 'output_name', "");
    if ~ismember(cfg.theta_scheme, {'target_constant', 'double_average'})
        error('theta_scheme must be target_constant or double_average.');
    end
    if ~ismember(cfg.filter_shift_mode, {'standard', 'corrected'})
        error('filter_shift_mode must be standard or corrected.');
    end
end

function write_v5_settings(results, out_dir)
    cfg = results.cfg;
    grid = results.grid;
    match = results.matching;
    file_id = fopen(fullfile(out_dir, 'pml_v5_consistency_settings.txt'), 'w');
    cleaner = onCleanup(@() fclose(file_id));
    fprintf(file_id, 'WaveHoltz/PML v5 five-source consistency settings\n');
    fprintf(file_id, 'physical domain: [0, %.6g]^2\n', cfg.L);
    fprintf(file_id, 'wavespeed: %.16g\n', cfg.c);
    fprintf(file_id, 'physical grid: %d x %d\n', grid.nx_phys, grid.ny_phys);
    fprintf(file_id, 'PML cells per side: %d\n', cfg.npml);
    fprintf(file_id, 'total interior grid: %d x %d\n', grid.nx, grid.ny);
    fprintf(file_id, 'target omega_H: %.16g\n', match.omega_helm);
    fprintf(file_id, 'matched omega_ts: %.16g\n', match.omega_ts);
    fprintf(file_id, 'c_w: %.16g\n', match.c_w);
    fprintf(file_id, 'dt: %.16g\n', match.dt);
    fprintf(file_id, 'steps per period: %d\n', match.nt);
    fprintf(file_id, 'sigma_max: %.16g\n', cfg.sigma_max);
    fprintf(file_id, 'sources: %s\n', strjoin(cellstr(results.source_list), ', '));
    fprintf(file_id, 'fixed-point tolerance/max iterations: %.3e / %d\n', cfg.tol, cfg.max_iter);
    fprintf(file_id, 'representative source: %s\n', cfg.representative_source);
    fprintf(file_id, 'theta scheme: %s\n', cfg.theta_scheme);
    fprintf(file_id, 'filter shift mode: %s\n', cfg.filter_shift_mode);
end

function write_v5_summary(results, out_dir)
    file_id = fopen(fullfile(out_dir, 'pml_v5_consistency_summary.txt'), 'w');
    cleaner = onCleanup(@() fclose(file_id));
    fprintf(file_id, 'WaveHoltz/PML v5 five-source consistency summary\n');
    fprintf(file_id, 'values are minimum / mean / maximum\n\n');
    write_stats(file_id, 'phasor error, full PML-extended domain', results.summary.relerr_full);
    write_stats(file_id, 'phasor error, physical region', results.summary.relerr_phys);
    write_stats(file_id, 'phasor error, PML region', results.summary.relerr_pml);
    write_stats(file_id, 'WaveHoltz Helmholtz residual', results.summary.wave_residual);
    write_stats(file_id, 'fixed-point iterations', results.summary.iterations);
    write_stats(file_id, 'WaveHoltz solve time (s)', results.summary.wave_solve_time);
end

function write_stats(file_id, label, values)
    fprintf(file_id, '%-48s %.6e / %.6e / %.6e\n', label, values(1), values(2), values(3));
end

function make_v5_paper_figures(results, fig_dir, out_dir)
    sources = cellstr(results.source_list);
    colors = lines(numel(sources));

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 980 720]);
    hold on;
    for k = 1:numel(sources)
        semilogy(1:numel(results.hist{k}), results.hist{k}, 'LineWidth', 3.0, ...
            'Color', colors(k, :), 'DisplayName', sources{k});
    end
    htol = yline(results.cfg.tol, '--k', 'stopping tolerance', ...
        'LabelHorizontalAlignment', 'left', 'HandleVisibility', 'off');
    htol.LineWidth = 2.0;
    htol.FontSize = 20;
    set(gca, 'YScale', 'log'); ylim([results.cfg.tol / 10, 2]);
    grid on; xlabel('fixed-point iteration'); ylabel('relative fixed-point change');
    legend('Location', 'northeast');
    title('Five-source WaveHoltz/PML convergence');
    style_v5_curve_axes(gca);
    save_v5_figure(fig, fullfile(fig_dir, 'pml_v5_five_source_residuals.png'), out_dir);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 980 720]);
    semilogy(1:numel(sources), results.relerr_full, '-o', 'LineWidth', 3.0, 'MarkerSize', 10);
    hold on;
    semilogy(1:numel(sources), results.relerr_phys, '-s', 'LineWidth', 3.0, 'MarkerSize', 10);
    semilogy(1:numel(sources), results.relerr_pml, '-d', 'LineWidth', 3.0, 'MarkerSize', 10);
    semilogy(1:numel(sources), results.wave_residual, '-^', 'LineWidth', 3.0, 'MarkerSize', 10);
    grid on; xlim([0.7, numel(sources) + 0.3]); xticks(1:numel(sources)); xticklabels(sources);
    ylabel('relative error / residual');
    legend({'full phasor', 'physical phasor', ...
        'PML phasor', 'Helmholtz residual'}, 'Location', 'best');
    title('WaveHoltz/PML consistency metrics');
    style_v5_curve_axes(gca);
    save_v5_figure(fig, fullfile(fig_dir, 'pml_v5_five_source_errors.png'), out_dir);

    rep_idx = find(results.source_list == string(results.cfg.representative_source), 1);
    if isempty(rep_idx), rep_idx = 1; end
    grid_data = results.grid;
    u_helm = reshape(results.u_helm(:, rep_idx), grid_data.nx, grid_data.ny);
    u_wave = reshape(results.u_wave(:, rep_idx), grid_data.nx, grid_data.ny);
    abs_diff = abs(u_wave - u_helm);
    x_int = grid_data.x(2:end-1);
    y_int = grid_data.y(2:end-1);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1200 920]);
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    field_limits = max(abs([real(u_helm(:)); real(u_wave(:))]));
    nexttile; imagesc(x_int, y_int, real(u_helm).'); axis image; set(gca, 'YDir', 'normal'); cb = colorbar;
    clim([-field_limits, field_limits]);
    title('Helmholtz real part'); xlabel('x'); ylabel('y');
    style_v5_field_axes(gca, cb);
    nexttile; imagesc(x_int, y_int, real(u_wave).'); axis image; set(gca, 'YDir', 'normal'); cb = colorbar;
    clim([-field_limits, field_limits]);
    title('WaveHoltz real part'); xlabel('x'); ylabel('y');
    style_v5_field_axes(gca, cb);
    nexttile; imagesc(x_int, y_int, log10(max(abs_diff.', 1e-20))); axis image; set(gca, 'YDir', 'normal'); cb = colorbar;
    title('$\log_{10}|U_{\rm WH}-U_{\rm H}|$', 'Interpreter', 'latex'); xlabel('x'); ylabel('y');
    style_v5_field_axes(gca, cb);
    nexttile; semilogy(1:numel(results.hist{rep_idx}), results.hist{rep_idx}, 'LineWidth', 3.0);
    hold on; htol = yline(results.cfg.tol, '--k', 'HandleVisibility', 'off'); htol.LineWidth = 2.0; grid on;
    set(gca, 'YScale', 'log'); ylim([results.cfg.tol / 100, 2]);
    xlabel('fixed-point iteration'); ylabel('relative fixed-point change');
    title(sprintf('%s-source convergence', sources{rep_idx}));
    style_v5_panel_curve_axes(gca);
    save_v5_figure(fig, fullfile(fig_dir, 'pml_v5_representative_point_source.png'), out_dir);
end

function style_v5_curve_axes(ax)
    set(ax, 'FontSize', 24, 'LineWidth', 1.6, 'Box', 'on');
    ax.Title.FontSize = 29;
    ax.XLabel.FontSize = 26;
    ax.YLabel.FontSize = 26;
    lgd = ax.Legend;
    if ~isempty(lgd) && isvalid(lgd)
        lgd.FontSize = 20;
        lgd.Box = 'on';
    end
end

function style_v5_field_axes(ax, cb)
    set(ax, 'FontSize', 20, 'LineWidth', 1.4, 'Box', 'on');
    ax.Title.FontSize = 22;
    ax.XLabel.FontSize = 22;
    ax.YLabel.FontSize = 22;
    if nargin >= 2 && ~isempty(cb) && isvalid(cb)
        cb.FontSize = 18;
        cb.LineWidth = 1.2;
    end
end

function style_v5_panel_curve_axes(ax)
    set(ax, 'FontSize', 20, 'LineWidth', 1.4, 'Box', 'on');
    ax.Title.FontSize = 24;
    ax.XLabel.FontSize = 22;
    ax.YLabel.FontSize = 22;
end

function save_v5_figure(fig, path, out_dir)
    exportgraphics(fig, path, 'Resolution', 300);
    [~, name, ext] = fileparts(path);
    copyfile(path, fullfile(out_dir, [name ext]));
    close(fig);
end

function val = relerr_local(a, b)
    val = norm(a - b) / max(norm(b), 1e-16);
end

function values = stats3_local(x)
    values = [min(x(:)), mean(x(:)), max(x(:))];
end

function cfg = set_default(cfg, name, value)
    if ~isfield(cfg, name) || isempty(cfg.(name))
        cfg.(name) = value;
    end
end
