function waveholtz_2d_pml_v5
    % ============================================================
    %  WaveHoltz vs Helmholtz with OUTER PML
    %  v5 update:
    %    1) leapfrog frequency matching (Stolk correction)
    %    2) PML sigma scaling for staggered-time consistency
    %    3) corrected theta sign in PML auxiliary update
    % ============================================================

    rng(0);

    % ---- Physical-domain parameters (target Helmholtz) ----
    L = 1.0;
    c = 1.0;
    omega = 3.5;
    max_iter = 1200;
    tol = 1e-14;

    % ---- Discretization on physical domain ----
    Nphys  = 120;
    nx_phys = Nphys; ny_phys = Nphys;
    hx = L/(nx_phys+1);
    hy = L/(ny_phys+1);

    % ---- PML outside physical domain ----
    ispml = 0;
    npml = ispml*30;          % thickness in number of INTERIOR unknown layers
    sigma_max = 80;

    % ---- Time step + frequency mapping ----
    CFL = 0.5;
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
    source_list = ["sin", "point", "polynomial", "gauss", "random"];
    do_plots = 1;

    for sid = 1:numel(source_list)
        source_type = source_list(sid);
        fprintf('\n===== Source: %s =====\n', source_type);

        [f_int, u_exact] = build_rhs_case(source_type, A, nx, ny, hx, hy, L, X, Y, I, J);

        % Helmholtz solve (total domain)
        u_helm_vec = A \ f_int;
        u_helm = zeros(nx+2, ny+2);
        u_helm(2:end-1,2:end-1) = reshape(u_helm_vec, nx, ny);

        % WaveHoltz solve (time stepping at omega_ts, target omega in forcing scale)
        [u_wave, hist] = waveholtz_iteration( ...
            nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, f_int, pml_wave, max_iter, tol, A);

        % Physical crop diagnostics
        Uh_phys_c = u_helm(I+1, J+1);
        Uw_phys_c = u_wave(I+1, J+1);
        Uh_phys = real(Uh_phys_c);
        Uw_phys = real(Uw_phys_c);
        res_real = norm(Uw_phys(:) - Uh_phys(:)) / max(norm(Uh_phys(:)), 1e-15);
        fprintf('Relative error in physical domain: %.5e\n', res_real);

        if ~isempty(u_exact)
            Ue_phys = real(u_exact(I+1, J+1));
            err_helm_exact = norm(Uh_phys(:) - Ue_phys(:)) / max(norm(Ue_phys(:)), 1e-15);
            err_wave_exact = norm(Uw_phys(:) - Ue_phys(:)) / max(norm(Ue_phys(:)), 1e-15);
            fprintf('Helm vs exact (physical): %.5e\n', err_helm_exact);
            fprintf('Wave vs exact (physical): %.5e\n', err_wave_exact);
        end

        if do_plots
            figure;
            subplot(2,2,1);
            imagesc(real(u_helm).'); axis equal tight;
            title(sprintf('Helmholtz (%s)', source_type)); colorbar;

            subplot(2,2,2);
            imagesc(real(u_wave).'); axis equal tight;
            title(sprintf('WaveHoltz (%s)', source_type)); colorbar;

            subplot(2,2,3);
            imagesc((real(Uw_phys - Uh_phys)).'); axis equal tight;
            title('Wave - Helm (physical)'); colorbar;

            subplot(2,2,4);
            semilogy(hist, 'LineWidth', 1.5); grid on;
            xlabel('iteration'); ylabel('rel-res');
            title('WaveHoltz iteration history');
        end
    end
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
% Now outputs snapshots of [u^n; v^n] where v^n is recovered by averaging
% staggered half-step velocities: v^n = 0.5*(v^{n-1/2}+v^{n+1/2})
function W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x0, f_int)

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
    nstate = nu + nv + naux;
    if numel(x0) ~= nstate
        error('x0 length mismatch: expect %d, got %d', nstate, numel(x0));
    end

    u0_int = x0(1:nu);
    v0_int = x0(nu+1:nu+nv);
    psi1_int = x0(nu+nv+1 : nu+nv+nu);
    psi2_int = x0(nu+nv+nu+1 : nu+nv+2*nu);
    theta_int = x0(nu+nv+2*nu+1 : end);

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
    W(1,:) = [u0_vec; v0_vec; psi1_int; psi2_int; theta_int].';

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

        % (4) Theta update at half step
        theta = theta + dt * ( (sx_full .* sy_full) .* (c_w^2) .* u );

        % Divergence of v^{n+1/2}
        div_v = zeros(nx+2, ny+2);
        div_v(2:end-1,:) = div_v(2:end-1,:) + (vx_new(2:end,:) - vx_new(1:end-1,:))/hx;
        div_v(:,2:end-1) = div_v(:,2:end-1) + (vy_new(:,2:end) - vy_new(:,1:end-1))/hy;

        % Forcing (physical omega)
        forcing = -(1.0/omega_helm) * f_2d .* sin(omega_ts * t_half);

        % (1) u update with u^{n+1/2} damping
        rhs_u = ...
              div_v(2:end-1, 2:end-1) ...
            - theta(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) ...
            + psi2(2:end-1, 2:end-1) ...
            + forcing;

        num_u = (1 - 0.5*dt*sigu(2:end-1,2:end-1)).*u(2:end-1,2:end-1) + dt*rhs_u;
        den_u = (1 + 0.5*dt*sigu(2:end-1,2:end-1));
        u(2:end-1,2:end-1) = num_u ./ den_u;

        % Shift half-step velocities
        vx = vx_new;
        vy = vy_new;

        v_vec = [reshape(vx(:,2:end-1), nvx, 1); reshape(vy(2:end-1,:), nvy, 1)];
        u_vec = reshape(u(2:end-1, 2:end-1), nu, 1);

        psi1_vec = reshape(psi1(2:end-1,2:end-1), nu, 1);
        psi2_vec = reshape(psi2(2:end-1,2:end-1), nu, 1);
        theta_vec = reshape(theta(2:end-1,2:end-1), nu, 1);

        W(n+1,:) = [u_vec; v_vec; psi1_vec; psi2_vec; theta_vec].';
    end
end

% -------------------------------------------------------------------------
% WaveHoltz fixed-point iteration using one-period time stepping
% Now iterates on x = [u0; v0] and filters both u and v together.
function [u_full, hist] = waveholtz_iteration(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, f_int, pml, max_iter, tol, A)

    nu = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nv = nvx + nvy;

    x = zeros(nu + nv + 3*nu, 1);         % [u0; v0; psi10; psi20; theta0]
    hist = zeros(max_iter,1);

    for k = 1:max_iter

        W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x, f_int);

        % Daniel's weights (keep as you requested)
        t_grid = (0:nt).' * dt;            % nt+1 points, includes t=T
        weights = cos(omega_ts*t_grid) - 0.25;
        
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

    % Recover complex phasor from converged [u0; v0]:
    %   u_hat = u0 +/- i*(1/omega_helm)*div(v0)
    % Choose sign by smaller Helmholtz residual.
    u0_int = x(1:nu);
    v0_int = x(nu+1:nu+nv);

    vx0 = zeros(nx+1, ny+2);
    vy0 = zeros(nx+2, ny+1);
    vx0(:,2:end-1) = reshape(v0_int(1:nvx), nx+1, ny);
    vy0(2:end-1,:) = reshape(v0_int(nvx+1:end), nx, ny+1);

    div_v0 = zeros(nx+2, ny+2);
    div_v0(2:end-1,:) = div_v0(2:end-1,:) + (vx0(2:end,:) - vx0(1:end-1,:))/hx;
    div_v0(:,2:end-1) = div_v0(:,2:end-1) + (vy0(:,2:end) - vy0(:,1:end-1))/hy;

    u0 = zeros(nx+2, ny+2);
    u0(2:end-1,2:end-1) = reshape(u0_int, nx, ny);

    uim = (1/omega_helm) * div_v0;

    u_plus = u0 + 1i*uim;
    u_minus = u0 - 1i*uim;

    r_plus = norm(A*reshape(u_plus(2:end-1,2:end-1), nu, 1) - f_int) / max(norm(f_int), 1e-15);
    r_minus = norm(A*reshape(u_minus(2:end-1,2:end-1), nu, 1) - f_int) / max(norm(f_int), 1e-15);

    if r_plus <= r_minus
        u_full = u_plus;
    else
        u_full = u_minus;
    end

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
