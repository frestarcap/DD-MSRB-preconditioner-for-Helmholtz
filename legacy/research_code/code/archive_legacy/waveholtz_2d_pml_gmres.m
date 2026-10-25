function waveholtz_2d_pml_gmres
 % ============================================================
%  WaveHoltz (outer PML): FULL-STATE alignment
%    - Fixed-Point: x_{k+1} = Pi_J(x_k)
%    - GMRES:       (I - S) x = Pi0
%  where x = [u0; v0], v0 = [vx0; vy0].
%
%  This file upgrades waveholtz_2d_pml_fp_vs_gmres_v3_fixed.m so that
%  FP/GMRES operate on the SAME full-state WaveHoltz mapping used by
%  waveholtz_2d_pml_v3.m.
%
%  Notes:
%    - The wave stepper uses physical omega in forcing, and uses
%      omega_eff=(2/dt)sin(omega dt/2) via alpha=omega_eff/omega for
%      time-discrete consistency (same as your v2/v3 codes).
%    - The Pi operator is the same trapezoidal-in-time filter:
%        Pi(x) = (2/T) ∫_0^T (cos(ωt) - 1/4) * state(t; x) dt
% ============================================================

    % ---- Physical-domain parameters ----
    L = 1.0;
    c = 1.0;
    omega = 33.5;

    % ---- Discretization on physical domain ----
    Nphys  = 100;
    nx_phys = Nphys; ny_phys = Nphys;
    hx = L/(nx_phys+1);
    hy = L/(ny_phys+1);

    % ---- PML outside physical domain ----
    ispml = 1;
    npml = ispml*ceil(0.3*Nphys);    % thickness in number of INTERIOR unknown layers
    sigma_max = 80;

    % ---- Time step (CFL) + enforce exact one-period sampling ----
    CFL = 0.5;
    dt0 = CFL*min(hx,hy)/c;
    [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt0);
    T_ts = nt*dt;

    do_helm = 1;   % optional Helmholtz-PML solve for comparison

    fprintf('Nphys=%d, dt0=%.3e, nt=%d, dt=%.3e, npml=%d\n', Nphys, dt0, nt, dt, npml);
    fprintf('omega(helm)=%.6f, omega(ts)=%.6f, c_w=%.6f\n', omega, omega_ts, c_w);

    % ---- Total computational interior unknowns (padded) ----
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

    % ---- PML (stretching uses omega_eff, as in v2/v3) ----
    pml_helm = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml);
    pml_wave = pml_helm;
    pml_wave.sigma_x = pml_helm.sigma_x / c_w;
    pml_wave.sigma_y = pml_helm.sigma_y / c_w;

    % ---- Source selection ----
    source_type = "gauss";   % "point" | "manufactured" | "polynomial" | "gauss"

    % ---- Build RHS f_int (size nx*ny) and optional exact field ----
    [f_int, u_exact] = build_rhs_v2(nx, ny, hx, hy, L, X, Y, I, J, omega, pml_helm, source_type);

    % ---- Dimensions of full state ----
    nu  = nx*ny;                 % u at interior nodes
    nvx = (nx+1)*ny;             % vx on x-faces, excluding y boundary
    nvy = nx*(ny+1);             % vy on y-faces, excluding x boundary
    nv  = nvx + nvy;
    nstate = nu + nv + 3*nu;

    % ============================================================
    % Fixed-point WaveHoltz (full-state)
    % ============================================================
    fp_max_iter = 2000;
    fp_tol      = 1e-14;
    t_FP = tic;
    [x_fp, hist_fp] = waveholtz_fixedpoint_fullstate(nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, f_int, pml_wave, fp_max_iter, fp_tol);
    time_FP = toc(t_FP);

    % ============================================================
    % GMRES WaveHoltz (full-state)
    % ============================================================
    gmres_tol     = 1e-14;
    gmres_restart = 30;
    gmres_maxit   = 60;

    t_gmres = tic;
    [x_gm, resvec_rel, info] = waveholtz_gmres_fullstate(nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, f_int, pml_wave, gmres_tol, gmres_restart, gmres_maxit);
    time_gmres = toc(t_gmres);

    % ============================================================
    % Optional Helmholtz-PML solve (u only, for reference)
    % ============================================================
    if do_helm
        A_helm = helmholtz_with_pml(nx, ny, hx, hy, omega, pml_helm);
        u_int  = A_helm \ f_int;
        u_helm = zeros(nx+2, ny+2);
        u_helm(2:end-1,2:end-1) = reshape(u_int, nx, ny);
        Uhelm_phys = real(u_helm(I+1, J+1));
    end

    % ============================================================
    % Extract u component and compare on PHYSICAL region
    % ============================================================
    u_fp_full = zeros(nx+2, ny+2);
    u_fp_full(2:end-1,2:end-1) = reshape(x_fp(1:nu), nx, ny);

    u_gm_full = zeros(nx+2, ny+2);
    u_gm_full(2:end-1,2:end-1) = reshape(x_gm(1:nu), nx, ny);

    Ufp_phys = real(u_fp_full(I+1, J+1));
    Ugm_phys = real(u_gm_full(I+1, J+1));

    relL2_phys = norm(Ugm_phys(:) - Ufp_phys(:)) / max(1e-16, norm(Ufp_phys(:)));

    fprintf('\n===== Summary (physical crop) =====\n');
    fprintf('WaveHoltz FULLSTATE FP: time=%.3f s\n', time_FP);
    fprintf('WaveHoltz FULLSTATE GM: time=%.3f s\n', time_gmres);
    fprintf('Speedup (time):         %.3fx\n', time_FP / max(1e-16, time_gmres));
    fprintf('GMRES flag=%d, relres=%.3e, iter=[%d %d]\n', info.flag, info.relres, info.iter(1), info.iter(2));
    fprintf('Relative L2 error (physical, GMRES vs FP) = %.6e\n', relL2_phys);

    if do_helm
        rel_err_gm_vs_helm = norm(Ugm_phys(:) - Uhelm_phys(:)) / max(1e-16, norm(Uhelm_phys(:)));
        fprintf('RelErr(GMRES vs Helmholtz) on physical = %.3e\n', rel_err_gm_vs_helm);
    end

    % ============================================================
    % Plots
    % ============================================================
    figure('Position',[100 100 1650 520]);
    subplot(1,4,1);
    imagesc([0 L],[0 L], Ufp_phys.'); set(gca,'YDir','normal');
    axis equal tight; title('WaveHoltz FULLSTATE Fixed-Point (u, physical)'); colorbar;

    subplot(1,4,2);
    imagesc([0 L],[0 L], Ugm_phys.'); set(gca,'YDir','normal');
    axis equal tight; title('WaveHoltz FULLSTATE GMRES (u, physical)'); colorbar;

    subplot(1,4,3);
    imagesc([0 L],[0 L], (Ugm_phys - Ufp_phys).'); set(gca,'YDir','normal');
    axis equal tight; title(sprintf('GMRES - FP (rel=%.2e)', relL2_phys)); colorbar;

    subplot(1,4,4);
    if do_helm
        imagesc([0 L],[0 L], (Ugm_phys - Uhelm_phys).'); set(gca,'YDir','normal');
        axis equal tight; title('Helm (physical)'); colorbar;
    end
    colormap(jet);
    sgtitle('WaveHoltz FULLSTATE: Fixed-Point vs GMRES (outer PML)');

    figure('Position',[100 100 1200 420]);
    subplot(1,2,1);
    semilogy(hist_fp,'LineWidth',1.2); grid on;
    xlabel('Fixed-point iter'); ylabel('rel-res'); title('Fixed-point convergence (full-state)');

    subplot(1,2,2);
    semilogy(resvec_rel,'LineWidth',1.2); grid on;
    xlabel('GMRES iter'); ylabel('||r||/||b||'); title('GMRES convergence (full-state)');
end

% -------------------------------------------------------------------------
% RHS builder matching the v2/v3 style
function [f_int, u_exact] = build_rhs_v2(nx, ny, hx, hy, L, X, Y, I, J, omega_eff, pml, source_type)
    u_exact = zeros(nx+2, ny+2);
    A = helmholtz_with_pml(nx, ny, hx, hy, omega_eff, pml);

    if source_type == "manufactured"
        Xp = X(I+1, J+1);
        Yp = Y(I+1, J+1);
        u_exact(I+1, J+1) = sin(pi*Xp/L) .* sin(pi*Yp/L);
        u_vec = reshape(u_exact(2:end-1,2:end-1), nx*ny, 1);
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
        Xp = X(I+1, J+1);
        Yp = Y(I+1, J+1);
        sigma = 0.05*L;
        r2 = (Xp - L/2).^2 + (Yp - L/2).^2;
        u_exact(I+1, J+1) = exp(-r2/(2*sigma^2));
        u_vec = reshape(u_exact(2:end-1,2:end-1), nx*ny, 1);
        f_int = u_vec;

    else
        error('Unknown source_type');
    end
end

% -------------------------------------------------------------------------
% Fixed-point on full state: x_{k+1} = Pi_J(x_k)
function [x, hist] = waveholtz_fixedpoint_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, f_int, pml, max_iter, tol)
    nu  = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    x = zeros(nu+nvx+nvy+3*nu, 1);

    hist = zeros(max_iter,1);

    for k = 1:max_iter
        x_new = apply_Pi_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x, f_int, true);

        if any(~isfinite(x_new))
            warning('FP blew up at iter %d (NaN/Inf).', k);
            hist = hist(1:max(k-1,1));
            break;
        end

        res = norm(x_new - x) / max(1e-16, norm(x_new));
        hist(k) = res;
        fprintf('FP iter %4d: rel-res = %.3e\n', k, res);

        x = x_new;
        if res < tol
            hist = hist(1:k);
            break;
        end
    end
end

% -------------------------------------------------------------------------
% GMRES solve on full state: (I - S)x = Pi0
function [x_sol, resvec_rel, info] = waveholtz_gmres_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, f_int, pml, tol, restart, maxit)
    nu  = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nstate = nu + nvx + nvy + 3*nu;

    Pi0 = apply_Pi_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, zeros(nstate,1), f_int, true);
    Sfun = @(x) apply_Pi_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x, f_int, false);
    Afun = @(x) x - Sfun(x);

    x0 = zeros(size(Pi0));
    verbose = true;
    [x_sol, flag, relres, iter, resvec] = gmres_restarted_verbose(Afun, Pi0, restart, tol, maxit, x0, verbose);

    resvec_rel = resvec / max(1e-16, norm(Pi0));
    info = struct('flag',flag,'relres',relres,'iter',iter);

    if flag ~= 0
        warning('GMRES stopped: flag=%d, relres=%.3e, iter=[%d %d].', flag, relres, iter(1), iter(2));
    end
end

% -------------------------------------------------------------------------
% Pi operator on full state
%   source_on=true  -> uses f_int in forcing
%   source_on=false -> zero forcing
function Pi_x = apply_Pi_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x0, f_int, source_on)
    if source_on
        W = solve_wave_one_period_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x0, f_int);
    else
        W = solve_wave_one_period_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x0, zeros(nx*ny,1));
    end

    t_grid  = (0:nt).' * dt;
    weights = cos(omega_ts*t_grid) - 0.25;

    integrand = bsxfun(@times, weights, W);
    integral  = trapz(t_grid, integrand, 1);

    Pi_x = (2.0/T_ts) * integral.';
end

% -------------------------------------------------------------------------
% FULL-STATE one-period wave stepping.
% Input x0 = [u0; vx0; vy0] where
%   u0  : nx*ny
%   vx0 : (nx+1)*ny    (x-faces, excluding y-boundary lines)
%   vy0 : nx*(ny+1)    (y-faces, excluding x-boundary lines)
% Output W: (nt+1) x (nu+nv) snapshot matrix with rows [u^n; vx^n; vy^n]
function W = solve_wave_one_period_fullstate(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x0, f_int)
    nu  = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nv = nvx + nvy;
    nstate = nu + nv + 3*nu;

    if numel(x0) ~= nstate
        error('x0 length mismatch: expect %d, got %d', nstate, numel(x0));
    end

    sx_full = zeros(nx+2, ny+2);
    sy_full = zeros(nx+2, ny+2);
    sx = pml.sigma_x; sy = pml.sigma_y;
    for i = 1:(nx+2), sx_full(i,:) = sx(i); end
    for j = 1:(ny+2), sy_full(:,j) = sy(j); end

    sigma_x_vx = 0.5*(sx_full(1:end-1,:) + sx_full(2:end,:)); % (nx+1, ny+2)
    sigma_y_vy = 0.5*(sy_full(:,1:end-1) + sy_full(:,2:end)); % (nx+2, ny+1)

    u0_int = x0(1:nu);
    v0_int = x0(nu+1:nu+nv);
    psi1_int = x0(nu+nv+1 : nu+nv+nu);
    psi2_int = x0(nu+nv+nu+1 : nu+nv+2*nu);
    theta_int = x0(nu+nv+2*nu+1 : end);

    u = zeros(nx+2, ny+2);
    u(2:end-1,2:end-1) = reshape(u0_int, nx, ny);

    vx = zeros(nx+1, ny+2);
    vy = zeros(nx+2, ny+1);
    vx(:, 2:end-1) = reshape(v0_int(1:nvx), nx+1, ny);
    vy(2:end-1, :) = reshape(v0_int(nvx+1:end), nx, ny+1);

    psi1 = zeros(nx+2, ny+2);
    psi2 = zeros(nx+2, ny+2);
    theta = zeros(nx+2, ny+2);
    psi1(2:end-1,2:end-1) = reshape(psi1_int, nx, ny);
    psi2(2:end-1,2:end-1) = reshape(psi2_int, nx, ny);
    theta(2:end-1,2:end-1) = reshape(theta_int, nx, ny);

    f_2d = reshape(f_int, nx, ny);
    c_w = cos(omega_ts*dt/2);
    sigu = sx_full + sy_full;

    W = zeros(nt+1, nstate);
    W(1,:) = pack_state_from_fields(u, vx, vy, psi1, psi2, theta, nx, ny);

    for n = 1:nt
        t = (n-1)*dt;
        t_half = t + 0.5*dt;

        du_dx = (u(2:end,:) - u(1:end-1,:))/hx;
        du_dy = (u(:,2:end) - u(:,1:end-1))/hy;
        vx_new = ((1 - 0.5*dt*sigma_x_vx).*vx + dt*c^2*du_dx) ./ (1 + 0.5*dt*sigma_x_vx);
        vy_new = ((1 - 0.5*dt*sigma_y_vy).*vy + dt*c^2*du_dy) ./ (1 + 0.5*dt*sigma_y_vy);

        vx_avg = 0.5*(vx + vx_new);
        vy_avg = 0.5*(vy + vy_new);

        grad_vx = (vx_avg(2:end,:) - vx_avg(1:end-1,:))/hx;
        tmp1 = zeros(nx+2, ny+2); tmp1(2:end-1,:) = grad_vx;

        grad_vy = (vy_avg(:,2:end) - vy_avg(:,1:end-1))/hy;
        tmp2 = zeros(nx+2, ny+2); tmp2(:,2:end-1) = grad_vy;

        psi1 = psi1 + dt * (sy_full .* tmp1);
        psi2 = psi2 + dt * (sx_full .* tmp2);
        theta = theta + dt * ((sx_full .* sy_full) .* (c_w^2) .* u);

        div_v = zeros(nx+2, ny+2);
        div_v(2:end-1,:) = div_v(2:end-1,:) + (vx_new(2:end,:) - vx_new(1:end-1,:))/hx;
        div_v(:,2:end-1) = div_v(:,2:end-1) + (vy_new(:,2:end) - vy_new(:,1:end-1))/hy;

        forcing = -(1.0/omega_helm) * f_2d .* sin(omega_ts * t_half);

        rhs_u = div_v(2:end-1, 2:end-1) ...
            - theta(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) ...
            + psi2(2:end-1, 2:end-1) ...
            + forcing;

        num_u = (1 - 0.5*dt*sigu(2:end-1,2:end-1)).*u(2:end-1,2:end-1) + dt*rhs_u;
        den_u = (1 + 0.5*dt*sigu(2:end-1,2:end-1));
        u(2:end-1,2:end-1) = num_u ./ den_u;

        vx = vx_new;
        vy = vy_new;

        W(n+1,:) = pack_state_from_fields(u, vx, vy, psi1, psi2, theta, nx, ny);
    end
end

function row = pack_state_from_fields(u, vx, vy, psi1, psi2, theta, nx, ny)
    % u: (nx+2, ny+2) nodes; take interior (nx,ny)
    u_int = reshape(u(2:end-1, 2:end-1), 1, nx*ny);

    % vx: (nx+1, ny+2) faces; store only y-interior columns 2:ny+1 => (nx+1, ny)
    vx_int = reshape(vx(:, 2:ny+1), 1, (nx+1)*ny);

    % vy: (nx+2, ny+1) faces; store only x-interior rows 2:nx+1 => (nx, ny+1)
    vy_int = reshape(vy(2:nx+1, :), 1, nx*(ny+1));

    psi1_int = reshape(psi1(2:end-1,2:end-1), 1, nx*ny);
    psi2_int = reshape(psi2(2:end-1,2:end-1), 1, nx*ny);
    theta_int = reshape(theta(2:end-1,2:end-1), 1, nx*ny);

    row = [u_int, vx_int, vy_int, psi1_int, psi2_int, theta_int];
end

% -------------------------------------------------------------------------
% Build PML coefficients on TOTAL domain: sigma nonzero only in OUTER layers
function pml = build_pml_outer(nx, ny, omega_stretch, npml, sigma_max, ispml)
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
    s_x = 1.0 + 1i*sigma_x/omega_stretch;
    s_y = 1.0 + 1i*sigma_y/omega_stretch;

    % half-step averages along axes
    s_x_half = 0.5*(s_x(1:end-1) + s_x(2:end));
    s_y_half = 0.5*(s_y(1:end-1) + s_y(2:end));

    pml = struct('sigma_x',sigma_x, 'sigma_y',sigma_y, ...
                 's_x',s_x, 's_y',s_y, ...
                 's_x_half',s_x_half, 's_y_half',s_y_half);
end

% -------------------------------------------------------------------------
% Helmholtz with PML (variable-coefficient discretization on interior unknowns)
% A ≈ (Δ_PML + omega^2 I)
function A_helm_pml = helmholtz_with_pml(nx, ny, hx, hy, omega_helm, pml)
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

    A_helm_pml = A + (omega_helm^2)*speye(Ntot);
end

% -------------------------------------------------------------------------
% Restarted GMRES with per-iteration residual print (same as your v3_fixed)
function [x, flag, relres, iter, resvec] = gmres_restarted_verbose(Afun, b, restart, tol, maxit_outer, x0, verbose)
% Solve Afun(x) = b using restarted GMRES(m), m=restart.
% Outputs resvec with absolute residual norms (starting from ||r0||).

    if nargin < 7, verbose = true; end
    if nargin < 6 || isempty(x0), x0 = zeros(size(b)); end

    x = x0;

    nb = norm(b);
    if nb == 0
        flag = 0; relres = 0; iter = [0,0]; resvec = 0;
        return;
    end

    r = b - Afun(x);
    beta = norm(r);
    resvec = beta;

    if verbose
        fprintf('GMRES init: ||r0||/||b|| = %.3e\n', beta/nb);
    end
    if beta/nb <= tol
        flag = 0; relres = beta/nb; iter = [0,0];
        return;
    end

    flag = 1;
    iter = [0,0];

    for outer = 1:maxit_outer
        V = zeros(numel(b), restart+1);
        H = zeros(restart+1, restart);
        cs = zeros(restart,1);
        sn = zeros(restart,1);
        e1 = zeros(restart+1,1); e1(1) = 1;

        V(:,1) = r / beta;
        g = beta * e1;

        inner_last = 0;
        for j = 1:restart
            inner_last = j;

            w = Afun(V(:,j));

            for i = 1:j
                H(i,j) = V(:,i)' * w;
                w = w - H(i,j)*V(:,i);
            end
            H(j+1,j) = norm(w);

            if H(j+1,j) ~= 0
                V(:,j+1) = w / H(j+1,j);
            else
                % breakdown
                V(:,j+1) = zeros(size(w));
            end

            % Apply previous Givens rotations
            for i = 1:j-1
                temp      =  cs(i)*H(i,j) + sn(i)*H(i+1,j);
                H(i+1,j)  = -sn(i)*H(i,j) + cs(i)*H(i+1,j);
                H(i,j)    =  temp;
            end

            % Form i-th rotation
            [cs(j), sn(j)] = givens_rotation(H(j,j), H(j+1,j));

            % Apply i-th rotation
            temp      = cs(j)*H(j,j) + sn(j)*H(j+1,j);
            H(j+1,j)  = 0;
            H(j,j)    = temp;

            % Update g
            temp    = cs(j)*g(j) + sn(j)*g(j+1);
            g(j+1)  = -sn(j)*g(j) + cs(j)*g(j+1);
            g(j)    = temp;

            resnorm = abs(g(j+1));
            resvec(end+1,1) = resnorm;

            if verbose
                fprintf('  GMRES iter %4d: ||r||/||b|| = %.3e\n', (outer-1)*restart + j, resnorm/nb);
            end

            if resnorm/nb <= tol
                y = H(1:j,1:j) \ g(1:j);
                x = x + V(:,1:j) * y;

                r = b - Afun(x);
                beta = norm(r);

                flag = 0;
                relres = beta/nb;
                iter = [outer, j];
                return;
            end
        end

        % restart update
        y = H(1:inner_last,1:inner_last) \ g(1:inner_last);
        x = x + V(:,1:inner_last) * y;

        r = b - Afun(x);
        beta = norm(r);

        iter = [outer, inner_last];

        if beta/nb <= tol
            flag = 0;
            relres = beta/nb;
            return;
        end
    end

    relres = beta/nb;
end

function [c,s] = givens_rotation(a,b)
    if b == 0
        c = 1; s = 0;
    elseif abs(b) > abs(a)
        t = a/b;
        s = 1 / sqrt(1 + t^2);
        c = s*t;
    else
        t = b/a;
        c = 1 / sqrt(1 + t^2);
        s = c*t;
    end
end

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
