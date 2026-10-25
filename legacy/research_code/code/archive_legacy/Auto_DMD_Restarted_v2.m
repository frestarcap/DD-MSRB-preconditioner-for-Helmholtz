function Auto_DMD_Restarted_v2
% ============================================================
% Auto_DMD_Restarted_v2
%   WaveHoltz (outer-PML, v2) + rolling/truncated Auto-DMD acceleration
%   Compare against Helmholtz-PML (using omega correction) and
%   report DMD vs plain WaveHoltz speedup.
%
% Outputs:
%   1) Helmholtz solution (physical crop)
%   2) DMD-accelerated WaveHoltz solution (physical crop)
%   3) Error image (DMD WaveHoltz - Helmholtz) on physical crop
%   4) Convergence curves + speedup metrics
% ============================================================

    % ----------------------
    % Parameters (v2-style)
    % ----------------------
    L = 1.0;
    c = 1.0;
    omega = 33.5;
    max_iter = 200;
    tol = 1e-12;

    % physical grid
    Nphys = 120;
    nx_phys = Nphys; ny_phys = Nphys;
    hx = L/(nx_phys+1);
    hy = L/(ny_phys+1);

    % outer PML
    ispml = 1;
    npml = ceil(0.3*Nphys);
    sigma_max = 80.0;

    % time step / one-period sampling
    CFL = 0.3;
    dt0 = CFL*min(hx,hy)/c;
    [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt0);
    T_ts = nt*dt;

    fprintf('Nphys=%d, npml=%d, nx_total=%d\n', Nphys, npml, nx_phys + 2*npml);
    fprintf('dt0=%.3e, nt=%d, dt=%.3e, CFL_actual=%.3f\n', dt0, nt, dt, c*dt/min(hx,hy));
    fprintf('omega(helm)=%.6f, omega(ts)=%.6f, c_w=%.6f\n', omega, omega_ts, c_w);
  
    % DMD controls
    dmd_max_snaps = 100;     % rolling snapshot window length
    dmd_max_rank = 40;       % cap DMD modes so U remains moderate
    dmd_predict_burst = 4;   % after a trigger, allow several predictions
    dmd_accept_growth = 1.0; % accept only if verified fixed-point residual improves
    epsSVD = 1e-10;

    % Baseline controls: "plain" fixed-point iteration or "gmres" on
    % (I - WaveHoltz map) x = Pi(0; f).
    baseline_solver = "gmres";
    baseline_gmres_restart = 30;
    baseline_gmres_maxit = max(1, ceil(max_iter / baseline_gmres_restart));

    % ----------------------
    % Total domain (with PML)
    % ----------------------
    nx = nx_phys + 2*npml;
    ny = ny_phys + 2*npml;

    Lpml_x = npml*hx;
    Lpml_y = npml*hy;
    x = linspace(-Lpml_x, L+Lpml_x, nx+2);
    y = linspace(-Lpml_y, L+Lpml_y, ny+2);
    [X, Y] = meshgrid(x, y);

    % physical interior indices in total interior grid
    I = (npml+1):(npml+nx_phys);
    J = (npml+1):(npml+ny_phys);

    % ----------------------
    % PML + Helmholtz operator
    % ----------------------
    pml_helm = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml);
    pml_wave = pml_helm;
    pml_wave.sigma_x = pml_helm.sigma_x / c_w;
    pml_wave.sigma_y = pml_helm.sigma_y / c_w;
    A = helmholtz_with_pml(nx, ny, hx, hy, omega, pml_helm);

    % ----------------------
    % Source selection
    %   source_type = "point" | "smooth" | "manufactured"
    % ----------------------
    source_type = "point";

    u_exact = zeros(nx+2, ny+2);

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
        % enforce outer Dirichlet nodes
        f_cont(1,:) = 0; f_cont(end,:) = 0;
        f_cont(:,1) = 0; f_cont(:,end) = 0;
        f_int = reshape(f_cont(2:end-1, 2:end-1), nx*ny, 1);
    else
        error('Unknown source_type');
    end

    % ----------------------
    % Helmholtz solve
    % ----------------------
    fprintf('\n===== Helmholtz-PML solve (omega_eff) =====\n');
    t_helm = tic;
    u_helm_vec = A \ f_int;
    time_helm = toc(t_helm);

    u_helm = zeros(nx+2, ny+2);
    u_helm(2:end-1,2:end-1) = reshape(u_helm_vec, nx, ny);

    % ----------------------
    % WaveHoltz baseline
    % ----------------------
    fprintf('\n===== WaveHoltz baseline (%s) =====\n', baseline_solver);
    t_reg = tic;
    switch baseline_solver
        case "plain"
            [u_wave, hist_reg, baseline_label] = waveholtz_iteration_plain( ...
                nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, f_int, pml_wave, max_iter, tol, A);
        case "gmres"
            [u_wave, hist_reg, baseline_label] = waveholtz_iteration_gmres_baseline( ...
                nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, f_int, pml_wave, ...
                baseline_gmres_restart, baseline_gmres_maxit, tol, A);
        otherwise
            error('Unknown baseline_solver: %s', baseline_solver);
    end
    time_reg = toc(t_reg);

    % ----------------------
    % WaveHoltz + Auto-DMD (rolling/truncated)
    % ----------------------
    fprintf('\n===== WaveHoltz + Auto-DMD (rolling/truncated) =====\n');
    t_dmd = tic;
    [u_dmd, hist_dmd, singular_vals, num_window_truncations, num_predictions] = waveholtz_iteration_dmd_restarted( ...
        nx, ny, hx, hy, c, omega_ts, omega, T_ts, dt, nt, f_int, pml_wave, max_iter, tol, ...
        dmd_max_snaps, dmd_max_rank, dmd_predict_burst, dmd_accept_growth, epsSVD, A);
    time_dmd = toc(t_dmd);

    % ----------------------
    % Metrics on physical crop
    % ----------------------
    Uh = real(u_helm(I+1, J+1));
    Uw = real(u_wave(I+1, J+1));
    Ud = real(u_dmd(I+1, J+1));

    rel_err_dmd_vs_helm  = norm(Ud(:) - Uh(:)) / max(1e-16, norm(Uh(:)));
    rel_err_wave_vs_helm = norm(Uw(:) - Uh(:)) / max(1e-16, norm(Uh(:)));
    rel_err_dmd_vs_wave  = norm(Ud(:) - Uw(:)) / max(1e-16, norm(Uw(:)));

    it_reg = numel(hist_reg);
    it_dmd = numel(hist_dmd);
    speedup_iter = it_reg / max(1, it_dmd);
    speedup_time = time_reg / max(1e-16, time_dmd);

    fprintf('\n===== Summary (physical crop) =====\n');
    fprintf('Helmholtz runtime: %.3f s\n', time_helm);
    fprintf('%-16s it=%d, time=%.3f s\n', char(baseline_label + ":"), it_reg, time_reg);
    fprintf('DMD WaveHoltz:     it=%d, time=%.3f s, window-trunc=%d, predictions=%d\n', ...
        it_dmd, time_dmd, num_window_truncations, num_predictions);
    fprintf('Speedup (iters):   %.3fx\n', speedup_iter);
    fprintf('Speedup (time):    %.3fx\n', speedup_time);
    fprintf('RelErr %s vs Helm: %.6e\n', char(baseline_label), rel_err_wave_vs_helm);
    fprintf('RelErr DMD   vs Helm: %.6e\n', rel_err_dmd_vs_helm);
    fprintf('RelErr DMD   vs Wave: %.6e\n', rel_err_dmd_vs_wave);

    % ----------------------
    % Figures
    % ----------------------
    figure('Name','Auto-DMD Restarted v2: Helm vs DMD-WaveHoltz','Position',[80 80 1550 700]);

    subplot(2,2,1);
    imagesc([0 L],[0 L], Uh.');
    set(gca,'YDir','normal'); axis equal tight; colorbar;
    title('Helmholtz-PML (physical crop)');

    subplot(2,2,2);
    imagesc([0 L],[0 L], Ud.');
    set(gca,'YDir','normal'); axis equal tight; colorbar;
    title(sprintf('DMD-accelerated WaveHoltz (crop) | it=%d', it_dmd));

    subplot(2,2,3);
    imagesc([0 L],[0 L], (Ud - Uh).');
    set(gca,'YDir','normal'); axis equal tight; colorbar;
    title(sprintf('Error: DMD-Wave - Helm | rel=%.2e', rel_err_dmd_vs_helm));

    subplot(2,2,4);
    semilogy(hist_reg,'LineWidth',1.5); hold on;
    semilogy(hist_dmd,'LineWidth',1.5);
    grid on; xlabel('iteration'); ylabel('rel-res');
    legend(baseline_label, 'DMD WaveHoltz', 'Location', 'northeast');
    title(sprintf('Convergence | speedup(it)=%.2fx, speedup(time)=%.2fx', speedup_iter, speedup_time));

    sgtitle(sprintf('Auto-DMD Restarted v2 | source=%s | omega=%.1f', source_type, omega));

    if ~isempty(singular_vals)
        figure('Name','DMD: singular values of Y_{neg}','Position',[120 120 760 360]);
        semilogy(singular_vals,'-o','LineWidth',1.2,'MarkerSize',4);
        grid on; xlabel('index'); ylabel('singular value');
        title('Singular values (latest pre-prediction window)');
    end
end

% =====================================================================
% v2 helper: OUTER PML stretching (omega_stretch is omega_eff)
% =====================================================================
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

    s_x = 1.0 + 1i*sigma_x/omega_stretch;
    s_y = 1.0 + 1i*sigma_y/omega_stretch;

    s_x_half = 0.5*(s_x(1:end-1) + s_x(2:end));
    s_y_half = 0.5*(s_y(1:end-1) + s_y(2:end));

    pml = struct('sigma_x',sigma_x, 'sigma_y',sigma_y, ...
                 's_x',s_x, 's_y',s_y, ...
                 's_x_half',s_x_half, 's_y_half',s_y_half);
end

% =====================================================================
% v2 helper: Helmholtz-PML operator using omega_helm=omega_eff
% =====================================================================
function A_helm_pml = helmholtz_with_pml(nx, ny, hx, hy, omega_helm, pml)
    Ntot = nx*ny;
    A = spalloc(Ntot, Ntot, 5*Ntot);
    idx = @(i,j) i + (j-1)*nx;

    for i = 1:nx
        for j = 1:ny
            row = idx(i,j);

            % x-direction
            a_x = 1.0 ./ pml.s_x(i+1);
            a_p = 1.0 ./ pml.s_x_half(i+1);
            a_m = 1.0 ./ pml.s_x_half(i);

            A(row,row) = A(row,row) + (-(a_p + a_m)*a_x)/(hx*hx);
            if i+1 <= nx, A(row, idx(i+1,j)) = A(row, idx(i+1,j)) + (a_x*a_p)/(hx*hx); end
            if i-1 >= 1,  A(row, idx(i-1,j)) = A(row, idx(i-1,j)) + (a_x*a_m)/(hx*hx); end

            % y-direction
            b_y = 1.0 ./ pml.s_y(j+1);
            b_p = 1.0 ./ pml.s_y_half(j+1);
            b_m = 1.0 ./ pml.s_y_half(j);

            A(row,row) = A(row,row) + (-(b_p + b_m)*b_y)/(hy*hy);
            if j+1 <= ny, A(row, idx(i,j+1)) = A(row, idx(i,j+1)) + (b_y*b_p)/(hy*hy); end
            if j-1 >= 1,  A(row, idx(i,j-1)) = A(row, idx(i,j-1)) + (b_y*b_m)/(hy*hy); end

            % omega^2 term
            A(row,row) = A(row,row) + omega_helm^2;
        end
    end

    A_helm_pml = A;
end

% =====================================================================
% v2 helper: One-period wave stepping (physical omega)
% =====================================================================
function W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x0, f_int)
    sx_full = zeros(nx+2, ny+2);
    sy_full = zeros(nx+2, ny+2);
    sx = pml.sigma_x; sy = pml.sigma_y;
    for i = 1:(nx+2), sx_full(i,:) = sx(i); end
    for j = 1:(ny+2), sy_full(:,j) = sy(j); end

    sigma_x_vx = 0.5*(sx_full(1:end-1,:) + sx_full(2:end,:)); % (nx+1, ny+2)
    sigma_y_vy = 0.5*(sy_full(:,1:end-1) + sy_full(:,2:end)); % (nx+2, ny+1)

    nu = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
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

    u = zeros(nx+2, ny+2);
    u(2:end-1, 2:end-1) = reshape(u0_int, nx, ny);

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

    v0_vec = [reshape(vx(:,2:end-1), nvx, 1); reshape(vy(2:end-1,:), nvy, 1)];
    W = zeros(nt+1, nstate);
    u0_vec = reshape(u(2:end-1, 2:end-1), nu, 1);
    W(1,:) = [u0_vec; v0_vec; psi1_int; psi2_int; theta_int].';

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

        psi1 = psi1 + dt * ( sy_full .* tmp1 );
        psi2 = psi2 + dt * ( sx_full .* tmp2 );
        theta = theta + dt * ( (sx_full .* sy_full) .* (c_w^2) .* u );

        div_v = zeros(nx+2, ny+2);
        div_v(2:end-1,:) = div_v(2:end-1,:) + (vx_new(2:end,:) - vx_new(1:end-1,:))/hx;
        div_v(:,2:end-1) = div_v(:,2:end-1) + (vy_new(:,2:end) - vy_new(:,1:end-1))/hy;

        forcing = -(1.0/omega_helm) * f_2d .* sin(omega_ts * t_half);

        rhs_u = ...
              div_v(2:end-1, 2:end-1) ...
            - theta(2:end-1, 2:end-1) ...
            + psi1(2:end-1, 2:end-1) ...
            + psi2(2:end-1, 2:end-1) ...
            + forcing;

        num_u = (1 - 0.5*dt*sigu(2:end-1,2:end-1)).*u(2:end-1,2:end-1) + dt*rhs_u;
        den_u = (1 + 0.5*dt*sigu(2:end-1,2:end-1));
        u(2:end-1,2:end-1) = num_u ./ den_u;

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

% =====================================================================
% Plain WaveHoltz fixed-point iteration (v2)
% =====================================================================
function [u_full, hist, label] = waveholtz_iteration_plain(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, f_int, pml, max_iter, tol, A)
    nu = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nv = nvx + nvy;
    x = zeros(nu + nv + 3*nu, 1);
    hist = zeros(max_iter,1);
    label = "plain WaveHoltz";

    t_grid = (0:nt).' * dt;
    weights = cos(omega_ts*t_grid) - 0.25;

    for k = 1:max_iter
        W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x, f_int);
        integrand = bsxfun(@times, weights, W);
        integral = trapz(t_grid, integrand, 1);
        x_new = (2.0/T_ts) * integral.';

        res = norm(x_new - x) / max(1e-16, norm(x_new));
        hist(k) = res;
        x = x_new;

        fprintf('plain iter %3d: rel-res = %.3e\n', k, res);
        if res < tol
            hist = hist(1:k);
            break;
        end
    end

    u_full = recover_complex_from_state(x, nx, ny, hx, hy, omega_helm, A, f_int);
end

% =====================================================================
% GMRES baseline for the same WaveHoltz fixed-point equation.
% =====================================================================
function [u_full, hist, label, info] = waveholtz_iteration_gmres_baseline( ...
    nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, f_int, pml, restart, maxit, tol, A)

    nu = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nv = nvx + nvy;
    nstate = nu + nv + 3*nu;
    label = "GMRES WaveHoltz";

    x0 = zeros(nstate, 1);
    Pi0 = waveholtz_apply_map(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x0, f_int);
    Sfun = @(x) waveholtz_apply_map(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x, zeros(nu, 1));
    Afun = @(x) x - Sfun(x);

    fprintf('GMRES baseline: restart=%d, maxit=%d, tol=%.1e\n', restart, maxit, tol);
    [x_sol, flag, relres, iter, resvec] = gmres(Afun, Pi0, restart, tol, maxit, [], [], x0);

    hist = resvec / max(1e-16, norm(Pi0));
    if numel(hist) > 1
        hist = hist(2:end);
    end

    iter_count = waveholtz_gmres_iter_count(iter, restart);
    fprintf('GMRES baseline stopped: flag=%d, iter=%d, relres=%.3e\n', flag, iter_count, relres);
    if flag ~= 0
        warning('GMRES baseline did not reach the requested tolerance.');
    end

    info = struct('flag', flag, 'relres', relres, 'iter', iter, 'iter_count', iter_count);
    u_full = recover_complex_from_state(x_sol, nx, ny, hx, hy, omega_helm, A, f_int);
end

function x_next = waveholtz_apply_map(nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, pml, x, f_int)
    t_grid = (0:nt).' * dt;
    weights = cos(omega_ts*t_grid) - 0.25;

    W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x, f_int);
    integrand = bsxfun(@times, weights, W);
    integral = trapz(t_grid, integrand, 1);
    x_next = (2.0/T_ts) * integral.';
end

function niter = waveholtz_gmres_iter_count(iter_pair, restart)
    if isempty(iter_pair)
        niter = 0;
    elseif numel(iter_pair) == 1
        niter = iter_pair(1);
    else
        niter = max(0, (iter_pair(1) - 1) * restart + iter_pair(2));
    end
end

% =====================================================================
% Rolling/truncated Auto-DMD accelerated WaveHoltz iteration (v2)
%   Full-state delta-DMD on x=[u;v;psi1;psi2;theta]:
%     x_{k+1} = Pi*x_k + b,   d_k = x_{k+1}-x_k  =>  d_{k+1} = Pi*d_k
%   Prediction uses:
%     x_* ~= x_k + (I-Pi)^{-1} d_k   (projected in DMD subspace)
% =====================================================================
function [u_full, hist, singular_vals, num_window_truncations, num_predictions] = waveholtz_iteration_dmd_restarted( ...
    nx, ny, hx, hy, c, omega_ts, omega_helm, T_ts, dt, nt, f_int, pml, max_iter, tol, ...
    max_snaps, max_rank, predict_burst, accept_growth, epsSVD, A)

    nu = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nv = nvx + nvy;
    nstate = nu + nv + 3*nu;

    x = zeros(nstate, 1);
    hist = zeros(max_iter,1);

    t_grid = (0:nt).' * dt;
    weights = cos(omega_ts*t_grid) - 0.25;

    % incremental thin SVD on full-state updates:
    % Y_neg = [d^1, ..., d^{m-1}], d = x_{k+1}-x_k in R^{nstate}
    U = []; S = []; V = [];

    DeltaSnaps = zeros(nstate, max_snaps);
    snap_count = 0;

    num_window_truncations = 0;
    num_predictions = 0;
    singular_vals = [];
    burst_remaining = 0;

    for k = 1:max_iter
        W = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x, f_int);
        integrand = bsxfun(@times, weights, W);
        integral = trapz(t_grid, integrand, 1);
        x_new = (2.0/T_ts) * integral.';

        delta = x_new - x;
        x_plain = x_new;
        plain_res = norm(delta) / max(1e-16, norm(x_plain));
        iter_res = plain_res;

        % Push into a rolling window.  Prediction no longer resets this
        % history; if the window is full, drop only the oldest update and
        % rebuild a truncated thin SVD for the retained data.
        window_truncated = false;
        if snap_count < max_snaps
            snap_count = snap_count + 1;
            DeltaSnaps(:, snap_count) = delta;
        else
            DeltaSnaps(:, 1:end-1) = DeltaSnaps(:, 2:end);
            DeltaSnaps(:, end) = delta;
            window_truncated = true;
            num_window_truncations = num_window_truncations + 1;
        end

        predicted = false;

        if snap_count >= 2
            % Update SVD with the previous delta, so U,S,V represent
            % Y_neg=[d_1,...,d_{m-1}].  If the rolling window shifted, rebuild
            % the compact factors from the retained snapshots.
            if window_truncated
                [U, S, V] = snapshotSVD(DeltaSnaps(:, 1:snap_count-1), epsSVD, max_rank);
            elseif snap_count == 2
                % initialize with d_1
                [U, S, V] = initSVD(DeltaSnaps(:,1), epsSVD);
                [U, S, V] = truncateSVD(U, S, V, epsSVD, max_rank);
            else
                % append d_{snap_count-1}
                [U, S, V] = incrementalSVD(DeltaSnaps(:, snap_count-1), epsSVD, U, S, V);
                [U, S, V] = truncateSVD(U, S, V, epsSVD, max_rank);
            end

            % Form Y_pos = [d_2, ..., d_m]
            Y_pos = DeltaSnaps(:, 2:snap_count);

            if ~isempty(U)
                % A_tilde = U^* Y_pos V S^{-1}
                A_tilde = (U') * Y_pos * V / S;

                % diagnostics: ||A_tilde||_2 and spectral radius
                [~, SigmaA, ~] = svd(A_tilde, 'econ');
                spec2 = max(diag(SigmaA));
                lam = eig(A_tilde);
                rho = max(abs(lam));

                sigmas = diag(S);
                max_sig = max(sigmas);
                min_sig = min(sigmas(sigmas > 1e-17));
                if isempty(min_sig), min_sig = 0; end
                cond_ratio = min_sig / max(1e-16, max_sig);

                fprintf('  DMD: m=%d, ||A||2=%.6f, rho=%.6f, min_sig=%.3e, max_sig=%.3e, cond_ratio=%.3e\n', ...
                    snap_count, spec2, rho, min_sig, max_sig, cond_ratio);

                trigger_prediction = cond_ratio < 1e-3 && rho < 0.999;
                if trigger_prediction
                    burst_remaining = max(burst_remaining, predict_burst);
                end

                % Once triggered, use the same strict stability condition for
                % a short burst.  The candidate is accepted only after one
                % additional WaveHoltz map confirms that its fixed-point
                % residual is no worse than the ordinary step residual.
                can_predict = burst_remaining > 0 && cond_ratio < 1e-3 && rho < 0.999;
                if can_predict
                    r = size(A_tilde,1);
                    Iden = eye(r);
                    delta_z = (Iden - A_tilde) \ ((U') * delta);
                    x_candidate = x + U * delta_z;

                    W_check = solve_wave_one_period(nx, ny, hx, hy, c, omega_ts, omega_helm, dt, nt, pml, x_candidate, f_int);
                    integrand_check = bsxfun(@times, weights, W_check);
                    integral_check = trapz(t_grid, integrand_check, 1);
                    x_check = (2.0/T_ts) * integral_check.';
                    pred_res = norm(x_check - x_candidate) / max(1e-16, norm(x_check));

                    if pred_res <= accept_growth * plain_res
                        x_new = x_candidate;
                        predicted = true;
                        iter_res = pred_res;
                        num_predictions = num_predictions + 1;
                        burst_remaining = max(0, burst_remaining - 1);
                        fprintf('  >>> DMD prediction accepted: pred-res=%.3e, plain-res=%.3e, burst remaining %d\n', ...
                            pred_res, plain_res, burst_remaining);
                    else
                        x_new = x_plain;
                        burst_remaining = 0;
                        fprintf('  >>> DMD prediction rejected: pred-res=%.3e, plain-res=%.3e\n', ...
                            pred_res, plain_res);
                    end
                end

                if predicted
                    singular_vals = diag(S);
                end
            end
        end

        % residual + accept
        res = iter_res;
        hist(k) = res;
        x = x_new;

        fprintf('dmd   iter %3d: rel-res = %.3e\n', k, res);
        if res < tol
            hist = hist(1:k);
            break;
        end
    end

    u_full = recover_complex_from_state(x, nx, ny, hx, hy, omega_helm, A, f_int);
end

function u_full = recover_complex_from_state(x, nx, ny, hx, hy, omega_helm, A, f_int)
    nu = nx*ny;
    nvx = (nx+1)*ny;
    nvy = nx*(ny+1);
    nv = nvx + nvy;

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

% =====================================================================
% Incremental thin SVD utilities
% =====================================================================
function [W, sigma, V] = initSVD(u, epsSVD)
    if norm(u) > epsSVD
        sigma = norm(u);
        W = u / sigma;
        V = 1;
        sigma = diag(sigma);
    else
        sigma = [];
        W = [];
        V = [];
    end
end

function [U, S, V] = snapshotSVD(Y, epsSVD, max_rank)
    if isempty(Y)
        U = [];
        S = [];
        V = [];
        return;
    end

    col_norms = vecnorm(Y);
    active = col_norms > epsSVD;
    if ~any(active)
        U = [];
        S = [];
        V = [];
        return;
    end

    [U, S, V] = svd(Y, 'econ');
    [U, S, V] = truncateSVD(U, S, V, epsSVD, max_rank);
end

function [U, S, V] = truncateSVD(U, S, V, epsSVD, max_rank)
    if isempty(S)
        U = [];
        V = [];
        return;
    end

    sigmas = diag(S);
    keep = find(sigmas > epsSVD);
    if nargin >= 5 && max_rank > 0 && numel(keep) > max_rank
        keep = keep(1:max_rank);
    end

    if isempty(keep)
        U = [];
        S = [];
        V = [];
        return;
    end

    U = U(:, keep);
    S = diag(sigmas(keep));
    V = V(:, keep);
end

function [U_r, S_r, V_r] = incrementalSVD(u, epsSVD, U, S, V)
    % Update thin SVD of Y with new column u:  Y_new = [Y, u]
    if isempty(U)
        [U_r, S_r, V_r] = initSVD(u, epsSVD);
        return;
    end

    % Projection + residual
    p = U' * u;
    r = u - U * p;
    r_norm = norm(r);

    if r_norm > epsSVD
        r_hat = r / r_norm;
        K = [S, p; zeros(1,size(S,2)), r_norm];
        [Uk, Sk, Vk] = svd(K, 'econ');

        U_r = [U, r_hat] * Uk;
        S_r = Sk;
        V_r = blkdiag(V,1) * Vk;
    else
        K = [S, p];
        [Uk, Sk, Vk] = svd(K, 'econ');

        U_r = U * Uk;
        S_r = Sk;
        V_r = blkdiag(V,1) * Vk;
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
