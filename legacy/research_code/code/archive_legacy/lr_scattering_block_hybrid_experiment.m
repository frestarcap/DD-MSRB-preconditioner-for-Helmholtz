function report = lr_scattering_block_hybrid_experiment(user_cfg)
%LR_SCATTERING_BLOCK_HYBRID_EXPERIMENT
% Step 1: blockwise rank/residual diagnostics
% Step 2: Hybrid Galerkin (near full-order + far reduced-order)
%
% This script does NOT modify existing scattering code. It reuses the same
% parameter setup and discretization as waveholtz_2d_pml_scattering_parametric.

    if nargin < 1
        user_cfg = struct();
    end
    cfg = lrh_default_cfg(user_cfg);

    if ~exist(cfg.output_dir, 'dir')
        mkdir(cfg.output_dir);
    end

    fprintf('=== lr_scattering_block_hybrid_experiment ===\n');
    fprintf('Output dir: %s\n', cfg.output_dir);

    [prob, ratio_list, log_ratio_list] = lrh_build_problem(cfg);
    nsample = numel(ratio_list);

    fprintf('Problem: nx=%d, ny=%d, nu=%d, nsample=%d\n', prob.nx, prob.ny, prob.nu, nsample);
    fprintf('ratio range: [%.6f, %.6f]\n', ratio_list(1), ratio_list(end));

    samples = cell(nsample, 1);
    for sid = 1:nsample
        fprintf('\n[%02d/%02d] Solving FOM at ratio=%.6f ...\n', sid, nsample, ratio_list(sid));
        samples{sid} = lrh_solve_one_sample(prob, ratio_list(sid), log_ratio_list(sid));
        fprintf('  solve done: ||u||=%.3e, nnz(A)=%d\n', norm(samples{sid}.u_vec), nnz(samples{sid}.A));
    end

    train_idx = 1:2:nsample;
    test_idx = 2:2:nsample;
    if isempty(test_idx)
        test_idx = train_idx;
    end

    fprintf('\nTrain size=%d, Test size=%d\n', numel(train_idx), numel(test_idx));

    blocks = lrh_build_radial_sector_blocks(prob, cfg, samples);
    step1 = lrh_run_step1(prob, cfg, samples, blocks, train_idx, test_idx);

    report = struct();
    report.cfg = cfg;
    report.prob = prob;
    report.train_idx = train_idx;
    report.test_idx = test_idx;
    report.blocks = blocks;
    report.step1 = step1;
    report.step2 = [];

    if ~step1.promising
        fprintf('\nStep 1 indicates low promise. Stop before Step 2.\n');
        report.stop_reason = 'step1_not_promising';
        lrh_write_report(cfg, report, ratio_list, samples);
        save(fullfile(cfg.output_dir, 'lr_scattering_block_hybrid_report.mat'), 'report', '-v7.3');
        return;
    end

    fprintf('\nStep 1 is promising. Proceeding to Step 2 Hybrid Galerkin.\n');
    step2 = lrh_run_step2(prob, cfg, samples, blocks, train_idx, test_idx, step1);
    report.step2 = step2;
    report.stop_reason = '';

    lrh_write_report(cfg, report, ratio_list, samples);
    save(fullfile(cfg.output_dir, 'lr_scattering_block_hybrid_report.mat'), 'report', '-v7.3');
end


function cfg = lrh_default_cfg(user_cfg)
    cfg = struct();

    % Match parametric baseline
    cfg.ratio_min = 9/10;
    cfg.ratio_max = 10/9;
    cfg.nsample = 24;

    cfg.L = 1.0;
    cfg.c = 1.0;
    cfg.omega = 18.0;

    cfg.Nphys = 85;
    cfg.ispml = 1;
    cfg.npml = 30;
    cfg.sigma_max = 80;

    cfg.center = [0.5, 0.5];
    cfg.radius_ref = 0.12;
    cfg.eta_boundary_min = 0.05;

    cfg.CFL = 0.20;

    % Blocking setup
    cfg.num_sectors = 8;
    cfg.r_near = 2.0 * cfg.radius_ref;
    cfg.r_mid = 3.4 * cfg.radius_ref;

    % Step-1 diagnostics
    cfg.rank_eps_list = [1e-2, 1e-3, 1e-4];
    cfg.rank_eps_for_map = 1e-3;
    cfg.step1_recon_tol = 1e-3;

    % Promise gate
    cfg.promise_near_rank_fraction_min = 0.70;
    cfg.promise_far_rank_fraction_max = 0.50;
    cfg.promise_contrast_min = 1.50;

    % Step-2 hybrid
    cfg.hybrid_rank_tol = 1e-3;
    cfg.full_order_rank_fraction = 0.75;
    cfg.gmres_tol = 1e-8;
    cfg.gmres_maxit = 400;
    cfg.gmres_restart = 60;

    cfg.output_dir = fullfile(pwd, 'lr_hybrid_outputs');

    cfg = lrh_apply_user_cfg(cfg, user_cfg);
end


function cfg = lrh_apply_user_cfg(cfg, user_cfg)
    if isempty(user_cfg)
        return;
    end
    fields = fieldnames(user_cfg);
    for k = 1:numel(fields)
        cfg.(fields{k}) = user_cfg.(fields{k});
    end
end


function [prob, ratio_list, log_ratio_list] = lrh_build_problem(cfg)
    prob = struct();
    prob.L = cfg.L;
    prob.c = cfg.c;
    prob.omega = cfg.omega;

    % Physical grid
    prob.nx_phys = cfg.Nphys;
    prob.ny_phys = cfg.Nphys;
    prob.hx = prob.L / (prob.nx_phys + 1);
    prob.hy = prob.L / (prob.ny_phys + 1);

    % PML-extended grid
    prob.nx = prob.nx_phys + 2 * cfg.npml;
    prob.ny = prob.ny_phys + 2 * cfg.npml;
    prob.nu = prob.nx * prob.ny;

    Lpml_x = cfg.npml * prob.hx;
    Lpml_y = cfg.npml * prob.hy;
    prob.x = linspace(-Lpml_x, prob.L + Lpml_x, prob.nx + 2);
    prob.y = linspace(-Lpml_y, prob.L + Lpml_y, prob.ny + 2);
    [prob.X, prob.Y] = ndgrid(prob.x, prob.y);

    prob.center = cfg.center;
    prob.radius_ref = cfg.radius_ref;
    prob.eta_boundary_min = cfg.eta_boundary_min;

    prob.I_phys = (cfg.npml + 1):(cfg.npml + prob.nx_phys);
    prob.J_phys = (cfg.npml + 1):(cfg.npml + prob.ny_phys);

    % Time-step matching (kept for consistency with baseline setup)
    dt0 = cfg.CFL * min(prob.hx, prob.hy) / prob.c;
    [prob.omega_ts, prob.c_w, prob.nt, prob.dt] = match_omega_and_period(prob.omega, dt0);

    fprintf('=== Time stepping match (baseline-consistent) ===\n');
    fprintf('omega(helm)=%.6f, omega(ts)=%.6f, c_w=%.6f\n', prob.omega, prob.omega_ts, prob.c_w);
    fprintf('dt=%.3e, nt=%d\n', prob.dt, prob.nt);

    % PML
    prob.pml = build_pml_outer(prob.nx, prob.ny, prob.omega, cfg.npml, cfg.sigma_max, cfg.ispml);

    % Incident wave
    prob.inc = build_incident_wave(prob.X, prob.omega, prob.c);

    log_ratio_list = linspace(log(cfg.ratio_min), log(cfg.ratio_max), cfg.nsample);
    ratio_list = exp(log_ratio_list);
end


function sample = lrh_solve_one_sample(prob, ratio_ab, log_ratio)
    semi_a = prob.radius_ref * sqrt(ratio_ab);
    semi_b = prob.radius_ref / sqrt(ratio_ab);

    geom = build_scatterer_geometry( ...
        prob.nx, prob.ny, prob.x, prob.y, prob.center, semi_a, semi_b, prob.eta_boundary_min);
    [A, b] = helmholtz_scattering_system(prob.nx, prob.ny, prob.hx, prob.hy, prob.omega, prob.pml, geom, prob.inc);

    u = A \ b;

    sample = struct();
    sample.ratio_ab = ratio_ab;
    sample.log_ratio = log_ratio;
    sample.semi_a = semi_a;
    sample.semi_b = semi_b;
    sample.geom = geom;
    sample.A = A;
    sample.b = b;
    sample.u_vec = u;
    sample.active_vec = reshape(geom.active(2:end-1, 2:end-1), prob.nu, 1);
end


function blocks = lrh_build_radial_sector_blocks(prob, cfg, samples)
    % Use geometry-independent partition on interior grid; keep only dofs
    % active for at least one training sample.
    nsample = numel(samples);
    active_any = false(prob.nu, 1);
    for k = 1:nsample
        active_any = active_any | samples{k}.active_vec;
    end

    x_int = prob.x(2:end-1);
    y_int = prob.y(2:end-1);
    [Xi, Yi] = ndgrid(x_int, y_int);
    R = hypot(Xi - prob.center(1), Yi - prob.center(2));
    Theta = atan2(Yi - prob.center(2), Xi - prob.center(1));

    Rv = reshape(R, prob.nu, 1);
    Tv = reshape(Theta, prob.nu, 1);

    nsec = cfg.num_sectors;
    edges = linspace(-pi, pi, nsec + 1);

    blocks = struct('name', {}, 'region', {}, 'dofs', {}, 'ring_id', {}, 'sector_id', {});

    % Near block (single block)
    near_mask = (Rv <= cfg.r_near) & active_any;
    if any(near_mask)
        b.name = 'near_all';
        b.region = 'near';
        b.dofs = find(near_mask);
        b.ring_id = 1;
        b.sector_id = 0;
        blocks(end + 1) = b; %#ok<AGROW>
    end

    % Mid annulus, sectorized
    mid_mask_all = (Rv > cfg.r_near) & (Rv <= cfg.r_mid) & active_any;
    for s = 1:nsec
        if s < nsec
            sm = Tv >= edges(s) & Tv < edges(s + 1);
        else
            sm = Tv >= edges(s) & Tv <= edges(s + 1);
        end
        mask = mid_mask_all & sm;
        if any(mask)
            b.name = sprintf('mid_s%02d', s);
            b.region = 'mid';
            b.dofs = find(mask);
            b.ring_id = 2;
            b.sector_id = s;
            blocks(end + 1) = b; %#ok<AGROW>
        end
    end

    % Far annulus, sectorized
    far_mask_all = (Rv > cfg.r_mid) & active_any;
    for s = 1:nsec
        if s < nsec
            sm = Tv >= edges(s) & Tv < edges(s + 1);
        else
            sm = Tv >= edges(s) & Tv <= edges(s + 1);
        end
        mask = far_mask_all & sm;
        if any(mask)
            b.name = sprintf('far_s%02d', s);
            b.region = 'far';
            b.dofs = find(mask);
            b.ring_id = 3;
            b.sector_id = s;
            blocks(end + 1) = b; %#ok<AGROW>
        end
    end

    fprintf('\nBuilt %d blocks: near=%d, mid=%d, far=%d\n', numel(blocks), ...
        nnz(strcmp({blocks.region}, 'near')), nnz(strcmp({blocks.region}, 'mid')), nnz(strcmp({blocks.region}, 'far')));

    % Coverage sanity
    covered = false(prob.nu, 1);
    for b = 1:numel(blocks)
        covered(blocks(b).dofs) = true;
    end
    fprintf('Active-any dofs=%d, covered=%d\n', nnz(active_any), nnz(covered & active_any));
end


function step1 = lrh_run_step1(prob, cfg, samples, blocks, train_idx, test_idx)
    fprintf('\n=== Step 1: blockwise rank + residual diagnostics ===\n');

    U_train = zeros(prob.nu, numel(train_idx));
    for k = 1:numel(train_idx)
        U_train(:, k) = samples{train_idx(k)}.u_vec;
    end

    rank_eps = cfg.rank_eps_list(:).';
    nblk = numel(blocks);
    ntrain = numel(train_idx);

    block_rank = zeros(nblk, numel(rank_eps));
    block_n = zeros(nblk, 1);
    block_sing = cell(nblk, 1);

    for b = 1:nblk
        dof = blocks(b).dofs;
        block_n(b) = numel(dof);
        Ub = U_train(dof, :);
        s = svd(Ub, 'econ');
        block_sing{b} = s;
        for e = 1:numel(rank_eps)
            block_rank(b, e) = lrh_energy_rank(s, rank_eps(e));
        end
        fprintf('  block %-10s: n=%5d, rank@1e-2=%2d, @1e-3=%2d, @1e-4=%2d\n', ...
            blocks(b).name, block_n(b), block_rank(b, 1), block_rank(b, 2), block_rank(b, 3));
    end

    map_eps = cfg.rank_eps_for_map;
    [~, ieps] = min(abs(rank_eps - map_eps));
    rank_for_map = block_rank(:, ieps);

    % Step-1 local basis with tolerance cfg.step1_recon_tol
    local_basis = lrh_build_local_bases(U_train, blocks, cfg.step1_recon_tol);

    res_rel = zeros(numel(test_idx), 1);
    err_rel = zeros(numel(test_idx), 1);
    point_accum = zeros(prob.nu, 1);

    for kt = 1:numel(test_idx)
        sid = test_idx(kt);
        A = samples{sid}.A;
        b = samples{sid}.b;
        u_ref = samples{sid}.u_vec;

        % Oracle projection onto local spaces (diagnostic upper bound)
        u_approx = zeros(prob.nu, 1);
        for bb = 1:numel(blocks)
            dof = blocks(bb).dofs;
            Vb = local_basis{bb}.V;
            if isempty(Vb)
                continue;
            end
            coeff = Vb' * u_ref(dof);
            u_approx(dof) = Vb * coeff;
        end

        r = A * u_approx - b;
        res_rel(kt) = norm(r) / max(norm(b), 1e-15);
        err_rel(kt) = norm(u_approx - u_ref) / max(norm(u_ref), 1e-15);
        point_accum = point_accum + abs(r);

        fprintf('  test sid=%2d ratio=%.6f: rel-res=%.3e, rel-err=%.3e\n', ...
            sid, samples{sid}.ratio_ab, res_rel(kt), err_rel(kt));
    end

    point_mean_abs_res = point_accum / max(numel(test_idx), 1);

    % Promise gate
    near_ids = find(strcmp({blocks.region}, 'near'));
    far_ids = find(strcmp({blocks.region}, 'far'));

    near_rank = max(rank_for_map(near_ids));
    if isempty(near_rank)
        near_rank = 0;
    end
    if isempty(far_ids)
        far_rank_med = ntrain;
    else
        far_rank_med = median(rank_for_map(far_ids));
    end

    near_frac = near_rank / max(ntrain, 1);
    far_frac = far_rank_med / max(ntrain, 1);
    contrast = near_rank / max(far_rank_med, 1);

    promising = (near_frac >= cfg.promise_near_rank_fraction_min) && ...
                (far_frac <= cfg.promise_far_rank_fraction_max) && ...
                (contrast >= cfg.promise_contrast_min);

    fprintf('\nStep-1 gate:\n');
    fprintf('  near rank frac=%.3f (>= %.3f?)\n', near_frac, cfg.promise_near_rank_fraction_min);
    fprintf('  far  rank frac=%.3f (<= %.3f?)\n', far_frac, cfg.promise_far_rank_fraction_max);
    fprintf('  contrast=%.3f (>= %.3f?)\n', contrast, cfg.promise_contrast_min);
    fprintf('  promising = %d\n', promising);

    % Save plots
    lrh_plot_step1_rank_map(prob, blocks, rank_for_map, ntrain, cfg);
    lrh_plot_step1_residual_map(prob, point_mean_abs_res, samples, cfg);
    lrh_plot_step1_spectrum(block_sing, blocks, cfg);

    step1 = struct();
    step1.rank_eps = rank_eps;
    step1.block_rank = block_rank;
    step1.block_n = block_n;
    step1.block_sing = block_sing;
    step1.rank_for_map = rank_for_map;
    step1.res_rel = res_rel;
    step1.err_rel = err_rel;
    step1.point_mean_abs_res = point_mean_abs_res;
    step1.promising = promising;
    step1.gate = struct('near_rank', near_rank, 'far_rank_med', far_rank_med, ...
                        'near_frac', near_frac, 'far_frac', far_frac, 'contrast', contrast);
    step1.local_basis = local_basis;
end


function local_basis = lrh_build_local_bases(U_train, blocks, tol)
    local_basis = cell(numel(blocks), 1);
    for b = 1:numel(blocks)
        dof = blocks(b).dofs;
        Ub = U_train(dof, :);
        [V, S, ~] = svd(Ub, 'econ');
        s = diag(S);
        r = lrh_energy_rank(s, tol);
        if r == 0
            Vuse = zeros(numel(dof), 0);
        else
            Vuse = V(:, 1:r);
        end
        local_basis{b} = struct('V', Vuse, 'rank', r, 'tol', tol);
    end
end


function rank = lrh_energy_rank(s, eps)
    if isempty(s)
        rank = 0;
        return;
    end
    e = abs(s).^2;
    total = sum(e);
    if total <= 0
        rank = 0;
        return;
    end
    prefix = [0; cumsum(e(:))];
    % smallest r such that sum_{j>r} s_j^2 / sum_j s_j^2 < eps^2
    rank = numel(s);
    for r = 0:numel(s)
        rem = total - prefix(r + 1);
        if rem / total < eps^2
            rank = r;
            break;
        end
    end
end


function lrh_plot_step1_rank_map(prob, blocks, block_rank, ntrain, cfg)
    rank_map = nan(prob.nu, 1);
    for b = 1:numel(blocks)
        rank_map(blocks(b).dofs) = block_rank(b) / max(ntrain, 1);
    end

    rank_grid = reshape(rank_map, prob.nx, prob.ny);

    fig = figure('Visible', 'off');
    imagesc(prob.x(2:end-1), prob.y(2:end-1), rank_grid.');
    axis equal tight;
    set(gca, 'YDir', 'normal');
    xlabel('x');
    ylabel('y');
    title(sprintf('Step1 rank fraction map (eps=%.0e)', cfg.rank_eps_for_map));
    cb = colorbar;
    ylabel(cb, 'rank / n_{train}');
    grid on;

    out_png = fullfile(cfg.output_dir, 'step1_rank_fraction_map.png');
    exportgraphics(fig, out_png, 'Resolution', 180);
    close(fig);
end


function lrh_plot_step1_residual_map(prob, point_mean_abs_res, samples, cfg)
    % mask by active-any for readability
    active_any = false(prob.nu, 1);
    for k = 1:numel(samples)
        active_any = active_any | samples{k}.active_vec;
    end

    val = point_mean_abs_res;
    val(~active_any) = nan;
    val_grid = reshape(val, prob.nx, prob.ny);

    fig = figure('Visible', 'off');
    imagesc(prob.x(2:end-1), prob.y(2:end-1), log10(val_grid.'));
    axis equal tight;
    set(gca, 'YDir', 'normal');
    xlabel('x');
    ylabel('y');
    title('Step1 mean abs residual map: log10(|r|)');
    colorbar;
    grid on;

    out_png = fullfile(cfg.output_dir, 'step1_mean_abs_residual_map.png');
    exportgraphics(fig, out_png, 'Resolution', 180);
    close(fig);
end


function lrh_plot_step1_spectrum(block_sing, blocks, cfg)
    fig = figure('Visible', 'off', 'Position', [100, 100, 1500, 820]);
    hold on;
    hlist = gobjects(0);
    labels = cell(0, 1);
    for b = 1:numel(blocks)
        s = block_sing{b};
        if isempty(s)
            continue;
        end
        sr = s / max(s(1), 1e-15);
        if strcmp(blocks(b).region, 'near')
            style = '-';
            lw = 2.0;
        elseif strcmp(blocks(b).region, 'mid')
            style = '--';
            lw = 1.3;
        else
            style = ':';
            lw = 1.3;
        end
        h = semilogy(1:numel(sr), sr, style, 'LineWidth', lw, ...
            'DisplayName', blocks(b).name);
        hlist(end + 1, 1) = h; %#ok<AGROW>
        labels{end + 1, 1} = blocks(b).name; %#ok<AGROW>
    end
    xlabel('index');
    ylabel('\sigma_i/\sigma_1 (log scale)');
    title('Step1 blockwise singular value decay (log-scale y)');
    set(gca, 'YScale', 'log');
    ylim([1e-12, 2e0]);
    grid on;
    if ~isempty(hlist)
        legend(hlist, labels, 'Location', 'eastoutside', 'Interpreter', 'none', 'FontSize', 9);
    end
    hold off;

    out_png = fullfile(cfg.output_dir, 'step1_block_singular_decay.png');
    exportgraphics(fig, out_png, 'Resolution', 180);
    close(fig);
end


function step2 = lrh_run_step2(prob, cfg, samples, blocks, train_idx, test_idx, step1)
    fprintf('\n=== Step 2: Hybrid Galerkin ===\n');

    U_train = zeros(prob.nu, numel(train_idx));
    for k = 1:numel(train_idx)
        U_train(:, k) = samples{train_idx(k)}.u_vec;
    end

    ntrain = numel(train_idx);
    [~, ieps] = min(abs(step1.rank_eps - cfg.hybrid_rank_tol));
    block_rank = step1.block_rank(:, ieps);

    full_blocks = find(block_rank >= ceil(cfg.full_order_rank_fraction * ntrain));
    red_blocks = setdiff(1:numel(blocks), full_blocks);

    fprintf('Hybrid partition: full blocks=%d, reduced blocks=%d\n', numel(full_blocks), numel(red_blocks));

    local_basis = cell(numel(blocks), 1);
    for b = 1:numel(blocks)
        dof = blocks(b).dofs;
        Ub = U_train(dof, :);
        [V, S, ~] = svd(Ub, 'econ');
        s = diag(S);
        r = lrh_energy_rank(s, cfg.hybrid_rank_tol);
        if isempty(find(red_blocks == b, 1))
            local_basis{b} = struct('is_full', true, 'rank', numel(dof), 'V', []);
        else
            if r > 0
                Vuse = V(:, 1:r);
            else
                Vuse = zeros(numel(dof), 0);
            end
            local_basis{b} = struct('is_full', false, 'rank', r, 'V', Vuse);
        end
    end

    [Q, q_meta] = lrh_build_hybrid_basis(prob.nu, blocks, local_basis);
    fprintf('Hybrid basis size: Q is %d x %d\n', size(Q, 1), size(Q, 2));

    nt = numel(test_idx);
    rel_res_h = zeros(nt, 1);
    rel_err_h = zeros(nt, 1);
    time_red = zeros(nt, 1);

    rel_res_gm0 = zeros(nt, 1);
    iter_gm0 = zeros(nt, 1);
    time_gm0 = zeros(nt, 1);

    rel_res_gmh = zeros(nt, 1);
    iter_gmh = zeros(nt, 1);
    time_gmh = zeros(nt, 1);

    for kt = 1:nt
        sid = test_idx(kt);
        A = samples{sid}.A;
        b = samples{sid}.b;
        u_ref = samples{sid}.u_vec;

        t0 = tic;
        AQ = A * Q;
        Ar = Q' * AQ;
        br = Q' * b;
        y = Ar \ br;
        u_h = Q * y;
        time_red(kt) = toc(t0);

        rel_res_h(kt) = norm(A * u_h - b) / max(norm(b), 1e-15);
        rel_err_h(kt) = norm(u_h - u_ref) / max(norm(u_ref), 1e-15);

        % Baseline GMRES from zero
        t1 = tic;
        [u0, ~, ~, iter0] = gmres(A, b, cfg.gmres_restart, cfg.gmres_tol, cfg.gmres_maxit);
        time_gm0(kt) = toc(t1);
        rel_res_gm0(kt) = norm(A * u0 - b) / max(norm(b), 1e-15);
        iter_gm0(kt) = lrh_gmres_iter_count(iter0, cfg.gmres_restart);

        % GMRES completion from hybrid initial guess
        rhs_corr = b - A * u_h;
        t2 = tic;
        [du, ~, ~, iterh] = gmres(A, rhs_corr, cfg.gmres_restart, cfg.gmres_tol, cfg.gmres_maxit);
        u_hc = u_h + du;
        time_gmh(kt) = toc(t2);
        rel_res_gmh(kt) = norm(A * u_hc - b) / max(norm(b), 1e-15);
        iter_gmh(kt) = lrh_gmres_iter_count(iterh, cfg.gmres_restart);

        fprintf(['  test sid=%2d ratio=%.6f | hybrid rel-res=%.3e rel-err=%.3e ' ...
                 '| gmres0 iter=%d time=%.3fs | gmres(hybrid) iter=%d time=%.3fs\n'], ...
            sid, samples{sid}.ratio_ab, rel_res_h(kt), rel_err_h(kt), ...
            round(iter_gm0(kt)), time_gm0(kt), round(iter_gmh(kt)), time_gmh(kt));
    end

    step2 = struct();
    step2.full_blocks = full_blocks;
    step2.red_blocks = red_blocks;
    step2.local_basis = local_basis;
    step2.q_meta = q_meta;
    step2.rel_res_hybrid = rel_res_h;
    step2.rel_err_hybrid = rel_err_h;
    step2.time_hybrid = time_red;
    step2.rel_res_gmres_zero = rel_res_gm0;
    step2.iter_gmres_zero = iter_gm0;
    step2.time_gmres_zero = time_gm0;
    step2.rel_res_gmres_hybrid = rel_res_gmh;
    step2.iter_gmres_hybrid = iter_gmh;
    step2.time_gmres_hybrid = time_gmh;

    step2.summary = struct();
    step2.summary.mean_rel_res_hybrid = mean(rel_res_h);
    step2.summary.mean_rel_err_hybrid = mean(rel_err_h);
    step2.summary.mean_time_hybrid = mean(time_red);
    step2.summary.mean_iter_gmres_zero = mean(iter_gm0);
    step2.summary.mean_iter_gmres_hybrid = mean(iter_gmh);
    step2.summary.mean_time_gmres_zero = mean(time_gm0);
    step2.summary.mean_time_gmres_hybrid = mean(time_gmh);
    step2.summary.iter_ratio_hybrid_vs_zero = mean(iter_gmh) / max(mean(iter_gm0), 1e-15);
    step2.summary.time_ratio_hybrid_vs_zero = mean(time_gmh) / max(mean(time_gm0), 1e-15);
end


function [Q, meta] = lrh_build_hybrid_basis(nu, blocks, local_basis)
    % Sparse assembly by triplets
    rows = [];
    cols = [];
    vals = [];
    col_ptr = 0;

    full_dof_count = 0;
    red_basis_count = 0;

    for b = 1:numel(blocks)
        dof = blocks(b).dofs;
        info = local_basis{b};

        if info.is_full
            nb = numel(dof);
            if nb == 0
                continue;
            end
            cidx = (col_ptr + 1):(col_ptr + nb);
            rows = [rows; dof(:)]; %#ok<AGROW>
            cols = [cols; cidx(:)]; %#ok<AGROW>
            vals = [vals; ones(nb, 1)]; %#ok<AGROW>
            col_ptr = col_ptr + nb;
            full_dof_count = full_dof_count + nb;
        else
            V = info.V;
            rb = size(V, 2);
            if rb == 0
                continue;
            end
            [ii, jj, vv] = find(V);
            rows = [rows; dof(ii)]; %#ok<AGROW>
            cols = [cols; (col_ptr + jj(:))]; %#ok<AGROW>
            vals = [vals; vv(:)]; %#ok<AGROW>
            col_ptr = col_ptr + rb;
            red_basis_count = red_basis_count + rb;
        end
    end

    Q = sparse(rows, cols, vals, nu, col_ptr);

    meta = struct();
    meta.total_cols = col_ptr;
    meta.full_dof_count = full_dof_count;
    meta.red_basis_count = red_basis_count;
end


function niter = lrh_gmres_iter_count(iter_pair, restart)
    if isempty(iter_pair) || numel(iter_pair) < 2
        niter = 0;
        return;
    end
    if restart <= 0
        restart = 1;
    end
    niter = (iter_pair(1) - 1) * restart + iter_pair(2);
end


function lrh_write_report(cfg, report, ratio_list, samples)
    p = fullfile(cfg.output_dir, 'REPORT_lr_scattering_block_hybrid.md');
    fid = fopen(p, 'w');
    if fid < 0
        warning('Cannot write report file: %s', p);
        return;
    end

    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, '# LR-style Blockwise Scattering Diagnostic + Hybrid Galerkin\n\n');
    fprintf(fid, 'Date: %s\n\n', datestr(now));

    fprintf(fid, '## Baseline-consistent settings\n');
    fprintf(fid, '- ratio range: [%.6f, %.6f], nsample=%d\n', report.cfg.ratio_min, report.cfg.ratio_max, report.cfg.nsample);
    fprintf(fid, '- L=%.3f, c=%.3f, omega=%.3f\n', report.cfg.L, report.cfg.c, report.cfg.omega);
    fprintf(fid, '- Nphys=%d, npml=%d, sigma_max=%.1f\n', report.cfg.Nphys, report.cfg.npml, report.cfg.sigma_max);
    fprintf(fid, '- center=(%.3f, %.3f), radius_ref=%.3f\n', report.cfg.center(1), report.cfg.center(2), report.cfg.radius_ref);
    fprintf(fid, '\n');

    fprintf(fid, '## Sample overview\n');
    fprintf(fid, '- solved samples: %d\n', numel(samples));
    fprintf(fid, '- ratios: min=%.6f, max=%.6f\n', min(ratio_list), max(ratio_list));
    fprintf(fid, '\n');

    s1 = report.step1;
    fprintf(fid, '## Step 1 diagnostics\n');
    fprintf(fid, '- near rank fraction (eps=%.0e): %.3f\n', report.cfg.rank_eps_for_map, s1.gate.near_frac);
    fprintf(fid, '- far median rank fraction (eps=%.0e): %.3f\n', report.cfg.rank_eps_for_map, s1.gate.far_frac);
    fprintf(fid, '- rank contrast near/far: %.3f\n', s1.gate.contrast);
    fprintf(fid, '- test mean oracle local-projection rel-residual: %.3e\n', mean(s1.res_rel));
    fprintf(fid, '- test mean oracle local-projection rel-error: %.3e\n', mean(s1.err_rel));
    fprintf(fid, '- promising: %d\n', s1.promising);
    fprintf(fid, '\n');

    fprintf(fid, 'Figures:\n');
    fprintf(fid, '- step1_rank_fraction_map.png\n');
    fprintf(fid, '- step1_mean_abs_residual_map.png\n');
    fprintf(fid, '- step1_block_singular_decay.png\n');
    fprintf(fid, '\n');

    if isempty(report.step2)
        fprintf(fid, '## Step 2\n');
        fprintf(fid, 'Skipped because Step 1 gate did not pass.\n');
        return;
    end

    s2 = report.step2;
    fprintf(fid, '## Step 2 Hybrid Galerkin\n');
    fprintf(fid, '- hybrid basis size: %d\n', s2.q_meta.total_cols);
    fprintf(fid, '- full-order dof columns: %d\n', s2.q_meta.full_dof_count);
    fprintf(fid, '- reduced columns: %d\n', s2.q_meta.red_basis_count);
    fprintf(fid, '- test mean rel-residual (hybrid projected solve): %.3e\n', s2.summary.mean_rel_res_hybrid);
    fprintf(fid, '- test mean rel-error (vs FOM): %.3e\n', s2.summary.mean_rel_err_hybrid);
    fprintf(fid, '- test mean reduced solve time: %.3fs\n', s2.summary.mean_time_hybrid);
    fprintf(fid, '- GMRES mean iter (zero init): %.2f\n', s2.summary.mean_iter_gmres_zero);
    fprintf(fid, '- GMRES mean iter (hybrid init): %.2f\n', s2.summary.mean_iter_gmres_hybrid);
    fprintf(fid, '- GMRES iter ratio (hybrid/zero): %.3f\n', s2.summary.iter_ratio_hybrid_vs_zero);
    fprintf(fid, '- GMRES mean time zero init: %.3fs\n', s2.summary.mean_time_gmres_zero);
    fprintf(fid, '- GMRES mean time hybrid init: %.3fs\n', s2.summary.mean_time_gmres_hybrid);
    fprintf(fid, '- GMRES time ratio (hybrid/zero): %.3f\n', s2.summary.time_ratio_hybrid_vs_zero);
end


% ==================== PDE helpers (copied from parametric baseline) ====================

function inc = build_incident_wave(X, omega, c)
    k = omega / c;
    u_inc_hat = exp(1i * k * X);
    inc = struct('k', k, 'u_inc_hat', u_inc_hat);
end


function geom = build_scatterer_geometry(nx, ny, x, y, center, semi_a, semi_b, eta_boundary_min)
    [X, Y] = ndgrid(x, y);
    interior = false(nx + 2, ny + 2);
    interior(2:end-1, 2:end-1) = true;

    ellipse_level = ((X - center(1)).^2) / (semi_a^2) + ((Y - center(2)).^2) / (semi_b^2);
    internal = interior & (ellipse_level <= 1.0);
    active = interior & ~internal;

    geom = struct();
    geom.center = center;
    geom.semi_a = semi_a;
    geom.semi_b = semi_b;
    geom.X = X;
    geom.Y = Y;
    geom.active = active;
    geom.internal = internal;
    geom.ghost_x = false(nx + 2, ny + 2);
    geom.ghost_y = false(nx + 2, ny + 2);
    geom.boundary_count_x = 0;
    geom.boundary_count_y = 0;

    zero_field = zeros(nx + 2, ny + 2);
    geom.gx_c2 = zero_field;
    geom.gx_c1 = zero_field;
    geom.gx_cg = zero_field;
    geom.gx_owner = zeros(nx + 2, ny + 2);
    geom.gx_u2 = zeros(nx + 2, ny + 2);
    geom.gx_gamma_x = zero_field;
    geom.gx_gamma_y = zero_field;

    geom.gy_c2 = zero_field;
    geom.gy_c1 = zero_field;
    geom.gy_cg = zero_field;
    geom.gy_owner = zeros(nx + 2, ny + 2);
    geom.gy_u2 = zeros(nx + 2, ny + 2);
    geom.gy_gamma_x = zero_field;
    geom.gy_gamma_y = zero_field;

    dims = [nx + 2, ny + 2];

    for i = 3:(nx)
        for j = 2:(ny + 1)
            if ~internal(i, j)
                continue;
            end

            left_active = active(i - 1, j);
            right_active = active(i + 1, j);
            root = 1 - ((y(j) - center(2))^2) / (semi_b^2);
            if root <= 0, error('Invalid x-ghost geometry at (%d,%d).', i, j); end

            if left_active && right_active
                % Robust fallback for rare pixel-level ambiguity:
                % keep the side with larger eta (more stable extrapolation).
                x_gamma_l = center(1) - semi_a * sqrt(root);
                x_gamma_r = center(1) + semi_a * sqrt(root);
                eta_l = abs(x_gamma_l - x(i - 1)) / (x(i) - x(i - 1));
                eta_r = abs(x(i + 1) - x_gamma_r) / (x(i + 1) - x(i));
                if eta_l >= eta_r
                    right_active = false;
                else
                    left_active = false;
                end
            end

            if left_active
                x_gamma = center(1) - semi_a * sqrt(root);
                eta = abs(x_gamma - x(i - 1)) / (x(i) - x(i - 1));
                if ~(eta > 0 && eta < 1), error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_x = geom.boundary_count_x + 1; end
                geom.ghost_x(i, j) = true;
                geom.gx_c2(i, j) = coeffs(1);
                geom.gx_c1(i, j) = coeffs(2);
                geom.gx_cg(i, j) = coeffs(3);
                geom.gx_owner(i, j) = sub2ind(dims, i - 1, j);
                geom.gx_u2(i, j) = sub2ind(dims, i - 2, j);
                geom.gx_gamma_x(i, j) = x_gamma;
                geom.gx_gamma_y(i, j) = y(j);
            elseif right_active
                x_gamma = center(1) + semi_a * sqrt(root);
                eta = abs(x(i + 1) - x_gamma) / (x(i + 1) - x(i));
                if ~(eta > 0 && eta < 1), error('x-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_x = geom.boundary_count_x + 1; end
                geom.ghost_x(i, j) = true;
                geom.gx_c2(i, j) = coeffs(1);
                geom.gx_c1(i, j) = coeffs(2);
                geom.gx_cg(i, j) = coeffs(3);
                geom.gx_owner(i, j) = sub2ind(dims, i + 1, j);
                geom.gx_u2(i, j) = sub2ind(dims, i + 2, j);
                geom.gx_gamma_x(i, j) = x_gamma;
                geom.gx_gamma_y(i, j) = y(j);
            end
        end
    end

    for i = 2:(nx + 1)
        for j = 3:(ny)
            if ~internal(i, j)
                continue;
            end

            down_active = active(i, j - 1);
            up_active = active(i, j + 1);
            root = 1 - ((x(i) - center(1))^2) / (semi_a^2);
            if root <= 0, error('Invalid y-ghost geometry at (%d,%d).', i, j); end

            if down_active && up_active
                % Robust fallback for rare pixel-level ambiguity.
                y_gamma_d = center(2) - semi_b * sqrt(root);
                y_gamma_u = center(2) + semi_b * sqrt(root);
                eta_d = abs(y_gamma_d - y(j - 1)) / (y(j) - y(j - 1));
                eta_u = abs(y(j + 1) - y_gamma_u) / (y(j + 1) - y(j));
                if eta_d >= eta_u
                    up_active = false;
                else
                    down_active = false;
                end
            end

            if down_active
                y_gamma = center(2) - semi_b * sqrt(root);
                eta = abs(y_gamma - y(j - 1)) / (y(j) - y(j - 1));
                if ~(eta > 0 && eta < 1), error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_y = geom.boundary_count_y + 1; end
                geom.ghost_y(i, j) = true;
                geom.gy_c2(i, j) = coeffs(1);
                geom.gy_c1(i, j) = coeffs(2);
                geom.gy_cg(i, j) = coeffs(3);
                geom.gy_owner(i, j) = sub2ind(dims, i, j - 1);
                geom.gy_u2(i, j) = sub2ind(dims, i, j - 2);
                geom.gy_gamma_x(i, j) = x(i);
                geom.gy_gamma_y(i, j) = y_gamma;
            elseif up_active
                y_gamma = center(2) + semi_b * sqrt(root);
                eta = abs(y(j + 1) - y_gamma) / (y(j + 1) - y(j));
                if ~(eta > 0 && eta < 1), error('y-ghost eta out of range at (%d,%d): %.3e', i, j, eta); end
                [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min);
                if boundary_mode, geom.boundary_count_y = geom.boundary_count_y + 1; end
                geom.ghost_y(i, j) = true;
                geom.gy_c2(i, j) = coeffs(1);
                geom.gy_c1(i, j) = coeffs(2);
                geom.gy_cg(i, j) = coeffs(3);
                geom.gy_owner(i, j) = sub2ind(dims, i, j + 1);
                geom.gy_u2(i, j) = sub2ind(dims, i, j + 2);
                geom.gy_gamma_x(i, j) = x(i);
                geom.gy_gamma_y(i, j) = y_gamma;
            end
        end
    end

    active_left = false(nx + 2, ny + 2);
    active_right = false(nx + 2, ny + 2);
    active_down = false(nx + 2, ny + 2);
    active_up = false(nx + 2, ny + 2);

    active_left(2:end, :) = active(1:end-1, :);
    active_right(1:end-1, :) = active(2:end, :);
    active_down(:, 2:end) = active(:, 1:end-1);
    active_up(:, 1:end-1) = active(:, 2:end);

    x_adjacent_internal = internal & (active_left | active_right);
    y_adjacent_internal = internal & (active_down | active_up);

    if any(x_adjacent_internal(:) & ~geom.ghost_x(:))
        error('Some x-adjacent internal nodes were not assigned ghost data.');
    end
    if any(y_adjacent_internal(:) & ~geom.ghost_y(:))
        error('Some y-adjacent internal nodes were not assigned ghost data.');
    end
end


function [coeffs, boundary_mode] = ghost_coefficients(eta, eta_boundary_min)
    if eta <= eta_boundary_min
        coeffs = [0, 0, 1];
        boundary_mode = true;
        return;
    end

    c2 = (1 - eta) / (1 + eta);
    c1 = 2 * (1 - eta) / eta;
    cg = 2 / (eta * (1 + eta));
    coeffs = [c2, c1, cg];
    boundary_mode = false;
end


function pml = build_pml_outer(nx, ny, omega, npml, sigma_max, ispml)
    sigma_x = zeros(nx + 2, 1);
    sigma_y = zeros(ny + 2, 1);

    for i = 1:npml
        if ispml == 1
            val = sigma_max * ((npml - (i - 1)) / npml)^2;
        else
            val = 0;
        end
        sigma_x(i) = val;
        sigma_x(end + 1 - i) = val;
        sigma_y(i) = val;
        sigma_y(end + 1 - i) = val;
    end

    s_x = 1.0 + 1i * sigma_x / omega;
    s_y = 1.0 + 1i * sigma_y / omega;
    s_x_half = 0.5 * (s_x(1:end-1) + s_x(2:end));
    s_y_half = 0.5 * (s_y(1:end-1) + s_y(2:end));

    pml = struct('sigma_x', sigma_x, 'sigma_y', sigma_y, ...
                 's_x', s_x, 's_y', s_y, ...
                 's_x_half', s_x_half, 's_y_half', s_y_half);
end


function [A_scatt, rhs] = helmholtz_scattering_system(nx, ny, hx, hy, omega, pml, geom, inc)
    Ntot = nx * ny;
    A_scatt = spalloc(Ntot, Ntot, 7 * Ntot);
    rhs = zeros(Ntot, 1);
    idx = @(ii, jj) ii + (jj - 1) * nx;

    dims_full = [nx + 2, ny + 2];

    for ii = 1:nx
        i = ii + 1;
        for jj = 1:ny
            j = jj + 1;
            row = idx(ii, jj);

            if geom.internal(i, j)
                A_scatt(row, row) = 1;
                continue;
            end

            a_x = 1.0 ./ pml.s_x(i);
            a_p = 1.0 ./ pml.s_x_half(i);
            a_m = 1.0 ./ pml.s_x_half(i - 1);
            b_y = 1.0 ./ pml.s_y(j);
            b_p = 1.0 ./ pml.s_y_half(j);
            b_m = 1.0 ./ pml.s_y_half(j - 1);

            A_scatt(row, row) = A_scatt(row, row) ...
                - a_x * (a_p + a_m) / (hx * hx) ...
                - b_y * (b_p + b_m) / (hy * hy) ...
                + omega^2;

            coef = a_x * a_p / (hx * hx);
            if geom.active(i + 1, j)
                A_scatt(row, idx(ii + 1, jj)) = A_scatt(row, idx(ii + 1, jj)) + coef;
            elseif geom.internal(i + 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i + 1, j, idx, dims_full, geom, inc);
            end

            coef = a_x * a_m / (hx * hx);
            if geom.active(i - 1, j)
                A_scatt(row, idx(ii - 1, jj)) = A_scatt(row, idx(ii - 1, jj)) + coef;
            elseif geom.internal(i - 1, j)
                [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, i - 1, j, idx, dims_full, geom, inc);
            end

            coef = b_y * b_p / (hy * hy);
            if geom.active(i, j + 1)
                A_scatt(row, idx(ii, jj + 1)) = A_scatt(row, idx(ii, jj + 1)) + coef;
            elseif geom.internal(i, j + 1)
                [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, i, j + 1, idx, dims_full, geom, inc);
            end

            coef = b_y * b_m / (hy * hy);
            if geom.active(i, j - 1)
                A_scatt(row, idx(ii, jj - 1)) = A_scatt(row, idx(ii, jj - 1)) + coef;
            elseif geom.internal(i, j - 1)
                [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, i, j - 1, idx, dims_full, geom, inc);
            end
        end
    end
end


function [A_scatt, rhs] = apply_ghost_x(A_scatt, rhs, row, coef, gi, gj, idx, dims_full, geom, inc)
    owner_lin = geom.gx_owner(gi, gj);
    if owner_lin == 0
        error('Missing x-ghost coefficients at (%d,%d).', gi, gj);
    end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    if ~geom.active(oi, oj)
        error('x-ghost owner at (%d,%d) is not active.', oi, oj);
    end

    ii = oi - 1;
    jj = oj - 1;
    A_scatt(row, idx(ii, jj)) = A_scatt(row, idx(ii, jj)) - coef * geom.gx_c1(gi, gj);

    u2_lin = geom.gx_u2(gi, gj);
    [u2i, u2j] = ind2sub(dims_full, u2_lin);
    if geom.active(u2i, u2j)
        A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * geom.gx_c2(gi, gj);
    end

    u_gamma = interp_incident(geom.gx_gamma_x(gi, gj), geom.gx_gamma_y(gi, gj), inc);
    rhs(row) = rhs(row) - coef * geom.gx_cg(gi, gj) * u_gamma;
end


function [A_scatt, rhs] = apply_ghost_y(A_scatt, rhs, row, coef, gi, gj, idx, dims_full, geom, inc)
    owner_lin = geom.gy_owner(gi, gj);
    if owner_lin == 0
        error('Missing y-ghost coefficients at (%d,%d).', gi, gj);
    end
    [oi, oj] = ind2sub(dims_full, owner_lin);
    if ~geom.active(oi, oj)
        error('y-ghost owner at (%d,%d) is not active.', oi, oj);
    end

    ii = oi - 1;
    jj = oj - 1;
    A_scatt(row, idx(ii, jj)) = A_scatt(row, idx(ii, jj)) - coef * geom.gy_c1(gi, gj);

    u2_lin = geom.gy_u2(gi, gj);
    [u2i, u2j] = ind2sub(dims_full, u2_lin);
    if geom.active(u2i, u2j)
        A_scatt(row, idx(u2i - 1, u2j - 1)) = A_scatt(row, idx(u2i - 1, u2j - 1)) + coef * geom.gy_c2(gi, gj);
    end

    u_gamma = interp_incident(geom.gy_gamma_x(gi, gj), geom.gy_gamma_y(gi, gj), inc);
    rhs(row) = rhs(row) - coef * geom.gy_cg(gi, gj) * u_gamma;
end


function u = interp_incident(xq, yq, inc)
    %#ok<INUSD> yq is unused for x-directed incident plane wave
    u = exp(1i * inc.k * xq);
end


function [omega_ts, c_w, nt, dt] = match_omega_and_period(omega, dt_init)
    dt = dt_init;
    if omega * dt / 2 >= 0.98
        dt = 0.98 * 2 / omega;
    end

    for it = 1:30
        omega_ts = (2 / dt) * asin(omega * dt / 2);
        nt = max(10, round(2 * pi / (omega_ts * dt)));
        dt = 2 * pi / (omega_ts * nt);
    end

    omega_ts = (2 / dt) * asin(omega * dt / 2);
    c_w = cos(omega_ts * dt / 2);
end
