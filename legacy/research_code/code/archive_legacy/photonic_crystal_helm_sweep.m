function results = photonic_crystal_helm_sweep(cfg)
%PHOTONIC_CRYSTAL_HELM_SWEEP
% Sweep Helmholtz-only photonic-crystal waveguide parameters.
%
% The current first diagnostic is lambda/a.  Each case reuses the fixed
% photonic-crystal setup and scores how much field stays in the central
% channel relative to the upper/lower crystal region.

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);

    ncase = numel(cfg.lambda_over_a_values);
    cases = repmat(empty_case(), ncase, 1);
    fprintf('=== photonic_crystal_helm_sweep ===\n');
    fprintf('cases=%d, pc_points_per_a=%g, rows=%d, rod=%s\n', ...
        ncase, cfg.pc_points_per_a, cfg.pc_num_rows, cfg.pc_rod_shape);

    for k = 1:ncase
        lambda_over_a = cfg.lambda_over_a_values(k);
        run_cfg = cfg.base_cfg;
        run_cfg.pc_points_per_a = cfg.pc_points_per_a;
        run_cfg.pc_num_rows = cfg.pc_num_rows;
        run_cfg.pc_rod_shape = cfg.pc_rod_shape;
        run_cfg.pc_lattice_a = cfg.pc_lattice_a;
        run_cfg.Lx = cfg.pc_num_cols * cfg.pc_lattice_a;
        run_cfg.Ly = (2 * cfg.pc_num_rows + 1) * cfg.pc_lattice_a;
        run_cfg.L = run_cfg.Ly;
        run_cfg.pc_radius = cfg.pc_radius_factor * cfg.pc_lattice_a;
        run_cfg.pc_square_halfwidth = cfg.pc_square_halfwidth_factor * cfg.pc_lattice_a;
        run_cfg.omega = 2 * pi / (lambda_over_a * cfg.pc_lattice_a);
        run_cfg.do_plots = false;
        run_cfg.wh_initial_guess = 'helm_lift';
        run_cfg.wh_auto_refine_against_helm = false;

        r = waveholtz_2d_pml_photonic_crystal(run_cfg);
        u_helm = r.u_helm_phys(:, :, 1);
        metric = score_helm_field(r, u_helm, cfg);

        cases(k).lambda_over_a = lambda_over_a;
        cases(k).omega = run_cfg.omega;
        cases(k).score = metric.score;
        cases(k).leak_ratio = metric.leak_ratio;
        cases(k).channel_power = metric.channel_power;
        cases(k).leak_power = metric.leak_power;
        cases(k).source_power = metric.source_power;
        cases(k).rel_wave_vs_helm = r.rel_wave_vs_helm_phys(1);
        cases(k).result = r;

        fprintf(['%2d/%2d: lambda/a=%.4f, omega=%.4f, score=%.3e, ' ...
            'leak/channel=%.3e, channel=%.3e, WH-Helm=%.1e\n'], ...
            k, ncase, cases(k).lambda_over_a, cases(k).omega, cases(k).score, ...
            cases(k).leak_ratio, cases(k).channel_power, cases(k).rel_wave_vs_helm);
    end

    [~, order] = sort([cases.score], 'descend');
    results = struct();
    results.cfg = cfg;
    results.cases = cases;
    results.rank_order = order(:);
    results.best = cases(order(1));

    fprintf('\n=== Best Helm candidates ===\n');
    nshow = min(cfg.plot_top_n, ncase);
    for j = 1:nshow
        c = cases(order(j));
        fprintf('#%d lambda/a=%.4f, omega=%.4f, score=%.3e, leak/channel=%.3e\n', ...
            j, c.lambda_over_a, c.omega, c.score, c.leak_ratio);
    end

    if cfg.do_plots
        make_sweep_plots(results);
    end
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'pc_lattice_a', 1.0);
    cfg = set_default(cfg, 'pc_points_per_a', 4);
    cfg = set_default(cfg, 'pc_num_rows', 5);
    cfg = set_default(cfg, 'pc_num_cols', 21);
    cfg = set_default(cfg, 'pc_rod_shape', 'circle');
    cfg = set_default(cfg, 'pc_radius_factor', 0.25);
    cfg = set_default(cfg, 'pc_square_halfwidth_factor', 0.25);
    cfg = set_default(cfg, 'lambda_over_a_values', 1.80:0.05:2.20);
    cfg = set_default(cfg, 'channel_halfwidth_a', 0.55);
    cfg = set_default(cfg, 'leak_start_a', 0.85);
    cfg = set_default(cfg, 'source_exclusion_a', 2.0);
    cfg = set_default(cfg, 'plot_top_n', 6);
    cfg = set_default(cfg, 'do_plots', true);
    cfg = set_default(cfg, 'base_cfg', struct());
end

function c = empty_case()
    c = struct();
    c.lambda_over_a = NaN;
    c.omega = NaN;
    c.score = NaN;
    c.leak_ratio = NaN;
    c.channel_power = NaN;
    c.leak_power = NaN;
    c.source_power = NaN;
    c.rel_wave_vs_helm = NaN;
    c.result = [];
end

function metric = score_helm_field(r, u_helm, cfg)
    prob = r.problem;
    [X, Y] = ndgrid(prob.x_phys, prob.y_phys);
    a = cfg.pc_lattice_a;
    y_mid = 0.5 * prob.Ly;
    x_src = prob.x_phys(r.cfg.source_col_phys);

    away_from_source = abs(X - x_src) >= cfg.source_exclusion_a * a;
    channel_mask = abs(Y - y_mid) <= cfg.channel_halfwidth_a * a;
    leak_mask = abs(Y - y_mid) >= cfg.leak_start_a * a;
    source_mask = ~away_from_source;

    power = abs(u_helm).^2;
    channel_power = sum(power(channel_mask & away_from_source), 'all');
    leak_power = sum(power(leak_mask & away_from_source), 'all');
    source_power = sum(power(source_mask), 'all');

    metric = struct();
    metric.channel_power = channel_power;
    metric.leak_power = leak_power;
    metric.source_power = source_power;
    metric.leak_ratio = leak_power / max(channel_power, 1e-30);
    metric.score = channel_power / max(leak_power, 1e-30);
end

function make_sweep_plots(results)
    cases = results.cases;
    order = results.rank_order;
    lambda_vals = [cases.lambda_over_a];
    leak_ratio = [cases.leak_ratio];
    score = [cases.score];

    figure('Name', 'Photonic Crystal Helm Sweep Metrics');
    subplot(1, 2, 1);
    semilogy(lambda_vals, leak_ratio, 'o-', 'LineWidth', 1.4);
    grid on;
    xlabel('\lambda/a');
    ylabel('leak / channel');
    title('Leakage metric');

    subplot(1, 2, 2);
    semilogy(lambda_vals, score, 'o-', 'LineWidth', 1.4);
    grid on;
    xlabel('\lambda/a');
    ylabel('channel / leak');
    title('Guiding score');

    nshow = min(results.cfg.plot_top_n, numel(cases));
    figure('Name', 'Top Helm Sweep Fields');
    tlo = tiledlayout(ceil(nshow / 2), 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    for j = 1:nshow
        c = cases(order(j));
        nexttile;
        r = c.result;
        imagesc(r.problem.x_phys, r.problem.y_phys, real(r.u_helm_phys(:, :, 1)).');
        set(gca, 'YDir', 'normal');
        axis equal tight;
        colorbar;
        clim_symmetric(real(r.u_helm_phys(:, :, 1)));
        title(sprintf('\\lambda/a=%.3f, leak/ch=%.2e', c.lambda_over_a, c.leak_ratio));
        xlabel('x');
        ylabel('y');
    end
    title(tlo, 'Best Helm fields by channel/leak score');
end

function clim_symmetric(field)
    vmax = max(abs(field(:)));
    if isfinite(vmax) && vmax > 0
        caxis([-vmax, vmax]);
    end
end

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
