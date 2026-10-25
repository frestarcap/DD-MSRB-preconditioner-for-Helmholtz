function results = photonic_crystal_waveguide_autodmd_standalone(cfg)
%PHOTONIC_CRYSTAL_WAVEGUIDE_AUTODMD_STANDALONE
% Isolated photonic-crystal waveguide test extracted from
% waveholtz_piecewise_scattering_standalone, with Auto-DMD restarted v2 as
% a WaveHoltz fixed-point accelerator.
%
% Example smoke run:
%   r = photonic_crystal_waveguide_autodmd_standalone(struct('show_plots', false));
%
% Example large reference run:
%   r = photonic_crystal_waveguide_autodmd_standalone(struct( ...
%       'pc_num_cols', 21, 'pc_num_rows', 7, 'pc_points_per_a', 6, ...
%       'wh_gmres_restart', 500, 'wh_gmres_maxit', 3, 'show_plots', true));

    if nargin < 1 || isempty(cfg)
        cfg = struct();
    end
    cfg = apply_defaults(cfg);

    base_cfg = make_waveguide_cfg(cfg);
    cases = {};
    if cfg.run_fixed_point
        cases{end + 1} = struct('label', 'plain_fixed_point', 'solver', 'fixed_point'); %#ok<AGROW>
    end
    if cfg.run_gmres
        cases{end + 1} = struct('label', 'gmres', 'solver', cfg.gmres_solver); %#ok<AGROW>
    end
    if cfg.run_dmd
        cases{end + 1} = struct('label', 'auto_dmd_restarted_v2', 'solver', 'auto_dmd_restarted_v2'); %#ok<AGROW>
    end
    if isempty(cases)
        error('At least one of run_fixed_point, run_gmres, or run_dmd must be true.');
    end

    fprintf('=== photonic_crystal_waveguide_autodmd_standalone ===\n');
    fprintf('pc: cols=%d, rows/side=%d, points/a=%d, c_pc=%.6f\n', ...
        cfg.pc_num_cols, cfg.pc_num_rows, cfg.pc_points_per_a, cfg.c_pc);
    fprintf('domain: Lx=%.4f, Ly=%.4f, Nphys=%d x %d, npml=%d\n', ...
        base_cfg.Lx, base_cfg.Ly, base_cfg.Nphys_x, base_cfg.Nphys_y, base_cfg.npml);
    fprintf('omega=%.6f, lambda/a=%.4f, filter_periods=%d\n', ...
        base_cfg.omega, cfg.lambda_over_a, cfg.filter_periods);

    results = struct();
    results.cfg = cfg;
    results.base_cfg = base_cfg;
    results.problem_name = 'photonic_crystal_waveguide_autodmd_standalone';
    results.cases = struct();
    results.summary = struct();

    for k = 1:numel(cases)
        run_cfg = base_cfg;
        run_cfg.wh_solver = cases{k}.solver;
        run_cfg.do_plots = false;
        run_cfg.wh_verbose_gmres = cfg.verbose_iter;
        fprintf('\n===== Case: %s =====\n', cases{k}.label);
        t_case = tic;
        run_result = waveholtz_piecewise_scattering_standalone(run_cfg);
        elapsed = toc(t_case);

        info = run_result.info(1);
        case_summary = struct();
        case_summary.label = cases{k}.label;
        case_summary.solver = cases{k}.solver;
        case_summary.flag = info.flag;
        case_summary.iter_count = info.iter_count;
        case_summary.relres = info.relres;
        case_summary.wave_vs_helm_phys = run_result.rel_wave_vs_helm_phys(1);
        case_summary.solve_time = info.solve_time;
        case_summary.total_time = elapsed;
        case_summary.recovery_method = info.recovery.method;
        case_summary.recovery_helm_relres = info.recovery.helm_relres;
        case_summary.converged = (info.flag == 0) && (info.relres <= run_cfg.wh_tol);
        if isfield(info.solver_extra, 'num_predictions')
            case_summary.dmd_predictions = info.solver_extra.num_predictions;
            case_summary.dmd_restarts = info.solver_extra.num_restarts;
            if isfield(info.solver_extra, 'num_window_truncations')
                case_summary.dmd_window_truncations = info.solver_extra.num_window_truncations;
            end
            case_summary.dmd_rejected = info.solver_extra.num_rejected;
            case_summary.dmd_last_rank = info.solver_extra.last_rank;
            case_summary.dmd_last_rho = info.solver_extra.last_rho;
            case_summary.dmd_last_cond_ratio = info.solver_extra.last_cond_ratio;
        end

        results.cases.(cases{k}.label) = run_result;
        results.summary.(cases{k}.label) = case_summary;
        print_case_summary(case_summary);
    end

    if isfield(results.summary, 'auto_dmd_restarted_v2')
        dmd_summary = results.summary.auto_dmd_restarted_v2;
        fprintf('\n===== Auto-DMD convergence verdict =====\n');
        if dmd_summary.converged
            fprintf('Auto-DMD converged: yes (relres=%.3e, iter=%d)\n', ...
                dmd_summary.relres, dmd_summary.iter_count);
        else
            fprintf('Auto-DMD converged: no (relres=%.3e, iter=%d, flag=%d)\n', ...
                dmd_summary.relres, dmd_summary.iter_count, dmd_summary.flag);
        end
        fprintf('Auto-DMD physical WaveHoltz-vs-Helmholtz error: %.3e\n', ...
            dmd_summary.wave_vs_helm_phys);
    end

    if cfg.show_plots
        make_comparison_plots(results);
    end
end

function cfg = apply_defaults(cfg)
    cfg = set_default(cfg, 'pc_lattice_a', 1.0);
    cfg = set_default(cfg, 'pc_num_cols', 9);
    cfg = set_default(cfg, 'pc_num_rows', 3);
    cfg = set_default(cfg, 'pc_points_per_a', 20);
    cfg = set_default(cfg, 'pc_radius', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'pc_rod_shape', 'circle');
    cfg = set_default(cfg, 'pc_square_halfwidth', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'pc_x_margin', 0.5 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'lambda_over_a', 2.0);
    cfg = set_default(cfg, 'omega', 2 * pi / (cfg.lambda_over_a * cfg.pc_lattice_a));
    cfg = set_default(cfg, 'c_pc', 1 / sqrt(11.4));
    cfg = set_default(cfg, 'npml_points_per_a', 2);
    cfg = set_default(cfg, 'CFL', 0.5);
    cfg = set_default(cfg, 'sigma_max', 80);
    cfg = set_default(cfg, 'source_type', 'point');
    cfg = set_default(cfg, 'point_source_profile', 'box');
    cfg = set_default(cfg, 'point_amplitude', 1.0);
    cfg = set_default(cfg, 'point_box_radius_x', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'point_box_radius_y', 0.25 * cfg.pc_lattice_a);
    cfg = set_default(cfg, 'wh_tol', 1e-6);
    cfg = set_default(cfg, 'wh_gmres_restart', 120);
    cfg = set_default(cfg, 'wh_gmres_maxit', 5);
    cfg = set_default(cfg, 'wh_gmres_reorth', true);
    cfg = set_default(cfg, 'wh_initial_guess', 'zero');
    cfg = set_default(cfg, 'wh_auto_refine_against_helm', false);
    cfg = set_default(cfg, 'wh_helm_align_tol', 1e-3);
    cfg = set_default(cfg, 'wh_auto_refine_max_rounds', 0);
    cfg = set_default(cfg, 'wh_auto_refine_maxit_factor', 2);
    cfg = set_default(cfg, 'wh_recovery_mode', 'best_helm');
    cfg = set_default(cfg, 'filter_shift', 0.25);
    cfg = set_default(cfg, 'filter_periods', 1);
    cfg = set_default(cfg, 'gmres_solver', 'custom_gmres');
    cfg = set_default(cfg, 'fixed_point_relax', 0.15);
    cfg = set_default(cfg, 'dmd_relax', 0.15);
    cfg = set_default(cfg, 'run_fixed_point', true);
    cfg = set_default(cfg, 'run_gmres', true);
    cfg = set_default(cfg, 'run_dmd', true);
    cfg = set_default(cfg, 'verbose_iter', false);
    cfg = set_default(cfg, 'show_plots', true);
    cfg = set_default(cfg, 'dmd_max_snaps', 40);
    cfg = set_default(cfg, 'dmd_epsSVD', 1e-10);
    cfg = set_default(cfg, 'dmd_svd_rel_tol', 1e-6);
    cfg = set_default(cfg, 'dmd_max_rank', 30);
    cfg = set_default(cfg, 'dmd_min_snaps_for_predict', 16);
    cfg = set_default(cfg, 'dmd_min_pred_interval', 20);
    cfg = set_default(cfg, 'dmd_cond_ratio_tol', 1e-3);
    cfg = set_default(cfg, 'dmd_rho_tol', 0.999);
    cfg = set_default(cfg, 'dmd_predict_burst', 1);
    cfg = set_default(cfg, 'dmd_accept_growth', 1.0);
    cfg = set_default(cfg, 'dmd_start_plain_res', inf);
    cfg = set_default(cfg, 'dmd_window_strategy', 'rolling');
end

function wh_cfg = make_waveguide_cfg(cfg)
    wh_cfg = struct();
    wh_cfg.pc_lattice_a = cfg.pc_lattice_a;
    wh_cfg.pc_num_cols = cfg.pc_num_cols;
    wh_cfg.pc_num_rows = cfg.pc_num_rows;
    wh_cfg.Lx = cfg.pc_num_cols * cfg.pc_lattice_a;
    wh_cfg.Ly = (2 * cfg.pc_num_rows + 1) * cfg.pc_lattice_a;
    wh_cfg.L = wh_cfg.Ly;
    wh_cfg.Nphys_x = round(wh_cfg.Lx / cfg.pc_lattice_a * cfg.pc_points_per_a);
    wh_cfg.Nphys_y = round(wh_cfg.Ly / cfg.pc_lattice_a * cfg.pc_points_per_a);
    wh_cfg.Nphys = wh_cfg.Nphys_y;
    wh_cfg.npml = cfg.npml_points_per_a * cfg.pc_points_per_a;
    wh_cfg.pc_radius = cfg.pc_radius;
    wh_cfg.pc_rod_shape = cfg.pc_rod_shape;
    wh_cfg.pc_square_halfwidth = cfg.pc_square_halfwidth;
    wh_cfg.pc_x_margin = cfg.pc_x_margin;
    wh_cfg.omega = cfg.omega;
    wh_cfg.CFL = cfg.CFL;
    wh_cfg.sigma_max = cfg.sigma_max;
    wh_cfg.c_right_values = cfg.c_pc;
    wh_cfg.material_model = 'photonic_crystal';
    wh_cfg.source_type = cfg.source_type;
    wh_cfg.point_source_profile = cfg.point_source_profile;
    wh_cfg.point_amplitude = cfg.point_amplitude;
    wh_cfg.point_box_radius_x = cfg.point_box_radius_x;
    wh_cfg.point_box_radius_y = cfg.point_box_radius_y;
    wh_cfg.source_col_phys = round((wh_cfg.Nphys_x + 1) / 2);
    wh_cfg.source_row_phys = round((wh_cfg.Nphys_y + 1) / 2);
    wh_cfg.compute_incident_reference = false;
    wh_cfg.wh_initial_guess = cfg.wh_initial_guess;
    wh_cfg.wh_tol = cfg.wh_tol;
    wh_cfg.wh_gmres_restart = cfg.wh_gmres_restart;
    wh_cfg.wh_gmres_maxit = cfg.wh_gmres_maxit;
    wh_cfg.wh_gmres_reorth = cfg.wh_gmres_reorth;
    wh_cfg.wh_auto_refine_against_helm = cfg.wh_auto_refine_against_helm;
    wh_cfg.wh_helm_align_tol = cfg.wh_helm_align_tol;
    wh_cfg.wh_auto_refine_max_rounds = cfg.wh_auto_refine_max_rounds;
    wh_cfg.wh_auto_refine_maxit_factor = cfg.wh_auto_refine_maxit_factor;
    wh_cfg.wh_recovery_mode = cfg.wh_recovery_mode;
    wh_cfg.filter_shift = cfg.filter_shift;
    wh_cfg.filter_periods = cfg.filter_periods;
    wh_cfg.fixed_point_relax = cfg.fixed_point_relax;
    wh_cfg.dmd_relax = cfg.dmd_relax;
    wh_cfg.dmd_max_snaps = cfg.dmd_max_snaps;
    wh_cfg.dmd_epsSVD = cfg.dmd_epsSVD;
    wh_cfg.dmd_svd_rel_tol = cfg.dmd_svd_rel_tol;
    wh_cfg.dmd_max_rank = cfg.dmd_max_rank;
    wh_cfg.dmd_min_snaps_for_predict = cfg.dmd_min_snaps_for_predict;
    wh_cfg.dmd_min_pred_interval = cfg.dmd_min_pred_interval;
    wh_cfg.dmd_cond_ratio_tol = cfg.dmd_cond_ratio_tol;
    wh_cfg.dmd_rho_tol = cfg.dmd_rho_tol;
    wh_cfg.dmd_predict_burst = cfg.dmd_predict_burst;
    wh_cfg.dmd_accept_growth = cfg.dmd_accept_growth;
    wh_cfg.dmd_start_plain_res = cfg.dmd_start_plain_res;
    wh_cfg.dmd_window_strategy = cfg.dmd_window_strategy;
end

function print_case_summary(s)
    fprintf('case %-24s solver=%s\n', s.label, s.solver);
    fprintf('  converged=%d, flag=%d, iter=%d, relres=%.3e\n', ...
        s.converged, s.flag, s.iter_count, s.relres);
    fprintf('  rel(WaveHoltz, Helm) physical = %.3e, recovery=%s residual=%.3e\n', ...
        s.wave_vs_helm_phys, s.recovery_method, s.recovery_helm_relres);
    fprintf('  solve_time=%.3fs, total_time=%.3fs\n', s.solve_time, s.total_time);
    if isfield(s, 'dmd_predictions')
        fprintf('  DMD predictions=%d, restarts=%d, rejected=%d, last_rank=%d\n', ...
            s.dmd_predictions, s.dmd_restarts, s.dmd_rejected, s.dmd_last_rank);
        if isfield(s, 'dmd_window_truncations')
            fprintf('  DMD window-truncations=%d\n', s.dmd_window_truncations);
        end
        fprintf('  DMD last rho=%.5f, last cond_ratio=%.3e\n', ...
            s.dmd_last_rho, s.dmd_last_cond_ratio);
    end
end

function make_comparison_plots(results)
    labels = fieldnames(results.cases);
    n = numel(labels);
    figure('Name', 'Photonic crystal waveguide: WaveHoltz solvers', 'Position', [80 80 420 * n 760]);
    tlo = tiledlayout(2, n, 'Padding', 'compact', 'TileSpacing', 'compact');
    for k = 1:n
        r = results.cases.(labels{k});
        prob = r.problem;
        field_phys = real(r.u_wave_phys(:, :, 1));
        nexttile;
        imagesc(prob.x_phys, prob.y_phys, field_phys.');
        set(gca, 'YDir', 'normal');
        axis equal tight;
        colorbar;
        set_robust_symmetric_color_limits(field_phys);
        title(strrep(labels{k}, '_', '\_'));
        xlabel('x');
        ylabel('y');
        hold on;
        contour(prob.x_phys, prob.y_phys, double(prob.chi_right_grid(prob.I, prob.J)).', [0.5, 0.5], ...
            'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.45);
        hold off;

        nexttile;
        resvec = r.info(1).resvec;
        relhist = abs(resvec(:)) / max(abs(resvec(1)), 1e-16);
        semilogy(relhist, 'LineWidth', 1.3);
        grid on;
        xlabel('iteration');
        ylabel('relative residual');
        title(sprintf('relres %.2e', r.info(1).relres));
    end
    title(tlo, 'Extracted photonic-crystal waveguide');
end

function set_robust_symmetric_color_limits(field_phys)
    vals = abs(real(field_phys(:)));
    vals = vals(isfinite(vals));
    if isempty(vals)
        return;
    end

    vals = sort(vals);
    idx = max(1, ceil(0.98 * numel(vals)));
    vmax = vals(idx);
    if vmax <= 0
        vmax = vals(end);
    end
    if vmax <= 0
        return;
    end

    caxis([-vmax, vmax]);
end

function cfg = set_default(cfg, field_name, value)
    if ~isfield(cfg, field_name) || isempty(cfg.(field_name))
        cfg.(field_name) = value;
    end
end
