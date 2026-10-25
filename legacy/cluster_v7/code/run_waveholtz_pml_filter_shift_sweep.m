function summary = run_waveholtz_pml_filter_shift_sweep(nt_values, cfg)
%RUN_WAVEHOLTZ_PML_FILTER_SHIFT_SWEEP Compare shifts while rebuilding G_tau.
    if nargin < 1 || isempty(nt_values), nt_values = [11 16 24 32 48]; end
    if nargin < 2, cfg = struct(); end
    ncase = numel(nt_values);
    rows = repmat(struct(), ncase, 1);
    for k = 1:ncase
        run_cfg = cfg;
        run_cfg.nt_override = nt_values(k);
        run_cfg.make_plot = false;
        run_cfg.save_outputs = false;
        fprintf('\n========== N_t = %d ==========\n', nt_values(k));
        r = run_waveholtz_pml_propagator_spectrum_experiment(run_cfg);
        rows(k).nt = r.nt;
        rows(k).dt = r.dt;
        rows(k).rho_G = r.rho_G;
        rows(k).standard_shift = r.standard.shift;
        rows(k).corrected_shift = r.corrected.shift;
        rows(k).standard_circle_max = r.standard.max_filter_on_unit_circle;
        rows(k).corrected_circle_max = r.corrected.max_filter_on_unit_circle;
        rows(k).standard_rho_S = r.standard.rho_S;
        rows(k).corrected_rho_S = r.corrected.rho_S;
        rows(k).standard_norm_S = r.standard.norm_S;
        rows(k).corrected_norm_S = r.corrected.norm_S;
        rows(k).standard_target_derivative = r.standard.target_modulus_phase_derivative;
        rows(k).corrected_target_derivative = r.corrected.target_modulus_phase_derivative;
    end
    summary = struct2table(rows);
    outdir = helm_output_path('outputs', 'pml_propagator_spectrum');
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    writetable(summary, fullfile(outdir, 'filter_shift_nt_sweep.csv'));
    save(fullfile(outdir, 'filter_shift_nt_sweep.mat'), 'summary', 'cfg', 'nt_values');
    disp(summary);
end
