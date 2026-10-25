function spec = helmholtz_3d_40pct_config(candidate)
%HELMHOLTZ_3D_40PCT_CONFIG Historical 3D memory-result settings.
% This reproduces the archived ten-cycle cohort, including achieved snapshot
% residuals above the requested tolerance. New verified runs use
% run_jsc_lens_tradeoff (twenty cycles and explicit rejection).

    if nargin < 1 || isempty(candidate)
        candidate = 'lens';
    end
    candidate = lower(strtrim(char(string(candidate))));

    common = struct();
    common.train_styles = {'global', 'dd'};
    common.dd_partition_mode = 'signed_depth_pml';
    common.dd_rank_strategy = 'marginal_storage';
    common.pml_depth_fraction = 0.5;
    common.physical_partition_mode = 'profile_bands';
    common.profile_partition_thresholds = [1e-3, 1e-2, 1e-1, 0.5];
    common.snapshot_solver = 'shifted_mg';
    common.snapshot_tol = 1e-7;
    common.snapshot_restart = 40;
    common.snapshot_maxit = 10;
    common.require_snapshot_tolerance = false;
    common.num_layers = 2;
    common.shift_beta = 1.2;
    common.gmres_tol = 1e-6;
    common.gmres_restart = 40;
    common.gmres_maxit = 20;
    common.gmres_reorthogonalize = true;
    common.direct_max_unknowns = 0;
    common.run_direct = false;
    common.save_models = true;
    common.parallel_train = true;
    common.parallel_workers = 4;
    common.mg_max_levels = 6;
    common.mg_coarse_max_dofs = 3000;
    common.mg_pre_sweeps = 2;
    common.mg_post_sweeps = 2;
    common.mg_jacobi_weight = 0.6;
    common.mg_cycles_per_apply = 1;
    common.train_count = 6;

    problem = struct();
    problem.background_speed = 1.0;
    problem.sigma_max = 80;
    problem.use_pml = true;
    problem.source_model = 'gaussian';
    problem.active_profile_cutoff = 0.08;

    expected = struct();
    switch candidate
        case {'lens', 'gaussian_lens'}
            candidate = 'lens';
            problem_name = 'gaussian_lens';
            problem.material_model = problem_name;
            problem.domain_lengths = [1, 1, 1];
            problem.gaussian_center = [0.55, 0.5, 0.5];
            problem.gaussian_width = [0.15, 0.18, 0.18];
            problem.source_location = [0.16, 0.5, 0.5];
            problem.parameter_range = [0.15, 0.55];
            problem.test_params = [0.283, 0.417];
            problem.Nphys = [63, 63, 63];
            problem.npml = [12, 12, 12];
            problem.omega = 32;
            common.physical_axial_sections = 3;
            common.layer_tail_tolerances = [0.012, 0.016];
            common.output_tag = 'lens_dd_memory_saving_46pct_reproduction';
            expected.global_basis_mib = 70.34;
            expected.dd_basis_mib = 37.73;
            expected.saving_fraction = 0.464;
            expected.shifted_snapshot_rom_offline_s = 123.5107;
            expected.non_snapshot_rom_overhead_s = 24.56;
        case {'waveguide', 'gaussian_waveguide'}
            candidate = 'waveguide';
            problem_name = 'gaussian_waveguide';
            problem.material_model = problem_name;
            problem.domain_lengths = [3, 1, 1];
            problem.waveguide_axis = 'x';
            problem.waveguide_center = [1.5, 0.5, 0.5];
            problem.waveguide_width = [0.14, 0.14];
            problem.source_location = [0.2, 0.5, 0.5];
            problem.parameter_range = [0.2, 0.6];
            problem.test_params = [0.317, 0.483];
            problem.Nphys = [143, 47, 47];
            problem.npml = [9, 9, 9];
            problem.omega = 24;
            common.physical_axial_sections = 5;
            common.layer_tail_tolerances = [0.0056, 0.0278];
            common.output_tag = 'waveguide_dd_memory_saving_41pct_reproduction';
            expected.global_basis_mib = 93.41;
            expected.dd_basis_mib = 55.14;
            expected.saving_fraction = 0.410;
            expected.shifted_snapshot_rom_offline_s = 243.2041;
            expected.non_snapshot_rom_overhead_s = 10.30;
        otherwise
            error('candidate must be lens or waveguide.');
    end

    problem.train_params = linspace(problem.parameter_range(1), ...
        problem.parameter_range(2), common.train_count);
    common.problem_names = {problem_name};
    common.([problem_name '_cfg']) = problem;

    spec = struct('name', candidate, 'mode', 'large', ...
        'problem_name', problem_name, 'problem_cfg', problem, ...
        'driver_cfg', common, 'expected', expected);
end
