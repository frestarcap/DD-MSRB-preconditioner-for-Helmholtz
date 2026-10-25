function results = run_helm_shifted_laplacian_completion_experiment(mode, user_cfg)
%RUN_HELM_SHIFTED_LAPLACIAN_COMPLETION_EXPERIMENT Solver-agnostic ROM test.
%
% This half-space experiment keeps the trajectory-aware ROM training setup from
% Experiment 1 and replaces WaveHoltz completion by shifted-Laplacian GMRES.

    if nargin < 1 || isempty(mode)
        mode = "paper";
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end
    mode = lower(string(mode));

    cfg = default_shifted_completion_cfg(mode);
    cfg = merge_structs(cfg, user_cfg);

    fprintf('=== Helm shifted-Laplacian completion experiment: %s ===\n', mode);
    results = helm_waveholtz_correction_rom(cfg);

    out_dir = helm_output_path('outputs', 'helm_shifted_laplacian_completion');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    tag = char(mode);
    if mode == "paper", tag = 'paper_tol1e12'; end
    archive_file = fullfile(out_dir, ['helm_shifted_laplacian_completion_' tag '.mat']);
    save(archive_file, ...
        'mode', 'cfg', 'results', '-v7.3');
    if mode == "paper" && cfg.do_plots && cfg.save_plots
        plot_helm_shifted_laplacian_completion_figure(archive_file,cfg.figure_subdir,0.8622);
    end
end

function cfg = default_shifted_completion_cfg(mode)
    cfg = struct();
    cfg.material_model = 'halfspace';
    cfg.figure_subdir = 'helm_shifted_laplacian_completion';
    cfg.benchmark_name = 'helm_shifted_laplacian_completion';
    cfg.rom_check_target = 'helm';
    cfg.reference_mode = 'helm_lift';
    cfg.online_completion_mode = 'shifted_laplacian';
    cfg.use_direct_rom_init = false;
    cfg.corr_basis_style = 'global';
    cfg.corr_num_layers = 3;
    cfg.corr_pod_singular_value_tols = [1e-5, 1e-5, 1e-5];
    cfg.run_wave_baselines = true;
    cfg.shifted_laplacian_beta = 0.2;
    cfg.shifted_laplacian_restart = 30;
    cfg.shifted_laplacian_maxit = 3;
    cfg.shifted_laplacian_tol = 1e-12;
    cfg.online_helm_accept_tol = 1e-12;
    cfg.helm_baseline_tol = 1e-12;
    cfg.shifted_laplacian_reorth = true;
    cfg.parallel_train = true;
    cfg.parallel_test = true;
    cfg.do_plots = true;
    cfg.save_plots = true;

    switch mode
        case "smoke"
            cfg.Nphys = 45;
            cfg.train_params = linspace(0.80, 0.95, 6);
            cfg.test_params = [0.8130, 0.9072];
            cfg.shifted_laplacian_tol = 1e-9;
            cfg.online_helm_accept_tol = 1e-9;
            cfg.helm_baseline_tol = 1e-9;
            cfg.shifted_laplacian_maxit = 8;
            cfg.helm_baseline_iters = 40;
            cfg.parallel_train = false;
            cfg.parallel_test = false;
            cfg.figure_subdir = 'helm_shifted_laplacian_completion_smoke';
            cfg.benchmark_name = 'helm_shifted_laplacian_completion_smoke';
        otherwise
            cfg.train_params = linspace(0.80, 0.95, 24);
            cfg.test_param_strategy = 'random_interval';
            cfg.test_param_rng_seed = 1;
            cfg.plot_test_index = [];
    end
end

function out = merge_structs(base, override)
    out = base;
    names = fieldnames(override);
    for k = 1:numel(names)
        out.(names{k}) = override.(names{k});
    end
end
