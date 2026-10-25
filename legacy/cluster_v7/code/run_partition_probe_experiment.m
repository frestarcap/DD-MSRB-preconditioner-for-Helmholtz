function results = run_partition_probe_experiment(mode, problems)
%RUN_PARTITION_PROBE_EXPERIMENT Compare candidate DD partitions on small cases.
%
% The probe keeps the correction-ROM and completion algorithms fixed and varies
% only the block partition used to store the layerwise correction bases.

    if nargin < 1 || isempty(mode)
        mode = "smoke";
    end
    mode = lower(string(mode));
    if mode ~= "smoke" && mode ~= "medium" && mode ~= "paper"
        error('mode must be smoke, medium, or paper.');
    end
    if nargin < 2 || isempty(problems)
        problems = "all";
    end
    problems = lower(string(problems));
    valid_problems = ["all", "halfspace", "pc"];
    if ~any(problems == valid_problems)
        error('problems must be all, halfspace, or pc.');
    end

    out_dir = helm_output_path('outputs', 'partition_probe');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    fprintf('=== Partition probe: %s (%s) ===\n', mode, problems);
    half = [];
    pc = [];
    tables = {};
    if problems == "all" || problems == "halfspace"
        half = run_halfspace_partition_probe(mode);
        tables{end + 1} = half.table; %#ok<AGROW>
    end
    if problems == "all" || problems == "pc"
        pc = run_pc_partition_probe(mode);
        tables{end + 1} = pc.table; %#ok<AGROW>
    end

    results = struct();
    results.mode = char(mode);
    results.problems = char(problems);
    results.halfspace = half;
    results.pc = pc;

    all_tbl = vertcat(tables{:});
    results.table = all_tbl;
    file_stem = ['partition_probe_' char(mode) '_' char(problems)];
    writetable(all_tbl, fullfile(out_dir, [file_stem '.csv']));
    save(fullfile(out_dir, [file_stem '.mat']), 'results', '-v7.3');
end

function out = run_halfspace_partition_probe(mode)
    base = struct();
    base.material_model = 'halfspace';
    base.rom_check_target = 'helm';
    base.reference_mode = 'helm_lift';
    base.online_completion_mode = 'waveholtz';
    base.online_wave_solver = 'fixedpoint';
    base.corr_num_layers = 3;
    base.corr_pod_singular_value_tols = [1e-4, 1e-3, 1e-2];
    base.run_nr_rom_comparison = false;
    base.do_plots = false;
    base.save_plots = false;
    base.save_benchmark = false;
    base.run_wave_baselines = true;
    base.parallel_train = false;
    base.parallel_test = false;

    if mode == "smoke"
        base.Nphys = 45;
        base.train_params = linspace(0.80, 0.95, 6);
        base.test_params = [0.8130, 0.9072];
        base.wh_max_iter = 80;
        base.fp_maxit = 40;
        base.wave_baseline_max_iter = 60;
    elseif mode == "medium"
        base.Nphys = 85;
        base.train_params = linspace(0.80, 0.95, 16);
        base.test_param_strategy = 'random_interval';
        base.test_param_rng_seed = 1;
        base.test_count = 5;
        base.wh_max_iter = 160;
        base.fp_maxit = 80;
        base.wave_baseline_max_iter = 120;
    else
        base.train_params = linspace(0.80, 0.95, 24);
        base.test_param_strategy = 'random_interval';
        base.test_param_rng_seed = 1;
    end

    variants = {
        make_variant('global', 'global', '', [])
        make_variant('radial-current', 'block3', 'radial', [])
        make_variant('left-right-pml', 'block3', 'halfspace_material', [])
        make_variant('source-interface-pml', 'block3', 'halfspace_source_interface', [])
    };
    out = run_variants('halfspace', variants, base, @helm_waveholtz_correction_rom);
end

function out = run_pc_partition_probe(mode)
    base = struct();
    base.material_model = 'photonic_crystal';
    base.rom_check_target = 'helm';
    base.reference_mode = 'helm_lift';
    base.online_completion_mode = 'waveholtz';
    base.online_wave_solver = 'gmres';
    base.corr_num_layers = 3;
    base.do_plots = false;
    base.save_plots = false;
    base.save_benchmark = false;
    base.run_wave_baselines = true;
    base.parallel_train = false;
    base.parallel_test = false;
    base.corr_pod_singular_value_tols = [1e-4, 1e-3, 1e-2];

    if mode == "smoke"
        base.pc_num_cols = 5;
        base.pc_num_rows = 1;
        base.pc_points_per_a = 8;
        base.npml = 8;
        base.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.98, 1.02, 4);
        base.wh_gmres_restart = 40;
        base.wh_gmres_maxit = 1;
        base.helm_baseline_iters = 20;
    elseif mode == "medium"
        base.pc_num_cols = 7;
        base.pc_num_rows = 2;
        base.pc_points_per_a = 10;
        base.npml = 10;
        base.c_pc_train_params = (1 / sqrt(11.4)) * linspace(0.98, 1.02, 6);
        base.test_param_strategy = 'random_interval';
        base.test_param_rng_seed = 1;
        base.test_count = 4;
        base.wh_gmres_restart = 60;
        base.wh_gmres_maxit = 2;
        base.helm_baseline_iters = 40;
    else
        base.test_param_strategy = 'random_interval';
        base.test_param_rng_seed = 1;
    end

    dd_default = struct();
    dd_default.corr_block3_rank_cap_near = Inf;
    dd_default.corr_block3_rank_cap_far = Inf;
    dd_default.corr_block3_rank_cap_pml = Inf;
    dd_default.corr_block3_tail_tol_near = [1, 1, 1];
    dd_default.corr_block3_tail_tol_far = [1, 1, 1];
    dd_default.corr_block3_tail_tol_pml = [1, 1, 1];
    dd_default.corr_block3_global_tail_tols = [1e-4, 1e-3, 1e-2];
    dd_default.corr_block3_global_tail_strategy = 'dof_weighted';

    lr_tighter = dd_default;

    narrow = dd_default;
    narrow.block3_waveguide_halfwidth = 0.75;

    four_block = dd_default;
    four_block.corr_blockn_tail_tols = ones(3, 4);
    four_block.corr_blockn_rank_caps = Inf(3, 4);

    variants = {
        make_variant('global', 'global', '', [])
        make_variant('channel-current', 'block3', 'pc_waveguide', dd_default)
        make_variant('narrow-channel', 'block3', 'pc_waveguide', narrow)
        make_variant('physical-lr-pml-tb-pml', 'block3', 'pc_physical_directional_pml', dd_default)
        make_variant('channel-cladding-lr-pml', 'block3', 'pc_channel_lr_pml', lr_tighter)
        make_variant('core-cladding-lr-tb-pml', 'block3', 'pc_core_cladding_lr_tb_pml', four_block)
    };
    out = run_variants('pc', variants, base, @photonic_crystal_waveguide_correction_rom);
end

function variant = make_variant(name, basis_style, partition_mode, extra_cfg)
    variant = struct();
    variant.name = name;
    variant.basis_style = basis_style;
    variant.partition_mode = partition_mode;
    if isempty(extra_cfg)
        extra_cfg = struct();
    end
    variant.extra_cfg = extra_cfg;
end

function out = run_variants(problem_name, variants, base_cfg, runner)
    rows = repmat(empty_row(), numel(variants), 1);
    raw = cell(numel(variants), 1);

    for i = 1:numel(variants)
        v = variants{i};
        cfg = base_cfg;
        cfg.corr_basis_style = v.basis_style;
        if strcmp(v.basis_style, 'block3')
            cfg.block3_partition_mode = v.partition_mode;
            cfg.corr_block3_global_tail_strategy = 'dof_weighted';
            if ~isfield(cfg, 'corr_block3_global_tail_tols') || isempty(cfg.corr_block3_global_tail_tols)
                cfg.corr_block3_global_tail_tols = cfg.corr_pod_singular_value_tols;
            end
        end
        cfg = merge_structs_local(cfg, v.extra_cfg);

        fprintf('\n--- %s partition variant: %s ---\n', problem_name, v.name);
        raw{i} = runner(cfg);
        rows(i) = summarize_run(problem_name, v.name, raw{i});
    end

    out = struct();
    out.raw = raw;
    out.table = struct2table(rows);
end

function row = summarize_run(problem_name, variant_name, r)
    row = empty_row();
    row.problem = string(problem_name);
    row.variant = string(variant_name);
    row.mean_precorr_relres = scalar_stat(r.summary.online_precorr_relres, 2);
    row.max_precorr_relres = scalar_stat(r.summary.online_precorr_relres, 3);
    row.mean_total_time = scalar_stat(r.summary.online_total_time, 2);
    row.mean_completion_iter = scalar_stat(r.summary.online_fp_iter, 2);
    row.mean_final_relres = scalar_stat(r.summary.online_fp_relres, 2);
    row.mean_plain_time = scalar_stat(r.summary.plain_fp_time, 2);

    if isfield(r.summary, 'corr_storage_block') && any(isfinite(r.summary.corr_storage_block(:)))
        row.total_basis_entries = sum(r.summary.corr_storage_block(:), 'omitnan');
        row.layer_ranks = string(mat2str(r.summary.corr_block_ranks));
        row.layer_dofs = string(mat2str(r.summary.corr_block_dofs));
    else
        dense_storage = r.summary.corr_storage_dense(:);
        if ~any(isfinite(dense_storage)) && isfield(r, 'problem') && isfield(r.problem, 'Ntot')
            dense_storage = r.problem.Ntot * r.summary.corr_ranks(:);
        end
        row.total_basis_entries = sum(dense_storage, 'omitnan');
        row.layer_ranks = string(mat2str(r.summary.corr_ranks(:).'));
        row.layer_dofs = "";
    end
end

function row = empty_row()
    row = struct();
    row.problem = "";
    row.variant = "";
    row.total_basis_entries = NaN;
    row.mean_precorr_relres = NaN;
    row.max_precorr_relres = NaN;
    row.mean_total_time = NaN;
    row.mean_completion_iter = NaN;
    row.mean_final_relres = NaN;
    row.mean_plain_time = NaN;
    row.layer_ranks = "";
    row.layer_dofs = "";
end

function val = scalar_stat(x, idx)
    val = NaN;
    if isnumeric(x) && numel(x) >= idx
        val = x(idx);
    end
end

function a = merge_structs_local(a, b)
    if isempty(b)
        return;
    end
    names = fieldnames(b);
    for i = 1:numel(names)
        a.(names{i}) = b.(names{i});
    end
end
