function results = run_helmholtz_3d_candidate_experiments(mode, user_cfg)
%RUN_HELMHOLTZ_3D_CANDIDATE_EXPERIMENTS Evaluate two hard 3D candidates.
%
% The candidates are a localized Gaussian lens and a long Gaussian
% waveguide. Each run compares sparse direct solution when feasible,
% shifted-Laplacian geometric-multigrid GMRES, and global/DD layered
% trajectory-aware reduced inverse-action preconditioning followed by the
% same shifted-Laplacian completion.

    if nargin < 1 || isempty(mode)
        mode = 'smoke';
    end
    if nargin < 2 || isempty(user_cfg)
        user_cfg = struct();
    end
    mode = lower(char(string(mode)));
    cfg = merge_structs(default_cfg(mode), user_cfg);
    cfg.mode = mode;
    cfg.dd_partition_mode = lower(char(string(cfg.dd_partition_mode)));
    cfg.dd_rank_strategy = lower(char(string(cfg.dd_rank_strategy)));
    cfg.physical_partition_mode = lower(char(string(cfg.physical_partition_mode)));
    validate_cfg(cfg);

    out_dir = helm_output_path('outputs', 'helmholtz_3d_candidates', mode);
    if ~isempty(cfg.output_dir), out_dir = cfg.output_dir; end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    cfg.output_dir=out_dir;

    problem_names = cellstr(string(cfg.problem_names));
    candidate_results = cell(numel(problem_names), 1);
    summary_tables = cell(numel(problem_names), 1);
    for k = 1:numel(problem_names)
        name = lower(strtrim(problem_names{k}));
        candidate_cfg = candidate_problem_cfg(name, cfg);
        fprintf('\n============================================================\n');
        fprintf('3D candidate %d/%d: %s (%s)\n', k, numel(problem_names), name, mode);
        fprintf('============================================================\n');
        candidate_results{k} = run_candidate(name, candidate_cfg, cfg, out_dir);
        summary_tables{k} = candidate_results{k}.summary_table;

        checkpoint = fullfile(out_dir, sprintf('%s_%s_checkpoint.mat', ...
            name, cfg.output_tag));
        candidate_result = candidate_results{k};
        if cfg.save_candidate_checkpoints
            save(checkpoint, 'candidate_result', '-v7.3');
        end
    end

    results = struct();
    results.mode = mode;
    results.cfg = cfg;
    results.candidates = candidate_results;
    results.summary_table = vertcat(summary_tables{:});
    results.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));
    results.matlab_version = version;
    results.artifact_manifest = struct( ...
        'schema_version', 1, ...
        'contains_full_models', logical(cfg.save_models && cfg.embed_full_models), ...
        'separate_training_checkpoint', logical(cfg.save_models), ...
        'contains_solution_fields', true, ...
        'contains_plot_data', true, ...
        'model_contents', {{'layer bases', 'affine reduced operators', ...
            'local singular spectra', 'snapshot solve diagnostics'}});

    mat_file = fullfile(out_dir, sprintf('helmholtz_3d_candidates_%s.mat', cfg.output_tag));
    csv_file = fullfile(out_dir, sprintf('helmholtz_3d_candidates_%s.csv', cfg.output_tag));
    paper_plot_data = make_paper_plot_data(results);
    save(mat_file, 'results', 'paper_plot_data', '-v7.3');
    writetable(results.summary_table, csv_file);
    fprintf('\nSaved %s\nSaved %s\n', mat_file, csv_file);
    disp(results.summary_table);
    if cfg.make_plots
        for k = 1:numel(candidate_results)
            one = candidate_results{k};
            plot_candidate_results(one.name, one.models, one.tests, out_dir, cfg.output_tag);
        end
    end
end

function cfg = default_cfg(mode)
    cfg = struct();
    cfg.problem_names = {'gaussian_lens', 'gaussian_waveguide'};
    cfg.train_styles = {'global', 'dd'};
    cfg.dd_partition_mode = 'directional_pml';
    cfg.dd_rank_strategy = 'marginal_storage';
    cfg.pml_depth_fraction = 0.5;
    cfg.physical_partition_mode = 'active_far';
    cfg.profile_partition_thresholds = [0.005, 0.08, 0.4];
    cfg.physical_axial_sections = 1;
    cfg.snapshot_solver = 'direct';
    cfg.snapshot_tol = 1e-9;
    cfg.snapshot_restart = 30;
    cfg.snapshot_maxit = 15;
    cfg.require_snapshot_tolerance = true;
    cfg.layer_tail_tolerances = [1e-4, 1e-3, 1e-2];
    cfg.num_layers = 3;
    cfg.shift_beta = 0.4;
    cfg.gmres_tol = 1e-6;
    cfg.gmres_restart = 30;
    cfg.gmres_maxit = 12;
    cfg.gmres_reorthogonalize = true;
    cfg.direct_max_unknowns = 150000;
    cfg.run_direct = true;
    cfg.save_models = true;
    cfg.embed_full_models = true;
    cfg.save_candidate_checkpoints = true;
    cfg.parallel_train = false;
    cfg.parallel_workers = 4;
    cfg.worker_threads = 1;
    cfg.hpc_mode = false;
    cfg.make_plots = true;
    cfg.output_dir = '';
    cfg.fast_reduced_assembly = false;
    cfg.reuse_first_layer = false;
    cfg.gmres_orthogonalization = 'mgs';
    cfg.store_preconditioned_basis = true;
    cfg.checkpoint_training = false;
    cfg.cache_snapshots = false;
    cfg.checkpoint_stop_after_layer = 0; % Fault-injection test only.
    cfg.output_tag = mode;
    cfg.mg_max_levels = 6;
    cfg.mg_coarse_max_dofs = 1500;
    cfg.mg_pre_sweeps = 2;
    cfg.mg_post_sweeps = 2;
    cfg.mg_jacobi_weight = 0.6;
    cfg.mg_cycles_per_apply = 1;

    switch mode
        case 'smoke'
            cfg.train_count = 4;
            cfg.num_layers = 2;
            cfg.layer_tail_tolerances = [1e-5, 1e-4];
            cfg.gmres_maxit = 6;
        case 'pilot'
            cfg.train_count = 8;
        case 'large'
            cfg.train_count = 12;
            cfg.snapshot_solver = 'shifted_mg';
            cfg.direct_max_unknowns = 0;
            cfg.run_direct = false;
            cfg.gmres_restart = 40;
            cfg.gmres_maxit = 20;
            cfg.shift_beta = 1.2;
            cfg.snapshot_tol = 1e-8;
            cfg.snapshot_restart = 40;
            cfg.snapshot_maxit = 12;
            cfg.mg_coarse_max_dofs = 3000;
            cfg.parallel_train = true;
        otherwise
            error('mode must be smoke, pilot, or large.');
    end
end

function validate_cfg(cfg)
    if cfg.num_layers < 1 || cfg.num_layers ~= round(cfg.num_layers)
        error('num_layers must be a positive integer.');
    end
    if numel(cfg.layer_tail_tolerances) < cfg.num_layers
        error('layer_tail_tolerances must contain at least num_layers entries.');
    end
    if cfg.train_count < 2 || cfg.train_count ~= round(cfg.train_count)
        error('train_count must be an integer of at least two.');
    end
    styles = cellstr(string(cfg.train_styles));
    if any(~ismember(styles, {'global', 'dd'}))
        error('train_styles may contain only global and dd.');
    end
    partition_modes = {'active_far_pml', 'directional_pml', ...
        'signed_directional_pml', 'directional_depth_pml', ...
        'signed_depth_pml', 'topological_pml'};
    if ~ismember(cfg.dd_partition_mode, partition_modes)
        error('Unsupported dd_partition_mode "%s".', cfg.dd_partition_mode);
    end
    if ~ismember(cfg.dd_rank_strategy, ...
            {'dof_weighted', 'marginal_storage', 'minimum_storage'})
        error(['dd_rank_strategy must be dof_weighted, marginal_storage, ' ...
            'or minimum_storage.']);
    end
    if ~(isscalar(cfg.pml_depth_fraction) && isfinite(cfg.pml_depth_fraction) && ...
            cfg.pml_depth_fraction > 0 && cfg.pml_depth_fraction < 1)
        error('pml_depth_fraction must be strictly between zero and one.');
    end
    if ~ismember(cfg.physical_partition_mode, {'active_far', 'profile_bands'})
        error('physical_partition_mode must be active_far or profile_bands.');
    end
    thresholds = cfg.profile_partition_thresholds(:);
    if isempty(thresholds) || any(~isfinite(thresholds)) || ...
            any(thresholds <= 0) || any(thresholds >= 1) || ...
            any(diff(thresholds) <= 0)
        error('profile_partition_thresholds must be strictly increasing in (0,1).');
    end
    if ~(isscalar(cfg.physical_axial_sections) && ...
            cfg.physical_axial_sections >= 1 && ...
            cfg.physical_axial_sections == round(cfg.physical_axial_sections))
        error('physical_axial_sections must be a positive integer.');
    end
end

function problem_cfg = candidate_problem_cfg(name, cfg)
    problem_cfg = struct();
    problem_cfg.background_speed = 1.0;
    problem_cfg.sigma_max = 80;
    problem_cfg.use_pml = true;
    problem_cfg.source_model = 'gaussian';
    problem_cfg.active_profile_cutoff = 0.08;

    switch name
        case 'affine_halfspace'
            problem_cfg.material_model = 'affine_halfspace';
            problem_cfg.domain_lengths = [1, 1, 1];
            problem_cfg.interface_x = .5;
            problem_cfg.source_location = [.2, .5, .5];
            problem_cfg.parameter_range = [.2, .6];
            problem_cfg.test_params = [.317, .483];
            problem_cfg.Nphys = [47, 47, 47];
            problem_cfg.npml = [9, 9, 9];
            problem_cfg.omega = 24;
            if strcmp(cfg.mode,'smoke')
                problem_cfg.Nphys=[11,11,11]; problem_cfg.npml=[3,3,3];
                problem_cfg.omega=6;
            end
        case 'gaussian_lens'
            problem_cfg.material_model = 'gaussian_lens';
            problem_cfg.domain_lengths = [1, 1, 1];
            problem_cfg.gaussian_center = [0.55, 0.5, 0.5];
            problem_cfg.gaussian_width = [0.15, 0.18, 0.18];
            problem_cfg.source_location = [0.16, 0.5, 0.5];
            problem_cfg.parameter_range = [0.15, 0.55];
            problem_cfg.test_params = [0.283, 0.417];
            switch cfg.mode
                case 'smoke'
                    problem_cfg.Nphys = [15, 15, 15];
                    problem_cfg.npml = [3, 3, 3];
                    problem_cfg.omega = 8;
                case 'pilot'
                    problem_cfg.Nphys = [23, 23, 23];
                    problem_cfg.npml = [4, 4, 4];
                    problem_cfg.omega = 12;
                case 'large'
                    problem_cfg.Nphys = [63, 63, 63];
                    problem_cfg.npml = [12, 12, 12];
                    problem_cfg.omega = 32;
            end

        case 'gaussian_waveguide'
            problem_cfg.material_model = 'gaussian_waveguide';
            problem_cfg.domain_lengths = [3, 1, 1];
            problem_cfg.waveguide_axis = 'x';
            problem_cfg.waveguide_center = [1.5, 0.5, 0.5];
            problem_cfg.waveguide_width = [0.14, 0.14];
            problem_cfg.source_location = [0.2, 0.5, 0.5];
            problem_cfg.parameter_range = [0.2, 0.6];
            problem_cfg.test_params = [0.317, 0.483];
            switch cfg.mode
                case 'smoke'
                    problem_cfg.Nphys = [35, 11, 11];
                    problem_cfg.npml = [3, 3, 3];
                    problem_cfg.omega = 6;
                case 'pilot'
                    problem_cfg.Nphys = [71, 23, 23];
                    problem_cfg.npml = [5, 5, 5];
                    problem_cfg.omega = 12;
                case 'large'
                    problem_cfg.Nphys = [143, 47, 47];
                    problem_cfg.npml = [9, 9, 9];
                    problem_cfg.omega = 24;
            end
        otherwise
            error('Unknown 3D candidate "%s".', name);
    end

    problem_cfg.train_params = linspace(problem_cfg.parameter_range(1), ...
        problem_cfg.parameter_range(2), cfg.train_count);
    override_name = matlab.lang.makeValidName([name '_cfg']);
    if isfield(cfg, override_name) && ~isempty(cfg.(override_name))
        problem_cfg = merge_structs(problem_cfg, cfg.(override_name));
        if ~isfield(cfg.(override_name), 'train_params')
            problem_cfg.train_params = linspace(problem_cfg.parameter_range(1), ...
                problem_cfg.parameter_range(2), cfg.train_count);
        end
    end
end

function result = run_candidate(name, problem_cfg, cfg, out_dir)
    setup_timer = tic;
    problem = helmholtz_3d_problem_interface(problem_cfg);
    setup_time = toc(setup_timer);
    if ~problem.affine.is_affine
        error('The 3D candidate driver requires an affine squared-slowness model.');
    end

    rhs = problem.rhs(problem_cfg.train_params(1));
    rhs = rhs / max(norm(rhs), eps);
    partition = make_rom_partition(problem, mean(problem_cfg.parameter_range), cfg);
    block_counts = [partition.blocks.count];
    min_speed = 1 / sqrt(1 / problem_cfg.background_speed^2 + max(problem_cfg.parameter_range));
    ppw = 2 * pi * min_speed / (problem_cfg.omega * max(problem.spacing));
    fprintf('grid=%s, N=%d, nnz(K)=%d, omega=%.3f, min PPW=%.2f\n', ...
        mat2str(problem.n), problem.N, nnz(problem.K_pml), problem_cfg.omega, ppw);
    fprintf('DD partition %s\n', cfg.dd_partition_mode);
    for block_index = 1:numel(partition.blocks)
        fprintf('  %s: %d unknowns\n', partition.blocks(block_index).name, ...
            partition.blocks(block_index).count);
    end

    styles = cellstr(string(cfg.train_styles));
    models = struct();
    shared_first = [];
    worker_problem = problem;
    if cfg.hpc_mode && cfg.parallel_train
        helm_3d_parallel_pool(cfg.parallel_workers, cfg.worker_threads);
        % Send only the operator data, without geometry callback closures.
        slim = struct('N', problem.N, 'n', problem.n, 'cfg',problem.cfg, ...
            'K_pml', problem.K_pml, 'affine', problem.affine);
        worker_problem = parallel.pool.Constant(slim);
        worker_cleanup = onCleanup(@() delete(worker_problem)); %#ok<NASGU>
    end
    frozen = isfield(cfg,'frozen_training_file') && ~isempty(cfg.frozen_training_file);
    if frozen
        saved=load(cfg.frozen_training_file,'models','partition','resolved_problem_cfg','rhs','cfg');
        current_problem=problem.cfg; saved_problem=saved.resolved_problem_cfg;
        current_problem.test_params=saved_problem.test_params;
        assert(isequaln(current_problem,saved_problem),'Frozen model geometry/training mismatch.');
        assert(isequaln(partition,saved.partition) && isequal(rhs,saved.rhs));
        for field={'layer_tail_tolerances','dd_rank_strategy','num_layers'}
            assert(isequaln(cfg.(field{1}),saved.cfg.(field{1})),'Frozen model policy mismatch.');
        end
        models=saved.models;
    end
    for k = 1:numel(styles)
        style = styles{k};
        if ~frozen
            fprintf('\n--- Training %s trajectory-aware ROM ---\n', upper(style));
            [models.(style), shared_first] = train_trajectory_rom(problem, problem_cfg.train_params, ...
                rhs, partition, cfg, style, worker_problem, shared_first);
        else
            fprintf('\n--- Reusing frozen %s model; no retraining ---\n',upper(style));
        end
        fprintf('%s offline %.2fs, layer ranks=%s, basis storage=%.2f MiB\n', ...
            upper(style), models.(style).offline_time, ...
            mat2str(models.(style).layer_ranks), ...
            models.(style).basis_bytes / 2^20);
    end
    clear shared_first
    if cfg.hpc_mode && cfg.parallel_train
        clear worker_cleanup worker_problem
        delete(gcp('nocreate'));
    end

    for k=1:numel(styles)
        style=styles{k}; index_bytes=0;
        if strcmp(style,'dd'), index_bytes=8*sum(block_counts); end
        models.(style).partition_bytes=index_bytes;
        models.(style).persistent_bytes=models.(style).basis_bytes + ...
            16*models.(style).reduced_entries + index_bytes;
    end

    geometry = struct('coordinates', {problem.coordinates}, ...
        'grid_size', problem.n, 'physical_grid_size', problem.nphys, ...
        'pml_cells', problem.npml, 'spacing', problem.spacing);
    if cfg.save_models
        checkpoint = fullfile(out_dir, sprintf('%s_%s_training.mat', ...
            name, cfg.output_tag));
        resolved_problem_cfg = problem.cfg;
        save(checkpoint, 'models', 'partition', 'geometry', 'rhs', ...
            'resolved_problem_cfg', 'cfg', '-v7.3');
        fprintf('Saved trained bases before online tests: %s\n', checkpoint);
    end

    tests = repmat(empty_test_result(), numel(problem_cfg.test_params), 1);
    for k = 1:numel(problem_cfg.test_params)
        mu = problem_cfg.test_params(k);
        fprintf('\n--- Online test mu=%.6f ---\n', mu);
        tests(k) = run_online_test(problem, rhs, mu, models, cfg);
    end

    summary_table = make_summary_table(name, problem, problem_cfg, ppw, ...
        setup_time, models, tests);

    result = struct();
    result.name = name;
    result.problem_cfg = problem.cfg;
    result.geometry = geometry;
    result.partition = partition;
    result.rhs = rhs;
    result.problem_summary = struct('N', problem.N, 'grid_size', problem.n, ...
        'nnz_K', nnz(problem.K_pml), 'spacing', problem.spacing, ...
        'ppw_minimum', ppw, 'setup_time', setup_time, ...
        'partition_names', {partition.block_names}, ...
        'partition_counts', block_counts);
    if cfg.save_models && cfg.embed_full_models
        result.models = models;
    else
        result.models = summarize_models(models);
    end
    result.tests = tests;
    result.summary_table = summary_table;
end

function partition = make_rom_partition(problem, mu, cfg)
    raw = problem.partition(mu);
    blocks = make_physical_blocks(problem, raw, cfg);

    pml_idx = find(raw.labels(:) == 3);
    switch cfg.dd_partition_mode
        case 'active_far_pml'
            blocks(end + 1) = make_partition_block('pml', 3, pml_idx);
        otherwise
            pml_blocks = make_refined_pml_blocks( ...
                problem, pml_idx, cfg.dd_partition_mode, cfg.pml_depth_fraction);
            blocks = [blocks, pml_blocks];
    end
    blocks = blocks([blocks.count] > 0);
    covered = sort(vertcat(blocks.idx));
    if ~isequal(covered, (1:problem.N).')
        error('The DD partition does not cover every unknown exactly once.');
    end
    partition = struct('n', problem.N, 'blocks', blocks, ...
        'block_names', {{blocks.name}}, ...
        'mode', cfg.dd_partition_mode, ...
        'physical_mode', cfg.physical_partition_mode);
end

function blocks = make_physical_blocks(problem, raw, cfg)
    switch cfg.physical_partition_mode
        case 'active_far'
            blocks = struct('name', {}, 'label', {}, 'idx', {}, 'count', {});
            blocks(end + 1) = make_partition_block( ...
                'active', 2, find(raw.labels(:) == 2));
            blocks(end + 1) = make_partition_block( ...
                'far', 1, find(raw.labels(:) == 1));
        case 'profile_bands'
            if ~isfield(problem, 'affine') || ...
                    ~isfield(problem.affine, 'mass_parameter_diagonal')
                error('profile_bands requires affine material-profile metadata.');
            end
            profile = abs(problem.affine.mass_parameter_diagonal(:)) / ...
                (problem.cfg.omega ^ 2);
            physical_idx = find(raw.labels(:) ~= 3);
            edges = [0; cfg.profile_partition_thresholds(:); inf];
            blocks = struct('name', {}, 'label', {}, 'idx', {}, 'count', {});
            for band = 1:numel(edges) - 1
                in_band = profile(physical_idx) >= edges(band) & ...
                    profile(physical_idx) < edges(band + 1);
                band_idx = physical_idx(in_band);
                blocks = append_axial_sections(blocks, problem, band_idx, ...
                    sprintf('profile_band_%d', band), 10 * band, ...
                    cfg.physical_axial_sections);
            end
    end
    blocks = blocks([blocks.count] > 0);
end

function blocks = append_axial_sections( ...
        blocks, problem, idx, base_name, base_label, section_count)
    if section_count == 1 || isempty(idx)
        blocks(end + 1) = make_partition_block(base_name, base_label, idx);
        return;
    end
    axis_dim = find(strcmp({'x', 'y', 'z'}, problem.cfg.waveguide_axis), 1);
    if isempty(axis_dim)
        axis_dim = 1;
    end
    [ix, iy, iz] = ind2sub(problem.n, idx);
    coords = {ix, iy, iz};
    physical_coordinate = coords{axis_dim} - problem.npml(axis_dim);
    section = min(floor((physical_coordinate - 1) * section_count / ...
        problem.nphys(axis_dim)) + 1, section_count);
    for k = 1:section_count
        blocks(end + 1) = make_partition_block( ...
            sprintf('%s_s%d', base_name, k), base_label + k, ...
            idx(section == k)); %#ok<AGROW>
    end
end

function blocks = make_refined_pml_blocks(problem, pml_idx, mode, depth_fraction)
    [ix, iy, iz] = ind2sub(problem.n, pml_idx);
    coords = {ix, iy, iz};
    axis_names = {'x', 'y', 'z'};
    axis_mask = false(numel(pml_idx), 3);
    low_mask = false(numel(pml_idx), 3);
    high_mask = false(numel(pml_idx), 3);
    depth = zeros(numel(pml_idx), 3);
    for d = 1:3
        low_mask(:, d) = coords{d} <= problem.npml(d);
        high_mask(:, d) = coords{d} > problem.n(d) - problem.npml(d);
        axis_mask(:, d) = low_mask(:, d) | high_mask(:, d);
        depth(low_mask(:, d), d) = problem.npml(d) - coords{d}(low_mask(:, d)) + 1;
        depth(high_mask(:, d), d) = ...
            coords{d}(high_mask(:, d)) - (problem.n(d) - problem.npml(d));
    end

    groups = struct('name', {}, 'idx', {});
    switch mode
        case {'directional_pml', 'signed_directional_pml', ...
                'directional_depth_pml', 'signed_depth_pml'}
            assigned = false(numel(pml_idx), 1);
            directional_mask = false(numel(pml_idx), 3);
            for d = 1:3
                directional_mask(:, d) = axis_mask(:, d) & ~assigned;
                assigned = assigned | directional_mask(:, d);
            end
            for d = 1:3
                mask = directional_mask(:, d);
                switch mode
                    case 'directional_pml'
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d}], pml_idx(mask)); %#ok<AGROW>
                    case 'signed_directional_pml'
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_low'], ...
                            pml_idx(mask & low_mask(:, d))); %#ok<AGROW>
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_high'], ...
                            pml_idx(mask & high_mask(:, d))); %#ok<AGROW>
                    case 'directional_depth_pml'
                        inner_cells = max(1, ceil(depth_fraction * problem.npml(d)));
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_inner'], ...
                            pml_idx(mask & depth(:, d) <= inner_cells)); %#ok<AGROW>
                        groups(end + 1) = make_pml_group( ...
                            ['pml_' axis_names{d} '_outer'], ...
                            pml_idx(mask & depth(:, d) > inner_cells)); %#ok<AGROW>
                    case 'signed_depth_pml'
                        inner_cells = max(1, ceil(depth_fraction * problem.npml(d)));
                        groups = append_signed_depth_groups(groups, pml_idx, mask, ...
                            low_mask(:, d), high_mask(:, d), depth(:, d), ...
                            inner_cells, axis_names{d});
                end
            end
        case 'topological_pml'
            face_count = sum(axis_mask, 2);
            topology_names = {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'};
            topology_masks = { ...
                face_count == 1 & axis_mask(:, 1), ...
                face_count == 1 & axis_mask(:, 2), ...
                face_count == 1 & axis_mask(:, 3), ...
                face_count == 2 & axis_mask(:, 1) & axis_mask(:, 2), ...
                face_count == 2 & axis_mask(:, 1) & axis_mask(:, 3), ...
                face_count == 2 & axis_mask(:, 2) & axis_mask(:, 3), ...
                face_count == 3};
            for k = 1:numel(topology_names)
                groups(end + 1) = make_pml_group( ...
                    ['pml_' topology_names{k}], ...
                    pml_idx(topology_masks{k})); %#ok<AGROW>
            end
        otherwise
            error('Unsupported refined PML partition mode "%s".', mode);
    end

    groups = groups(arrayfun(@(group) ~isempty(group.idx), groups));
    blocks = struct('name', {}, 'label', {}, 'idx', {}, 'count', {});
    for k = 1:numel(groups)
        blocks(end + 1) = make_partition_block( ...
            groups(k).name, 30 + k, groups(k).idx); %#ok<AGROW>
    end
end

function groups = append_signed_depth_groups( ...
        groups, pml_idx, mask, low, high, depth, inner_cells, axis_name)
    side_names = {'low', 'high'};
    side_masks = {low, high};
    for side = 1:2
        groups(end + 1) = make_pml_group( ...
            sprintf('pml_%s_%s_inner', axis_name, side_names{side}), ...
            pml_idx(mask & side_masks{side} & depth <= inner_cells)); %#ok<AGROW>
        groups(end + 1) = make_pml_group( ...
            sprintf('pml_%s_%s_outer', axis_name, side_names{side}), ...
            pml_idx(mask & side_masks{side} & depth > inner_cells)); %#ok<AGROW>
    end
end

function group = make_pml_group(name, idx)
    group = struct('name', name, 'idx', idx(:));
end

function block = make_partition_block(name, label, idx)
    block = struct('name', name, 'label', label, 'idx', idx(:), ...
        'count', numel(idx));
end

function [model, shared_first] = train_trajectory_rom(problem, train_params, rhs, partition, cfg, style, worker_problem, shared_first)
    timer = tic;
    ns = numel(train_params);
    nlayers = cfg.num_layers;
    checkpoint=fullfile(cfg.output_dir,sprintf('%s_%s_%s_progress.mat', ...
        problem.cfg.material_model,cfg.output_tag,style));
    policy=training_checkpoint_policy(cfg);
    resumed=false; prior_elapsed=0; first_layer=1;
    q_history = cell(ns, 1);
    for k = 1:ns
        q_history{k} = complex(zeros(problem.N, nlayers + 1));
        q_history{k}(:, 1) = rhs / max(norm(rhs), eps);
    end

    layers = cell(nlayers, 1);
    snapshot_time = zeros(nlayers, ns);
    snapshot_relres = zeros(nlayers, ns);
    snapshot_cache_hits = false(nlayers,ns);
    breakdowns = false(nlayers, ns);
    reused_snapshot_seconds = 0;
    stage_times = zeros(nlayers,4);
    if cfg.checkpoint_training && isfile(checkpoint)
        saved=load(checkpoint,'payload'); state=saved.payload;
        assert(isequaln(state.policy,policy),'Training checkpoint policy mismatch.');
        current=problem.cfg; current.test_params=state.problem_cfg.test_params;
        assert(isequaln(current,state.problem_cfg) && isequaln(partition,state.partition), ...
            'Training checkpoint geometry/partition mismatch.');
        q_history=state.q_history; layers=state.layers;
        snapshot_time=state.snapshot_time; snapshot_relres=state.snapshot_relres;
        snapshot_cache_hits=state.snapshot_cache_hits;
        breakdowns=state.breakdowns; stage_times=state.stage_times;
        shared_first=state.shared_first;
        reused_snapshot_seconds=state.reused_snapshot_seconds;
        prior_elapsed=state.elapsed; first_layer=state.completed_layer+1;
        resumed=true;
        fprintf('RESUME %s: %d completed layers from %s\n',upper(style),state.completed_layer,checkpoint);
        clear saved state
    end
    for layer_index = first_layer:nlayers
        q_inputs = complex(zeros(problem.N, ns));
        for k = 1:ns
            q = q_history{k}(:, layer_index);
            if norm(q) <= 1e-14
                breakdowns(layer_index, k) = true;
                continue;
            end
            q_inputs(:, k) = q;
        end
        snapshot_timer = tic;
        if layer_index == 1 && ~isempty(shared_first)
            snapshots = shared_first.snapshots;
            layer_times = shared_first.times;
            layer_relres = shared_first.relative_residuals;
            layer_cache_hits = shared_first.cache_hits;
            reused_snapshot_seconds = reused_snapshot_seconds + shared_first.wall_time;
        else
            [snapshots, layer_times, layer_relres, layer_cache_hits] = collect_inverse_snapshots( ...
                worker_problem, problem.N, train_params, q_inputs, breakdowns(layer_index, :), cfg, ...
                fullfile(cfg.output_dir,'snapshot_cache',problem.cfg.material_model,style,sprintf('layer_%02d',layer_index)));
            if layer_index == 1 && cfg.reuse_first_layer
                shared_first = struct('snapshots', snapshots, 'times', layer_times, ...
                    'relative_residuals', layer_relres, 'cache_hits',layer_cache_hits, ...
                    'wall_time', toc(snapshot_timer));
            end
        end
        snapshot_time(layer_index, :) = layer_times;
        snapshot_relres(layer_index, :) = layer_relres;
        snapshot_cache_hits(layer_index,:) = layer_cache_hits;
        stage_times(layer_index,1) = toc(snapshot_timer);

        tail_tol = cfg.layer_tail_tolerances(layer_index);
        stage_timer = tic;
        [space, spectrum] = build_snapshot_space( ...
            snapshots, partition, tail_tol, style, cfg.dd_rank_strategy);
        stage_times(layer_index,2) = toc(stage_timer);
        stage_timer = tic;
        layer = assemble_rom_layer(problem, space, spectrum, tail_tol, cfg);
        stage_times(layer_index,3) = toc(stage_timer);
        layers{layer_index} = layer;

        stage_timer = tic;
        for k = 1:ns
            if breakdowns(layer_index, k)
                continue;
            end
            mu = train_params(k);
            z = apply_rom_layer(layer, mu, q_history{k}(:, layer_index));
            w = apply_affine_operator(problem, mu, z);
            for j = 1:layer_index
                w = w - q_history{k}(:, j) * (q_history{k}(:, j)' * w);
            end
            wnorm = norm(w);
            if wnorm > 1e-13
                q_history{k}(:, layer_index + 1) = w / wnorm;
            else
                breakdowns(layer_index, k) = true;
            end
        end
        stage_times(layer_index,4) = toc(stage_timer);
        fprintf(['layer %d/%d: snapshots %d x %d, rank=%d, storage %.2f MiB, ' ...
            'snapshot residual mean %.3e\n'], layer_index, nlayers, ...
            size(snapshots, 1), size(snapshots, 2), layer.rank, ...
            layer.basis_bytes / 2^20, mean(snapshot_relres(layer_index, :), 'omitnan'));
        clear snapshots
        if cfg.checkpoint_training
            state=struct(); state.policy=policy; state.problem_cfg=problem.cfg;
            state.partition=partition; state.completed_layer=layer_index;
            state.layers=layers; state.q_history=q_history;
            state.snapshot_time=snapshot_time; state.snapshot_relres=snapshot_relres;
            state.snapshot_cache_hits=snapshot_cache_hits;
            state.breakdowns=breakdowns; state.stage_times=stage_times;
            state.shared_first=shared_first; state.reused_snapshot_seconds=reused_snapshot_seconds;
            state.elapsed=prior_elapsed+toc(timer);
            state.job_id=getenv('SLURM_JOB_ID'); state.created_at=datestr(now,30);
            atomic_training_save(checkpoint,state);
            clear state
            fprintf('CHECKPOINT %s layer %d saved\n',upper(style),layer_index);
            if cfg.checkpoint_stop_after_layer==layer_index
                error('helm:CheckpointTestStop','Intentional stop after a completed checkpoint.');
            end
        end
    end

    model = struct();
    model.style = style;
    model.layers = layers;
    model.num_layers = nlayers;
    model.train_params = train_params;
    model.layer_ranks = cellfun(@(v) v.rank, layers).';
    model.layer_basis_entries = cellfun(@(v) v.basis_entries, layers).';
    model.basis_entries = sum(model.layer_basis_entries);
    model.basis_bytes = sum(cellfun(@(v) v.basis_bytes, layers));
    model.reduced_entries = sum(cellfun(@(v) 2 * v.rank^2, layers));
    model.persistent_bytes = model.basis_bytes + 16 * model.reduced_entries;
    model.snapshot_solve_time = snapshot_time;
    model.snapshot_relative_residual = snapshot_relres;
    model.snapshot_cache_hits = snapshot_cache_hits;
    model.breakdowns = breakdowns;
    model.offline_time = prior_elapsed + toc(timer);
    model.resumed = resumed || any(snapshot_cache_hits(:));
    model.measured_offline_time = model.offline_time;
    model.stage_wall_seconds = stage_times;
    model.stage_names = {'snapshots','SVD','projection','replay'};
    model.shared_snapshot_time = 0;
    if cfg.reuse_first_layer, model.shared_snapshot_time = shared_first.wall_time; end
    % Standalone comparison includes the shared solve cost for both methods.
    model.offline_time = model.offline_time + reused_snapshot_seconds;
end

function [snapshots, solve_times, relative_residuals, cache_hits] = collect_inverse_snapshots( ...
        problem, n, train_params, q_inputs, skip, cfg, cache_dir)
    ns = numel(train_params);
    snapshots = complex(zeros(n, ns));
    solve_times = zeros(1, ns);
    relative_residuals = zeros(1, ns);
    cache_hits=false(1,ns);
    if cfg.cache_snapshots && ~isfolder(cache_dir), mkdir(cache_dir); end

    if cfg.parallel_train
        if ~cfg.hpc_mode, ensure_parallel_pool(cfg.parallel_workers); end
        previous_threads = maxNumCompThreads(1);
        thread_cleanup = onCleanup(@() maxNumCompThreads(previous_threads)); %#ok<NASGU>
        parfor k = 1:ns
            if ~skip(k)
                if isa(problem, 'parallel.pool.Constant')
                    local_problem = problem.Value;
                else
                    local_problem = problem;
                end
                [snapshot, solve_info] = cached_inverse_snapshot( ...
                    local_problem, train_params(k), q_inputs(:, k), cfg, ...
                    fullfile(cache_dir,sprintf('sample_%03d.mat',k)));
                snapshots(:, k) = snapshot;
                solve_times(k) = solve_info.total_time;
                relative_residuals(k) = solve_info.relative_residual;
                cache_hits(k)=solve_info.cache_hit;
            end
        end
    else
        for k = 1:ns
            if ~skip(k)
                [snapshots(:, k), solve_info] = cached_inverse_snapshot( ...
                    problem, train_params(k), q_inputs(:, k), cfg, ...
                    fullfile(cache_dir,sprintf('sample_%03d.mat',k)));
                solve_times(k) = solve_info.total_time;
                relative_residuals(k) = solve_info.relative_residual;
                cache_hits(k)=solve_info.cache_hit;
            end
        end
    end
    failed=~skip & (~isfinite(relative_residuals) | relative_residuals>cfg.snapshot_tol*(1+1e-6));
    if cfg.require_snapshot_tolerance && any(failed)
        error('helm:SnapshotTolerance', ...
            '%d/%d snapshots failed tolerance. No ROM layer accepted; inspect parameter caches when enabled.', ...
            nnz(failed),nnz(~skip));
    end
end

function policy=training_checkpoint_policy(cfg)
    policy=struct();
    for name={'num_layers','layer_tail_tolerances','snapshot_solver','snapshot_tol', ...
            'require_snapshot_tolerance','shift_beta','gmres_reorthogonalize', ...
            'gmres_orthogonalization','store_preconditioned_basis','dd_rank_strategy', ...
            'reuse_first_layer','mg_max_levels','mg_coarse_max_dofs','mg_pre_sweeps', ...
            'mg_post_sweeps','mg_jacobi_weight','mg_cycles_per_apply','snapshot_restart'}
        policy.(name{1})=cfg.(name{1});
    end
end

function [snapshot,info]=cached_inverse_snapshot(problem,mu,rhs,cfg,file)
    if ~cfg.cache_snapshots
        unchecked=cfg; unchecked.require_snapshot_tolerance=false;
        [snapshot,info]=solve_inverse_snapshot(problem,mu,rhs,unchecked); info.cache_hit=false; return;
    end
    policy=training_checkpoint_policy(cfg);
    if isfile(file)
        saved=load(file,'payload'); cache=saved.payload;
        current=problem.cfg; current.test_params=cache.problem_cfg.test_params;
        assert(isequaln(cache.policy,policy) && isequaln(current,cache.problem_cfg) && ...
            cache.mu==mu && isequaln(cache.rhs,rhs),'Snapshot cache mismatch; use a new output tag.');
        snapshot=cache.snapshot; info=cache.info; info.cache_hit=true;
        rho=norm(rhs-apply_affine_operator(problem,mu,snapshot))/max(norm(rhs),eps);
        assert(isfinite(rho) && rho<=cfg.snapshot_tol*(1+1e-6),'Cached snapshot residual failed.');
        fprintf('CACHE mu=%.6f true=%.3e\n',mu,rho); return;
    end
    unchecked=cfg; unchecked.require_snapshot_tolerance=false;
    [snapshot,info]=solve_inverse_snapshot(problem,mu,rhs,unchecked);
    info.cache_hit=false;
    cache=struct(); cache.policy=policy; cache.problem_cfg=problem.cfg;
    cache.mu=mu; cache.rhs=rhs; cache.snapshot=snapshot; cache.info=info;
    cache.job_id=getenv('SLURM_JOB_ID'); cache.created_at=datestr(now,30);
    passed=isfinite(info.relative_residual) && info.relative_residual<=cfg.snapshot_tol*(1+1e-6);
    if passed
        atomic_training_save(file,cache);
    else
        % Preserve unsuccessful parameter points independently of future retries.
        atomic_training_save([tempname(fileparts(file)) '_failed.mat'],cache);
        fprintf('Snapshot mu=%.6f failed true residual %.3e; failed artifact retained.\n',mu,info.relative_residual);
    end
end

function atomic_training_save(file,payload)
    folder=fileparts(file); if ~isfolder(folder), mkdir(folder); end
    temporary=[tempname(folder) '.mat'];
    save(temporary,'payload','-v7.3','-nocompression');
    [ok,message]=movefile(temporary,file,'f');
    assert(ok,'Checkpoint rename failed: %s',message);
end

function ensure_parallel_pool(worker_count)
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('Processes', worker_count);
    elseif pool.NumWorkers ~= worker_count
        delete(pool);
        parpool('Processes', worker_count);
    end
end

function [snapshot, info] = solve_inverse_snapshot(problem, mu, rhs, cfg)
    timer = tic;
    A = assemble_affine_matrix(problem, mu);
    solver = lower(char(string(cfg.snapshot_solver)));
    switch solver
        case 'direct'
            factor_timer = tic;
            factor = decomposition(A, 'lu');
            setup_time = toc(factor_timer);
            solve_timer = tic;
            snapshot = factor \ rhs;
            solve_time = toc(solve_timer);
            iteration_count = 0;
        case {'shifted_mg', 'mg'}
            mass_diagonal = affine_mass_diagonal(problem, mu);
            Ashift = A + 1i * cfg.shift_beta * ...
                spdiags(mass_diagonal, 0, problem.N, problem.N);
            [Msolve, setup] = build_shifted_preconditioner(Ashift, problem, cfg);
            setup_time = setup.setup_time;
            gmres_opts = make_gmres_opts(cfg, true);
            [snapshot, solve] = shifted_laplacian_gmres(@(v) A * v, ...
                Msolve, rhs, [], gmres_opts);
            solve_time = solve.total_time;
            iteration_count = solve.iter_count;
        otherwise
            error('Unknown snapshot_solver "%s".', solver);
    end
    relative_residual = norm(rhs - A * snapshot) / max(norm(rhs), eps);
    if cfg.hpc_mode
        fprintf('snapshot mu=%.6f it=%d true=%.3e wall=%.2fs\n', ...
            mu, iteration_count, relative_residual, toc(timer));
    end
    if cfg.require_snapshot_tolerance && (~isfinite(relative_residual) || ...
            relative_residual > cfg.snapshot_tol*(1+1e-6))
        error('helm:SnapshotTolerance', ...
            'Actual snapshot residual %.6e exceeds requested %.6e; increase snapshot_maxit.', ...
            relative_residual,cfg.snapshot_tol);
    end
    info = struct('solver', solver, 'setup_time', setup_time, ...
        'solve_time', solve_time, 'total_time', toc(timer), ...
        'iteration_count', iteration_count, ...
        'relative_residual', relative_residual);
end

function [space, spectrum] = build_snapshot_space( ...
        snapshots, partition, tail_tol, style, dd_rank_strategy)
    switch style
        case 'global'
            [U, S, ~] = svd(snapshots, 'econ');
            singular_values = diag(S);
            rank_value = tail_rank(singular_values, tail_tol);
            space = rom_reduced_space_utils('make_dense_space', U(:, 1:rank_value));
            spectrum = struct('global', singular_values(:));
        case 'dd'
            truncation = struct('rank_global_tail_tol', tail_tol, ...
                'rank_global_tail_strategy', dd_rank_strategy, ...
                'rank_rtol', 1e-14);
            space = rom_reduced_space_utils('build_blockn_space', ...
                snapshots, partition, truncation);
            spectrum = struct();
            for k = 1:numel(space.blocks)
                field = matlab.lang.makeValidName(space.blocks(k).name);
                spectrum.(field) = space.blocks(k).meta.singular_values;
            end
        otherwise
            error('Unknown ROM style "%s".', style);
    end
end

function rank_value = tail_rank(singular_values, tolerance)
    if isempty(singular_values) || norm(singular_values) <= eps
        rank_value = 0;
        return;
    end
    tail_energy = flipud(cumsum(flipud(abs(singular_values(:)).^2)));
    tail_after = [tail_energy(2:end); 0];
    relative_tail = sqrt(tail_after) / norm(singular_values);
    rank_value = find(relative_tail <= tolerance, 1, 'first');
    if isempty(rank_value)
        rank_value = numel(singular_values);
    end
end

function layer = assemble_rom_layer(problem, space, spectrum, tail_tol, cfg)
    A0 = @(v) problem.K_pml * v + problem.affine.mass_base * v;
    A1 = @(v) problem.affine.mass_parameter_diagonal .* v;
    if cfg.fast_reduced_assembly
        [Ar0, Ar1] = assemble_3d_rom_affine_sparse(problem, space);
    else
        Ar0 = rom_reduced_space_utils('assemble_reduced_matrix', A0, space);
        Ar1 = rom_reduced_space_utils('assemble_reduced_matrix', A1, space);
    end
    [basis_entries, basis_bytes] = space_storage(space);
    layer = struct('space', space, 'Ar0', Ar0, 'Ar1', Ar1, ...
        'rank', rom_reduced_space_utils('dimension', space), ...
        'tail_tolerance', tail_tol, 'spectrum', spectrum, ...
        'basis_entries', basis_entries, 'basis_bytes', basis_bytes);
end

function z = apply_rom_layer(layer, mu, rhs)
    if layer.rank == 0
        z = zeros(size(rhs), 'like', rhs);
        return;
    end
    reduced_rhs = rom_reduced_space_utils('project', layer.space, rhs);
    coefficient = (layer.Ar0 + mu * layer.Ar1) \ reduced_rhs;
    z = rom_reduced_space_utils('lift', layer.space, coefficient);
end

function test = run_online_test(problem, rhs, mu, models, cfg)
    test = empty_test_result();
    test.mu = mu;
    A = assemble_affine_matrix(problem, mu);
    Afun = @(v) A * v;

    if cfg.run_direct && problem.N <= cfg.direct_max_unknowns
        test.direct = benchmark_direct(A, rhs);
        fprintf('direct: setup %.2fs, solve %.2fs, residual %.3e\n', ...
            test.direct.setup_time, test.direct.solve_time, test.direct.relative_residual);
    else
        test.direct.stop_reason = 'skipped_by_size';
        fprintf('direct: skipped at N=%d\n', problem.N);
    end

    mass_diagonal = affine_mass_diagonal(problem, mu);
    Ashift = A + 1i * cfg.shift_beta * spdiags(mass_diagonal, 0, problem.N, problem.N);
    [Msolve, setup] = build_shifted_preconditioner(Ashift, problem, cfg);
    gmres_opts = make_gmres_opts(cfg, false);
    [plain_solution, plain_info] = shifted_laplacian_gmres(Afun, Msolve, rhs, [], gmres_opts);
    test.shifted = pack_iterative_result(plain_solution, plain_info, A, rhs, setup);
    fprintf('shifted-MG: setup %.2fs, solve %.2fs, it=%d, residual %.3e\n', ...
        test.shifted.setup_time, test.shifted.solve_time, ...
        test.shifted.iterations, test.shifted.relative_residual);

    styles = fieldnames(models);
    for k = 1:numel(styles)
        style = styles{k};
        model = models.(style);
        precorrection_timer = tic;
        [x0, precorrection] = layered_fgmres(Afun, rhs, model, mu, ...
            cfg.gmres_reorthogonalize);
        precorrection_time = toc(precorrection_timer);
        if precorrection.relative_residual <= cfg.gmres_tol
            x = x0;
            completion = struct('total_time', 0, 'iter_count', 0, ...
                'relres_hist', precorrection.relative_residual, 'flag', 0, ...
                'stop_reason', 'precorrection_residual');
            method_setup = struct('setup_time', 0, 'factor_bytes', 0);
        else
            [x, completion] = shifted_laplacian_gmres( ...
                Afun, Msolve, rhs, x0, gmres_opts);
            method_setup = setup;
        end

        one = pack_iterative_result(x, completion, A, rhs, method_setup);
        one.precorrection_time = precorrection_time;
        one.precorrection_iterations = precorrection.iterations;
        one.precorrection_relative_residual = precorrection.relative_residual;
        one.precorrection_residual_history = precorrection.residual_history;
        one.precorrection_solution = x0;
        one.total_time = method_setup.setup_time + precorrection_time + completion.total_time;
        test.(style) = one;
        fprintf(['%s TA-ROM: precorrection %.2fs (%d steps, %.3e), ' ...
            'completion %.2fs (%d it), final %.3e\n'], upper(style), ...
            precorrection_time, precorrection.iterations, ...
            precorrection.relative_residual, completion.total_time, ...
            completion.iter_count, one.relative_residual);
    end
end

function [x, info] = layered_fgmres(Afun, rhs, model, mu, reorthogonalize)
    n = numel(rhs);
    steps = model.num_layers;
    rhs_norm = max(norm(rhs), eps);
    Q = complex(zeros(n, steps + 1));
    Z = complex(zeros(n, steps));
    H = complex(zeros(steps + 1, steps));
    Q(:, 1) = rhs / rhs_norm;
    residual_history = ones(steps + 1, 1);
    x = complex(zeros(n, 1));
    completed = 0;

    for j = 1:steps
        if model.layers{j}.rank == 0
            break;
        end
        z = apply_rom_layer(model.layers{j}, mu, Q(:, j));
        Z(:, j) = z;
        w = Afun(z);
        for i = 1:j
            H(i, j) = Q(:, i)' * w;
            w = w - H(i, j) * Q(:, i);
        end
        if reorthogonalize
            for i = 1:j
                correction = Q(:, i)' * w;
                H(i, j) = H(i, j) + correction;
                w = w - correction * Q(:, i);
            end
        end
        H(j + 1, j) = norm(w);
        if H(j + 1, j) > 1e-13
            Q(:, j + 1) = w / H(j + 1, j);
        end
        least_squares_rhs = complex(zeros(j + 1, 1));
        least_squares_rhs(1) = rhs_norm;
        coefficients = H(1:j + 1, 1:j) \ least_squares_rhs;
        x = Z(:, 1:j) * coefficients;
        residual_history(j + 1) = norm(rhs - Afun(x)) / rhs_norm;
        completed = j;
        if H(j + 1, j) <= 1e-13
            break;
        end
    end

    info = struct('iterations', completed, ...
        'relative_residual', residual_history(completed + 1), ...
        'residual_history', residual_history(1:completed + 1));
end

function A = assemble_affine_matrix(problem, mu)
    diagonal = affine_mass_diagonal(problem, mu);
    A = problem.K_pml + spdiags(diagonal, 0, problem.N, problem.N);
end

function diagonal = affine_mass_diagonal(problem, mu)
    diagonal = problem.affine.mass_base + mu * problem.affine.mass_parameter_diagonal;
end

function y = apply_affine_operator(problem, mu, x)
    y = problem.K_pml * x + affine_mass_diagonal(problem, mu) .* x;
end

function [Msolve, setup] = build_shifted_preconditioner(Ashift, problem, cfg)
    opts = struct('solver', 'mg', 'grid_size', problem.n, ...
        'mg_max_levels', cfg.mg_max_levels, ...
        'mg_coarse_max_dofs', cfg.mg_coarse_max_dofs, ...
        'mg_pre_sweeps', cfg.mg_pre_sweeps, ...
        'mg_post_sweeps', cfg.mg_post_sweeps, ...
        'mg_jacobi_weight', cfg.mg_jacobi_weight, ...
        'mg_cycles_per_apply', cfg.mg_cycles_per_apply);
    [Msolve, setup] = build_shifted_laplacian_preconditioner(Ashift, opts);
end

function opts = make_gmres_opts(cfg, snapshot_mode)
    if snapshot_mode
        opts = struct('restart', cfg.snapshot_restart, ...
            'maxit', cfg.snapshot_maxit, 'tol', cfg.snapshot_tol, ...
            'reorthogonalize', cfg.gmres_reorthogonalize, 'verbose', false);
    else
        opts = struct('restart', cfg.gmres_restart, ...
            'maxit', cfg.gmres_maxit, 'tol', cfg.gmres_tol, ...
            'reorthogonalize', cfg.gmres_reorthogonalize, 'verbose', false);
    end
    opts.orthogonalization = cfg.gmres_orthogonalization;
    opts.store_preconditioned_basis = cfg.store_preconditioned_basis;
    opts.fixed_linear_preconditioner = true; % Fixed shifted-MG V-cycle.
end

function result = benchmark_direct(A, rhs)
    result = empty_solver_result();
    timer = tic;
    factor = decomposition(A, 'lu');
    result.setup_time = toc(timer);
    factor_info = whos('factor');
    result.factor_bytes = factor_info.bytes;
    timer = tic;
    x = factor \ rhs;
    result.solve_time = toc(timer);
    result.total_time = result.setup_time + result.solve_time;
    result.relative_residual = norm(rhs - A * x) / max(norm(rhs), eps);
    result.iterations = 0;
    result.flag = 0;
    result.stop_reason = 'direct';
    result.solution = x;
end

function result = pack_iterative_result(x, solve, A, rhs, setup)
    result = empty_solver_result();
    result.setup_time = setup.setup_time;
    result.solve_time = solve.total_time;
    result.total_time = setup.setup_time + solve.total_time;
    result.iterations = solve.iter_count;
    result.relative_residual = norm(rhs - A * x) / max(norm(rhs), eps);
    result.residual_history = solve.relres_hist;
    result.factor_bytes = setup.factor_bytes;
    result.flag = solve.flag;
    result.stop_reason = solve.stop_reason;
    result.solution = x;
end

function [entries, bytes] = space_storage(space)
    switch space.type
        case 'dense'
            entries = numel(space.basis);
            is_real = isreal(space.basis);
        case 'blockn'
            entries = 0;
            is_real = true;
            for k = 1:numel(space.blocks)
                entries = entries + numel(space.blocks(k).basis);
                is_real = is_real && isreal(space.blocks(k).basis);
            end
        otherwise
            error('Unsupported reduced-space type "%s".', space.type);
    end
    bytes = entries * 8 * (1 + ~is_real);
end

function table_out = make_summary_table(name, problem, problem_cfg, ppw, ...
        setup_time, models, tests)
    methods = {'direct', 'shifted', 'global', 'dd'};
    row_count = numel(tests) * numel(methods);
    candidate = strings(row_count, 1);
    mu = zeros(row_count, 1);
    method = strings(row_count, 1);
    N = repmat(problem.N, row_count, 1);
    omega = repmat(problem_cfg.omega, row_count, 1);
    ppw_min = repmat(ppw, row_count, 1);
    problem_setup_s = repmat(setup_time, row_count, 1);
    offline_s = zeros(row_count, 1);
    preconditioner_setup_s = NaN(row_count, 1);
    precorrection_s = zeros(row_count, 1);
    solve_s = NaN(row_count, 1);
    total_s = NaN(row_count, 1);
    iterations = NaN(row_count, 1);
    precorrection_relres = NaN(row_count, 1);
    final_relres = NaN(row_count, 1);
    flag = NaN(row_count, 1);
    basis_entries = zeros(row_count, 1);
    basis_MiB = zeros(row_count, 1);
    reduced_entries = zeros(row_count, 1);
    persistent_MiB = zeros(row_count, 1);
    preconditioner_MiB = NaN(row_count, 1);
    speedup_vs_shifted = NaN(row_count, 1);
    break_even_queries_vs_shifted = NaN(row_count, 1);
    layer_ranks = strings(row_count, 1);

    row = 0;
    for k = 1:numel(tests)
        for j = 1:numel(methods)
            row = row + 1;
            key = methods{j};
            one = tests(k).(key);
            candidate(row) = string(name);
            mu(row) = tests(k).mu;
            method(row) = string(key);
            preconditioner_setup_s(row) = one.setup_time;
            precorrection_s(row) = one.precorrection_time;
            solve_s(row) = one.solve_time;
            total_s(row) = one.total_time;
            iterations(row) = one.iterations;
            precorrection_relres(row) = one.precorrection_relative_residual;
            final_relres(row) = one.relative_residual;
            flag(row) = one.flag;
            preconditioner_MiB(row) = one.factor_bytes / 2^20;
            if isfinite(one.total_time) && one.total_time > 0
                speedup_vs_shifted(row) = tests(k).shifted.total_time / one.total_time;
            end
            if isfield(models, key)
                offline_s(row) = models.(key).offline_time;
                basis_entries(row) = models.(key).basis_entries;
                basis_MiB(row) = models.(key).basis_bytes / 2^20;
                reduced_entries(row) = models.(key).reduced_entries;
                persistent_MiB(row) = models.(key).persistent_bytes / 2^20;
                layer_ranks(row) = string(mat2str(models.(key).layer_ranks));
                online_saving = tests(k).shifted.total_time - one.total_time;
                if online_saving > 0
                    break_even_queries_vs_shifted(row) = ...
                        models.(key).offline_time / online_saving;
                end
            end
        end
    end
    table_out = table(candidate, mu, method, N, omega, ppw_min, ...
        problem_setup_s, offline_s, preconditioner_setup_s, precorrection_s, ...
        solve_s, total_s, iterations, precorrection_relres, final_relres, ...
        flag, basis_entries, basis_MiB, reduced_entries, persistent_MiB, ...
        preconditioner_MiB, speedup_vs_shifted, ...
        break_even_queries_vs_shifted, layer_ranks);
end

function plot_candidate_results(name, models, tests, out_dir, output_tag)
    colors = lines(3);
    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 760, 520]);
    ax = axes(fig);
    semilogy(ax, 0:numel(tests(1).shifted.residual_history) - 1, ...
        tests(1).shifted.residual_history, '-o', 'Color', colors(1, :), ...
        'LineWidth', 1.4, 'DisplayName', 'Shifted Laplacian + MG');
    hold(ax, 'on');
    styles = intersect({'global', 'dd'}, fieldnames(models), 'stable');
    for k = 1:numel(styles)
        style = styles{k};
        precorr = tests(1).(style).precorrection_residual_history(:);
        completion = tests(1).(style).residual_history(:);
        combined = [precorr; completion(2:end)];
        semilogy(ax, 0:numel(combined) - 1, combined, '-s', ...
            'Color', colors(k + 1, :), 'LineWidth', 1.4, ...
            'DisplayName', sprintf('%s TA-ROM + shifted MG', upper(style)));
    end
    xlabel(ax, 'iteration / pre-correction step');
    ylabel(ax, 'relative residual');
    title(ax, sprintf('%s, \\mu=%.3f', strrep(name, '_', ' '), tests(1).mu));
    grid(ax, 'on');
    legend(ax, 'Location', 'southwest');
    residual_file = fullfile(out_dir, ...
        sprintf('%s_%s_residual.png', name, output_tag));
    exportgraphics(fig, residual_file, 'Resolution', 180);
    close(fig);

    if all(isfield(models, {'global', 'dd'}))
        fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 620, 460]);
        values = [models.global.basis_bytes, models.dd.basis_bytes] / 2^20;
        bar(values, 0.55);
        set(gca, 'XTickLabel', {'Global', 'DD'});
        ylabel('stored basis (MiB)');
        title(strrep(name, '_', ' '));
        grid on;
        storage_file = fullfile(out_dir, ...
            sprintf('%s_%s_storage.png', name, output_tag));
        exportgraphics(fig, storage_file, 'Resolution', 180);
        close(fig);
    end
end

function summaries = summarize_models(models)
    summaries = struct();
    names = fieldnames(models);
    for k = 1:numel(names)
        model = models.(names{k});
        if isfield(model, 'layers')
            model = rmfield(model, 'layers');
        end
        summaries.(names{k}) = model;
    end
end

function paper_plot_data = make_paper_plot_data(results)
    paper_plot_data = struct();
    paper_plot_data.schema_version = 1;
    paper_plot_data.mode = results.mode;
    paper_plot_data.cfg = results.cfg;
    paper_plot_data.created_at = results.created_at;
    paper_plot_data.matlab_version = results.matlab_version;
    paper_plot_data.summary_table = results.summary_table;
    paper_plot_data.candidates = cell(size(results.candidates));
    for k = 1:numel(results.candidates)
        candidate = results.candidates{k};
        candidate.models = summarize_models(candidate.models);
        candidate = rmfield(candidate, {'partition', 'rhs'});
        methods = {'direct', 'shifted', 'global', 'dd'};
        for test_index = 1:numel(candidate.tests)
            for method_index = 1:numel(methods)
                key = methods{method_index};
                candidate.tests(test_index).(key) = rmfield( ...
                    candidate.tests(test_index).(key), ...
                    {'solution', 'precorrection_solution'});
            end
        end
        paper_plot_data.candidates{k} = candidate;
    end
end

function result = empty_test_result()
    empty = empty_solver_result();
    result = struct('mu', NaN, 'direct', empty, 'shifted', empty, ...
        'global', empty, 'dd', empty);
end

function result = empty_solver_result()
    result = struct('setup_time', NaN, 'solve_time', NaN, 'total_time', NaN, ...
        'iterations', NaN, 'relative_residual', NaN, 'residual_history', [], ...
        'factor_bytes', NaN, 'flag', NaN, 'stop_reason', '', ...
        'precorrection_time', 0, 'precorrection_iterations', 0, ...
        'precorrection_relative_residual', NaN, ...
        'precorrection_residual_history', [], ...
        'solution', [], 'precorrection_solution', []);
end

function out = merge_structs(base, override)
    out = base;
    names = fieldnames(override);
    for k = 1:numel(names)
        out.(names{k}) = override.(names{k});
    end
end
