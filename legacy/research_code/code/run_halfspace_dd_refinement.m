function results = run_halfspace_dd_refinement(mode, user_cfg)
%RUN_HALFSPACE_DD_REFINEMENT Replay each DD candidate with the paper tail budget.
% pilot: paper grid/training, five held-out tests; paper: all 23 held-out tests.
% Results are checkpointed separately, including deployable bases and spectra.
    if nargin < 1, mode = 'pilot'; end
    if nargin < 2, user_cfg = struct(); end
    mode = char(mode);
    assert(ismember(mode, {'smoke','pilot','paper'}), 'Unknown experiment mode.');
    cfg = struct('material_model', 'halfspace', 'reference_mode', 'helm_lift', ...
        'rom_check_target', 'helm', 'online_completion_mode', 'waveholtz', ...
        'online_wave_solver', 'fixedpoint', 'use_direct_rom_init', false, ...
        'build_direct_rom_when_unused', false, 'run_wave_baselines', false, ...
        'run_nr_rom_comparison', false, 'do_plots', false, 'save_plots', false, ...
        'save_benchmark', false, 'parallel_train', false, 'parallel_test', false, ...
        'train_params', linspace(0.8, 0.95, 24), 'test_param_rng_seed', 1, ...
        'corr_num_layers', 3, 'corr_pod_singular_value_tols', [1e-5,1e-5,1e-5]);
    if strcmp(mode, 'pilot')
        old_rng = rng;
        rng(1, 'twister');
        all_tests = cfg.train_params(1:end-1) + diff(cfg.train_params) .* rand(1,23);
        rng(old_rng);
        cfg.test_params = all_tests([1,6,12,18,23]);
    elseif strcmp(mode, 'smoke')
        cfg.Nphys = 25;
        cfg.omega = 12;
        cfg.train_params = linspace(0.8,0.95,6);
        cfg.test_params = [0.813,0.9072];
        cfg.fp_tol = 1e-10;
        cfg.fp_maxit = 80;
    end
    if isfield(user_cfg, 'solver_cfg'), cfg = merge(cfg, user_cfg.solver_cfg); end
    variants = candidates();
    if strcmp(mode, 'paper') && ~isfield(user_cfg, 'variant_names')
        user_cfg.variant_names = {'global','material3-dof','slabs1-aligned-sym8-dof', ...
            'slabs2-aligned-sym8','slabs3-aligned-sym8'};
    end
    if isfield(user_cfg, 'variant_names')
        requested = string(user_cfg.variant_names);
        assert(~isempty(requested), 'Select at least one variant.');
        assert(all(ismember(requested, string({variants.name}))), 'Unknown variant.');
        variants = variants(ismember(string({variants.name}), requested));
    end
    tag = mode;
    if isfield(user_cfg, 'tag'), tag = char(user_cfg.tag); end
    assert(~isempty(regexp(tag, '^[A-Za-z0-9_-]+$', 'once')), 'Invalid output tag.');
    out = helm_output_path('outputs', 'halfspace_dd_refinement', tag);
    if ~exist(out, 'dir'), mkdir(out); end
    resume = true;
    if isfield(user_cfg, 'resume'), resume = user_cfg.resume; end
    rows = struct([]);
    for k = 1:numel(variants)
        variant = variants(k);
        run_cfg = merge(cfg, variant.cfg);
        if strcmp(run_cfg.corr_basis_style, 'block3')
            run_cfg.corr_block3_global_tail_tols = run_cfg.corr_pod_singular_value_tols;
        end
        file = fullfile(out, [variant.name '.mat']);
        if resume && isfile(file)
            saved = load(file, 'requested_cfg', 'row');
            assert(isequaln(saved.requested_cfg, run_cfg), ...
                'Cached configuration differs; use a new tag or resume=false.');
            row = saved.row;
        else
            fprintf('\n=== Half-space DD refinement: %s / %s ===\n', tag, variant.name);
            r = helm_waveholtz_correction_rom(run_cfg);
            [row, artifact] = summarize(r, variant.name);
            requested_cfg = run_cfg;
            save(file, 'row', 'artifact', 'requested_cfg', '-v7.3');
            clear r artifact;
        end
        if k == 1
            rows = row;
        else
            rows(k) = row;
        end
        tbl = struct2table(rows);
        g = find(tbl.variant == "global", 1);
        tbl.saved_basis_percent = nan(height(tbl),1);
        tbl.saved_rom_payload_percent = nan(height(tbl),1);
        if ~isempty(g)
            tbl.saved_basis_percent = 100 * (1 - tbl.basis_bytes / tbl.basis_bytes(g));
            tbl.saved_rom_payload_percent = 100 * ...
                (1 - tbl.rom_payload_bytes / tbl.rom_payload_bytes(g));
        end
        writetable(tbl, fullfile(out, 'summary.csv'));
        save(fullfile(out, 'summary.mat'), 'tbl', 'variants', 'cfg');
        disp(tbl(:, {'variant','saved_basis_percent','saved_rom_payload_percent', ...
            'mean_completion_iter','mean_online_s','max_final_helm_residual'}));
    end
    results = struct('table', tbl, 'output_dir', out);
end

function v = candidates()
    v = make('global', struct('corr_basis_style','global'));
    legacy = struct('corr_basis_style','block3', 'block3_partition_mode','halfspace_material', ...
        'corr_block3_global_tail_strategy','dof_weighted');
    v(end+1) = make('material3-dof', legacy);
    legacy.corr_block3_global_tail_strategy = 'marginal_storage';
    v(end+1) = make('material3-greedy', legacy);
    refined = struct('corr_basis_style','block3', 'block3_partition_mode','halfspace_refined', ...
        'corr_block3_global_tail_strategy','marginal_storage', ...
        'halfspace_dd_physical_mode','slabs', 'halfspace_dd_sections',1, ...
        'halfspace_dd_transverse_sections',1, 'halfspace_dd_pml_mode','signed', ...
        'halfspace_dd_depth_edges',0.5);
    v(end+1) = make('material-pml4', refined);
    refined.halfspace_dd_pml_mode = 'signed_depth';
    v(end+1) = make('material-pml8', refined);
    for n = [2,3,4,6]
        c = refined;
        c.halfspace_dd_sections = n;
        v(end+1) = make(sprintf('slabs%d-pml8',n), c); %#ok<AGROW>
    end
    c = refined;
    c.halfspace_dd_sections = 2;
    c.halfspace_dd_transverse_sections = 2;
    v(end+1) = make('grid4x2-pml8', c);
    c.halfspace_dd_transverse_sections = 4;
    v(end+1) = make('grid4x4-pml8', c);
    c = refined;
    c.halfspace_dd_physical_mode = 'source_rings';
    c.halfspace_dd_radii = [0.15,0.3,0.5];
    v(end+1) = make('rings4-pml8', c);
    c.halfspace_dd_radii = [0.1,0.2,0.3,0.45,0.65];
    v(end+1) = make('rings6-pml8', c);
    c = refined;
    c.halfspace_dd_sections = 3;
    c.corr_block3_global_tail_strategy = 'dof_weighted';
    v(end+1) = make('slabs3-pml8-dof', c);
    c = refined;
    c.halfspace_dd_sections = 3;
    c.halfspace_dd_depth_edges = [1/3,2/3];
    v(end+1) = make('slabs3-pml12', c);
    for n = [1,2,3]
        c = refined;
        c.halfspace_dd_sections = n;
        c.halfspace_dd_pml_mode = 'material_symmetric_depth';
        v(end+1) = make(sprintf('slabs%d-aligned-sym8',n), c); %#ok<AGROW>
        c.corr_block3_global_tail_strategy = 'dof_weighted';
        v(end+1) = make(sprintf('slabs%d-aligned-sym8-dof',n), c); %#ok<AGROW>
    end
    c = refined;
    c.halfspace_dd_sections = 2;
    c.halfspace_dd_pml_mode = 'material_signed_depth';
    v(end+1) = make('slabs2-aligned12', c);
    c = refined;
    c.corr_block3_global_tail_strategy = 'dof_weighted';
    c.halfspace_dd_pml_mode = 'signed';
    v(end+1) = make('material-pml4-dof', c);
end

function v = make(name, cfg)
    v = struct('name',name, 'cfg',cfg);
end

function [row, artifact] = summarize(r, name)
    layers = r.correction_rom.layers;
    basis_bytes = 0;
    operator_bytes = 0;
    partition_bytes = 0;
    deploy_layers = cell(size(layers));
    for j = 1:numel(layers)
        layer = layers{j};
        space = layer.basis_space;
        diag = struct('ranks', [], 'dofs', [], 'energy', [], 'local_tails', [], ...
            'global_tail', NaN, 'labels', strings(1,0));
        if strcmp(space.type, 'dense')
            bases = {space.basis};
            diag.ranks = space.k;
            diag.dofs = space.n;
            diag.labels = "global";
        elseif strcmp(space.type, 'block3')
            bases = {space.Un,space.Uf,space.Up};
            diag.ranks = [space.kn,space.kf,space.kp];
            diag.dofs = [space.partition.count_near,space.partition.count_far,space.partition.count_pml];
            diag.labels = ["near","far","pml"];
        else
            bases = {space.blocks.basis};
            diag.ranks = [space.blocks.k];
            diag.dofs = [space.blocks.count];
            diag.labels = string({space.blocks.name});
        end
        for b = 1:numel(bases), basis_bytes = basis_bytes + bytes(bases{b}); end
        operator_bytes = operator_bytes + bytes(layer.Ar_base) + bytes(layer.Ar_right);
        if ~strcmp(space.type, 'dense')
            if j == 1
                % Count one shared double index map; training duplicates are excluded.
                partition_bytes = 8 * space.n;
            end
            for b = 1:numel(diag.ranks)
                if strcmp(space.type, 'block3')
                    meta = space.block_meta.(char(diag.labels(b)));
                else
                    meta = space.blocks(b).meta;
                end
                diag.energy(b) = sum(meta.singular_values.^2);
                diag.local_tails(b) = meta.local_relative_tail;
                diag.global_tail = meta.global_relative_tail;
            end
        end
        deploy_layers{j} = struct('basis_space',space, 'Ar_base',layer.Ar_base, ...
            'Ar_right',layer.Ar_right, 'eta_snapshots',layer.eta_snapshots, ...
            'singular_values',layer.pod.singular_values, 'diagnostics',diag, ...
            'trajectory_diagnostics',{layer.trajectory_diagnostics});
    end
    full_residual = zeros(numel(r.test.params),1);
    for k = 1:numel(full_residual)
        alpha = 1/r.test.params(k)^2 - 1;
        A = r.problem.A_base + alpha * r.problem.A_right;
        full_residual(k) = norm(A*r.test.online_final_u(:,k)-r.rhs)/norm(r.rhs);
    end
    row = struct('variant',string(name), 'basis_bytes',basis_bytes, ...
        'operator_bytes',operator_bytes, 'partition_bytes',partition_bytes, ...
        'rom_payload_bytes',basis_bytes+operator_bytes+partition_bytes, ...
        'mean_precorr_residual',mean(r.test.online_precorr_relres_final), ...
        'max_precorr_residual',max(r.test.online_precorr_relres_final), ...
        'mean_completion_iter',mean(r.test.online_fp_iter), ...
        'max_completion_iter',max(r.test.online_fp_iter), ...
        'mean_online_s',mean(r.test.online_total_time), ...
        'mean_precorr_s',mean(r.test.online_precorr_time), ...
        'max_final_helm_residual',max(full_residual), ...
        'max_final_completion_residual',max(r.test.online_fp_relres_final), ...
        'max_final_error',max(r.test.online_final_err_helm_full), ...
        'offline_s',r.timing.offline, 'space_dims',string(mat2str(r.summary.corr_ranks(:).')));
    artifact = struct('cfg',r.cfg, 'layers',{deploy_layers}, 'summary',r.summary, ...
        'timing',r.timing, 'rhs',r.rhs, 'x_int',r.problem.x_int, ...
        'y_int',r.problem.y_int, 'physical_i',r.problem.I, 'physical_j',r.problem.J, ...
        'test',r.test, 'final_helm_residual',full_residual);
end

function n = bytes(value) %#ok<INUSD>
    info = whos('value');
    n = info.bytes;
end

function a = merge(a, b)
    fields = fieldnames(b);
    for k = 1:numel(fields), a.(fields{k}) = b.(fields{k}); end
end
