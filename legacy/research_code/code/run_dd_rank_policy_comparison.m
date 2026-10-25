function report = run_dd_rank_policy_comparison(case_name, tag)
%RUN_DD_RANK_POLICY_COMPARISON Cross policies on fixed publication partitions.
% Uses full trajectory retraining, original tests, and original tolerances.
% Raw models/fields stay in the existing experiment output directories.
    if nargin < 1, case_name = 'all'; end
    if nargin < 2, tag = 'policy_crosscheck_20260905'; end
    case_name = char(case_name);
    tag = char(tag);
    assert(ismember(case_name, {'all','halfspace','waveguide'}), 'Unknown case.');
    assert(~isempty(regexp(tag, '^[A-Za-z0-9_-]+$', 'once')), 'Invalid tag.');
    out = helm_output_path('outputs', 'dd_rank_policy_comparison', tag);
    if ~exist(out, 'dir'), mkdir(out); end

    if ismember(case_name, {'all','halfspace'})
        compare_halfspace(tag, out);
    end
    if ismember(case_name, {'all','waveguide'})
        compare_waveguide(tag, out);
    end
    summaries = cell(0,1);
    for name = {'halfspace','waveguide'}
        file = fullfile(out, [name{1} '.mat']);
        if isfile(file)
            saved = load(file, 'comparison');
            summaries{end+1,1} = saved.comparison.summary; %#ok<AGROW>
        end
    end
    report = struct('created_at', datetime('now'), 'matlab_version', version, ...
        'tag', tag, 'summary', vertcat(summaries{:}));
    save(fullfile(out, 'comparison.mat'), 'report');
    writetable(report.summary, fullfile(out, 'comparison.csv'));
    disp(report.summary(:, {'problem','policy','basis_saved_percent', ...
        'rom_saved_percent','completion_mean','online_mean_s', ...
        'speedup_vs_shifted','max_final_residual'}));
end

function compare_halfspace(tag, out)
    variants = {'global','slabs1-aligned-sym8','slabs1-aligned-sym8-dof'};
    runs = run_halfspace_dd_refinement('paper', struct('tag', tag, ...
        'variant_names', {variants}));
    data = cellfun(@(name) load(fullfile(runs.output_dir, [name '.mat']), ...
        'row', 'artifact', 'requested_cfg'), ...
        variants, 'UniformOutput', false);
    assert(isequaln(data{2}.artifact.layers{1}.basis_space.partition, ...
        data{3}.artifact.layers{1}.basis_space.partition), 'Partitions differ.');
    left = data{2}.requested_cfg;
    right = data{3}.requested_cfg;
    left.corr_block3_global_tail_strategy = right.corr_block3_global_tail_strategy;
    assert(isequaln(left, right), 'Non-policy DD settings differ.');
    policies = {'global','marginal_storage','dof_weighted'};
    rows = repmat(empty_row(), numel(data), 1);
    for k = 1:numel(data)
        one = data{k};
        assert(isequal(one.artifact.test.params, data{1}.artifact.test.params));
        assert(numel(one.artifact.test.params) == 23, 'Expected all 23 tests.');
        assert(one.row.max_final_completion_residual <= ...
            1.05 * one.artifact.cfg.fp_tol, 'Completion tolerance missed.');
        assert(isfinite(one.row.max_final_helm_residual) && ...
            one.row.max_final_helm_residual < 1e-12, 'Helmholtz check failed.');
        row = empty_row();
        row.problem = "halfspace";
        row.policy = string(policies{k});
        row.test_count = numel(one.artifact.test.params);
        row.basis_MiB = one.row.basis_bytes / 2^20;
        row.rom_MiB = one.row.rom_payload_bytes / 2^20;
        row.basis_saved_percent = 100*(1-one.row.basis_bytes/data{1}.row.basis_bytes);
        row.rom_saved_percent = 100*(1-one.row.rom_payload_bytes/data{1}.row.rom_payload_bytes);
        row.completion_mean = one.row.mean_completion_iter;
        row.completion_max = one.row.max_completion_iter;
        row.online_mean_s = one.row.mean_online_s;
        row.max_final_residual = one.row.max_final_helm_residual;
        row.mean_precorrection_residual = one.row.mean_precorr_residual;
        row.offline_s = one.row.offline_s;
        row.layer_ranks = one.row.space_dims;
        row.source_file = string(fullfile(runs.output_dir, [variants{k} '.mat']));
        rows(k) = row;
    end
    comparison = struct('summary', struct2table(rows), ...
        'test_params', data{1}.artifact.test.params, ...
        'accounting', 'Basis + two affine reduced operators + one shared index map.');
    save(fullfile(out, 'halfspace.mat'), 'comparison');
    writetable(comparison.summary, fullfile(out, 'halfspace.csv'));
    disp(comparison.summary);
end

function compare_waveguide(tag, out)
    spec = helmholtz_3d_40pct_config('waveguide');
    policies = {'dof_weighted','marginal_storage'};
    rows = repmat(empty_row(), 3, 1);
    baseline = [];
    for k = 1:numel(policies)
        cfg = spec.driver_cfg;
        % The original 400-step cap can stop before the snapshot tolerance.
        cfg.snapshot_maxit = max(cfg.snapshot_maxit, 20);
        cfg.dd_rank_strategy = policies{k};
        cfg.output_tag = [tag '_waveguide_' policies{k}];
        if k == 2, cfg.train_styles = {'dd'}; end
        file = helm_output_path('outputs', 'helmholtz_3d_candidates', 'large', ...
            ['helmholtz_3d_candidates_' cfg.output_tag '.mat']);
        if isfile(file)
            saved = load(file, 'results');
            result = saved.results;
            for field = fieldnames(cfg).'
                assert(isequaln(result.cfg.(field{1}), cfg.(field{1})), ...
                    'Cached setting %s differs; use a new tag.', field{1});
            end
        else
            result = run_helmholtz_3d_candidate_experiments(spec.mode, cfg);
        end
        one = result.candidates{1};
        validate_3d_artifact(one, result.cfg);
        if k == 1
            baseline = one.models.global;
            partition = one.partition;
            rows(1) = summarize_3d(one, 'global', 'global', baseline, file);
        else
            assert(isequaln(partition, one.partition), 'Partitions differ.');
        end
        rows(k+1) = summarize_3d(one, 'dd', policies{k}, baseline, file);
        clear result saved one;
    end
    comparison = struct('summary', struct2table(rows), ...
        'test_params', spec.problem_cfg.test_params, ...
        'accounting', 'Publication 3D accounting: basis + two affine reduced operators.');
    save(fullfile(out, 'waveguide.mat'), 'comparison');
    writetable(comparison.summary, fullfile(out, 'waveguide.csv'));
    disp(comparison.summary);
end

function validate_3d_artifact(one, cfg)
    for key = fieldnames(one.models).'
        name = key{1};
        model = one.models.(name);
        assert(isfield(model, 'layers'), 'Complete bases were not saved.');
        assert(all(isfinite(model.snapshot_relative_residual(:))) && ...
            max(model.snapshot_relative_residual(:)) <= 1.05*cfg.snapshot_tol, ...
            'Snapshot solves did not reach the requested tolerance.');
        for j = 1:numel(model.layers)
            layer = model.layers{j};
            if strcmp(name, 'dd')
                for b = 1:numel(layer.space.blocks)
                    tail = layer.space.blocks(b).meta.global_relative_tail;
                    assert(tail <= layer.tail_tolerance*(1+1e-8), ...
                        'Snapshot-tail budget missed.');
                end
            end
        end
        tests = arrayfun(@(test) test.(name), one.tests);
        assert(all([tests.flag] == 0) && all(isfinite([tests.relative_residual])) && ...
            max([tests.relative_residual]) <= 1.05*cfg.gmres_tol, ...
            'Full-system convergence check failed.');
    end
end

function row = summarize_3d(one, method, policy, baseline, file)
    model = one.models.(method);
    tests = arrayfun(@(test) test.(method), one.tests);
    shifted = [one.tests.shifted];
    row = empty_row();
    row.problem = "waveguide";
    row.policy = string(policy);
    row.test_count = numel(tests);
    row.basis_MiB = model.basis_bytes / 2^20;
    row.rom_MiB = model.persistent_bytes / 2^20;
    row.basis_saved_percent = 100*(1-model.basis_bytes/baseline.basis_bytes);
    row.rom_saved_percent = 100*(1-model.persistent_bytes/baseline.persistent_bytes);
    row.completion_mean = mean([tests.iterations]);
    row.completion_max = max([tests.iterations]);
    row.online_mean_s = mean([tests.total_time]);
    row.max_final_residual = max([tests.relative_residual]);
    row.mean_precorrection_residual = mean([tests.precorrection_relative_residual]);
    row.offline_s = model.offline_time;
    row.speedup_vs_shifted = mean([shifted.total_time])/row.online_mean_s;
    row.layer_ranks = string(mat2str(model.layer_ranks));
    row.source_file = string(file);
end

function row = empty_row()
    row = struct('problem', "", 'policy', "", 'test_count', 0, ...
        'basis_MiB', NaN, 'rom_MiB', NaN, 'basis_saved_percent', NaN, ...
        'rom_saved_percent', NaN, 'completion_mean', NaN, 'completion_max', NaN, ...
        'online_mean_s', NaN, 'max_final_residual', NaN, ...
        'mean_precorrection_residual', NaN, 'offline_s', NaN, ...
        'speedup_vs_shifted', NaN, 'layer_ranks', "", 'source_file', "");
end
