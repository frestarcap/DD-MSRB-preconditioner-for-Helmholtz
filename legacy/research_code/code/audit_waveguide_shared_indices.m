function rows = audit_waveguide_shared_indices(source, output_file, reference_dir)
%AUDIT_WAVEGUIDE_SHARED_INDICES Compare actual objects and frozen actions.
% No training, full-order assembly, or completion solves. All measured objects
% retain the same online arrays; only duplicate layer indices are removed.
    if nargin < 3, reference_dir = ''; end
    timer = tic;
    saved = load(source, 'models', 'resolved_problem_cfg', 'rhs');
    previous_rng = rng; restore_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
    rng(73);
    rows = struct([]);
    names = {'global','dd'};
    for m = 1:numel(names)
        method = names{m}; original = saved.models.(method);
        old = build_jsc_3d_deployment_model(original, 'per_layer');
        model = build_jsc_3d_deployment_model(original, 'shared');
        old_info = whos('old'); new_info = whos('model');
        if ~isempty(reference_dir)
            reference = load(fullfile(reference_dir, ['waveguide_' method '.mat']), 'model');
            assert(isequaln(reference.model, old));
            clear reference
        end
        indices = {};
        if strcmp(method, 'dd')
            assert(isfield(model, 'shared_block_indices'));
            indices = model.shared_block_indices;
            reconstructed = rmfield(model, 'shared_block_indices');
            for j = 1:model.num_layers
                assert(~isfield(model.layers{j}.space.blocks, 'idx'));
                for b = 1:numel(indices)
                    reconstructed.layers{j}.space.blocks(b).idx = indices{b};
                end
                reconstructed.layers{j}.space.blocks = orderfields( ...
                    reconstructed.layers{j}.space.blocks, old.layers{j}.space.blocks);
            end
            assert(isequaln(reconstructed, old), 'Retained fields changed.');
            clear reconstructed
        else
            assert(isequaln(old, model));
        end
        n = old.layers{1}.space.n;
        q = complex(randn(n,1), randn(n,1)); q = q / norm(q);
        checked_actions = 0;
        for j = 1:model.num_layers
            l = model.layers{j}; prev = old.layers{j};
            c = rom_reduced_space_utils('project', prev.space, q);
            assert(isequaln(c, rom_reduced_space_utils('project', l.space, q, indices)));
            a = complex(randn(l.rank,1), randn(l.rank,1));
            assert(isequaln(rom_reduced_space_utils('lift', prev.space, a), ...
                rom_reduced_space_utils('lift', l.space, a, indices)));
            for mu = [0.237 0.317 0.483]
                zold = rom_reduced_space_utils('lift', prev.space, (prev.Ar0+mu*prev.Ar1)\c);
                znew = rom_reduced_space_utils('lift', l.space, (l.Ar0+mu*l.Ar1)\ ...
                    rom_reduced_space_utils('project', l.space, q, indices), indices);
                assert(isequaln(zold, znew), 'Reduced inverse action changed.');
                checked_actions = checked_actions + 1;
            end
        end
        row = struct('omega', saved.resolved_problem_cfg.omega, 'method', method, ...
            'N', n, 'layers', model.num_layers, 'old_whos_bytes', old_info.bytes, ...
            'shared_whos_bytes', new_info.bytes, 'removed_logical_bytes', old_info.bytes-new_info.bytes, ...
            'old_MiB', old_info.bytes/2^20, 'shared_MiB', new_info.bytes/2^20, ...
            'removed_MiB', (old_info.bytes-new_info.bytes)/2^20, ...
            'checked_inverse_actions', checked_actions, 'actions_bitwise_equal', true, ...
            'retained_values_equal', true, 'matlab_version', version, ...
            'matlab_release', version('-release'), 'node', getenv('SLURMD_NODENAME'), ...
            'job_id', getenv('SLURM_JOB_ID'));
        if m == 1, rows = row; else, rows(m) = row; end
        fprintf('SHARED_INDEX_RESULT %s\n', jsonencode(row));
        clear model old original l prev q zold znew indices
    end
    result = struct('source', source, 'elapsed_seconds', toc(timer), ...
        'protocol', 'Actual old/shared deployment whos; identical retained arrays and reduced actions; no physical RAM or online timing inference', ...
        'validated', true, 'rows', rows);
    fid = fopen(output_file, 'w'); assert(fid >= 0);
    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s\n', jsonencode(result));
    fprintf('SHARED_INDEX_PASS omega=%g\n', saved.resolved_problem_cfg.omega);
end
