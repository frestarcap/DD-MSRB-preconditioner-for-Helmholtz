function validate_shared_index_deployment(reference_file, output_dir)
%VALIDATE_SHARED_INDEX_DEPLOYMENT Save/load and complete an existing query.
% Reference is an existing legacy deployment; numerical settings are unchanged.
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    legacy = load(reference_file);
    compact = legacy;
    compact.model = build_jsc_3d_deployment_model(legacy.model);
    assert(isfield(compact.model, 'shared_block_indices'));
    oldfile = fullfile(output_dir, 'legacy_dd.mat');
    newfile = fullfile(output_dir, 'shared_dd.mat');
    save(oldfile, '-struct', 'legacy', '-v7.3');
    save(newfile, '-struct', 'compact', '-v7.3');
    restored = load(newfile, 'model');
    assert(isequaln(restored.model, compact.model));
    clear legacy compact restored
    old = run_jsc_3d_deployment(oldfile, 'lazy', 0.317, 1, true);
    new = run_jsc_3d_deployment(newfile, 'lazy', 0.317, 1, true);
    fields = {'parameter','completion_iterations','precompletion_residual','true_residual'};
    assert(isequaln(old(:,fields), new(:,fields)), 'Online numerical results changed.');
    olddiag = readtable(fullfile(output_dir, 'legacy_dd_lazy_diagnostics.csv'));
    newdiag = readtable(fullfile(output_dir, 'shared_dd_lazy_diagnostics.csv'));
    assert(isequaln(olddiag, newdiag), 'Layer diagnostics changed.');
    % Check that unequal partitions are never silently merged.
    toy.blocks = struct('idx', {[1;2], [3;4]}, 'basis', {eye(2), eye(2)}, ...
        'k', {2,2}, 'meta', {[],[]});
    toy.type = 'blockn'; toy.n = 4; toy.k = 4;
    layer = struct('space', toy, 'Ar0', eye(4), 'Ar1', eye(4), 'rank', 4);
    original = struct('layers', {{layer,layer}});
    original.layers{2}.space.blocks(1).idx = [1;3];
    original.layers{2}.space.blocks(2).idx = [2;4];
    fallback = build_jsc_3d_deployment_model(original);
    assert(~isfield(fallback, 'shared_block_indices'));
    assert(isequaln(fallback, build_jsc_3d_deployment_model(original, 'per_layer')));
    fprintf('SHARED_INDEX_DEPLOYMENT_PASS: save/load, complete query, layer diagnostics, unequal-partition fallback\n');
end
