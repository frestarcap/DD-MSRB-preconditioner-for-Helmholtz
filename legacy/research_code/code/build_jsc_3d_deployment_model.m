function model = build_jsc_3d_deployment_model(original, index_layout)
%BUILD_JSC_3D_DEPLOYMENT_MODEL Retain online arrays, sharing a fixed partition.
% Shared indices are ordinary numeric cells, not handles. whos reports logical
% storage; this representation does not claim a change in unique physical RAM.
% The legacy layout is available for reproducible before/after measurements.
    if nargin < 2, index_layout = 'shared'; end
    assert(ismember(index_layout, {'shared','per_layer'}));
    layers = cell(size(original.layers));
    for j = 1:numel(layers)
        l = original.layers{j}; sp = l.space;
        if strcmp(sp.type, 'dense')
            sp = rom_reduced_space_utils('make_dense_space', sp.basis);
        else
            assert(strcmp(sp.type, 'blockn'), 'Expected dense or blockn deployment.');
            sp = rmfield(sp, intersect(fieldnames(sp), {'partition','meta','block_meta'}));
            for b = 1:numel(sp.blocks), sp.blocks(b).meta = []; end
        end
        layers{j} = struct('space', sp, 'Ar0', l.Ar0, 'Ar1', l.Ar1, 'rank', l.rank);
    end
    model = struct('layers', {layers}, 'num_layers', numel(layers));
    if isempty(layers) || strcmp(index_layout, 'per_layer') || ...
            ~all(cellfun(@(l) strcmp(l.space.type, 'blockn'), layers))
        return;
    end
    indices = {layers{1}.space.blocks.idx};
    for j = 2:numel(layers)
        % Different layer partitions must retain their own maps.
        if layers{j}.space.n ~= layers{1}.space.n || ...
                ~isequal(indices, {layers{j}.space.blocks.idx})
            return;
        end
    end
    model.shared_block_indices = indices;
    for j = 1:numel(layers)
        model.layers{j}.space.blocks = rmfield(model.layers{j}.space.blocks, 'idx');
    end
end
