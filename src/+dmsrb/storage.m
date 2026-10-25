function s = storage(model)
%STORAGE Basis, deployed payload and MATLAB object size of a model in MiB.
%   basis   = 16 bytes per complex basis entry;
%   payload = basis + affine reduced operators (16 bytes per entry) + one index map
%             of 8 bytes per grid node for a subdomain model (shared by all layers);
%   object  = whos size of the deployed model (dmsrb.deploy), with per-layer indices.
    basis = 0; operators = 0; index = 0;
    ranks = zeros(1, numel(model.layers));
    local = cell(1, numel(model.layers));
    for j = 1:numel(model.layers)
        l = model.layers{j};
        sp = l.space;
        ranks(j) = sp.k;
        if strcmp(sp.type, 'dense')
            basis = basis + 16 * numel(sp.basis);
        else
            basis = basis + 16 * sum(cellfun(@numel, sp.basis));
            index = max(index, 8 * sum(cellfun(@numel, sp.idx)));
            local{j} = sp.ranks;
        end
        if isfield(l, 'operators')
            operators = operators + 16 * sum(cellfun(@numel, l.operators));
        end
    end
    deployed = dmsrb.deploy(model); %#ok<NASGU>
    w = whos('deployed');
    MiB = 2^20;
    s = struct('ranks', ranks, 'local_ranks', {local}, 'basis_MiB', basis / MiB, ...
        'operators_MiB', operators / MiB, 'index_MiB', index / MiB, ...
        'payload_MiB', (basis + operators + index) / MiB, 'object_MiB', w.bytes / MiB);
end
