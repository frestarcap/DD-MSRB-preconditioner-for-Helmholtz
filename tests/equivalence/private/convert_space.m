function sp = convert_space(old)
%CONVERT_SPACE Original reduced space (dense, blockn or block3) to dmsrb layout.
    switch old.type
        case 'dense'
            sp = struct('type', 'dense', 'n', size(old.basis, 1), 'k', size(old.basis, 2), 'basis', old.basis);
        case 'blockn'
            sp = struct('type', 'blocks', 'n', old.n, 'k', old.k, 'ranks', [old.blocks.k], ...
                'idx', {{old.blocks.idx}}, 'basis', {{old.blocks.basis}});
        case 'block3'
            p = old.partition;
            basis = {old.Un, old.Uf, old.Up};
            sp = struct('type', 'blocks', 'n', old.n, 'k', old.k, 'ranks', cellfun(@(B) size(B, 2), basis), ...
                'idx', {{p.idx_near, p.idx_far, p.idx_pml}}, 'basis', {basis});
    end
end
