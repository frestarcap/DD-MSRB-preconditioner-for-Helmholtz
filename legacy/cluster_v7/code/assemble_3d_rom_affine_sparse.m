function [Ar0, Ar1] = assemble_3d_rom_affine_sparse(problem, space)
%ASSEMBLE_3D_ROM_AFFINE_SPARSE Preserve every cross-block Galerkin coupling.
% DD bases are temporarily embedded sparsely, never as an N-by-total-rank
% dense matrix. The persistent ROM representation remains local/dense.
    if strcmp(space.type, 'dense')
        B = space.basis;
    else
        assert(strcmp(space.type, 'blockn'));
        nb = numel(space.blocks);
        rows = cell(nb, 1); cols = rows; values = rows;
        offset = 0;
        for b = 1:nb
            block = space.blocks(b);
            rows{b} = repmat(block.idx(:), block.k, 1);
            cols{b} = repelem((offset + (1:block.k)).', numel(block.idx));
            cols{b} = cols{b}(:); % Keep rank-zero blocks as 0-by-1 columns.
            values{b} = block.basis(:);
            offset = offset + block.k;
        end
        B = sparse(vertcat(rows{:}), vertcat(cols{:}), vertcat(values{:}), ...
            problem.N, space.k);
        clear rows cols values
    end
    Ar0 = full(B' * (problem.K_pml * B)) + ...
        problem.affine.mass_base * full(B' * B);
    if issparse(B)
        DB = spdiags(problem.affine.mass_parameter_diagonal, 0, ...
            problem.N, problem.N) * B;
    else
        DB = problem.affine.mass_parameter_diagonal .* B;
    end
    Ar1 = full(B' * DB);
end
