function Ar = galerkin(A, space)
%GALERKIN Reduced matrix Psi'*A*Psi.
%   For a subdomain space the blocks Psi_i'*A(idx_i,idx_j)*Psi_j are contracted
%   directly, skipping subdomain pairs without coupling.
    if strcmp(space.type, 'dense')
        Ar = space.basis' * (A * space.basis);
        return;
    end
    Ar = zeros(space.k, space.k);
    offsets = [0, cumsum(space.ranks)];
    for i = 1:numel(space.basis)
        ri = offsets(i) + (1:space.ranks(i));
        if isempty(ri), continue; end
        for j = 1:numel(space.basis)
            rj = offsets(j) + (1:space.ranks(j));
            if isempty(rj), continue; end
            Aij = A(space.idx{i}, space.idx{j});
            if nnz(Aij) == 0, continue; end
            Ar(ri, rj) = space.basis{i}' * (Aij * space.basis{j});
        end
    end
end
