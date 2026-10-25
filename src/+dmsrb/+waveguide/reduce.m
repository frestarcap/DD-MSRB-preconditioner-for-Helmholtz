function operators = reduce(P, space)
%REDUCE Affine reduced operators {Psi'*(K + omega^2 I)*Psi, Psi'*diag(omega^2 g)*Psi}.
%   A subdomain basis is embedded as a sparse N-by-k matrix, so all couplings
%   between subdomains are kept without forming a dense N-by-k basis.
    if strcmp(space.type, 'dense')
        B = space.basis;
    else
        nb = numel(space.basis);
        rows = cell(nb, 1); cols = rows; values = rows;
        offset = 0;
        for b = 1:nb
            k = space.ranks(b);
            rows{b} = repmat(space.idx{b}(:), k, 1);
            cols{b} = repelem((offset + (1:k)).', numel(space.idx{b}));
            cols{b} = cols{b}(:);
            values{b} = space.basis{b}(:);
            offset = offset + k;
        end
        B = sparse(vertcat(rows{:}), vertcat(cols{:}), vertcat(values{:}), P.N, space.k);
    end
    Ar0 = full(B' * (P.K * B)) + P.mass_base * full(B' * B);
    if issparse(B)
        DB = spdiags(P.mass_parameter, 0, P.N, P.N) * B;
    else
        DB = P.mass_parameter .* B;
    end
    operators = {Ar0, full(B' * DB)};
end
