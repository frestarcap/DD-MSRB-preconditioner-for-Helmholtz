function c = project(space, v)
%PROJECT Coefficients Psi'*v of v in a reduced space (subdomain-wise for 'blocks').
    if strcmp(space.type, 'dense')
        c = space.basis' * v;
        return;
    end
    c = zeros(space.k, 1);
    offset = 0;
    for b = 1:numel(space.basis)
        kb = space.ranks(b);
        if kb > 0
            c(offset + (1:kb)) = space.basis{b}' * v(space.idx{b});
            offset = offset + kb;
        end
    end
end
