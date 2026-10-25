function v = lift(space, c)
%LIFT Full-grid vector Psi*c of reduced coefficients c.
    if strcmp(space.type, 'dense')
        v = space.basis * c;
        return;
    end
    v = zeros(space.n, 1, 'like', c);
    offset = 0;
    for b = 1:numel(space.basis)
        kb = space.ranks(b);
        if kb > 0
            v(space.idx{b}) = space.basis{b} * c(offset + (1:kb));
            offset = offset + kb;
        end
    end
end
