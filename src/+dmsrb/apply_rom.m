function z = apply_rom(layer, A, theta, q)
%APPLY_ROM Reduced inverse action z = Psi*(Psi'*A*Psi)^{-1}*Psi'*q of one layer.
%   Affine layers (field 'operators') assemble the reduced matrix from the
%   precomputed terms, sum_q theta(q)*Ar_q with theta = model.theta(mu), and solve
%   with backslash.  Otherwise (theta = []) the reduced matrix is assembled from A,
%   and an ill-conditioned reduced system (rcond < 1e-12) is solved by pinv.
    sp = layer.space;
    if sp.k == 0
        z = zeros(size(q), 'like', q);
        return;
    end
    br = dmsrb.project(sp, q);
    if isfield(layer, 'operators')
        Ar = layer.operators{1};
        for t = 2:numel(theta)
            Ar = Ar + theta(t) * layer.operators{t};
        end
        coeff = Ar \ br;
    else
        Ar = full(dmsrb.galerkin(A, sp));
        if rcond(Ar) < 1e-12
            coeff = pinv(Ar) * br;
        else
            coeff = Ar \ br;
        end
    end
    z = dmsrb.lift(sp, coeff);
end
