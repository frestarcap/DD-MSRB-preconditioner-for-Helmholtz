function stats = mg_response(info, r, A)
%MG_RESPONSE Contraction of a residual by the components of one MG V-cycle.
%   For the pre-smoothing correction x_pre, the prolonged coarse correction z_c
%   and the post-smoothing correction z_post of one fine-grid V-cycle applied to
%   r, kappa = min_alpha ||r - alpha*A*z|| / ||r|| (A the unshifted Helmholtz
%   operator).  info is the second output of dmsrb.sl.mg.
    levels = info.levels; opts = info.options;
    level = levels{1};
    if numel(levels) == 1                          % no coarsening: one direct solve
        x = level.coarse_factor \ r;
        zero = zeros(size(r), 'like', r);
        stats = struct('pre', contraction(r, zero), 'coarse', contraction(r, A * x), ...
            'post', contraction(r, zero), 'combined', contraction(r, A * x));
        return;
    end
    x_pre = zeros(size(r), 'like', r);
    for sweep = 1:opts.mg_pre_sweeps
        x_pre = x_pre + opts.mg_jacobi_weight * level.diagonal_inverse .* (r - level.A * x_pre);
    end
    z_coarse = level.P * info.vcycle(2, level.P' * (r - level.A * x_pre));
    x = x_pre + z_coarse;
    after_coarse = x;
    for sweep = 1:opts.mg_post_sweeps
        x = x + opts.mg_jacobi_weight * level.diagonal_inverse .* (r - level.A * x);
    end
    z_post = x - after_coarse;
    stats = struct('pre', contraction(r, A * x_pre), 'coarse', contraction(r, A * z_coarse), ...
        'post', contraction(r, A * z_post), 'combined', contraction(r, A * x));
end

function kappa = contraction(r, image)
    if norm(image) == 0, kappa = 1; return; end
    u = image / norm(image);
    kappa = norm(r - (u' * r) * u) / norm(r);
end
