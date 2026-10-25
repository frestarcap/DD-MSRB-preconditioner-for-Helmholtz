function [model, info] = train(family, params, eta, part)
%TRAIN Trajectory-aware training of iteration-wise reduced spaces (Algorithm 1).
%   [model, info] = dmsrb.train(family, params, eta, part) builds one reduced
%   space per ROM-preconditioned FGMRES iteration j = 1..numel(eta) from the
%   training parameters params.  family.system(mu) returns [A, b]; affine
%   families also provide family.terms {A_1,...,A_Q} and family.theta(mu) with
%   A(mu) = sum_q theta_q(mu) A_q.  part is [] (global POD) or a partition.
%
%   Layer 1 compresses the exact inverse actions z_1 = A\b/norm(b).  For j > 1
%   every training trajectory is advanced with the compressed layer j-1:
%   z_rom = P_{j-1} q_{j-1}, and the exact inverse action of the next Arnoldi
%   vector q_j follows from the recurrence (dmsrb.verified_snapshot), so the
%   snapshots of layer j are taken on the trajectory that the compressed
%   preconditioners generate online.
    count = numel(params);
    depth = numel(eta);
    affine = isfield(family, 'terms');
    clock = tic;
    A = cell(1, count); Q = cell(1, count); Z = cell(1, count);
    timer = tic;
    for s = 1:count
        [A{s}, b] = family.system(params(s));
        u = A{s} \ b;
        nb = norm(b);
        Q{s} = b / nb;
        Z{s} = u / nb;
    end
    info = struct('solve_s', toc(timer), 'snapshot_s', 0, 'compress_s', 0, 'assembly_s', 0, ...
        'spectra', {cell(depth, 1)}, 'snapshots', {cell(depth, 1)}, 'fallbacks', 0);
    model = struct('layers', {cell(depth, 1)}, 'affine', affine);
    if affine, model.theta = family.theta; end
    for j = 1:depth
        E = zeros(size(Q{1}, 1), count, 'like', Z{1});
        timer = tic;
        for s = 1:count
            if j > 1
                q = Q{s}(:, j - 1);
                if norm(q) <= 1e-15
                    [Q{s}(:, j), Z{s}(:, j)] = deal(zeros(size(q), 'like', q));
                else
                    zrom = dmsrb.apply_rom(model.layers{j - 1}, A{s}, coefficients(model, params(s)), q);
                    [Q{s}(:, j), Z{s}(:, j), ~, ~, check] = dmsrb.verified_snapshot( ...
                        A{s}, zrom, Q{s}(:, 1:j - 1), Z{s}(:, 1:j - 1));
                    info.fallbacks = info.fallbacks + check.direct_fallback;
                end
            end
            E(:, s) = Z{s}(:, j);
        end
        info.snapshot_s = info.snapshot_s + toc(timer);
        info.snapshots{j} = E;
        timer = tic;
        [space, info.spectra{j}] = dmsrb.compress(E, part, eta(j));
        info.compress_s = info.compress_s + toc(timer);
        layer = struct('space', space);
        if affine
            timer = tic;
            layer.operators = cellfun(@(T) dmsrb.galerkin(T, space), family.terms, 'UniformOutput', false);
            info.assembly_s = info.assembly_s + toc(timer);
        end
        model.layers{j} = layer;
    end
    info.total_s = toc(clock);
end

function c = coefficients(model, mu)
    c = [];
    if model.affine, c = model.theta(mu); end
end
