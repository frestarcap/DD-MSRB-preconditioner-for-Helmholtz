function [model, info] = train_nt(family, params, ranks)
%TRAIN_NT Non-trajectory control (NT-ROM) with prescribed global ranks.
%   Layer j compresses the exact inverse actions A\q_j of the plain
%   (unpreconditioned) Arnoldi vectors q_1 = b/norm(b), q_{j+1} ~ A*q_j, so the
%   spaces ignore the trajectory that the compressed preconditioners produce.
%   ranks(j) fixes the dimension of layer j (the trajectory-aware global ranks).
    count = numel(params);
    clock = tic;
    A = cell(1, count); Q = cell(1, count);
    for s = 1:count
        [A{s}, b] = family.system(params(s));
        Q{s} = b / norm(b);
    end
    model = struct('layers', {cell(numel(ranks), 1)}, 'affine', true, 'theta', family.theta);
    for j = 1:numel(ranks)
        E = complex(zeros(size(Q{1}, 1), count));
        for s = 1:count
            q = Q{s}(:, j);
            if norm(q) > 1e-15, E(:, s) = A{s} \ q; end
        end
        [U, ~, ~] = svd(E, 'econ');
        space = struct('type', 'dense', 'n', size(E, 1), 'k', ranks(j), 'basis', U(:, 1:ranks(j)));
        model.layers{j} = struct('space', space, 'operators', ...
            {cellfun(@(T) dmsrb.galerkin(T, space), family.terms, 'UniformOutput', false)});
        for s = 1:count
            w = A{s} * Q{s}(:, j);
            for i = 1:size(Q{s}, 2)
                w = w - (Q{s}(:, i)' * w) * Q{s}(:, i);
            end
            next = zeros(size(w));
            if norm(w) > 1e-15, next = w / norm(w); end
            Q{s}(:, j + 1) = next;
        end
    end
    info = struct('total_s', toc(clock));
end
