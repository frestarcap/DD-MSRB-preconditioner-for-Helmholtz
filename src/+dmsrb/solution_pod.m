function [model, info] = solution_pod(family, params, eta, depth)
%SOLUTION_POD Solution-snapshot control (SS-ROM): one POD space of the training
%   solutions u(mu_s) = A(mu_s)\b, truncated at relative tail eta and reused as
%   the reduced space of every one of the depth ROM-preconditioned iterations.
    clock = tic;
    S = [];
    for s = 1:numel(params)
        [A, b] = family.system(params(s));
        S(:, s) = A \ b; %#ok<AGROW>
    end
    [space, spectrum] = dmsrb.compress(S, [], eta);
    layer = struct('space', space, 'operators', ...
        {cellfun(@(T) dmsrb.galerkin(T, space), family.terms, 'UniformOutput', false)});
    model = struct('layers', {repmat({layer}, depth, 1)}, 'affine', true, 'theta', family.theta);
    info = struct('total_s', toc(clock), 'spectrum', spectrum);
end
