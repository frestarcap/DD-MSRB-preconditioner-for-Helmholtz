function [space, spectra] = compress(E, part, eta)
%COMPRESS Reduced space of the snapshot columns E with relative tail eta.
%   [space, spectra] = dmsrb.compress(E, [], eta) returns the global POD basis
%   (type 'dense') of the smallest rank with relative Frobenius tail <= eta, and
%   an empty space when the snapshots vanish (a converged trajectory).
%   [space, spectra] = dmsrb.compress(E, part, eta) compresses the rows of each
%   subdomain part.blocks(b).idx separately and allocates the local ranks with
%   dmsrb.greedy_ranks (type 'blocks').  spectra holds the singular values (one
%   cell per subdomain for 'blocks').
    if isempty(part)
        [U, S, ~] = svd(E, 'econ');
        spectra = diag(S);
        r = 0;
        if norm(E, 'fro') > 1e-15, r = max(dmsrb.tail_rank(spectra, eta), 1); end
        space = struct('type', 'dense', 'n', size(E, 1), 'k', r, 'basis', U(:, 1:r));
        return;
    end
    count = numel(part.blocks);
    idx = {part.blocks.idx};
    U = cell(1, count);
    spectra = cell(1, count);
    for b = 1:count
        [U{b}, S, ~] = svd(E(idx{b}, :), 'econ');
        spectra{b} = diag(S);
    end
    ranks = dmsrb.greedy_ranks(spectra, cellfun(@numel, idx), eta);
    basis = cell(1, count);
    for b = 1:count
        basis{b} = U{b}(:, 1:ranks(b));
    end
    space = struct('type', 'blocks', 'n', size(E, 1), 'k', sum(ranks), ...
        'ranks', ranks, 'idx', {idx}, 'basis', {basis});
end
