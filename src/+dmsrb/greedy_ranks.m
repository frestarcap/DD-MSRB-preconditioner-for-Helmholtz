function ranks = greedy_ranks(svals, nrows, eta)
%GREEDY_RANKS Local ranks under one global snapshot-tail budget.
%   ranks = dmsrb.greedy_ranks(svals, nrows, eta) starts from zero ranks and
%   repeatedly adds the mode with the largest discarded energy per stored entry,
%   argmax_b sigma_{b,r_b+1}^2 / n_b, until the remaining energy over all
%   subdomains is at most eta^2 times the total snapshot energy.  svals{b} are the
%   singular values of the snapshot rows of subdomain b and nrows(b) its size.
%   Ties go to the lower subdomain number.
    count = numel(svals);
    total = 0;
    suffix = cell(1, count);
    remaining = zeros(1, count);
    for b = 1:count
        s2 = abs(svals{b}(:)).^2;
        total = total + sum(s2);
        suffix{b} = [flipud(cumsum(flipud(s2))); 0];
        remaining(b) = suffix{b}(1);
    end
    ranks = zeros(1, count);
    if total <= 1e-30, return; end
    target = eta^2 * total;
    tail2 = sum(remaining);
    while tail2 > target * (1 + 1e-12)
        score = -inf(1, count);
        for b = 1:count
            if ranks(b) < numel(svals{b})
                score(b) = abs(svals{b}(ranks(b) + 1))^2 / max(nrows(b), 1);
            end
        end
        [best, b] = max(score);
        assert(isfinite(best), 'No feasible ranks under the global tail budget.');
        ranks(b) = ranks(b) + 1;
        remaining(b) = suffix{b}(ranks(b) + 1);
        tail2 = sum(remaining);
    end
end
