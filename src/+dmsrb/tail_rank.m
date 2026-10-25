function r = tail_rank(s, eta)
%TAIL_RANK Smallest rank whose relative Frobenius tail is at most eta.
%   r = dmsrb.tail_rank(s, eta) for singular values s returns the smallest
%   r >= 1 with sqrt(sum(s(r+1:end).^2)) <= eta*norm(s), or 0 for a zero matrix.
    total = norm(s);
    if total <= 1e-30
        r = 0;
        return;
    end
    tail = flipud(cumsum(flipud(abs(s(:)).^2)));
    tail_after = [tail(2:end); 0];
    r = find(sqrt(tail_after) / total <= eta, 1, 'first');
    if isempty(r), r = numel(s); end
end
