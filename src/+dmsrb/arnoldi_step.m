function [q, h, hn, breakdown] = arnoldi_step(w, Q)
%ARNOLDI_STEP Orthogonalize w against the columns of Q (two-pass MGS).
%   Near breakdown is declared when the remainder is below 100*eps relative to
%   the norm of w.
    scale = norm(w);
    h = zeros(size(Q, 2), 1, 'like', w);
    for pass = 1:2
        for i = 1:size(Q, 2)
            dh = Q(:, i)' * w;
            h(i) = h(i) + dh;
            w = w - Q(:, i) * dh;
        end
    end
    hn = norm(w);
    breakdown = ~isfinite(hn) || hn <= 100 * eps * max(scale, realmin);
    q = zeros(size(w), 'like', w);
    if ~breakdown, q = w / hn; end
end
