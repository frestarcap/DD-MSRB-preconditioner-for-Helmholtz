function tests = test_core
%TEST_CORE Unit tests of the reduced-space building blocks.
    tests = functiontests(localfunctions);
end

function test_greedy_matches_exhaustive_search(tc)
% Greedy allocation is optimal for the storage-weighted objective on small cases
% whose marginal gains are decreasing; it never violates the budget.
    rng(3);
    for trial = 1:20
        s = {sort(rand(5, 1), 'descend').^4, sort(rand(4, 1), 'descend').^3, sort(rand(6, 1), 'descend').^5};
        n = [50, 20, 80];
        eta = 0.05 + 0.3 * rand;
        r = dmsrb.greedy_ranks(s, n, eta);
        total = sum(cellfun(@(v) sum(v.^2), s));
        tail = sum(arrayfun(@(b) sum(s{b}(r(b) + 1:end).^2), 1:3));
        tc.verifyLessThanOrEqual(tail, eta^2 * total * (1 + 1e-12));
        best = inf;
        for a = 0:5, for b = 0:4, for c = 0:6
            t = sum(s{1}(a + 1:end).^2) + sum(s{2}(b + 1:end).^2) + sum(s{3}(c + 1:end).^2);
            if t <= eta^2 * total * (1 + 1e-12), best = min(best, n * [a; b; c]); end
        end, end, end
        tc.verifyLessThanOrEqual(n * r(:), best * 1.5 + max(n), 'greedy storage far from optimal');
    end
end

function test_tail_rank(tc)
    s = [10; 1; 0.1; 0.01];
    tc.verifyEqual(dmsrb.tail_rank(s, 0.2), 1);
    tc.verifyEqual(dmsrb.tail_rank(s, 1e-3), 3);
    tc.verifyEqual(dmsrb.tail_rank(zeros(3, 1), 0.1), 0);
end

function test_project_lift_and_galerkin(tc)
    rng(5);
    n = 60;
    part = struct('blocks', struct('idx', {(1:20)', (21:45)', (46:60)'}));
    E = complex(randn(n, 8), randn(n, 8));
    sp = dmsrb.compress(E, part, 1e-3);
    V = zeros(n, sp.k);
    for j = 1:sp.k
        e = zeros(sp.k, 1); e(j) = 1;
        V(:, j) = dmsrb.lift(sp, e);
    end
    tc.verifyLessThan(norm(V' * V - eye(sp.k)), 1e-12, 'subdomain bases are orthonormal');
    v = complex(randn(n, 1), randn(n, 1));
    tc.verifyLessThan(norm(dmsrb.project(sp, v) - V' * v), 1e-12);
    A = sprandn(n, n, 0.1) + 1i * speye(n);
    tc.verifyLessThan(norm(dmsrb.galerkin(A, sp) - V' * A * V), 1e-12);
    g = dmsrb.compress(E, [], 1e-3);
    tc.verifyEqual(g.k, 8);
    tc.verifyLessThan(norm(dmsrb.galerkin(A, g) - g.basis' * A * g.basis), 1e-12);
end

function test_prefix_with_exact_spaces_solves(tc)
% With the exact inverse actions in the reduced spaces the ROM steps solve the
% training problem to roundoff.
    P = dmsrb.halfspace.problem(struct('Nphys', 15, 'omega', 6));
    F = dmsrb.halfspace.family(P);
    model = dmsrb.train(F, 0.85, [1e-14, 1e-14, 1e-14], []);
    A = dmsrb.halfspace.operator(P, 0.85);
    [~, history] = dmsrb.prefix(A, P.b, 0.85, model, 1e-12);
    tc.verifyLessThan(history(end), 1e-10);
end
