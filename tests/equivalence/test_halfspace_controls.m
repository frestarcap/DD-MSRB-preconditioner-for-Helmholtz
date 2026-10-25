function tests = test_halfspace_controls
%TEST_HALFSPACE_CONTROLS NT-ROM and SS-ROM rebuilt from scratch equal the paper controls (L5).
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = oracle();
    tc.assumeNotEmpty(root, 'DMSRB_ORACLE is not set.');
    file = fullfile(root, 'outputs', 'jcp_shifted_20260914', 'frozen_cohorts.mat');
    tc.assumeTrue(isfile(file), 'Frozen cohorts are not available.');
    s = load(file, 'cohorts');
    tc.TestData.c = s.cohorts{1};
    tc.TestData.P = dmsrb.halfspace.problem();
end

function test_nt_rom(tc)
    c = tc.TestData.c; P = tc.TestData.P;
    % The equal-rank NT-ROM keeps 19 modes in iteration 3, although its snapshot
    % spectrum reaches roundoff (sigma_19 ~ 1e-18) there: those directions are not
    % reproducible, but they do not change the preconditioned iterates.
    model = dmsrb.train_nt(dmsrb.halfspace.family(P), c.train_parameters, [7 11 19]);
    for j = 1:2
        V = model.layers{j}.space.basis;
        W = c.models{3}.layers{j}.space.basis;
        tc.verifyLessThan(norm(V * V' - W * W', 'fro'), 1e-6, sprintf('NT-ROM space %d', j));
    end
    check_fields(tc, model, c.expected{3});
end

function test_ss_rom(tc)
    c = tc.TestData.c; P = tc.TestData.P;
    model = dmsrb.solution_pod(dmsrb.halfspace.family(P), c.train_parameters, 1e-5, 3);
    V = model.layers{1}.space.basis;
    W = c.models{4}.layers{1}.space.basis;
    tc.verifyEqual(size(V, 2), 7);
    tc.verifyLessThan(norm(V * V' - W * W', 'fro'), 1e-9, 'SS-ROM space');
end

function check_fields(tc, model, expected)
    c = tc.TestData.c; P = tc.TestData.P;
    for q = 1:numel(c.parameters)
        mu = c.parameters(q);
        u = dmsrb.prefix(dmsrb.halfspace.operator(P, mu), P.b, mu, model, 1e-12);
        tc.verifyLessThan(norm(u - expected(:, q)) / norm(expected(:, q)), 1e-9, sprintf('query %d', q));
    end
end
