function tests = test_halfspace_online
%TEST_HALFSPACE_ONLINE The paper's frozen Experiment 1 models, solved online with
%   the refactored code (L4/L5): ROM fields equal the stored fields and the SL step
%   counts equal those of the published timing run, for all 23 test queries.
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = oracle();
    tc.assumeNotEmpty(root, 'DMSRB_ORACLE is not set.');
    file = fullfile(root, 'outputs', 'jcp_shifted_20260914', 'frozen_cohorts.mat');
    tc.assumeTrue(isfile(file), 'Frozen cohorts are not available.');
    s = load(file, 'cohorts');
    tc.TestData.c = s.cohorts{1};
    tc.TestData.paper = readtable(fullfile(root, 'outputs', 'jcp_shifted_20261022', 'paper_timings.csv'), ...
        'TextType', 'string');
end

function test_prefix_fields_and_sl_steps(tc)
    c = tc.TestData.c;
    P = dmsrb.halfspace.problem();
    tc.verifyTrue(isequal(P.A_base, c.A_base) && isequal(P.A_right, c.A_right) && isequal(P.b, c.b));
    theta = P.theta;
    models = cellfun(@(m) convert_model(m, theta), c.models(1:3), 'UniformOutput', false);
    ss = convert_model(c.models{4}, theta);                % SS-ROM: one space reused three times
    ss.layers = repmat(ss.layers(1), 3, 1);
    models{4} = ss;
    sl = struct('tol', 1e-12, 'precond', @(A, mu) dmsrb.sl.ilu(A + 1i * 0.2 * spdiags( ...
        dmsrb.halfspace.sl_mass(P, mu), 0, P.N, P.N), 1e-3), ...
        'gmres', struct('tol', 1e-12, 'restart', 40, 'maxit', 20));
    paper = tc.TestData.paper(tc.TestData.paper.problem == "halfspace" & tc.TestData.paper.repeat == 1, :);
    names = [c.names, {'shifted'}];
    for q = 1:numel(c.parameters)
        mu = c.parameters(q);
        A = dmsrb.halfspace.operator(P, mu);
        for m = 1:numel(names)
            if m <= 3
                u = dmsrb.prefix(A, P.b, mu, models{m}, 1e-12);
                old = c.expected{m}(:, q);
                tc.verifyLessThan(norm(u - old) / norm(old), 1e-9, sprintf('%s query %d', names{m}, q));
            end
            model = [];
            if m <= 4, model = models{m}; end
            [~, info] = dmsrb.solve(A, P.b, mu, model, sl);
            row = paper(paper.method == names{m} & paper.query == q, :);
            tc.verifyEqual(info.sl_steps, row.completion_iterations, sprintf('SL steps %s query %d', names{m}, q));
            tc.verifyLessThanOrEqual(info.rho, 1e-12);
        end
    end
end
