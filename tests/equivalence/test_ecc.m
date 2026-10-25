function tests = test_ecc
%TEST_ECC Experiment 2 against the original code and the frozen paper models.
%   L0: systems equal the stored test systems bitwise.  L1: partition.
%   L3: retrained models have the paper ranks (snapshots of iterations 2-3 are
%       roundoff sensitive, see test_training).  L4: frozen models
%   solved online give the published SL step counts.
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    % The paper runs of Experiment 2 used one computational thread; the reduced
    % systems are ill-conditioned enough for BLAS rounding to matter.
    tc.TestData.threads = maxNumCompThreads(1);
    root = oracle();
    tc.assumeNotEmpty(root, 'DMSRB_ORACLE is not set.');
    folder = fullfile(root, 'outputs', 'ecc_dd_budget_20260919');
    tc.assumeTrue(isfolder(folder), 'Frozen ECC models are not available.');
    tc.TestData.folder = folder;
    tc.TestData.P = dmsrb.ecc.problem();
    tc.TestData.params = exp2_params();
    v = load(fullfile(folder, 'validate_results.mat'), 'c');
    tc.TestData.c = v.c;
end

function teardownOnce(tc)
    maxNumCompThreads(tc.TestData.threads);
end

function test_systems(tc)
    P = tc.TestData.P; c = tc.TestData.c; p = tc.TestData.params;
    tc.verifyEqual(p.test, c.parameters(:).');
    for q = 1:numel(p.test)
        [A, b] = dmsrb.ecc.system(P, p.test(q));
        tc.verifyTrue(isequal(A, c.systems{q}.A) && isequal(b, c.systems{q}.b), sprintf('system %d', q));
    end
    tc.verifyTrue(all(arrayfun(@(mu) dmsrb.ecc.admissible(P, mu), [p.train, p.test])));
end

function test_training(tc)
    P = tc.TestData.P; p = tc.TestData.params;
    part = dmsrb.ecc.partition(P, p.train);
    for style = {'global', 'dd'}
        s = load(fullfile(tc.TestData.folder, ['uniform5_' style{1} '.mat']), 'model', 'snapshots', 'cfg');
        tc.verifyEqual(s.cfg.ratio_train(:).', p.train);
        sp = [];
        if strcmp(style{1}, 'dd')
            sp = part;
            old = s.model.layers{1}.space.partition;
            tc.verifyTrue(isequal({part.blocks.idx}, {old.idx_near, old.idx_far, old.idx_pml}), 'partition');
        end
        [model, info] = dmsrb.train(dmsrb.ecc.family(P), p.train, 1e-5 * [1 1 1], sp);
        for j = 1:3
            old = convert_space(s.model.layers{j}.space);
            new = model.layers{j}.space;
            tc.verifyEqual(new.k, old.k, sprintf('%s rank %d', style{1}, j));
            if strcmp(style{1}, 'dd'), tc.verifyEqual(new.ranks(:), old.ranks(:)); end
            E = s.snapshots{j};
            difference = norm(info.snapshots{j} - E, 'fro') / norm(E, 'fro');
            fprintf('%s iteration %d: relative snapshot difference %.2e\n', style{1}, j, difference);
            % Iterations 2 and 3 amplify roundoff through ill-conditioned reduced
            % systems: the original code itself reproduces its 2026-09-20 snapshots
            % only to 2e-7 and 4e-4.  The ranks are robust.
            tolerance = [1e-12, 1e-5, 1e-2];
            tc.verifyLessThan(difference, tolerance(j));
        end
    end
end

function test_training_original_code(tc)
    % The refactored trainer reproduces the original trainer (commit 097edc8) bitwise.
    s = load(fullfile(oracle(), 'outputs', 'ecc_shifted_tuning_20260919', 'global_n96_d3_eta1e-06.mat'), 'cfg');
    cfg = s.cfg; eta = 1e-5 * [1 1 1];
    cfg.corr_rank_rtol = []; cfg.corr_rank_rtols = eta; cfg.corr_block3_global_tail_tols = eta;
    cfg.corr_block3_global_tail_strategy = 'marginal_storage';
    for key = {'near', 'far', 'pml'}
        cfg.(['corr_block3_tail_tol_' key{1}]) = []; cfg.(['corr_block3_rtol_' key{1}]) = [];
        cfg.(['corr_block3_rank_cap_' key{1}]) = Inf;
    end
    cfg.corr_num_layers = 3; cfg.corr_normalize_snapshots = false;
    cfg.corr_pod_truncation_rule = 'relative_frobenius_tail'; cfg.helm_gmres_track_tol = 1e-12;
    cfg.shifted_laplacian_tol = 1e-12; cfg.parallel_train = false; cfg.parallel_test = false;
    cfg.do_plots = false; cfg.save_plots = false; cfg.save_benchmark = false;
    cfg.ratio_test = cfg.ratio_test(1); cfg.ratio_test_auto_generated = false;
    P = tc.TestData.P; p = tc.TestData.params;
    for style = {'global', 'block3'}
        cfg.corr_basis_style = style{1};
        old = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
        part = [];
        if strcmp(style{1}, 'block3'), part = dmsrb.ecc.partition(P, p.train); end
        [model, info] = dmsrb.train(dmsrb.ecc.family(P), p.train, eta, part);
        for j = 1:3
            tc.verifyEqual(info.snapshots{j}, old.correction_rom.layers{j}.eta_snapshots, sprintf('%s %d', style{1}, j));
            tc.verifyEqual(model.layers{j}.space.k, old.correction_rom.layers{j}.space_dim);
        end
    end
end

function test_online(tc)
    P = tc.TestData.P; c = tc.TestData.c;
    t = readtable(fullfile(tc.TestData.folder, 'uniform5_timings.csv'), 'TextType', 'string');
    t = t(t.repeat == 1, :);
    sl = struct('tol', 1e-12, 'precond', @(A, mu) dmsrb.sl.ilu(A + 1i * 0.2 * spdiags( ...
        dmsrb.ecc.sl_mass(P, mu), 0, P.N, P.N), 1e-3), 'gmres', struct('tol', 1e-12, 'restart', 40, 'maxit', 20));
    names = {'uniform5_global', 'uniform5_dd', 'shifted'};
    models = cell(1, 3);
    for m = 1:2
        s = load(fullfile(tc.TestData.folder, [names{m} '.mat']), 'model');
        models{m} = convert_model(s.model, []);
    end
    for q = 1:numel(c.parameters)
        A = c.systems{q}.A; b = c.systems{q}.b;
        for m = 1:3
            [u, info] = dmsrb.solve(A, b, c.parameters(q), models{m}, sl);
            row = t(t.method == names{m} & t.query == q, :);
            tc.verifyEqual(info.sl_steps, row.iterations, sprintf('%s query %d', names{m}, q));
            tc.verifyLessThan(abs(log10(info.rho_rom / row.post)), 1e-2, sprintf('rho_rom %s %d', names{m}, q));
            tc.verifyLessThanOrEqual(info.rho, 1e-12);
            tc.verifyLessThan(norm(u - c.references(:, q)) / norm(c.references(:, q)), 1e-7);
        end
    end
end
