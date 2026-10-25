function tests = test_halfspace_training
%TEST_HALFSPACE_TRAINING Experiment 1 training against the original code (L3).
%   (a) The refactored trainer reproduces the original trainer (commit 097edc8):
%       snapshots, global and local ranks, reduced operators.
%   (b) It reproduces the ranks of the frozen paper models.  Their training
%       (2026-09-05) used one-pass Gram--Schmidt without the residual check of
%       the recurrence, so the snapshots of iterations 2 and 3 differ slightly.
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    root = oracle();
    tc.assumeNotEmpty(root, 'DMSRB_ORACLE is not set.');
    tc.TestData.frozen = fullfile(root, 'outputs', 'halfspace_dd_refinement', 'review_repaired_20260905');
    tc.TestData.P = dmsrb.halfspace.problem();
    tc.TestData.params = linspace(0.8, 0.95, 24);
end

function test_global(tc)
    run_case(tc, 'global', [], 'global.mat');
end

function test_material_depth(tc)
    run_case(tc, 'dd', dmsrb.halfspace.partition(tc.TestData.P, 'material_depth'), 'slabs1-aligned-sym8.mat');
end

function run_case(tc, style, part, frozen_file)
    [model, info] = dmsrb.train(dmsrb.halfspace.family(tc.TestData.P), tc.TestData.params, 1e-5 * [1 1 1], part);
    % (a) original trainer of commit 097edc8
    cfg = struct('reference_mode', 'helm_lift', 'online_completion_mode', 'shifted_laplacian', ...
        'shifted_laplacian_solver', 'ilu', 'shifted_laplacian_ilu_droptol', 1e-3, ...
        'run_wave_baselines', false, 'use_direct_rom_init', false, 'build_direct_rom_when_unused', false, ...
        'train_params', tc.TestData.params, 'test_params', 0.8622, 'corr_num_layers', 3, ...
        'corr_pod_singular_value_tols', 1e-5 * [1 1 1], 'corr_basis_style', 'global', ...
        'do_plots', false, 'save_benchmark', false, 'parallel_train', false, 'parallel_test', false);
    if strcmp(style, 'dd'), cfg = halfspace_paper_dd_config(cfg); end
    old = helm_waveholtz_correction_rom(cfg);
    compare_layers(tc, model, info, old.correction_rom.layers, 1e-12, 'original code');
    % (b) frozen paper model
    s = load(fullfile(tc.TestData.frozen, frozen_file), 'artifact');
    compare_layers(tc, model, info, s.artifact.layers, [1e-12, Inf, Inf], 'paper model');
end

function compare_layers(tc, model, info, layers, snapshot_tol, label)
    if isscalar(snapshot_tol), snapshot_tol = snapshot_tol * [1 1 1]; end
    for j = 1:3
        l = layers{j};
        sp = l.basis_space;
        new = model.layers{j}.space;
        tc.verifyEqual(new.k, sp.k, sprintf('%s: rank of iteration %d', label, j));
        if strcmp(sp.type, 'blockn')
            tc.verifyEqual(new.ranks(:), [sp.blocks.k]', sprintf('%s: local ranks of iteration %d', label, j));
        end
        E = l.eta_snapshots;
        difference = norm(info.snapshots{j} - E, 'fro') / norm(E, 'fro');
        fprintf('%s, iteration %d: relative snapshot difference %.2e\n', label, j, difference);
        tc.verifyLessThan(difference, snapshot_tol(j), sprintf('%s: snapshots of iteration %d', label, j));
        if isfinite(snapshot_tol(j))
            D = phases(new, sp);
            tc.verifyLessThan(norm(D' * model.layers{j}.operators{1} * D - l.Ar_base, 'fro') / ...
                norm(l.Ar_base, 'fro'), 1e-10, sprintf('%s: reduced operator %d', label, j));
        end
    end
end

function D = phases(new, old)
    if strcmp(old.type, 'dense')
        V = {new.basis}; W = {old.basis};
    else
        V = new.basis; W = {old.blocks.basis};
    end
    d = [];
    for b = 1:numel(V)
        d = [d; diag(W{b}' * V{b})]; %#ok<AGROW>
    end
    D = diag(d ./ abs(d));
end
