function tests = test_waveguide
%TEST_WAVEGUIDE Experiment 3 pipeline against the code that ran on the cluster (L6).
%   The cluster source (v7) and the refactored code train and solve the waveguide
%   on reduced grids (omega = 6 and 12 at the paper's points per wavelength).
%   Partitions, ranks, reduced operators, ROM residuals and SL step counts agree.
%   Set DMSRB_V7 to the v7 source tree (containing code/).
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    v7 = getenv('DMSRB_V7');
    tc.assumeTrue(~isempty(v7) && isfolder(fullfile(v7, 'code')), 'DMSRB_V7 is not set.');
    addpath(fullfile(v7, 'code'), '-begin');
    tc.TestData.v7 = v7;
    tc.TestData.out = tempname;
    mkdir(tc.TestData.out);
end

function teardownOnce(tc)
    rmpath(fullfile(tc.TestData.v7, 'code'));
    rmdir(tc.TestData.out, 's');
end

function test_omega6(tc)
    compare(tc, 6);
end

function test_omega12(tc)
    compare(tc, 12);
end

function compare(tc, omega)
    old = run_v7(tc, omega);
    c = exp3_config(omega);
    c.workers = 1; c.orthogonalization = 'cgs2';
    P = dmsrb.waveguide.problem(omega);
    part = dmsrb.waveguide.partition(P);
    tc.verifyTrue(isequal({part.blocks.idx}, {old.partition.blocks.idx}), 'partition');
    tc.verifyEqual(P.b, old.rhs, 'source');
    first = [];
    for style = {'global', 'dd'}
        sp = [];
        if strcmp(style{1}, 'dd'), sp = part; end
        [model, ~, first] = train_waveguide(P, c, sp, first, fullfile(tc.TestData.out, sprintf('w%d_%s.mat', omega, style{1})));
        v = old.models.(style{1});
        for j = 1:3
            new = model.layers{j}.space; ref = convert_space(v.layers{j}.space);
            tc.verifyEqual(new.k, ref.k, sprintf('%s rank %d', style{1}, j));
            if strcmp(style{1}, 'dd'), tc.verifyEqual(new.ranks(:), ref.ranks(:)); end
            D = phases(new, ref);
            Ar0 = v.layers{j}.Ar0; Ar1 = v.layers{j}.Ar1;
            tc.verifyLessThan(norm(D' * model.layers{j}.operators{1} * D - Ar0, 'fro') / norm(Ar0, 'fro'), 1e-9);
            tc.verifyLessThan(norm(D' * model.layers{j}.operators{2} * D - Ar1, 'fro') / norm(Ar1, 'fro'), 1e-9);
        end
        models.(style{1}) = model;
    end
    for k = 1:numel(old.tests)
        t = old.tests(k);
        A = dmsrb.waveguide.operator(P, t.mu);
        shift = 1i * c.beta * spdiags(dmsrb.waveguide.mass(P, t.mu), 0, P.N, P.N);
        sl = struct('tol', c.tol, 'precond', @(A, ~) dmsrb.sl.mg(A + shift, P.n, c.mg), ...
            'gmres', waveguide_gmres_options(c, c.tol, c.maxit));
        for style = {'shifted', 'global', 'dd'}
            model = [];
            if ~strcmp(style{1}, 'shifted'), model = models.(style{1}); end
            [~, info] = dmsrb.solve(A, P.b, t.mu, model, sl);
            ref = t.(style{1});
            tc.verifyEqual(info.sl_steps, ref.iterations, sprintf('omega %d mu %.3f %s SL steps', omega, t.mu, style{1}));
            if ~strcmp(style{1}, 'shifted')
                tc.verifyLessThan(abs(info.rho_rom / ref.precorrection_relative_residual - 1), 1e-6, ...
                    sprintf('omega %d mu %.3f %s rho_ROM', omega, t.mu, style{1}));
            end
        end
    end
end

function r = run_v7(tc, omega)
    spec = helmholtz_3d_40pct_config('waveguide');
    cfg = spec.driver_cfg;
    scale = omega / 24;
    geom = cfg.gaussian_waveguide_cfg;
    geom.Nphys = round([144 48 48] * scale) - 1;
    geom.npml = max(3, round([9 9 9] * scale));
    geom.omega = omega;
    cfg.train_count = 8;
    geom.train_params = linspace(.2, .6, 8);
    geom.test_params = [.237 .317 .483];
    cfg.num_layers = 3;
    cfg.layer_tail_tolerances = [.003 .008 .02];
    solver = struct('shift_beta', 2, 'snapshot_restart', 40, 'snapshot_maxit', 40, ...
        'gmres_restart', 40, 'gmres_maxit', 40, 'mg_coarse_max_dofs', 12000);
    for name = fieldnames(solver).', cfg.(name{1}) = solver.(name{1}); end
    cfg.require_snapshot_tolerance = true; cfg.hpc_mode = true; cfg.parallel_train = false;
    cfg.gmres_orthogonalization = 'cgs2'; cfg.store_preconditioned_basis = false;
    cfg.fast_reduced_assembly = true; cfg.reuse_first_layer = true; cfg.make_plots = false;
    cfg.checkpoint_training = false; cfg.cache_snapshots = false; cfg.embed_full_models = true;
    cfg.save_models = true; cfg.save_candidate_checkpoints = false;
    cfg.output_dir = tc.TestData.out; cfg.output_tag = sprintf('l6_w%d', omega);
    cfg.gaussian_waveguide_cfg = geom;
    results = run_helmholtz_3d_candidate_experiments('large', cfg);
    r = results.candidates{1};
    r.tests = arrayfun(@(t) t, r.tests);
end

function D = phases(new, ref)
    if strcmp(ref.type, 'dense')
        V = {new.basis}; W = {ref.basis};
    else
        V = new.basis; W = ref.basis;
    end
    d = [];
    for b = 1:numel(V), d = [d; diag(W{b}' * V{b})]; end %#ok<AGROW>
    D = diag(d ./ abs(d));
end
