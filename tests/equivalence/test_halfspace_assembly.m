function tests = test_halfspace_assembly
%TEST_HALFSPACE_ASSEMBLY Matrices, source and partitions equal the original code (L0, L1).
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    tc.assumeNotEmpty(oracle(), 'DMSRB_ORACLE is not set.');
end

function test_paper_grid(tc)
    check(tc, struct());
end

function test_small_grid(tc)
    check(tc, struct('Nphys', 25, 'omega', 12));
end

function check(tc, opts)
    P = dmsrb.halfspace.problem(opts);
    cfg = struct('setup_only', true, 'Nphys', P.Nphys, 'omega', P.omega, ...
        'do_plots', false, 'save_benchmark', false, 'parallel_train', false, 'parallel_test', false);
    r = helm_waveholtz_correction_rom(cfg);
    tc.verifyTrue(isequal(P.A_base, r.problem.A_base), 'A_base');
    tc.verifyTrue(isequal(P.A_right, r.problem.A_right), 'A_right');
    tc.verifyTrue(isequal(P.b, r.rhs), 'rhs');
    for mu = [0.81, 0.8622, 0.94]
        alpha = (1 / mu^2) - 1;
        tc.verifyTrue(isequal(dmsrb.halfspace.operator(P, mu), r.problem.A_base + alpha * r.problem.A_right));
    end
    old = struct('material_model', 'halfspace', 'interface_x', 0.5, 'Lx', 1, 'Ly', 1);
    old = halfspace_paper_dd_config(old);
    modes = {'material_depth', 'material_symmetric_depth'; 'signed_depth', 'signed_depth'; 'merged', 'merged'};
    for k = 1:size(modes, 1)
        old.halfspace_dd_pml_mode = modes{k, 2};
        ref = build_halfspace_refined_partition(r.problem, old);
        new = dmsrb.halfspace.partition(P, modes{k, 1});
        tc.verifyTrue(isequal({new.blocks.idx}, {ref.blocks.idx}), modes{k, 1});
    end
end
