function tests = test_appendixA
%TEST_APPENDIXA Residual diagnostics of Appendix A against the original code.
%   MG component contractions and transverse roughness of a fixed residual on the
%   omega = 8 waveguide grid (two multigrid levels).
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    tc.assumeNotEmpty(oracle(), 'DMSRB_ORACLE is not set.');
end

function test_probe_and_roughness(tc)
    omega = 8; mu = 0.317;
    c = exp3_config(omega);
    P = dmsrb.waveguide.problem(omega);
    A = dmsrb.waveguide.operator(P, mu);
    Ashift = A + 1i * c.beta * spdiags(dmsrb.waveguide.mass(P, mu), 0, P.N, P.N);
    rng(11);
    r = P.b + 1e-3 * complex(randn(P.N, 1), randn(P.N, 1));
    [~, info] = dmsrb.sl.mg(Ashift, P.n, c.mg);
    tc.verifyGreaterThan(info.level_count, 1);
    new = mg_response(info, r, A);
    roughness = transverse_roughness(P, r);
    opts = c.mg; opts.solver = 'mg'; opts.grid_size = P.n;
    [~, probe] = build_dd_mechanism_mg_probe(Ashift, P.n, opts);
    old = probe(r, @(v) A * v);
    tc.verifyEqual([new.pre, new.coarse, new.combined], ...
        [old.component_optimal_contraction(1:2), old.combined_optimal_contraction], 'RelTol', 1e-10);
    spec = helmholtz_3d_40pct_config('waveguide');
    g = spec.problem_cfg;
    g.Nphys = P.Nphys; g.npml = P.npml; g.omega = omega;
    problem = helmholtz_3d_problem_interface(g);
    m = dd_mechanism_component_metrics(r, problem, []);
    tc.verifyEqual(roughness, m.roughness_transverse_over_omega_squared, 'RelTol', 1e-10);
end
