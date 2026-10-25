function result = test_ecc_vectorized_assembly()
%TEST_ECC_VECTORIZED_ASSEMBLY Compare vectorized and legacy ECC systems.

    base = struct();
    base.setup_only = true;
    base.Nphys = 45;
    base.npml = 12;
    base.ratio_train = [0.94, 1.06];
    base.ratio_test = 1.0;
    base.parallel_train = false;
    base.parallel_test = false;
    base.do_plots = false;
    base.save_plots = false;
    base.save_benchmark = false;

    cfg_vec = base;
    cfg_vec.vectorized_scattering_assembly = true;
    vectorized = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg_vec);

    cfg_legacy = base;
    cfg_legacy.vectorized_scattering_assembly = false;
    legacy = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg_legacy);

    A_relerr = norm(vectorized.system.A - legacy.system.A, 'fro') ...
        / max(norm(legacy.system.A, 'fro'), eps);
    b_relerr = norm(vectorized.system.b - legacy.system.b) ...
        / max(norm(legacy.system.b), eps);
    assert(isequal(spones(vectorized.system.A), spones(legacy.system.A)), ...
        'ECC sparsity patterns differ.');
    assert(A_relerr <= 20 * eps, 'ECC matrix relative error is %.3e.', A_relerr);
    assert(b_relerr <= 20 * eps, 'ECC RHS relative error is %.3e.', b_relerr);

    result = struct('A_relative_error', A_relerr, 'b_relative_error', b_relerr, ...
        'vectorized_time', vectorized.assembly_time, ...
        'legacy_time', legacy.assembly_time, ...
        'speedup', legacy.assembly_time / vectorized.assembly_time);
    fprintf(['ECC assembly A relerr %.3e, b relerr %.3e, vectorized %.4fs, ' ...
        'legacy %.4fs, speedup %.1fx\n'], A_relerr, b_relerr, ...
        result.vectorized_time, result.legacy_time, result.speedup);
end
