function run_jsc_final_checks
%RUN_JSC_FINAL_CHECKS Freeze holdouts, rebuild publication data, verify helpers.
    root=helm_output_path('outputs','revision_20260907');
    rng(9072026,'twister'); parameters=[.13,sort(.15+.4*rand(1,3)),.57];
    save(fullfile(root,'deployment','lens_primary_holdout_parameters.mat'),'parameters');
    for method={'global','dd'}
        base=fullfile(root,'deployment',['lens_tight_' method{1}]);
        source=[base '.mat']; target=[base '_holdout.mat'];
        copyfile(source,target);
        run_jsc_3d_deployment(target,'lazy',parameters,1);
    end
    run_jsc_waveguide_diagnostics;
    export_jsc_paper_data;
    audit_revision_evidence('revision20260907');
    write_revision_audit_tables('revision20260907');
    plot_halfspace_dd_paper;
    redraw_paper_output_figures('pc');
    plot_helmholtz_3d_paper_figures;
    test_jsc_revision_helpers;
    test_rom_verified_snapshot_step;
    names={'helm_waveholtz_correction_rom','waveholtz_2d_pml_scattering_ecc_correction_rom', ...
        'photonic_crystal_waveguide_correction_rom'};
    for k=1:numel(names)
        [files,products]=matlab.codetools.requiredFilesAndProducts(which(names{k}));
        fprintf('Dependency check %s: %d files, %d products\n',names{k},numel(files),numel(products));
    end
end
