function test_jsc_revision_helpers
%TEST_JSC_REVISION_HELPERS Complex bound, zero RHS, partition and frozen reuse.
    rng(7); A=randn(12)+1i*randn(12)+6*eye(12); b=randn(12,1);
    spaces=cell(1,3); B=spaces;
    for j=1:3
        [U,~]=qr(randn(12,5)+1i*randn(12,5),0);
        spaces{j}=rom_reduced_space_utils('make_dense_space',U); B{j}=U'*A*U;
    end
    [~,d]=rom_true_action_diagnostics(A,b,spaces,B);
    assert(all(d.FGMRES_relative<=d.product_bound+1e-12));
    [x,d0]=rom_true_action_diagnostics(A,zeros(12,1),spaces,B);
    assert(norm(x)==0 && isempty(d0));
    full=rom_reduced_space_utils('make_dense_space',eye(12));
    [x,df]=rom_true_action_diagnostics(A,b,{full},{A});
    assert(norm(b-A*x)/norm(b)<1e-12 && height(df)==1);

    setup=photonic_crystal_waveguide_correction_rom(struct('setup_only',true, ...
        'pc_points_per_a',6,'do_plots',false,'save_benchmark',false));
    for sections=[1,3]
        for signed=[false,true]
            cfg=setup.cfg; cfg.pc_dd_axial_sections=sections; cfg.pc_dd_signed_pml=signed;
            part=build_pc_refined_partition(setup.problem,cfg);
            ids=vertcat(part.blocks.idx);
            assert(isequal(sort(ids),(1:setup.problem.Ntot).'));
        end
    end
    cfg=struct('Nphys',25,'npml',6,'omega',8,'train_params',linspace(.8,.95,6), ...
        'test_params',.873,'reference_mode','helm_lift','corr_basis_style','global', ...
        'corr_num_layers',2,'corr_pod_singular_value_tols',[1e-4,1e-4], ...
        'use_direct_rom_init',false,'build_direct_rom_when_unused',false, ...
        'run_wave_baselines',false,'run_nr_rom_comparison',false, ...
        'do_plots',false,'save_plots',false,'save_benchmark',false, ...
        'parallel_train',false,'parallel_test',false, ...
        'online_completion_mode','shifted_laplacian','online_helm_accept_tol',1e-9, ...
        'shifted_laplacian_tol',1e-9);
    r=helm_waveholtz_correction_rom(cfg); frozen=helm_waveholtz_correction_rom(r.cfg,r);
    assert(frozen.timing.offline==0);
    assert(norm(r.test.online_final_u-frozen.test.online_final_u)<1e-10);
    changed=r.cfg; changed.omega=9;
    rejected=false;
    try, helm_waveholtz_correction_rom(changed,r); catch e
        rejected=contains(e.message,'Frozen ROM');
    end
    assert(rejected,'Frozen model must reject a different full-order problem.');
    fprintf('JSC revision numerical helper tests passed.\n');
end
