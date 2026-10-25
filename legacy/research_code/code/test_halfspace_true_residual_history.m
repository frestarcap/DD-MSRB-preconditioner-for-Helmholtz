function test_halfspace_true_residual_history
% Verify that recording does not change the fixed-point trajectory.
    cfg=struct('Nphys',15,'npml',4,'omega',8, ...
        'train_params',[0.8,0.85,0.9,0.95],'test_params',0.873, ...
        'corr_basis_style','global','reference_mode','helm_lift', ...
        'online_wave_solver','fixedpoint','run_nr_rom_comparison',false, ...
        'run_wave_baselines',true,'use_direct_rom_init',false, ...
        'build_direct_rom_when_unused',false,'fp_maxit',4,'fp_tol',1e-14, ...
        'online_helm_accept_tol',[],'do_plots',false,'save_plots',false, ...
        'save_benchmark',false,'parallel_train',false,'parallel_test',false);
    off=helm_waveholtz_correction_rom(cfg);
    cfg.record_true_helm_history=true;
    on=helm_waveholtz_correction_rom(cfg,off);
    assert(isequal(off.test.online_fp_relres_hist,on.test.online_fp_relres_hist));
    assert(isequal(off.test.plain_fp_relres_hist,on.test.plain_fp_relres_hist));
    assert(isequal(off.test.online_final_x,on.test.online_final_x));
    assert(isequal(off.test.online_final_u,on.test.online_final_u));
    A=on.problem.A_base+(cfg.test_params^(-2)-1)*on.problem.A_right;
    t=on.test;
    h=t.online_true_helm_relres_hist{1};
    assert(numel(h)==t.online_fp_iter(1)+1 && all(isfinite(h)));
    assert(abs(h(end)-norm(on.rhs-A*t.online_final_u)/norm(on.rhs))<1e-14);
    p=t.plain_true_helm_relres_hist{1};
    assert(numel(p)==t.plain_fp_iter(1)+1 && all(isfinite(p)));
    assert(abs(p(end)-norm(on.rhs-A*t.plain_fp_u)/norm(on.rhs))<1e-14);
    % A truncated rerun at each prefix independently recovers that state.
    for steps=1:3
        prefix=cfg; prefix.fp_maxit=steps;
        q=helm_waveholtz_correction_rom(prefix,off);
        rho=norm(on.rhs-A*q.test.online_final_u)/norm(on.rhs);
        assert(abs(rho-h(steps+1))<1e-11*max(1,rho));
    end
    disp('PASS: unchanged native trajectories; true histories match independent recovery.');
end
