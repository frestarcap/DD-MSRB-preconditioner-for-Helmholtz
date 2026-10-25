function fixed = run_halfspace_fixed_pod_residual
% Interpolation-only fixed solution-POD comparison for Figure 5.
    out=helm_output_path('outputs','halfspace_dd_paper');
    d=load(fullfile(out,'true_residual_representative.mat'),'diagnostic'); d=d.diagnostic;
    source=helm_output_path('outputs','revision_20260907','halfspace','fixedpod_n24_eta-5.mat');
    s=load(source,'r'); p=s.r; clear s;
    assert(isequal(p.cfg.train_params,d.cfg.train_params));
    assert(d.parameter>min(p.cfg.train_params) && d.parameter<max(p.cfg.train_params));
    cfg=p.cfg; cfg.test_params=d.parameter; cfg.plot_test_index=1;
    cfg.online_completion_mode='waveholtz'; cfg.online_wave_solver='fixedpoint';
    cfg.fp_tol=d.cfg.fp_tol; cfg.fp_maxit=d.cfg.fp_maxit;
    cfg.fp_step_tol=d.cfg.fp_step_tol; cfg.online_helm_accept_tol=1e-12;
    cfg.record_true_helm_history=true;
    cfg.run_wave_baselines=false; cfg.run_nr_rom_comparison=false;
    cfg.do_plots=false; cfg.save_plots=false; cfg.save_benchmark=false;
    cfg.parallel_train=false; cfg.parallel_test=false;
    r=helm_waveholtz_correction_rom(cfg,p);
    A=r.problem.A_base+(d.parameter^(-2)-1)*r.problem.A_right;
    rho=norm(r.rhs-A*r.test.online_final_u(:,1))/norm(r.rhs);
    assert(rho<=1e-12);
    if r.test.online_fp_iter(1)>0
        assert(abs(rho-r.test.online_true_helm_relres_hist{1}(end))<1e-14);
    end
    fixed=struct('test',r.test,'parameter',d.parameter,'tail',1e-5, ...
        'rank',r.correction_rom.layers{1}.space_dim,'source',source,'true_residual',rho);
    save(fullfile(out,'fixed_pod_residual_interpolation.mat'),'fixed','-v7.3');
    fprintf('Fixed POD rank %d, initial residual %.6g, completion %d, final %.6g\n', ...
        fixed.rank,r.test.online_precorr_relres_final,r.test.online_fp_iter,rho);
    plot_halfspace_true_residual_figure(d);
end
