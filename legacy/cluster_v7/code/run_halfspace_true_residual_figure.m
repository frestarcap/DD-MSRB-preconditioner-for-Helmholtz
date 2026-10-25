function diagnostic = run_halfspace_true_residual_figure(mode)
%RUN_HALFSPACE_TRUE_RESIDUAL_FIGURE Recompute only the representative query.
% Saved paper timing batches are not overwritten by this diagnostic run.
    if nargin<1, mode='run'; end
    out=helm_output_path('outputs','halfspace_dd_paper');
    file=fullfile(out,'true_residual_representative.mat');
    if strcmp(mode,'redraw')
        s=load(file,'diagnostic'); diagnostic=s.diagnostic;
    else
        s=load(fullfile(out,'paper_data.mat'),'paper'); paper=s.paper;
        [~,k]=min(abs(paper.global.test.params-0.8622));
        cfg=paper.aux.cfg;
        cfg.test_params=paper.global.test.params(k);
        cfg.plot_test_index=1;
        cfg.record_true_helm_history=true;
        cfg.do_plots=false; cfg.save_plots=false; cfg.save_benchmark=false;
        cfg.parallel_train=false; cfg.parallel_test=false;
        r=helm_waveholtz_correction_rom(cfg);
        % Same native trajectory and final fields as the published batch.
        nr=r.nr_comparison.test;
        old=paper.aux.nr;
        assert(nr.online_fp_iter(1)==old.online_fp_iter(k));
        assert(norm(nr.online_fp_relres_hist{1}-old.online_fp_relres_hist{k},inf)<1e-11);
        assert(norm(nr.online_final_u(:,1)-old.online_final_u(:,k))/norm(old.online_final_u(:,k))<1e-10);
        A=r.problem.A_base+(cfg.test_params^(-2)-1)*r.problem.A_right;
        rho=norm(r.rhs-A*nr.online_final_u(:,1))/norm(r.rhs);
        assert(abs(rho-nr.online_true_helm_relres_hist{1}(end))<1e-14);
        rho_plain=norm(r.rhs-A*r.test.plain_fp_u(:,1))/norm(r.rhs);
        assert(abs(rho_plain-r.test.plain_true_helm_relres_hist{1}(end))<1e-14);
        diagnostic=struct('global',r.test,'nr',nr,'dd',paper.dd.test, ...
            'dd_index',k,'parameter',cfg.test_params,'cfg',r.cfg, ...
            'source','paper_data.mat; representative fixedpoint replay', ...
            'timing',r.timing);
        save(file,'diagnostic','-v7.3');
    end
    plot_halfspace_true_residual_figure(diagnostic);
end
