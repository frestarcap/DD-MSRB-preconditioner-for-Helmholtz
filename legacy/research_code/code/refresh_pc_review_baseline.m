function refresh_pc_review_baseline
%REFRESH_PC_REVIEW_BASELINE A shared, converged baseline for both repaired ROMs.
    root=helm_output_path('outputs','helm_trajectory_evidence','review_repaired_20260905');
    file=fullfile(root,'pc_residual_ablation_paper.mat');
    s=load(file,'global_results','dd_results','base_cfg');
    baseline_file=fullfile(root,'pc_shared_converged_baseline.mat');
    if isfile(baseline_file)
        saved=load(baseline_file,'baseline'); baseline=saved.baseline;
    else
        cfg=s.global_results.cfg;
        cfg.corr_num_layers=0; cfg.corr_pod_singular_value_tols=[];
        cfg.corr_block3_global_tail_tols=[]; cfg.corr_basis_style='global';
        cfg.build_direct_rom_when_unused=false; cfg.use_direct_rom_init=false;
        cfg.wh_gmres_maxit=6; cfg.run_wave_baselines=true;
        cfg.do_plots=false; cfg.save_benchmark=false;
        r=helm_waveholtz_correction_rom(cfg);
        baseline=r.test;
        save(baseline_file,'baseline','cfg','-v7.3');
    end
    assert(isequal(baseline.params,s.global_results.test.params));
    assert(max(baseline.plain_fp_relres_final)<=1e-6);
    for name={'global_results','dd_results'}
        for field=fieldnames(baseline)'
            if startsWith(field{1},'plain_')
                s.(name{1}).test.(field{1})=baseline.(field{1});
            end
        end
    end
    s.baseline_provenance=baseline_file;
    save(file,'-struct','s','-v7.3');
    fprintf('Shared PC baseline converged at all %d parameters.\n',numel(baseline.params));
end
