function run_3d_frozen_holdout_audit
%RUN_3D_FROZEN_HOLDOUT_AUDIT New parameters with the previously frozen models.
    rng(9052026,'twister');
    for name={'lens','waveguide'}
        spec=helmholtz_3d_40pct_config(name{1}); cfg=spec.driver_cfg;
        p=spec.problem_cfg; interval=p.parameter_range;
        interior=interval(1)+diff(interval)*rand(1,3);
        p.test_params=sort([interval(1)-0.02,interior,interval(2)+0.02]);
        assert(~any(ismember(p.test_params,[p.train_params,spec.problem_cfg.test_params])));
        cfg.frozen_training_file=helm_output_path('outputs','helmholtz_3d_candidates','large', ...
            sprintf('%s_%s_training.mat',spec.problem_name,cfg.output_tag));
        cfg.([spec.problem_name '_cfg'])=p;
        cfg.output_tag=[name{1} '_frozen_holdout_20260905'];
        run_helmholtz_3d_candidate_experiments('large',cfg);
    end
end
