function report=test_helm_3d_checkpoint()
%TEST_HELM_3D_CHECKPOINT Simulate interruption before and after a layer save.
    previous=maxNumCompThreads(2);
    cleanup=onCleanup(@() maxNumCompThreads(previous)); %#ok<NASGU>
    root=helm_output_path('outputs','scnet_3d_20260909','checkpoint_tests');
    if ~isfolder(root), mkdir(root); end
    out=tempname(root); mkdir(out);
    spec=helmholtz_3d_40pct_config('waveguide'); cfg=spec.driver_cfg;
    cfg.gaussian_waveguide_cfg.Nphys=[35 11 11];
    cfg.gaussian_waveguide_cfg.npml=[3 3 3];
    cfg.gaussian_waveguide_cfg.omega=6;
    cfg.gaussian_waveguide_cfg.train_params=linspace(.2,.6,4);
    cfg.train_count=4; cfg.num_layers=3; cfg.layer_tail_tolerances=[.003 .008 .02];
    cfg.parallel_train=false; cfg.make_plots=false;
    cfg.snapshot_maxit=40; cfg.require_snapshot_tolerance=true;
    cfg.fast_reduced_assembly=true; cfg.gmres_orthogonalization='cgs2';
    cfg.store_preconditioned_basis=false; cfg.reuse_first_layer=true;
    cfg.embed_full_models=false; cfg.save_candidate_checkpoints=false;
    cfg.checkpoint_training=true; cfg.cache_snapshots=true;
    cfg.checkpoint_stop_after_layer=1;
    cfg.output_dir=out; cfg.output_tag='checkpoint_test';
    expect_interruption(cfg);
    progress=fullfile(out,'gaussian_waveguide_checkpoint_test_global_progress.mat');
    assert(isfile(progress));
    cache=fullfile(out,'snapshot_cache','gaussian_waveguide','global','layer_01','sample_001.mat');
    stamp=dir(cache); assert(~isempty(stamp));
    % Remove only the layer marker: the next attempt must reuse the accepted
    % per-parameter snapshots, as after a mid-layer scheduler interruption.
    movefile(progress,[progress '.held']);
    expect_interruption(cfg);
    after=dir(cache); assert(stamp.datenum==after.datenum && stamp.bytes==after.bytes);
    cfg.checkpoint_stop_after_layer=0;
    resumed=run_helmholtz_3d_candidate_experiments('large',cfg);
    assert(resumed.candidates{1}.models.global.resumed);
    reference_cfg=cfg; reference_cfg.checkpoint_training=false; reference_cfg.cache_snapshots=false;
    reference_cfg.output_dir=fullfile(out,'fresh');
    fresh=run_helmholtz_3d_candidate_experiments('large',reference_cfg);
    errors=[];
    for method={'global','dd'}
        a=resumed.candidates{1}; b=fresh.candidates{1};
        assert(isequal(a.models.(method{1}).layer_ranks,b.models.(method{1}).layer_ranks));
        for k=1:numel(a.tests)
            x=a.tests(k).(method{1}).solution; y=b.tests(k).(method{1}).solution;
            errors(end+1)=norm(x-y)/norm(y); %#ok<AGROW>
            assert(a.tests(k).(method{1}).relative_residual<=1.000001e-6);
        end
    end
    assert(max(errors)<1e-8);
    report=struct('solution_differences',errors,'cache_reused',true,'output_dir',out);
    save(fullfile(out,'checkpoint_verification.mat'),'report');
    disp(report); fprintf('HELM_3D_CHECKPOINT_PASS\n');
end

function expect_interruption(cfg)
    stopped=false;
    try
        run_helmholtz_3d_candidate_experiments('large',cfg);
    catch ME
        if strcmp(ME.identifier,'helm:CheckpointTestStop'), stopped=true;
        else, rethrow(ME); end
    end
    assert(stopped,'Expected an intentional checkpoint interruption.');
end
