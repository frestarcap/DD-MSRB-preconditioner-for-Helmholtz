function report=test_helm_3d_failure_collection()
%TEST_HELM_3D_FAILURE_COLLECTION Preserve every numerically failed parameter.
    spec=helmholtz_3d_40pct_config('waveguide'); cfg=spec.driver_cfg;
    cfg.gaussian_waveguide_cfg.Nphys=[35 11 11];
    cfg.gaussian_waveguide_cfg.npml=[3 3 3];
    cfg.gaussian_waveguide_cfg.omega=6;
    cfg.gaussian_waveguide_cfg.train_params=linspace(.2,.6,4);
    cfg.train_count=4; cfg.parallel_train=true; cfg.hpc_mode=true;
    cfg.parallel_workers=2; cfg.worker_threads=1;
    cfg.snapshot_restart=1; cfg.snapshot_maxit=1;
    cfg.require_snapshot_tolerance=true; cfg.cache_snapshots=true;
    cfg.checkpoint_training=true; cfg.make_plots=false;
    cfg.gmres_orthogonalization='cgs2'; cfg.store_preconditioned_basis=false;
    root=helm_output_path('outputs','scnet_3d_20260909','failure_collection_tests');
    if ~isfolder(root), mkdir(root); end
    cfg.output_dir=tempname(root); mkdir(cfg.output_dir);
    cfg.output_tag='all_failures'; failed=false;
    try
        run_helmholtz_3d_candidate_experiments('large',cfg);
    catch ME
        if strcmp(ME.identifier,'helm:SnapshotTolerance'), failed=true;
        else, rethrow(ME); end
    end
    pool=gcp('nocreate'); if ~isempty(pool), delete(pool); end
    assert(failed,'An underresolved inverse action must not create a ROM layer.');
    files=dir(fullfile(cfg.output_dir,'snapshot_cache','gaussian_waveguide','global','layer_01','*_failed.mat'));
    assert(numel(files)==4,'All four failed parameter artifacts must survive.');
    assert(isempty(dir(fullfile(cfg.output_dir,'*_progress.mat'))));
    parameters=zeros(1,4); residuals=parameters;
    for k=1:numel(files)
        saved=load(fullfile(files(k).folder,files(k).name),'payload'); s=saved.payload;
        p=helmholtz_3d_problem_interface(s.problem_cfg);
        [A,~]=p.assemble(s.mu);
        rho=norm(s.rhs-A*s.snapshot)/norm(s.rhs);
        assert(rho>cfg.snapshot_tol && abs(rho-s.info.relative_residual)<1e-12);
        parameters(k)=s.mu; residuals(k)=rho;
    end
    assert(max(abs(sort(parameters)-cfg.gaussian_waveguide_cfg.train_params))<1e-14);
    report=struct('parameters',parameters,'failed_residuals',residuals,'output_dir',cfg.output_dir);
    save(fullfile(cfg.output_dir,'failure_collection_verification.mat'),'report');
    fprintf('HELM_3D_FAILURE_COLLECTION_PASS count=%d\n',numel(files));
end
