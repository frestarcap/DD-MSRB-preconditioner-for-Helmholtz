function results = run_scnet_3d_experiment(action, omega, workers, threads, problem_name, solver_options)
%RUN_SCNET_3D_EXPERIMENT Headless, fixed-resolution-per-wavelength 3D study.
    if nargin<1, action='full'; end
    if nargin<2, omega=24; end
    if nargin<3, workers=8; end
    if nargin<4, threads=1; end
    if nargin<5, problem_name='waveguide'; end
    if nargin<6, solver_options=struct(); end
    assert(ismember(problem_name,{'waveguide','halfspace'}));
    solver=struct('shift_beta',1.2,'snapshot_restart',40,'snapshot_maxit',40, ...
        'gmres_restart',40,'gmres_maxit',40,'mg_coarse_max_dofs',3000);
    for name=fieldnames(solver_options).'
        assert(isfield(solver,name{1}),'Unsupported solver option %s.',name{1});
        value=solver_options.(name{1});
        validateattributes(value,{'numeric'},{'scalar','finite','positive'});
        if ~strcmp(name{1},'shift_beta'), assert(value==fix(value)); end
        solver.(name{1})=value;
    end
    allocation=str2double(getenv('SLURM_CPUS_PER_TASK'));
    if ~isfinite(allocation), allocation=workers*threads+1; end
    % Memory-bound sparse kernels rarely benefit from all 64 BLAS threads.
    maxNumCompThreads(min(8,allocation));
    tag=sprintf('%s_w%g_p%d_t%d',action,omega,workers,threads);
    if strcmp(problem_name,'halfspace')
        tag=sprintf('%s_halfspace_w%g_p%d_t%d',action,omega,workers,threads);
    end
    if ~isempty(fieldnames(solver_options))
        tag=sprintf('%s_b%g_sr%d_or%d_sm%d_om%d_c%d',tag,solver.shift_beta, ...
            solver.snapshot_restart,solver.gmres_restart,solver.snapshot_maxit,solver.gmres_maxit, ...
            solver.mg_coarse_max_dofs);
    end
    out=helm_output_path('outputs','scnet_3d_20260909',tag);
    if ~isfolder(out), mkdir(out); end
    metadata=struct('job_id',getenv('SLURM_JOB_ID'),'node',getenv('SLURMD_NODENAME'), ...
        'cpus',allocation,'workers',workers,'worker_threads',threads, ...
        'matlab',version,'created_at',datestr(now,30));
    save(fullfile(out,'metadata.mat'),'metadata');
    fid=fopen(fullfile(out,'metadata.json'),'w'); fprintf(fid,'%s\n',jsonencode(metadata)); fclose(fid);
    if strcmp(action,'verify')
        test_helm_3d_hpc;
        test_helmholtz_3d_problem_interface;
        test_shifted_laplacian_multigrid_3d;
        omega=6;
    end
    spec=helmholtz_3d_40pct_config('waveguide');
    cfg=spec.driver_cfg;
    scale=omega/24;
    geom=cfg.gaussian_waveguide_cfg;
    geom.Nphys=round([144 48 48]*scale)-1;
    geom.npml=max(3,round([9 9 9]*scale));
    geom.omega=omega;
    if strcmp(problem_name,'halfspace')
        geom.material_model='affine_halfspace';
        geom.domain_lengths=[1 1 1]; geom.interface_x=.5;
        geom.Nphys=repmat(round(48*scale)-1,1,3);
        geom.source_location=[.2 .5 .5];
        cfg.problem_names={'affine_halfspace'};
        cfg.physical_axial_sections=3;
    end
    % Preserve physical PML thickness (up to integer rounding); source spans
    % 1.5 cells, as in the reference. Report both choices in the run notes.
    cfg.train_count=8;
    geom.train_params=linspace(.2,.6,cfg.train_count);
    geom.test_params=[.237 .317 .483];
    cfg.num_layers=3;
    cfg.layer_tail_tolerances=[.003 .008 .02];
    for name=fieldnames(solver).', cfg.(name{1})=solver.(name{1}); end
    cfg.require_snapshot_tolerance=true;
    cfg.hpc_mode=true;
    cfg.parallel_train=true;
    cfg.parallel_workers=workers; cfg.worker_threads=threads;
    cfg.gmres_orthogonalization='cgs2';
    if exist('helm_gmres_cgs2_mex','file')==3
        cfg.gmres_orthogonalization='cgs2_mex';
    end
    cfg.store_preconditioned_basis=false;
    cfg.fast_reduced_assembly=true;
    cfg.reuse_first_layer=true;
    cfg.make_plots=false;
    cfg.checkpoint_training=true;
    cfg.cache_snapshots=true;
    cfg.embed_full_models=false;
    cfg.save_candidate_checkpoints=false;
    cfg.output_dir=out; cfg.output_tag=tag;
    cfg.gaussian_waveguide_cfg=geom;
    if strcmp(action,'verify')
        cfg.train_count=4; cfg.gaussian_waveguide_cfg.train_params=linspace(.2,.6,4);
        cfg.gaussian_waveguide_cfg.test_params=[.317 .483];
    end
    if strcmp(problem_name,'halfspace')
        cfg.affine_halfspace_cfg=cfg.gaussian_waveguide_cfg;
        cfg=rmfield(cfg,'gaussian_waveguide_cfg');
    end
    n=prod(geom.Nphys+2*geom.npml);
    % Conservative planning estimate, not measured RSS. Includes MATLAB
    % runtimes, Krylov basis, Galerkin MG temporaries and client ROM storage.
    per_worker=2*2^30+n*(16*(cfg.snapshot_restart+1)+1600);
    client=4*2^30+n*16*(cfg.train_count*(cfg.num_layers+5)+150);
    estimated=workers*per_worker+client;
    configuration=struct('cfg',cfg,'N',n,'estimated_peak_GiB',estimated/2^30);
    fid=fopen(fullfile(out,'configuration.json'),'w');
    fprintf(fid,'%s\n',jsonencode(configuration)); fclose(fid);
    fprintf('Memory planning envelope %.1f GiB; N=%d\n',estimated/2^30,n);
    mem_mib=str2double(getenv('SLURM_MEM_PER_NODE'));
    if isfinite(mem_mib)
        assert(estimated < .85*mem_mib*2^20,'Memory planning envelope exceeds 85%% of job limit.');
    end
    timer=tic;
    results=run_helmholtz_3d_candidate_experiments('large',cfg);
    c=results.candidates{1};
    report=struct('problem_name',c.name,'omega',omega,'N',n,'ppw',c.problem_summary.ppw_minimum, ...
        'elapsed_seconds',toc(timer),'metadata',metadata, ...
        'global_basis_MiB',c.models.global.basis_bytes/2^20, ...
        'dd_basis_MiB',c.models.dd.basis_bytes/2^20, ...
        'global_persistent_MiB',c.models.global.persistent_bytes/2^20, ...
        'dd_persistent_MiB',c.models.dd.persistent_bytes/2^20);
    report.basis_saving_percent=100*(1-report.dd_basis_MiB/report.global_basis_MiB);
    report.orthogonalization=cfg.gmres_orthogonalization;
    report.solver_config=solver;
    report.training_resumed=c.models.global.resumed || c.models.dd.resumed;
    report.snapshot_max_relres=max([c.models.global.snapshot_relative_residual(:); ...
        c.models.dd.snapshot_relative_residual(:)]);
    test_rows=cell(numel(c.tests),1);
    for k=1:numel(c.tests)
        q=c.tests(k); row=struct('mu',q.mu);
        for key={'shifted','global','dd'}
            s=q.(key{1});
            row.(key{1})=struct('true_residual',s.relative_residual,'flag',s.flag, ...
                'iterations',s.iterations,'seconds',s.total_time, ...
                'setup_seconds',s.setup_time,'solve_seconds',s.solve_time, ...
                'history',s.residual_history);
            if ~strcmp(key{1},'shifted')
                row.(key{1}).rom_residual=s.precorrection_relative_residual;
                row.(key{1}).rom_seconds=s.precorrection_time;
                row.(key{1}).rom_history=s.precorrection_residual_history;
            end
        end
        test_rows{k}=row;
    end
    report.tests=vertcat(test_rows{:});
    final_r=[];
    for k=1:numel(report.tests)
        for key={'shifted','global','dd'}
            final_r(end+1)=report.tests(k).(key{1}).true_residual; %#ok<AGROW>
        end
    end
    report.all_converged=all(isfinite(final_r) & final_r<=cfg.gmres_tol*(1+1e-6));
    mid=ceil(c.geometry.grid_size(3)/2);
    field=reshape(c.tests(2).dd.solution,c.geometry.grid_size);
    reference=reshape(c.tests(2).shifted.solution,c.geometry.grid_size);
    report.field_slice=struct('mu',c.tests(2).mu,'z',c.geometry.coordinates{3}(mid), ...
        'x',c.geometry.coordinates{1},'y',c.geometry.coordinates{2}, ...
        'real',real(field(:,:,mid)),'amplitude',abs(field(:,:,mid)), ...
        'relative_difference_to_shifted',norm(field(:)-reference(:))/norm(reference(:)));
    fid=fopen(fullfile(out,'report.json'),'w'); fprintf(fid,'%s\n',jsonencode(report)); fclose(fid);
    save(fullfile(out,'report.mat'),'report');
    assert(report.all_converged,'Online residual acceptance failed; all results retained.');
    fprintf('SCNET_3D_EXPERIMENT_PASS omega=%g N=%d saving=%.2f%%\n', ...
        omega,n,report.basis_saving_percent);
end
