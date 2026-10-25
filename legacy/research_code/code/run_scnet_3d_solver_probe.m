function report=run_scnet_3d_solver_probe(snapshot_file, output_tag, options)
%RUN_SCNET_3D_SOLVER_PROBE Re-solve a retained difficult RHS without changing it.
% Eight concurrent exploratory beta/restart/coarse-grid combinations. Every result,
% including nonconvergence, is saved. Timings include concurrent contention.
    if nargin<2, output_tag='halfspace_w48_solver_probe'; end
    if nargin<3, options=struct(); end
    defaults=struct('workers',8,'threads',4,'iteration_budget',1600, ...
        'variants',[1.2 40 3000;1.2 80 3000;1.2 120 3000;2 80 3000; ...
                    1.2 40 12000;1.2 80 12000;2 40 12000;.8 80 12000]);
    for name=fieldnames(defaults).'
        if ~isfield(options,name{1}), options.(name{1})=defaults.(name{1}); end
    end
    loaded=load(snapshot_file,'payload'); cache=loaded.payload; clear loaded
    problem=helmholtz_3d_problem_interface(cache.problem_cfg);
    [A,~]=problem.assemble(cache.mu); rhs=cache.rhs;
    mass=problem.mass_diagonal(cache.mu);
    baseline=struct('artifact',snapshot_file,'mu',cache.mu, ...
        'shift_beta',cache.policy.shift_beta,'restart',cache.policy.snapshot_restart, ...
        'true_residual',norm(rhs-A*cache.snapshot)/max(norm(rhs),eps), ...
        'iterations',cache.info.iteration_count,'seconds',cache.info.total_time);
    cfg=cache.policy; problem_cfg=problem.cfg; clear cache
    if size(options.variants,2)==2
        options.variants(:,3)=cfg.mg_coarse_max_dofs;
    end
    assert(size(options.variants,2)==3 && all(options.variants(:)>0));
    data=struct('A',A,'rhs',rhs,'mass',mass,'grid_size',problem.n);
    out=helm_output_path('outputs','scnet_3d_20260909',output_tag);
    if ~isfolder(out), mkdir(out); end
    write_json(fullfile(out,'baseline.json'),baseline);
    backend='cgs2';
    if exist('helm_gmres_cgs2_mex','file')==3, backend='cgs2_mex'; end
    previous=maxNumCompThreads(1);
    restore=onCleanup(@() maxNumCompThreads(previous)); %#ok<NASGU>
    pool=helm_3d_parallel_pool(options.workers,options.threads);
    shared=parallel.pool.Constant(data);
    cleanup=onCleanup(@() delete(shared)); %#ok<NASGU>
    variants=options.variants; rows=cell(size(variants,1),1);
    timer=tic;
    parfor k=1:size(variants,1)
        rows{k}=solve_variant(shared.Value,cfg,problem_cfg,baseline,variants(k,:), ...
            options.iteration_budget,backend,out);
    end
    report=struct('baseline',baseline,'problem_cfg',problem_cfg,'N',problem.N, ...
        'options',options,'backend',backend,'job_id',getenv('SLURM_JOB_ID'), ...
        'rows',vertcat(rows{:}),'batch_seconds',toc(timer), ...
        'note','Concurrent exploratory timings; same retained RHS and unchanged true-residual tolerance.');
    report.passing_count=sum([report.rows.converged]);
    save(fullfile(out,'probe.mat'),'report');
    write_json(fullfile(out,'probe.json'),report);
    clear cleanup shared
    delete(pool);
    fprintf('SCNET_3D_SOLVER_PROBE_COMPLETE passed=%d/%d\n',report.passing_count,numel(rows));
end

function row=solve_variant(data,cfg,problem_cfg,baseline,variant,budget,backend,out)
    beta=variant(1); restart=variant(2); cycles=max(1,floor(budget/restart));
    row=struct('mu',baseline.mu,'shift_beta',beta,'restart',restart, ...
        'mg_coarse_max_dofs',variant(3),'level_count',NaN,'coarse_grid_size',[], ...
        'maximum_iterations',cycles*restart,'converged',false, ...
        'true_residual',NaN,'iterations',NaN,'flag',NaN, ...
        'setup_seconds',NaN,'solve_seconds',NaN,'seconds',NaN, ...
        'history',[],'error','');
    x=[]; rhs=data.rhs; timer=tic;
    try
        n=numel(rhs);
        shifted=data.A+1i*beta*spdiags(data.mass,0,n,n);
        cfg.mg_coarse_max_dofs=variant(3);
        [M,setup]=build_shifted_laplacian_multigrid_preconditioner(shifted,data.grid_size,cfg);
        opts=struct('restart',restart,'maxit',cycles,'tol',cfg.snapshot_tol, ...
            'orthogonalization',backend,'store_preconditioned_basis',false, ...
            'fixed_linear_preconditioner',true);
        [x,info]=shifted_laplacian_gmres(@(v)data.A*v,M,rhs,[],opts);
        row.setup_seconds=setup.setup_time; row.solve_seconds=info.total_time;
        row.level_count=setup.level_count; row.coarse_grid_size=setup.grid_sizes{end};
        row.true_residual=norm(rhs-data.A*x)/max(norm(rhs),eps);
        row.iterations=info.iter_count; row.flag=info.flag; row.history=info.relres_hist;
        row.converged=isfinite(row.true_residual) && row.true_residual<=cfg.snapshot_tol*(1+1e-6);
    catch ME
        row.error=getReport(ME,'extended','hyperlinks','off');
    end
    row.seconds=toc(timer);
    tag=sprintf('beta_%03d_restart_%03d_coarse_%05d',round(100*beta),restart,variant(3));
    save(fullfile(out,[tag '.mat']),'x','rhs','row','problem_cfg','-v7.3','-nocompression');
    write_json(fullfile(out,[tag '.json']),row);
    fprintf('SOLVER_PROBE beta=%.2f restart=%d coarse=%d it=%g true=%.3e time=%.2fs passed=%d\n', ...
        beta,restart,variant(3),row.iterations,row.true_residual,row.seconds,row.converged);
end

function write_json(file,data)
    temporary=[tempname(fileparts(file)) '.json'];
    fid=fopen(temporary,'w'); assert(fid>=0);
    fprintf(fid,'%s\n',jsonencode(data)); fclose(fid);
    [ok,message]=movefile(temporary,file,'f'); assert(ok,'%s',message);
end
