function report = run_scnet_3d_scaling()
%RUN_SCNET_3D_SCALING Same eight inverse actions, vary concurrency/threads.
    pool=helm_3d_parallel_pool(8,7);
    maxNumCompThreads(1);
    f=parfevalOnAll(pool,@maxNumCompThreads,0,1); fetchOutputs(f);
    configs=[1 1;4 1;8 1;8 2;8 4;8 7;8 1];
    optimized='cgs2';
    if exist('helm_gmres_cgs2_mex','file')==3, optimized='cgs2_mex'; end
    parameters=linspace(.2,.6,8);
    % Warm up all workers before timing the first configuration.
    parfor k=1:8, solve_one(parameters(k),6,optimized); end
    row_cells=cell(size(configs,1),1);
    out=helm_output_path('outputs','scnet_3d_20260909','scaling');
    if ~isfolder(out), mkdir(out); end
    for k=1:size(configs,1)
        nw=configs(k,1); nt=configs(k,2);
        backend=optimized;
        if k==size(configs,1), backend='cgs2'; end
        f=parfevalOnAll(pool,@maxNumCompThreads,0,nt); fetchOutputs(f);
        actual=fetchOutputs(parfevalOnAll(pool,@maxNumCompThreads,1));
        if iscell(actual), actual=cell2mat(actual); end
        assert(all(actual(:)==nt),'Requested thread layout is unavailable.');
        residuals=zeros(1,8); iterations=residuals; times=residuals;
        timer=tic;
        parfor (j=1:8,nw)
            [residuals(j),iterations(j),times(j)]=solve_one(parameters(j),12,backend);
        end
        row=struct('workers',nw,'threads',nt,'worker_thread_limits',actual, ...
            'backend',backend,'wall_seconds',toc(timer), ...
            'max_true_residual',max(residuals),'iterations',iterations,'task_seconds',times);
        row_cells{k}=row;
        fprintf('SCALING workers=%d threads=%d backend=%s wall=%.3fs max_residual=%.3e\n', ...
            nw,nt,backend,row.wall_seconds,max(residuals));
        partial=vertcat(row_cells{1:k});
        save(fullfile(out,'scaling_progress.mat'),'partial');
        fid=fopen(fullfile(out,'scaling_progress.json'),'w');
        fprintf(fid,'%s\n',jsonencode(partial)); fclose(fid);
    end
    rows=vertcat(row_cells{:});
    [~,best]=min([rows.wall_seconds]);
    report=struct('rows',rows,'best',rows(best),'allocation',64,'omega',12, ...
        'note','One timing per layout; pool startup excluded; eight identical parameter tasks.');
    save(fullfile(out,'scaling.mat'),'report');
    fid=fopen(fullfile(out,'scaling.json'),'w'); fprintf(fid,'%s\n',jsonencode(report)); fclose(fid);
    delete(pool);
    assert(all([rows.max_true_residual]<=1.01e-7));
    fprintf('SCNET_3D_SCALING_PASS\n');
end

function [r,it,elapsed]=solve_one(mu,omega,backend)
    timer=tic;
    p=helmholtz_3d_problem_interface(struct('Nphys',[6 2 2]*omega-1, ...
        'npml',repmat(max(3,round(9*omega/24)),1,3),'omega',omega, ...
        'domain_lengths',[3 1 1],'material_model','gaussian_waveguide', ...
        'waveguide_width',[.14 .14],'source_model','gaussian', ...
        'source_location',[.2 .5 .5]));
    [A,b]=p.assemble(mu);
    B=p.assemble_shifted(mu,1.2);
    [M,~]=build_shifted_laplacian_multigrid_preconditioner(B,p.n, ...
        struct('mg_coarse_max_dofs',3000));
    opts=struct('restart',40,'maxit',40,'tol',1e-7,'orthogonalization',backend, ...
        'store_preconditioned_basis',false,'fixed_linear_preconditioner',true);
    [x,info]=shifted_laplacian_gmres(@(v)A*v,M,b,[],opts);
    r=norm(b-A*x)/norm(b); it=info.iter_count; elapsed=toc(timer);
end
