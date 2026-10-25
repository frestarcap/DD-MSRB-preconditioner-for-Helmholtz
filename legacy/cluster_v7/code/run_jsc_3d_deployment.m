function summary=run_jsc_3d_deployment(file,policy,parameters,repetitions,collect_diagnostics)
%RUN_JSC_3D_DEPLOYMENT Serial online timings with eager or lazy MG construction.
% Run this in a separate MATLAB process under /usr/bin/time -l for peak RSS.
% Full-order assembly and MAT load are excluded from query time but included
% in process RSS. Every recorded query includes its own MG setup when used.
    if nargin<2, policy='lazy'; end
    if nargin<5, collect_diagnostics=true; end
    if strcmp(policy,'baseline')
        s=load(file,'cfg','problem_cfg','rhs');
        model=struct('layers',{{}},'num_layers',0);
    else
        s=load(file); model=s.model;
    end
    cfg=s.cfg;
    if nargin<3 || isempty(parameters), parameters=s.problem_cfg.test_params; end
    if nargin<4, repetitions=5; end
    problem=helmholtz_3d_problem_interface(s.problem_cfg); b=s.rhs;
    assert(ismember(policy,{'lazy','eager','baseline'}));
    opts=struct('solver','mg','grid_size',problem.n, ...
        'mg_max_levels',cfg.mg_max_levels,'mg_coarse_max_dofs',cfg.mg_coarse_max_dofs, ...
        'mg_pre_sweeps',cfg.mg_pre_sweeps,'mg_post_sweeps',cfg.mg_post_sweeps, ...
        'mg_jacobi_weight',cfg.mg_jacobi_weight,'mg_cycles_per_apply',cfg.mg_cycles_per_apply);
    gmopts=struct('restart',cfg.gmres_restart,'maxit',cfg.gmres_maxit, ...
        'tol',cfg.gmres_tol,'reorthogonalize',true,'verbose',false);
    spaces=cellfun(@(l)l.space,model.layers,'UniformOutput',false);
    rows=table(); diagnostics=table();
    for mu=parameters(:).'
        mass=problem.affine.mass_base+mu*problem.affine.mass_parameter_diagonal;
        A=problem.K_pml+spdiags(mass,0,problem.N,problem.N);
        if collect_diagnostics
        reduced=cellfun(@(l)l.Ar0+mu*l.Ar1,model.layers,'UniformOutput',false);
        [~,d]=rom_true_action_diagnostics(A,b,spaces,reduced);
        d.parameter=repmat(mu,height(d),1); diagnostics=[diagnostics;d]; %#ok<AGROW>
        end
        for rep=0:repetitions
            timer=tic; setup_s=0; mg_bytes=0; M=[];
            if strcmp(policy,'eager')
                [M,mg]=build_shifted_laplacian_preconditioner( ...
                    A+1i*cfg.shift_beta*spdiags(mass,0,problem.N,problem.N),opts);
                setup_s=mg.setup_time; mg_bytes=mg.factor_bytes;
            end
            if strcmp(policy,'baseline')
                x=zeros(size(b),'like',b); pre=1; pre_s=0;
            else
                tr=tic; [x,pre]=reduced_steps(A,b,model,mu); pre_s=toc(tr);
            end
            completion_s=0; iterations=0;
            if pre>cfg.gmres_tol
                if isempty(M)
                    [M,mg]=build_shifted_laplacian_preconditioner( ...
                        A+1i*cfg.shift_beta*spdiags(mass,0,problem.N,problem.N),opts);
                    setup_s=mg.setup_time; mg_bytes=mg.factor_bytes;
                end
                [x,info]=shifted_laplacian_gmres(@(v)A*v,M,b,x,gmopts);
                completion_s=info.total_time; iterations=info.iter_count;
            end
            rho=norm(b-A*x)/norm(b); total_s=toc(timer);
            assert(rho<=cfg.gmres_tol,'Final true residual failed.');
            if rep>0
                rows=[rows;table(mu,rep,pre_s,setup_s,completion_s,total_s, ...
                    iterations,pre,rho,mg_bytes/2^20,'VariableNames', ...
                    {'parameter','repetition','reduced_s','mg_setup_s','completion_s', ...
                    'total_s','completion_iterations','precompletion_residual', ...
                    'true_residual','mg_MiB'})]; %#ok<AGROW>
            end
            clear M mg x;
        end
    end
    [folder,name]=fileparts(file); stem=fullfile(folder,[name '_' policy]);
    writetable(rows,[stem '_timings.csv']); writetable(diagnostics,[stem '_diagnostics.csv']);
    summary=rows; save([stem '_results.mat'],'summary','diagnostics','policy','parameters','file');
    disp(rows);
end

function [x,rho]=reduced_steps(A,b,model,mu)
    k=model.num_layers; n=numel(b); beta=norm(b);
    x=zeros(n,1,'like',b); rho=0;
    if beta==0, return; end
    Q=complex(zeros(n,k+1)); Z=complex(zeros(n,k)); H=zeros(k+1,k);
    Q(:,1)=b/beta;
    for j=1:k
        l=model.layers{j};
        z=rom_reduced_space_utils('lift',l.space, ...
            (l.Ar0+mu*l.Ar1)\rom_reduced_space_utils('project',l.space,Q(:,j)));
        Z(:,j)=z; [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*z,Q(:,1:j));
        g=zeros(j+1,1); g(1)=beta;
        x=Z(:,1:j)*(H(1:j+1,1:j)\g);
        rho=norm(b-A*x)/beta;
        if stop, break; end
    end
end
