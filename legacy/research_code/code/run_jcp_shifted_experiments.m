function run_jcp_shifted_experiments(mode)
% Frozen Experiment 1 representative cohort and Experiment 2, frequency only.
% No paper files or historical evidence are overwritten. All times are serial,
% one computational thread, including per-query shifted-ILU setup when needed.
    if nargin<1, mode='paper'; end
    assert(ismember(mode,{'smoke','paper','paper_reused','ssrom_reuse'}));
    if strcmp(mode,'ssrom_reuse'), run_ssrom_reuse(); return; end
    % paper_reused: same protocol, but SS-ROM reuses its space in every ROM
    % step. Frozen models are read only; results go to a separate folder.
    reuse_ss=strcmp(mode,'paper_reused');
    if reuse_ss, mode='paper'; end
    old_threads=maxNumCompThreads(1);
    cleanup=onCleanup(@()maxNumCompThreads(old_threads)); %#ok<NASGU>
    out=helm_output_path('outputs','jcp_shifted_20260914');
    model_file=fullfile(out,'frozen_cohorts.mat');
    if reuse_ss, out=helm_output_path('outputs','jcp_shifted_20261022'); end
    if ~exist(out,'dir'), mkdir(out); end
    if ~isfile(model_file)
        cohorts=prepare_cohorts(out);
        save(model_file,'cohorts','-v7.3');
    else
        s=load(model_file,'cohorts'); cohorts=s.cohorts;
    end
    % ECC smoke at 1e-8 exposed O(1e-6) field errors. Use a common tighter
    % true-residual criterion for the final comparison, keeping every basis.
    if cohorts{2}.tol~=1e-10
        assert(~reuse_ss);
        cohorts{2}.tol=1e-10; save(model_file,'cohorts','-v7.3');
    end
    if reuse_ss, cohorts{1}=reuse_ss_space(cohorts{1}); end
    opts=struct('tol',1e-12,'restart',40,'maxit',20,'reorthogonalize',true);
    ilu_opts=struct('solver','ilu','ilu_droptol',1e-3);
    beta=0.2; repetitions=5;
    if strcmp(mode,'smoke'), repetitions=1; end
    rows=table(); checks=table(); cases=cell(2,1);
    for ci=1:numel(cohorts)
        c=cohorts{ci}; opts.tol=c.tol;
        queries=1:numel(c.parameters);
        if strcmp(mode,'smoke'), queries=unique([c.representative,numel(c.parameters)]); end
        methods=[{'shifted','direct'},c.names];
        cases{ci}=cell(numel(c.parameters),numel(methods));
        for qi=queries
            if strcmp(c.name,'halfspace')
                alpha=c.parameters(qi)^(-2)-1;
                A=c.A_base+alpha*c.A_right; b=c.b;
                mass=c.omega^2*ones(numel(b),1)+alpha*full(diag(c.A_right));
            else
                A=c.systems{qi}.A; b=c.systems{qi}.b;
                mass=c.omega^2*ones(numel(b),1);
            end
            ref=c.references(:,qi); nb=norm(b);
            % Verify the saved reference independently of every timed solve.
            assert(norm(b-A*ref)/nb<=c.tol);
            for mi=1:numel(c.names)
                [u,h]=prefix(A,b,c.parameters(qi),c.models{mi},c.track_tol);
                old=c.expected{mi}(:,qi);
                field_error=norm(u-old)/max(norm(old),eps);
                assert(field_error<1e-9,'Frozen prefix changed: %s / %s / %d.',c.name,c.names{mi},qi);
                checks=[checks;table(string(c.name),string(c.names{mi}),qi,field_error,h(end), ...
                    'VariableNames',{'problem','method','query','prefix_field_difference','post_ROM_residual'})]; %#ok<AGROW>
            end
            for rep=0:repetitions
                order=circshift(1:numel(methods),[0,mod(qi+rep,numel(methods))]);
                for mi=order
                    method=methods{mi}; total_clock=tic;
                    prefix_s=0; setup_s=0; solve_s=0; factor_bytes=0;
                    hist=1; iters=0; flag=0; completion_hist=[];
                    if strcmp(method,'direct')
                        tt=tic; u=A\b; solve_s=toc(tt); post=NaN;
                    else
                        if mi>2
                            tt=tic;
                            [u,hist]=prefix(A,b,c.parameters(qi),c.models{mi-2},c.track_tol);
                            prefix_s=toc(tt);
                        else
                            u=zeros(size(b));
                        end
                        post=norm(b-A*u)/nb;
                        if post>c.tol
                            tt=tic;
                            Ashift=A+1i*beta*spdiags(mass,0,numel(b),numel(b));
                            [M,setup]=build_shifted_laplacian_preconditioner(Ashift,ilu_opts);
                            setup_s=toc(tt); factor_bytes=setup.factor_bytes;
                            tt=tic;
                            [u,info]=shifted_laplacian_gmres(@(v)A*v,M,b,u,opts);
                            solve_s=toc(tt); iters=info.iter_count; flag=info.flag;
                            completion_hist=info.relres_hist;
                            clear M Ashift;
                        end
                    end
                    rho=norm(b-A*u)/nb;
                    total_s=toc(total_clock);
                    error=norm(u-ref)/max(norm(ref),eps);
                    if rep>0
                        rows=[rows;table(string(c.name),string(method),qi,c.parameters(qi),rep, ...
                            c.tol,total_s,prefix_s,setup_s,solve_s,iters,post,rho,error,flag,factor_bytes, ...
                            'VariableNames',{'problem','method','query','parameter','repeat','tolerance', ...
                            'total_s','ROM_s','setup_s','solve_s','completion_iterations', ...
                            'post_ROM_residual','true_residual','relative_error','flag','factor_bytes'})]; %#ok<AGROW>
                    end
                    if rep==1
                        cases{ci}{qi,mi}=struct('method',method,'parameter',c.parameters(qi), ...
                            'u',u,'ROM_history',hist,'completion_history',completion_hist, ...
                            'true_residual',rho,'reference_error',error,'flag',flag);
                    end
                    % Save failures before stopping, never silently relax a tolerance.
                    if flag~=0 || rho>c.tol
                        save(fullfile(out,[mode '_failure.mat']),'rows','checks','cases','opts','ilu_opts');
                        error('JCP:Residual','%s/%s query %d failed: flag=%d, rho=%.3e.',c.name,method,qi,flag,rho);
                    end
                end
                fprintf('JCP_SHIFTED %s query=%d/%d rep=%d/%d passed\n',c.name,qi,numel(c.parameters),rep,repetitions);
            end
            writetable(rows,fullfile(out,[mode '_timings.csv']));
            writetable(checks,fullfile(out,[mode '_prefix_checks.csv']));
            save(fullfile(out,[mode '_results.mat']),'rows','checks','cases','opts','ilu_opts','beta','repetitions','model_file','-v7.3');
        end
    end
    histories=table(); plain=table();
    for ci=1:numel(cohorts)
        c=cohorts{ci};
        for qi=1:size(cases{ci},1)
            for mi=1:size(cases{ci},2)
                v=cases{ci}{qi,mi};
                if isempty(v) || strcmp(v.method,'direct'), continue; end
                h=v.ROM_history(:); depth=numel(h)-1;
                if ~isempty(v.completion_history), h=[h;v.completion_history(2:end)]; end %#ok<AGROW>
                steps=(0:numel(h)-1).';
                histories=[histories;table(repmat(string(c.name),numel(h),1), ...
                    repmat(string(v.method),numel(h),1),repmat(qi,numel(h),1), ...
                    steps,h,repmat(depth,numel(h),1),'VariableNames', ...
                    {'problem','method','query','step','relative_residual','ROM_depth'})]; %#ok<AGROW>
            end
        end
        qi=c.representative;
        if strcmp(c.name,'halfspace')
            A=c.A_base+(c.parameters(qi)^(-2)-1)*c.A_right; b=c.b;
        else, A=c.systems{qi}.A; b=c.systems{qi}.b; end
        plain_opts=opts; plain_opts.tol=c.tol; plain_opts.restart=100; plain_opts.maxit=1;
        [u,info]=shifted_laplacian_gmres(@(v)A*v,@(v)v,b,[],plain_opts);
        h=info.relres_hist; rho=norm(b-A*u)/norm(b);
        plain=[plain;table(repmat(string(c.name),numel(h),1), ...
            repmat(qi,numel(h),1),(0:numel(h)-1).',h,repmat(rho,numel(h),1), ...
            'VariableNames',{'problem','query','step','relative_residual','final_true_residual'})]; %#ok<AGROW>
    end
    writetable(histories,fullfile(out,[mode '_histories.csv']));
    writetable(plain,fullfile(out,[mode '_plain_gmres.csv']));
    fprintf('JCP_SHIFTED_%s_PASS rows=%d prefix_checks=%d\n',upper(mode),height(rows),height(checks));
end

function cohorts=prepare_cohorts(out)
    root=helm_output_path('outputs');
    g=load(fullfile(root,'halfspace_dd_refinement','review_repaired_20260905','global.mat'),'artifact'); g=g.artifact;
    d=load(fullfile(root,'halfspace_dd_refinement','review_repaired_20260905','slabs1-aligned-sym8.mat'),'artifact'); d=d.artifact;
    full=load(fullfile(root,'revision_20260907','halfspace','global_n24_eta-5.mat'),'r'); r=full.r; clear full;
    assert(isequal(g.cfg.train_params,r.cfg.train_params,d.cfg.train_params));
    assert(isequal(g.summary.corr_ranks(:),r.summary.corr_ranks(:)));
    parameters=g.test.params(:).'; [~,representative]=min(abs(parameters-0.8622));
    c=struct('name','halfspace','parameters',parameters,'representative',representative, ...
        'omega',g.cfg.omega,'tol',1e-12,'track_tol',g.cfg.helm_gmres_track_tol, ...
        'A_base',r.problem.A_base,'A_right',r.problem.A_right,'b',g.rhs, ...
        'references',g.test.helm_u_ref,'train_parameters',g.cfg.train_params);
    assert(isequal(g.rhs,r.rhs));
    c.names={'global','dd','nr','fixed_pod'};
    c.models={lean_model(g.layers,false),lean_model(d.layers,false)};
    c.expected={g.test.online_precorr_u,d.test.online_precorr_u};
    % Reconstruct the existing equal-rank NR control through its active trainer.
    % Global training is frozen; no time-domain solve is used.
    cfg=r.cfg; cfg.test_params=parameters; cfg.run_nr_rom_comparison=true;
    cfg.online_completion_mode='shifted_laplacian'; cfg.online_helm_accept_tol=c.tol;
    cfg.shifted_laplacian_solver='ilu'; cfg.shifted_laplacian_ilu_droptol=1e-3;
    cfg.shifted_laplacian_beta=.2; cfg.shifted_laplacian_restart=40;
    cfg.shifted_laplacian_maxit=20; cfg.shifted_laplacian_tol=c.tol;
    cfg.run_wave_baselines=false; cfg.do_plots=false; cfg.save_plots=false;
    cfg.save_benchmark=false; cfg.parallel_train=false; cfg.parallel_test=false;
    nrfile=fullfile(out,'nr_control.mat');
    if isfile(nrfile), nr=load(nrfile,'model','expected');
    else
        rebuilt=helm_waveholtz_correction_rom(cfg,r);
        model=lean_model(rebuilt.nr_comparison.correction_rom.layers,false);
        expected=rebuilt.nr_comparison.test.online_precorr_u;
        assert(all(rebuilt.nr_comparison.test.online_final_err_helm_full<1e-9));
        save(nrfile,'model','expected','cfg','-v7.3');
        nr=struct('model',model,'expected',expected); clear rebuilt;
    end
    c.models{3}=nr.model; c.expected{3}=nr.expected;
    % Independent reproduction target for NR from the historical paper cohort.
    old=load(fullfile(root,'halfspace_dd_paper','paper_data.mat'),'paper');
    for qi=1:numel(parameters)
        rho=norm(c.b-(c.A_base+(parameters(qi)^(-2)-1)*c.A_right)*nr.expected(:,qi))/norm(c.b);
        assert(abs(rho-old.paper.aux.nr.online_precorr_relres_final(qi))<1e-9);
    end
    p=load(fullfile(root,'revision_20260907','halfspace','fixedpod_n24_eta-5.mat'),'r'); p=p.r;
    assert(isequal(c.A_base,p.problem.A_base) && isequal(c.A_right,p.problem.A_right));
    assert(isfield(p.correction_rom,'fixed_pod_initial') && p.correction_rom.fixed_pod_initial);
    c.models{4}=lean_model(p.correction_rom.layers,true);
    % This cached study has different queries: calculate its Galerkin initial
    % field independently from the saved dense POD basis on the paper queries.
    V=p.correction_rom.layers{1}.basis_space.basis;
    c.expected{4}=complex(zeros(numel(c.b),numel(parameters)));
    for qi=1:numel(parameters)
        A=c.A_base+(parameters(qi)^(-2)-1)*c.A_right;
        c.expected{4}(:,qi)=V*((V'*A*V)\(V'*c.b));
    end
    cohorts={c}; clear c p r g d nr old;
    s=load(fullfile(root,'ecc_two_layer_paper_20260911','paper_data.mat'),'global_results','dd_results');
    g=s.global_results; d=s.dd_results; clear s;
    parameters=g.test.ratio(:).'; [~,representative]=min(abs(parameters-.9072));
    c=struct('name','ecc','parameters',parameters,'representative',representative, ...
        'omega',g.cfg.omega,'tol',1e-10,'track_tol',g.cfg.helm_gmres_track_tol, ...
        'references',g.test.helm_u_ref,'train_parameters',g.cfg.ratio_train);
    c.names={'global','dd'};
    c.models={lean_model(g.correction_rom.layers(1:2),false),lean_model(d.correction_rom.layers(1:2),false)};
    c.expected={g.test.online_precorr_u,d.test.online_precorr_u};
    c.systems=cell(size(g.test.systems));
    for k=1:numel(c.systems)
        assert(isequal(g.test.systems{k}.A,d.test.systems{k}.A));
        c.systems{k}=struct('A',g.test.systems{k}.A,'b',g.test.systems{k}.b);
    end
    cohorts{2}=c;
end

function model=lean_model(layers,fixed)
    model=struct('layers',{cell(size(layers))},'fixed_pod',fixed);
    for j=1:numel(layers)
        l=layers{j}; model.layers{j}=struct('space',l.basis_space);
        if isfield(l,'Ar_base')
            model.layers{j}.Ar_base=l.Ar_base; model.layers{j}.Ar_right=l.Ar_right;
        end
    end
end

function [u,history]=prefix(A,b,mu,model,track_tol)
    n=numel(b); depth=numel(model.layers); nb=norm(b);
    u=zeros(n,1); history=1;
    if model.fixed_pod
        u=apply_inverse(A,b,mu,model.layers{1});
        history=[1;norm(b-A*u)/nb]; return;
    end
    Q=zeros(n,depth+1); Z=zeros(n,depth); H=zeros(depth+1,depth); Q(:,1)=b/nb;
    for j=1:depth
        Z(:,j)=apply_inverse(A,Q(:,j),mu,model.layers{j});
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*Z(:,j),Q(:,1:j));
        g=zeros(j+1,1); g(1)=nb;
        if isfield(model.layers{j},'Ar_base'), y=H(1:j+1,1:j)\g;
        else, y=lsqminnorm(H(1:j+1,1:j),g); end
        u=Z(:,1:j)*y;
        history(end+1,1)=norm(b-A*u)/nb; %#ok<AGROW>
        if stop || history(end)<=track_tol, break; end
    end
end

function z=apply_inverse(A,b,mu,layer)
    sp=layer.space;
    br=rom_reduced_space_utils('project',sp,b);
    if isfield(layer,'Ar_base')
        Ar=layer.Ar_base+(mu^(-2)-1)*layer.Ar_right; coeff=Ar\br;
    else
        Ar=full(rom_reduced_space_utils('assemble_reduced_matrix',A,sp));
        if rcond(Ar)<1e-12, coeff=pinv(Ar)*br; else, coeff=Ar\br; end
    end
    z=rom_reduced_space_utils('lift',sp,coeff);
end

function run_ssrom_reuse()
% Diagnostic only, no timing: reuse the SS-ROM space as the preconditioner in
% every ROM step of the frozen half-space cohort. Reads the frozen models and
% paper timings read-only; writes to a separate output folder.
    src=helm_output_path('outputs','jcp_shifted_20260914');
    out=helm_output_path('outputs','jcp_ssrom_reuse_20261022');
    if ~exist(out,'dir'), mkdir(out); end
    s=load(fullfile(src,'frozen_cohorts.mat'),'cohorts'); c=s.cohorts{1}; clear s;
    assert(strcmp(c.name,'halfspace'));
    ig=c.models{strcmp(c.names,'fixed_pod')}; assert(ig.fixed_pod);
    gl=c.models{strcmp(c.names,'global')};
    depth=numel(gl.layers);
    reused=struct('layers',{repmat(ig.layers(1),1,depth)},'fixed_pod',false);
    sp=ig.layers{1}.space;
    r=numel(rom_reduced_space_utils('project',sp,c.b));
    V=complex(zeros(numel(c.b),r)); I=eye(r);
    for i=1:r, V(:,i)=rom_reduced_space_utils('lift',sp,I(:,i)); end
    paper=readtable(fullfile(src,'paper_timings.csv'),'TextType','string');
    opts=struct('tol',c.tol,'restart',40,'maxit',20,'reorthogonalize',true);
    ilu_opts=struct('solver','ilu','ilu_droptol',1e-3); beta=0.2;
    rows=table(); histories=table();
    for qi=1:numel(c.parameters)
        mu=c.parameters(qi); alpha=mu^(-2)-1;
        A=c.A_base+alpha*c.A_right; b=c.b; nb=norm(b);
        mass=c.omega^2*ones(numel(b),1)+alpha*full(diag(c.A_right));
        bound=norm(b-A*(V*((A*V)\b)))/nb;
        [~,hg]=prefix(A,b,mu,gl,c.track_tol);
        [u0,h0]=prefix(A,b,mu,ig,c.track_tol);
        assert(norm(u0-c.expected{strcmp(c.names,'fixed_pod')}(:,qi))<=1e-9*norm(u0));
        [u1,h1]=prefix(A,b,mu,reused,c.track_tol);
        it=zeros(1,2); ch=cell(1,2); u={u0,u1};
        for k=1:2
            if norm(b-A*u{k})/nb>c.tol
                Ashift=A+1i*beta*spdiags(mass,0,numel(b),numel(b));
                M=build_shifted_laplacian_preconditioner(Ashift,ilu_opts);
                [u{k},info]=shifted_laplacian_gmres(@(v)A*v,M,b,u{k},opts);
                assert(info.flag==0); it(k)=info.iter_count; ch{k}=info.relres_hist(:);
            end
            assert(norm(b-A*u{k})/nb<=c.tol);
        end
        p=paper(paper.problem=="halfspace" & paper.method=="fixed_pod" & paper.query==qi,:);
        assert(all(p.completion_iterations==it(1)),'SS-ROM SL steps differ from the paper run.');
        rows=[rows;table(qi,mu,r,hg(2),h0(end),it(1),h1(2),h1(end),numel(h1)-1,it(2),bound, ...
            'VariableNames',{'query','parameter','ss_dimension','global_step1_residual', ...
            'ss_initial_residual','ss_initial_SL_steps','reused_step1_residual', ...
            'reused_ROM_residual','reused_ROM_steps','reused_SL_steps','range_lower_bound'})]; %#ok<AGROW>
        h=[h1(:);ch{2}(2:end)];
        histories=[histories;table(repmat(qi,numel(h),1),(0:numel(h)-1).',h, ...
            repmat(numel(h1)-1,numel(h),1),'VariableNames',{'query','step','relative_residual','ROM_depth'})]; %#ok<AGROW>
        fprintf('SSROM_REUSE query=%d/%d ig=%.3e(%d SL) reused=%.3e(%d SL) bound=%.3e\n', ...
            qi,numel(c.parameters),h0(end),it(1),h1(end),it(2),bound);
    end
    writetable(rows,fullfile(out,'ssrom_reuse_summary.csv'));
    writetable(histories,fullfile(out,'ssrom_reuse_histories.csv'));
    fprintf('SSROM_REUSE_PASS queries=%d\n',height(rows));
end

function c=reuse_ss_space(c)
% Replace the SS-ROM initial guess by the same space reused in all ROM steps.
    k=strcmp(c.names,'fixed_pod'); ig=c.models{k}; assert(ig.fixed_pod);
    depth=numel(c.models{strcmp(c.names,'global')}.layers);
    c.models{k}=struct('layers',{repmat(ig.layers(1),1,depth)},'fixed_pod',false);
    for qi=1:numel(c.parameters)
        A=c.A_base+(c.parameters(qi)^(-2)-1)*c.A_right;
        c.expected{k}(:,qi)=prefix(A,c.b,c.parameters(qi),c.models{k},c.track_tol);
    end
end
