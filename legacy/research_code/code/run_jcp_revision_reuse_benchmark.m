function run_jcp_revision_reuse_benchmark(mode)
% Frozen-query, storage-budgeted fixed-space competitors for JCP review.
% New experiment only: no paper, historical model, or shared solver is edited.
    if nargin<1, mode='paper'; end
    assert(ismember(mode,{'prepare','smoke','paper'}));
    old_threads=maxNumCompThreads(1);
    cleanup=onCleanup(@()maxNumCompThreads(old_threads)); %#ok<NASGU>
    out=helm_output_path('outputs','jcp_revision_experiments_20260914','reuse');
    if ~exist(out,'dir'), mkdir(out); end
    model_file=fullfile(out,'models.mat');
    if ~isfile(model_file), prepare_models(out); end
    if strcmp(mode,'prepare'), return; end
    s=load(model_file,'experiment'); e=s.experiment; clear s;
    c=e.cohort; methods=[{'shifted','direct'},e.names];
    repetitions=3; queries=1:numel(c.parameters);
    if strcmp(mode,'smoke'), repetitions=1; queries=[1,c.representative,numel(c.parameters)]; end
    opts=struct('tol',1e-12,'restart',40,'maxit',20,'reorthogonalize',true);
    ilu_opts=struct('solver','ilu','ilu_droptol',1e-3); beta=.2;
    rows=table(); checks=table(); cases=cell(numel(c.parameters),numel(methods));
    for qi=queries
        mu=c.parameters(qi); alpha=mu^(-2)-1;
        A=c.A_base+alpha*c.A_right; b=c.b; nb=norm(b); ref=c.references(:,qi);
        mass=c.omega^2*ones(numel(b),1)+alpha*full(diag(c.A_right));
        assert(norm(b-A*ref)/nb<=opts.tol);
        for mi=1:2
            [u,h]=prefix(A,b,mu,e.models{mi},c.track_tol);
            err=norm(u-c.expected{mi}(:,qi))/max(norm(c.expected{mi}(:,qi)),eps);
            assert(err<1e-9,'Frozen trajectory replay changed.');
            checks=[checks;table(string(e.names{mi}),qi,err,h(end), ...
                'VariableNames',{'method','query','prefix_field_difference','post_ROM_residual'})]; %#ok<AGROW>
        end
        for rep=0:repetitions
            order=circshift(1:numel(methods),[0,mod(qi+rep,numel(methods))]);
            for mi=order
                method=methods{mi}; all_clock=tic;
                rom_s=0; setup_s=0; solve_s=0; factor_bytes=0;
                history=1; completion_history=[]; iters=0; flag=0;
                if strcmp(method,'direct')
                    tt=tic; u=A\b; solve_s=toc(tt); post=NaN;
                else
                    if mi>2
                        tt=tic; [u,history]=prefix(A,b,mu,e.models{mi-2},c.track_tol); rom_s=toc(tt);
                    else, u=zeros(size(b)); end
                    post=norm(b-A*u)/nb;
                    if post>opts.tol
                        tt=tic;
                        Ashift=A+1i*beta*spdiags(mass,0,numel(b),numel(b));
                        [M,setup]=build_shifted_laplacian_preconditioner(Ashift,ilu_opts);
                        setup_s=toc(tt); factor_bytes=setup.factor_bytes;
                        tt=tic;
                        [u,info]=shifted_laplacian_gmres(@(v)A*v,M,b,u,opts);
                        solve_s=toc(tt); iters=info.iter_count; flag=info.flag;
                        completion_history=info.relres_hist; clear M Ashift;
                    end
                end
                rho=norm(b-A*u)/nb; total_s=toc(all_clock);
                err=norm(u-ref)/max(norm(ref),eps);
                if rep>0
                    rows=[rows;table(string(method),qi,mu,rep,total_s,rom_s,setup_s,solve_s, ...
                        iters,post,rho,err,flag,factor_bytes, ...
                        'VariableNames',{'method','query','parameter','repeat','total_s','ROM_s', ...
                        'setup_s','solve_s','completion_iterations','post_ROM_residual', ...
                        'true_residual','relative_error','flag','factor_bytes'})]; %#ok<AGROW>
                end
                if rep==1
                    cases{qi,mi}=struct('method',method,'parameter',mu,'u',u, ...
                        'ROM_history',history,'completion_history',completion_history, ...
                        'true_residual',rho,'relative_error',err,'flag',flag);
                end
                if flag~=0 || ~isfinite(rho) || rho>opts.tol
                    save(fullfile(out,[mode '_failure.mat']),'rows','checks','cases','opts','ilu_opts','-v7.3');
                    writetable(rows,fullfile(out,[mode '_failure_rows.csv']));
                    error('JCPRevision:Residual','%s query%d: flag=%d rho=%.4e',method,qi,flag,rho);
                end
            end
            fprintf('JCP_REUSE %s query=%d/%d rep=%d/%d passed\n',mode,qi,numel(c.parameters),rep,repetitions);
        end
        writetable(rows,fullfile(out,[mode '_timings.csv']));
        writetable(checks,fullfile(out,[mode '_prefix_checks.csv']));
        save(fullfile(out,[mode '_results.mat']),'rows','checks','cases','opts','ilu_opts','beta','model_file','-v7.3');
    end
    fprintf('JCP_REUSE_%s_PASS rows=%d checks=%d\n',upper(mode),height(rows),height(checks));
end

function prepare_models(out)
    root=helm_output_path('outputs');
    source_cohort=fullfile(root,'jcp_shifted_20260914','frozen_cohorts.mat');
    s=load(source_cohort,'cohorts'); c=s.cohorts{1}; clear s;
    source_global=fullfile(root,'revision_20260907','halfspace','global_n24_eta-5.mat');
    s=load(source_global,'r'); g=s.r; clear s;
    source_dd=fullfile(root,'halfspace_dd_refinement','review_repaired_20260905','slabs1-aligned-sym8.mat');
    s=load(source_dd,'artifact'); d=s.artifact; clear s;
    assert(isequal(c.train_parameters,g.cfg.train_params,d.cfg.train_params));
    assert(isequal(c.A_base,g.problem.A_base) && isequal(c.A_right,g.problem.A_right));
    sg=rom_deployment_storage(g.correction_rom.layers); sd=rom_deployment_storage(d.layers);
    solution_ranks=[7,12,16,20,24]; n=numel(c.b);
    budgets=[sd.payload_MiB,sg.payload_MiB]; pooled_ranks=zeros(1,2);
    for j=1:2
        candidates=1:72; payload=(16*n*candidates+32*candidates.^2)/2^20;
        pooled_ranks(j)=max(candidates(payload<=budgets(j)));
    end
    manifest=struct('created',datestr(now,30),'problem','halfspace','unknowns',n, ...
        'omega',c.omega,'training_parameters',c.train_parameters,'test_parameters',c.parameters, ...
        'source_cohort',source_cohort,'source_global',source_global,'source_dd',source_dd, ...
        'solution_ranks',solution_ranks,'pooled_ranks',pooled_ranks,'pooled_budget_MiB',budgets, ...
        'rule','Ranks frozen before test solves. Solution POD uses all 24 available columns; pooled POD uses column-normalized 72 Global trajectory snapshots. Largest integer rank with dense basis and two reduced matrices within each target payload.', ...
        'completion','shifted ILUTP beta .2 droptol 1e-3; GMRES restart40 maxit20; true residual 1e-12', ...
        'timing','one computational thread, serial; one warmup and three measured repetitions; rotated method order; full per-query setup included; loading and common full matrix assembly excluded');
    fid=fopen(fullfile(out,'manifest.json'),'w'); fprintf(fid,'%s\n',jsonencode(manifest,PrettyPrint=true)); fclose(fid);
    % Store the rule before evaluating any query with a newly built space.
    names={'global','dd'}; models=c.models(1:2);
    model_rows=table();
    for j=1:2
        if j==1, stats=sg; old_offline=g.timing.offline; else, stats=sd; old_offline=d.timing.offline; end
        model=models{j}; w=whos('model');
        model_rows=[model_rows;table(string(names{j}),string(mat2str(stats.ranks)), ...
            stats.basis_MiB,stats.payload_MiB,w.bytes/2^20,NaN,NaN,old_offline, ...
            'VariableNames',{'method','ranks','basis_MiB','payload_MiB','matlab_object_MiB', ...
            'target_payload_MiB','target_utilization','historical_offline_s'})]; %#ok<AGROW>
    end
    S=g.train_reference.u_ref_snapshots;
    P=cat(2,g.correction_rom.layers{1}.eta_snapshots, ...
        g.correction_rom.layers{2}.eta_snapshots,g.correction_rom.layers{3}.eta_snapshots);
    P=P./max(vecnorm(P,2,1),realmin);
    t=tic; [Us,Ds,~]=svd(S,'econ'); solution_svd_s=toc(t);
    t=tic; [Up,Dp,~]=svd(P,'econ'); pooled_svd_s=toc(t);
    singular_values=struct('solution',diag(Ds),'pooled',diag(Dp));
    singular_value=singular_values.solution; mode_index=(1:numel(singular_value)).';
    relative_singular_value=singular_value/singular_value(1);
    writetable(table(mode_index,singular_value,relative_singular_value),fullfile(out,'solution_spectrum.csv'));
    construction=table();
    for k=solution_ranks
        name=sprintf('solution_pod%d',k); tt=tic;
        model=fixed_model(Us(:,1:k),c); build_s=toc(tt);
        names{end+1}=name; models{end+1}=model; %#ok<AGROW>
        model_rows=[model_rows;fixed_storage_row(name,model,NaN,NaN)]; %#ok<AGROW>
        construction=[construction;table(string(name),solution_svd_s,build_s, ...
            'VariableNames',{'method','svd_s','reduced_assembly_s'})]; %#ok<AGROW>
    end
    for j=1:2
        tags={'dd','global'}; name=['pooled_pod_' tags{j}]; tt=tic;
        model=fixed_model(Up(:,1:pooled_ranks(j)),c); build_s=toc(tt);
        names{end+1}=name; models{end+1}=model; %#ok<AGROW>
        model_rows=[model_rows;fixed_storage_row(name,model,budgets(j),g.timing.offline)]; %#ok<AGROW>
        construction=[construction;table(string(name),pooled_svd_s,build_s, ...
            'VariableNames',{'method','svd_s','reduced_assembly_s'})]; %#ok<AGROW>
    end
    % Isolated offline measurement. Existing Global/DD model remains frozen;
    % retraining is only to measure its actual serial training implementation.
    offline=table();
    for rep=1:3
        tt=tic;
        for k=1:numel(c.train_parameters)
            mu=c.train_parameters(k); A=c.A_base+(mu^(-2)-1)*c.A_right;
            Uref=A\c.b; assert(norm(Uref-S(:,k))/max(norm(S(:,k)),eps)<1e-10);
        end
        reference_s=toc(tt);
        tt=tic; [Q,~,~]=svd(S,'econ'); svd_s=toc(tt);
        for k=solution_ranks
            tt=tic; model=fixed_model(Q(:,1:k),c); reduced_s=toc(tt); %#ok<NASGU>
            offline=[offline;offline_row(sprintf('solution_pod%d',k),rep,reference_s,svd_s,reduced_s,0)]; %#ok<AGROW>
        end
        fresh_times=zeros(1,2);
        for j=1:2
            if j==1, cfg=g.cfg; else, cfg=d.cfg; end
            cfg.test_params=c.parameters(c.representative); cfg.plot_test_index=1;
            cfg.parallel_train=false; cfg.parallel_test=false;
            cfg.do_plots=false; cfg.save_plots=false; cfg.save_benchmark=false;
            cfg.run_wave_baselines=false; cfg.run_nr_rom_comparison=false;
            cfg.reference_mode='helm_lift'; cfg.online_completion_mode='shifted_laplacian';
            cfg.shifted_laplacian_solver='ilu'; cfg.shifted_laplacian_ilu_droptol=1e-3;
            cfg.shifted_laplacian_tol=1e-12; cfg.online_helm_accept_tol=1e-12;
            fresh=helm_waveholtz_correction_rom(cfg);
            fresh_times(j)=fresh.timing.offline;
            offline=[offline;offline_row(names{j},rep,0,0,0,fresh_times(j))]; %#ok<AGROW>
            assert(isequal(fresh.summary.corr_ranks(:).',eval_rank(model_rows.ranks(j))));
            clear fresh;
        end
        tt=tic;
        P=cat(2,g.correction_rom.layers{1}.eta_snapshots, ...
            g.correction_rom.layers{2}.eta_snapshots,g.correction_rom.layers{3}.eta_snapshots);
        P=P./max(vecnorm(P,2,1),realmin); [Q,~,~]=svd(P,'econ'); svd_s=toc(tt);
        for j=1:2
            tt=tic; model=fixed_model(Q(:,1:pooled_ranks(j)),c); reduced_s=toc(tt); %#ok<NASGU>
            tags={'dd','global'};
            offline=[offline;offline_row(['pooled_pod_' tags{j}],rep,0,svd_s,reduced_s,fresh_times(1))]; %#ok<AGROW>
        end
        writetable(offline,fullfile(out,'offline_timings.csv'));
        fprintf('JCP_REUSE_OFFLINE repeat=%d/3 passed\n',rep);
    end
    experiment=struct('cohort',c,'names',{names},'models',{models},'manifest',manifest, ...
        'storage',model_rows,'singular_values',singular_values,'offline',offline);
    save(fullfile(out,'models.mat'),'experiment','-v7.3');
    writetable(model_rows,fullfile(out,'storage.csv')); writetable(construction,fullfile(out,'construction.csv'));
    save(fullfile(out,'singular_values.mat'),'singular_values');
    fprintf('JCP_REUSE_PREPARE_PASS pooled_ranks=%s\n',mat2str(pooled_ranks));
end

function r=eval_rank(s)
    r=sscanf(erase(erase(char(s),'['),']'),'%f').';
end

function row=offline_row(name,rep,ref,svd_s,reduced,ta)
    row=table(string(name),rep,ref,svd_s,reduced,ta,ref+svd_s+reduced+ta, ...
        'VariableNames',{'method','repeat','reference_s','svd_s','reduced_assembly_s','TA_training_s','offline_total_s'});
end

function row=fixed_storage_row(name,model,target,old_offline)
    k=model.layers{1}.space.k; n=model.layers{1}.space.n; basis=16*n*k/2^20;
    payload=basis+32*k*k/2^20; w=whos('model');
    row=table(string(name),string(k),basis,payload,w.bytes/2^20,target,payload/target,old_offline, ...
        'VariableNames',{'method','ranks','basis_MiB','payload_MiB','matlab_object_MiB', ...
        'target_payload_MiB','target_utilization','historical_offline_s'});
end

function model=fixed_model(U,c)
    l=struct('space',rom_reduced_space_utils('make_dense_space',U), ...
        'Ar_base',U'*c.A_base*U,'Ar_right',U'*c.A_right*U);
    model=struct('layers',{{l}},'fixed_pod',true);
end

function [u,history]=prefix(A,b,mu,model,track_tol)
    n=numel(b); depth=numel(model.layers); nb=norm(b); u=zeros(n,1); history=1;
    if model.fixed_pod
        u=apply_inverse(b,mu,model.layers{1}); history=[1;norm(b-A*u)/nb]; return;
    end
    Q=zeros(n,depth+1); Z=zeros(n,depth); H=zeros(depth+1,depth); Q(:,1)=b/nb;
    for j=1:depth
        Z(:,j)=apply_inverse(Q(:,j),mu,model.layers{j});
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*Z(:,j),Q(:,1:j));
        rhs=zeros(j+1,1); rhs(1)=nb; y=H(1:j+1,1:j)\rhs;
        u=Z(:,1:j)*y; history(end+1,1)=norm(b-A*u)/nb; %#ok<AGROW>
        if stop || history(end)<=track_tol, break; end
    end
end

function z=apply_inverse(b,mu,layer)
    br=rom_reduced_space_utils('project',layer.space,b);
    Ar=layer.Ar_base+(mu^(-2)-1)*layer.Ar_right;
    z=rom_reduced_space_utils('lift',layer.space,Ar\br);
end
