function benchmark_ecc_dd_budget(mode)
% Paired serial benchmark for strict global-tail allocation.
    old=maxNumCompThreads(1); cleanup=onCleanup(@()maxNumCompThreads(old)); %#ok<NASGU>
    out=helm_output_path('outputs','ecc_dd_budget_20260919');
    legacy=helm_output_path('outputs','ecc_shifted_tuning_20260919');
    s=load(helm_output_path('outputs','jcp_shifted_20260914','frozen_cohorts.mat'),'cohorts');
    c=s.cohorts{2}; names={'shifted','direct'};models={[],[]};training_sets={};
    specs={'uniform6','last4','last3','layered'};
    if ~strcmp(mode,'screen'),specs={'uniform6','last3'};end
    if strcmp(mode,'uniform3'),specs={'last3','uniform3'};end
    if strcmp(mode,'two6'),specs={'last3','two6'};end
    if strcmp(mode,'uniform5'),specs={'last3','uniform5'};end
    for si=1:numel(specs)
        for method={'global','dd'}
            tag=[specs{si} '_' method{1}];s=load(fullfile(out,[tag '.mat']),'cfg','model');
            cfg=s.cfg;names{end+1}=tag;models{end+1}=s.model;training_sets{end+1}=cfg.ratio_train; %#ok<AGROW>
        end
    end
    repetitions=3; validation_seeds=[]; validation_candidates=table();
    if ismember(mode,{'uniform3','two6','uniform5'})
        v=load(fullfile(out,'validate_results.mat'),'c');c=v.c;repetitions=5;
    elseif strcmp(mode,'validate')
        cfg.ratio_train=c.train_parameters; cfg.setup_only=true;
        % Geometry repair can map distinct draws to the same supported point.
        % Reject overlaps before any solver evaluation and collect 22 queries.
        parameters=[];
        vold=load(fullfile(legacy,'validate_results.mat'),'c');
        excluded=[c.parameters(:);c.train_parameters(:);vold.c.parameters(:)];
        for k=1:numel(training_sets), excluded=[excluded;training_sets{k}(:)]; end %#ok<AGROW>
        for seed=20260923:20260960
            validation_seeds(end+1)=seed; %#ok<AGROW>
            cfg.ratio_test=[]; cfg.ratio_test_rng_seed=seed;
            v=waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
            for parameter=v.cfg.ratio_test
                eligible=all(abs([excluded;parameters(:)]-parameter)>1e-12);
                validation_candidates=[validation_candidates;table(seed,parameter,eligible)]; %#ok<AGROW>
                if eligible, parameters(end+1)=parameter; end %#ok<AGROW>
                if numel(parameters)==22, break; end
            end
            if numel(parameters)==22, break; end
        end
        assert(numel(parameters)==22);
        parameters=unique(parameters);
        assert(all(min(abs(c.parameters(:)-parameters(:).'),[],1)>1e-12));
        c.parameters=parameters; c.systems=cell(1,numel(c.parameters));
        for k=1:numel(training_sets)
            assert(all(min(abs(training_sets{k}(:)-c.parameters(:).'),[],1)>1e-12));
        end
        assert(all(min(abs(c.train_parameters(:)-c.parameters(:).'),[],1)>1e-12));
        c.references=zeros(numel(v.system.b),numel(c.parameters));
        for qi=1:numel(c.parameters)
            cfg.ratio_test=c.parameters(qi);
            v=waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
            c.systems{qi}=struct('A',v.system.A,'b',v.system.b);
            c.references(:,qi)=v.system.A\v.system.b;
        end
        repetitions=5;
    else
        assert(ismember(mode,{'screen','confirm'}));
        if strcmp(mode,'confirm'), repetitions=5; end
    end
    opts=struct('tol',1e-12,'restart',40,'maxit',20,'reorthogonalize',true);
    ilu_opts=struct('solver','ilu','ilu_droptol',1e-3); beta=.2;
    rows=table(); cases=cell(numel(c.parameters),numel(names)); checks=table();
    for qi=1:numel(c.parameters)
        A=c.systems{qi}.A; b=c.systems{qi}.b; nb=norm(b); ref=c.references(:,qi);
        for rep=0:repetitions
            order=circshift(1:numel(names),[0,mod(qi+rep,numel(names))]);
            for mi=order
                total=tic; rom_s=0; setup_s=0; solve_s=0; it=0; flag=0; ch=[]; h=1;
                if strcmp(names{mi},'direct')
                    tt=tic; u=A\b; solve_s=toc(tt); post=NaN;
                else
                    u=zeros(size(b));
                    if ~isempty(models{mi})
                        tt=tic; [u,h]=prefix(A,b,models{mi},opts.tol); rom_s=toc(tt);
                    end
                    post=norm(b-A*u)/nb;
                    if post>opts.tol
                        tt=tic;
                        Ashift=A+1i*beta*c.omega^2*speye(numel(b));
                        M=build_shifted_laplacian_preconditioner(Ashift,ilu_opts);
                        setup_s=toc(tt); tt=tic;
                        [u,info]=shifted_laplacian_gmres(@(v)A*v,M,b,u,opts);
                        solve_s=toc(tt); it=info.iter_count; flag=info.flag; ch=info.relres_hist;
                        clear M Ashift;
                    end
                end
                rho=norm(b-A*u)/nb; total_s=toc(total); error=norm(u-ref)/norm(ref);
                assert(flag==0 && rho<=opts.tol,'ECC benchmark failed.');
                if rep>0
                    row=table(string(names{mi}),qi,c.parameters(qi),rep,total_s,rom_s,setup_s, ...
                        solve_s,it,post,rho,error,'VariableNames',{'method','query','parameter', ...
                        'repeat','total_s','ROM_s','setup_s','solve_s','iterations','post','residual','error'});
                    rows=[rows;row]; %#ok<AGROW>
                end
                if rep==1
                    cases{qi,mi}=struct('method',names{mi},'u',u,'ROM_history',h,'completion_history',ch);
                end
            end
        end
        fprintf('ECC_BENCHMARK %s query=%d/%d passed\n',mode,qi,numel(c.parameters));
        writetable(rows,fullfile(out,[mode '_timings.csv']));
    end
    summary=table(); baseline=mean(rows.total_s(rows.method=="shifted"));
    for name=names
        t=rows(rows.method==string(name{1}),:);
        summary=[summary;table(string(name{1}),mean(t.total_s),mean(t.ROM_s),mean(t.setup_s), ...
            mean(t.solve_s),mean(t.iterations),baseline/mean(t.total_s),sum(t.setup_s==0)/repetitions, ...
            max(t.residual),max(t.error),'VariableNames',{'method','total_s','ROM_s','setup_s', ...
            'solve_s','iterations','speedup','no_setup_queries','max_residual','max_error'})]; %#ok<AGROW>
    end
    writetable(summary,fullfile(out,[mode '_summary.csv']));
    save(fullfile(out,[mode '_results.mat']),'rows','summary','checks','cases','names','models','c', ...
        'opts','ilu_opts','beta','repetitions','specs','training_sets','validation_seeds', ...
        'validation_candidates','-v7.3');
    if ~isempty(validation_candidates)
        writetable(validation_candidates,fullfile(out,'validation_candidates.csv'));
    end
    disp(summary); fprintf('ECC_BENCHMARK_%s_PASS\n',upper(mode));
end

function [u,h]=prefix(A,b,model,tol)
    n=numel(b); depth=numel(model.layers); nb=norm(b); u=zeros(n,1); h=1;
    Q=zeros(n,depth+1); Z=zeros(n,depth); H=zeros(depth+1,depth); Q(:,1)=b/nb;
    for j=1:depth
        sp=model.layers{j}.space;
        Ar=full(rom_reduced_space_utils('assemble_reduced_matrix',A,sp));
        br=rom_reduced_space_utils('project',sp,Q(:,j));
        if isempty(Ar), coeff=zeros(0,1);
        elseif rcond(Ar)<1e-12, coeff=pinv(Ar)*br; else, coeff=Ar\br; end
        Z(:,j)=rom_reduced_space_utils('lift',sp,coeff);
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*Z(:,j),Q(:,1:j));
        rhs=zeros(j+1,1); rhs(1)=nb;
        u=Z(:,1:j)*lsqminnorm(H(1:j+1,1:j),rhs);
        h(end+1,1)=norm(b-A*u)/nb; %#ok<AGROW>
        if stop || h(end)<=tol, break; end
    end
end
