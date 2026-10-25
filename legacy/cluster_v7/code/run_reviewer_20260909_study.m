function summary = run_reviewer_20260909_study(mode)
%RUN_REVIEWER_20260909_STUDY Frequency, two-material, and partition checks.
% Scope: Helm half-space. Uses the active PML assembly, local-POD allocation,
% two-pass Arnoldi, and shifted-Laplacian GMRES helpers. No WaveHoltz lifting.
% train: checkpoint exact inverse-action snapshots and frozen deployed models.
% timing: serial warm-up + five repetitions, after all training has finished.
% diagnostics: conditional truncation constants on the actual query trajectory.
% smoke: small independent problem; all outputs remain below outputs/.
    if nargin < 1, mode = 'train'; end
    assert(ismember(mode, {'train','timing','diagnostics','smoke', ...
        'explore_train','explore_timing','holdout'}));
    out = helm_output_path('outputs','reviewer_20260909');
    if ~isfolder(out), mkdir(out); end
    cases = protocol();
    if startsWith(mode,'explore_')
        cases=exploration_protocol(); mode=erase(mode,'explore_');
        summary_tag=['explore_' mode];
    else
        summary_tag=mode;
    end
    repeats=5;
    if strcmp(summary_tag,'explore_timing'), repeats=3; end
    if strcmp(mode,'holdout')
        selection=load(fullfile(out,'selected_holdout.mat'),'selected');
        cases=selection.selected;
    end
    if strcmp(mode,'smoke')
        cases=cases(1); cases.name='smoke'; cases.scale=0.25;
        cases.training=[ones(6,1),linspace(0.8,0.95,6)'];
        cases.queries=[1,0.837;1,0.917]; cases.variants={'global','dd'};
    end
    summary = table();
    for ic=1:numel(cases)
        c=cases(ic); folder=fullfile(out,c.name);
        if ~isfolder(folder), mkdir(folder); end
        setupfile=fullfile(folder,'problem.mat');
        if isfile(setupfile)
            s=load(setupfile); assert(isequaln(c,s.c),'Cached protocol changed.');
        else
            assert(ismember(mode,{'train','smoke'}),'Train first.');
            cfg=struct('setup_only',true,'Nphys',round(126*c.scale)-1, ...
                'omega',53.5*c.scale,'sigma_max',80*c.scale, ...
                'train_params',unique(c.training(:,2))', ...
                'test_params',unique(c.queries(:,2))', ...
                'parallel_train',false,'parallel_test',false,'do_plots',false, ...
                'save_plots',false,'save_benchmark',false);
            r=helm_waveholtz_correction_rom(cfg); p=r.problem; b=r.rhs; cfg=r.cfg;
            % Both material coefficients extend normally through the PML.
            Aleft=p.omega^2*speye(p.Ntot)-p.A_right;
            ops={p.A_base,p.A_right};
            if any(c.training(:,1)~=1), ops{3}=Aleft; end
            save(setupfile,'c','p','b','cfg','ops','-v7.3');
            s=load(setupfile); clear r p b cfg ops Aleft;
        end
        p=s.p; b=s.b; ops=s.ops; cfg=s.cfg; clear s;
        for iv=1:numel(c.variants)
            variant=c.variants{iv}; file=fullfile(folder,[variant '.mat']);
            if isfile(file)
                d=load(file,'model','offline','audit');
                model=d.model; offline=d.offline; audit=d.audit; clear d;
            else
                assert(ismember(mode,{'train','smoke'}),'Train first.');
                fprintf('%s / %s: training\n',c.name,variant);
                part=partition(p,cfg,variant);
                [model,offline,audit]=train(ops,b,c.training,part,c.eta);
                save(file,'model','offline','audit','c','variant','-v7.3');
            end
            st=storage(model); info=whos('model');
            row=table(string(c.name),string(variant),p.omega,p.Ntot, ...
                size(c.training,1),string(mat2str(st.ranks)),st.basis, ...
                st.operators,st.index,st.payload,info.bytes/2^20, ...
                offline.total,offline.inverse,offline.replay,offline.pod, ...
                offline.assembly,max([audit.tail]),max([audit.snapshot_residual]), ...
                'VariableNames',{'study','variant','omega','N','training_count', ...
                'ranks','basis_MiB','operators_MiB','index_MiB','payload_MiB', ...
                'object_MiB','offline_s','inverse_s','replay_s','pod_s', ...
                'reduced_assembly_s','max_tail','max_snapshot_residual'});
            summary=[summary;row]; %#ok<AGROW>
            writetable(summary,fullfile(out,['summary_' summary_tag '.csv']));
            if strcmp(mode,'timing') || strcmp(mode,'smoke')
                timingfile=fullfile(folder,[variant '_timing.csv']);
                if ~isfile(timingfile)
                    rows=benchmark(ops,b,c.queries,model,variant,c.tol,repeats);
                    writetable(rows,timingfile);
                end
            elseif strcmp(mode,'holdout')
                width=(max(c.training(:,1))-min(c.training(:,1)))/2;
                [ll,rr]=ndgrid(1+width*[-0.89,0.17,0.91],[0.807,0.889,0.947]);
                rows=benchmark(ops,b,[ll(:),rr(:)],model,variant,c.tol,5);
                writetable(rows,fullfile(folder,[variant '_holdout.csv']));
            elseif strcmp(mode,'diagnostics') && strcmp(c.name,'frequency_1') ...
                    && ismember(variant,{'global','dd'})
                rows=diagnostics(ops,b,[1,0.8761],model,audit,c.eta);
                rows.variant=repmat(string(variant),height(rows),1);
                writetable(rows,fullfile(folder,[variant '_constants.csv']));
            end
            clear model audit;
        end
        if strcmp(mode,'timing') || strcmp(mode,'smoke') || strcmp(mode,'holdout')
            for variant={'direct','shifted'}
                queries=c.queries; suffix='_timing.csv';
                if strcmp(mode,'holdout')
                    queries=[ll(:),rr(:)]; suffix='_holdout.csv';
                end
                timingfile=fullfile(folder,[variant{1} suffix]);
                if ~isfile(timingfile)
                    shared=find_baseline(out,c,ops,queries,variant{1},suffix);
                    if isempty(shared)
                        rows=benchmark(ops,b,queries,{},variant{1},c.tol,repeats);
                        writetable(rows,timingfile);
                    else
                        copyfile(shared,timingfile);
                        provenance=struct('source',shared,'operator_equality',true, ...
                            'query_equality',true,'tolerance',c.tol);
                        save([timingfile '.provenance.mat'],'provenance');
                    end
                end
            end
        end
        clear p b ops;
    end
    disp(summary);
end

function file=find_baseline(out,c,ops,queries,variant,suffix)
    file='';
    if ~strcmp(suffix,'_timing.csv'), return; end
    dirs=dir(out);
    for k=1:numel(dirs)
        if ~dirs(k).isdir || startsWith(dirs(k).name,'.'), continue; end
        candidate=fullfile(out,dirs(k).name,[variant suffix]);
        pf=fullfile(out,dirs(k).name,'problem.mat');
        if ~isfile(candidate) || ~isfile(pf), continue; end
        s=load(pf,'c');
        if s.c.tol~=c.tol || ~isequal(s.c.queries,queries), continue; end
        s=load(pf,'ops');
        if isequaln(s.ops,ops), file=candidate; return; end
    end
end

function cs=exploration_protocol()
    base=protocol(); reference=base(end); cs=reference;
    for width=[0.02,0.05,0.10]
        for eta=[1e-3,1e-4,1e-5]
            if width==0.05 && eta==1e-5, continue; end
            c=reference;
            c.name=sprintf('explore_w%02d_n6_e%d',round(100*width),round(log10(eta)));
            [l,r]=ndgrid(linspace(1-width,1+width,6),linspace(0.8,0.95,6));
            c.training=[l(:),r(:)];
            [l,r]=ndgrid(1+width*[-0.74,-0.06,0.74],[0.813,0.871,0.938]);
            c.queries=[l(:),r(:)]; c.eta=eta*[1,1,1]; cs(end+1)=c; %#ok<AGROW>
        end
    end
    for n=[4,8]
        c=reference; c.name=sprintf('explore_w05_n%d_e-4',n);
        [l,r]=ndgrid(linspace(0.95,1.05,n),linspace(0.8,0.95,n));
        c.training=[l(:),r(:)]; c.eta=1e-4*[1,1,1]; cs(end+1)=c; %#ok<AGROW>
    end
    c=reference; c.name='explore_w05_n6_relaxed'; c.eta=[1e-5,1e-4,1e-3];
    cs(end+1)=c;
end

function cs=protocol()
    cs=struct([]);
    for scale=[1,1.5,2]
        c=struct('name',sprintf('frequency_%g',scale),'scale',scale, ...
            'training',[ones(24,1),linspace(0.8,0.95,24)'], ...
            'queries',[ones(5,1),[0.8053;0.8297;0.8761;0.9229;0.9431]], ...
            'variants',{{'global','dd'}},'eta',1e-5*[1,1,1],'tol',1e-10);
        if scale==1, c.variants={'global','dd','merged','signed','cartesian'}; end
        if isempty(cs), cs=c; else, cs(end+1)=c; end %#ok<AGROW>
    end
    c.name='two_parameter'; c.scale=1;
    [l,r]=ndgrid(linspace(0.95,1.05,6),linspace(0.8,0.95,6));
    c.training=[l(:),r(:)];
    [l,r]=ndgrid([0.963,0.997,1.037],[0.813,0.871,0.938]);
    c.queries=[l(:),r(:)]; c.variants={'global','dd'};
    cs(end+1)=c;
end

function part=partition(p,cfg,variant)
    part=[]; if strcmp(variant,'global'), return; end
    cfg=halfspace_paper_dd_config(cfg);
    if strcmp(variant,'merged'), cfg.halfspace_dd_pml_mode='merged'; end
    if strcmp(variant,'signed'), cfg.halfspace_dd_pml_mode='signed_depth'; end
    part=build_halfspace_refined_partition(p,cfg);
    if strcmp(variant,'cartesian')
        [i,j]=ndgrid(1:p.nx,1:p.ny);
        labels=1+min(3,floor(4*(i-1)/p.nx))+4*min(3,floor(4*(j-1)/p.ny));
        blocks=repmat(struct('name','','idx',[],'count',0),16,1);
        for k=1:16
            blocks(k).name=sprintf('tile_%d',k); blocks(k).idx=find(labels==k);
            blocks(k).count=numel(blocks(k).idx);
        end
        part.blocks=blocks; part.labels=uint16(labels); part.mode='cartesian4x4';
    end
    indices=vertcat(part.blocks.idx);
    assert(isequal(sort(indices),(1:p.Ntot)'),'Partition must cover once.');
end

function a=coefficients(mu,nops)
    a=[1,mu(2)^(-2)-1];
    if nops==3, a(3)=mu(1)^(-2)-1; end
end

function A=assemble(ops,mu)
    a=coefficients(mu,numel(ops)); A=a(1)*ops{1};
    for k=2:numel(ops), A=A+a(k)*ops{k}; end
end

function [model,t,audit]=train(ops,b,params,part,eta)
    alltime=tic; model=cell(0,1); audit=struct([]);
    t=struct('inverse',0,'replay',0,'pod',0,'assembly',0);
    for j=1:numel(eta)
        E=complex(zeros(numel(b),size(params,1))); worst=0;
        for k=1:size(params,1)
            tt=tic; A=assemble(ops,params(k,:));
            [~,Q]=replay(A,b,params(k,:),model);
            t.replay=t.replay+toc(tt); q=Q(:,end);
            tt=tic; E(:,k)=A\q; t.inverse=t.inverse+toc(tt);
            worst=max(worst,norm(A*E(:,k)-q));
        end
        assert(worst<1e-9,'Inverse-action snapshot failed.');
        tt=tic;
        if isempty(part)
            [U,S,~]=svd(E,'econ'); sig=diag(S);
            tails=sqrt([flipud(cumsum(flipud(sig(2:end).^2)));0])/norm(sig);
            r=find(tails<=eta(j),1); sp=rom_reduced_space_utils('make_dense_space',U(:,1:r));
        else
            sp=rom_reduced_space_utils('build_blockn_space',E,part, ...
                struct('rank_global_tail_tol',eta(j),'rank_global_tail_strategy','marginal_storage'));
        end
        t.pod=t.pod+toc(tt);
        tail2=0;
        for column=1:size(E,2)
            v=E(:,column);
            tail2=tail2+norm(v-rom_reduced_space_utils('lift',sp, ...
                rom_reduced_space_utils('project',sp,v)))^2;
        end
        tail=sqrt(tail2)/norm(E,'fro');
        assert(tail<=eta(j)*(1+1e-3),'Measured snapshot tail exceeds budget.');
        sp=deploy_space(sp); tt=tic; reduced=cell(size(ops));
        for k=1:numel(ops)
            reduced{k}=rom_reduced_space_utils('assemble_reduced_matrix',ops{k},sp);
        end
        t.assembly=t.assembly+toc(tt);
        model{j}=struct('space',sp,'operators',{reduced});
        one=struct('E',E,'tail',tail,'snapshot_residual',worst);
        if isempty(audit), audit=one; else, audit(j)=one; end
        fprintf('  layer %d: rank %d, tail %.3g, snapshot %.3g\n',j,sp.k,tail,worst);
    end
    t.total=toc(alltime);
end

function sp=deploy_space(sp)
    if strcmp(sp.type,'dense'), return; end
    sp=rmfield(sp,intersect(fieldnames(sp),{'partition','block_names','block_meta'}));
    sp.blocks=rmfield(sp.blocks,intersect(fieldnames(sp.blocks),{'meta','label','name'}));
end

function [x,Q,H,Z]=replay(A,b,mu,model)
    n=numel(b); beta=norm(b); k=numel(model);
    Q=complex(zeros(n,k+1)); Q(:,1)=b/beta;
    Z=complex(zeros(n,k)); H=complex(zeros(k+1,k)); x=zeros(n,1,'like',Q);
    for j=1:k
        l=model{j}; B=assemble(l.operators,mu);
        z=rom_reduced_space_utils('lift',l.space, ...
            B\rom_reduced_space_utils('project',l.space,Q(:,j)));
        Z(:,j)=z;
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*z,Q(:,1:j));
        g=zeros(j+1,1); g(1)=beta; x=Z(:,1:j)*(H(1:j+1,1:j)\g);
        if stop, Q=Q(:,1:j+1); return; end
    end
end

function st=storage(model)
    st=struct('basis',0,'operators',0,'index',0,'ranks',zeros(1,numel(model)));
    for j=1:numel(model)
        l=model{j}; sp=l.space; st.ranks(j)=sp.k;
        if strcmp(sp.type,'dense'), entries=numel(sp.basis);
        else
            entries=sum(arrayfun(@(b)numel(b.basis),sp.blocks));
            st.index=max(st.index,8*sum([sp.blocks.count])/2^20);
        end
        st.basis=st.basis+16*entries/2^20;
        st.operators=st.operators+16*sum(cellfun(@numel,l.operators))/2^20;
    end
    st.payload=st.basis+st.operators+st.index;
end

function rows=benchmark(ops,b,queries,model,variant,tol,repeats)
    rows=table();
    for k=1:size(queries,1)
        mu=queries(k,:); A=assemble(ops,mu);
        % A_base=K_pml+omega^2 I; right and left masks cover the full grid.
        w2=max(full(diag(ops{2}))); mass=w2*ones(numel(b),1);
        a=coefficients(mu,numel(ops));
        for i=2:numel(ops), mass=mass+a(i)*full(diag(ops{i})); end
        for rep=0:repeats
            tt=tic; iterations=0; pre=NaN; setup=0;
            if strcmp(variant,'direct')
                x=A\b;
            else
                if strcmp(variant,'shifted'), x=zeros(size(b));
                else, x=replay(A,b,mu,model); end
                pre=norm(b-A*x)/norm(b);
                if pre>tol
                    [M,mi]=build_shifted_laplacian_preconditioner( ...
                        A+1i*0.2*spdiags(mass,0,numel(b),numel(b)), ...
                        struct('solver','ilu','ilu_droptol',1e-3));
                    setup=mi.setup_time;
                    [x,si]=shifted_laplacian_gmres(@(v)A*v,M,b,x, ...
                        struct('tol',tol,'restart',40,'maxit',20,'reorthogonalize',true));
                    iterations=si.iter_count; clear M;
                end
            end
            rho=norm(b-A*x)/norm(b); elapsed=toc(tt);
            assert(rho<=tol,'Final true Helmholtz residual failed.');
            if rep>0
                row=table(string(variant),mu(1),mu(2),rep,elapsed,setup,iterations,pre,rho, ...
                    'VariableNames',{'variant','c_left','c_right','repetition','total_s', ...
                    'completion_setup_s','completion_iterations','post_ROM_residual','true_residual'});
                rows=[rows;row]; %#ok<AGROW>
            end
        end
        fprintf('  %s query %d/%d: %.4gs, %d completion, rho %.2g\n', ...
            variant,k,size(queries,1),elapsed,iterations,rho);
    end
end

function rows=diagnostics(ops,b,mu,model,audit,eta)
    A=assemble(ops,mu); [~,Q]=replay(A,b,mu,model); rows=table();
    % Kbar is a certified norm upper bound; Khat is a power-method estimate.
    Anormbar=sqrt(norm(A,1)*norm(A,inf));
    for j=1:2
        sp=model{j}.space; B=assemble(model{j}.operators,mu);
        U=complex(zeros(sp.n,sp.k));
        for column=1:sp.k
            unit=zeros(sp.k,1); unit(column)=1;
            U(:,column)=rom_reduced_space_utils('lift',sp,unit);
        end
        AU=A*U; AtU=A'*U;
        f=@(v)A*v-AU*(B\(U'*(A*v)));
        ft=@(v)A'*v-AtU*(B'\(AU'*v));
        rng(900+j); v=randn(numel(b),1)+1i*randn(numel(b),1); v=v/norm(v);
        for it=1:100, w=ft(f(v)); v=w/norm(w); end
        Khat=norm(f(v)); eigen_residual=norm(ft(f(v))-Khat^2*v)/Khat^2;
        Kbar=Anormbar*(1+Anormbar/min(svd(B)));
        q=Q(:,j); zstar=A\q; E=audit(j).E;
        a=pinv(E)*zstar; e=zstar-E*a;
        trunc=norm(E,'fro')*norm(a)*eta(j); transfer=norm(e);
        Chateta=Khat*trunc; deltahat=Khat*transfer;
        Cbareta=Kbar*trunc; deltabar=Kbar*transfer;
        z=U*(B\(U'*q)); defect=norm(A*z-q);
        % Directly applied decomposition avoids replacing every factor by a norm.
        compression=norm(f(E*a)); coverage=norm(f(e));
        assert(defect<=Cbareta+deltabar+1e-10);
        assert(defect<=compression+coverage+1e-9);
        rows=[rows;table(j,Khat,eigen_residual,Kbar,Chateta,deltahat,Cbareta, ...
            deltabar,compression,coverage,defect)]; %#ok<AGROW>
    end
end
