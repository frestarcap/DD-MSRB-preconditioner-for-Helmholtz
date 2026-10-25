function result = run_marmousi_rom_experiment(mode, reuse_dir)
%RUN_MARMOUSI_ROM_EXPERIMENT Independent Helm workflow on gridded Marmousi.
% Reuses the production PML, verified TA replay, reduced-space and shifted
% GMRES routines. Constant-density acoustic physics; no anisotropy claim.
% Download the input with scripts/reproduce_marmousi.sh first.
    if nargin<1, mode='paired'; end
    if nargin<2, reuse_dir=''; end
    root=helm_output_path('outputs','marmousi_20260920');
    out=fullfile(root,mode); if ~exist(out,'dir'), mkdir(out); end
    cfg=struct('h',.02,'npml',25,'ntrain',32,'depth',3,'tail',1e-5, ...
        'frequencies',linspace(3,5,32),'tests',[3.137,3.523,4.117,4.487,4.863], ...
        'fixed_frequency',4.117,'repetitions',3,'tol',1e-8,'beta',.2, ...
        'droptol',1e-3,'seed',20260920,'random_train',true,'shared_setup_timing',false);
    if strcmp(mode,'smoke')
        cfg.h=.04; cfg.npml=12; cfg.ntrain=10; cfg.frequencies=linspace(1.5,2.3,10);
        cfg.tests=[1.637,2.173]; cfg.fixed_frequency=2.173;
        cfg.repetitions=1; cfg.random_train=false;
    elseif strcmp(mode,'compact')
        cfg.tail=1e-3; cfg.random_train=false; cfg.droptol=1e-4;
        cfg.shared_setup_timing=true;
    elseif strcmp(mode,'paired')
        cfg.droptol=1e-4; cfg.shared_setup_timing=true;
    elseif strcmp(mode,'strong')
        cfg.droptol=1e-4;
    else
        assert(strcmp(mode,'pilot'),'Use smoke, pilot, strong, paired, or compact.');
    end
    old=maxNumCompThreads(1); cleanup=onCleanup(@()maxNumCompThreads(old));
    rng(cfg.seed,'twister'); wall=tic;
    input=fullfile(root,'data','vp_marmousi_bi');
    fid=fopen(input,'r','ieee-le'); assert(fid>=0,'Missing Marmousi input.');
    v=fread(fid,[401,1601],'single=>double').'; fclose(fid);
    assert(isequal(size(v),[1601,401]) && all(isfinite(v(:))));
    % Devito's model uses km/s and 7.5 m spacing (12 km by 3 km).
    assert(min(v(:))>=1 && max(v(:))<10,'Unexpected velocity units.');
    p=problem(v,cfg); fprintf('MARMOUSI %s N=%d, physical=%dx%d, h=%.1fm, c=[%.3f,%.3f]km/s\n', ...
        mode,p.n,p.nxp,p.nzp,1000*cfg.h,min(p.v(:)),max(p.v(:)));
    assert(min(p.v(:))/(max(cfg.frequencies)*cfg.h)>=10,'Insufficient PPW.');
    names={'shifted','direct','frequency_global','frequency_dd'};
    models={[],[],[],[]}; trainrows=table();
    families={'frequency'}; if cfg.random_train, families{end+1}='iid5'; end
    if ~isempty(reuse_dir)
        if cfg.random_train, names=[names,{'iid5_global','iid5_dd'}]; end
        for mi=3:numel(names)
            cached=load(fullfile(reuse_dir,[names{mi} '_model.mat']),'model','cfg');
            for key={'h','npml','tail','ntrain','depth','frequencies','fixed_frequency','seed'}
                assert(isequaln(cached.cfg.(key{1}),cfg.(key{1})),'Cached training mismatch.');
            end
            models{mi}=cached.model;
        end
        trainrows=readtable(fullfile(reuse_dir,'training.csv'));
        writetable(trainrows,fullfile(out,'training.csv'));
        fprintf('Reusing verified-equation training from %s\n',reuse_dir);
    else
    for fi=1:numel(families)
        family=families{fi}; systems=cell(cfg.ntrain,1); solutions=zeros(p.n,cfg.ntrain);
        t=tic;
        for k=1:cfg.ntrain
            if fi==1
                freq=cfg.frequencies(k); velocity=p.v;
            else
                freq=cfg.fixed_frequency; velocity=perturb(p,.05,'iid',cfg.seed+1000+k);
            end
            mass=1./extend(velocity,cfg.npml).^2;
            A=p.K+(2*pi*freq)^2*spdiags(mass(:),0,p.n,p.n);
            systems{k}=A; solutions(:,k)=A\p.b;
            assert(norm(A*solutions(:,k)-p.b)/norm(p.b)<1e-10);
        end
        reference_s=toc(t);
        for style={'global','dd'}
            t=tic; model=train(systems,solutions,p,cfg,style{1}); training_s=toc(t);
            model.reference_s=reference_s; model.training_s=training_s;
            method=[family '_' style{1}];
            if fi==1, mi=2+find(strcmp(style{1},{'global','dd'}));
            else, mi=numel(names)+1; names{mi}=method; end
            models{mi}=model;
            ranks=cellfun(@(x)x.space.k,model.layers);
            entries=sum(cellfun(@(x)basis_entries(x.space),model.layers));
            trainrows=[trainrows;table(string(method),cfg.ntrain,reference_s,training_s, ...
                reference_s+training_s,entries,max(ranks),model.max_snapshot_residual, ...
                'VariableNames',{'method','ntrain','reference_s','training_s','offline_s', ...
                'basis_entries','max_rank','max_snapshot_residual'})]; %#ok<AGROW>
            save(fullfile(out,[method '_model.mat']),'model','cfg','-v7.3','-nocompression');
            writetable(trainrows,fullfile(out,'training.csv'));
            fprintf('TRAIN %s ranks=%s, entries=%d, offline=%.2fs\n', ...
                method,mat2str(ranks),entries,reference_s+training_s);
        end
        clear systems solutions;
    end
    end
    % Query identifiers and seeds are fixed before timing and never selected
    % according to solver outcome. No test frequency occurs in training.
    queries=struct('group',{},'frequency',{},'amplitude',{},'kind',{},'seed',{});
    for f=cfg.tests
        assert(all(abs(f-cfg.frequencies)>1e-10));
        queries(end+1)=query('frequency',f,0,'none',0); %#ok<AGROW>
    end
    for kind={'iid','smooth'}
        for amplitude=[.01,.05,.10]
            for seed=1:2
                queries(end+1)=query(sprintf('%s%d',kind{1},round(100*amplitude)), ...
                    cfg.fixed_frequency,amplitude,kind{1},cfg.seed+2000+seed); %#ok<AGROW>
            end
        end
    end
    save(fullfile(out,'checkpoint.mat'),'p','cfg','names','queries','trainrows','-v7.3');
    rows=table(); fields=cell(numel(queries),numel(names));
    opts=struct('tol',cfg.tol,'restart',40,'maxit',30,'reorthogonalize',true, ...
        'orthogonalization','cgs2');
    for qi=1:numel(queries)
        q=queries(qi); velocity=perturb(p,q.amplitude,q.kind,q.seed);
        mass=1./extend(velocity,cfg.npml).^2;
        mass=mass(:); alpha=(2*pi*q.frequency)^2;
        A=p.K+alpha*spdiags(mass,0,p.n,p.n); ref=A\p.b;
        ppw=min(velocity(:))/(q.frequency*cfg.h);
        assert(ppw>=10,'Query has fewer than 10 points per wavelength.');
        for rep=0:cfg.repetitions
            if cfg.shared_setup_timing
                tt=tic;
                Ashift=A+1i*cfg.beta*alpha*spdiags(mass,0,p.n,p.n);
                shared_M=build_shifted_laplacian_preconditioner(Ashift, ...
                    struct('solver','ilu','ilu_droptol',cfg.droptol));
                shared_setup_s=toc(tt); clear Ashift;
            end
            order=circshift(1:numel(names),[0,mod(qi+rep,numel(names))]);
            for mi=order
                t=tic; rom_s=0; setup_s=0; solve_s=0; it=0; ch=[]; rh=1;
                if mi==2
                    tt=tic; u=A\p.b; solve_s=toc(tt); post=NaN; flag=0;
                else
                    u=zeros(p.n,1); flag=0;
                    if mi>2
                        tt=tic; [u,rh]=prefix(A,p.b,models{mi},alpha,mass,p.mass,cfg.tol);
                        rom_s=toc(tt);
                    end
                    post=norm(p.b-A*u)/norm(p.b);
                    if post>cfg.tol
                        if cfg.shared_setup_timing
                            M=shared_M; setup_s=shared_setup_s;
                        else
                        tt=tic;
                        Ashift=A+1i*cfg.beta*alpha*spdiags(mass,0,p.n,p.n);
                        M=build_shifted_laplacian_preconditioner(Ashift, ...
                            struct('solver','ilu','ilu_droptol',cfg.droptol));
                        setup_s=toc(tt);
                        end
                        tt=tic;
                        [u,info]=shifted_laplacian_gmres(@(x)A*x,M,p.b,u,opts);
                        solve_s=toc(tt); it=info.iter_count; flag=info.flag; ch=info.relres_hist;
                        clear M Ashift;
                    end
                end
                rho=norm(p.b-A*u)/norm(p.b); total_s=toc(t);
                if cfg.shared_setup_timing, total_s=total_s+setup_s; end
                err=norm(u-ref)/norm(ref); accepted=(flag==0 && rho<=cfg.tol);
                assert(accepted,'Solver failed: %s, query %d, residual %.3e',names{mi},qi,rho);
                if rep>0
                    row=table(string(names{mi}),qi,string(q.group),q.frequency,rep, ...
                        total_s,rom_s,setup_s,solve_s,it,post,rho,err,ppw,accepted, ...
                        'VariableNames',{'method','query','group','frequency','repeat', ...
                        'total_s','ROM_s','setup_s','solve_s','iterations','post_ROM', ...
                        'residual','error','ppw','accepted'});
                    rows=[rows;row]; %#ok<AGROW>
                end
                if rep==1
                    fields{qi,mi}=struct('u',u,'ROM_history',rh,'completion_history',ch);
                end
            end
        end
        writetable(rows,fullfile(out,'timings.csv'));
        fprintf('QUERY %d/%d %s %.3fHz passed; shifted/global/dd %.3f/%.3f/%.3fs\n', ...
            qi,numel(queries),q.group,q.frequency, ...
            mean(rows.total_s(rows.query==qi & rows.method=="shifted")), ...
            mean(rows.total_s(rows.query==qi & rows.method=="frequency_global")), ...
            mean(rows.total_s(rows.query==qi & rows.method=="frequency_dd")));
    end
    elapsed_s=toc(wall); matlab_version=version;
    save(fullfile(out,'results.mat'),'rows','fields','p','cfg','models','names','queries', ...
        'trainrows','elapsed_s','matlab_version','-v7.3','-nocompression');
    result=struct('rows',rows,'training',trainrows,'elapsed_s',elapsed_s);
    fprintf('MARMOUSI_%s_PASS elapsed=%.2fs\n',upper(mode),elapsed_s);
end

function p=problem(v,cfg)
    h=cfg.h; np=cfg.npml;
    p.x=(h:h:12-h).'; p.z=(h:h:3-h).';
    [X,Z]=ndgrid(p.x,p.z);
    F=griddedInterpolant({(0:1600)'*.0075,(0:400)'*.0075},v,'linear','nearest');
    p.v=F(X,Z); p.nxp=numel(p.x); p.nzp=numel(p.z);
    p.nx=p.nxp+2*np; p.nz=p.nzp+2*np; p.n=p.nx*p.nz;
    sx=stretch(p.nx,np); sz=stretch(p.nz,np);
    pml=struct('s_x',sx,'s_y',sz,'s_x_half',(sx(1:end-1)+sx(2:end))/2, ...
        's_y_half',(sz(1:end-1)+sz(2:end))/2);
    p.K=pml_stiffness_2d(p.nx,p.nz,h,h,pml);
    p.mass=1./extend(p.v,np).^2; p.mass=p.mass(:);
    p.b=zeros(p.nx,p.nz); [~,ix]=min(abs(p.x-6)); [~,iz]=min(abs(p.z-.1));
    p.b(ix+np,iz+np)=1; p.b=p.b(:);
    % Three geological columns; four separate absorbing sides. Covers every
    % unknown exactly once; top/bottom include corner nodes.
    label=zeros(p.nx,p.nz); label(1:np,:)=4; label(end-np+1:end,:)=5;
    label(:,1:np)=6; label(:,end-np+1:end)=7;
    for k=1:3
        ix=find(p.x>=(k-1)*4 & p.x<k*4);
        label(np+ix,np+(1:p.nzp))=k;
    end
    assert(all(label(:)>0)); p.labels=label;
    p.partition=struct('n',p.n,'blocks',struct('idx',{},'name',{}));
    for k=1:7
        p.partition.blocks(k)=struct('idx',find(label(:)==k),'name',sprintf('region%d',k));
    end
end

function s=stretch(n,np)
    sigma=zeros(n+2,1);
    for j=1:np
        sigma(j)=2.5*((np-j+1)/np)^2; sigma(end+1-j)=sigma(j);
    end
    s=1+1i*sigma;
end

function v=extend(v,np)
    v=v([ones(1,np),1:size(v,1),size(v,1)*ones(1,np)], ...
        [ones(1,np),1:size(v,2),size(v,2)*ones(1,np)]);
end

function velocity=perturb(p,a,kind,seed)
    if a==0, velocity=p.v; return; end
    rng(seed,'twister'); noise=2*rand(size(p.v))-1;
    if strcmp(kind,'smooth')
        % 250 m Gaussian correlation scale, normalized to the requested
        % maximum relative perturbation. Still generated on the full grid.
        h=p.x(2)-p.x(1); s=.25/h; t=(-ceil(3*s):ceil(3*s));
        g=exp(-t.^2/(2*s^2)); g=g/sum(g);
        noise=conv2(g,g,noise,'same'); noise=noise/max(abs(noise(:)));
    end
    velocity=p.v.*(1+a*noise);
end

function q=query(group,f,a,kind,seed)
    q=struct('group',group,'frequency',f,'amplitude',a,'kind',kind,'seed',seed);
end

function model=train(systems,solutions,p,cfg,style)
    ns=numel(systems); Q=cell(ns,1); E=cell(ns,1);
    for k=1:ns, Q{k}=p.b/norm(p.b); E{k}=solutions(:,k)/norm(p.b); end
    model=struct('layers',{{}},'max_snapshot_residual',0,'style',style);
    for j=1:cfg.depth
        S=zeros(p.n,ns);
        for k=1:ns, S(:,k)=E{k}(:,j); end
        norms=vecnorm(S); S=S./max(norms,realmin);
        if strcmp(style,'global')
            [U,D,~]=svd(S,'econ'); d=diag(D);
            if norm(d)==0
                rank=0;
            else
                energy_tail=[flipud(cumsum(flipud(d.^2)));0];
                tail=sqrt(energy_tail(2:end)/energy_tail(1));
                rank=find(tail<=cfg.tail,1);
            end
            sp=rom_reduced_space_utils('make_dense_space',U(:,1:rank));
            singular_values=d;
        else
            tc=struct('rank_rtol',1e-13,'rank_global_tail_tol',cfg.tail, ...
                'rank_global_tail_strategy','marginal_storage');
            sp=rom_reduced_space_utils('build_blockn_space',S,p.partition,tc);
            singular_values=[];
        end
        Kr=rom_reduced_space_utils('assemble_reduced_matrix',p.K,sp);
        Mr=rom_reduced_space_utils('assemble_reduced_matrix',spdiags(p.mass,0,p.n,p.n),sp);
        layer=struct('space',sp,'Kr',Kr,'Mr',Mr,'singular_values',singular_values, ...
            'snapshots',S,'diagnostics',{{}});
        if j<cfg.depth
            for k=1:ns
                A=systems{k}; Ar=rom_reduced_space_utils('assemble_reduced_matrix',A,sp);
                z=inverse(sp,Ar,Q{k}(:,j));
                [qn,en,~,~,diagnostic]=rom_verified_snapshot_step(A,z,Q{k},E{k});
                Q{k}(:,j+1)=qn; E{k}(:,j+1)=en;
                layer.diagnostics{k}=diagnostic;
                if ~diagnostic.breakdown
                    model.max_snapshot_residual=max(model.max_snapshot_residual,diagnostic.accepted_residual);
                end
            end
        end
        model.layers{j}=layer;
        fprintf('  %s layer %d rank=%d\n',style,j,sp.k);
    end
end

function z=inverse(sp,Ar,q)
    if sp.k==0, z=zeros(size(q)); return; end
    br=rom_reduced_space_utils('project',sp,q);
    if rcond(full(Ar))<1e-12, coeff=pinv(full(Ar))*br; else, coeff=Ar\br; end
    z=rom_reduced_space_utils('lift',sp,coeff);
end

function [u,hist]=prefix(A,b,model,alpha,mass,base_mass,tol)
    n=numel(b); depth=numel(model.layers); nb=norm(b);
    Q=zeros(n,depth+1); Z=zeros(n,depth); H=zeros(depth+1,depth);
    Q(:,1)=b/nb; hist=1; u=zeros(n,1); changed=~isequal(mass,base_mass);
    for j=1:depth
        layer=model.layers{j}; sp=layer.space;
        if changed
            Mr=rom_reduced_space_utils('assemble_reduced_matrix',spdiags(mass,0,n,n),sp);
        else
            Mr=layer.Mr;
        end
        Z(:,j)=inverse(sp,layer.Kr+alpha*Mr,Q(:,j));
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*Z(:,j),Q(:,1:j));
        rhs=zeros(j+1,1); rhs(1)=nb;
        u=Z(:,1:j)*lsqminnorm(H(1:j+1,1:j),rhs);
        hist(end+1,1)=norm(b-A*u)/nb; %#ok<AGROW>
        if stop || hist(end)<=tol, break; end
    end
end

function n=basis_entries(sp)
    if strcmp(sp.type,'dense'), n=numel(sp.basis);
    else, n=sum(arrayfun(@(b)numel(b.basis),sp.blocks)); end
end
