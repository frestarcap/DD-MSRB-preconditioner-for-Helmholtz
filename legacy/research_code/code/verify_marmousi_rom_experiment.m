function verify_marmousi_rom_experiment(mode)
% Independently check saved full fields and reduced operator actions.
    if nargin<1, mode='paired'; end
    old=maxNumCompThreads(1); cleanup=onCleanup(@()maxNumCompThreads(old));
    out=helm_output_path('outputs','marmousi_20260920',mode);
    s=load(fullfile(out,'results.mat'));
    p=s.p; cfg=s.cfg; rows=table(); space_rows=table();
    maxres=0; maxerr=0; nb=norm(p.b);
    for qi=1:numel(s.queries)
        q=s.queries(qi); velocity=p.v;
        if q.amplitude>0
            rng(q.seed,'twister'); noise=2*rand(size(p.v))-1;
            if strcmp(q.kind,'smooth')
                width=.25/cfg.h; t=-ceil(3*width):ceil(3*width);
                g=exp(-t.^2/(2*width^2)); g=g/sum(g);
                noise=conv2(g,g,noise,'same'); noise=noise/max(abs(noise(:)));
            end
            velocity=p.v.*(1+q.amplitude*noise);
        end
        np=cfg.npml; vel=velocity([ones(1,np),1:p.nxp,p.nxp*ones(1,np)], ...
            [ones(1,np),1:p.nzp,p.nzp*ones(1,np)]);
        mass=1./vel(:).^2; alpha=(2*pi*q.frequency)^2;
        A=p.K+spdiags(alpha*mass,0,p.n,p.n);
        direct=s.fields{qi,2}.u;
        for mi=1:numel(s.names)
            u=s.fields{qi,mi}.u; rho=norm(A*u-p.b)/nb;
            err=norm(u-direct)/norm(direct);
            recorded=s.rows(s.rows.query==qi & s.rows.method==string(s.names{mi}) & s.rows.repeat==1,:);
            assert(height(recorded)==1 && rho<=cfg.tol && abs(recorded.residual-rho)<1e-12);
            maxres=max(maxres,rho); maxerr=max(maxerr,err);
            rows=[rows;table(qi,string(s.names{mi}),rho,err, ...
                'VariableNames',{'query','method','residual','error'})]; %#ok<AGROW>
        end
        if qi==1 || qi==numel(s.queries)
            rng(9283,'twister');
            for mi=3:numel(s.models)
                for j=1:cfg.depth
                    l=s.models{mi}.layers{j}; sp=l.space;
                    x=randn(sp.k,1)+1i*randn(sp.k,1);
                    Mr=rom_reduced_space_utils('assemble_reduced_matrix',spdiags(mass,0,p.n,p.n),sp);
                    lhs=(l.Kr+alpha*Mr)*x;
                    rhs=rom_reduced_space_utils('project',sp,A*rom_reduced_space_utils('lift',sp,x));
                    action_error=norm(lhs-rhs)/max(norm(rhs),eps);
                    projected=zeros(size(l.snapshots));
                    if strcmp(sp.type,'dense')
                        projected=sp.basis*(sp.basis'*l.snapshots);
                    else
                        for bi=1:numel(sp.blocks)
                            ids=sp.blocks(bi).idx; U=sp.blocks(bi).basis;
                            projected(ids,:)=U*(U'*l.snapshots(ids,:));
                        end
                    end
                    tail=norm(l.snapshots-projected,'fro')/max(norm(l.snapshots,'fro'),realmin);
                    assert(action_error<1e-10 && tail<=cfg.tail*(1+1e-5), ...
                        'Space %s layer %d: action=%.3e tail=%.3e target=%.3e', ...
                        s.names{mi},j,action_error,tail,cfg.tail);
                    space_rows=[space_rows;table(qi,string(s.names{mi}),j,action_error,tail, ...
                        'VariableNames',{'query','method','layer','action_error','snapshot_tail'})]; %#ok<AGROW>
                end
            end
        end
    end
    storage=table(); ranks=table();
    for mi=3:numel(s.models)
        basis_bytes=0; reduced_bytes=0;
        for j=1:cfg.depth
            l=s.models{mi}.layers{j}; sp=l.space;
            reduced_bytes=reduced_bytes+16*(numel(l.Kr)+numel(l.Mr));
            if strcmp(sp.type,'dense')
                basis_bytes=basis_bytes+16*numel(sp.basis);
                ranks=[ranks;table(string(s.names{mi}),j,"global",p.n,sp.k, ...
                    'VariableNames',{'method','layer','region','dofs','rank'})]; %#ok<AGROW>
            else
                basis_bytes=basis_bytes+16*sum(arrayfun(@(b)numel(b.basis),sp.blocks));
                for bi=1:numel(sp.blocks)
                    b=sp.blocks(bi);
                    ranks=[ranks;table(string(s.names{mi}),j,string(b.name),numel(b.idx),b.k, ...
                        'VariableNames',{'method','layer','region','dofs','rank'})]; %#ok<AGROW>
                end
            end
        end
        % Explicit deployment accounting with one shared partition. Excludes
        % training snapshots, diagnostics, full operators and MATLAB headers.
        index_bytes=0;
        if strcmp(s.models{mi}.style,'dd'), index_bytes=8*p.n; end
        numerical_bytes=basis_bytes+reduced_bytes+index_bytes;
        storage=[storage;table(string(s.names{mi}),basis_bytes,reduced_bytes,index_bytes,numerical_bytes, ...
            'VariableNames',{'method','basis_bytes','reduced_bytes','shared_index_bytes','numerical_bytes'})]; %#ok<AGROW>
    end
    writetable(storage,fullfile(out,'storage.csv'));
    writetable(ranks,fullfile(out,'ranks.csv'));
    writetable(rows,fullfile(out,'verification.csv'));
    writetable(space_rows,fullfile(out,'space_verification.csv'));
    x=p.x; z=p.z; velocity=p.v; labels=p.labels;
    names=s.names; query_groups={s.queries.group}; frequencies=[s.queries.frequency];
    % Saved compact plotting bundle contains actual full-field histories.
    histories=cell(size(s.fields)); solution=reshape(s.fields{numel(cfg.tests),2}.u,p.nx,p.nz);
    solution=solution(np+(1:p.nxp),np+(1:p.nzp));
    for k=1:numel(s.fields)
        field=s.fields{k}; h=field.ROM_history(:); ch=field.completion_history(:);
        if isempty(ch), histories{k}=h; else, histories{k}=[h;ch(2:end)]; end
    end
    save(fullfile(out,'plot_data.mat'),'x','z','velocity','labels','solution','names', ...
        'query_groups','frequencies','histories','cfg','-v7');
    fprintf('MARMOUSI_VERIFY_PASS fields=%d max_residual=%.3e max_error=%.3e\n',height(rows),maxres,maxerr);
end
