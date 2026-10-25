function audit = audit_revision_evidence(mode)
%AUDIT_REVISION_EVIDENCE Independent checks of frozen paper artifacts.
    if nargin<1, mode='legacy'; end
    current=strcmp(mode,'revision20260907');
    repaired=strcmp(mode,'repaired') || current;
    assert(repaired || strcmp(mode,'legacy'));
    out = helm_output_path('outputs',['revision_audit_20260905_' mode]);
    if current, out=helm_output_path('outputs','revision_20260907','audit'); end
    if ~exist(out,'dir'), mkdir(out); end
    audit = struct('compression',table(),'stability',table(), ...
        'reference',table(),'direct',table(),'offline',table(),'solution',table());
    tagged = helm_output_path('outputs','helm_trajectory_evidence','greedy_adoption_20260905');
    if repaired, tagged=helm_output_path('outputs','helm_trajectory_evidence','review_repaired_20260905'); end
    for problem = ["halfspace","ecc","pc"]
        if problem == "halfspace"
            root = helm_output_path('outputs','halfspace_dd_refinement','greedy_paper_helm1e12_fulltiming');
            if repaired, root=helm_output_path('outputs','halfspace_dd_refinement','review_repaired_20260905'); end
            g=load(fullfile(root,'global.mat'),'artifact');
            d=load(fullfile(root,'slabs1-aligned-sym8.mat'),'artifact');
            pair={g.artifact,d.artifact};
            cfg=g.artifact.cfg; cfg.setup_only=true;
            setup=helm_waveholtz_correction_rom(cfg);
        else
            file = 'ecc_residual_ablation_paper.mat';
            if problem == "pc"
                file='pc_waveholtz_residual_ablation_paper.mat';
                if repaired, file='pc_residual_ablation_paper.mat'; end
            end
            source=fullfile(tagged,file);
            if current && problem=="pc"
                source=helm_output_path('outputs','revision_20260907','pc','paper_data.mat');
            end
            s=load(source,'global_results','dd_results');
            pair={s.global_results,s.dd_results};
        end
        if problem=="halfspace", dl=pair{2}.layers;
        else, dl=pair{2}.correction_rom.layers; end
        partition=dl{1}.basis_space.partition;
        for m=1:2
            r=pair{m}; style="Global"; if m==2, style="DD"; end
            if problem=="halfspace"
                layers=r.layers; params=r.cfg.train_params;
                testparams=r.test.params; rhs=r.rhs;
                Aof=@(k) setup.problem.A_base+(1/params(k)^2-1)*setup.problem.A_right;
                Atest=@(k) setup.problem.A_base+(1/testparams(k)^2-1)*setup.problem.A_right;
                bof=@(~) rhs;
            elseif problem=="pc"
                layers=r.correction_rom.layers; params=r.train_reference.params;
                testparams=r.test.params; rhs=r.rhs;
                Aof=@(k) r.problem.A_base+(1/params(k)^2-1)*r.problem.A_right;
                Atest=@(k) r.problem.A_base+(1/testparams(k)^2-1)*r.problem.A_right;
                bof=@(~) rhs;
            else
                layers=r.correction_rom.layers; params=r.train_reference.ratio;
                testparams=r.test.ratio;
                Aof=@(k) r.train_reference.systems{k}.A;
                Atest=@(k) r.test.systems{k}.A;
                bof=@(k) r.train_reference.systems{k}.b;
            end
            for j=1:numel(layers)
                E=layers{j}.eta_snapshots; eta=r.cfg.corr_pod_singular_value_tols(j);
                sv=svd(E,'econ'); rankg=tail_rank(sv,eta);
                opts=struct('rank_global_tail_tol',eta,'rank_global_tail_strategy','marginal_storage');
                sp=rom_reduced_space_utils('build_block3_space',E,partition,opts);
                if strcmp(sp.type,'blockn')
                    entries=sum(arrayfun(@(b) numel(b.basis),sp.blocks));
                    actualtail=sp.blocks(1).meta.global_relative_tail;
                else
                    entries=numel(sp.Un)+numel(sp.Uf)+numel(sp.Up);
                    actualtail=sp.block_meta.near.global_relative_tail;
                end
                ge=size(E,1)*rankg;
                audit.compression=[audit.compression;table(problem,style,j,eta,ge,entries, ...
                    100*(1-entries/ge),actualtail,'VariableNames', ...
                    {'problem','snapshot_trajectory','layer','budget','global_entries','greedy_entries','saving_percent','greedy_tail'})];
                conds=zeros(numel(testparams),1);
                for k=1:numel(testparams)
                    Ar=rom_reduced_space_utils('assemble_reduced_matrix',Atest(k),layers{j}.basis_space);
                    conds(k)=cond(Ar);
                end
                hs=nan(numel(params),1); defects=zeros(numel(params),1);
                direct_errors=defects; orth1=defects; orth2=defects;
                snapshot_change=defects;
                for k=1:numel(params)
                    A=Aof(k); b=bof(k); q=b/norm(b); Q=q;
                    Q2=q; q2=q; Z2=layers{1}.eta_snapshots(:,k);
                    for l=1:j-1
                        sp0=layers{l}.basis_space;
                        z=rom_apply(A,sp0,q,layers{l},params(k));
                        w=A*z;
                        if repaired
                            [q,~,hs(k),stop]=rom_arnoldi_step(w,Q);
                            assert(~stop);
                        else
                            for i=1:l
                                h=Q(:,i)'*w; w=w-Q(:,i)*h;
                            end
                            hs(k)=norm(w); q=w/hs(k);
                        end
                        Q(:,l+1)=q;
                        z2=rom_apply(A,sp0,q2,layers{l},params(k)); w2=A*z2; h2=zeros(l,1);
                        for pass=1:2
                            for i=1:l
                                dh=Q2(:,i)'*w2; h2(i)=h2(i)+dh;
                                w2=w2-Q2(:,i)*dh;
                            end
                        end
                        hn=norm(w2); q2=w2/hn;
                        Z2(:,l+1)=(z2-Z2(:,1:l)*h2)/hn;
                        Q2(:,l+1)=q2;
                    end
                    zsaved=E(:,k); zexact=A\q;
                    defects(k)=norm(A*zsaved-q);
                    direct_errors(k)=norm(zsaved-zexact)/norm(zexact);
                    orth1(k)=norm(Q'*Q-eye(j),2); orth2(k)=norm(Q2'*Q2-eye(j),2);
                    snapshot_change(k)=norm(Z2(:,j)-zsaved)/norm(zsaved);
                end
                audit.stability=[audit.stability;table(problem,style,j,max(conds), ...
                    min(hs),max(defects),max(direct_errors),max(orth1),max(orth2),max(snapshot_change), ...
                    'VariableNames',{'problem','method','layer','max_reduced_condition','min_hsub', ...
                    'max_snapshot_residual','max_snapshot_direct_error','max_orthogonality_single', ...
                    'max_orthogonality_double','max_double_snapshot_change'})];
            end
            ref_res=zeros(numel(params),1); ref_err=ref_res; flags=0;
            for k=1:numel(params)
                A=Aof(k); b=bof(k);
                if problem=="halfspace"
                    u=layers{1}.eta_snapshots(:,k)*norm(b);
                elseif problem=="pc"
                    u=r.train_reference.u_ref_snapshots(:,k);
                else
                    u=r.train_reference.u_ref(:,k);
                end
                ref_res(k)=norm(A*u-b)/norm(b);
                exact=A\b; ref_err(k)=norm(u-exact)/norm(exact);
            end
            if problem=="pc", flags=nnz(r.train_reference.flag); end
            audit.reference=[audit.reference;table(problem,style,flags,max(ref_res),max(ref_err), ...
                'VariableNames',{'problem','method','failed_native_reference_flags','max_reference_H_residual','max_reference_error'})];
            offline=NaN;
            if isfield(r,'timing'), offline=r.timing.offline; end
            audit.offline=[audit.offline;table(problem,style,offline, ...
                'VariableNames',{'problem','method','total_offline_s'})];
            residuals=zeros(numel(testparams),1);
            for k=1:numel(testparams)
                A=Atest(k);
                if problem=="ecc", b=r.test.systems{k}.b; else, b=rhs; end
                residuals(k)=norm(A*r.test.online_final_u(:,k)-b)/norm(b);
            end
            audit.solution=[audit.solution;table(problem,style,max(residuals), ...
                max(r.test.online_final_err_helm_full),'VariableNames', ...
                {'problem','method','max_true_H_residual','max_solution_error'})];
            if m==1
                A=Atest(1);
                if problem=="ecc", b=r.test.systems{1}.b; else, b=rhs; end
                u=A\b;
                times=zeros(7,1);
                for rep=1:7, t=tic; u=A\b; times(rep)=toc(t); end
                audit.direct=[audit.direct;table(problem,numel(b),testparams(1),mean(times),std(times), ...
                    norm(A*u-b)/norm(b),r.test.online_total_time(1),NaN,'VariableNames', ...
                    {'problem','N','parameter','mean_direct_s','std_direct_s','relative_residual', ...
                    'global_online_s','dd_online_s'})];
            else
                audit.direct.dd_online_s(end)=r.test.online_total_time(1);
            end
            save(fullfile(out,'audit.mat'),'audit','-v7.3');
            fprintf('Audited %s %s\n',problem,style);
        end
        clear pair r g d s setup;
    end
    for name=fieldnames(audit)'
        writetable(audit.(name{1}),fullfile(out,[name{1} '.csv']));
        disp(audit.(name{1}));
    end
end

function r=tail_rank(s,tol)
    tail=[flipud(cumsum(flipud(abs(s(:)).^2)));0];
    r=find(tail<=tol^2*sum(abs(s).^2),1)-1;
end

function z=rom_apply(A,sp,q,layer,param)
    if isfield(layer,'Ar_base')
        Ar=layer.Ar_base+(1/param^2-1)*layer.Ar_right;
    else
        Ar=rom_reduced_space_utils('assemble_reduced_matrix',A,sp);
    end
    c=Ar\rom_reduced_space_utils('project',sp,q);
    z=rom_reduced_space_utils('lift',sp,c);
end
