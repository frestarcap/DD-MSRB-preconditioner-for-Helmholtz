function run_jcp_revision_rank_policy(mode)
% New rank allocation ablation; no manuscript or active solver edits.
% train: complete independent replay for three policies at two budgets.
% timing: frozen models, one warmup + five rotated-order measured repeats.
% smoke: smaller numerical check; paper: train followed by timing.
    if nargin<1, mode='paper'; end
    assert(ismember(mode,{'smoke','train','timing','paper'}));
    previous=maxNumCompThreads(1); cleanup=onCleanup(@()maxNumCompThreads(previous)); %#ok<NASGU>
    out=helm_output_path('outputs','jcp_revision_experiments_20260914','rank_policy');
    if strcmp(mode,'smoke'), out=fullfile(out,'smoke'); end
    raw=fullfile(out,'raw'); generated=fullfile(raw,'generated');
    if ~isfolder(generated), mkdir(generated); end
    policies={'marginal_storage','uniform_relative','energy_greedy'};
    etas=[1e-5,1e-4];
    cfg=base_config();
    if strcmp(mode,'smoke')
        cfg.Nphys=31; cfg.omega=13.375; cfg.sigma_max=20;
        cfg.train_params=linspace(.8,.95,6); cfg.test_params=[.8130,.9072]; etas=1e-4;
    end
    % Copy the active self-contained function into ignored output and rename
    % only its function and utility calls. This avoids changing its defaults.
    source=which('helm_waveholtz_correction_rom'); source_text=fileread(source);
    clone=strrep(source_text,'function results = helm_waveholtz_correction_rom(cfg, frozen)', ...
        'function results = jcp_revision_rank_helm_generated(cfg, frozen)');
    clone=strrep(clone,'rom_reduced_space_utils(', 'jcp_revision_rank_space_utils(');
    generated_file=fullfile(generated,'jcp_revision_rank_helm_generated.m');
    if ~strcmp(mode,'timing')
        write_text(generated_file,clone);
        write_text(fullfile(raw,'active_helm_source.m.txt'),source_text);
    else
        assert(strcmp(fileread(fullfile(raw,'active_helm_source.m.txt')),source_text), ...
            'Active Helm source changed after training; inspect before timing.');
    end
    addpath(generated); path_cleanup=onCleanup(@()rmpath(generated)); %#ok<NASGU>
    manifests=table(); compression=table(); models={}; names={}; budgets=[];
    for eta=etas
        for pi=1:numel(policies)
            policy=policies{pi}; name=sprintf('%s_eta%d',policy,round(log10(eta)));
            file=fullfile(raw,[name '.mat']);
            if ~isfile(file)
                assert(~strcmp(mode,'timing'),'Training checkpoint missing.');
                c=cfg; c.corr_pod_singular_value_tols=eta*[1,1,1];
                c=halfspace_paper_dd_config(c); c.corr_block3_global_tail_strategy=policy;
                fprintf('JCP_RANK_TRAIN %s\n',name);
                r=jcp_revision_rank_helm_generated(c);
                [model,storage]=deploy_model(r.correction_rom.layers);
                matrix_base=r.problem.A_base; matrix_right=r.problem.A_right; rhs=r.rhs;
                references=r.test.helm_u_ref; expected=r.test.online_precorr_u;
                config=r.cfg; offline_s=r.timing.offline;
                block_rows=inspect_compression(r,name,eta);
                selfcheck=check_prefixes(model,matrix_base,matrix_right,rhs,config.test_params, ...
                    expected,config.helm_gmres_track_tol);
                s=rom_deployment_storage(r.correction_rom.layers);
                manifest=table(string(name),string(policy),eta,r.problem.Ntot, ...
                    numel(c.train_params),numel(c.test_params),string(mat2str(s.ranks)), ...
                    sum(s.ranks),s.basis_MiB,s.payload_MiB,storage/2^20,offline_s, ...
                    max(block_rows.aggregate_relative_tail),selfcheck, ...
                    'VariableNames',{'candidate','policy','budget','N','training_count', ...
                    'query_count','ranks','total_rank','basis_MiB','payload_MiB', ...
                    'model_object_MiB','offline_s','max_aggregate_tail','max_prefix_difference'});
                common_snapshots=table();
                if strcmp(policy,'marginal_storage')
                    common_snapshots=common_snapshot_comparison(r,policies,name,eta);
                end
                save(file,'model','config','matrix_base','matrix_right','rhs','references', ...
                    'expected','offline_s','manifest','block_rows','common_snapshots','-v7.3');
                clear r;
            end
            d=load(file,'manifest','block_rows'); manifests=[manifests;d.manifest]; %#ok<AGROW>
            compression=[compression;d.block_rows]; %#ok<AGROW>
            models{end+1}=file; names{end+1}=name; budgets(end+1)=eta; %#ok<AGROW>
            writetable(manifests,fullfile(out,'models.csv'));
            writetable(compression,fullfile(out,'compression.csv'));
        end
    end
    if strcmp(mode,'train'), fprintf('JCP_RANK_TRAIN_PASS\n'); return; end
    if ismember(mode,{'smoke','timing','paper'})
        repetitions=5; if strcmp(mode,'smoke'), repetitions=1; end
        measure_models(models,names,budgets,repetitions,out);
    end
end

function cfg=base_config()
    cfg=struct('Nphys',125,'omega',53.5,'sigma_max',80, ...
        'material_model','halfspace','reference_mode','helm_lift','rom_check_target','helm', ...
        'online_completion_mode','shifted_laplacian','use_direct_rom_init',false, ...
        'build_direct_rom_when_unused',false,'run_wave_baselines',false, ...
        'run_nr_rom_comparison',false,'do_plots',false,'save_plots',false,'save_benchmark',false, ...
        'parallel_train',false,'parallel_test',false,'train_params',linspace(.8,.95,24), ...
        'test_params',[.8031,.8130,.8297,.8513,.8761,.9007,.9229,.9431,.9473], ...
        'corr_num_layers',3,'helm_baseline_iters',1, ...
        'online_helm_accept_tol',1e-12,'online_true_helm_tol',1e-12, ...
        'shifted_laplacian_solver','ilu','shifted_laplacian_ilu_droptol',1e-3, ...
        'shifted_laplacian_tol',1e-12,'shifted_laplacian_restart',40, ...
        'shifted_laplacian_maxit',20,'shifted_laplacian_beta',.2);
end

function [model,bytes]=deploy_model(layers)
    model=struct('layers',{cell(size(layers))},'shared_indices',{{}});
    for j=1:numel(layers)
        sp=layers{j}.basis_space;
        blocks=struct('basis',{},'k',{});
        for b=1:numel(sp.blocks)
            if j==1, model.shared_indices{b}=sp.blocks(b).idx;
            else, assert(isequal(model.shared_indices{b},sp.blocks(b).idx)); end
            blocks(b)=struct('basis',sp.blocks(b).basis,'k',sp.blocks(b).k);
        end
        space=struct('type','blockn','n',sp.n,'k',sp.k,'blocks',blocks);
        model.layers{j}=struct('space',space,'Ar_base',layers{j}.Ar_base,'Ar_right',layers{j}.Ar_right);
    end
    w=whos('model'); bytes=w.bytes;
end

function rows=inspect_compression(r,name,eta)
    rows=table();
    for j=1:numel(r.correction_rom.layers)
        layer=r.correction_rom.layers{j}; sp=layer.basis_space; S=layer.eta_snapshots;
        total=norm(S,'fro')^2; tail=0;
        for b=1:numel(sp.blocks)
            z=sp.blocks(b); residual=S(z.idx,:)-z.basis*(z.basis'*S(z.idx,:));
            tail=tail+norm(residual,'fro')^2;
        end
        aggregate=sqrt(tail/total); assert(aggregate<=eta*(1+1e-7));
        for b=1:numel(sp.blocks)
            z=sp.blocks(b);
            rows=[rows;table(string(name),j,string(z.name),numel(z.idx),z.k, ...
                z.meta.local_relative_tail,aggregate,eta, ...
                'VariableNames',{'candidate','layer','block','block_dofs','rank', ...
                'local_relative_tail','aggregate_relative_tail','budget'})]; %#ok<AGROW>
        end
    end
end

function rows=common_snapshot_comparison(r,policies,name,eta)
    rows=table();
    for j=1:numel(r.correction_rom.layers)
        l=r.correction_rom.layers{j};
        for pi=1:numel(policies)
            c=struct('rank_global_tail_tol',eta,'rank_global_tail_strategy',policies{pi},'rank_rtol',eta);
            sp=jcp_revision_rank_space_utils('build_block3_space',l.eta_snapshots,l.basis_space.partition,c);
            bytes=16*sum(arrayfun(@(b)numel(b.basis),sp.blocks));
            rows=[rows;table(string(name),j,string(policies{pi}),sp.k,bytes/2^20, ...
                sp.blocks(1).meta.global_relative_tail,'VariableNames', ...
                {'snapshot_source','layer','allocation_policy','rank','basis_MiB','aggregate_tail'})]; %#ok<AGROW>
        end
    end
end

function delta=check_prefixes(model,A0,A1,b,params,expected,track_tol)
    delta=0;
    for qi=1:numel(params)
        A=A0+(params(qi)^(-2)-1)*A1;
        u=prefix(A,b,params(qi),model,track_tol);
        one=norm(u-expected(:,qi))/max(norm(expected(:,qi)),eps);
        delta=max(delta,one); assert(one<1e-9,'Independent ROM prefix mismatch.');
    end
end

function measure_models(files,names,budgets,repetitions,out)
    d=load(files{1}); A0=d.matrix_base; A1=d.matrix_right; b=d.rhs;
    refs=d.references; params=d.config.test_params; omega=d.config.omega;
    track_tol=d.config.helm_gmres_track_tol; tol=d.config.online_helm_accept_tol;
    common=table(); bank=cell(size(files));
    for mi=1:numel(files)
        v=load(files{mi},'model','config','common_snapshots'); bank{mi}=v.model;
        assert(isequal(v.config.test_params,params)); common=[common;v.common_snapshots]; %#ok<AGROW>
    end
    writetable(common,fullfile(out,'common_snapshot_allocation.csv'));
    names=[names,{'shifted','direct'}]; budgets=[budgets,NaN,NaN];
    opts=struct('tol',tol,'restart',40,'maxit',20,'reorthogonalize',true);
    ilu_opts=struct('solver','ilu','ilu_droptol',1e-3); beta=.2;
    rows=table(); fields=cell(numel(params),numel(names));
    for qi=1:numel(params)
        mu=params(qi); A=A0+(mu^(-2)-1)*A1; nb=norm(b);
        mass=omega^2*ones(size(b))+(mu^(-2)-1)*full(diag(A1));
        assert(norm(b-A*refs(:,qi))/nb<=tol);
        for rep=0:repetitions
            order=circshift(1:numel(names),[0,mod(qi+rep,numel(names))]);
            for mi=order
                t=tic; prefix_s=0; setup_s=0; solve_s=0; iterations=0; flag=0; post=NaN;
                if strcmp(names{mi},'direct')
                    ts=tic; u=A\b; solve_s=toc(ts);
                else
                    u=zeros(size(b));
                    if mi<=numel(bank)
                        ts=tic; u=prefix(A,b,mu,bank{mi},track_tol); prefix_s=toc(ts);
                    end
                    post=norm(b-A*u)/nb;
                    if post>tol
                        ts=tic; Ashift=A+1i*beta*spdiags(mass,0,numel(b),numel(b));
                        M=build_shifted_laplacian_preconditioner(Ashift,ilu_opts); setup_s=toc(ts);
                        ts=tic; [u,info]=shifted_laplacian_gmres(@(v)A*v,M,b,u,opts);
                        solve_s=toc(ts); iterations=info.iter_count; flag=info.flag;
                        clear M Ashift;
                    end
                end
                rho=norm(b-A*u)/nb; elapsed=toc(t);
                err=norm(u-refs(:,qi))/max(norm(refs(:,qi)),eps);
                assert(flag==0 && rho<=tol,'True residual acceptance failed.');
                if rep>0
                    rows=[rows;table(string(names{mi}),budgets(mi),qi,mu,rep,elapsed, ...
                        prefix_s,setup_s,solve_s,iterations,post,rho,err, ...
                        'VariableNames',{'candidate','budget','query','parameter','repeat','total_s', ...
                        'ROM_s','setup_s','solve_s','completion_iterations','post_ROM_residual', ...
                        'true_residual','reference_error'})]; %#ok<AGROW>
                end
                if rep==1, fields{qi,mi}=u; end
            end
        end
        writetable(rows,fullfile(out,'timings.csv'));
        save(fullfile(out,'raw','timings.mat'),'rows','fields','files','opts','ilu_opts','beta','repetitions','-v7.3');
        fprintf('JCP_RANK_TIMING query=%d/%d passed\n',qi,numel(params));
    end
    fprintf('JCP_RANK_PASS measured=%d maxrho=%.4g\n',height(rows),max(rows.true_residual));
end

function u=prefix(A,b,mu,model,track_tol)
    n=numel(b); depth=numel(model.layers); nb=norm(b); u=zeros(n,1);
    Q=zeros(n,depth+1); Z=zeros(n,depth); H=zeros(depth+1,depth); Q(:,1)=b/nb;
    for j=1:depth
        l=model.layers{j}; sp=l.space;
        br=rom_reduced_space_utils('project',sp,Q(:,j),model.shared_indices);
        Ar=l.Ar_base+(mu^(-2)-1)*l.Ar_right;
        Z(:,j)=rom_reduced_space_utils('lift',sp,Ar\br,model.shared_indices);
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*Z(:,j),Q(:,1:j));
        g=zeros(j+1,1); g(1)=nb; u=Z(:,1:j)*(H(1:j+1,1:j)\g);
        if stop || norm(b-A*u)/nb<=track_tol, break; end
    end
end

function write_text(file,value)
    fid=fopen(file,'w'); assert(fid>0); cleanup=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fwrite(fid,value,'char');
end
