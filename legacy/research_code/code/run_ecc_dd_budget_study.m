function run_ecc_dd_budget_study(mode)
% Strict paper allocation: a single layerwise global tail budget, zero-start
% marginal-storage greedy, no independent local tail constraints or rank caps.
    if nargin<1,mode='train';end
    old=maxNumCompThreads(1); cleanup=onCleanup(@()maxNumCompThreads(old)); %#ok<NASGU>
    out=helm_output_path('outputs','ecc_dd_budget_20260919');
    if ~isfolder(out),mkdir(out);end
    legacy=helm_output_path('outputs','ecc_shifted_tuning_20260919');
    specs={ 'uniform6',[1e-6,1e-6,1e-6]; ...
            'last4',[1e-6,1e-6,1e-4]; ...
            'last3',[1e-6,1e-6,1e-3]; ...
            'layered',[1e-5,1e-5,1e-3]};
    summary_prefix='';
    if strcmp(mode,'uniform3_train')
        specs={'uniform3',[1e-3,1e-3,1e-3]};summary_prefix='uniform3_';
    end
    if strcmp(mode,'two6_train')
        specs={'two6',[1e-6,1e-6]};summary_prefix='two6_';
    end
    if strcmp(mode,'uniform5_train')
        specs={'uniform5',[1e-5,1e-5,1e-5]};summary_prefix='uniform5_';
    end
    if strcmp(mode,'audit')
        audit_legacy(legacy,out);return;
    elseif ismember(mode,{'train','uniform3_train','two6_train','uniform5_train'})
        rows=table(); audits=table();
        for si=1:size(specs,1)
            for method={'global','dd'}
                tag=[specs{si,1} '_' method{1}];file=fullfile(out,[tag '.mat']);
                if isfile(file)
                    s=load(file,'row','audit');row=s.row;audit=s.audit;
                else
                    s=load(fullfile(legacy,'global_n96_d3_eta1e-06.mat'),'cfg');cfg=s.cfg;
                    cfg.corr_rank_rtol=[];cfg.corr_rank_rtols=specs{si,2};
                    cfg.corr_block3_global_tail_tols=specs{si,2};
                    cfg.corr_block3_global_tail_strategy='marginal_storage';
                    for key={'near','far','pml'}
                        cfg.(['corr_block3_tail_tol_' key{1}])=[];
                        cfg.(['corr_block3_rtol_' key{1}])=[];
                        cfg.(['corr_block3_rank_cap_' key{1}])=Inf;
                    end
                    cfg.corr_num_layers=numel(specs{si,2});cfg.corr_normalize_snapshots=false;
                    cfg.corr_pod_truncation_rule='relative_frobenius_tail';
                    cfg.corr_basis_style='global';
                    if strcmp(method{1},'dd'),cfg.corr_basis_style='block3';end
                    cfg.helm_gmres_track_tol=1e-12;cfg.shifted_laplacian_tol=1e-12;
                    cfg.parallel_train=false;cfg.parallel_test=false;
                    cfg.do_plots=false;cfg.save_plots=false;cfg.save_benchmark=false;
                    fprintf('ECC_BUDGET_TRAIN %s\n',tag);
                    r=waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
                    cfg=r.cfg;depth=cfg.corr_num_layers;model=struct('layers',{cell(depth,1)});snapshots=cell(depth,1);
                    audit=table();training=cell(depth,1);entries=0;map=0;ranks=[];
                    for j=1:depth
                        l=r.correction_rom.layers{j};sp=l.basis_space;E=l.eta_snapshots;
                        model.layers{j}=struct('space',sp);snapshots{j}=E;
                        a=audit_layer(tag,j,E,sp,cfg.corr_rank_rtols(j));audit=[audit;a]; %#ok<AGROW>
                        entries=entries+a.basis_entries;ranks(end+1)=a.dimension; %#ok<AGROW>
                        if strcmp(sp.type,'block3'),map=8*(numel(sp.partition.idx_near)+numel(sp.partition.idx_far)+numel(sp.partition.idx_pml));end
                        training{j}=struct('residual',l.train_fgmres_relres, ...
                            's',l.pod.singular_values,'diagnostics',{l.trajectory_diagnostics});
                        for di=1:numel(l.trajectory_diagnostics)
                            d=l.trajectory_diagnostics{di};
                            if ~isempty(d) && isfield(d,'accepted_residual') && ~d.breakdown
                                assert(d.accepted_residual<=1e-10);
                            end
                        end
                    end
                    assert(all(r.test.online_fp_relres_final<=1e-12));
                    row=table(string(tag),string(method{1}),string(mat2str(cfg.corr_rank_rtols)), ...
                        entries*16/2^20,(entries*16+map)/2^20,string(mat2str(ranks)), ...
                        mean(r.test.online_precorr_relres_final),mean(r.test.online_fp_iter), ...
                        mean(r.test.online_total_time),r.timing.offline,max(r.test.online_fp_relres_final), ...
                        max(r.test.online_final_err_helm_full),'VariableNames', ...
                        {'tag','method','tails','basis_MiB','payload_MiB','ranks','mean_post', ...
                        'iterations','exploratory_total_s','offline_s','max_residual','max_error'});
                    test=r.test;test=rmfield(test,intersect(fieldnames(test),{'systems'}));
                    save(file,'cfg','model','snapshots','training','row','audit','test','-v7.3');
                    clear r model snapshots training test;
                end
                rows=[rows;row];audits=[audits;audit]; %#ok<AGROW>
                writetable(rows,fullfile(out,[summary_prefix 'training_summary.csv']));
                writetable(audits,fullfile(out,[summary_prefix 'tail_audit.csv']));disp(row);
            end
        end
    elseif ismember(mode,{'screen','confirm','validate'})
        benchmark_ecc_dd_budget(mode);
    else,error('Unknown mode %s',mode);end
    fprintf('ECC_DD_BUDGET_%s_PASS\n',upper(mode));
end

function audit_legacy(legacy,out)
    rows=table();
    for method={'global','dd'}
        s=load(fullfile(legacy,[method{1} '_n96_d3_eta1e-06.mat']),'cfg','model','training');
        for j=1:3
            sp=s.model.layers{j}.space;eta=s.cfg.corr_rank_rtols(j);
            if strcmp(sp.type,'dense')
                sv=s.training{j}.s;tail=norm(sv(sp.k+1:end))/norm(sv);actual=sp.k;expected=sp.k;local=false;
            else
                sv={sp.block_meta.near.singular_values,sp.block_meta.far.singular_values,sp.block_meta.pml.singular_values};
                n=[size(sp.Un,1),size(sp.Uf,1),size(sp.Up,1)];actual=[sp.kn,sp.kf,sp.kp];
                expected=independent_greedy(sv,n,eta);
                total=sum(cellfun(@(v)sum(abs(v).^2),sv));tail2=0;
                for b=1:3,tail2=tail2+sum(abs(sv{b}(actual(b)+1:end)).^2);end
                tail=sqrt(tail2/total);local=~isempty(s.cfg.corr_block3_tail_tol_near);
            end
            assert(tail<=eta*(1+1e-7));
            rows=[rows;table(string(method{1}),j,eta,tail,local,string(mat2str(actual)), ...
                string(mat2str(expected)),isequal(actual,expected),'VariableNames', ...
                {'method','layer','budget','achieved_tail','extra_local_constraints', ...
                'saved_ranks','zero_start_greedy_ranks','matches_paper_allocation'})]; %#ok<AGROW>
        end
    end
    disp(rows);writetable(rows,fullfile(out,'legacy_allocation_audit.csv'));
end

function a=audit_layer(tag,j,E,sp,eta)
    if strcmp(sp.type,'dense')
        B=sp.basis;error=norm(E-B*(B'*E),'fro')/norm(E,'fro');
        sv=svd(E,'econ');expected=find(arrayfun(@(k)norm(sv(k+1:end))/norm(sv),1:numel(sv))<=eta,1);
        if isempty(expected),expected=numel(sv);end
        assert(sp.k==expected);spectral=norm(sv(sp.k+1:end))/norm(sv);
        entries=numel(B);ranks=sp.k;
    else
        idx={sp.partition.idx_near,sp.partition.idx_far,sp.partition.idx_pml};
        B={sp.Un,sp.Uf,sp.Up};n=cellfun(@numel,idx);ranks=cellfun(@(v)size(v,2),B);
        sv={sp.block_meta.near.singular_values,sp.block_meta.far.singular_values,sp.block_meta.pml.singular_values};
        expected=independent_greedy(sv,n,eta);assert(isequal(ranks,expected));
        mask=true(size(E,1),1);energy=0;spectral2=0;
        for b=1:3
            mask(idx{b})=false;Eb=E(idx{b},:);
            energy=energy+norm(Eb-B{b}*(B{b}'*Eb),'fro')^2;
            spectral2=spectral2+sum(abs(sv{b}(ranks(b)+1:end)).^2);
        end
        energy=energy+norm(E(mask,:),'fro')^2;
        error=sqrt(energy)/norm(E,'fro');spectral=sqrt(spectral2)/norm(E,'fro');entries=sum(n.*ranks);
    end
    assert(error<=eta*(1+1e-7) && spectral<=eta*(1+1e-7),'Actual snapshot tail exceeds budget.');
    a=table(string(tag),j,eta,spectral,error,string(mat2str(ranks)),sum(ranks),entries, ...
        'VariableNames',{'tag','layer','budget','spectral_tail','projection_tail','ranks','dimension','basis_entries'});
end

function ranks=independent_greedy(sv,n,eta)
% Independent zero-start implementation, using the remaining singular values.
    ranks=zeros(1,numel(sv));energy=sum(cellfun(@(v)sum(abs(v).^2),sv));
    while true
        tail=0;score=-inf(size(ranks));
        for b=1:numel(sv)
            tail=tail+sum(abs(sv{b}(ranks(b)+1:end)).^2);
            if ranks(b)<numel(sv{b}),score(b)=abs(sv{b}(ranks(b)+1))^2/max(n(b),1);end
        end
        if tail<=eta^2*energy*(1+1e-12),break;end
        [val,b]=max(score);assert(isfinite(val));ranks(b)=ranks(b)+1;
    end
end
