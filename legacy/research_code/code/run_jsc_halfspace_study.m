function summary=run_jsc_halfspace_study(mode)
%RUN_JSC_HALFSPACE_STUDY Fixed solution-POD competitor and training sensitivity.
% Frozen test set and geometry; all methods use shifted-Laplacian completion
% and true Helmholtz acceptance at 1e-12. Timings run separately from training.
    if nargin<1, mode='train'; end
    out=helm_output_path('outputs','revision_20260907','halfspace');
    if ~exist(out,'dir'), mkdir(out); end
    tests=[0.785,0.8053,0.8297,0.8513,0.8761,0.9007,0.9229,0.9431,0.965];
    summary=table(); diagnostics=table();
    for nt=[6,12,24]
        etas=1e-5; if nt==24, etas=[1e-4,1e-5,1e-6]; end
        for eta=etas
            for method={'global','dd'}
                cfg=base_config(nt,eta,tests);
                if strcmp(method{1},'dd'), cfg=halfspace_paper_dd_config(cfg); end
                name=sprintf('%s_n%d_eta%d',method{1},nt,round(log10(eta)));
                file=fullfile(out,[name '.mat']);
                if isfile(file)
                    d=load(file,'r'); r=d.r;
                    assert(isequal(r.cfg.test_params,tests));
                else
                    assert(strcmp(mode,'train'),'Run training before timing.');
                    r=helm_waveholtz_correction_rom(cfg);
                    save(file,'r','-v7.3');
                end
                [row,diag]=inspect(r,name,mode);
                summary=[summary;row]; diagnostics=[diagnostics;diag]; %#ok<AGROW>
                if strcmp(method{1},'global') && eta==1e-5
                    for ptol=[1e-5,1e-12,1e-14]
                        podname=sprintf('fixedpod_n%d_eta%d',nt,round(log10(ptol)));
                        podfile=fullfile(out,[podname '.mat']);
                        if isfile(podfile)
                            d=load(podfile,'r'); p=d.r;
                        else
                            p=make_fixed_pod(r,ptol);
                            p=helm_waveholtz_correction_rom(p.cfg,p);
                            p.timing.offline=p.correction_rom.standalone_offline_s;
                            old=r; r=p; save(podfile,'r','-v7.3'); r=old; clear old;
                        end
                        [row,~]=inspect(p,podname,mode); summary=[summary;row]; %#ok<AGROW>
                    end
                end
                writetable(summary,fullfile(out,['summary_' mode '.csv']));
                writetable(diagnostics,fullfile(out,'online_action_diagnostics.csv'));
            end
        end
    end
    disp(summary);
end

function cfg=base_config(nt,eta,tests)
    cfg=struct('material_model','halfspace','reference_mode','helm_lift', ...
        'rom_check_target','helm','online_completion_mode','shifted_laplacian', ...
        'use_direct_rom_init',false,'build_direct_rom_when_unused',false, ...
        'run_wave_baselines',false,'run_nr_rom_comparison',false, ...
        'do_plots',false,'save_plots',false,'save_benchmark',false, ...
        'parallel_train',false,'parallel_test',false, ...
        'train_params',linspace(0.8,0.95,nt),'test_params',tests, ...
        'corr_num_layers',3,'corr_pod_singular_value_tols',eta*[1,1,1], ...
        'corr_basis_style','global','online_helm_accept_tol',1e-12, ...
        'shifted_laplacian_tol',1e-12,'shifted_laplacian_restart',40, ...
        'shifted_laplacian_maxit',20,'shifted_laplacian_beta',0.2);
end

function p=make_fixed_pod(r,tol)
    timer=tic; S=r.train_reference.u_ref_snapshots;
    [U,D,~]=svd(S,'econ'); sig=diag(D);
    tails=sqrt([flipud(cumsum(flipud(sig(2:end).^2)));0])/norm(sig);
    k=find(tails<=tol,1); U=U(:,1:k);
    sp=rom_reduced_space_utils('make_dense_space',U);
    l=r.correction_rom.layers{1}; l.basis_space=sp; l.basis_u=U;
    l.space_dim=k; l.pod.basis=U; l.pod.rank=k; l.pod.rank_used=k;
    l.Ar_base=U'*r.problem.A_base*U; l.Ar_right=U'*r.problem.A_right*U;
    p=r; p.correction_rom.layers={l}; p.correction_rom.num_layers=1;
    p.correction_rom.fixed_pod_initial=true;
    p.correction_rom.standalone_offline_s=sum(r.train_reference.solve_time)+toc(timer);
    p.cfg.corr_num_layers=1; p.cfg.corr_pod_singular_value_tols=tol;
    p.cfg.corr_block3_global_tail_tols=[];
    fields=fieldnames(p.cfg);
    for q=1:numel(fields)
        key=fields{q}; value=p.cfg.(key);
        if startsWith(key,'corr_block3_') && isnumeric(value) && isvector(value) && numel(value)==3
            p.cfg.(key)=value(1);
        end
    end
end

function [row,diags]=inspect(r,name,mode)
    t=r.test; s=rom_deployment_storage(r.correction_rom.layers);
    actual=actual_residual(r); assert(all(actual<=1e-12),'Final true residual target missed.');
    times=t.online_total_time(:).'; dev=nan(size(times)); repetitions=1;
    if strcmp(mode,'timing')
        repetitions=5; samples=zeros(5,numel(times));
        cfg=r.cfg; cfg.do_reference_check=false;
        warm=helm_waveholtz_correction_rom(cfg,r); %#ok<NASGU>
        for rep=1:5
            repeated=helm_waveholtz_correction_rom(cfg,r);
            assert(all(actual_residual(repeated)<=1e-12));
            samples(rep,:)=repeated.test.online_total_time;
        end
        times=mean(samples,1); dev=std(samples,0,1);
        out=helm_output_path('outputs','revision_20260907','halfspace');
        save(fullfile(out,[name '_timings.mat']),'samples','cfg');
    end
    row=table(string(name),numel(r.cfg.train_params),string(mat2str(s.ranks)), ...
        s.basis_MiB,s.payload_MiB,r.timing.offline,mean(times),max(times), ...
        max(dev),repetitions,max(t.online_fp_iter),max(actual), ...
        max(t.online_precorr_relres_final), ...
        'VariableNames',{'candidate','training_count','ranks','basis_MiB', ...
        'payload_MiB','offline_s','online_mean_s','online_worst_s', ...
        'max_query_timing_sd','repetitions','completion_max','true_residual_max', ...
        'precompletion_residual_max'});
    diags=table();
    if isfield(r.correction_rom,'fixed_pod_initial'), return; end
    for k=1:numel(t.params)
        mu=t.params(k); alpha=1/mu^2-1; A=r.problem.A_base+alpha*r.problem.A_right;
        ls=r.correction_rom.layers;
        spaces=cellfun(@(l)l.basis_space,ls,'UniformOutput',false);
        reduced=cellfun(@(l)l.Ar_base+alpha*l.Ar_right,ls,'UniformOutput',false);
        [~,d]=rom_true_action_diagnostics(A,r.rhs,spaces,reduced);
        d.candidate=repmat(string(name),height(d),1); d.parameter=repmat(mu,height(d),1);
        diags=[diags;d]; %#ok<AGROW>
    end
end

function rho=actual_residual(r)
    rho=zeros(numel(r.test.params),1);
    for k=1:numel(rho)
        A=r.problem.A_base+(1/r.test.params(k)^2-1)*r.problem.A_right;
        rho(k)=norm(r.rhs-A*r.test.online_final_u(:,k))/norm(r.rhs);
    end
end
