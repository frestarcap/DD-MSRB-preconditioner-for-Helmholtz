function benchmark_ecc_two_layer_paper
% Matched repeated online timings for the two-layer Experiment 2 models.
% Four independent query workers, each capped at one computational thread.
    out=helm_output_path('outputs','ecc_two_layer_paper_20260911');
    if ~exist(out,'dir'), mkdir(out); end
    source=helm_output_path('outputs','helm_trajectory_evidence', ...
        'review_repaired_20260905','ecc_residual_ablation_paper.mat');
    s=load(source,'global_results','dd_results');
    names={'baseline','global','dd'}; configs=cell(1,3);
    for m=1:3
        if m==3, r=s.dd_results; else, r=s.global_results; end
        if m==1, depth=0; else, depth=2; end
        model=struct('layers',{r.correction_rom.layers(1:depth)},'num_layers',depth);
        for j=1:depth, model.layers{j}=struct('basis_space',model.layers{j}.basis_space); end
        file=fullfile(out,[names{m} '_model.mat']); save(file,'model','-v7.3');
        cfg=r.cfg; cfg.parallel_test=false; cfg.parallel_train=false;
        cfg.do_plots=false; cfg.save_plots=false; cfg.save_benchmark=false;
        cfg.deployment_file=file; cfg.deployment_audit=true;
        configs{m}=cfg;
    end
    ratios=s.global_results.test.ratio;
    expected={[],s.global_results.test.corr_u_layers{2},s.dd_results.test.corr_u_layers{2}};
    clear s r model;
    pool=gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers~=4, delete(pool); pool=[]; end
    if isempty(pool), pool=parpool('Processes',4); end
    future=parfevalOnAll(pool,@maxNumCompThreads,0,1); wait(future);
    parfor w=1:4
        for m=1:3
            cfg=configs{m}; cfg.ratio_test=ratios(1);
            waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
        end
    end
    rows={};
    for rep=1:3
        blocks=cell(numel(ratios),1);
        parfor k=1:numel(ratios)
            local={}; order=circshift(1:3,mod(k+rep-2,3));
            for a=1:3
                m=order(a); cfg=configs{m}; cfg.ratio_test=ratios(k);
                result=waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
                assert(result.ratio==ratios(k));
                assert(result.completion.step_hist(end)<cfg.wh_tol);
                if m>1
                    e=norm(result.precorr_u-expected{m}(:,k))/norm(expected{m}(:,k));
                    assert(e<1e-10);
                    rom=result.precorr.total_time;
                else
                    e=0; rom=0;
                end
                c=result.completion;
                local(end+1,:)={names{m},k,rep,ratios(k),rom,c.lift_time,c.wave_time, ...
                    rom+c.total_time,c.iter_count,result.true_residual,e,c.step_hist(end)}; %#ok<AGROW>
                if rep==1
                    save_case(fullfile(out,sprintf('query%02d_%s.mat',k,names{m})),result,cfg);
                end
                fprintf('ECC_TIMING rep=%d query=%d %s total=%.6f it=%d\n',rep,k,names{m},rom+c.total_time,c.iter_count);
            end
            blocks{k}=local;
        end
        rows=[rows;vertcat(blocks{:})]; %#ok<AGROW>
        T=cell2table(rows,'VariableNames',{'method','query','repeat','ratio','rom_seconds', ...
            'lift_seconds','completion_seconds','total_seconds','completion_iters', ...
            'true_residual','prefix_error','final_update'});
        writetable(T,fullfile(out,'timings.csv'));
    end
    assert(height(T)==99);
    for m=1:3
        for k=1:numel(ratios)
            t=T(strcmp(T.method,names{m}) & T.query==k,:);
            assert(height(t)==3 && numel(unique(t.completion_iters))==1);
        end
    end
    save(fullfile(out,'timings.mat'),'T','source','configs','ratios');
    delete(pool);
    fprintf('ECC_TWO_LAYER_TIMING_PASS\n');
end

function save_case(file,result,cfg)
    save(file,'result','cfg','-v7.3');
end
