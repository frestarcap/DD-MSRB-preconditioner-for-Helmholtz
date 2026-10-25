function run_ecc_layer3_audit(mode)
% Frozen two/three-layer ablation for the current Experiment 2 cohort.
% 'residual' independently checks every saved iterate; 'completion' also
% replays both prefixes through the original ECC kernel and WaveHoltz stop.
% No training or manuscript evidence is overwritten. Timings are exploratory
% single passes, rotated by query, not repeated publication benchmarks.
    if nargin<1, mode='residual'; end
    assert(ismember(mode,{'residual','completion'}));
    out=helm_output_path('outputs','ecc_layer3_audit_20260911');
    if ~exist(out,'dir'), mkdir(out); end
    source=helm_output_path('outputs','helm_trajectory_evidence', ...
        'review_repaired_20260905','ecc_residual_ablation_paper.mat');
    s=load(source,'global_results','dd_results');
    names={'global','dd'}; rows={}; trainrows={}; storage={};
    for m=1:2
        r=s.([names{m} '_results']);
        assert(~r.cfg.use_direct_rom_init && r.correction_rom.num_layers==3);
        for k=1:numel(r.test.ratio)
            sys=r.test.systems{k}; rho=zeros(1,3); err=rho;
            for j=1:3
                u=r.test.corr_u_layers{j}(:,k);
                rho(j)=norm(sys.b-sys.A*u)/norm(sys.b);
                err(j)=norm(u-r.test.helm_u_ref(:,k))/norm(r.test.helm_u_ref(:,k));
            end
            saved=r.test.online_precorr_relres_hist{k};
            assert(max(abs(rho-saved(2:4).'))<1e-12);
            assert(all(diff(rho)<=1e-12));
            rows(end+1,:)={names{m},k,r.test.ratio(k),rho(1),rho(2),rho(3), ...
                100*(1-rho(3)/rho(2)),err(2),err(3), ...
                r.test.corr_time_layers(3,k)}; %#ok<AGROW>
        end
        for k=1:numel(r.train_reference.ratio)
            rho=zeros(1,3);
            sys=r.train_reference.systems{k};
            for j=1:3
                u=r.correction_rom.u_train_after_layer{j}(:,k);
                rho(j)=norm(sys.b-sys.A*u)/norm(sys.b);
                assert(abs(rho(j)-r.correction_rom.layers{j}.train_fgmres_relres(k))<1e-12);
            end
            trainrows(end+1,:)={names{m},k,r.train_reference.ratio(k),rho(1),rho(2),rho(3),100*(1-rho(3)/rho(2))}; %#ok<AGROW>
        end
        for j=1:3
            sp=r.correction_rom.layers{j}.basis_space;
            if strcmp(sp.type,'dense')
                entries=numel(sp.basis); map=0;
            else
                entries=numel(sp.Un)+numel(sp.Uf)+numel(sp.Up);
                map=8*(numel(sp.partition.idx_near)+numel(sp.partition.idx_far)+numel(sp.partition.idx_pml));
            end
            storage(end+1,:)={names{m},j,r.correction_rom.layers{j}.space_dim,entries,16*entries/2^20,map/2^20}; %#ok<AGROW>
        end
        for depth=2:3
            model=struct('layers',{r.correction_rom.layers(1:depth)},'num_layers',depth);
            % Retain only deployed bases: omit offline snapshot/POD duplicates.
            for j=1:depth, model.layers{j}=struct('basis_space',model.layers{j}.basis_space); end
            save(fullfile(out,sprintf('%s_L%d.mat',names{m},depth)),'model','-v7.3');
        end
    end
    T=cell2table(rows,'VariableNames',{'method','query','ratio','rho1','rho2','rho3','decrease_pct','error2','error3','saved_layer3_seconds'});
    writetable(T,fullfile(out,'residuals.csv'));
    writetable(cell2table(trainrows,'VariableNames',{'method','query','ratio','rho1','rho2','rho3','decrease_pct'}),fullfile(out,'training_residuals.csv'));
    writetable(cell2table(storage,'VariableNames',{'method','layer','rank','basis_entries','basis_MiB','shared_map_MiB'}),fullfile(out,'storage.csv'));
    disp(T);
    if strcmp(mode,'residual'), return; end
    rows={};
    for k=1:numel(s.global_results.test.ratio)
        pairs=[1 2;1 3;2 2;2 3];
        pairs=circshift(pairs,mod(k-1,4),1);
        for a=1:4
            m=pairs(a,1); depth=pairs(a,2); name=names{m};
            r=s.([name '_results']); cfg=r.cfg;
            cfg.ratio_test=r.test.ratio(k);
            cfg.parallel_test=false; cfg.parallel_train=false;
            cfg.do_plots=false; cfg.save_plots=false; cfg.save_benchmark=false;
            cfg.deployment_file=fullfile(out,sprintf('%s_L%d.mat',name,depth));
            cfg.deployment_audit=true;
            fprintf('START query=%d method=%s depth=%d ratio=%.15g\n',k,name,depth,cfg.ratio_test);
            result=waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
            expected=r.test.corr_u_layers{depth}(:,k);
            replay_error=norm(result.precorr_u-expected)/max(norm(expected),1e-16);
            assert(replay_error<1e-10,'Frozen prefix differs from the original trajectory.');
            assert(result.ratio==r.test.ratio(k));
            sys=r.test.systems{k};
            true_residual=norm(sys.b-sys.A*result.final_u)/norm(sys.b);
            assert(abs(true_residual-result.true_residual)<1e-12);
            assert(result.completion.step_hist(end)<cfg.wh_tol);
            if depth==3
                assert(result.completion_iterations==r.test.online_fp_iter(k),'Three-layer control changed completion count.');
            end
            final_error=norm(result.final_u-r.test.helm_u_ref(:,k))/norm(r.test.helm_u_ref(:,k));
            c=result.completion;
            rows(end+1,:)={name,k,depth,result.ratio,result.precorr.relres_final, ...
                result.completion_iterations,true_residual,final_error, ...
                result.precorr.total_time,c.lift_time,c.wave_time, ...
                result.precorr.total_time+c.total_time,replay_error,c.step_hist(end)}; %#ok<AGROW>
            save(fullfile(out,sprintf('query%02d_%s_L%d.mat',k,name,depth)),'result','cfg','source','replay_error','-v7.3');
            C=cell2table(rows,'VariableNames',{'method','query','layers','ratio','pre_residual','completion_iters','final_residual','final_error','rom_seconds','lift_seconds','wave_seconds','total_seconds','prefix_replay_error','final_update'});
            writetable(C,fullfile(out,'completion.csv'));
            fprintf('DONE query=%d %s L%d: iterations=%d rho=%.4e total=%.4fs\n',k,name,depth,result.completion_iterations,true_residual,C.total_seconds(end));
        end
    end
end
