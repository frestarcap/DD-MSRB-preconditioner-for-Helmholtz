function prepare_ecc_two_layer_paper
% Assemble the accepted two-layer paper cohort without retraining.
    out=helm_output_path('outputs','ecc_two_layer_paper_20260911');
    source=helm_output_path('outputs','helm_trajectory_evidence', ...
        'review_repaired_20260905','ecc_residual_ablation_paper.mat');
    S=load(source,'global_results','dd_results');
    timing=readtable(fullfile(out,'timings.csv'));
    assert(height(timing)==99);
    names={'global','dd'}; storage_rows={};
    for m=1:2
        r=S.([names{m} '_results']);
        fields=fieldnames(r.cfg);
        for f=1:numel(fields)
            key=fields{f}; v=r.cfg.(key);
            if startsWith(key,'corr_') && isnumeric(v) && isvector(v) && numel(v)==3
                r.cfg.(key)=v(1:2);
            end
        end
        r.cfg.corr_num_layers=2;
        r.correction_rom.num_layers=2;
        r.correction_rom.layers=r.correction_rom.layers(1:2);
        r.correction_rom.u_train_after_layer=r.correction_rom.u_train_after_layer(1:2);
        r.test.corr_u_layers=r.test.corr_u_layers(1:2);
        r.test.corr_time_layers=r.test.corr_time_layers(1:2,:);
        r.test.layer_err_helm_full=r.test.layer_err_helm_full(1:2,:);
        r.test.layer_err_helm_phys=r.test.layer_err_helm_phys(1:2,:);
        for k=1:numel(r.test.ratio)
            a=load(fullfile(out,sprintf('query%02d_%s.mat',k,names{m})),'result');
            v=a.result; p=v.precorr; c=v.completion;
            assert(p.iter_count==2 && v.ratio==r.test.ratio(k));
            t=timing(strcmp(timing.method,names{m}) & timing.query==k,:);
            r.test.online_precorr_u(:,k)=v.precorr_u;
            r.test.online_final_u(:,k)=v.final_u;
            r.test.online_precorr_relres_hist{k}=p.relres_hist;
            r.test.online_precorr_relres_final(k)=p.relres_final;
            r.test.online_precorr_matvec(k)=p.matvec_count;
            r.test.online_precorr_time(k)=mean(t.rom_seconds);
            r.test.online_lift_time(k)=mean(t.lift_seconds);
            r.test.online_fp_time(k)=mean(t.completion_seconds);
            r.test.online_total_time(k)=mean(t.total_seconds);
            r.test.online_fp_iter(k)=c.iter_count;
            r.test.online_fp_op_apply(k)=c.op_apply_count;
            r.test.online_fp_relres_hist{k}=c.relres_hist;
            r.test.online_fp_relres_final(k)=norm(r.test.systems{k}.b-r.test.systems{k}.A*v.final_u)/norm(r.test.systems{k}.b);
            r.test.online_fp_step_hist{k}=c.step_hist;
            r.test.online_fp_stop_reason{k}=c.stop_reason;
            r.test.online_combined_relres_hist{k}=[p.relres_hist(:);c.relres_hist(:)];
            r.test.online_combined_transition_idx(k)=3;
            ref=r.test.helm_u_ref(:,k); idx=r.test.systems{k}.active_vec_idx;
            ef=norm(v.final_u-ref)/norm(ref);
            ep=norm(v.final_u(idx)-ref(idx))/norm(ref(idx));
            r.test.online_final_err_helm_full(k)=ef;
            r.test.online_final_err_helm_phys(k)=ep;
            r.test.online_final_err_wave_full(k)=ef;
            r.test.online_final_err_wave_phys(k)=ep;
            a=load(fullfile(out,sprintf('query%02d_baseline.mat',k)),'result');
            b=a.result; bt=timing(strcmp(timing.method,'baseline') & timing.query==k,:);
            r.test.plain_fp_u(:,k)=b.final_u;
            r.test.plain_fp_time(k)=mean(bt.total_seconds);
            r.test.plain_fp_iter(k)=b.completion.iter_count;
            r.test.plain_fp_op_apply(k)=b.completion.op_apply_count;
            r.test.plain_fp_relres_final(k)=b.true_residual;
            r.test.plain_fp_relres_hist{k}=b.completion.relres_hist;
            r.test.plain_fp_step_hist{k}=b.completion.step_hist;
            r.test.plain_fp_stop_reason{k}=b.completion.stop_reason;
        end
        old=r.summary;
        r.summary=struct('representative_test_index',old.representative_test_index, ...
            'representative_ratio',old.representative_ratio,'corr_num_layers',2);
        r.timing=struct('offline',NaN,'online',sum(r.test.online_total_time));
        S.([names{m} '_results'])=r;
        b=load(fullfile(out,[names{m} '_model.mat']),'model'); model=b.model;
        for j=1:numel(model.layers)
            sp=model.layers{j}.basis_space;
            sp=rmfield(sp,intersect(fieldnames(sp),{'meta','block_meta'}));
            if strcmp(sp.type,'block3')
                keep={'idx_near','idx_far','idx_pml','count_near','count_far','count_pml'};
                sp.partition=rmfield(sp.partition,setdiff(fieldnames(sp.partition),keep));
            end
            model.layers{j}.basis_space=sp;
        end
        save(fullfile(out,[names{m} '_lean_model.mat']),'model','-v7.3');
        w=whos('model');
        entries=0;
        for j=1:2
            sp=model.layers{j}.basis_space;
            if strcmp(sp.type,'dense'), entries=entries+numel(sp.basis); map=0;
            else
                entries=entries+numel(sp.Un)+numel(sp.Uf)+numel(sp.Up);
                map=8*(numel(sp.partition.idx_near)+numel(sp.partition.idx_far)+numel(sp.partition.idx_pml));
            end
        end
        ranks=cellfun(@(l)rom_reduced_space_utils('dimension',l.basis_space),model.layers);
        storage_rows(end+1,:)={names{m},16*entries/2^20,map/2^20,(16*entries+map)/2^20,w.bytes/2^20,16*max(ranks)^2/2^20}; %#ok<AGROW>
    end
    global_results=S.global_results; dd_results=S.dd_results;
    base_cfg=global_results.cfg;
    save(fullfile(out,'paper_data.mat'),'global_results','dd_results','base_cfg','source','-v7.3');
    writetable(cell2table(storage_rows,'VariableNames',{'method','basis_MiB','map_MiB','payload_MiB','object_MiB','peak_reduced_MiB'}),fullfile(out,'storage.csv'));
    redraw_paper_output_figures('ecc');
    from=helm_output_path('outputs','figures','ecc_correction_rom');
    to=helm_output_path('LaTeX_DL_468198_240419','figures','ecc_correction_rom');
    files={'ecc_global_spectral_energy.pdf','ecc_global_spectral_energy.png', ...
        'ecc_joint_combined_residual_decay_full_field_ratio_0_9072.png', ...
        'ecc_ratio_0_9072_real_joint_correction_full_field.png'};
    for f=1:numel(files),copyfile(fullfile(from,files{f}),fullfile(to,files{f}));end
    fprintf('ECC_TWO_LAYER_PAPER_PASS\n');
end
