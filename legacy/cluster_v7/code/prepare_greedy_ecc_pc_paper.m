function prepare_greedy_ecc_pc_paper(tag)
%PREPARE_GREEDY_ECC_PC_PAPER Audit tagged ECC and promote saved greedy PC data.
    if nargin<1, tag='review_repaired_20260905'; end
    out = helm_output_path('outputs','helm_trajectory_evidence',tag);
    for style = {'global','dd'}
        s = load(fullfile(out,['ecc_paper_' style{1} '_checkpoint.mat']),'r');
        ecc.(style{1}) = s.r;
    end
    assert(isequal(ecc.global.test.ratio,ecc.dd.test.ratio));
    assert(numel(ecc.dd.test.ratio)==11 && ecc.dd.cfg.wh_tol==1e-8);
    assert(strcmp(ecc.dd.cfg.corr_block3_global_tail_strategy,'marginal_storage'));
    for style = {'global','dd'}
        r = ecc.(style{1});
        assert(all(strcmp(r.test.online_fp_stop_reason,'fixed_point_step')));
        assert(max(r.test.online_fp_relres_final)<1e-8);
        assert(max(r.test.online_final_err_helm_full)<1e-6);
        for k = 1:numel(r.test.ratio)
            sys = r.test.systems{k};
            rho = norm(sys.A*r.test.online_final_u(:,k)-sys.b)/norm(sys.b);
            assert(abs(rho-r.test.online_fp_relres_final(k))<1e-12);
        end
    end
    if strcmp(tag,'review_repaired_20260905')
        baseline_repetitions=[ecc.global.test.plain_fp_time,ecc.dd.test.plain_fp_time];
        shared_baseline=mean(baseline_repetitions,2);
        ecc.global.test.plain_fp_time=shared_baseline;
        ecc.dd.test.plain_fp_time=shared_baseline;
        save(fullfile(out,'ecc_shared_baseline_timings.mat'), ...
            'baseline_repetitions','shared_baseline');
    end
    summarize_pair(ecc.global,ecc.dd,'ecc',out);

    if strcmp(tag,'review_repaired_20260905')
        s=load(fullfile(out,'pc_residual_ablation_paper.mat'),'global_results','dd_results');
        assert(strcmp(s.global_results.cfg.reference_mode,'helm_lift'));
        summarize_pair(s.global_results,s.dd_results,'pc',out);
        return;
    end

    pc_file = fullfile(out,'pc_waveholtz_residual_ablation_paper.mat');
    if isfile(pc_file)
        s = load(pc_file,'global_results','dd_results');
        assert(strcmp(s.dd_results.cfg.corr_block3_global_tail_strategy,'marginal_storage'));
        assert(isequal(s.global_results.test.params,s.dd_results.test.params));
        summarize_pair(s.global_results,s.dd_results,'pc',out);
        return;
    end

    old = helm_output_path('outputs','helm_trajectory_evidence', ...
        'pc_waveholtz_residual_ablation_paper.mat');
    s = load(old,'global_results','dd_results');
    global_results = s.global_results;
    old_dd = s.dd_results;
    source = helm_output_path('outputs','pc_dd_storage_exploration', ...
        'pc_dd_storage_paper_waveholtz_narrow4_greedy_full.mat');
    s = load(source,'run_result');
    dd_results = s.run_result;
    assert(strcmp(dd_results.cfg.corr_block3_global_tail_strategy,'marginal_storage'));
    assert(isequal(global_results.test.params,dd_results.test.params));
    assert(isequal(global_results.train_reference.params,dd_results.train_reference.params));
    assert(isequaln(global_results.problem.A_base,dd_results.problem.A_base));
    assert(isequaln(global_results.problem.A_right,dd_results.problem.A_right));
    assert(isequal(global_results.rhs,dd_results.rhs));
    for field = {'corr_num_layers','corr_pod_singular_value_tols','wh_tol', ...
            'wh_gmres_restart','wh_gmres_maxit','reference_mode','online_completion_mode'}
        assert(isequaln(global_results.cfg.(field{1}),dd_results.cfg.(field{1})), ...
            'PC configuration differs: %s',field{1});
    end
    for j = 1:3
        sp = dd_results.correction_rom.layers{j}.basis_space;
        old_sp = old_dd.correction_rom.layers{j}.basis_space;
        assert(isequal([sp.blocks.k],[old_sp.blocks.k]), 'PC local ranks changed.');
        assert(all([sp.blocks.count]==[old_sp.blocks.count]));
        assert(sp.blocks(1).meta.global_relative_tail <= ...
            dd_results.cfg.corr_pod_singular_value_tols(j));
    end
    assert(max(dd_results.test.online_fp_relres_final)<1.05*dd_results.cfg.wh_tol);
    base_cfg = dd_results.cfg;
    provenance = struct('global_source',old,'dd_source',source, ...
        'note','Actual saved greedy retraining; not relabeled DOF data.');
    save(pc_file, ...
        'global_results','dd_results','base_cfg','provenance','-v7.3');
    summarize_pair(global_results,dd_results,'pc',out);
end

function summarize_pair(g,d,name,out)
    layers = table(); rows = table();
    for m = 1:2
        if m==1, r=g; label="Global TA-ROM"; else, r=d; label="DD-TA-ROM"; end
        basis = 0; operators = 0; peak = 0; index_bytes = 0;
        for j = 1:numel(r.correction_rom.layers)
            l = r.correction_rom.layers{j}; sp=l.basis_space;
            if strcmp(sp.type,'dense')
                entries = numel(sp.basis); ranks = sp.k; tail=NaN;
            elseif strcmp(sp.type,'block3')
                entries = numel(sp.Un)+numel(sp.Uf)+numel(sp.Up);
                ranks=[sp.kn sp.kf sp.kp]; tail=sp.block_meta.near.global_relative_tail;
                index_bytes=8*(size(sp.Un,1)+size(sp.Uf,1)+size(sp.Up,1));
            else
                entries = sum(arrayfun(@(b) numel(b.basis),sp.blocks));
                ranks=[sp.blocks.k]; tail=sp.blocks(1).meta.global_relative_tail;
                index_bytes=8*sum([sp.blocks.count]);
            end
            if m==2, assert(tail<=r.cfg.corr_pod_singular_value_tols(j)); end
            basis=basis+16*entries;
            if isfield(l,'Ar_base')
                operators=operators+16*(numel(l.Ar_base)+numel(l.Ar_right));
            else
                peak=max(peak,16*sum(ranks)^2);
            end
            layers=[layers;table(label,j,string(mat2str(ranks)),entries,tail, ...
                'VariableNames',{'method','layer','ranks','entries','tail'})]; %#ok<AGROW>
        end
        t=r.test;
        row=table(label,basis/2^20,operators/2^20,index_bytes/2^20,peak/2^20, ...
            (basis+operators+index_bytes+peak)/2^20, ...
            mean(t.online_precorr_relres_final),mean(t.online_fp_iter), ...
            mean(t.online_precorr_time),mean(t.online_lift_time),mean(t.online_fp_time), ...
            mean(t.online_total_time),mean(t.plain_fp_time), ...
            mean(t.online_fp_relres_final),max(t.online_fp_relres_final), ...
            mean(t.online_final_err_helm_full),max(t.online_final_err_helm_full), ...
            'VariableNames',{'method','basis_MiB','operators_MiB','index_MiB','peak_reduced_MiB', ...
            'rom_MiB','mean_precorr_residual','mean_completion_iterations', ...
            'mean_precorr_s','mean_lift_s','mean_completion_s','mean_online_s', ...
            'mean_plain_s','mean_final_residual','max_final_residual', ...
            'mean_final_error','max_final_error'});
        rows=[rows;row]; %#ok<AGROW>
    end
    rows.basis_saved_percent=100*(1-rows.basis_MiB/rows.basis_MiB(1));
    rows.rom_saved_percent=100*(1-rows.rom_MiB/rows.rom_MiB(1));
    writetable(rows,fullfile(out,[name '_greedy_summary.csv']));
    writetable(layers,fullfile(out,[name '_greedy_layers.csv']));
    disp(rows); disp(layers);
end
