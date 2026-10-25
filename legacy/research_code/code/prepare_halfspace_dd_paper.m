function paper = prepare_halfspace_dd_paper(mode)
%PREPARE_HALFSPACE_DD_PAPER Assemble the accepted Experiment 1 evidence.
% assemble: reuse matching main/tolerance caches, run missing NR/baselines.
% redraw: read the assembled MAT only; no solves or training.
    if nargin < 1, mode = 'assemble'; end
    out = helm_output_path('outputs', 'halfspace_dd_paper');
    if ~exist(out, 'dir'), mkdir(out); end
    file = fullfile(out, 'paper_data.mat');
    if strcmp(mode, 'redraw')
        s = load(file, 'paper');
        paper = s.paper;
        plot_halfspace_dd_paper(paper);
        return;
    end
    assert(strcmp(mode, 'assemble'), 'Use assemble or redraw.');
    variants = {'global','slabs1-aligned-sym8'};
    root = helm_output_path('outputs', 'halfspace_dd_refinement');
    main_tag = 'review_repaired_20260905';
    run_halfspace_dd_refinement('paper', struct('tag',main_tag, ...
        'variant_names',{variants}, 'solver_cfg', ...
        struct('fp_tol',1e-12,'online_helm_accept_tol',1e-12)));
    g = load(fullfile(root,main_tag,'global.mat'));
    d = load(fullfile(root,main_tag,'slabs1-aligned-sym8.mat'));
    audit_tags = {'policy_crosscheck_20260905','greedy_adoption_tol1e-13', ...
        'greedy_adoption_tol1e-12'};
    audit_tols = [1e-14,1e-13,1e-12];
    audit = table();
    audit_sources = cell(0,1);
    for j = 1:3
        if ~all(cellfun(@(v) isfile(fullfile(root,audit_tags{j},[v '.mat'])),variants))
            run_halfspace_dd_refinement('paper', struct('tag',audit_tags{j}, ...
                'variant_names',{variants}, 'solver_cfg',struct('fp_tol',audit_tols(j))));
        end
        for m = 1:2
            afile = fullfile(root,audit_tags{j},[variants{m} '.mat']);
            a = load(afile,'row','artifact');
            assert(a.artifact.cfg.fp_tol == audit_tols(j));
            assert(isequal(a.artifact.test.params,g.artifact.test.params));
            assert(isequal(a.artifact.cfg.corr_pod_singular_value_tols, ...
                g.artifact.cfg.corr_pod_singular_value_tols));
            row = table(audit_tols(j),m,a.row.mean_completion_iter, ...
                a.row.mean_online_s,nnz(a.artifact.test.online_fp_iter==0), ...
                a.row.max_final_helm_residual, 'VariableNames', ...
                {'wh_tolerance','method','mean_iterations','mean_online_s', ...
                'zero_update_count','max_helm_residual'});
            audit = [audit;row]; %#ok<AGROW>
            audit_sources{end+1,1} = afile; %#ok<AGROW>
        end
    end
    cfg = g.requested_cfg;
    cfg.run_nr_rom_comparison = true;
    cfg.run_wave_baselines = true;
    cfg.plot_test_index = 10;
    aux_file = fullfile(out, 'nr_baseline_greedy_tol1e12_fulltiming.mat');
    if isfile(aux_file)
        s = load(aux_file, 'aux', 'requested_cfg');
        assert(isequaln(s.requested_cfg,cfg), 'Auxiliary configuration changed.');
        aux = s.aux;
    else
        r = helm_waveholtz_correction_rom(cfg);
        aux = struct('cfg',r.cfg, 'global_ranks',r.summary.corr_ranks, ...
            'global_test',compact_test(r.test), ...
            'nr',compact_test(r.nr_comparison.test), ...
            'nr_ranks',r.nr_comparison.summary.corr_ranks);
        requested_cfg = cfg;
        save(aux_file, 'aux', 'requested_cfg', '-v7.3');
        clear r;
    end
    assert(isequal(g.artifact.test.params,d.artifact.test.params,aux.nr.params));
    assert(isequal(g.artifact.summary.corr_ranks(:),aux.global_ranks(:),aux.nr_ranks(:)));
    expected = halfspace_paper_dd_config(d.requested_cfg);
    assert(isequaln(expected,d.requested_cfg), 'DD configuration is not the accepted one.');
    paper = struct('global',compact_artifact(g.artifact), ...
        'dd',compact_artifact(d.artifact), 'aux',aux, ...
        'partition',d.artifact.layers{1}.basis_space.partition, ...
        'main_rows',struct2table([g.row;d.row]), ...
        'tolerance_rows',audit, ...
        'source_files',{{fullfile(root,main_tag,'global.mat'), ...
        fullfile(root,main_tag,'slabs1-aligned-sym8.mat'),aux_file}}, ...
        'audit_source_files',{audit_sources});
    paper.main_rows.helm_accepted_count = [ ...
        nnz(strcmp(g.artifact.test.online_fp_stop_reason,'helmholtz_residual')); ...
        nnz(strcmp(d.artifact.test.online_fp_stop_reason,'helmholtz_residual'))];
    paper.main_rows.saved_basis_percent = 100*(1-paper.main_rows.basis_bytes/g.row.basis_bytes);
    paper.main_rows.saved_rom_payload_percent = 100*(1-paper.main_rows.rom_payload_bytes/g.row.rom_payload_bytes);
    paper.block_table = block_table(d.artifact);
    paper.layer_table = layer_table(g.artifact,d.artifact);
    save(file, 'paper', '-v7.3');
    writetable(paper.main_rows, fullfile(out,'main_results.csv'));
    writetable(paper.tolerance_rows, fullfile(out,'completion_tolerance.csv'));
    writetable(paper.block_table, fullfile(out,'block_ranks.csv'));
    writetable(paper.layer_table, fullfile(out,'layer_storage.csv'));
    disp(paper.main_rows);
    disp(paper.layer_table);
    disp(paper.block_table);
    plot_halfspace_dd_paper(paper);
end

function a = compact_artifact(a)
    a.test = compact_test(a.test);
    for j = 1:numel(a.layers)
        a.layers{j} = struct('diagnostics',a.layers{j}.diagnostics, ...
            'singular_values',a.layers{j}.singular_values);
    end
end

function t = compact_test(t)
    keep = {'params','online_final_u','helm_u_ref','online_precorr_relres_final', ...
        'online_precorr_relres_hist','online_true_helm_relres_hist','plain_true_helm_relres_hist', ...
        'online_fp_relres_final','online_combined_relres_hist','online_fp_iter', ...
        'online_combined_transition_idx','online_fp_relres_hist','online_fp_stop_reason', ...
        'online_total_time','online_precorr_time','online_lift_time','online_fp_time', ...
        'online_final_err_helm_full','plain_fp_relres_hist','plain_fp_iter', ...
        'plain_fp_time','plain_fp_relres_final','plain_fp_stop_reason', ...
        'helm_baseline_relres_hist'};
    t = rmfield(t,setdiff(fieldnames(t),keep));
end

function t = block_table(a)
    b = a.layers{1}.diagnostics;
    t = table((1:numel(b.ranks))',b.labels(:),b.dofs(:), ...
        'VariableNames',{'block','label','dofs'});
    for j = 1:numel(a.layers)
        d = a.layers{j}.diagnostics;
        t.(sprintf('rank_%d',j)) = d.ranks(:);
        t.(sprintf('tail_%d',j)) = d.local_tails(:);
    end
end

function t = layer_table(g,d)
    n = numel(g.layers);
    ranks = zeros(n,1); dd_ranks = ranks; entries = ranks; dd_entries = ranks;
    for j = 1:n
        a = g.layers{j}.diagnostics; b = d.layers{j}.diagnostics;
        ranks(j) = sum(a.ranks); dd_ranks(j) = sum(b.ranks);
        entries(j) = sum(a.ranks.*a.dofs); dd_entries(j) = sum(b.ranks.*b.dofs);
    end
    t = table((1:n)',ranks,dd_ranks,entries,dd_entries,100*(1-dd_entries./entries), ...
        'VariableNames',{'layer','global_rank','dd_rank','global_entries','dd_entries','saved_percent'});
end
