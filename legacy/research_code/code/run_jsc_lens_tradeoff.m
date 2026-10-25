function summary=run_jsc_lens_tradeoff(which)
%RUN_JSC_LENS_TRADEOFF Fully retrained storage/time candidates on the paper grid.
    if nargin<1, which='all'; end
    names={'edge','balanced','tight','simple_tight','middle','simple_middle'};
    tails={[0.012,0.016],[1e-3,1e-3],[1e-4,1e-4],[1e-4,1e-4],[3e-4,3e-4],[3e-4,3e-4]};
    out=helm_output_path('outputs','revision_20260907','lens');
    if ~exist(out,'dir'), mkdir(out); end
    summary=table();
    for k=1:numel(names)
        if ~strcmp(which,'all') && ~strcmp(which,names{k}), continue; end
        spec=helmholtz_3d_40pct_config('lens'); cfg=spec.driver_cfg;
        cfg.snapshot_maxit=20; cfg.require_snapshot_tolerance=true;
        cfg.layer_tail_tolerances=tails{k};
        cfg.output_tag=['jsc_20260907_lens_' names{k}];
        if ismember(k,[4,6])
            cfg.physical_axial_sections=1;
            cfg.profile_partition_thresholds=[0.005,0.08,0.4];
        end
        % Original two development queries are used for selection. New frozen
        % holdouts are evaluated only after selecting the candidate.
        file=helm_output_path('outputs','helmholtz_3d_candidates','large', ...
            ['helmholtz_3d_candidates_' cfg.output_tag '.mat']);
        if isfile(file)
            s=load(file,'results'); r=s.results;
            assert(isequaln(r.cfg.layer_tail_tolerances,cfg.layer_tail_tolerances));
            assert(r.cfg.snapshot_maxit==cfg.snapshot_maxit);
        else
            r=run_helmholtz_3d_candidate_experiments('large',cfg);
        end
        c=r.candidates{1};
        for st={'global','dd'}
            m=c.models.(st{1}); t=[c.tests.(st{1})];
            row=table(string(names{k}),string(st{1}), ...
                string(mat2str(m.layer_ranks)),m.basis_bytes/2^20, ...
                m.persistent_bytes/2^20,m.offline_time, ...
                mean([t.total_time]),max([t.iterations]), ...
                max([t.relative_residual]),max(m.snapshot_relative_residual(:)), ...
                string(file),'VariableNames',{'candidate','method','ranks', ...
                'basis_MiB','payload_MiB','offline_s','online_mean_s', ...
                'completion_max','true_residual_max','snapshot_residual_max','file'});
            summary=[summary;row]; %#ok<AGROW>
        end
        writetable(summary,fullfile(out,['summary_' which '.csv']));
        clear r s c;
    end
    disp(summary);
end
