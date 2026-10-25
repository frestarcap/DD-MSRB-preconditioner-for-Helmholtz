function rows=run_reviewer_waveholtz_prefix_probe(scale)
%RUN_REVIEWER_WAVEHOLTZ_PREFIX_PROBE Freeze layer 1 and exercise WH completion.
% One diagnostic query; does not replace the three-layer frequency study.
    if nargin<1, scale=4; end
    out=helm_output_path('outputs','reviewer_20260909','high_frequency_waveholtz');
    rows=table();
    for method={'global','dd'}
        tag=sprintf('%s_scale%g',method{1},scale);
        d=load(fullfile(out,[tag '.mat']),'artifact'); a=d.artifact; clear d;
        model=struct(); model.layers=a.layers(1); model.num_layers=1;
        A_base=a.problem.A_base; A_right=a.problem.A_right; rhs=a.rhs;
        cfg=a.cfg; clear a;
        names=fieldnames(cfg);
        for k=1:numel(names)
            key=names{k}; value=cfg.(key);
            if startsWith(key,'corr_') && isnumeric(value) && isvector(value) && numel(value)==3
                cfg.(key)=value(1);
            end
        end
        cfg.corr_num_layers=1; cfg.corr_pod_singular_value_tols=1e-5;
        if ~isempty(cfg.corr_block3_global_tail_tols), cfg.corr_block3_global_tail_tols=1e-5; end
        cfg.test_params=0.8761; cfg.run_wave_baselines=false; cfg.wh_verbose_gmres=true;
        file=fullfile(out,[tag '_prefix_model.mat']);
        save(file,'model','A_base','A_right','rhs','-v7.3');
        cfg.frozen_online_file=file;
        result=helm_waveholtz_correction_rom(cfg);
        assert(all([result.rows.completion_iterations]>0),'Probe should exercise completion.');
        save(fullfile(out,[tag '_prefix_result.mat']),'result');
        row=struct2table(rmfield(result.rows,{'u','native_history','ROM_history'}));
        row.method=repmat(string(method{1}),height(row),1); row.scale=repmat(scale,height(row),1);
        rows=[rows;row]; %#ok<AGROW>
        writetable(rows,fullfile(out,sprintf('prefix_scale%g.csv',scale)));
        fprintf('%s prefix: %d WH iterations, true residual %.3g, %.2fs\n', ...
            tag,row.completion_iterations,row.true_residual,row.total_s);
    end
end
