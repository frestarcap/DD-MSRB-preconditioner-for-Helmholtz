function summary = run_waveguide_threshold_tradeoff(phase, train_indices, out_override)
%RUN_WAVEGUIDE_THRESHOLD_TRADEOFF Fixed-geometry Experiment 3 threshold sweep.
% Run with -singleCompThread. All five points are retained. Training uses
% four process workers. Primary timing is orchestrated by
% scripts/run_waveguide_isolated_timings.py, one fresh process per model.
    if nargin < 1, phase = 'train'; end
    if nargin < 2, train_indices = [3,1,2,4,5]; end
    assert(all(ismember(train_indices,1:5)));
    assert(ismember(phase, {'train','plot'}), ...
        'Use train/plot here and scripts/run_waveguide_isolated_timings.py for timing.');
    out = helm_output_path('outputs','waveguide_threshold_tradeoff_20260908');
    if strcmp(phase,'plot'), out=fullfile(out,'isolated_timing'); end
    if nargin>=3 && ~isempty(out_override)
        assert(strcmp(phase,'plot'),'Output override is for collection/plotting only.');
        out=char(out_override);
    end
    if ~exist(out,'dir'), mkdir(out); end
    scales = [0.25,0.5,1,2,4];
    labels = {'quarter','half','reference','double','quadruple'};
    spec = helmholtz_3d_40pct_config('waveguide');
    base = spec.driver_cfg;
    base.snapshot_maxit = 20;
    base.require_snapshot_tolerance = true;
    repetitions = 3;
    if strcmp(phase,'train')
        % Fix the full protocol before collecting results.
        if ~isfile(fullfile(out,'protocol.mat'))
            save(fullfile(out,'protocol.mat'),'scales','labels','base','repetitions');
        end
        for k = train_indices
            cfg = base;
            cfg.layer_tail_tolerances = scales(k)*base.layer_tail_tolerances;
            cfg.output_tag = ['threshold_20260908_' labels{k}];
            file = result_file(cfg.output_tag);
            fprintf('\nSWEEP TRAIN %s alpha=%g eta=%s\n', labels{k},scales(k),mat2str(cfg.layer_tail_tolerances));
            if isfile(file)
                saved = load(file,'results'); result = saved.results;
                assert(isequaln(result.cfg.layer_tail_tolerances,cfg.layer_tail_tolerances));
                assert(result.cfg.require_snapshot_tolerance && result.cfg.snapshot_maxit==20);
                assert(isequaln(result.cfg.gaussian_waveguide_cfg,base.gaussian_waveguide_cfg));
            else
                result = run_helmholtz_3d_candidate_experiments('large',cfg);
            end
            c = result.candidates{1};
            for key = {'global','dd'}
                m = c.models.(key{1}); t = [c.tests.(key{1})];
                assert(all(m.snapshot_relative_residual(:)<=base.snapshot_tol*(1+1e-6)));
                assert(all([t.relative_residual]<=base.gmres_tol));
                for j=1:numel(m.layers)
                    l=m.layers{j};
                    if strcmp(key{1},'global')
                        v=l.spectrum.global;
                        tail=norm(v(l.rank+1:end))/norm(v);
                    else
                        tail=l.space.blocks(1).meta.global_relative_tail;
                    end
                    assert(tail<=cfg.layer_tail_tolerances(j)*(1+1e-10));
                end
                export_model(c, cfg, key{1}, out, labels{k}, file);
            end
            assert(all(arrayfun(@(t)t.shifted.relative_residual,c.tests)<=base.gmres_tol));
            close all;
            clear saved result c m t;
        end
    end
    summary = table();
    if strcmp(phase,'plot')
        baseline = readtable(fullfile(out,'reference_global_baseline_timings.csv'));
        raw = table();
        for k = 1:numel(scales)
            for key = {'global','dd'}
                method = key{1};
                file = fullfile(out,[labels{k} '_' method '.mat']);
                s = load(file,'original_model_summary'); m = s.original_model_summary;
                t = readtable(fullfile(out,[labels{k} '_' method '_lazy_timings.csv']));
                assert(height(t)==2*repetitions && all(t.true_residual<=base.gmres_tol));
                row = table(scales(k),scales(k)*base.layer_tail_tolerances(1), ...
                    scales(k)*base.layer_tail_tolerances(2),string(method), ...
                    string(mat2str(m.layer_ranks)),m.basis_bytes/2^20,m.persistent_bytes/2^20, ...
                    m.offline_time,max(m.snapshot_relative_residual(:)), ...
                    mean(t.completion_iterations),mean(t.total_s), ...
                    mean(baseline.total_s)/mean(t.total_s), ...
                    exp(mean(log(t.precompletion_residual))),max(t.true_residual), ...
                    'VariableNames',{'alpha','eta1','eta2','method','ranks','basis_MiB', ...
                    'payload_MiB','offline_s','snapshot_residual_max','completion_mean', ...
                    'online_mean_s','speedup','precompletion_geomean','true_residual_max'});
                summary = [summary;row]; %#ok<AGROW>
                t.alpha = repmat(scales(k),height(t),1);
                t.method = repmat(string(method),height(t),1);
                raw = [raw;t]; %#ok<AGROW>
            end
        end
        writetable(summary,fullfile(out,'summary.csv'));
        writetable(raw,fullfile(out,'timing_all.csv'));
        save(fullfile(out,'summary.mat'),'summary','raw','baseline','base','scales','repetitions');
        plot_waveguide_threshold_tradeoff(out);
        disp(summary);
    end
end

function file = result_file(tag)
    file = helm_output_path('outputs','helmholtz_3d_candidates','large', ...
        ['helmholtz_3d_candidates_' tag '.mat']);
end

function export_model(c,cfg,method,out,label,source)
    model = c.models.(method);
    original_model_summary = rmfield(model,'layers');
    layers = cell(size(model.layers));
    for j = 1:numel(layers)
        l = model.layers{j}; sp = l.space;
        if strcmp(sp.type,'dense')
            sp = rom_reduced_space_utils('make_dense_space',sp.basis);
        else
            sp = rmfield(sp,intersect(fieldnames(sp),{'partition','meta','block_meta'}));
            for b = 1:numel(sp.blocks), sp.blocks(b).meta=[]; end
        end
        layers{j} = struct('space',sp,'Ar0',l.Ar0,'Ar1',l.Ar1,'rank',l.rank);
    end
    model = struct('layers',{layers},'num_layers',numel(layers));
    problem_cfg = c.problem_cfg; rhs = c.rhs;
    save(fullfile(out,[label '_' method '.mat']),'model','cfg','problem_cfg', ...
        'rhs','source','original_model_summary','-v7.3');
end
