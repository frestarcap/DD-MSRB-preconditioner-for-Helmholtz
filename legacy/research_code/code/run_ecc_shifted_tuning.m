function run_ecc_shifted_tuning(mode)
% HISTORICAL: DD uses additional local tail caps; not the paper zero-start allocation.
% Use run_ecc_dd_budget_study for the strict single-global-budget experiment.
% Research-only ECC tuning; preserves the published cohort and solver settings.
% Run with one computational thread. All numerical evidence goes to outputs.
    if nargin < 1, mode = 'pilot'; end
    old = maxNumCompThreads(1);
    cleanup = onCleanup(@()maxNumCompThreads(old)); %#ok<NASGU>
    out = helm_output_path('outputs','ecc_shifted_tuning_20260919');
    if ~isfolder(out), mkdir(out); end
    source = helm_output_path('outputs','ecc_two_layer_paper_20260911','paper_data.mat');
    s = load(source,'global_results'); base = s.global_results.cfg; clear s;
    if strcmp(mode,'pilot')
        specs = [12,2,1e-5; 12,4,1e-5; 24,4,1e-5; 48,4,1e-6];
    elseif strcmp(mode,'refine')
        specs = [96,3,1e-6; 96,3,1e-8];
    else
        error('Unknown mode: %s',mode);
    end
    rows = table();
    for si = 1:size(specs,1)
        for method = {'global','dd'}
            name = method{1}; ns = specs(si,1); depth = specs(si,2); eta = specs(si,3);
            tag = sprintf('%s_n%d_d%d_eta%.0e',name,ns,depth,eta);
            file = fullfile(out,[tag '.mat']);
            if isfile(file)
                v = load(file,'row'); row = v.row;
            else
                cfg = base;
                if ns ~= numel(base.ratio_train)
                    cfg.ratio_train = exp(linspace(log(base.ratio_min),log(base.ratio_max),ns));
                end
                cfg.ratio_ntrain = ns;
                cfg.corr_num_layers = depth;
                cfg.corr_rank_rtol = [];
                cfg.corr_rank_rtols = repmat(eta,1,depth);
                for key = {'corr_block3_rtol_near','corr_block3_rtol_far','corr_block3_rtol_pml', ...
                        'corr_block3_tail_tol_near','corr_block3_tail_tol_far','corr_block3_tail_tol_pml'}
                    cfg.(key{1}) = cfg.corr_rank_rtols;
                end
                cfg.corr_block3_global_tail_tols = cfg.corr_rank_rtols;
                cfg.corr_block3_global_tail_strategy = 'marginal_storage';
                cfg.corr_basis_style = 'global';
                if strcmp(name,'dd'), cfg.corr_basis_style = 'block3'; end
                cfg.reference_mode = 'helmholtz'; cfg.rom_check_target = 'helm';
                cfg.online_completion_mode = 'shifted_laplacian';
                cfg.shifted_laplacian_solver = 'ilu'; cfg.shifted_laplacian_beta = .2;
                cfg.shifted_laplacian_ilu_droptol = 1e-3;
                cfg.shifted_laplacian_restart = 40; cfg.shifted_laplacian_maxit = 20;
                cfg.shifted_laplacian_tol = 1e-10; cfg.helm_gmres_track_tol = 1e-10;
                cfg.parallel_train = false; cfg.parallel_test = false;
                cfg.run_wave_baselines = false; cfg.helm_baseline_iters = 1;
                cfg.do_plots = false; cfg.save_plots = false; cfg.save_benchmark = false;
                fprintf('ECC_TUNING_START %s\n',tag);
                r = waveholtz_2d_pml_scattering_ecc_correction_rom(cfg);
                assert(isequal(r.cfg.ratio_test,base.ratio_test),'Test geometries changed.');
                assert(all(min(abs(r.cfg.ratio_train(:)-base.ratio_test(:).'),[],1)>1e-12));
                cfg = r.cfg; model = struct('layers',{cell(depth,1)},'fixed_pod',false);
                basis_entries = 0; ranks = zeros(1,depth); training = cell(depth,1);
                for j = 1:depth
                    l = r.correction_rom.layers{j}; sp = l.basis_space;
                    model.layers{j} = struct('space',sp);
                    ranks(j) = rom_reduced_space_utils('dimension',sp);
                    if strcmp(sp.type,'dense'), basis_entries = basis_entries+numel(sp.basis);
                    else, basis_entries = basis_entries+numel(sp.Un)+numel(sp.Uf)+numel(sp.Up); end
                    training{j} = struct('s',l.pod.singular_values, ...
                        'residual',l.train_fgmres_relres,'diagnostics',{l.trajectory_diagnostics});
                end
                post = r.test.online_precorr_relres_final;
                residual = r.test.online_fp_relres_final;
                assert(all(residual<=1e-10),'Completion failed.');
                row = table(string(tag),string(name),ns,depth,eta,mean(post),max(post), ...
                    mean(r.test.online_fp_iter),mean(r.test.online_precorr_time), ...
                    mean(r.test.online_total_time),basis_entries*16/2^20,r.timing.offline, ...
                    max(residual),max(r.test.online_final_err_helm_full),string(mat2str(ranks)), ...
                    'VariableNames',{'tag','method','ntrain','depth','eta','mean_post','max_post', ...
                    'mean_iterations','ROM_s','exploratory_total_s','basis_MiB','offline_s', ...
                    'max_residual','max_error','ranks'});
                test = r.test;
                test = rmfield(test,intersect(fieldnames(test),{'systems'}));
                save(file,'model','cfg','row','test','training','source','-v7.3');
                clear r model test training;
            end
            rows = [rows;row]; %#ok<AGROW>
            disp(row);
            writetable(rows,fullfile(out,[mode '_training_summary.csv']));
        end
    end
    fprintf('ECC_TUNING_%s_PASS\n',upper(mode));
end
