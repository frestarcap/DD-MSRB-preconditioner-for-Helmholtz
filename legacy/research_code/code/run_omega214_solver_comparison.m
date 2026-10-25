function rows = run_omega214_solver_comparison(stage)
%RUN_OMEGA214_SOLVER_COMPARISON Matched zero-start Helm half-space baselines.
% Run smoke, algebraic, then waveholtz in separate singleCompThread processes.
% The paper's frozen grid/PML/source config is checked against saved matrices.
% All costs include solver setup and an independent true-residual check;
% common problem assembly and file I/O are excluded. No ROM is trained.
    if nargin < 1, stage = 'smoke'; end
    assert(ismember(stage, {'smoke', 'algebraic', 'waveholtz'}));
    assert(maxNumCompThreads == 1, 'Launch MATLAB with -singleCompThread.');
    out = helm_output_path('outputs', 'omega214_solver_comparison_20260910');
    if ~isfolder(out), mkdir(out); end
    archive = helm_output_path('outputs', 'reviewer_20260909', 'high_frequency_waveholtz');
    scale = 4;
    if strcmp(stage, 'smoke'), scale = 0.25; end
    saved = load(fullfile(archive, sprintf('global_scale%g_prefix_result.mat', scale)), 'result');
    cfg = saved.result.cfg; clear saved;
    cfg = rmfield(cfg, 'frozen_online_file');
    cfg.setup_only = true;
    cfg.test_params = 0.8761;
    cfg.wh_verbose_gmres = false;
    cfg.wh_tol = 1e-10; cfg.wh_gmres_restart = 40; cfg.wh_gmres_maxit = 20;
    cfg.online_true_helm_tol = 1e-10;
    cfg.online_helm_accept_tol = 1e-10;
    cfg.shifted_laplacian_beta = 0.2;
    cfg.shifted_laplacian_solver = 'ilu';
    cfg.shifted_laplacian_ilu_droptol = 1e-3;
    cfg.shifted_laplacian_restart = 40; cfg.shifted_laplacian_maxit = 20;
    cfg.shifted_laplacian_tol = 1e-10; cfg.shifted_laplacian_reorth = true;
    r = helm_waveholtz_correction_rom(cfg);
    frozen = load(fullfile(archive, sprintf('global_scale%g_prefix_model.mat', scale)), ...
        'A_base', 'A_right', 'rhs');
    assert(isequaln(r.problem.A_base, frozen.A_base));
    assert(isequaln(r.problem.A_right, frozen.A_right));
    assert(isequaln(r.rhs, frozen.rhs));
    cfg = r.cfg; clear r frozen;
    cfg.setup_only = false;
    metadata = struct('matlab_version', version, 'computer', computer, ...
        'threads', maxNumCompThreads, 'created_at', char(datetime('now')), ...
        'matched_archived_operator_and_rhs', true, 'cfg', cfg);
    save(fullfile(out, [stage '_metadata.mat']), 'metadata');
    queries = [0.8297, 0.8761, 0.9431];
    methods = {'direct', 'shifted_laplacian'};
    reps = 0:3; % One full-size warm-up per method/query, three recorded runs.
    if strcmp(stage, 'waveholtz')
        methods = {'waveholtz'}; reps = 1;
    elseif strcmp(stage, 'smoke')
        methods = {'direct', 'shifted_laplacian', 'waveholtz'};
        queries = 0.8761; reps = 1;
    end
    rows = table();
    for mu = queries
        for rep = reps
            order = methods;
            if mod(rep, 2) == 0, order = fliplr(order); end
            for im = 1:numel(order)
                method = order{im};
                tag = sprintf('%s_%s_mu%.4f_rep%d', stage, method, mu, rep);
                file = fullfile(out, [tag '.mat']);
                if isfile(file)
                    old = load(file, 'row'); row = old.row;
                else
                    cfg.test_params = mu; cfg.baseline_probe_solver = method;
                    fprintf('START %s %s\n', tag, char(datetime('now')));
                    result = helm_waveholtz_correction_rom(cfg);
                    row = result.row; row.repetition = rep;
                    reference_error = NaN;
                    ref_file = fullfile(out, sprintf('%s_direct_mu%.4f_rep1.mat', stage, mu));
                    if strcmp(stage, 'waveholtz')
                        ref_file = fullfile(out, sprintf('algebraic_direct_mu%.4f_rep1.mat', mu));
                    end
                    if ~strcmp(method, 'direct') && isfile(ref_file)
                        ref = load(ref_file, 'result');
                        reference_error = norm(result.u-ref.result.u)/norm(ref.result.u);
                        clear ref;
                    end
                    row.relative_field_difference_from_direct = reference_error;
                    save(file, 'row', 'result', '-v7.3');
                    clear result;
                    fprintf('DONE %s: %.6fs, %d iterations, rho %.3e, accepted %d\n', ...
                        tag, row.total_s, row.iterations, row.true_residual, row.accepted);
                end
                rows = [rows; struct2table(row)]; %#ok<AGROW>
                writetable(rows, fullfile(out, [stage '_samples.csv']));
            end
        end
    end
    assert(all(rows.accepted), 'One or more solves failed the true-residual target.');
    fprintf('SOLVER_COMPARISON_PASS %s\n', stage);
    disp(rows(:, {'method', 'parameter', 'repetition', 'total_s', 'iterations', 'true_residual'}));
end
