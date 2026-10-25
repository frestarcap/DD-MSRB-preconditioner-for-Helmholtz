function summary = run_halfspace_default_model_retime()
%RUN_HALFSPACE_DEFAULT_MODEL_RETIME Partition and omega=53.5 timings with the main DD model.
% Scope: Helm half-space, Experiment 1.  The partition study and the
% omega=53.5 point of the frequency study (run_reviewer_20260909_study) used a
% separately trained default DD model with ranks [44 80 122].  The rest of
% Experiment 1 uses the main model of halfspace_dd_refinement/review_repaired
% (ranks [44 80 123]).  This driver converts the main model to the reviewer
% format, checks it against reduced operators assembled from the reviewer
% operators, and times all partition variants and both baselines in one
% session with the reviewer protocol (five queries, tolerance 1e-10, one
% untimed warm-up plus five timed repetitions).  The trained models are reused
% unchanged; nothing is retrained.
    out = helm_output_path('outputs','halfspace_default_retime_20261024');
    if ~isfolder(out), mkdir(out); end
    src = helm_output_path('outputs','reviewer_20260909','frequency_1');
    s = load(fullfile(src,'problem.mat'),'c','b','ops');
    c = s.c; b = s.b; ops = s.ops; clear s;
    assert(c.scale == 1 && c.tol == 1e-10 && size(c.queries,1) == 5);

    d = load(helm_output_path('outputs','halfspace_dd_refinement', ...
        'review_repaired_20260905','slabs1-aligned-sym8.mat'),'artifact');
    a = d.artifact; clear d;
    assert(norm(a.rhs - b) == 0, 'Main model was built for a different right-hand side.');
    models.dd = cell(1, numel(a.layers));
    for j = 1:numel(a.layers)
        sp = deploy_space(a.layers{j}.basis_space);
        reduced = cell(size(ops));
        for k = 1:numel(ops)
            reduced{k} = rom_reduced_space_utils('assemble_reduced_matrix', ops{k}, sp);
        end
        stored = {a.layers{j}.Ar_base, a.layers{j}.Ar_right};
        for k = 1:numel(ops)
            assert(norm(reduced{k} - stored{k}, 'fro') <= 1e-10*norm(stored{k}, 'fro'), ...
                'Reduced operator mismatch in layer %d.', j);
        end
        models.dd{j} = struct('space', sp, 'operators', {reduced});
    end
    for v = {'global','merged','signed','cartesian'}
        m = load(fullfile(src, [v{1} '.mat']), 'model'); models.(v{1}) = m.model;
    end

    variants = {'global','dd','merged','signed','cartesian','shifted','direct'};
    summary = table();
    for iv = 1:numel(variants)
        variant = variants{iv};
        if isfield(models, variant), model = models.(variant); else, model = {}; end
        rows = benchmark(ops, b, c.queries, model, variant, c.tol, 5);
        writetable(rows, fullfile(out, [variant '_timing.csv']));
        st = storage(model);
        summary = [summary; table(string(variant), string(mat2str(st.ranks)), st.basis, ...
            st.operators, st.index, st.payload, mean(rows.total_s), ...
            max(rows.post_ROM_residual), mean(rows.completion_iterations), ...
            'VariableNames', {'variant','ranks','basis_MiB','operators_MiB','index_MiB', ...
            'payload_MiB','mean_total_s','max_post_ROM_residual','mean_completion_iterations'})]; %#ok<AGROW>
    end
    writetable(summary, fullfile(out, 'summary.csv'));
    disp(summary);
end

function sp = deploy_space(sp)
    % Same deployed representation as run_reviewer_20260909_study.
    if strcmp(sp.type, 'dense'), return; end
    sp = rmfield(sp, intersect(fieldnames(sp), {'partition','block_names','block_meta'}));
    sp.blocks = rmfield(sp.blocks, intersect(fieldnames(sp.blocks), {'meta','label','name'}));
end

function st = storage(model)
    st = struct('basis',0,'operators',0,'index',0,'ranks',zeros(1,numel(model)));
    for j = 1:numel(model)
        l = model{j}; sp = l.space; st.ranks(j) = sp.k;
        if strcmp(sp.type,'dense'), entries = numel(sp.basis);
        else
            entries = sum(arrayfun(@(b)numel(b.basis), sp.blocks));
            st.index = max(st.index, 8*sum([sp.blocks.count])/2^20);
        end
        st.basis = st.basis + 16*entries/2^20;
        st.operators = st.operators + 16*sum(cellfun(@numel, l.operators))/2^20;
    end
    st.payload = st.basis + st.operators + st.index;
end

function a = coefficients(mu, nops)
    a = [1, mu(2)^-2 - 1];
    if nops == 3, a(3) = mu(1)^-2 - 1; end
end

function A = assemble(ops, mu)
    a = coefficients(mu, numel(ops)); A = a(1)*ops{1};
    for k = 2:numel(ops), A = A + a(k)*ops{k}; end
end

function rows = benchmark(ops, b, queries, model, variant, tol, repeats)
    % Same protocol as the benchmark of run_reviewer_20260909_study.
    rows = table();
    for k = 1:size(queries,1)
        mu = queries(k,:); A = assemble(ops, mu);
        w2 = max(full(diag(ops{2}))); mass = w2*ones(numel(b),1);
        a = coefficients(mu, numel(ops));
        for i = 2:numel(ops), mass = mass + a(i)*full(diag(ops{i})); end
        for rep = 0:repeats
            tt = tic; iterations = 0; pre = NaN; setup = 0;
            if strcmp(variant,'direct')
                x = A\b;
            else
                if strcmp(variant,'shifted'), x = zeros(size(b));
                else, x = rom_replay_inverse_action(A, b, mu, model); end
                pre = norm(b - A*x)/norm(b);
                if pre > tol
                    [M, mi] = build_shifted_laplacian_preconditioner( ...
                        A + 1i*0.2*spdiags(mass, 0, numel(b), numel(b)), ...
                        struct('solver','ilu','ilu_droptol',1e-3));
                    setup = mi.setup_time;
                    [x, si] = shifted_laplacian_gmres(@(v)A*v, M, b, x, ...
                        struct('tol',tol,'restart',40,'maxit',20,'reorthogonalize',true));
                    iterations = si.iter_count; clear M;
                end
            end
            rho = norm(b - A*x)/norm(b); elapsed = toc(tt);
            assert(rho <= tol, 'Final true Helmholtz residual failed.');
            if rep > 0
                rows = [rows; table(string(variant), mu(1), mu(2), rep, elapsed, setup, ...
                    iterations, pre, rho, 'VariableNames', {'variant','c_left','c_right', ...
                    'repetition','total_s','completion_setup_s','completion_iterations', ...
                    'post_ROM_residual','true_residual'})]; %#ok<AGROW>
            end
        end
    end
end
