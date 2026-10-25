function run_exp1(stage)
%RUN_EXP1 Experiment 1: half-space Helmholtz problem.
%   run_exp1(stage) with stage
%     'train'        train every model of the experiment (results/exp1/models)
%     'main'         six methods on the 23 test queries (Table 1, Figures 2-3)
%     'coverage'     training-set size n_s = 6, 12, 24 (Table 2)
%     'sensitivity'  partitions at omega = 53.5 and frequency scaling (Table 3, Figure 5)
%     'offline'      training times of the coverage models (Table 2): median of three
%                    trainings after an untimed warm-up
%     'export'       write results/paper/exp1 in the schema of paper_data/exp1
%     'all'          all of the above, in order
%     'smoke'        a small problem through the whole pipeline (about a minute)
%   All computations use one computational thread, as in the paper.
    if nargin < 1, stage = 'all'; end
    old = maxNumCompThreads(1);
    restore = onCleanup(@() maxNumCompThreads(old));
    c = exp1_config();
    if strcmp(stage, 'smoke'), smoke(c); return; end
    stages = {stage};
    if strcmp(stage, 'all'), stages = {'train', 'main', 'coverage', 'sensitivity', 'offline', 'export'}; end
    for s = stages
        fprintf('=== Experiment 1: %s ===\n', s{1});
        switch s{1}
            case 'train', train_models(c);
            case 'main', main_study(c);
            case 'coverage', coverage_study(c);
            case 'sensitivity', sensitivity_study(c);
            case 'offline', offline_study(c);
            case 'export', export_paper(c);
            otherwise, error('Unknown stage %s.', s{1});
        end
    end
end

%% Training
function train_models(c)
    P = exp1_problem(1);
    F = dmsrb.halfspace.family(P);
    g = fit(F, c.train, c.eta, [], 'global');
    fit(F, c.train, c.eta, dmsrb.halfspace.partition(P, 'material_depth'), 'dd');
    keep('nt', @() dmsrb.train_nt(F, c.train, g.model_ranks));
    keep('ss', @() dmsrb.solution_pod(F, c.train, c.ss_eta, numel(c.eta)));
    for mode = c.partitions(2:end)
        fit(F, c.train, c.eta, dmsrb.halfspace.partition(P, mode{1}), mode{1});
    end
    for n = setdiff(c.coverage_counts, numel(c.train))
        train = linspace(c.train(1), c.train(end), n);
        fit(F, train, c.eta, [], sprintf('global_n%d', n));
        fit(F, train, c.eta, dmsrb.halfspace.partition(P, 'material_depth'), sprintf('dd_n%d', n));
    end
    for scale = c.scales(2:end)
        Ps = exp1_problem(scale);
        Fs = dmsrb.halfspace.family(Ps);
        fit(Fs, c.train, c.eta, [], sprintf('global_s%g', scale));
        fit(Fs, c.train, c.eta, dmsrb.halfspace.partition(Ps, 'material_depth'), sprintf('dd_s%g', scale));
    end
end

function out = fit(F, train, eta, part, name)
    out = keep(name, @() dmsrb.train(F, train, eta, part));
end

function out = keep(name, trainer)
% Train once; later stages load the saved model.
    file = model_file(name);
    if isfile(file)
        out = load(file);
        return;
    end
    [model, info] = trainer();
    info = rmfield_if(info, 'snapshots');
    s = dmsrb.storage(model);
    model_ranks = s.ranks; %#ok<NASGU>
    save(file, 'model', 'info', 's', 'model_ranks', '-v7.3');
    fprintf('%s: ranks %s, payload %.3f MiB, offline %.2f s\n', name, mat2str(s.ranks), ...
        s.payload_MiB, info.total_s);
    out = load(file);
end

%% Main comparison (Table 1, Figures 1-3)
function main_study(c)
    P = exp1_problem(1);
    names = {'shifted', 'direct', 'global', 'dd', 'nt', 'ss'};
    methods = method_list(names, @(n) load_model(n));
    [rows, cases] = dmsrb.benchmark(queries(P, c.test, c), methods, sl_options(c, c.tol), c.repetitions);
    writetable(rows, result_file('main_timings.csv'));
    q = c.representative;
    histories = table();
    for m = 1:numel(names)
        if strcmp(names{m}, 'direct'), continue; end
        h = cases{q, m}.history(:);
        histories = [histories; table(repmat(string(names{m}), numel(h), 1), (0:numel(h) - 1)', h, ...
            'VariableNames', {'method', 'step', 'relative_residual'})]; %#ok<AGROW>
    end
    A = dmsrb.halfspace.operator(P, c.test(q));
    [~, g] = dmsrb.sl.gmres(@(v) A * v, @(v) v, P.b, [], struct('tol', c.tol, 'restart', 100, 'maxit', 1));
    h = g.relres_hist(:);
    histories = [histories; table(repmat("plain_gmres", numel(h), 1), (0:numel(h) - 1)', h, ...
        'VariableNames', {'method', 'step', 'relative_residual'})];
    writetable(histories, result_file('main_histories.csv'));
    u = cases{q, strcmp(names, 'global')}.u; %#ok<NASGU>
    save(result_file('main_field.mat'), 'u', '-v7');
end

%% Training coverage (Table 2)
function coverage_study(c)
    P = exp1_problem(1);
    names = {};
    for n = c.coverage_counts
        names = [names, {coverage_name('global', n, c), coverage_name('dd', n, c)}]; %#ok<AGROW>
    end
    methods = method_list(names, @(n) load_model(n));
    rows = dmsrb.benchmark(queries(P, c.coverage_test, c), methods, sl_options(c, c.tol), c.repetitions);
    writetable(rows, result_file('coverage_timings.csv'));
end

function name = coverage_name(style, n, c)
    name = style;
    if n ~= numel(c.train), name = sprintf('%s_n%d', style, n); end
end

%% Partitions and frequency (Table 3, Figure 5)
function sensitivity_study(c)
    all_rows = table();
    for scale = c.scales
        P = exp1_problem(scale);
        if scale == 1
            names = [{'shifted', 'direct', 'global', 'dd'}, c.partitions(2:end)];
        else
            names = {'shifted', 'direct', sprintf('global_s%g', scale), sprintf('dd_s%g', scale)};
        end
        methods = method_list(names, @(n) load_model(n));
        [rows, cases] = dmsrb.benchmark(queries(P, c.sensitivity_test, c), methods, ...
            sl_options(c, c.sensitivity_tol), c.repetitions);
        rows.omega = repmat(P.omega, height(rows), 1);
        all_rows = [all_rows; rows]; %#ok<AGROW>
        if scale == c.scales(end)
            q = find(abs(c.sensitivity_test - 0.8761) < 1e-12);
            u = cases{q, 4}.u; mu = c.sensitivity_test(q); rho = cases{q, 4}.rho; %#ok<NASGU>
            save(result_file('frequency_field.mat'), 'u', 'mu', 'rho', '-v7');
        end
    end
    writetable(all_rows, result_file('sensitivity_timings.csv'));
end

%% Export in the paper_data schema (figures/make_all.py --data results/paper)
function export_paper(c)
    out = fullfile(results_root(), 'paper', 'exp1');
    if ~isfolder(out), mkdir(out); end
    P = exp1_problem(1);
    g = load(model_file('global'));
    f = load(result_file('main_field.mat'));
    part = dmsrb.halfspace.partition(P, 'material_depth');
    field = struct('U', reshape(f.u, P.nx, P.ny).', 'x', P.x_int, 'y', P.y_int, ...
        'labels', double(part.labels.'), 'source', P.source_xy);
    for j = 1:numel(g.info.spectra), field.(sprintf('spectrum_%d', j)) = g.info.spectra{j}; end
    save(fullfile(out, 'field_omega53.5.mat'), '-struct', 'field', '-v7');
    P4 = exp1_problem(c.scales(end));
    f = load(result_file('frequency_field.mat'));
    weights = abs(reshape(P4.b, P4.nx, P4.ny));
    source = [P4.x_int.' * sum(weights, 2), sum(weights, 1) * P4.y_int] / sum(weights(:));
    field = struct('U', reshape(f.u, P4.nx, P4.ny).', 'x', P4.x_int, 'y', P4.y_int, ...
        'source', source, 'rho', f.rho);
    save(fullfile(out, sprintf('field_omega%g.mat', P4.omega)), '-struct', 'field', '-v7');

    copyfile(result_file('main_timings.csv'), fullfile(out, 'online_timings.csv'));
    copyfile(result_file('main_histories.csv'), fullfile(out, sprintf('residual_histories_query%d.csv', c.representative)));
    rows = table();
    for name = {'global', 'dd'}
        s = load(model_file(name{1}), 's'); s = s.s;
        rows = [rows; table(string(name{1}), s.basis_MiB, s.payload_MiB, s.object_MiB, ...
            'VariableNames', {'method', 'basis_MiB', 'payload_MiB', 'object_MiB'})]; %#ok<AGROW>
    end
    writetable(rows, fullfile(out, 'storage.csv'));

    t = readtable(result_file('coverage_timings.csv'), 'TextType', 'string');
    rows = table();
    for n = c.coverage_counts
        for style = {'global', 'dd'}
            name = coverage_name(style{1}, n, c);
            m = load(model_file(name), 's', 'info');
            r = t(t.method == name, :);
            offline = m.info.total_s;
            if isfile(result_file('offline_timings.csv'))
                o = readtable(result_file('offline_timings.csv'), 'TextType', 'string');
                offline = o.offline_s(o.n_s == n & o.method == style{1});
            end
            rows = [rows; table(n, string(style{1}), string(mat2str(m.s.ranks)), m.s.payload_MiB, ...
                offline, mean(r.total_s), max(r.rho_rom), max(r.true_residual), max(r.repeat), ...
                'VariableNames', {'n_s', 'method', 'ranks', 'payload_MiB', 'offline_s', 'online_mean_s', ...
                'max_rho_rom', 'max_true_residual', 'repetitions'})]; %#ok<AGROW>
        end
    end
    writetable(rows, fullfile(out, 'training_coverage.csv'));

    t = readtable(result_file('sensitivity_timings.csv'), 'TextType', 'string');
    base = t(t.omega == P.omega, :);
    rows = table();
    for variant = [{'global'}, c.partitions]
        name = variant{1};
        if strcmp(name, 'material_depth'), name = 'dd'; end
        s = load(model_file(name), 's'); s = s.s;
        r = base(base.method == name, :);
        rows = [rows; table(string(variant{1}), string(mat2str(s.ranks)), s.basis_MiB, s.operators_MiB, ...
            s.index_MiB, s.payload_MiB, mean(r.total_s), max(r.rho_rom), 'VariableNames', {'variant', ...
            'ranks', 'basis_MiB', 'operators_MiB', 'index_MiB', 'payload_MiB', 'mean_total_s', 'max_rho_rom'})]; %#ok<AGROW>
    end
    writetable(rows, fullfile(out, 'partitions.csv'));

    storage = table(); timings = table();
    for scale = c.scales
        omega = 53.5 * scale;
        r = t(abs(t.omega - omega) < 1e-12, :);
        for style = {'global', 'dd', 'shifted', 'direct'}
            name = style{1};
            if scale ~= 1 && ismember(name, {'global', 'dd'}), name = sprintf('%s_s%g', name, scale); end
            if ismember(style{1}, {'global', 'dd'})
                s = load(model_file(name), 's');
                storage = [storage; table(omega, string(style{1}), s.s.payload_MiB, ...
                    'VariableNames', {'omega', 'method', 'payload_MiB'})]; %#ok<AGROW>
            end
            m = r(r.method == name, :);
            g = groupsummary(m, 'parameter', {'mean'}, 'total_s');
            timings = [timings; table(repmat(omega, height(g), 1), repmat(string(style{1}), height(g), 1), ...
                g.parameter, g.mean_total_s, g.GroupCount, 'VariableNames', ...
                {'omega', 'method', 'c_right', 'mean_total_s', 'repetitions'})]; %#ok<AGROW>
        end
    end
    writetable(sortrows(storage, {'method', 'omega'}), fullfile(out, 'frequency_storage.csv'));
    writetable(timings, fullfile(out, 'frequency_timings.csv'));
    fprintf('exported %s\n', out);
end

%% Offline training times (Table 2)
function offline_study(c)
    P = exp1_problem(1);
    F = dmsrb.halfspace.family(P);
    rows = table();
    for n = c.coverage_counts
        train = linspace(c.train(1), c.train(end), n);
        for style = {'global', 'dd'}
            fit = @() dmsrb.train(F, train, c.eta, partition_for(P, style{1}));   % partition timed too
            seconds = offline_seconds(fit, 3);
            rows = [rows; table(n, string(style{1}), seconds, median(seconds), ...
                'VariableNames', {'n_s', 'method', 'runs_s', 'offline_s'})]; %#ok<AGROW>
            fprintf('n_s = %d %s: offline %.2f s (median of %s)\n', n, style{1}, median(seconds), mat2str(seconds, 4));
        end
    end
    writetable(rows, result_file('offline_timings.csv'));
end

function part = partition_for(P, style)
    part = [];
    if strcmp(style, 'dd'), part = dmsrb.halfspace.partition(P, 'material_depth'); end
end

function seconds = offline_seconds(fit, repetitions)
% Wall-clock time of a complete training (partition included), after one warm-up.
    fit();
    seconds = zeros(1, repetitions);
    for r = 1:repetitions
        timer = tic;
        fit();
        seconds(r) = toc(timer);
    end
end

%% Shared helpers
function Q = queries(P, params, c)
    Q = struct('mu', {}, 'A', {}, 'b', {}, 'reference', {}, 'precond', {});
    for k = 1:numel(params)
        mu = params(k);
        A = dmsrb.halfspace.operator(P, mu);
        shift = 1i * c.beta * spdiags(dmsrb.halfspace.sl_mass(P, mu), 0, P.N, P.N);
        Q(k) = struct('mu', mu, 'A', A, 'b', P.b, 'reference', A \ P.b, ...
            'precond', @(A) dmsrb.sl.ilu(A + shift, c.ilu_droptol));
    end
end

function sl = sl_options(c, tol)
    sl = struct('tol', tol, 'gmres', struct('tol', tol, 'restart', c.restart, 'maxit', c.maxit));
end

function methods = method_list(names, loader)
    methods = struct('name', names, 'model', []);
    for m = 1:numel(names)
        switch names{m}
            case 'shifted', methods(m).model = [];
            case 'direct', methods(m).model = 'direct';
            otherwise, methods(m).model = loader(names{m});
        end
    end
end

function model = load_model(name)
    s = load(model_file(name), 'model');
    model = s.model;
end

function file = model_file(name)
    folder = fullfile(results_root(), 'exp1', 'models');
    if ~isfolder(folder), mkdir(folder); end
    file = fullfile(folder, [name '.mat']);
end

function file = result_file(name)
    folder = fullfile(results_root(), 'exp1');
    if ~isfolder(folder), mkdir(folder); end
    file = fullfile(folder, name);
end

function root = results_root()
    root = getenv('DMSRB_RESULTS');
    if isempty(root)
        root = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');
    end
end

function s = rmfield_if(s, name)
    if isfield(s, name), s = rmfield(s, name); end
end

%% Smoke test
function smoke(c)
    P = dmsrb.halfspace.problem(struct('Nphys', 31, 'omega', 13.4, 'sigma_max', 20));
    F = dmsrb.halfspace.family(P);
    train = linspace(0.8, 0.95, 8);
    g = dmsrb.train(F, train, c.eta, []);
    d = dmsrb.train(F, train, c.eta, dmsrb.halfspace.partition(P, 'material_depth'));
    methods = struct('name', {'shifted', 'direct', 'global', 'dd'}, 'model', {[], 'direct', g, d});
    rows = dmsrb.benchmark(queries(P, [0.83, 0.91], c), methods, sl_options(c, c.tol), 1);
    disp(rows(:, {'method', 'query', 'sl_steps', 'rho_rom', 'true_residual'}));
    sg = dmsrb.storage(g); sd = dmsrb.storage(d);
    fprintf('smoke: global ranks %s, DD ranks %s, payload %.3f vs %.3f MiB\n', ...
        mat2str(sg.ranks), mat2str(sd.ranks), sg.payload_MiB, sd.payload_MiB);
end
