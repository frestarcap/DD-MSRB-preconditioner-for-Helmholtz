function run_exp2(stage)
%RUN_EXP2 Experiment 2: elliptic-obstacle scattering (geometric, non-affine parameter).
%   run_exp2(stage) with stage
%     'train'   global MSRB and DD-MSRB on the 96 training geometries
%     'time'    four methods on the 22 test geometries (cost table and figures)
%     'offline' training times: median of three trainings after an untimed warm-up
%     'export'  write results/paper/exp2 in the schema of paper_data/exp2
%     'all'     all of the above, in order
%     'smoke'   a coarse grid through the whole pipeline
%   All computations use one computational thread, as in the paper.
    if nargin < 1, stage = 'all'; end
    old = maxNumCompThreads(1);
    restore = onCleanup(@() maxNumCompThreads(old));
    c = exp2_config();
    if strcmp(stage, 'smoke'), smoke(c); return; end
    stages = {stage};
    if strcmp(stage, 'all'), stages = {'train', 'time', 'offline', 'export'}; end
    for s = stages
        fprintf('=== Experiment 2: %s ===\n', s{1});
        switch s{1}
            case 'train', train_models(c);
            case 'time', time_study(c);
            case 'offline', offline_study(c);
            case 'export', export_paper(c);
            otherwise, error('Unknown stage %s.', s{1});
        end
    end
end

function c = exp2_config()
    p = exp2_params();
    c = struct('train', p.train, 'test', p.test, 'representative', p.representative, ...
        'eta', 1e-5 * [1, 1, 1], 'tol', 1e-12, 'beta', 0.2, 'ilu_droptol', 1e-3, ...
        'restart', 40, 'maxit', 20, 'repetitions', 5);
end

function train_models(c)
    P = dmsrb.ecc.problem();
    F = dmsrb.ecc.family(P);
    keep('global', @() dmsrb.train(F, c.train, c.eta, []));
    keep('dd', @() dmsrb.train(F, c.train, c.eta, dmsrb.ecc.partition(P, c.train)));
end

function keep(name, trainer)
    file = model_file(name);
    if isfile(file), return; end
    [model, info] = trainer();
    info = rmfield(info, 'snapshots');
    s = dmsrb.storage(model);
    save(file, 'model', 'info', 's', '-v7.3');
    fprintf('%s: ranks %s, payload %.3f MiB, offline %.2f s\n', name, mat2str(s.ranks), s.payload_MiB, info.total_s);
end

function time_study(c)
    P = dmsrb.ecc.problem();
    methods = struct('name', {'shifted', 'direct', 'global', 'dd'}, ...
        'model', {[], 'direct', load_model('global'), load_model('dd')});
    [rows, cases] = dmsrb.benchmark(queries(P, c.test, c), methods, sl_options(c), c.repetitions);
    writetable(rows, result_file('timings.csv'));
    q = c.representative;
    histories = table();
    for m = [3, 4, 1]
        h = cases{q, m}.history(:);
        histories = [histories; table(repmat(string(methods(m).name), numel(h), 1), (0:numel(h) - 1)', h, ...
            'VariableNames', {'method', 'step', 'relative_residual'})]; %#ok<AGROW>
    end
    writetable(histories, result_file('histories.csv'));
    u = cases{q, 4}.u; rho = cases{q, 4}.rho; %#ok<NASGU>
    save(result_file('field.mat'), 'u', 'rho', '-v7');
end

function offline_study(c)
% Wall-clock training time (DD: partition included), median of three after a warm-up.
    P = dmsrb.ecc.problem();
    F = dmsrb.ecc.family(P);
    rows = table();
    for style = {'global', 'dd'}
        if strcmp(style{1}, 'dd')
            fit = @() dmsrb.train(F, c.train, c.eta, dmsrb.ecc.partition(P, c.train));
        else
            fit = @() dmsrb.train(F, c.train, c.eta, []);
        end
        fit();
        seconds = zeros(1, 3);
        for r = 1:3
            timer = tic; fit(); seconds(r) = toc(timer);
        end
        rows = [rows; table(string(style{1}), seconds, median(seconds), ...
            'VariableNames', {'method', 'runs_s', 'offline_s'})]; %#ok<AGROW>
        fprintf('%s: offline %.2f s (median of %s)\n', style{1}, median(seconds), mat2str(seconds, 4));
    end
    writetable(rows, result_file('offline_timings.csv'));
end

function export_paper(c)
    out = fullfile(results_root(), 'paper', 'exp2');
    if ~isfolder(out), mkdir(out); end
    P = dmsrb.ecc.problem();
    f = load(result_file('field.mat'));
    field = struct('U', reshape(f.u, P.nx, P.ny).', 'x', P.x(2:end-1).', 'y', P.y(2:end-1).', ...
        'aspect_ratio', c.test(c.representative), 'rho', f.rho);
    save(fullfile(out, 'field.mat'), '-struct', 'field', '-v7');
    g = load(model_file('global'));
    spectra = table();
    for j = 1:numel(g.info.spectra)
        s = g.info.spectra{j}(:);
        tail = sqrt([flipud(cumsum(flipud(s.^2))); 0]) / norm(s);
        energy = [NaN; s.^2 / sum(s.^2)];
        spectra = [spectra; table(repmat(j, numel(tail), 1), (0:numel(s))', energy, tail, ...
            'VariableNames', {'iteration', 'rank', 'energy', 'tail'})]; %#ok<AGROW>
    end
    writetable(spectra, fullfile(out, 'tail_spectra.csv'));
    copyfile(result_file('histories.csv'), fullfile(out, sprintf('residual_histories_query%d.csv', c.representative)));
    copyfile(result_file('timings.csv'), fullfile(out, 'online_timings.csv'));
    ranks = table(); storage = table();
    for name = {'global', 'dd'}
        m = load(model_file(name{1}));
        for j = 1:numel(m.model.layers)
            sp = m.model.layers{j}.space;
            if strcmp(sp.type, 'dense')
                local = sp.k; entries = numel(sp.basis);
            else
                local = sp.ranks; entries = sum(cellfun(@numel, sp.basis));
            end
            ranks = [ranks; table(string(name{1}), j, sp.k, string(mat2str(local)), entries, ...
                'VariableNames', {'method', 'iteration', 'dimension', 'ranks', 'basis_entries'})]; %#ok<AGROW>
        end
        offline = m.info.total_s;
        if isfile(result_file('offline_timings.csv'))
            o = readtable(result_file('offline_timings.csv'), 'TextType', 'string');
            offline = o.offline_s(o.method == name{1});
        end
        storage = [storage; table(string(name{1}), m.s.basis_MiB, m.s.payload_MiB, m.s.object_MiB, ...
            offline, 'VariableNames', {'method', 'basis_MiB', 'payload_MiB', 'object_MiB', 'offline_s'})]; %#ok<AGROW>
    end
    writetable(ranks, fullfile(out, 'rank_storage.csv'));
    writetable(storage, fullfile(out, 'storage.csv'));
    fprintf('exported %s\n', out);
end

function Q = queries(P, params, c)
    Q = struct('mu', {}, 'A', {}, 'b', {}, 'reference', {}, 'precond', {});
    shift = 1i * c.beta * spdiags(dmsrb.ecc.sl_mass(P), 0, P.N, P.N);
    for k = 1:numel(params)
        [A, b] = dmsrb.ecc.system(P, params(k));
        Q(k) = struct('mu', params(k), 'A', A, 'b', b, 'reference', A \ b, ...
            'precond', @(A) dmsrb.sl.ilu(A + shift, c.ilu_droptol));
    end
end

function sl = sl_options(c)
    sl = struct('tol', c.tol, 'gmres', struct('tol', c.tol, 'restart', c.restart, 'maxit', c.maxit));
end

function model = load_model(name)
    s = load(model_file(name), 'model');
    model = s.model;
end

function file = model_file(name)
    folder = fullfile(results_root(), 'exp2', 'models');
    if ~isfolder(folder), mkdir(folder); end
    file = fullfile(folder, [name '.mat']);
end

function file = result_file(name)
    folder = fullfile(results_root(), 'exp2');
    if ~isfolder(folder), mkdir(folder); end
    file = fullfile(folder, name);
end

function root = results_root()
    root = getenv('DMSRB_RESULTS');
    if isempty(root)
        root = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');
    end
end

function smoke(c)
    P = dmsrb.ecc.problem(struct('Nphys', 41, 'npml', 14));
    F = dmsrb.ecc.family(P);
    train = c.train(1:8:end);
    test = c.test(1:4:end);
    train = train(arrayfun(@(mu) dmsrb.ecc.admissible(P, mu), train));
    test = test(arrayfun(@(mu) dmsrb.ecc.admissible(P, mu), test));
    g = dmsrb.train(F, train, c.eta, []);
    d = dmsrb.train(F, train, c.eta, dmsrb.ecc.partition(P, train));
    methods = struct('name', {'shifted', 'direct', 'global', 'dd'}, 'model', {[], 'direct', g, d});
    rows = dmsrb.benchmark(queries(P, test(1:2), c), methods, sl_options(c), 1);
    disp(rows(:, {'method', 'query', 'sl_steps', 'rho_rom', 'true_residual'}));
end
