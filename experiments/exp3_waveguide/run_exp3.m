function run_exp3(stage, omega)
%RUN_EXP3 Experiment 3: 3D Gaussian waveguide (omega = 24, 36, 48, 60).
%   run_exp3(stage, omega) with stage
%     'train'         global MSRB and DD-MSRB at this omega (concurrent training solves)
%     'online'        SL baseline, global and DD at the three common queries
%     'validation'    omega = 48 only: ten queries, global (k = 3 and k = 2) and DD
%     'export'        results/paper/exp3 in the schema of paper_data/exp3 (all omegas)
%     'verify'        omega = 6: train and solve end to end on a workstation (minutes)
%   The full problems need a cluster node (omega = 48: N = 5,543,003, about 100 GB);
%   see experiments/exp3_waveguide/README.md and slurm/.
    if nargin < 2, omega = 6; end
    if strcmp(stage, 'verify'), omega = 6; end
    c = exp3_config(omega);
    switch stage
        case {'train', 'verify'}
            train_models(c);
            if strcmp(stage, 'verify'), online_study(c); end
        case 'online', online_study(c);
        case 'validation', validation_study(c);
        case 'export', export_paper();
        otherwise, error('Unknown stage %s.', stage);
    end
end

function train_models(c)
    P = dmsrb.waveguide.problem(c.omega);
    fprintf('omega = %g: grid %s, N = %d\n', c.omega, mat2str(P.n), P.N);
    pool = open_pool(c);
    cleanup = onCleanup(@() delete(pool));
    first = [];
    for style = {'global', 'dd'}
        file = model_file(c, style{1});
        if isfile(file), continue; end
        part = [];
        if strcmp(style{1}, 'dd'), part = dmsrb.waveguide.partition(P); end
        [model, info, first] = train_waveguide(P, c, part, first, [file(1:end-4) '_progress.mat']);
        s = dmsrb.storage(model);
        save(file, 'model', 'info', 's', '-v7.3');
        fprintf('%s: ranks %s, payload %.2f MiB, offline %.0f s (total of all training solves %.0f s)\n', ...
            style{1}, mat2str(s.ranks), s.payload_MiB, info.total_s, sum(info.solve_s(:)));
    end
end

function pool = open_pool(c)
    pool = [];
    if c.workers <= 1 || isempty(ver('parallel')), return; end
    pool = gcp('nocreate');
    if isempty(pool)
        cluster = parcluster('Processes');
        cluster.NumThreads = c.worker_threads;
        pool = parpool(cluster, c.workers);
    end
    fetchOutputs(parfevalOnAll(pool, @maxNumCompThreads, 0, c.worker_threads));
end

function online_study(c)
    maxNumCompThreads(c.online_threads);
    P = dmsrb.waveguide.problem(c.omega);
    models = {[], load_model(c, 'global'), load_model(c, 'dd')};
    names = {'shifted', 'global', 'dd'};
    rows = table(); histories = table();
    for mu = c.test
        [A, sl] = online_setup(P, c, mu);
        for m = 1:3
            [u, info] = dmsrb.solve(A, P.b, mu, models{m}, sl);
            assert(info.rho <= c.tol, 'Online residual target missed.');
            rows = [rows; table(c.omega, P.N, mu, string(names{m}), info.total_s, info.setup_s, ...
                info.solve_s, info.ROM_s, info.sl_steps, info.rho_rom, info.rho, 'VariableNames', ...
                {'omega', 'N', 'mu', 'method', 'online_s', 'setup_s', 'solve_s', 'rom_s', 'sl_steps', ...
                'rho_rom', 'true_residual'})]; %#ok<AGROW>
            if mu == c.representative
                h = info.history(:);
                histories = [histories; table(repmat(string(names{m}), numel(h), 1), (0:numel(h) - 1)', h, ...
                    'VariableNames', {'method', 'step', 'relative_residual'})]; %#ok<AGROW>
                if strcmp(names{m}, 'dd'), save_slice(P, c, u); end
            end
            fprintf('omega=%g mu=%.3f %s: %.1f s, %d SL steps, rho_ROM %.2e\n', c.omega, mu, names{m}, ...
                info.total_s, info.sl_steps, info.rho_rom);
        end
    end
    writetable(rows, result_file(c, 'online.csv'));
    writetable(histories, result_file(c, 'histories.csv'));
end

function [A, sl] = online_setup(P, c, mu)
    A = dmsrb.waveguide.operator(P, mu);
    shift = 1i * c.beta * spdiags(dmsrb.waveguide.mass(P, mu), 0, P.N, P.N);
    sl = struct('tol', c.tol, 'precond', @(A, ~) dmsrb.sl.mg(A + shift, P.n, c.mg), ...
        'gmres', waveguide_gmres_options(c, c.tol, c.maxit));
end

function save_slice(P, c, u)
    U = reshape(u, P.n);
    mid = ceil(P.n(3) / 2);
    x = P.coordinates{1}; y = P.coordinates{2}; real_part = real(U(:, :, mid));
    mu = c.representative; z = P.coordinates{3}(mid); %#ok<NASGU>
    save(result_file(c, 'field.mat'), 'x', 'y', 'real_part', 'mu', 'z', '-v7');
end

function validation_study(c)
% Ten queries at omega = 48: the three common queries timed three times after a
% warm-up, seven added queries once; rotating order; one SL baseline per query.
    assert(c.omega == 48, 'The validation study is defined at omega = 48.');
    maxNumCompThreads(c.online_threads);
    P = dmsrb.waveguide.problem(c.omega);
    g = load_model(c, 'global'); d = load_model(c, 'dd');
    g2 = g; g2.layers = g.layers(1:2);
    models = {g, d, g2}; names = {'global3', 'dd3', 'global2'};
    queries = [c.test, c.added];
    rows = table();
    for q = 1:numel(queries)
        mu = queries(q);
        [A, sl] = online_setup(P, c, mu);
        common = q <= numel(c.test);
        for rep = (1 - common):(1 + 2 * common)
            for m = circshift(1:3, [0, mod(q + rep - 1, 3)])
                [~, info] = dmsrb.solve(A, P.b, mu, models{m}, sl);
                if rep > 0, rows = [rows; row(q, mu, common, names{m}, rep, info)]; end %#ok<AGROW>
            end
        end
        [~, info] = dmsrb.solve(A, P.b, mu, [], sl);
        rows = [rows; row(q, mu, common, 'shifted_mg', 1, info)]; %#ok<AGROW>
        writetable(rows, result_file(c, 'validation.csv'));
    end
end

function r = row(q, mu, common, method, rep, info)
    groups = {'added', 'common'};
    r = table(q, mu, string(groups{common + 1}), string(method), rep, info.total_s, info.sl_steps, ...
        info.rho_rom, info.rho, 'VariableNames', {'query_index', 'mu', 'group', 'method', 'repeat', ...
        'total_s', 'sl_steps', 'rho_rom', 'true_residual'});
end

function export_paper()
    out = fullfile(results_root(), 'paper', 'exp3');
    if ~isfolder(out), mkdir(out); end
    queries = table(); storage = table(); offline = table();
    for omega = [24, 36, 48, 60]
        c = exp3_config(omega);
        if ~isfile(result_file(c, 'online.csv')), continue; end
        queries = [queries; readtable(result_file(c, 'online.csv'), 'TextType', 'string')]; %#ok<AGROW>
        row = table(omega, 'VariableNames', {'omega'});
        for style = {'global', 'dd'}
            m = load(model_file(c, style{1}), 's', 'info');
            row.([style{1} '_payload_MiB']) = m.s.payload_MiB;
            offline = [offline; table(omega, string(style{1}), sum(m.info.solve_s(:)), ...
                'VariableNames', {'omega', 'method', 'offline_total_s'})]; %#ok<AGROW>
            if omega == 48
                storage = [storage; table(string(style{1}), m.s.basis_MiB, m.s.payload_MiB, m.s.object_MiB, ...
                    'VariableNames', {'method', 'basis_MiB', 'payload_MiB', 'object_MiB'})]; %#ok<AGROW>
            end
        end
        row.payload_saving_percent = 100 * (1 - row.dd_payload_MiB / row.global_payload_MiB);
        if omega == 24, frequency = row; else, frequency = [frequency; row]; end %#ok<AGROW>
    end
    writetable(queries, fullfile(out, 'frequency_queries.csv'));
    writetable(frequency, fullfile(out, 'frequency_storage.csv'));
    writetable(storage, fullfile(out, 'storage_omega48.csv'));
    writetable(offline, fullfile(out, 'offline_times.csv'));
    c = exp3_config(48);
    copyfile(result_file(c, 'histories.csv'), fullfile(out, 'residual_histories_omega48_mu0.317.csv'));
    f = load(result_file(c, 'field.mat'));
    field = struct('x', f.x, 'y', f.y, 'real', f.real_part, 'mu', f.mu, 'z', f.z);
    save(fullfile(out, 'field_omega48_mu0.317.mat'), '-struct', 'field', '-v7');
    if isfile(result_file(c, 'validation.csv'))
        v = readtable(result_file(c, 'validation.csv'), 'TextType', 'string');
        g = groupsummary(v, {'query_index', 'mu', 'group', 'method'}, {'mean', 'std'}, {'total_s', 'sl_steps'});
        g.std_total_s(g.GroupCount == 1) = NaN;
        t = table(g.query_index, g.mu, g.group, g.method, g.GroupCount, g.mean_total_s, g.std_total_s, ...
            g.mean_sl_steps, 'VariableNames', {'query_index', 'mu', 'group', 'method', 'repetitions', ...
            'mean_total_s', 'sd_total_s', 'mean_sl_steps'});
        writetable(t, fullfile(out, 'validation_omega48.csv'));
    end
    fprintf('exported %s\n', out);
end

function model = load_model(c, style)
    s = load(model_file(c, style), 'model');
    model = s.model;
end

function file = model_file(c, style)
    folder = fullfile(results_root(), 'exp3', sprintf('omega%g', c.omega));
    if ~isfolder(folder), mkdir(folder); end
    file = fullfile(folder, [style '.mat']);
end

function file = result_file(c, name)
    file = fullfile(fileparts(model_file(c, 'x')), name);
end

function root = results_root()
    root = getenv('DMSRB_RESULTS');
    if isempty(root)
        root = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');
    end
end
