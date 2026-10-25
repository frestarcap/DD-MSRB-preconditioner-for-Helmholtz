function [model, info, first] = train_waveguide(P, c, part, first, checkpoint)
%TRAIN_WAVEGUIDE Trajectory-aware training of the waveguide reduced spaces.
%   As dmsrb.train, but every snapshot z = A(mu)\q is computed by SL-MG GMRES to a
%   relative residual of c.snapshot_tol (the training solves run concurrently on a
%   process pool when one is open), and each trajectory is advanced with one pass
%   of Gram--Schmidt.  The first-iteration snapshots do not depend on the space
%   and can be shared between the global and the DD model (argument first).
%   Progress is checkpointed after every iteration and every accepted solve.
    clock = tic;
    count = numel(c.train);
    depth = numel(c.eta);
    Q = repmat({complex(zeros(P.N, depth + 1))}, 1, count);
    for s = 1:count, Q{s}(:, 1) = P.b / max(norm(P.b), eps); end
    model = struct('layers', {cell(depth, 1)}, 'affine', true, 'theta', P.theta);
    info = struct('solve_s', zeros(depth, count), 'residual', zeros(depth, count), ...
        'spectra', {cell(depth, 1)}, 'shared_s', 0);
    done = 0;
    if isfile(checkpoint)
        saved = load(checkpoint, 'state');
        [Q, model, info, done] = deal(saved.state.Q, saved.state.model, saved.state.info, saved.state.done);
        fprintf('resuming after iteration %d (%s)\n', done, checkpoint);
    end
    cache = [checkpoint(1:end-4) '_snapshots'];
    for j = done + 1:depth
        if j == 1 && ~isempty(first)
            E = first.snapshots;
            info.solve_s(1, :) = first.solve_s;
            info.residual(1, :) = first.residual;
            info.shared_s = first.wall_s;
        else
            timer = tic;
            [E, info.solve_s(j, :), info.residual(j, :)] = snapshots(P, c, Q, j, fullfile(cache, sprintf('iteration_%d', j)));
            if j == 1, first = struct('snapshots', E, 'solve_s', info.solve_s(1, :), ...
                    'residual', info.residual(1, :), 'wall_s', toc(timer)); end
        end
        [space, info.spectra{j}] = dmsrb.compress(E, part, c.eta(j));
        clear E
        layer = struct('space', space, 'operators', {dmsrb.waveguide.reduce(P, space)});
        model.layers{j} = layer;
        for s = 1:count
            mu = c.train(s);
            if norm(Q{s}(:, j)) <= 1e-14, continue; end
            z = dmsrb.apply_rom(layer, [], P.theta(mu), Q{s}(:, j));
            w = P.K * z + dmsrb.waveguide.mass(P, mu) .* z;
            for i = 1:j
                w = w - Q{s}(:, i) * (Q{s}(:, i)' * w);
            end
            if norm(w) > 1e-13, Q{s}(:, j + 1) = w / norm(w); end
        end
        fprintf('iteration %d: rank %d, max snapshot residual %.2e\n', j, space.k, max(info.residual(j, :)));
        state = struct('Q', {Q}, 'model', model, 'info', info, 'done', j); %#ok<NASGU>
        save([checkpoint '.tmp'], 'state', '-v7.3');
        movefile([checkpoint '.tmp'], checkpoint, 'f');
    end
    info.total_s = toc(clock) + info.shared_s;
end

function [E, times, residuals] = snapshots(P, c, Q, j, cache)
    count = numel(c.train);
    E = complex(zeros(P.N, count));
    times = zeros(1, count);
    residuals = zeros(1, count);
    if ~isfolder(cache), mkdir(cache); end
    q = cellfun(@(V) V(:, j), Q, 'UniformOutput', false);
    pool = gcp('nocreate');
    if isempty(pool)
        for s = 1:count
            [E(:, s), times(s), residuals(s)] = cached_solve(P, c, c.train(s), q{s}, cache, s);
        end
    else
        constant = parallel.pool.Constant(P);
        parfor s = 1:count
            [E(:, s), times(s), residuals(s)] = cached_solve(constant.Value, c, c.train(s), q{s}, cache, s);
        end
    end
    assert(all(residuals <= c.snapshot_tol * (1 + 1e-6)), 'dmsrb:snapshot', ...
        'A training solve missed the snapshot tolerance; increase snapshot_maxit.');
end

function [z, seconds, residual] = cached_solve(P, c, mu, q, cache, s)
    file = fullfile(cache, sprintf('sample_%02d.mat', s));
    A = dmsrb.waveguide.operator(P, mu);
    if isfile(file)
        saved = load(file, 'z', 'seconds');
        z = saved.z; seconds = saved.seconds;
        residual = norm(q - A * z) / max(norm(q), eps);
        if residual <= c.snapshot_tol * (1 + 1e-6), return; end
    end
    timer = tic;
    M = dmsrb.sl.mg(A + 1i * c.beta * spdiags(dmsrb.waveguide.mass(P, mu), 0, P.N, P.N), P.n, c.mg);
    z = dmsrb.sl.gmres(@(v) A * v, M, q, [], waveguide_gmres_options(c, c.snapshot_tol, c.snapshot_maxit));
    seconds = toc(timer);
    residual = norm(q - A * z) / max(norm(q), eps);
    fprintf('snapshot mu=%.6f true=%.3e wall=%.2fs\n', mu, residual, seconds);
    if residual <= c.snapshot_tol * (1 + 1e-6), save(file, 'z', 'seconds', '-v7.3'); end
end
