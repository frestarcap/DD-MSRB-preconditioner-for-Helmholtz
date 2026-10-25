function out = rom_reduced_space_utils(action, varargin)
%ROM_REDUCED_SPACE_UTILS
% Shared reduced-space helpers for dense POD and blockwise reduced spaces.
%
% Supported actions:
%   make_dense_space(basis)
%   build_block3_partition(grid)
%   build_block3_space(snapshot_matrix, partition, cfg)
%   build_blockn_space(snapshot_matrix, partition, cfg)
%   dimension(space)
%   is_empty(space)
%   project(space, vec, shared_indices_optional)
%   lift(space, coeff, shared_indices_optional)
%   assemble_reduced_matrix(Aop, space)

    if nargin < 1 || ~ischar(action)
        error('rom_reduced_space_utils requires action as a string.');
    end

    switch lower(strtrim(action))
        case 'make_dense_space'
            out = local_make_dense_space(varargin{:});
        case 'build_block3_partition'
            out = local_build_block3_partition(varargin{:});
        case 'build_block3_space'
            out = local_build_block3_space(varargin{:});
        case 'build_blockn_space'
            out = local_build_blockn_space(varargin{:});
        case 'dimension'
            out = local_dimension(varargin{:});
        case 'is_empty'
            out = local_dimension(varargin{:}) == 0;
        case 'project'
            out = local_project(varargin{:});
        case 'lift'
            out = local_lift(varargin{:});
        case 'assemble_reduced_matrix'
            out = local_assemble_reduced_matrix(varargin{:});
        otherwise
            error('rom_reduced_space_utils: unknown action "%s".', action);
    end
end


function space = local_make_dense_space(basis)
    if ~isnumeric(basis)
        error('Dense basis must be numeric.');
    end
    space = struct();
    space.type = 'dense';
    space.n = size(basis, 1);
    space.k = size(basis, 2);
    space.basis = basis;
end


function part = local_build_block3_partition(grid)
    req = {'nx', 'ny', 'x', 'y', 'npml', 'center', 'r_near'};
    for i = 1:numel(req)
        if ~isfield(grid, req{i})
            error('build_block3_partition missing field: %s', req{i});
        end
    end

    nx = grid.nx;
    ny = grid.ny;

    x_int = grid.x;
    y_int = grid.y;
    if numel(x_int) == nx + 2
        x_int = x_int(2:end-1);
    end
    if numel(y_int) == ny + 2
        y_int = y_int(2:end-1);
    end
    if numel(x_int) ~= nx || numel(y_int) ~= ny
        error('x/y lengths are inconsistent with nx/ny.');
    end

    [Xi, Yi] = ndgrid(x_int, y_int);
    R = hypot(Xi - grid.center(1), Yi - grid.center(2));

    ii = repmat((1:nx).', 1, ny);
    jj = repmat(1:ny, nx, 1);
    pml_mask = (ii <= grid.npml) | (ii > nx - grid.npml) | (jj <= grid.npml) | (jj > ny - grid.npml);
    phys_mask = ~pml_mask;

    if isfield(grid, 'active_mask') && ~isempty(grid.active_mask)
        active_mask = logical(reshape(grid.active_mask, nx, ny));
    else
        active_mask = true(nx, ny);
    end

    near_mask = phys_mask & (R <= grid.r_near) & active_mask;
    far_mask = phys_mask & ~near_mask & active_mask;
    pml_mask = pml_mask & active_mask;

    covered = near_mask | far_mask | pml_mask;
    missing = active_mask & ~covered;
    if any(missing(:))
        far_mask = far_mask | missing;
    end

    part = struct();
    part.n = nx * ny;
    part.nx = nx;
    part.ny = ny;
    part.idx_near = find(near_mask(:));
    part.idx_far = find(far_mask(:));
    part.idx_pml = find(pml_mask(:));
    part.count_near = numel(part.idx_near);
    part.count_far = numel(part.idx_far);
    part.count_pml = numel(part.idx_pml);
end


function space = local_build_block3_space(snapshot_matrix, partition, cfg)
    if isfield(partition, 'blocks') && ~isempty(partition.blocks)
        space = local_build_blockn_space(snapshot_matrix, partition, cfg);
        return;
    end

    if nargin < 3 || isempty(cfg)
        cfg = struct();
    end
    if ~isfield(cfg, 'rank_rtol') || isempty(cfg.rank_rtol)
        cfg.rank_rtol = 1e-13;
    end

    n = size(snapshot_matrix, 1);
    if n ~= partition.n
        error('Snapshot row dimension (%d) does not match partition size (%d).', n, partition.n);
    end

    trunc_near = local_block_trunc_cfg(cfg, 'near');
    trunc_far = local_block_trunc_cfg(cfg, 'far');
    trunc_pml = local_block_trunc_cfg(cfg, 'pml');

    if local_has_global_tail_budget(cfg)
        [Un_all, sn] = local_block_svd(snapshot_matrix(partition.idx_near, :));
        [Uf_all, sf] = local_block_svd(snapshot_matrix(partition.idx_far, :));
        [Up_all, sp] = local_block_svd(snapshot_matrix(partition.idx_pml, :));
        svals = {sn, sf, sp};
        nrows = [numel(partition.idx_near), numel(partition.idx_far), numel(partition.idx_pml)];
        truncs = {trunc_near, trunc_far, trunc_pml};
        [ranks, global_meta] = local_select_global_tail_ranks(svals, nrows, truncs, cfg);
        [Un, meta_near] = local_finalize_block_basis(Un_all, sn, trunc_near, ranks(1), global_meta, 1);
        [Uf, meta_far] = local_finalize_block_basis(Uf_all, sf, trunc_far, ranks(2), global_meta, 2);
        [Up, meta_pml] = local_finalize_block_basis(Up_all, sp, trunc_pml, ranks(3), global_meta, 3);
    else
        [Un, meta_near] = local_block_basis(snapshot_matrix(partition.idx_near, :), trunc_near);
        [Uf, meta_far] = local_block_basis(snapshot_matrix(partition.idx_far, :), trunc_far);
        [Up, meta_pml] = local_block_basis(snapshot_matrix(partition.idx_pml, :), trunc_pml);
    end

    space = struct();
    space.type = 'block3';
    space.n = n;
    space.partition = partition;
    space.Un = Un;
    space.Uf = Uf;
    space.Up = Up;
    space.kn = size(Un, 2);
    space.kf = size(Uf, 2);
    space.kp = size(Up, 2);
    space.k = space.kn + space.kf + space.kp;
    space.block_meta = struct();
    space.block_meta.near = meta_near;
    space.block_meta.far = meta_far;
    space.block_meta.pml = meta_pml;
end


function space = local_build_blockn_space(snapshot_matrix, partition, cfg)
    if nargin < 3 || isempty(cfg)
        cfg = struct();
    end
    if ~isfield(partition, 'blocks') || isempty(partition.blocks)
        error('blockn partition must contain a nonempty blocks array.');
    end
    if ~isfield(cfg, 'rank_rtol') || isempty(cfg.rank_rtol)
        cfg.rank_rtol = 1e-13;
    end

    n = size(snapshot_matrix, 1);
    if n ~= partition.n
        error('Snapshot row dimension (%d) does not match partition size (%d).', n, partition.n);
    end

    block_count = numel(partition.blocks);
    block_names = strings(1, block_count);
    truncs = cell(1, block_count);
    for b = 1:block_count
        if ~isfield(partition.blocks(b), 'idx')
            error('blockn partition block %d is missing idx.', b);
        end
        if isfield(partition.blocks(b), 'name') && ~isempty(partition.blocks(b).name)
            block_names(b) = string(partition.blocks(b).name);
        else
            block_names(b) = sprintf('block%d', b);
        end
        truncs{b} = local_blockn_trunc_cfg(cfg, char(block_names(b)), b);
    end

    Ucell = cell(1, block_count);
    metacell = cell(1, block_count);
    if local_has_global_tail_budget(cfg)
        Uall = cell(1, block_count);
        svals = cell(1, block_count);
        nrows = zeros(1, block_count);
        for b = 1:block_count
            idx = partition.blocks(b).idx;
            [Uall{b}, svals{b}] = local_block_svd(snapshot_matrix(idx, :));
            nrows(b) = numel(idx);
        end
        [ranks, global_meta] = local_select_global_tail_ranks(svals, nrows, truncs, cfg);
        for b = 1:block_count
            [Ucell{b}, metacell{b}] = local_finalize_block_basis( ...
                Uall{b}, svals{b}, truncs{b}, ranks(b), global_meta, b);
        end
    else
        for b = 1:block_count
            idx = partition.blocks(b).idx;
            [Ucell{b}, metacell{b}] = local_block_basis(snapshot_matrix(idx, :), truncs{b});
        end
    end

    blocks = partition.blocks;
    total_dim = 0;
    block_meta = struct();
    for b = 1:block_count
        blocks(b).basis = Ucell{b};
        blocks(b).k = size(Ucell{b}, 2);
        blocks(b).meta = metacell{b};
        blocks(b).label = char(block_names(b));
        total_dim = total_dim + blocks(b).k;
        field_name = matlab.lang.makeValidName(char(block_names(b)));
        block_meta.(field_name) = metacell{b};
    end

    space = struct();
    space.type = 'blockn';
    space.n = n;
    space.partition = partition;
    space.blocks = blocks;
    space.block_names = cellstr(block_names);
    space.k = total_dim;
    space.block_meta = block_meta;
end


function tf = local_has_global_tail_budget(cfg)
    tf = isfield(cfg, 'rank_global_tail_tol') && ~isempty(cfg.rank_global_tail_tol);
    if tf && ~(isscalar(cfg.rank_global_tail_tol) && isfinite(cfg.rank_global_tail_tol) && cfg.rank_global_tail_tol > 0)
        error('rank_global_tail_tol must be empty or a positive finite scalar.');
    end
end


function trunc = local_block_trunc_cfg(cfg, block_name)
    if nargin < 2 || isempty(block_name)
        block_name = 'near';
    end

    trunc = struct();
    trunc.rank_rtol = 1e-13;
    trunc.rank_tail_tol = [];
    trunc.rank_cap = inf;

    if isfield(cfg, 'rank_rtol') && ~isempty(cfg.rank_rtol)
        trunc.rank_rtol = cfg.rank_rtol;
    end
    rtol_field = ['rank_rtol_' block_name];
    if isfield(cfg, rtol_field) && ~isempty(cfg.(rtol_field))
        trunc.rank_rtol = cfg.(rtol_field);
    end

    if isfield(cfg, 'rank_tail_tol') && ~isempty(cfg.rank_tail_tol)
        trunc.rank_tail_tol = cfg.rank_tail_tol;
    end
    tail_field = ['rank_tail_tol_' block_name];
    if isfield(cfg, tail_field) && ~isempty(cfg.(tail_field))
        trunc.rank_tail_tol = cfg.(tail_field);
    end

    if isfield(cfg, 'rank_cap') && ~isempty(cfg.rank_cap)
        trunc.rank_cap = cfg.rank_cap;
    end
    cap_field = ['rank_cap_' block_name];
    if isfield(cfg, cap_field) && ~isempty(cfg.(cap_field))
        trunc.rank_cap = cfg.(cap_field);
    end

    if ~(isfinite(trunc.rank_rtol) && trunc.rank_rtol > 0)
        error('rank_rtol for block %s must be positive finite.', block_name);
    end
    if ~isempty(trunc.rank_tail_tol) && ...
            ~(isfinite(trunc.rank_tail_tol) && trunc.rank_tail_tol > 0)
        error('rank_tail_tol for block %s must be empty or positive finite.', block_name);
    end
    if ~(isinf(trunc.rank_cap) || (isfinite(trunc.rank_cap) && trunc.rank_cap >= 0))
        error('rank_cap for block %s must be nonnegative finite or inf.', block_name);
    end
end


function trunc = local_blockn_trunc_cfg(cfg, block_name, block_index)
    trunc = local_block_trunc_cfg(cfg, block_name);

    if isfield(cfg, 'rank_rtols') && ~isempty(cfg.rank_rtols)
        if numel(cfg.rank_rtols) < block_index
            error('rank_rtols does not contain block %d.', block_index);
        end
        trunc.rank_rtol = cfg.rank_rtols(block_index);
    end
    if isfield(cfg, 'rank_tail_tols') && ~isempty(cfg.rank_tail_tols)
        if numel(cfg.rank_tail_tols) < block_index
            error('rank_tail_tols does not contain block %d.', block_index);
        end
        trunc.rank_tail_tol = cfg.rank_tail_tols(block_index);
    end
    if isfield(cfg, 'rank_caps') && ~isempty(cfg.rank_caps)
        if numel(cfg.rank_caps) < block_index
            error('rank_caps does not contain block %d.', block_index);
        end
        trunc.rank_cap = cfg.rank_caps(block_index);
    end

    if ~(isfinite(trunc.rank_rtol) && trunc.rank_rtol > 0)
        error('rank_rtol for block %s must be positive finite.', block_name);
    end
    if ~isempty(trunc.rank_tail_tol) && ...
            ~(isfinite(trunc.rank_tail_tol) && trunc.rank_tail_tol > 0)
        error('rank_tail_tol for block %s must be empty or positive finite.', block_name);
    end
    if ~(isinf(trunc.rank_cap) || (isfinite(trunc.rank_cap) && trunc.rank_cap >= 0))
        error('rank_cap for block %s must be nonnegative finite or inf.', block_name);
    end
end

function [Uuse, meta] = local_block_basis(S, trunc)
    nrow = size(S, 1);
    meta = struct();
    meta.nrow = nrow;
    meta.rank = 0;
    meta.rank_numeric = 0;
    meta.rank_cap = trunc.rank_cap;
    meta.rank_rtol = trunc.rank_rtol;
    meta.rank_tail_tol = trunc.rank_tail_tol;
    meta.rank_rule = 'relative_singular_value';
    meta.singular_values = zeros(0, 1);
    if nrow == 0 || isempty(S)
        Uuse = zeros(nrow, 0);
        return;
    end

    [U, Sigma, ~] = svd(S, 'econ');
    s = diag(Sigma);
    if isempty(trunc.rank_tail_tol)
        r_numeric = local_numeric_rank_from_rtol(s, trunc.rank_rtol);
    else
        r_numeric = local_numeric_rank_from_tail_tol(s, trunc.rank_tail_tol);
        meta.rank_rule = 'relative_frobenius_tail';
    end
    r_use = r_numeric;
    if isfinite(trunc.rank_cap)
        r_use = min(r_use, floor(trunc.rank_cap));
    end
    meta.rank_numeric = r_numeric;
    meta.rank = r_use;
    meta.singular_values = s(:);
    if r_use == 0
        Uuse = zeros(nrow, 0);
    else
        Uuse = U(:, 1:r_use);
    end
end


function [U, s] = local_block_svd(S)
    nrow = size(S, 1);
    if nrow == 0 || isempty(S)
        U = zeros(nrow, 0);
        s = zeros(0, 1);
        return;
    end
    [U, Sigma, ~] = svd(S, 'econ');
    s = diag(Sigma);
end


function [Uuse, meta] = local_finalize_block_basis(U, s, trunc, r_use, global_meta, block_index)
    nrow = size(U, 1);
    meta = struct();
    meta.nrow = nrow;
    meta.rank = r_use;
    meta.rank_numeric = r_use;
    meta.rank_cap = trunc.rank_cap;
    meta.rank_rtol = trunc.rank_rtol;
    meta.rank_tail_tol = trunc.rank_tail_tol;
    meta.rank_rule = global_meta.strategy;
    meta.singular_values = s(:);
    meta.rank_global_tail_tol = global_meta.tol;
    meta.global_energy_weight = global_meta.weights(block_index);
    meta.local_relative_tail = global_meta.local_relative_tail(block_index);
    meta.global_relative_tail_contribution = global_meta.global_relative_tail_contribution(block_index);
    meta.global_relative_tail = global_meta.global_relative_tail;
    meta.generated_rank_tail_tol = global_meta.generated_rank_tail_tol(block_index);
    if r_use == 0
        Uuse = zeros(nrow, 0);
    else
        Uuse = U(:, 1:r_use);
    end
end


function [ranks, meta] = local_select_global_tail_ranks(svals, nrows, truncs, cfg)
    strategy = 'dof_weighted_global_frobenius_tail';
    if isfield(cfg, 'rank_global_tail_strategy') && ~isempty(cfg.rank_global_tail_strategy)
        strategy = local_normalize_global_tail_strategy(cfg.rank_global_tail_strategy);
    end

    block_count = numel(svals);
    total_energy = 0;
    block_energy = zeros(1, block_count);
    tail_abs = cell(1, block_count);
    rel_tail = cell(1, block_count);
    max_ranks = zeros(1, block_count);
    for b = 1:block_count
        s = abs(svals{b}(:));
        block_energy(b) = sum(s.^2);
        total_energy = total_energy + block_energy(b);
        max_ranks(b) = numel(s);
        if isfinite(truncs{b}.rank_cap)
            max_ranks(b) = min(max_ranks(b), floor(truncs{b}.rank_cap));
        end
        tail_abs{b} = zeros(numel(s) + 1, 1);
        rel_tail{b} = zeros(numel(s) + 1, 1);
        denom = max(sqrt(block_energy(b)), 1e-30);
        for r = 0:numel(s)
            tail_abs{b}(r + 1) = sqrt(sum(s(r + 1:end).^2));
            rel_tail{b}(r + 1) = tail_abs{b}(r + 1) / denom;
        end
    end

    if total_energy <= 1e-30
        ranks = zeros(1, block_count);
        meta = local_global_tail_meta(cfg.rank_global_tail_tol, strategy, block_energy, total_energy, ...
            ranks, rel_tail, tail_abs, nan(1, block_count));
        return;
    end

    switch strategy
        case 'equal_absolute_global_frobenius_tail'
            ranks = local_select_equal_absolute_ranks(cfg.rank_global_tail_tol, block_energy, rel_tail, max_ranks, truncs);
            generated_tol = cfg.rank_global_tail_tol ./ sqrt(block_count * max(block_energy / total_energy, 1e-30));
            for b = 1:block_count
                if ~isempty(truncs{b}.rank_tail_tol)
                    generated_tol(b) = min(generated_tol(b), truncs{b}.rank_tail_tol);
                end
            end
        case 'dof_weighted_global_frobenius_tail'
            [ranks, generated_tol] = local_select_dof_weighted_ranks( ...
                cfg.rank_global_tail_tol, block_energy, nrows, rel_tail, ...
                max_ranks, truncs);
        case 'minimum_storage_global_frobenius_tail'
            [ranks, generated_tol] = local_select_minimum_storage_ranks( ...
                cfg.rank_global_tail_tol, total_energy, tail_abs, rel_tail, ...
                nrows, max_ranks, truncs);
        case 'marginal_storage_global_frobenius_tail'
            [ranks, generated_tol] = local_select_marginal_storage_ranks( ...
                cfg.rank_global_tail_tol, total_energy, svals, rel_tail, ...
                nrows, max_ranks, truncs);
        otherwise
            error('Unknown global tail strategy: %s.', strategy);
    end

    meta = local_global_tail_meta(cfg.rank_global_tail_tol, strategy, block_energy, total_energy, ...
        ranks, rel_tail, tail_abs, generated_tol);
end


function strategy = local_normalize_global_tail_strategy(strategy)
    strategy = lower(strtrim(char(strategy)));
    switch strategy
        case {'equal_absolute', 'equal_abs', 'equal_budget', 'equal_absolute_global_frobenius_tail'}
            strategy = 'equal_absolute_global_frobenius_tail';
        case {'dof_weighted', 'equal_per_dof', 'spatially_uniform', ...
                'dof_weighted_global_frobenius_tail'}
            strategy = 'dof_weighted_global_frobenius_tail';
        case {'minimum_storage', 'min_storage', 'storage_optimal', ...
                'minimum_storage_global_frobenius_tail'}
            strategy = 'minimum_storage_global_frobenius_tail';
        case {'marginal_storage', 'storage_greedy', 'greedy_storage', ...
                'marginal_storage_global_frobenius_tail'}
            strategy = 'marginal_storage_global_frobenius_tail';
        otherwise
            error(['rank_global_tail_strategy=%s must be equal_absolute, ' ...
                'dof_weighted, marginal_storage, or minimum_storage.'], strategy);
    end
end


function ranks = local_select_equal_absolute_ranks(global_tol, block_energy, rel_tail, max_ranks, truncs)
    block_count = numel(block_energy);
    weights = block_energy / max(sum(block_energy), 1e-30);
    ranks = zeros(1, block_count);
    for b = 1:block_count
        local_tol = global_tol / sqrt(block_count * max(weights(b), 1e-30));
        if ~isempty(truncs{b}.rank_tail_tol)
            local_tol = min(local_tol, truncs{b}.rank_tail_tol);
        end
        candidates = find(rel_tail{b} <= local_tol) - 1;
        candidates = candidates(candidates <= max_ranks(b));
        if isempty(candidates)
            error('No feasible rank for block %d under equal absolute global tail budget.', b);
        end
        ranks(b) = candidates(1);
    end
end


function [ranks, generated_tol] = local_select_dof_weighted_ranks( ...
        global_tol, block_energy, nrows, rel_tail, max_ranks, truncs)
    % Allocate squared tail energy in proportion to block size.  This keeps
    % the assembled Frobenius-tail bound equal to the requested global one.
    block_count = numel(block_energy);
    energy_weights = block_energy / max(sum(block_energy), 1e-30);
    dof_weights = nrows / max(sum(nrows), 1);
    generated_tol = global_tol * sqrt(dof_weights ./ max(energy_weights, 1e-30));
    ranks = zeros(1, block_count);
    for b = 1:block_count
        local_tol = generated_tol(b);
        if ~isempty(truncs{b}.rank_tail_tol)
            local_tol = min(local_tol, truncs{b}.rank_tail_tol);
            generated_tol(b) = local_tol;
        end
        candidates = find(rel_tail{b} <= local_tol * (1 + 1e-12)) - 1;
        candidates = candidates(candidates <= max_ranks(b));
        if isempty(candidates)
            error('No feasible rank for block %d under the DOF-weighted global tail budget.', b);
        end
        ranks(b) = candidates(1);
    end
end


function [ranks, effective_local_tols] = local_select_minimum_storage_ranks( ...
        global_tol, total_energy, tail_abs, rel_tail, nrows, max_ranks, truncs)
    % Research-only exact diagnostic.  This is a multiple-choice discrete
    % optimization; its Pareto frontier can grow exponentially with blocks.
    block_count = numel(tail_abs);
    target_tail2 = (global_tol ^ 2) * total_energy;

    state_cost = 0;
    state_tail2 = 0;
    state_ranks = zeros(1, 0);
    for b = 1:block_count
        rank_candidates = (0:max_ranks(b)).';
        if ~isempty(truncs{b}.rank_tail_tol)
            feasible_local = rel_tail{b}(rank_candidates + 1) <= ...
                truncs{b}.rank_tail_tol * (1 + 1e-12);
            rank_candidates = rank_candidates(feasible_local);
        end
        if isempty(rank_candidates)
            error('No feasible rank candidates for block %d under its local tail cap.', b);
        end

        candidate_cost = nrows(b) * rank_candidates;
        candidate_tail2 = tail_abs{b}(rank_candidates + 1) .^ 2;
        old_count = numel(state_cost);
        candidate_count = numel(rank_candidates);
        new_cost = repelem(state_cost, candidate_count, 1) + ...
            repmat(candidate_cost, old_count, 1);
        new_tail2 = repelem(state_tail2, candidate_count, 1) + ...
            repmat(candidate_tail2, old_count, 1);
        new_ranks = [repelem(state_ranks, candidate_count, 1), ...
            repmat(rank_candidates, old_count, 1)];

        [~, order] = sortrows([new_cost, new_tail2], [1, 2]);
        new_cost = new_cost(order);
        new_tail2 = new_tail2(order);
        new_ranks = new_ranks(order, :);

        keep = false(size(new_cost));
        best_tail2 = inf;
        for k = 1:numel(new_cost)
            if new_tail2(k) < best_tail2 * (1 - 1e-14)
                keep(k) = true;
                best_tail2 = new_tail2(k);
            end
        end
        state_cost = new_cost(keep);
        state_tail2 = new_tail2(keep);
        state_ranks = new_ranks(keep, :);
    end

    feasible = find(state_tail2 <= target_tail2 * (1 + 1e-12));
    if isempty(feasible)
        error('No feasible block ranks under the minimum-storage global tail budget.');
    end
    feasible_cost = state_cost(feasible);
    min_cost = min(feasible_cost);
    tied = feasible(feasible_cost == min_cost);
    [~, tie_index] = min(state_tail2(tied));
    best_index = tied(tie_index);
    ranks = state_ranks(best_index, :);

    effective_local_tols = zeros(1, block_count);
    for b = 1:block_count
        effective_local_tols(b) = rel_tail{b}(ranks(b) + 1);
    end
end


function [ranks, effective_local_tols] = local_select_marginal_storage_ranks( ...
        global_tol, total_energy, svals, rel_tail, nrows, max_ranks, truncs)
    block_count = numel(svals);
    target_tail2 = (global_tol ^ 2) * total_energy;
    ranks = zeros(1, block_count);

    for b = 1:block_count
        if ~isempty(truncs{b}.rank_tail_tol)
            candidates = find(rel_tail{b}(1:max_ranks(b) + 1) <= ...
                truncs{b}.rank_tail_tol * (1 + 1e-12)) - 1;
            if isempty(candidates)
                error('No feasible rank for block %d under its local tail cap.', b);
            end
            ranks(b) = candidates(1);
        end
    end

    % Sum the remaining tails directly. Subtracting retained leading energy
    % from the total loses small tails when the requested budget is tight.
    suffix_energy = cell(1,block_count);
    remaining_energy = zeros(1,block_count);
    for b = 1:block_count
        suffix_energy{b} = [flipud(cumsum(flipud(abs(svals{b}(:)).^2)));0];
        remaining_energy(b) = suffix_energy{b}(ranks(b)+1);
    end
    tail2 = sum(remaining_energy);
    while tail2 > target_tail2 * (1 + 1e-12)
        score = -inf(1, block_count);
        for b = 1:block_count
            if ranks(b) < max_ranks(b)
                next_sigma2 = abs(svals{b}(ranks(b) + 1)) ^ 2;
                score(b) = next_sigma2 / max(nrows(b), 1);
            end
        end
        [best_score, best_block] = max(score);
        if ~isfinite(best_score)
            error('No feasible block ranks under the marginal-storage global tail budget.');
        end
        ranks(best_block) = ranks(best_block) + 1;
        remaining_energy(best_block) = suffix_energy{best_block}(ranks(best_block)+1);
        tail2 = sum(remaining_energy);
    end

    effective_local_tols = zeros(1, block_count);
    for b = 1:block_count
        effective_local_tols(b) = rel_tail{b}(ranks(b) + 1);
    end
end


function meta = local_global_tail_meta(global_tol, strategy, block_energy, total_energy, ranks, rel_tail, tail_abs, generated_tol)
    weights = block_energy / max(total_energy, 1e-30);
    local_tail = zeros(1, numel(ranks));
    tail2 = 0;
    for b = 1:numel(ranks)
        local_tail(b) = rel_tail{b}(ranks(b) + 1);
        tail2 = tail2 + tail_abs{b}(ranks(b) + 1)^2;
    end
    meta = struct();
    meta.tol = global_tol;
    meta.strategy = strategy;
    meta.weights = weights;
    meta.local_relative_tail = local_tail;
    meta.global_relative_tail_contribution = sqrt(weights) .* local_tail;
    meta.global_relative_tail = sqrt(tail2 / max(total_energy, 1e-30));
    meta.generated_rank_tail_tol = generated_tol;
end


function r = local_numeric_rank_from_rtol(s, rank_rtol)
    if isempty(s)
        r = 0;
        return;
    end
    denom = max(abs(s(1)), 1e-30);
    srel = abs(s) / denom;
    r = nnz(srel >= rank_rtol);
end


function r = local_numeric_rank_from_tail_tol(s, rank_tail_tol)
    if isempty(s)
        r = 0;
        return;
    end
    total = norm(s);
    if total <= 1e-30
        r = 0;
        return;
    end
    tail_energy = flipud(cumsum(flipud(abs(s(:)).^2)));
    tail_after = [tail_energy(2:end); 0];
    rel_tail = sqrt(tail_after) / total;
    r = find(rel_tail <= rank_tail_tol, 1, 'first');
    if isempty(r)
        r = numel(s);
    end
end


function k = local_dimension(space)
    if isempty(space)
        k = 0;
        return;
    end
    if isstruct(space) && isfield(space, 'k')
        k = space.k;
        return;
    end
    if isnumeric(space)
        k = size(space, 2);
        return;
    end
    error('Unsupported space type in dimension().');
end


function coeff = local_project(space, vec, shared_indices)
    if nargin < 3, shared_indices = {}; end
    if isnumeric(space)
        coeff = space' * vec;
        return;
    end

    if strcmp(space.type, 'dense')
        coeff = space.basis' * vec;
        return;
    end

    if strcmp(space.type, 'block3')
        part = space.partition;
        coeff = zeros(space.k, 1);
        cur = 0;
        if space.kn > 0
            coeff(cur + (1:space.kn)) = space.Un' * vec(part.idx_near);
            cur = cur + space.kn;
        end
        if space.kf > 0
            coeff(cur + (1:space.kf)) = space.Uf' * vec(part.idx_far);
            cur = cur + space.kf;
        end
        if space.kp > 0
            coeff(cur + (1:space.kp)) = space.Up' * vec(part.idx_pml);
        end
        return;
    end

    if strcmp(space.type, 'blockn')
        coeff = zeros(space.k, 1);
        cur = 0;
        for b = 1:numel(space.blocks)
            kb = space.blocks(b).k;
            if kb > 0
                if isempty(shared_indices), idx = space.blocks(b).idx; else, idx = shared_indices{b}; end
                coeff(cur + (1:kb)) = space.blocks(b).basis' * vec(idx);
                cur = cur + kb;
            end
        end
        return;
    end

    error('Unsupported space type in project().');
end


function vec = local_lift(space, coeff, shared_indices)
    if nargin < 3, shared_indices = {}; end
    if isnumeric(space)
        vec = space * coeff;
        return;
    end

    if strcmp(space.type, 'dense')
        vec = space.basis * coeff;
        return;
    end

    if strcmp(space.type, 'block3')
        part = space.partition;
        % Avoid reallocating a full real vector on the first complex block.
        vec = zeros(space.n, 1, 'like', coeff);
        cur = 0;

        if space.kn > 0
            a = coeff(cur + (1:space.kn));
            vec(part.idx_near) = space.Un * a;
            cur = cur + space.kn;
        end
        if space.kf > 0
            a = coeff(cur + (1:space.kf));
            vec(part.idx_far) = space.Uf * a;
            cur = cur + space.kf;
        end
        if space.kp > 0
            a = coeff(cur + (1:space.kp));
            vec(part.idx_pml) = space.Up * a;
        end
        return;
    end

    if strcmp(space.type, 'blockn')
        % Avoid reallocating a full real vector on the first complex block.
        vec = zeros(space.n, 1, 'like', coeff);
        cur = 0;
        for b = 1:numel(space.blocks)
            kb = space.blocks(b).k;
            if kb > 0
                a = coeff(cur + (1:kb));
                if isempty(shared_indices), idx = space.blocks(b).idx; else, idx = shared_indices{b}; end
                vec(idx) = space.blocks(b).basis * a;
                cur = cur + kb;
            end
        end
        return;
    end

    error('Unsupported space type in lift().');
end


function Ar = local_assemble_reduced_matrix(Aop, space)
    if isnumeric(space)
        if isa(Aop, 'function_handle')
            r = size(space, 2);
            AV = zeros(size(space, 1), r);
            for j = 1:r
                AV(:, j) = Aop(space(:, j));
            end
            Ar = space' * AV;
        else
            Ar = space' * (Aop * space);
        end
        return;
    end

    if strcmp(space.type, 'dense')
        B = space.basis;
        if isa(Aop, 'function_handle')
            r = size(B, 2);
            AV = zeros(size(B, 1), r);
            for j = 1:r
                AV(:, j) = Aop(B(:, j));
            end
            Ar = B' * AV;
        else
            Ar = B' * (Aop * B);
        end
        return;
    end

    if strcmp(space.type, 'block3') || strcmp(space.type, 'blockn')
        k = space.k;
        Ar = zeros(k, k);
        if isnumeric(Aop)
            % Contract the supported blocks directly. This is the same
            % Galerkin matrix as lifting every column to the full grid,
            % but batches sparse products and avoids full-grid zero vectors.
            if strcmp(space.type, 'block3')
                indices = {space.partition.idx_near,space.partition.idx_far,space.partition.idx_pml};
                bases = {space.Un,space.Uf,space.Up};
            else
                indices = {space.blocks.idx};
                bases = {space.blocks.basis};
            end
            offsets = [0,cumsum(cellfun(@(B)size(B,2),bases))];
            for i = 1:numel(bases)
                ri = offsets(i)+(1:size(bases{i},2));
                if isempty(ri), continue; end
                for j = 1:numel(bases)
                    rj = offsets(j)+(1:size(bases{j},2));
                    if isempty(rj), continue; end
                    Aij = Aop(indices{i},indices{j});
                    if nnz(Aij) == 0, continue; end
                    Ar(ri,rj) = bases{i}' * (Aij * bases{j});
                end
            end
            return;
        end
        for j = 1:k
            ej = zeros(k, 1);
            ej(j) = 1;
            uj = local_lift(space, ej);
            wj = local_apply_operator(Aop, uj);
            Ar(:, j) = local_project(space, wj);
        end
        return;
    end

    error('Unsupported space type in assemble_reduced_matrix().');
end


function y = local_apply_operator(Aop, x)
    if isa(Aop, 'function_handle')
        y = Aop(x);
    else
        y = Aop * x;
    end
end
