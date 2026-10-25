function [Msolve, info] = mg(A, grid_size, opts)
%MG Geometric multigrid V-cycle for the shifted-Laplacian operator on a 3D grid.
%
% Coarse operators are formed by Galerkin projection so the hierarchy
% preserves the heterogeneous mass term and complex PML coefficients.

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    opts = set_default(opts, 'mg_max_levels', 6);
    opts = set_default(opts, 'mg_coarse_max_dofs', 12000);
    opts = set_default(opts, 'mg_pre_sweeps', 2);
    opts = set_default(opts, 'mg_post_sweeps', 2);
    opts = set_default(opts, 'mg_jacobi_weight', 0.6);
    opts = set_default(opts, 'mg_cycles_per_apply', 1);

    grid_size = round(grid_size(:).');
    if numel(grid_size) ~= 3 || any(grid_size < 1) || prod(grid_size) ~= size(A, 1)
        error('grid_size must be a positive three-vector whose product matches A.');
    end
    if size(A, 1) ~= size(A, 2)
        error('A must be square.');
    end

    setup_timer = tic;
    levels = cell(opts.mg_max_levels, 1);
    levels{1} = make_level(A, grid_size);
    level_count = 1;

    while level_count < opts.mg_max_levels
        fine = levels{level_count};
        if size(fine.A, 1) <= opts.mg_coarse_max_dofs || any(fine.grid_size <= 3)
            break;
        end

        coarse_size = max(1, floor((fine.grid_size - 1) / 2));
        if any(coarse_size >= fine.grid_size)
            break;
        end
        Px = linear_interpolation_1d(fine.grid_size(1), coarse_size(1));
        Py = linear_interpolation_1d(fine.grid_size(2), coarse_size(2));
        Pz = linear_interpolation_1d(fine.grid_size(3), coarse_size(3));
        P = kron(Pz, kron(Py, Px));
        levels{level_count}.P = P;

        coarse_A = P' * (fine.A * P);
        level_count = level_count + 1;
        levels{level_count} = make_level(coarse_A, coarse_size);
    end
    levels = levels(1:level_count);
    levels{end}.coarse_factor = decomposition(levels{end}.A, 'lu');

    hierarchy_bytes = estimate_hierarchy_bytes(levels);
    setup_time = toc(setup_timer);
    Msolve = @(rhs) apply_cycles(levels, rhs, opts);
    info = struct('setup_time', setup_time, 'factor_bytes', hierarchy_bytes, ...
        'level_count', level_count, 'grid_sizes', {cellfun(@(v) v.grid_size, ...
        levels, 'UniformOutput', false)}, 'options', opts);
    if nargout > 1
        % For diagnostics of single cycle components (Appendix A).
        info.levels = levels;
        info.vcycle = @(level_index, rhs) vcycle(levels, level_index, rhs, opts);
    end
end

function level = make_level(A, grid_size)
    diagonal = diag(A);
    diagonal(abs(diagonal) < eps) = 1;
    level = struct('A', A, 'grid_size', grid_size, ...
        'diagonal_inverse', 1 ./ diagonal, 'P', [], 'coarse_factor', []);
end

function P = linear_interpolation_1d(nfine, ncoarse)
    fine_coordinate = (1:nfine).' / (nfine + 1);
    coarse_index = fine_coordinate * (ncoarse + 1);
    left = floor(coarse_index);
    fraction = coarse_index - left;

    rows = zeros(2 * nfine, 1);
    cols = zeros(2 * nfine, 1);
    values = zeros(2 * nfine, 1);
    count = 0;
    for j = 1:nfine
        if left(j) >= 1
            count = count + 1;
            rows(count) = j;
            cols(count) = left(j);
            values(count) = 1 - fraction(j);
        end
        right = left(j) + 1;
        if right <= ncoarse
            count = count + 1;
            rows(count) = j;
            cols(count) = right;
            values(count) = fraction(j);
        end
    end
    P = sparse(rows(1:count), cols(1:count), values(1:count), nfine, ncoarse);
end

function x = apply_cycles(levels, rhs, opts)
    x = zeros(size(rhs), 'like', rhs);
    for cycle = 1:opts.mg_cycles_per_apply
        residual = rhs - levels{1}.A * x;
        x = x + vcycle(levels, 1, residual, opts);
    end
end

function x = vcycle(levels, level_index, rhs, opts)
    level = levels{level_index};
    if level_index == numel(levels)
        x = level.coarse_factor \ rhs;
        return;
    end

    x = zeros(size(rhs), 'like', rhs);
    for sweep = 1:opts.mg_pre_sweeps
        x = x + opts.mg_jacobi_weight * level.diagonal_inverse .* (rhs - level.A * x);
    end
    residual = rhs - level.A * x;
    coarse_correction = vcycle(levels, level_index + 1, level.P' * residual, opts);
    x = x + level.P * coarse_correction;
    for sweep = 1:opts.mg_post_sweeps
        x = x + opts.mg_jacobi_weight * level.diagonal_inverse .* (rhs - level.A * x);
    end
end

function bytes = estimate_hierarchy_bytes(levels)
    bytes = 0;
    for k = 1:numel(levels)
        level = levels{k};
        bytes = bytes + sparse_matrix_bytes(level.A) + 16 * numel(level.diagonal_inverse);
        if ~isempty(level.P)
            bytes = bytes + sparse_matrix_bytes(level.P);
        end
    end
    factor_info = whos('levels');
    bytes = max(bytes, factor_info.bytes);
end

function bytes = sparse_matrix_bytes(A)
    value_bytes = 8 * (1 + ~isreal(A));
    bytes = nnz(A) * (value_bytes + 8) + 8 * (size(A, 2) + 1);
end

function s = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end
