function [Msolve, probe, info] = build_dd_mechanism_mg_probe(A, grid_size, opts)
%BUILD_DD_MECHANISM_MG_PROBE Original V-cycle with component diagnostics.
%
% Coarse operators are formed by Galerkin projection so the hierarchy
% preserves the heterogeneous mass term and complex PML coefficients.
% The hierarchy and Msolve arithmetic match
% build_shifted_laplacian_multigrid_preconditioner. probe(r,Afun) examines ONE
% fine-grid cycle, applying Afun (the unshifted Helmholtz operator) to its
% pre-smoothing, prolonged coarse, and post-smoothing corrections. Only
% scalar/small-array diagnostics are returned; no full fields are retained.
% Optimal contractions are geometric span diagnostics, not actual GMRES
% convergence estimates. A large isolated component can cancel another one.

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    opts = set_default(opts, 'mg_max_levels', 6);
    opts = set_default(opts, 'mg_coarse_max_dofs', 4000);
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
    probe = @(rhs, Afun) probe_cycle(levels, rhs, Afun, opts);
    info = struct('setup_time', setup_time, 'factor_bytes', hierarchy_bytes, ...
        'level_count', level_count, 'grid_sizes', {cellfun(@(v) v.grid_size, ...
        levels, 'UniformOutput', false)}, 'options', opts);
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

function stats = probe_cycle(levels, rhs, Afun, opts)
    if opts.mg_cycles_per_apply ~= 1
        error('dd_mechanism:singleCycleRequired', ...
            'Component probing requires mg_cycles_per_apply=1.');
    end
    if ~isvector(rhs) || size(rhs, 2) ~= 1 || numel(rhs) ~= size(levels{1}.A, 1)
        error('dd_mechanism:rhsShape', 'Probe rhs must be a single column vector.');
    end
    if ~isa(Afun, 'function_handle')
        error('dd_mechanism:operatorType', 'Afun must apply the true Helmholtz operator.');
    end
    rhs_norm = norm(rhs);
    if rhs_norm == 0
        error('dd_mechanism:zeroResidual', 'Normalized diagnostics need a nonzero residual.');
    end

    level = levels{1};
    if numel(levels) == 1
        x_pre = zeros(size(rhs), 'like', rhs);
        z_coarse = level.coarse_factor \ rhs;
        x = z_coarse;
        z_post = zeros(size(rhs), 'like', rhs);
    else
        x_pre = zeros(size(rhs), 'like', rhs);
        for sweep = 1:opts.mg_pre_sweeps
            x_pre = x_pre + opts.mg_jacobi_weight * level.diagonal_inverse .* ...
                (rhs - level.A * x_pre);
        end
        coarse_rhs = level.P' * (rhs - level.A * x_pre);
        z_coarse = level.P * vcycle(levels, 2, coarse_rhs, opts);
        x_after_coarse = x_pre + z_coarse;
        x = x_after_coarse;
        for sweep = 1:opts.mg_post_sweeps
            x = x + opts.mg_jacobi_weight * level.diagonal_inverse .* ...
                (rhs - level.A * x);
        end
        z_post = x - x_after_coarse;
        clear x_after_coarse coarse_rhs
    end
    z_sum = (x_pre + z_coarse) + z_post;
    z_reference = apply_cycles(levels, rhs, opts);
    reference_norm = norm(z_reference);
    denom = max(reference_norm, realmin);
    stats = struct();
    stats.protocol = 'one fine-grid V-cycle; Afun is the unshifted Helmholtz operator';
    stats.component_names = {'pre_smoothing', 'prolonged_coarse', 'post_smoothing'};
    stats.gram_names = [{'residual'}, strcat('A_', stats.component_names)];
    stats.cycle_count = 1;
    stats.level_count = numel(levels);
    stats.rhs_norm = rhs_norm;
    stats.correction_norms = [norm(x_pre), norm(z_coarse), norm(z_post)];
    stats.combined_correction_norm = norm(x);
    stats.relative_sum_vs_cycle_error = norm(z_sum - x) / denom;
    stats.relative_sum_vs_Msolve_error = norm(z_sum - z_reference) / denom;
    stats.relative_cycle_vs_Msolve_error = norm(x - z_reference) / denom;
    clear z_sum z_reference

    images = cell(1, 3);
    images{1} = Afun(x_pre); clear x_pre
    images{2} = Afun(z_coarse); clear z_coarse
    images{3} = Afun(z_post); clear z_post
    a_combined = Afun(x); clear x
    image_norms = cellfun(@norm, images);
    stats.operator_image_norms = image_norms;
    stats.operator_image_norms_over_rhs = image_norms / rhs_norm;
    stats.combined_operator_image_norm = norm(a_combined);
    a_sum = (images{1} + images{2}) + images{3};
    stats.relative_operator_linearity_error = norm(a_sum - a_combined) / ...
        max(norm(a_combined), realmin);
    clear a_sum
    [stats.combined_optimal_alpha, stats.combined_optimal_contraction] = ...
        scalar_contraction(rhs, a_combined, rhs_norm);
    stats.combined_unit_step_contraction = norm(rhs - a_combined) / rhs_norm;
    clear a_combined
    stats.component_optimal_alpha = complex(zeros(1, 3));
    stats.component_optimal_contraction = ones(1, 3);
    stats.component_unit_step_contraction = ones(1, 3);
    for j = 1:3
        [stats.component_optimal_alpha(j), stats.component_optimal_contraction(j)] = ...
            scalar_contraction(rhs, images{j}, rhs_norm);
        stats.component_unit_step_contraction(j) = norm(rhs - images{j}) / rhs_norm;
    end

    % Unit-normalized columns prevent raw scale from controlling rank decisions.
    % A zero component is represented by a zero row/column of this Gram matrix.
    gram = complex(zeros(4));
    gram(1, 1) = 1;
    r_unit = rhs / rhs_norm;
    for j = 1:3
        if image_norms(j) > 0
            images{j} = images{j} / image_norms(j);
            gram(j + 1, j + 1) = 1;
            gram(1, j + 1) = r_unit' * images{j};
            gram(j + 1, 1) = conj(gram(1, j + 1));
        end
        for k = 1:j-1
            gram(k + 1, j + 1) = images{k}' * images{j};
            gram(j + 1, k + 1) = conj(gram(k + 1, j + 1));
        end
    end
    stats.normalized_complex_gram = gram;
    H = (gram(2:4, 2:4) + gram(2:4, 2:4)') / 2;
    [V, D] = eig(H, 'vector');
    eigenvalues = real(D);
    % Gram eigenvalues are squared singular values. This conservative cutoff
    % suppresses spurious cancellation from nearly dependent components.
    cutoff = 100 * eps * max(1, max(eigenvalues));
    retained = eigenvalues > cutoff;
    coeff = complex(zeros(3, 1));
    if any(retained)
        coeff = V(:, retained) * ((V(:, retained)' * gram(2:4, 1)) ./ eigenvalues(retained));
    end
    joint_residual = r_unit;
    for j = 1:3
        joint_residual = joint_residual - coeff(j) * images{j};
    end
    stats.joint_span_contraction = norm(joint_residual);
    stats.joint_span_rank = nnz(retained);
    stats.joint_gram_eigenvalues = sort(eigenvalues, 'descend');
    stats.joint_gram_eigenvalue_cutoff = cutoff;
    stats.joint_span_coefficients_normalized = coeff;
    stats.joint_span_coefficients = complex(zeros(3, 1));
    nonzero = image_norms(:) > 0;
    stats.joint_span_coefficients(nonzero) = rhs_norm * coeff(nonzero) ./ image_norms(nonzero).';
end

function [alpha, contraction] = scalar_contraction(rhs, image, rhs_norm)
    image_norm = norm(image);
    if image_norm == 0
        alpha = 0;
        contraction = 1;
        return;
    end
    image_unit = image / image_norm;
    alpha_unit = image_unit' * rhs;
    alpha = alpha_unit / image_norm;
    contraction = norm(rhs - alpha_unit * image_unit) / rhs_norm;
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
