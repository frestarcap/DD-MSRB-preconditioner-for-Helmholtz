function results = test_dd_mechanism_mg_probe(output_dir)
%TEST_DD_MECHANISM_MG_PROBE Reproducible checks, with saved scalar evidence.
    if nargin < 1
        output_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
            'outputs', 'dd_highfreq_mechanism_20260913', 'mg_probe');
    end
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    rng(1739, 'twister');
    grid_size = [17, 15, 13];
    n = prod(grid_size);
    T = cell(1, 3);
    for j = 1:3
        e = ones(grid_size(j), 1);
        T{j} = spdiags([-e, 2*e, -e], -1:1, grid_size(j), grid_size(j));
    end
    L = kron(speye(grid_size(3)), kron(speye(grid_size(2)), T{1})) + ...
        kron(speye(grid_size(3)), kron(T{2}, speye(grid_size(1)))) + ...
        kron(T{3}, speye(grid_size(1)*grid_size(2)));
    [gx, gy, gz] = ndgrid(linspace(0, 1, grid_size(1)), ...
        linspace(0, 1, grid_size(2)), linspace(0, 1, grid_size(3)));
    mass = 1.5 + 0.4*(gx(:) > 0.5);
    attenuation = 0.2 + 0.5*(gx(:).^8 + gy(:).^8 + gz(:).^8);
    A = L - spdiags(mass + 1i*attenuation, 0, n, n);
    Ashift = A - 1.2i*spdiags(mass, 0, n, n);
    rhs = randn(n, 1) + 1i*randn(n, 1);
    opts = struct('mg_coarse_max_dofs', 40, 'mg_cycles_per_apply', 1);
    results = struct('matlab_version', version, 'seed', 1739, ...
        'grid_size', grid_size, 'n', n, 'options', opts);
    [original, info_original] = build_shifted_laplacian_multigrid_preconditioner(Ashift, grid_size, opts);
    [diagnostic, probe, info_probe] = build_dd_mechanism_mg_probe(Ashift, grid_size, opts);
    z_original = original(rhs);
    z_diagnostic = diagnostic(rhs);
    results.original_vs_probe_Msolve = norm(z_original - z_diagnostic) / norm(z_original);
    assert(results.original_vs_probe_Msolve == 0, 'Msolve arithmetic changed.');
    assert(isequal(info_original.grid_sizes, info_probe.grid_sizes));
    results.multilevel = probe(rhs, @(v) A*v);
    check_stats(results.multilevel);
    assert(abs(results.multilevel.combined_unit_step_contraction - ...
        norm(rhs - A*z_original)/norm(rhs)) < 1e-13);

    scaled_stats = probe((2 - 3i)*rhs, @(v) A*v);
    results.complex_scaling_gram_error = norm(scaled_stats.normalized_complex_gram - ...
        results.multilevel.normalized_complex_gram, 'fro');
    results.complex_scaling_contraction_error = abs(scaled_stats.joint_span_contraction - ...
        results.multilevel.joint_span_contraction);
    assert(results.complex_scaling_gram_error < 1e-12);
    assert(results.complex_scaling_contraction_error < 1e-12);

    opts_single = opts; opts_single.mg_max_levels = 1;
    [single_M, single_probe] = build_dd_mechanism_mg_probe(Ashift, grid_size, opts_single);
    results.single_level = single_probe(rhs, @(v) A*v);
    check_stats(results.single_level);
    assert(results.single_level.operator_image_norms(1) == 0);
    assert(results.single_level.operator_image_norms(3) == 0);
    assert(results.single_level.joint_span_rank == 1);
    assert(abs(results.single_level.joint_span_contraction - ...
        results.single_level.combined_optimal_contraction) < 1e-12);
    assert(norm(single_M(rhs) - Ashift\rhs)/norm(single_M(rhs)) < 1e-12);

    opts_zero = opts; opts_zero.mg_pre_sweeps = 0; opts_zero.mg_post_sweeps = 0;
    [zero_M, zero_probe] = build_dd_mechanism_mg_probe(Ashift, grid_size, opts_zero);
    zero_original = build_shifted_laplacian_multigrid_preconditioner(Ashift, grid_size, opts_zero);
    results.zero_sweeps = zero_probe(rhs, @(v) A*v);
    check_stats(results.zero_sweeps);
    assert(isequal(zero_M(rhs), zero_original(rhs)));
    assert(results.zero_sweeps.joint_span_rank == 1);

    opts_two = opts; opts_two.mg_cycles_per_apply = 2;
    [two_M, two_probe] = build_dd_mechanism_mg_probe(Ashift, grid_size, opts_two);
    two_original = build_shifted_laplacian_multigrid_preconditioner(Ashift, grid_size, opts_two);
    results.two_cycle_Msolve_error = norm(two_M(rhs) - two_original(rhs));
    assert(results.two_cycle_Msolve_error == 0);
    results.two_cycle_probe_rejected = false;
    try
        two_probe(rhs, @(v) A*v);
    catch err
        results.two_cycle_probe_rejected = strcmp(err.identifier, 'dd_mechanism:singleCycleRequired');
    end
    assert(results.two_cycle_probe_rejected);
    results.all_passed = true;
    save(fullfile(output_dir, 'synthetic_probe_validation.mat'), ...
        'results', 'A', 'Ashift', 'rhs', 'grid_size', 'opts', '-v7');
    f = fopen(fullfile(output_dir, 'synthetic_probe_validation.json'), 'w');
    cleanup = onCleanup(@() fclose(f));
    fprintf(f, '%s\n', jsonencode(json_safe(results), PrettyPrint=true));
    fprintf('MG component probe: all checks passed, %d DOFs, %d levels.\n', n, info_probe.level_count);
    fprintf('Original Msolve error %.3g, decomposition error %.3g, joint contraction %.6g.\n', ...
        results.original_vs_probe_Msolve, results.multilevel.relative_sum_vs_Msolve_error, ...
        results.multilevel.joint_span_contraction);
end

function value = json_safe(value)
    if isstruct(value)
        fields = fieldnames(value);
        for k = 1:numel(fields)
            value.(fields{k}) = json_safe(value.(fields{k}));
        end
    elseif isnumeric(value) && ~isreal(value)
        value = struct('real', real(value), 'imag', imag(value));
    end
end

function check_stats(s)
    assert(s.relative_cycle_vs_Msolve_error < 1e-14);
    assert(s.relative_sum_vs_Msolve_error < 1e-14);
    assert(s.relative_operator_linearity_error < 1e-13);
    G = s.normalized_complex_gram;
    assert(norm(G - G', 'fro') < 1e-14);
    assert(min(eig((G + G')/2)) > -1e-12);
    assert(all(s.component_optimal_contraction <= 1 + 1e-12));
    assert(s.joint_span_contraction <= min(s.component_optimal_contraction) + 1e-11);
    assert(s.joint_span_contraction <= s.combined_optimal_contraction + 1e-11);
end
