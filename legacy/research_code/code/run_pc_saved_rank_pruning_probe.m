function results = run_pc_saved_rank_pruning_probe(mat_file, rank_caps, tag)
%RUN_PC_SAVED_RANK_PRUNING_PROBE Re-evaluate FGMRES after local rank pruning.
%
% This research-only diagnostic reloads a trained block ROM, removes trailing
% local basis vectors, reassembles the reduced operators, and recomputes the
% three-step Helmholtz FGMRES residual. It does not rerun offline references or
% WaveHoltz completion.

    if nargin < 1 || isempty(mat_file)
        mat_file = helm_output_path('outputs', 'pc_dd_storage_exploration', ...
            'pc_dd_storage_paper_waveholtz_narrow4_greedy_full.mat');
    end
    if nargin < 2 || isempty(rank_caps)
        rank_caps = [5, 5, 4, 3; 6, 6, 5, 5; 9, 10, 5, 5];
    end
    if nargin < 3 || isempty(tag)
        tag = 'pml_20pct';
    end

    loaded = load(mat_file, 'run_result');
    if ~isfield(loaded, 'run_result')
        error('%s does not contain run_result.', mat_file);
    end
    source = loaded.run_result;
    corr_rom = prune_correction_rom(source.problem, source.correction_rom, rank_caps);

    params = source.test.params(:);
    ntest = numel(params);
    nlayers = corr_rom.num_layers;
    relres = nan(ntest, nlayers);
    final_error = nan(ntest, 1);
    for k = 1:ntest
        [u, one_relres] = apply_layered_fgmres( ...
            source.problem, source.rhs, params(k), corr_rom);
        relres(k, :) = one_relres;
        A = helmholtz_operator(source.problem, params(k));
        u_exact = A \ source.rhs;
        final_error(k) = norm(u - u_exact) / max(norm(u_exact), 1e-30);
    end

    baseline = source.test.online_precorr_relres_final(:);
    tbl = table(params, baseline, relres(:, end), relres(:, end) ./ baseline, ...
        final_error, 'VariableNames', {'parameter', 'baseline_precorr_relres', ...
        'pruned_precorr_relres', 'pruned_over_baseline', 'pruned_exact_error'});
    for j = 1:nlayers
        tbl.(sprintf('pruned_layer%d_relres', j)) = relres(:, j);
    end

    old_entries = basis_entries(source.correction_rom);
    new_entries = basis_entries(corr_rom);
    global_entries = source.problem.Ntot * sum([5, 7, 9]);
    summary = struct();
    summary.rank_caps = rank_caps;
    summary.old_entries = old_entries;
    summary.new_entries = new_entries;
    summary.saved_vs_global_percent = 100 * (1 - new_entries / global_entries);
    summary.saved_vs_source_percent = 100 * (1 - new_entries / old_entries);
    summary.baseline_mean_relres = mean(baseline);
    summary.pruned_mean_relres = mean(relres(:, end));
    summary.pruned_max_relres = max(relres(:, end));
    summary.mean_residual_ratio = mean(relres(:, end) ./ baseline);
    summary.mean_exact_error = mean(final_error);

    out_dir = helm_output_path('outputs', 'pc_dd_storage_exploration');
    fig_dir = helm_output_path('outputs', 'figures', 'pc_dd_storage_exploration');
    ensure_dir(out_dir);
    ensure_dir(fig_dir);
    stem = ['pc_saved_rank_pruning_' char(tag)];
    csv_file = fullfile(out_dir, [stem '.csv']);
    mat_out = fullfile(out_dir, [stem '.mat']);
    fig_file = fullfile(fig_dir, [stem '.png']);
    writetable(tbl, csv_file);
    save(mat_out, 'mat_file', 'rank_caps', 'summary', 'tbl');
    plot_probe(tbl, summary, fig_file);

    results = struct('summary', summary, 'table', tbl, ...
        'csv_file', csv_file, 'mat_file', mat_out, 'figure_file', fig_file);
    fprintf(['Pruned entries %.0f -> %.0f; saving vs Global %.2f%%; ' ...
        'mean residual %.3e -> %.3e (%.2fx).\n'], ...
        old_entries, new_entries, summary.saved_vs_global_percent, ...
        summary.baseline_mean_relres, summary.pruned_mean_relres, ...
        summary.mean_residual_ratio);
    fprintf('Saved pruning probe: %s\n', csv_file);
end


function corr_rom = prune_correction_rom(prob, corr_rom, rank_caps)
    if size(rank_caps, 1) ~= corr_rom.num_layers
        error('rank_caps must contain one row per correction layer.');
    end
    for j = 1:corr_rom.num_layers
        layer = corr_rom.layers{j};
        space = layer.basis_space;
        if ~isfield(space, 'blocks') || isempty(space.blocks)
            error('Layer %d is not a block-N reduced space.', j);
        end
        if size(rank_caps, 2) ~= numel(space.blocks)
            error('rank_caps must contain one column per block.');
        end
        total_rank = 0;
        for b = 1:numel(space.blocks)
            old_rank = space.blocks(b).k;
            new_rank = min(old_rank, rank_caps(j, b));
            space.blocks(b).basis = space.blocks(b).basis(:, 1:new_rank);
            space.blocks(b).k = new_rank;
            space.blocks(b).meta.rank = new_rank;
            total_rank = total_rank + new_rank;
        end
        space.k = total_rank;
        layer.basis_space = space;
        layer.space_dim = total_rank;
        layer.Ar_base = rom_reduced_space_utils('assemble_reduced_matrix', prob.A_base, space);
        layer.Ar_right = rom_reduced_space_utils('assemble_reduced_matrix', prob.A_right, space);
        corr_rom.layers{j} = layer;
    end
end


function [u, layer_relres] = apply_layered_fgmres(prob, rhs, parameter, corr_rom)
    nlayers = corr_rom.num_layers;
    A = helmholtz_operator(prob, parameter);
    rhs_norm = max(norm(rhs), 1e-30);
    beta = norm(rhs);
    Q = zeros(prob.Ntot, nlayers + 1);
    Z = zeros(prob.Ntot, nlayers);
    H = zeros(nlayers + 1, nlayers);
    Q(:, 1) = rhs / beta;
    layer_relres = nan(1, nlayers);
    u = zeros(prob.Ntot, 1);
    for j = 1:nlayers
        layer = corr_rom.layers{j};
        alpha = (1 / parameter^2) - 1;
        Ar = layer.Ar_base + alpha * layer.Ar_right;
        br = rom_reduced_space_utils('project', layer.basis_space, Q(:, j));
        z = rom_reduced_space_utils('lift', layer.basis_space, Ar \ br);
        Z(:, j) = z;
        w = A * z;
        for i = 1:j
            H(i, j) = Q(:, i)' * w;
            w = w - H(i, j) * Q(:, i);
        end
        H(j + 1, j) = norm(w);
        if H(j + 1, j) > 1e-15
            Q(:, j + 1) = w / H(j + 1, j);
        end
        least_squares_rhs = zeros(j + 1, 1);
        least_squares_rhs(1) = beta;
        coeff = H(1:j + 1, 1:j) \ least_squares_rhs;
        u = Z(:, 1:j) * coeff;
        layer_relres(j) = norm(rhs - A * u) / rhs_norm;
    end
end


function A = helmholtz_operator(prob, parameter)
    alpha = (1 / parameter^2) - 1;
    A = prob.A_base + alpha * prob.A_right;
end


function entries = basis_entries(corr_rom)
    entries = 0;
    for j = 1:corr_rom.num_layers
        blocks = corr_rom.layers{j}.basis_space.blocks;
        entries = entries + sum(arrayfun(@(block) block.count * block.k, blocks));
    end
end


function plot_probe(tbl, summary, out_file)
    fig = figure('Visible', 'off', 'Position', [100, 100, 920, 520]);
    semilogy(tbl.parameter, tbl.baseline_precorr_relres, '-o', ...
        'LineWidth', 1.8, 'MarkerSize', 7, 'DisplayName', 'DOF-weighted DD');
    hold on;
    semilogy(tbl.parameter, tbl.pruned_precorr_relres, '-s', ...
        'LineWidth', 1.8, 'MarkerSize', 7, ...
        'DisplayName', sprintf('PML-pruned DD (%.1f%% saved)', ...
        summary.saved_vs_global_percent));
    hold off;
    grid on;
    xlabel('photonic-crystal wave speed');
    ylabel('three-step FGMRES relative residual');
    title('PC saved-ROM PML rank-pruning probe');
    legend('Location', 'best');
    set(findall(fig, '-property', 'FontSize'), 'FontSize', 15);
    exportgraphics(fig, out_file, 'Resolution', 220);
    close(fig);
end


function ensure_dir(path_name)
    if ~exist(path_name, 'dir')
        mkdir(path_name);
    end
end
