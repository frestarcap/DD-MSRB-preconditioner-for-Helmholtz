function redraw_paper_output_figures(target)
%REDRAW_PAPER_OUTPUT_FIGURES Redraw LaTeX-cited output figures from saved data.
% This script does not rerun numerical experiments.  It reloads saved paper
% result MAT-files and overwrites the PNG files cited by template.tex with
% larger fonts/legends suitable for two figures per row.

    if nargin < 1 || isempty(target)
        target = "all";
    end
    target = lower(string(target));

    close all;

    switch target
        case {"fields", "solutions"}
            plot_halfspace_dd_paper([], [], 'solution');
            redraw_ecc_figures(ecc_paper_results_file(), true);
            redraw_pc_figures(pc_paper_results_file(), true);
            plot_helmholtz_3d_paper_figures(struct('target','solutions'));
        case "all"
            prepare_halfspace_dd_paper('redraw');
            redraw_shifted_laplacian_figure();
            redraw_ecc_figures(ecc_paper_results_file());
            redraw_pc_figures(pc_paper_results_file());
            export_jsc_paper_data;
            plot_helmholtz_3d_paper_figures;
            redraw_waveguide_threshold_tradeoff();
        case {"shifted", "shifted_laplacian", "shifted_lap"}
            redraw_shifted_laplacian_figure();
        case {"pc", "photonic_crystal"}
            redraw_pc_figures(pc_paper_results_file());
            export_jsc_paper_data;
        case "pc_display"
            redraw_pc_figures(pc_paper_results_file());
        case "ecc_residual"
            redraw_ecc_figures(ecc_paper_results_file(), false, true);
        case {"ecc", "scattering"}
            redraw_ecc_figures(ecc_paper_results_file());
        case {"helm", "halfspace", "half_space"}
            prepare_halfspace_dd_paper('redraw');
        case {"3d", "helmholtz_3d", "three_dimensional"}
            plot_helmholtz_3d_paper_figures;
            redraw_waveguide_threshold_tradeoff();
        otherwise
            error('target=%s must be all, fields, helm, shifted, pc, pc_display, ecc, ecc_residual, or 3d.', target);
    end
end

function redraw_waveguide_threshold_tradeoff()
    out=helm_output_path('outputs','waveguide_threshold_tradeoff_20260908','isolated_timing');
    if isfile(fullfile(out,'summary.mat'))
        plot_waveguide_threshold_tradeoff(out);
    end
end

function redraw_shifted_laplacian_figure()
    plot_helm_shifted_laplacian_completion_figure( ...
        "paper", "helm_shifted_laplacian_completion", 0.8622);
end

function mat_file = ecc_paper_results_file()
    mat_file = resolve_paper_results_file('ecc', { ...
        helm_output_path('outputs', 'helm_trajectory_evidence', 'review_repaired_20260905', 'ecc_residual_ablation_paper.mat'), ...
        helm_output_path('outputs', 'helm_trajectory_evidence', 'greedy_adoption_20260905', 'ecc_residual_ablation_paper.mat'), ...
        helm_output_path('outputs', 'helm_trajectory_evidence', 'ecc_residual_ablation_paper.mat')}, ...
        'helmholtz', [1e-3, 1e-2, 1e-1]);
end

function mat_file = pc_paper_results_file()
    current=helm_output_path('outputs','revision_20260907','pc','paper_data.mat');
    if isfile(current)
        [valid,reason]=is_current_paper_results_file(current,'helm_lift',[1e-4,1e-3,1e-2]);
        assert(valid,'Primary revision PC data are invalid: %s',reason);
        mat_file=current;
        fprintf('Primary PC figure source: %s\n',mat_file);
        return;
    end
    mat_file = resolve_paper_results_file('pc', { ...
        helm_output_path('outputs','revision_20260907','pc','paper_data.mat'), ...
        helm_output_path('outputs', 'helm_trajectory_evidence', 'review_repaired_20260905', 'pc_residual_ablation_paper.mat'), ...
        helm_output_path('outputs', 'helm_trajectory_evidence', 'greedy_adoption_20260905', 'pc_waveholtz_residual_ablation_paper.mat'), ...
        helm_output_path('outputs', 'helm_trajectory_evidence', 'pc_waveholtz_residual_ablation_paper.mat'), ...
        helm_output_path('outputs', 'helm_trajectory_evidence', 'pc_residual_ablation_paper.mat')}, ...
        'helm_lift', [1e-4, 1e-3, 1e-2]);
end

function mat_file = resolve_paper_results_file(problem_type, candidates, expected_reference, expected_tols)
    skipped = strings(0, 1);
    for k = 1:numel(candidates)
        candidate = candidates{k};
        if ~exist(candidate, 'file')
            continue;
        end
        [is_clean, reason] = is_current_paper_results_file( ...
            candidate, expected_reference, expected_tols);
        if is_clean
            mat_file = candidate;
            return;
        end
        skipped(end + 1) = sprintf('%s (%s)', candidate, reason); %#ok<AGROW>
    end

    if ~isempty(skipped)
        error('No current %s paper results file found. Skipped:%s%s', ...
            upper(problem_type), newline, strjoin(skipped, newline));
    end
    error('No %s paper results file found.', upper(problem_type));
end

function [tf, reason] = is_current_paper_results_file(mat_file, expected_reference, expected_tols)
    tf = false;
    cfg = load_paper_cfg(mat_file);
    if isempty(cfg)
        reason = 'missing cfg';
        return;
    end
    if ~isfield(cfg, 'reference_mode') || ~strcmpi(cfg.reference_mode, expected_reference)
        reason = sprintf('reference_mode is not %s', expected_reference);
        return;
    end
    if ~isfield(cfg, 'online_completion_mode') || ~strcmpi(cfg.online_completion_mode, 'waveholtz')
        reason = 'online_completion_mode is not waveholtz';
        return;
    end
    if ~isfield(cfg, 'corr_pod_singular_value_tols') || ...
            ~isequal(size(cfg.corr_pod_singular_value_tols), size(expected_tols)) || ...
            any(abs(cfg.corr_pod_singular_value_tols - expected_tols) > ...
                10 * eps(max(1, max(expected_tols))))
        reason = 'layerwise tail schedule is stale';
        return;
    end
    if ~isfield(cfg, 'corr_block3_global_tail_strategy') || ...
            ~strcmpi(cfg.corr_block3_global_tail_strategy, 'marginal_storage')
        reason = 'DD allocation strategy is not marginal_storage';
        return;
    end
    tf = true;
    reason = 'clean';
end

function cfg = load_paper_cfg(mat_file)
    cfg = [];
    S = load(mat_file, 'base_cfg');
    if isfield(S, 'base_cfg')
        cfg = S.base_cfg;
        return;
    end

    S = load(mat_file, 'results');
    if ~isfield(S, 'results')
        return;
    end
    R = S.results;
    if isfield(R, 'base_cfg')
        cfg = R.base_cfg;
    elseif isfield(R, 'global_results') && isfield(R.global_results, 'cfg')
        cfg = R.global_results.cfg;
    elseif isfield(R, 'cfg')
        cfg = R.cfg;
    end
end


function redraw_ecc_figures(mat_file, fields_only, residual_only)
    if nargin < 2, fields_only = false; end
    if nargin < 3, residual_only = false; end
    S = load(mat_file);
    [R, DD] = unpack_global_dd_results(S);
    out_dir = helm_output_path('outputs', 'figures', 'ecc_correction_rom');
    ensure_dir(out_dir);

    k = R.summary.representative_test_index;
    ratio = R.summary.representative_ratio;
    tag = 'ratio_0_9072';
    dd_idx = [];
    if ~isempty(DD)
        dd_idx = choose_matching_index(DD.test.ratio, ratio, DD.summary.representative_test_index);
    end

    block_source = R;
    if ~isempty(DD)
        block_source = DD;
    end
    if ~fields_only
        if ~residual_only && has_block_partition(block_source)
            fig = plot_ecc_partition(block_source);
            save_png(fig, fullfile(out_dir, 'ecc_domain_decomposed_partition.png'));
        end

        tests = {R.test}; indices = k; labels = {'Global TA-ROM'};
        if ~isempty(DD)
            tests{end+1} = DD.test;
            indices(end+1) = dd_idx;
            labels{end+1} = 'DD-TA-ROM';
        end
        fig = new_wide_figure(); ax = axes(fig); hold(ax,'on');
        colors = {[0.85,0.05,0.05],[0.05,0.45,0.25]};
        markers = {'o','s'};
        for j=1:numel(tests)
            t=tests{j}; q=indices(j);
            h=t.online_precorr_relres_hist{q}; h=h(:);
            w=t.online_fp_relres_hist{q}; w=w(:); m=numel(h)-1;
            % Retain both measurements at the transition, without scaling.
            x=[(0:m)'; m+(0:numel(w)-1)']; y=[h;w];
            semilogy(ax,x,y,'Color',colors{j},'Marker',markers{j}, ...
                'LineWidth',3.2,'MarkerSize',9,'MarkerFaceColor','w', ...
                'MarkerIndices',unique(round(linspace(1,numel(y),22))), ...
                'DisplayName',labels{j});
        end
        plot_log_curve(ax,R.test.plain_fp_relres_hist{k},'-',[0,0,0], ...
            'Plain WaveHoltz',[],1e-15);
        plot_log_curve(ax,R.test.helm_baseline_relres_hist{k},':',[0.4,0,0.5], ...
            'Plain Helmholtz GMRES',[],1e-15);
        xlabel(ax,'online step index'); ylabel(ax,'residual');
        title(ax,sprintf('Scattering, a/b=%.4f',ratio));
        style_log_axes(ax); set(ax,'FontSize',28); ylim(ax,[1e-11,2]);
        legend(ax,'Location','northeast','FontSize',25);
        save_png(fig, fullfile(out_dir, ['ecc_joint_combined_residual_decay_full_field_' tag '.png']));
    end

    if residual_only, return; end

    obstacle = R.test.systems{k}.geom.internal;
    ref_grid = embed_field(R.test.wave_u_ref(:, k), R.problem.nx, R.problem.ny);
    if ~isempty(DD)
        corr_u = DD.test.online_final_u(:, dd_idx);
        field_title = 'DD-TA-ROM real part';
    else
        corr_u = R.test.online_final_u(:, k);
        field_title = 'Global TA-ROM real part';
    end
    corr_grid = embed_field(corr_u, R.problem.nx, R.problem.ny);
    fig = plot_field(R.problem.x_int, R.problem.y_int, real(ref_grid(2:end-1, 2:end-1).'), ...
        'Reference real part', 'real(u)', obstacle);
    save_png(fig, fullfile(out_dir, ['ecc_' tag '_real_reference_full_field.png']));
    fig = plot_field(R.problem.x_int, R.problem.y_int, real(corr_grid(2:end-1, 2:end-1).'), ...
        field_title, 'real(u)', obstacle);
    save_png(fig, fullfile(out_dir, ['ecc_' tag '_real_joint_correction_full_field.png']));
    if fields_only, return; end

    plot_ecc_global_spectral_energy(mat_file);
    if has_block_partition(block_source)
        fig = plot_block_singular_values(block_source.direct_rom.basis_space, 'Direct block spectra');
        save_png(fig, fullfile(out_dir, 'ecc_direct_blockwise_singular_decay.png'));
        fig = plot_block_singular_values(block_source.correction_rom.layers{1}.basis_space, 'Layer 1 block spectra');
        save_png(fig, fullfile(out_dir, 'ecc_layer_01_blockwise_singular_decay.png'));
    end

end

function redraw_pc_figures(mat_file, fields_only)
    if nargin < 2, fields_only = false; end
    S = load(mat_file);
    [R, DD] = unpack_global_dd_results(S);
    out_dir = helm_output_path('outputs', 'figures', 'photonic_crystal_correction_rom');
    ensure_dir(out_dir);

    [~,k] = min(abs(R.test.params-0.3068));
    c_rep = R.test.params(k);
    tag = 'c_right_0_3068';
    plot_floor = residual_plot_floor('pc');
    dd_idx = [];
    if ~isempty(DD)
        dd_idx = choose_matching_index(DD.test.params, c_rep, DD.summary.representative_test_index);
    end

    if ~fields_only
        series={R.test}; idx=k; labels={'Global TA-ROM'};
        if ~isempty(DD)
            series{2}=DD.test; idx(2)=dd_idx; labels{2}='DD-TA-ROM';
        end
        fig=plot_rom_stage_residuals(series,idx,labels,R.test,k, ...
            sprintf('Photonic crystal, c=%.4f',c_rep),'pc');
        save_png(fig, fullfile(out_dir, ['pc_joint_combined_residual_decay_full_field_' tag '.png']));
    end

    if ~isempty(DD)
        field_u = DD.test.online_final_u(:, dd_idx);
    else
        field_u = R.test.online_final_u(:, k);
    end
    grid = embed_field(field_u, R.problem.nx, R.problem.ny);
    field = grid(2:end-1, 2:end-1).';
    limit = max(abs([real(field(:));imag(field(:))]));
    if limit == 0, limit = 1; end
    fig = new_wide_figure(); fig.Position(4) = 840;
    tiles = tiledlayout(fig,2,1,'Padding','compact','TileSpacing','compact');
    components = {real(field),imag(field)};
    headings = {'Real part','Imaginary part'};
    labels = {'real(u)','imag(u)'};
    for m = 1:2
        ax = nexttile(tiles);
        imagesc(ax,R.problem.x_int,R.problem.y_int,components{m});
        set(ax,'YDir','normal'); axis(ax,'equal'); axis(ax,'tight');
        ylim(ax,[1,6]); clim(ax,[-limit,limit]);
        colormap(ax,helm_solution_colormap([-limit,limit]));
        style_image_axes(ax);
        title(ax,headings{m}); xlabel(ax,'x'); ylabel(ax,'y');
        cb = colorbar(ax); cb.FontSize = 22;
        ylabel(cb,labels{m},'FontSize',24);
        if m == 1, ax.XTickLabel = []; ax.XLabel.String = ''; end
    end
    save_png(fig, fullfile(out_dir, ['pc_' tag '_real_joint_correction_full_field.png']));

end


function fig = plot_ecc_partition(R)
    part = R.direct_rom.basis_space.partition;
    vals = zeros(part.nx * part.ny, 1);
    vals(part.idx_pml) = 3;
    vals(part.idx_far) = 2;
    vals(part.idx_near) = 1;
    img = reshape(vals, part.nx, part.ny).';

    fig = new_wide_figure();
    ax = axes(fig);
    imagesc(ax, R.problem.x_int, R.problem.y_int, img);
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal');
    axis(ax, 'tight');
    colormap(ax, dd_partition_colormap(3));
    clim(ax, [1, 3]);
    style_image_axes(ax);
    title(ax, 'Near / far / PML partition');
    xlabel(ax, 'x');
    ylabel(ax, 'y');
    cb = colorbar(ax, 'Ticks', [1, 2, 3], 'TickLabels', {'near', 'far', 'PML'});
    cb.FontSize = 22;
end

function tf = has_block_partition(R)
    tf = isfield(R, 'direct_rom') && isfield(R.direct_rom, 'basis_space') && ...
        isfield(R.direct_rom.basis_space, 'partition') && ...
        isfield(R.direct_rom.basis_space, 'block_meta');
end

function fig = plot_block_singular_values(space, ttl)
    fig = new_wide_figure();
    ax = axes(fig);
    hold(ax, 'on');
    plot_one_block(ax, space.block_meta.near, '-', [0.82 0.36 0.30], 'near');
    plot_one_block(ax, space.block_meta.far, '--', [0.25 0.52 0.64], 'far');
    plot_one_block(ax, space.block_meta.pml, ':', [0.66 0.60 0.78], 'PML');
    hold(ax, 'off');
    style_log_axes(ax);
    max_modes = max([numel(space.block_meta.near.singular_values), ...
        numel(space.block_meta.far.singular_values), ...
        numel(space.block_meta.pml.singular_values), 2]);
    xlim(ax, [1, max_modes]);
    xticks(ax, unique([1, get(ax, 'XTick')]));
    xlabel(ax, 'mode index');
    ylabel(ax, 'singular value');
    title(ax, ttl);
    add_legend(ax, 'northeast');
end

function plot_one_block(ax, meta, ls, color, name)
    sv = meta.singular_values;
    if has_curve(sv)
        semilogy(ax, 1:numel(sv), max(sv(:), 1e-16), ls, ...
            'Color', color, 'LineWidth', 3.2, 'Marker', 'o', 'MarkerSize', 8, ...
            'DisplayName', sprintf('%s (r=%d)', name, meta.rank));
    end
end

function [global_results, dd_results] = unpack_global_dd_results(S)
    dd_results = [];
    if isfield(S, 'global_results')
        global_results = S.global_results;
        if isfield(S, 'dd_results')
            dd_results = S.dd_results;
        end
        return;
    end
    if isfield(S, 'results') && isfield(S.results, 'global_results')
        global_results = S.results.global_results;
        if isfield(S.results, 'dd_results')
            dd_results = S.results.dd_results;
        end
        return;
    end
    global_results = S.results;
end

function fig = plot_field(x, y, data, ttl, cbar_label, obstacle_mask)
    if nargin < 6
        obstacle_mask = [];
    end
    fig = new_field_figure();
    ax = axes(fig);
    cdata = data;
    alpha_data = true(size(cdata));
    if ~isempty(obstacle_mask)
        interior = obstacle_mask(2:end-1, 2:end-1).';
        cdata(interior) = NaN;
        alpha_data(interior) = false;
    end
    h = imagesc(ax, x, y, cdata);
    if ~isempty(obstacle_mask)
        set(h, 'AlphaData', double(alpha_data));
        set(ax, 'Color', 'w');
    end
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal');
    axis(ax, 'tight');
    style_image_axes(ax);
    xlabel(ax, 'x');
    ylabel(ax, 'y');
    title(ax, ttl);
    colormap(ax, helm_solution_colormap(clim(ax)));
    cb = colorbar(ax);
    cb.FontSize = 22;
    ylabel(cb, cbar_label, 'FontSize', 24);
end

function plot_log_curve(ax, hist, line_style, color, label, max_step, floor_value)
    if ~has_curve(hist)
        return;
    end
    if nargin < 6 || isempty(max_step)
        max_step = Inf;
    end
    if nargin < 7 || isempty(floor_value)
        floor_value = residual_plot_floor('helm');
    end
    y = max(hist(:), floor_value);
    last_idx = min(numel(y), floor(max_step) + 1);
    y = y(1:last_idx);
    semilogy(ax, 0:(numel(y) - 1), y, line_style, ...
        'Color', color, 'LineWidth', 3.2, 'MarkerSize', 9, ...
        'MarkerFaceColor', 'w', 'DisplayName', label);
end

function floor_value = residual_plot_floor(problem_type)
    switch lower(string(problem_type))
        case "helm"
            floor_value = 1e-16;
        case "ecc"
            floor_value = 1e-11;
        case "pc"
            floor_value = 1e-10;
        otherwise
            error('Unknown residual plot type: %s.', problem_type);
    end
end

function grid = embed_field(u, nx, ny)
    grid = zeros(nx + 2, ny + 2);
    grid(2:end-1, 2:end-1) = reshape(u, nx, ny);
end

function idx = choose_matching_index(params, target, fallback)
    if nargin < 3 || isempty(fallback)
        fallback = 1;
    end
    if isempty(params)
        idx = fallback;
    else
        [~, idx] = min(abs(params(:) - target));
    end
end

function tf = has_curve(vals)
    tf = ~isempty(vals) && any(isfinite(vals(:)));
end

function fig = new_wide_figure()
    fig = figure('Color', 'w', 'Position', [80 80 980 720], 'Visible', 'off');
end

function fig = new_field_figure()
    fig = figure('Color', 'w', 'Position', [80 80 900 760], 'Visible', 'off');
end

function style_log_axes(ax)
    set(ax, 'FontSize', 25, 'LineWidth', 1.8, 'Box', 'on', ...
        'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.28, 'MinorGridAlpha', 0.18);
    ax.XMinorGrid = 'off';
    ax.YMinorGrid = 'on';
    ax.Title.FontSize = 30;
    ax.Title.FontWeight = 'bold';
    ax.XLabel.FontSize = 28;
    ax.YLabel.FontSize = 28;
end


function style_image_axes(ax)
    set(ax, 'FontSize', 24, 'LineWidth', 1.6, 'Box', 'on');
    ax.Title.FontSize = 29;
    ax.Title.FontWeight = 'bold';
    ax.XLabel.FontSize = 26;
    ax.YLabel.FontSize = 26;
end

function add_legend(ax, loc)
    lgd = legend(ax, 'Location', loc);
    lgd.FontSize = 21;
    lgd.Box = 'on';
    lgd.LineWidth = 1.1;
end

function save_png(fig, out_file)
    exportgraphics(fig, out_file, 'Resolution', 300);
    prefix = [helm_output_path('outputs','figures'),filesep];
    relative = extractAfter(out_file,prefix);
    published = helm_output_path('LaTeX_DL_468198_240419','figures',relative);
    if isfile(published), copyfile(out_file,published); end
    fprintf('redrew %s\n', out_file);
    close(fig);
end

function ensure_dir(dir_name)
    if ~exist(dir_name, 'dir')
        mkdir(dir_name);
    end
end
