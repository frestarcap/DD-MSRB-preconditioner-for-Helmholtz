function report = run_helmholtz_3d_40pct_reproduction(candidate)
%RUN_HELMHOLTZ_3D_40PCT_REPRODUCTION Reproduce the 3D 40% DD results.
%
% candidate is lens, waveguide, or all. The run trains global and DD spaces
% from the same settings and checks storage reduction and online residuals.

    if nargin < 1 || isempty(candidate)
        candidate = 'all';
    end
    candidate = lower(strtrim(char(string(candidate))));
    if strcmp(candidate, 'all')
        names = {'lens', 'waveguide'};
    elseif ismember(candidate, {'lens', 'waveguide'})
        names = {candidate};
    else
        error('candidate must be lens, waveguide, or all.');
    end

    rows = repmat(empty_report_row(), numel(names), 1);
    summarized_results = cell(numel(names), 1);
    artifact_files = strings(numel(names), 1);
    out_dir = helm_output_path('outputs', 'helmholtz_3d_candidates', 'large');
    for k = 1:numel(names)
        spec = helmholtz_3d_40pct_config(names{k});
        fprintf('\nReproducing %s 3D DD memory result\n', spec.name);
        fprintf('Expected basis storage: global %.2f MiB, DD %.2f MiB\n', ...
            spec.expected.global_basis_mib, spec.expected.dd_basis_mib);
        run_result = run_helmholtz_3d_candidate_experiments( ...
            spec.mode, spec.driver_cfg);

        one = run_result.candidates{1};
        validate_complete_model_artifacts(one);
        artifact_file = fullfile(out_dir, sprintf( ...
            'helmholtz_3d_candidates_%s.mat', spec.driver_cfg.output_tag));
        summarized_results{k} = validate_saved_artifact_file(artifact_file);
        artifact_files(k) = string(artifact_file);
        global_mib = one.models.global.basis_bytes / 2^20;
        dd_mib = one.models.dd.basis_bytes / 2^20;
        saving = 1 - dd_mib / global_mib;
        dd_relres = arrayfun(@(test) test.dd.relative_residual, one.tests);
        shifted_time = arrayfun(@(test) test.shifted.total_time, one.tests);
        dd_time = arrayfun(@(test) test.dd.total_time, one.tests);
        speedup = shifted_time ./ dd_time;

        rows(k) = struct('candidate', string(spec.name), ...
            'N', one.problem_summary.N, ...
            'omega', one.problem_cfg.omega, ...
            'global_basis_MiB', global_mib, ...
            'dd_basis_MiB', dd_mib, ...
            'saving_percent', 100 * saving, ...
            'max_final_relres', max(dd_relres), ...
            'speedup_min', min(speedup), ...
            'speedup_max', max(speedup));

        if saving < spec.expected.saving_fraction - 0.02
            error('%s storage saving %.1f%% is outside the expected range.', ...
                spec.name, 100 * saving);
        end
        if max(dd_relres) > 1.05 * spec.driver_cfg.gmres_tol
            error('%s DD solve did not reach the requested residual.', spec.name);
        end
    end

    summary = struct2table(rows);
    report = struct('created_at', datetime('now'), 'summary', summary, ...
        'results', {summarized_results}, 'artifact_files', artifact_files);
    save(fullfile(out_dir, 'helmholtz_3d_40pct_reproduction.mat'), ...
        'report', '-v7.3');
    writetable(summary, fullfile(out_dir, ...
        'helmholtz_3d_40pct_reproduction.csv'));
    disp(summary);
end

function plot_data = validate_saved_artifact_file(filename)
    if ~isfile(filename)
        error('Expected saved artifact file was not created: %s', filename);
    end
    saved = load(filename, 'results', 'paper_plot_data');
    if ~isfield(saved, 'results') || ~isfield(saved, 'paper_plot_data')
        error('%s must contain results and paper_plot_data.', filename);
    end
    if ~isfield(saved.results, 'artifact_manifest') || ...
            ~saved.results.artifact_manifest.contains_full_models
        error('%s does not declare complete saved ROM models.', filename);
    end
    for k = 1:numel(saved.results.candidates)
        validate_complete_model_artifacts(saved.results.candidates{k});
    end
    if isempty(saved.paper_plot_data.candidates)
        error('%s contains no lightweight plotting data.', filename);
    end
    plot_data = saved.paper_plot_data;
end

function validate_complete_model_artifacts(candidate)
    n = candidate.problem_summary.N;
    assert(numel(candidate.rhs) == n, 'Saved right-hand side has wrong size.');
    indices = sort(vertcat(candidate.partition.blocks.idx));
    assert(isequal(indices, (1:n).'), 'Saved DD partition is incomplete.');
    styles = {'global', 'dd'};
    for style_index = 1:numel(styles)
        style = styles{style_index};
        if ~isfield(candidate.models, style) || ...
                ~isfield(candidate.models.(style), 'layers')
            error('%s is missing complete %s ROM layers.', candidate.name, style);
        end
        layers = candidate.models.(style).layers;
        if numel(layers) ~= candidate.models.(style).num_layers
            error('%s %s ROM has an inconsistent layer count.', ...
                candidate.name, style);
        end
        for layer_index = 1:numel(layers)
            layer = layers{layer_index};
            required = {'space', 'Ar0', 'Ar1', 'spectrum'};
            if ~all(isfield(layer, required))
                error('%s %s layer %d is missing saved ROM artifacts.', ...
                    candidate.name, style, layer_index);
            end
            assert(isequal(size(layer.Ar0), [layer.rank, layer.rank]) && ...
                isequal(size(layer.Ar1), [layer.rank, layer.rank]), ...
                'Saved reduced operators have inconsistent dimensions.');
            entries = validate_saved_space(layer.space, n);
            assert(entries == layer.basis_entries, 'Saved basis entry count differs.');
        end
    end
    for k = 1:numel(candidate.tests)
        for key = {'shifted', 'global', 'dd'}
            result = candidate.tests(k).(key{1});
            assert(numel(result.solution) == n && all(isfinite(result.solution)), ...
                'Saved final solution is missing or nonfinite.');
        end
    end
end

function entries = validate_saved_space(space, n)
    switch space.type
        case 'dense'
            assert(size(space.basis, 1) == n, 'Saved global basis has wrong size.');
            entries = numel(space.basis);
        case 'blockn'
            entries = 0;
            for b = 1:numel(space.blocks)
                block = space.blocks(b);
                assert(size(block.basis, 1) == numel(block.idx) && ...
                    size(block.basis, 2) == block.k, 'Saved local basis has wrong size.');
                entries = entries + numel(block.basis);
            end
        otherwise
            error('Unknown saved reduced-space type: %s', space.type);
    end
end

function row = empty_report_row()
    row = struct('candidate', "", 'N', 0, 'omega', 0, ...
        'global_basis_MiB', NaN, 'dd_basis_MiB', NaN, ...
        'saving_percent', NaN, 'max_final_relres', NaN, ...
        'speedup_min', NaN, 'speedup_max', NaN);
end
