function diagnostics = verify_helmholtz_3d_saved_results(mat_file)
%VERIFY_HELMHOLTZ_3D_SAVED_RESULTS Check saved bases, operators, and solutions.
% This reloads a completed candidate MAT without training or solving again.

    saved = load(mat_file, 'results', 'paper_plot_data');
    assert(isfield(saved, 'paper_plot_data'), 'Missing lightweight plotting data.');
    results = saved.results;
    assert(results.artifact_manifest.contains_full_models, ...
        'The MAT file contains summaries only.');
    diagnostics = struct('candidate', {}, 'max_galerkin_error', {}, ...
        'max_residual_discrepancy', {}, 'basis_entries', {});
    for k = 1:numel(results.candidates)
        candidate = results.candidates{k};
        problem = helmholtz_3d_problem_interface(candidate.problem_cfg);
        n = problem.N;
        indices = sort(vertcat(candidate.partition.blocks.idx));
        assert(isequal(indices, (1:n).'), 'Saved DD partition does not cover the grid.');
        assert(numel(candidate.rhs) == n && abs(norm(candidate.rhs) - 1) < 1e-12, ...
            'Saved normalized right-hand side is invalid.');
        mu = candidate.tests(1).mu;
        galerkin_error = 0;
        stored_entries = struct();
        styles = fieldnames(candidate.models);
        for s = 1:numel(styles)
            style = styles{s};
            model = candidate.models.(style);
            assert(isfield(model, 'layers') && numel(model.layers) == model.num_layers, ...
                'Saved ROM layers are incomplete.');
            entries = 0;
            for j = 1:model.num_layers
                layer = model.layers{j};
                space = layer.space;
                assert(rom_reduced_space_utils('dimension', space) == layer.rank);
                assert(isequal(size(layer.Ar0), [layer.rank, layer.rank]) && ...
                    isequal(size(layer.Ar1), [layer.rank, layer.rank]));
                switch space.type
                    case 'dense'
                        assert(size(space.basis, 1) == n);
                        count = numel(space.basis);
                    case 'blockn'
                        count = 0;
                        for b = 1:numel(space.blocks)
                            block = space.blocks(b);
                            assert(isequal(size(block.basis), [numel(block.idx), block.k]));
                            count = count + numel(block.basis);
                        end
                    otherwise
                        error('Unsupported saved space type: %s', space.type);
                end
                assert(count == layer.basis_entries, 'Saved basis storage is inconsistent.');
                entries = entries + count;
                coefficient = sin((1:layer.rank).') + 1i * cos((1:layer.rank).');
                lifted = rom_reduced_space_utils('lift', space, coefficient);
                projected = rom_reduced_space_utils('project', space, ...
                    problem.apply(mu, lifted));
                reduced = (layer.Ar0 + mu * layer.Ar1) * coefficient;
                err = norm(projected - reduced) / max(norm(reduced), eps);
                assert(isfinite(err) && err < 1e-9, ...
                    'Saved basis/reduced-operator mismatch: %.3e.', err);
                galerkin_error = max(galerkin_error, err);
            end
            assert(entries == model.basis_entries);
            stored_entries.(style) = entries;
        end
        residual_discrepancy = 0;
        methods = [{'direct', 'shifted'}, styles.'];
        for t = 1:numel(candidate.tests)
            test = candidate.tests(t);
            for method = methods
                one = test.(method{1});
                if strcmp(method{1}, 'direct') && isempty(one.solution)
                    continue;
                end
                assert(numel(one.solution) == n && all(isfinite(one.solution)));
                rho = norm(candidate.rhs - problem.apply(test.mu, one.solution)) / ...
                    norm(candidate.rhs);
                discrepancy = abs(rho - one.relative_residual);
                assert(isfinite(discrepancy) && discrepancy < 1e-11, ...
                    'Saved final residual mismatch: %.3e.', discrepancy);
                residual_discrepancy = max(residual_discrepancy, discrepancy);
                if ismember(method{1}, styles)
                    assert(numel(one.precorrection_solution) == n && ...
                        all(isfinite(one.precorrection_solution)));
                    rho0 = norm(candidate.rhs - ...
                        problem.apply(test.mu, one.precorrection_solution)) / norm(candidate.rhs);
                    discrepancy = abs(rho0 - one.precorrection_relative_residual);
                    assert(isfinite(discrepancy) && discrepancy < 1e-11, ...
                        'Saved precorrection residual mismatch: %.3e.', discrepancy);
                    residual_discrepancy = max(residual_discrepancy, discrepancy);
                end
            end
        end
        light = saved.paper_plot_data.candidates{k};
        assert(isequaln(light.summary_table, candidate.summary_table));
        assert(~isfield(light.tests(1).shifted, 'solution'));
        for s = 1:numel(styles)
            assert(~isfield(light.models.(styles{s}), 'layers'));
        end
        diagnostics(k) = struct('candidate', candidate.name, ...
            'max_galerkin_error', galerkin_error, ...
            'max_residual_discrepancy', residual_discrepancy, ...
            'basis_entries', stored_entries);
        fprintf('%s: saved Galerkin error %.3e, residual discrepancy %.3e\n', ...
            candidate.name, galerkin_error, residual_discrepancy);
    end
end
