function rows = measure_waveguide_rom_whos(source, output_file, report_file, reference_dir)
%MEASURE_WAVEGUIDE_ROM_WHOS Inspect frozen deployment objects; no solver run.
% Reproduces the legacy per-layer-index deployment layout. whos counts
% logical variable storage, not unique allocations or process RSS. Repeated
% per-layer index values remain in the actual deployment structure.
    if nargin < 3, report_file = ''; end
    if nargin < 4, reference_dir = ''; end
    timer = tic;
    saved = load(source, 'models', 'resolved_problem_cfg');
    omega = saved.resolved_problem_cfg.omega;
    names = {'global', 'dd'};
    rows = struct([]);
    if ~isempty(report_file)
        report = jsondecode(fileread(report_file));
        assert(report.all_converged && report.omega == omega);
    end
    for method_index = 1:numel(names)
        method = names{method_index};
        original = saved.models.(method);
        layers = cell(size(original.layers));
        for j = 1:numel(layers)
            l = original.layers{j}; sp = l.space;
            if strcmp(sp.type, 'dense')
                sp = rom_reduced_space_utils('make_dense_space', sp.basis);
            else
                assert(strcmp(sp.type, 'blockn'));
                sp = rmfield(sp, intersect(fieldnames(sp), {'partition','meta','block_meta'}));
                for b = 1:numel(sp.blocks), sp.blocks(b).meta = []; end
            end
            layers{j} = struct('space', sp, 'Ar0', l.Ar0, 'Ar1', l.Ar1, 'rank', l.rank);
            assert(isequaln(l.Ar0, layers{j}.Ar0) && isequaln(l.Ar1, layers{j}.Ar1));
        end
        model = struct('layers', {layers}, 'num_layers', numel(layers));
        % Measure this one object, not all aliases still in the workspace.
        info = whos('model');
        row = struct('omega', omega, 'method', method, 'N', layers{1}.space.n, ...
            'matlab_version', version, 'matlab_release', version('-release'), ...
            'computer', computer, 'job_id', getenv('SLURM_JOB_ID'), ...
            'node', getenv('SLURMD_NODENAME'), 'model_whos_bytes', info.bytes, ...
            'basis_whos_bytes', 0, 'reduced_whos_bytes', 0, ...
            'index_whos_bytes_all_layers', 0, 'index_whos_bytes_one_map', 0, ...
            'layer_ranks', cellfun(@(l) l.rank, layers), 'layers', struct([]));
        for j = 1:model.num_layers
            l = model.layers{j}; sp = l.space; old = original.layers{j};
            n = sp.n; rank = l.rank;
            assert(isequal(size(l.Ar0), [rank rank]) && isequal(size(l.Ar1), [rank rank]));
            lr = struct('layer', j, 'rank', rank, 'basis_bytes', 0, ...
                'reduced_bytes', array_bytes(l.Ar0) + array_bytes(l.Ar1), 'index_bytes', 0);
            if strcmp(sp.type, 'dense')
                assert(isequaln(sp.basis, old.space.basis));
                assert(isequal(size(sp.basis), [n rank]));
                lr.basis_bytes = array_bytes(sp.basis);
            else
                counts = 0; ranks = 0; seen = false(n, 1);
                assert(numel(sp.blocks) == numel(old.space.blocks));
                for b = 1:numel(sp.blocks)
                    block = sp.blocks(b); previous = old.space.blocks(b);
                    assert(isequaln(block.basis, previous.basis) && isequal(block.idx, previous.idx));
                    assert(isequal(size(block.basis), [numel(block.idx), block.k]));
                    assert(isa(block.idx, 'double') && all(block.idx >= 1 & block.idx <= n));
                    if j > 1
                        assert(isequal(block.idx, model.layers{1}.space.blocks(b).idx));
                    end
                    seen(block.idx) = true;
                    counts = counts + numel(block.idx); ranks = ranks + block.k;
                    lr.basis_bytes = lr.basis_bytes + array_bytes(block.basis);
                    lr.index_bytes = lr.index_bytes + array_bytes(block.idx);
                end
                assert(counts == n && all(seen) && ranks == rank);
                clear seen
            end
            assert(lr.basis_bytes == old.basis_bytes);
            assert(lr.reduced_bytes == 32 * rank^2);
            if j == 1, row.layers = lr; else, row.layers(j) = lr; end
            row.basis_whos_bytes = row.basis_whos_bytes + lr.basis_bytes;
            row.reduced_whos_bytes = row.reduced_whos_bytes + lr.reduced_bytes;
            row.index_whos_bytes_all_layers = row.index_whos_bytes_all_layers + lr.index_bytes;
            if j == 1, row.index_whos_bytes_one_map = lr.index_bytes; end
        end
        row.numeric_bytes_per_layer_indices = row.basis_whos_bytes + ...
            row.reduced_whos_bytes + row.index_whos_bytes_all_layers;
        row.numeric_bytes_one_shared_map = row.basis_whos_bytes + ...
            row.reduced_whos_bytes + row.index_whos_bytes_one_map;
        row.container_and_auxiliary_bytes = row.model_whos_bytes - row.numeric_bytes_per_layer_indices;
        row.stored_training_persistent_bytes = original.persistent_bytes;
        row.formula_persistent_bytes = original.basis_bytes + ...
            16 * original.reduced_entries + 8 * row.N * strcmp(method, 'dd');
        assert(row.container_and_auxiliary_bytes >= 0);
        assert(row.numeric_bytes_one_shared_map == row.formula_persistent_bytes);
        if ~isempty(report_file)
            assert(row.stored_training_persistent_bytes == row.formula_persistent_bytes);
            assert(abs(row.formula_persistent_bytes / 2^20 - report.([method '_persistent_MiB'])) < 1e-8);
        end
        % Independent regression against an existing deployment export, if supplied.
        if ~isempty(reference_dir)
            reference = load(fullfile(reference_dir, ['waveguide_' method '.mat']), 'model');
            assert(isequaln(reference.model, model), 'Deployment field/value mismatch.');
            clear reference
        end
        row.model_MiB = row.model_whos_bytes / 2^20;
        if method_index == 1, rows = row; else, rows(method_index) = row; end
        fprintf('WHOS_RESULT omega=%g method=%s bytes=%.0f MiB=%.9f basis=%.0f reduced=%.0f indices=%.0f overhead=%.0f\n', ...
            omega, method, row.model_whos_bytes, row.model_MiB, row.basis_whos_bytes, ...
            row.reduced_whos_bytes, row.index_whos_bytes_all_layers, row.container_and_auxiliary_bytes);
        clear model original layers l sp old block previous
    end
    metadata = dir(source);
    result = struct('source', source, 'source_file_bytes', metadata.bytes, ...
        'source_file_date', metadata.date, 'elapsed_seconds', toc(timer), ...
        'protocol', 'Legacy prepare_jsc_3d_deployment fields; whos logical object; per-layer indices retained; no RSS inference', ...
        'validated', true, 'rows', rows);
    fid = fopen(output_file, 'w'); assert(fid >= 0);
    cleaner = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(result));
    fprintf('WAVEGUIDE_WHOS_PASS omega=%g\n', omega);
end

function bytes = array_bytes(value)
    assert(isnumeric(value) && ~issparse(value));
    s = whos('value'); bytes = s.bytes;
end
