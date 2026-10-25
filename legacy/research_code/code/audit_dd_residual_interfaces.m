function report = audit_dd_residual_interfaces(training_file, vectors_dir, output_json)
%AUDIT_DD_RESIDUAL_INTERFACES Saved-field interface diagnostics, no assembly.
% Loads only partition, geometry, resolved_problem_cfg from frozen training.
% Existing inputs are read-only; an existing output_json is never overwritten.
% Energy is the Euclidean sum |v|^2, not a Helmholtz energy norm.
    assert(~isfile(output_json), 'Refusing to overwrite an existing interface audit.');
    t = load(training_file, 'partition', 'geometry', 'resolved_problem_cfg');
    g = t.geometry;
    n = double(g.grid_size(:).');
    np = double(g.physical_grid_size(:).');
    pml = double(g.pml_cells(:).');
    h = double(g.spacing(:).');
    assert(numel(n)==3 && all(n==np+2*pml), 'Inconsistent saved grid metadata.');
    assert(numel(h)==3 && all(h>0), 'Missing or invalid saved grid spacing.');
    N = prod(n);
    label = zeros(N, 1, 'uint32');
    count = 0;
    names = cell(1, numel(t.partition.blocks));
    for b = 1:numel(t.partition.blocks)
        ids = t.partition.blocks(b).idx;
        assert(all(ids>=1 & ids<=N & ids==round(ids)), 'Invalid block index.');
        assert(~any(label(ids)), 'Partition blocks overlap.');
        label(ids) = uint32(b);
        count = count + numel(ids);
        names{b} = t.partition.blocks(b).name;
    end
    assert(count==N && all(label>0), 'Partition must cover every node once.');
    physical = false(n);
    physical(pml(1)+(1:np(1)), pml(2)+(1:np(2)), pml(3)+(1:np(3))) = true;
    physical = physical(:);
    [masks, geometry_stats] = make_masks(label, physical, n);
    files = dir(fullfile(vectors_dir, '*_vectors.mat'));
    assert(~isempty(files), 'No *_vectors.mat files were found.');
    [~, order] = sort({files.name}); files = files(order);
    report = struct('definition_version', 'dd_residual_interfaces_v1', ...
        'training_file', training_file, 'vectors_dir', vectors_dir, ...
        'matlab_version', version, 'omega', t.resolved_problem_cfg.omega, ...
        'N', N, 'grid_size', n, 'physical_grid_size', np, 'pml_cells', pml, ...
        'spacing', h, 'block_names', {names}, 'geometry', geometry_stats);
    report.definitions = struct( ...
        'energy', 'Euclidean squared norm sum(abs(v).^2), not Helmholtz energy', ...
        'internal_edge', 'Unoriented nearest-neighbor grid pair, counted once; no edge to an exterior ghost node', ...
        'all_interblock', 'Both endpoints of internal edges with distinct positive block labels', ...
        'physical_interblock', 'Both endpoints of those distinct-label edges whose endpoints are both physical', ...
        'physical_pml_transition', 'Both endpoints of internal edges with exactly one physical endpoint, regardless of block labels', ...
        'physical_conditional', 'Mask intersect physical; energy divided by all physical energy, volume by number of physical nodes', ...
        'enrichment', 'Energy fraction divided by volume fraction; 1 is uniform mean squared amplitude', ...
        'edge_categories', 'Physical inter-block and physical within-block edges; both endpoints physical', ...
        'edge_mean_squared_difference', 'Mean abs(v(j)-v(i))^2 per eligible edge; optional h scaling uses the edge axis spacing', ...
        'edge_scale_invariant', 'Per-edge mean squared difference divided by mean squared field amplitude on physical nodes', ...
        'rafter', 'Residual after fixed-work normalized-RHS completion; raw energy is not on the same scale as r0', ...
        'overlap', 'Masks may overlap; their fractions need not sum to one', ...
        'undefined', 'A ratio with zero denominator is NaN (JSON null)');
    report.records = struct([]);
    for k = 1:numel(files)
        source = fullfile(files(k).folder, files(k).name);
        meta = load(source, 'mu', 'label');
        record = struct('source_file', source, 'mu', meta.mu, 'variant', meta.label);
        for field_name = {'r0', 'e0', 'rafter'}
            f = field_name{1}; s = load(source, f);
            assert(isfield(s,f), 'Missing saved field %s.', f);
            v = s.(f); clear s
            assert(isvector(v) && numel(v)==N, 'Field/grid mismatch.');
            record.(f) = summarize_field(v(:), label, physical, masks, n, h, geometry_stats);
            clear v
        end
        if isempty(report.records), report.records=record; else, report.records(end+1)=record; end
        fprintf('INTERFACE_AUDIT omega=%g %s\n', report.omega, files(k).name);
    end
    report.validated = true;
    parent = fileparts(output_json);
    if ~isempty(parent) && ~isfolder(parent), mkdir(parent); end
    fid = fopen(output_json, 'w'); assert(fid>=0, 'Cannot create audit output.');
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, '%s\n', jsonencode(report, PrettyPrint=true));
end

function [masks, s] = make_masks(labels, physical, n)
    N = prod(n); masks = false(N, 3);
    edge_count = zeros(2, 3); transition_count = zeros(1, 3);
    all_count = zeros(1, 3); stride = [1, n(1), n(1)*n(2)];
    for d = 1:3
        for start = 1:65536:N-stride(d)
            ids = (start:min(start+65535,N-stride(d))).';
            ids = ids(mod(floor((ids-1)/stride(d)),n(d)) < n(d)-1);
            js = ids+stride(d);
            different = labels(ids)>0 & labels(js)>0 & labels(ids)~=labels(js);
            phys_edge = physical(ids) & physical(js);
            transition = xor(physical(ids),physical(js));
            selectors = {different, different & phys_edge, transition};
            for b = 1:3
                keep = selectors{b};
                masks(ids(keep),b)=true; masks(js(keep),b)=true;
            end
            edge_count(:,d) = edge_count(:,d) + [nnz(different & phys_edge);nnz(~different & phys_edge)];
            transition_count(d)=transition_count(d)+nnz(transition);
            all_count(d)=all_count(d)+numel(ids);
        end
    end
    s = struct('mask_names', {{'all_interblock','physical_interblock','physical_pml_transition'}}, ...
        'mask_node_count', sum(masks,1), 'physical_mask_node_count', sum(masks & physical,1), ...
        'physical_node_count', nnz(physical), 'volume_fraction', sum(masks,1)/N, ...
        'physical_volume_fraction', sum(masks & physical,1)/nnz(physical), ...
        'axis_names', {{'x','y','z'}}, 'edge_category_names', {{'physical_interblock','physical_withinblock'}}, ...
        'physical_edge_count_by_axis', edge_count, 'transition_edge_count_by_axis', transition_count, ...
        'all_internal_edge_count_by_axis', all_count);
end

function s = summarize_field(v, labels, physical, masks, n, h, geometry_stats)
    N=prod(n); total=0; phys_energy=0; mask_energy=zeros(1,3); mask_phys_energy=zeros(1,3);
    for start = 1:65536:N
        ids = start:min(start+65535,N); values=v(ids);
        assert(all(isfinite(values)), 'Nonfinite saved field.');
        power = abs(values).^2; total=total+sum(power);
        p=physical(ids); phys_energy=phys_energy+sum(power(p));
        for k=1:3
            m=masks(ids,k); mask_energy(k)=mask_energy(k)+sum(power(m));
            mask_phys_energy(k)=mask_phys_energy(k)+sum(power(m & p));
        end
    end
    sums=zeros(2,3); stride=[1,n(1),n(1)*n(2)];
    for d=1:3
        for start=1:65536:N-stride(d)
            ids=(start:min(start+65535,N-stride(d))).';
            ids=ids(mod(floor((ids-1)/stride(d)),n(d)) < n(d)-1);
            js=ids+stride(d); keep=physical(ids) & physical(js);
            ids=ids(keep); js=js(keep); different=labels(ids)~=labels(js);
            power=abs(v(js)-v(ids)).^2;
            sums(:,d)=sums(:,d)+[sum(power(different));sum(power(~different))];
        end
    end
    fraction=safe_divide(mask_energy,total);
    phys_fraction=safe_divide(mask_phys_energy,phys_energy);
    counts=geometry_stats.physical_edge_count_by_axis;
    means=safe_divide(sum(sums,2),sum(counts,2));
    gradient_sums=sums./(h.^2);
    gradient_means=safe_divide(sum(gradient_sums,2),sum(counts,2));
    average_power=phys_energy/geometry_stats.physical_node_count;
    s=struct('energy_total',total,'energy_physical',phys_energy, ...
        'mask_energy',mask_energy,'physical_mask_energy',mask_phys_energy, ...
        'energy_fraction',fraction,'physical_energy_fraction',phys_fraction, ...
        'enrichment',safe_divide(fraction,geometry_stats.volume_fraction), ...
        'physical_enrichment',safe_divide(phys_fraction,geometry_stats.physical_volume_fraction), ...
        'edge_squared_difference_sum_by_axis',sums, ...
        'edge_mean_squared_difference_by_axis',safe_divide(sums,counts), ...
        'edge_mean_squared_difference',means, ...
        'edge_mean_squared_gradient_by_axis',safe_divide(gradient_sums,counts), ...
        'edge_mean_squared_gradient',gradient_means, ...
        'edge_mean_squared_difference_over_physical_mean_power',safe_divide(means,average_power), ...
        'edge_mean_squared_gradient_over_physical_mean_power',safe_divide(gradient_means,average_power), ...
        'interblock_to_withinblock_difference_ratio',safe_divide(means(1),means(2)), ...
        'interblock_to_withinblock_gradient_ratio',safe_divide(gradient_means(1),gradient_means(2)));
end

function value = safe_divide(a,b)
    value=a./b;
    value(~isfinite(value))=NaN;
end
