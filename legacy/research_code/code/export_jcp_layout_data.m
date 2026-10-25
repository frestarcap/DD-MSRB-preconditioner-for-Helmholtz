function export_jcp_layout_data
%EXPORT_JCP_LAYOUT_DATA Export frozen plotting arrays; no solve or retraining.
    root = helm_project_root();
    out = fullfile(root, 'outputs', 'jcp_layout_20260925');
    if ~exist(out, 'dir'), mkdir(out); end
    s = load(fullfile(root, 'outputs', 'jcp_shifted_20260914', ...
        'paper_results.mat'), 'cases');
    v = s.cases{1}{10,3};
    assert(strcmp(v.method, 'global') && abs(v.parameter-.862209674352196)<1e-13);
    p = load(fullfile(root, 'outputs', 'halfspace_dd_paper', ...
        'paper_data.mat'), 'paper');
    x = p.paper.global.x_int;
    y = p.paper.global.y_int;
    U = reshape(v.u, numel(x), numel(y)).';
    labels = p.paper.partition.labels.';
    spectra = cellfun(@(layer) layer.singular_values, ...
        p.paper.global.layers, 'UniformOutput', false);
    parameter = v.parameter;
    assert(isequal(size(U), size(labels)) && max(labels(:)) == 10);
    save(fullfile(out, 'halfspace_plot_data.mat'), 'x', 'y', 'U', ...
        'labels', 'spectra', 'parameter', '-v7');

    % The ECC magnitude requires the saved complex field, not its real CSV.
    ecc = fullfile(root, 'outputs', 'ecc_dd_budget_20260919');
    s = load(fullfile(ecc, 'uniform5_results.mat'), 'cases', 'names');
    mi = find(strcmp(s.names, 'uniform5_dd'));
    assert(isscalar(mi));
    x = readmatrix(fullfile(ecc, 'paper_x.csv'));
    y = readmatrix(fullfile(ecc, 'paper_y.csv'));
    U = reshape(s.cases{1,mi}.u, numel(x), numel(y)).';
    previous = readmatrix(fullfile(ecc, 'paper_dd_field.csv'));
    assert(norm(real(U)-previous, 'fro') < 1e-12*norm(previous, 'fro'));
    rows = readtable(fullfile(ecc, 'paper_representative.csv'));
    row = rows(strcmp(rows.method, 'uniform5_dd'), :);
    parameter = row.parameter;
    residual = row.residual;
    assert(abs(parameter-.909176375080444)<1e-12 && residual<=1e-12);
    save(fullfile(out, 'ecc_plot_data.mat'), 'x', 'y', 'U', ...
        'parameter', 'residual', '-v7');
end
