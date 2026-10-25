function model = convert_model(old, theta)
%CONVERT_MODEL Original lean model (layers with space, Ar_base, Ar_right) to dmsrb layout.
    model = struct('layers', {cell(numel(old.layers), 1)}, 'affine', isfield(old.layers{1}, 'Ar_base'));
    if model.affine, model.theta = theta; end
    for j = 1:numel(old.layers)
        l = old.layers{j};
        layer = struct('space', convert_space(l.space));
        if model.affine, layer.operators = {l.Ar_base, l.Ar_right}; end
        model.layers{j} = layer;
    end
end
