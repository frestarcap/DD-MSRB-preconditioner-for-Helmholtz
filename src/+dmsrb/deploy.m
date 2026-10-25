function deployed = deploy(model)
%DEPLOY Online data of a model: per layer the reduced space and reduced operators.
    deployed = struct('layers', {cell(size(model.layers))});
    for j = 1:numel(model.layers)
        l = model.layers{j};
        layer = struct('space', l.space);
        if isfield(l, 'operators'), layer.operators = l.operators; end
        deployed.layers{j} = layer;
    end
end
