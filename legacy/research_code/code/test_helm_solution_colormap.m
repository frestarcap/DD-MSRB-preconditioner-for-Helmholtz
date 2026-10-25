function test_helm_solution_colormap
%TEST_HELM_SOLUTION_COLORMAP Check palette consistency and plotting invariants.
    anchors = [0.10 0.20 0.55; 0.15 0.55 0.90; 1 1 1; ...
        0.90 0.30 0.18; 0.55 0.02 0.10];
    expected = interp1([-1; -.5; 0; .5; 1], anchors, ...
        asinh(8 * linspace(-1,1,257).') / asinh(8), 'linear');
    map = helm_solution_colormap();
    assert(isequal(map, expected));
    assert(isequal(map(129,:), [1 1 1]));
    scaled = helm_solution_colormap([-1e-4,1e-4]);
    assert(max(abs(scaled(:)-map(:))) < 1e-14);
    asymmetric = helm_solution_colormap([-1,3],257);
    assert(isequal(asymmetric(65,:), [1 1 1]));
    assert(all(isfinite(asymmetric(:))) && ...
        all(asymmetric(:) >= 0 & asymmetric(:) <= 1));
    rejected = false;
    try
        helm_solution_colormap([1,1]);
    catch err
        rejected = strcmp(err.identifier,'helm:InvalidColorLimits');
    end
    assert(rejected);
    fig = figure('Visible','off');
    cleanup = onCleanup(@() close(fig));
    ax = axes(fig);
    data = [-1,NaN,3; -.1,0,.1];
    h = imagesc(ax,data);
    before = clim(ax);
    colormap(ax,helm_solution_colormap(before));
    assert(isequal(clim(ax),before) && isequaln(h.CData,data));
    fprintf('Solution palette: original Fig22 map, scaling, zero, and data/CLim checks passed.\n');
end
