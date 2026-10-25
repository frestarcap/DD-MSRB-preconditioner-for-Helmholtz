function colors = helm_solution_colormap(limits, n)
%HELM_SOLUTION_COLORMAP Contrast-enhanced blue/white/red for signed fields.
% Pass the existing CLim so zero stays neutral even for asymmetric ranges.
% Only the color mapping is nonlinear; amplitudes and colorbar labels are not.
    if nargin < 1 || isempty(limits), limits = [-1, 1]; end
    if nargin < 2 || isempty(n), n = 257; end
    validateattributes(limits, {'numeric'}, {'real','finite','vector','numel',2});
    validateattributes(n, {'numeric'}, {'scalar','integer','>=',2});
    assert(limits(2) > limits(1), 'helm:InvalidColorLimits', ...
        'Color limits must be strictly increasing.');
    amplitudes = linspace(limits(1), limits(2), n).' / max(abs(limits));
    levels = asinh(8 * amplitudes) / asinh(8);
    colors = interp1([-1; -.5; 0; .5; 1], ...
        [0.10 0.20 0.55; 0.15 0.55 0.90; 1 1 1; ...
         0.90 0.30 0.18; 0.55 0.02 0.10], levels, 'linear');
end
