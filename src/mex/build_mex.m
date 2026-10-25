function folder = build_mex()
%BUILD_MEX Compile the two-pass CGS kernel of the 3D SL GMRES (optional).
%   Requires a C compiler supported by MATLAB.  The binary goes to
%   build/mex/<arch>, which startup.m adds to the path.  The kernel is checked
%   against the MATLAB implementation of the same orthogonalization.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    folder = fullfile(root, 'build', 'mex', computer('arch'));
    if ~isfolder(folder), mkdir(folder); end
    source = fullfile(fileparts(mfilename('fullpath')), 'dmsrb_cgs2_mex.c');
    mex('-R2018a', '-O', '-outdir', folder, source, '-lmwblas');
    addpath(folder);
    rng(7);
    V = orth(complex(randn(500, 12), randn(500, 12)));
    w = complex(randn(500, 1), randn(500, 1));
    [w1, h1] = dmsrb_cgs2_mex(V, w, 12);
    h = V' * w; v = w - V * h; h2 = V' * v; w2 = v - V * h2;
    assert(norm(w1 - w2) <= 1e-13 * norm(w) && norm(h1 - (h + h2)) <= 1e-13 * norm(w), ...
        'MEX kernel disagrees with the MATLAB CGS2 implementation.');
    fprintf('dmsrb_cgs2_mex built and checked: %s\n', folder);
end
