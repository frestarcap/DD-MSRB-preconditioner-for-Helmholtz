function startup()
%STARTUP Add the DD-MSRB code to the MATLAB path (run from the repository root).
    root = fileparts(mfilename('fullpath'));
    addpath(fullfile(root, 'src'));
    addpath(fullfile(root, 'src', 'mex'));
    addpath(fullfile(root, 'figures'));
    for e = {'exp1_halfspace', 'exp2_ecc', 'exp3_waveguide', 'appendixA'}
        addpath(fullfile(root, 'experiments', e{1}));
    end
    if isfolder(fullfile(root, 'build', 'mex')), addpath(fullfile(root, 'build', 'mex')); end
end
