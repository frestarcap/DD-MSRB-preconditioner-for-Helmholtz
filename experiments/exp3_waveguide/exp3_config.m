function c = exp3_config(omega)
%EXP3_CONFIG Settings of Experiment 3 (3D Gaussian waveguide), as stated in the paper.
    c = struct('omega', omega);
    c.train = linspace(0.2, 0.6, 8);                % training parameters mu
    c.test = [0.237, 0.317, 0.483];                 % common test queries at every omega
    c.added = c.train(1:end-1) + diff(c.train) .* [0.31, 0.67, 0.43, 0.59, 0.37, 0.71, 0.53];
    c.representative = 0.317;
    c.eta = [0.003, 0.008, 0.02];                   % k = 3 reduced spaces
    c.tol = 1e-6;                                   % true relative residual target
    c.beta = 2;                                     % SL shift
    c.mg = struct('mg_max_levels', 6, 'mg_coarse_max_dofs', 12000, 'mg_pre_sweeps', 2, ...
        'mg_post_sweeps', 2, 'mg_jacobi_weight', 0.6, 'mg_cycles_per_apply', 1);
    c.restart = 40;
    c.maxit = 40;                                   % online SL cycles
    c.snapshot_tol = 1e-7;                          % training solves
    c.snapshot_maxit = 40;
    if omega >= 60, c.snapshot_maxit = 120; end     % omega = 60 needs more cycles
    c.orthogonalization = 'cgs2';
    if exist('dmsrb_cgs2_mex', 'file') == 3, c.orthogonalization = 'cgs2_mex'; end
    c.workers = 8;                                  % concurrent training solves
    c.worker_threads = 4;
    c.online_threads = 8;
    if omega >= 60, c.workers = 4; end
    if omega < 24, c.workers = 1; end               % small verification grids run serially
end
