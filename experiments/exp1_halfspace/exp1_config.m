function c = exp1_config()
%EXP1_CONFIG Settings of Experiment 1 (half-space), as stated in the paper.
    c = struct();
    c.train = linspace(0.8, 0.95, 24);            % training speeds c_right
    c.test = test_parameters(c.train, 1);          % 23 test queries, one per training interval
    c.representative = 10;                         % c_right = 0.8622
    c.coverage_counts = [6, 12, 24];               % training-set sizes of the coverage study
    c.coverage_test = [0.785, 0.8053, 0.8297, 0.8513, 0.8761, 0.9007, 0.9229, 0.9431, 0.965];
    c.sensitivity_test = [0.8053, 0.8297, 0.8761, 0.9229, 0.9431];   % partitions and frequency
    c.partitions = {'material_depth', 'merged', 'signed_depth', 'uniform4x4'};
    c.scales = [1, 1.5, 2, 4];                     % omega = 53.5*scale at fixed points per wavelength
    c.eta = 1e-5 * [1, 1, 1];                      % k = 3 reduced spaces
    c.ss_eta = 1e-5;                               % SS-ROM truncation
    c.tol = 1e-12;                                 % true relative residual target (main, coverage)
    c.sensitivity_tol = 1e-10;                     % partitions and frequency
    c.beta = 0.2;                                  % SL shift
    c.ilu_droptol = 1e-3;
    c.restart = 40;
    c.maxit = 20;
    c.repetitions = 5;
end

function p = test_parameters(train, seed)
% One uniformly distributed query inside each training interval (fixed seed).
    old = rng;
    rng(seed, 'twister');
    lo = train(1:end-1); hi = train(2:end);
    p = lo + rand(size(lo)) .* (hi - lo);
    rng(old);
end
