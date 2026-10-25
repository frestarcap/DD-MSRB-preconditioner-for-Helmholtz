function [Msolve, info] = build_shifted_laplacian_preconditioner(Ashift, opts)
%BUILD_SHIFTED_LAPLACIAN_PRECONDITIONER Build a shifted inverse action.

    if nargin < 2
        opts = struct();
    end
    opts = set_default(opts, 'solver', 'lu');
    opts = set_default(opts, 'ilu_type', 'ilutp');
    opts = set_default(opts, 'ilu_droptol', 1e-2);
    opts = set_default(opts, 'ilu_thresh', 1.0);
    opts = set_default(opts, 'ilu_milu', 'off');
    opts = set_default(opts, 'grid_size', []);

    solver = lower(strtrim(char(string(opts.solver))));
    t_setup = tic;
    switch solver
        case {'lu', 'exact_lu', 'direct'}
            factor = decomposition(Ashift, 'lu');
            Msolve = @(r) factor \ r;
            factor_stats = whos('factor');
            solver = 'lu';

        case {'ilu', 'ilutp'}
            ilu_setup = struct('type', opts.ilu_type, ...
                'droptol', opts.ilu_droptol, 'thresh', opts.ilu_thresh, ...
                'milu', opts.ilu_milu);
            [L, U] = ilu(Ashift, ilu_setup);
            Msolve = @(r) U \ (L \ r);
            factor_stats = whos('L', 'U');
            solver = 'ilu';

        case {'mg', 'multigrid', 'geometric_multigrid'}
            [Msolve, mg_info] = build_shifted_laplacian_multigrid_preconditioner( ...
                Ashift, opts.grid_size, opts);
            factor_stats = struct('bytes', mg_info.factor_bytes);
            solver = 'mg';

        otherwise
            error('Unknown shifted-Laplacian solver "%s".', solver);
    end

    info = struct();
    info.solver = solver;
    info.setup_time = toc(t_setup);
    info.factor_bytes = sum([factor_stats.bytes]);
    info.ilu_type = opts.ilu_type;
    info.ilu_droptol = opts.ilu_droptol;
    info.ilu_thresh = opts.ilu_thresh;
    info.ilu_milu = opts.ilu_milu;
    if exist('mg_info', 'var')
        info.multigrid = mg_info;
    end
end

function s = set_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end
