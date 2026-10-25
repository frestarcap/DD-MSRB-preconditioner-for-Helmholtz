function [x, info] = gmres(Afun, Msolve, b, x0, opts)
%GMRES Right-preconditioned restarted GMRES for the shifted-Laplacian (SL) steps.
%
%   [x, info] = dmsrb.sl.gmres(Afun, Msolve, b, x0, opts)
%   applies restarted GMRES to A M^{-1} y = b - A*x0 and returns
%   x = x0 + M^{-1} y.  The preconditioner action Msolve(r) is typically a
%   shifted-Laplacian solve.  Residual histories are measured in the original
%   equation, norm(b - A*x) / norm(b), so they can be compared directly with
%   unpreconditioned Helmholtz residuals.

    if nargin < 5
        opts = struct();
    end
    opts = apply_shifted_laplacian_gmres_defaults(opts);

    n = numel(b);
    if nargin < 4 || isempty(x0)
        x0 = zeros(n, 1);
    end

    x = x0;
    nb = max(norm(b), 1e-16);
    r = b - Afun(x);
    beta = norm(r);

    relres_hist = beta / nb;
    iter_count = 0;
    op_apply_count = 1;
    precond_apply_count = 0;
    flag = 1;
    stop_reason = 'maxit';
    tsolve = tic;

    if relres_hist(1) <= opts.tol
        flag = 0;
        stop_reason = 'residual';
        info = finish_shifted_laplacian_info(tsolve, iter_count, op_apply_count, ...
            precond_apply_count, relres_hist, flag, stop_reason, x);
        return;
    end

    for outer = 1:opts.maxit
        V = zeros(n, opts.restart + 1, 'like', r);
        if strcmp(opts.orthogonalization,'cgs2_mex'), V=complex(V); end
        if opts.store_preconditioned_basis
            Z = zeros(n, opts.restart, 'like', r);
        end
        H = zeros(opts.restart + 1, opts.restart);
        rotations = cell(opts.restart, 1);
        transformed_rhs = zeros(opts.restart + 1, 1);
        transformed_rhs(1) = beta;
        V(:, 1) = r / beta;
        inner_last = 0;
        cycle_finished = false;

        for j = 1:opts.restart
            inner_last = j;
            iter_count = iter_count + 1;

            z = Msolve(V(:, j));
            precond_apply_count = precond_apply_count + 1;
            if opts.store_preconditioned_basis
                Z(:, j) = z;
            end
            w = Afun(z);
            op_apply_count = op_apply_count + 1;

            if strcmp(opts.orthogonalization, 'cgs2_mex')
                [w,H(1:j,j)]=dmsrb_cgs2_mex(V,w,j);
            elseif strcmp(opts.orthogonalization, 'cgs2')
                % Two BLAS matrix-vector passes; always reorthogonalize.
                h = V(:, 1:j)' * w;
                w = w - V(:, 1:j) * h;
                h2 = V(:, 1:j)' * w;
                w = w - V(:, 1:j) * h2;
                H(1:j, j) = h + h2;
            else
                for i = 1:j
                    H(i, j) = V(:, i)' * w;
                    w = w - H(i, j) * V(:, i);
                end
                if opts.reorthogonalize
                    for i = 1:j
                        h_corr = V(:, i)' * w;
                        H(i, j) = H(i, j) + h_corr;
                        w = w - h_corr * V(:, i);
                    end
                end
            end
            H(j + 1, j) = norm(w);
            arnoldi_subdiagonal = H(j + 1, j);
            if H(j + 1, j) > 0
                V(:, j + 1) = w / H(j + 1, j);
            end

            for i = 1:j - 1
                H(i:i + 1, j) = rotations{i} * H(i:i + 1, j);
            end
            [rotations{j}, rotated_column] = planerot(H(j:j + 1, j));
            H(j:j + 1, j) = rotated_column;
            transformed_rhs(j:j + 1) = rotations{j} * transformed_rhs(j:j + 1);
            relres = abs(transformed_rhs(j + 1)) / nb;
            relres_hist(end + 1, 1) = relres; %#ok<AGROW>

            if opts.verbose
                fprintf('  SL-GMRES iter %4d: ||r||/||b|| = %.3e\n', iter_count, relres);
            end

            arnoldi_breakdown = arnoldi_subdiagonal <= ...
                10 * eps(max(1, norm(H(1:j, j))));
            if relres <= opts.tol || arnoldi_breakdown
                y = solve_small_least_squares(H(1:j, 1:j), transformed_rhs(1:j));
                if opts.store_preconditioned_basis
                    x_candidate = x + Z(:, 1:j) * y;
                else
                    x_candidate = x + Msolve(V(:, 1:j) * y);
                    precond_apply_count = precond_apply_count + 1;
                end
                r_candidate = b - Afun(x_candidate);
                op_apply_count = op_apply_count + 1;
                true_relres = norm(r_candidate) / nb;
                relres_hist(end) = true_relres;
            else
                true_relres = Inf;
            end

            if true_relres <= opts.tol
                x = x_candidate;
                flag = 0;
                stop_reason = 'residual';
                info = finish_shifted_laplacian_info(tsolve, iter_count, ...
                    op_apply_count, precond_apply_count, relres_hist, flag, stop_reason, x);
                return;
            end
            if arnoldi_breakdown
                x = x_candidate;
                r = r_candidate;
                stop_reason = 'breakdown';
                cycle_finished = true;
                break;
            end
        end

        if ~cycle_finished
            y = solve_small_least_squares(H(1:inner_last, 1:inner_last), ...
                transformed_rhs(1:inner_last));
            if opts.store_preconditioned_basis
                x = x + Z(:, 1:inner_last) * y;
            else
                x = x + Msolve(V(:, 1:inner_last) * y);
                precond_apply_count = precond_apply_count + 1;
            end
            r = b - Afun(x);
            op_apply_count = op_apply_count + 1;
        end
        beta = norm(r);
        relres_hist(end) = beta / nb;
        if beta / nb <= opts.tol
            flag = 0;
            stop_reason = 'residual';
            break;
        end
        if cycle_finished
            break;
        end
    end

    info = finish_shifted_laplacian_info(tsolve, iter_count, op_apply_count, ...
        precond_apply_count, relres_hist, flag, stop_reason, x);
end

function opts = apply_shifted_laplacian_gmres_defaults(opts)
    opts = set_struct_default(opts, 'restart', 40);
    opts = set_struct_default(opts, 'maxit', 20);
    opts = set_struct_default(opts, 'tol', 1e-12);
    opts = set_struct_default(opts, 'verbose', false);
    opts = set_struct_default(opts, 'reorthogonalize', true);
    opts = set_struct_default(opts, 'orthogonalization', 'mgs');
    opts = set_struct_default(opts, 'store_preconditioned_basis', true);
    opts = set_struct_default(opts, 'fixed_linear_preconditioner', false);
    assert(ismember(opts.orthogonalization, {'mgs', 'cgs2', 'cgs2_mex'}));
    if strcmp(opts.orthogonalization,'cgs2_mex')
        assert(exist('dmsrb_cgs2_mex','file')==3,'Build the CGS2 MEX kernel first.');
    end
    assert(opts.store_preconditioned_basis || opts.fixed_linear_preconditioner, ...
        'Discarding Z requires an explicitly declared fixed linear preconditioner.');
    opts.restart = max(1, round(opts.restart));
    opts.maxit = max(1, round(opts.maxit));
end

function s = set_struct_default(s, name, value)
    if ~isfield(s, name) || isempty(s.(name))
        s.(name) = value;
    end
end

function y = solve_small_least_squares(H, g)
    [U, S, V] = svd(H, 'econ');
    sigma = diag(S);
    if isempty(sigma)
        y = zeros(size(H, 2), 1);
        return;
    end
    tol = max(size(H)) * eps(max(sigma));
    keep = sigma > tol;
    if ~any(keep)
        y = zeros(size(H, 2), 1);
        return;
    end
    y = V(:, keep) * ((U(:, keep)' * g) ./ sigma(keep));
end

function info = finish_shifted_laplacian_info(tsolve, iter_count, op_apply_count, ...
        precond_apply_count, relres_hist, flag, stop_reason, x)
    info = struct();
    info.total_time = toc(tsolve);
    info.iter_count = iter_count;
    info.op_apply_count = op_apply_count;
    info.precond_apply_count = precond_apply_count;
    info.relres_hist = relres_hist(:);
    info.relres_final = relres_hist(end);
    info.flag = flag;
    info.stop_reason = stop_reason;
    info.u_final = x;
end
