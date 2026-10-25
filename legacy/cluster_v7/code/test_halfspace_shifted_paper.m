function test_halfspace_shifted_paper
%TEST_HALFSPACE_SHIFTED_PAPER Verify the saved common-tolerance Figure 6 run.
    s = load(helm_output_path('outputs','helm_shifted_laplacian_completion', ...
        'helm_shifted_laplacian_completion_paper_tol1e12.mat'),'results');
    r = s.results; t = r.test;
    assert(r.cfg.shifted_laplacian_tol == 1e-12);
    assert(r.cfg.online_helm_accept_tol == 1e-12);
    assert(r.cfg.helm_baseline_tol == 1e-12);
    assert(isequal(r.summary.corr_ranks(:),[7;11;19]));
    assert(numel(t.params) == 23 && all(t.online_fp_iter == 0));
    assert(all(t.online_combined_transition_idx == 4));
    assert(all(cellfun(@numel,t.online_combined_relres_hist) == 4));
    assert(all(t.online_precorr_relres_final <= 1e-12));
    assert(all(t.plain_fp_relres_final <= 1e-12));
    assert(all(t.plain_fp_iter <= 90));
    rho = zeros(numel(t.params),1);
    for k = 1:numel(t.params)
        A = r.problem.A_base + (1/t.params(k)^2-1)*r.problem.A_right;
        rho(k) = norm(A*t.online_final_u(:,k)-r.rhs)/norm(r.rhs);
    end
    assert(all(rho <= 1e-12));
    fprintf('Figure 6: 23 ROM queries accepted without completion; max true residual %.3e.\n',max(rho));
end
