function test_halfspace_dd_paper
%TEST_HALFSPACE_DD_PAPER Audit the saved numerical evidence behind Experiment 1.
    s = load(helm_output_path('outputs','halfspace_dd_paper','paper_data.mat'),'paper');
    p = s.paper;
    assert(isequal(p.global.test.params,p.dd.test.params,p.aux.nr.params));
    assert(numel(p.global.test.params)==23);
    assert(isequal(p.global.cfg.train_params,p.dd.cfg.train_params));
    assert(isequal(p.global.cfg.corr_pod_singular_value_tols,[1e-5 1e-5 1e-5]));
    assert(isequal(p.dd.cfg.corr_block3_global_tail_tols,[1e-5 1e-5 1e-5]));
    assert(isequaln(halfspace_paper_dd_config(p.dd.cfg),p.dd.cfg));
    assert(p.dd.cfg.omega==53.5 && p.dd.cfg.Nphys==125 && p.dd.cfg.npml==32);
    assert(p.global.cfg.fp_tol==1e-12 && p.dd.cfg.fp_tol==1e-12);
    assert(p.global.cfg.online_helm_accept_tol==1e-12 && p.dd.cfg.online_helm_accept_tol==1e-12);
    assert(numel(p.partition.blocks)==10);
    idx = vertcat(p.partition.blocks.idx);
    assert(isequal(sort(idx),(1:35721)'));
    assert(sum(p.layer_table.global_entries)*16==p.main_rows.basis_bytes(1));
    assert(sum(p.layer_table.dd_entries)*16==p.main_rows.basis_bytes(2));
    assert(isequal(p.layer_table.global_rank(:),p.aux.nr_ranks(:)));
    for j = 1:3
        d = p.dd.layers{j}.diagnostics;
        assert(d.global_tail<=1e-5);
        assert(abs(sqrt(sum(d.energy.*d.local_tails.^2)/sum(d.energy))-d.global_tail)<1e-14);
    end
    for k = 1:2
        assert(p.main_rows.rom_payload_bytes(k)==p.main_rows.basis_bytes(k)+ ...
            p.main_rows.operator_bytes(k)+p.main_rows.partition_bytes(k));
    end
    cfg = p.global.cfg; cfg.setup_only = true;
    r = helm_waveholtz_correction_rom(cfg);
    for name = {'global','dd'}
        a = p.(name{1}); t = a.test;
        accepted = strcmp(t.online_fp_stop_reason,'helmholtz_residual');
        assert(all(t.online_precorr_relres_final(accepted)<=1e-12));
        assert(all(t.online_fp_iter(accepted)==0));
        assert(all(t.online_lift_time(accepted)==0 & t.online_fp_time(accepted)==0));
        assert(all(isnan(t.online_fp_relres_final(accepted))));
        assert(all(cellfun(@isempty,t.online_fp_relres_hist(accepted))));
        assert(all(t.online_fp_relres_final(~accepted)<=1e-12));
        assert(max(abs(t.online_total_time-t.online_precorr_time- ...
            t.online_lift_time-t.online_fp_time))<1e-12);
        for k = 1:23
            A = r.problem.A_base+(1/t.params(k)^2-1)*r.problem.A_right;
            residual = norm(A*t.online_final_u(:,k)-r.rhs)/norm(r.rhs);
            err = norm(t.online_final_u(:,k)-t.helm_u_ref(:,k))/norm(t.helm_u_ref(:,k));
            assert(abs(residual-a.final_helm_residual(k))<1e-16);
            assert(residual<=1e-12 && err<1e-12);
        end
    end
    assert(isequal(p.main_rows.helm_accepted_count,[23;21]));
    assert(isequal(p.tolerance_rows.wh_tolerance,[1e-14;1e-14;1e-13;1e-13;1e-12;1e-12]));
    fprintf('Saved half-space paper evidence checks passed.\n');
end
