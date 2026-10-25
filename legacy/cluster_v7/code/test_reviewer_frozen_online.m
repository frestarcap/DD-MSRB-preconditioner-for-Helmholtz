function test_reviewer_frozen_online
%TEST_REVIEWER_FROZEN_ONLINE Validate frozen-model deployment guards.
% Positive completion is exercised by the separate scale-0.25 prefix smoke.
    out=helm_output_path('outputs','reviewer_20260909','high_frequency_waveholtz');
    for method={'global','dd'}
        s=load(fullfile(out,[method{1} '_scale0.25_prefix_result.mat']),'result');
        assert(s.result.rows.completion_iterations>0);
        assert(s.result.rows.true_residual<=s.result.cfg.online_true_helm_tol);
        cfg=s.result.cfg;
        cfg.omega=cfg.omega*1.01;
        expect_mismatch(cfg);
    end
    cfg=s.result.cfg;
    saved=load(cfg.frozen_online_file);
    saved.rhs(1)=saved.rhs(1)+1;
    scratch=fullfile(out,'guard_test_mismatched_rhs.mat');
    save(scratch,'-struct','saved','-v7.3');
    cleanup=onCleanup(@() delete(scratch));
    cfg.frozen_online_file=scratch;
    expect_mismatch(cfg);
    fprintf('Frozen deployment: positive smoke, operator and RHS guards passed.\n');
end

function expect_mismatch(cfg)
    rejected=false;
    try
        helm_waveholtz_correction_rom(cfg);
    catch e
        rejected=contains(e.message,'Frozen online model and requested Helmholtz problem differ');
        if ~rejected, rethrow(e); end
    end
    assert(rejected,'A mismatched model must be rejected before deployment.');
end
