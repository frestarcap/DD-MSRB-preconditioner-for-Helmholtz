function test_review_saved_evidence
%TEST_REVIEW_SAVED_EVIDENCE Assertions for the repaired paper audit tables.
    s=load(helm_output_path('outputs','revision_audit_20260905_repaired','audit.mat'),'audit');
    a=s.audit;
    assert(all(a.stability.max_snapshot_residual<=1e-10));
    assert(all(isfinite(a.stability.max_reduced_condition)));
    assert(all(a.reference.failed_native_reference_flags==0));
    assert(all(a.reference.max_reference_H_residual<=1e-10));
    assert(all(a.compression.greedy_tail<=a.compression.budget));
    assert(all(a.compression.saving_percent>0));
    assert(all(a.offline.total_offline_s>0));
    for name={'lens','waveguide'}
        f=helm_output_path('outputs','helmholtz_3d_candidates','large', ...
            ['helmholtz_3d_candidates_' name{1} '_frozen_holdout_20260905.csv']);
        t=readtable(f,'TextType','string'); t=t(t.method~="direct",:);
        assert(numel(unique(t.mu))==5 && height(t)==15);
        assert(all(t.flag==0) && all(t.final_relres<=1e-6));
        assert(all(t.speedup_vs_shifted(t.method=="dd")>1));
    end
    fprintf('Repaired snapshots, references, compression and frozen 3D holdouts verified.\n');
end
