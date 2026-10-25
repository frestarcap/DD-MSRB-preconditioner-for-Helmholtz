function test_marginal_storage_budget
% Low-energy blocks may be omitted; tiny tails must not cancel numerically.
    part=struct('n',102,'idx_near',1,'idx_far',(2:101).','idx_pml',102, ...
        'count_near',1,'count_far',100,'count_pml',1);
    E=zeros(102,3); E(1,1)=1; E(2,2)=1e-4; E(3,3)=1e-5;
    cfg=struct('rank_global_tail_tol',1e-3,'rank_global_tail_strategy','marginal_storage');
    s=rom_reduced_space_utils('build_block3_space',E,part,cfg);
    assert(isequal([s.kn,s.kf,s.kp],[1,0,0]));
    assert(s.block_meta.near.global_relative_tail<=cfg.rank_global_tail_tol);
    % Explicit local constraints remain supported, but are not the paper rule.
    cfg.rank_tail_tol_far=1e-3;
    s=rom_reduced_space_utils('build_block3_space',E,part,cfg);
    assert(s.kf==2);
    cfg=rmfield(cfg,'rank_tail_tol_far');
    E(2,2)=1e-8; E(3,3)=1e-9; cfg.rank_global_tail_tol=1e-10;
    s=rom_reduced_space_utils('build_block3_space',E,part,cfg);
    assert(isequal([s.kn,s.kf,s.kp],[1,2,0]));
    assert(s.block_meta.near.global_relative_tail<=cfg.rank_global_tail_tol);
    fprintf('MARGINAL_STORAGE_BUDGET_PASS\n');
end
