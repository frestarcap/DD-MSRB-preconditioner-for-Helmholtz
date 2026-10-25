function test_rom_verified_snapshot_step
%TEST_ROM_VERIFIED_SNAPSHOT_STEP Exercise accurate, contaminated, and zero steps.
    rng(17); A=diag(2:21)+0.01*randn(20); q=randn(20,1); q=q/norm(q);
    exact=A\q; zrom=exact+0.01*randn(20,1);
    [qn,z,~,~,info]=rom_verified_snapshot_step(A,zrom,q,exact);
    assert(~info.breakdown && norm(A*z-qn)<1e-10);
    [qn,z,~,~,info]=rom_verified_snapshot_step(A,zrom,q,exact+1e-3);
    assert(info.direct_fallback && norm(A*z-qn)<1e-10);
    [~,z,~,~,info]=rom_verified_snapshot_step(A,zeros(20,1),q,exact);
    assert(info.breakdown && all(z==0));
    [Q,~]=qr(randn(20,4),0); w=Q(:,1)+1e-9*randn(20,1);
    [qn,~,~,stop]=rom_arnoldi_step(w,Q);
    assert(~stop && norm(Q'*qn)<1e-12);
    fprintf('Verified snapshot and Arnoldi tests passed.\n');
end
