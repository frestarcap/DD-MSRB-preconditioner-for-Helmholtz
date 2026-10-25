function [q,z,h,hn,info] = rom_verified_snapshot_step(A,zrom,Q,Z)
%ROM_VERIFIED_SNAPSHOT_STEP Verify the recurrence before admitting a snapshot.
    [q,h,hn,stop]=rom_arnoldi_step(A*zrom,Q);
    info=struct('hsub',hn,'breakdown',stop,'direct_fallback',false, ...
        'recurrence_residual',NaN,'accepted_residual',NaN);
    z=zeros(size(zrom),'like',zrom);
    if stop, return; end
    z=(zrom-Z*h)/hn;
    info.recurrence_residual=norm(A*z-q);
    if ~isfinite(info.recurrence_residual) || info.recurrence_residual>1e-10
        z=A\q;
        info.direct_fallback=true;
    end
    info.accepted_residual=norm(A*z-q);
    assert(isfinite(info.accepted_residual) && info.accepted_residual<=1e-10, ...
        'ROM:UnverifiedSnapshot','Inverse-action snapshot failed the residual check.');
end
