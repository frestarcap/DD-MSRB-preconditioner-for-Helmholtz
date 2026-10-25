function [q,h,hn,breakdown] = rom_arnoldi_step(w,Q)
%ROM_ARNOLDI_STEP Two-pass MGS and scale-relative near-breakdown detection.
    scale=norm(w); h=zeros(size(Q,2),1,'like',w);
    for pass=1:2
        for i=1:size(Q,2)
            dh=Q(:,i)'*w; h(i)=h(i)+dh; w=w-Q(:,i)*dh;
        end
    end
    hn=norm(w);
    breakdown=~isfinite(hn) || hn<=100*eps*max(scale,realmin);
    q=zeros(size(w),'like',w);
    if ~breakdown, q=w/hn; end
end
