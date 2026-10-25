function benchmark_halfspace_residual_repair
% Uniform one-step residual minimization before unchanged ILU fallback.
    maxNumCompThreads(1);
    s=load(helm_output_path('outputs','jcp_shifted_20260914','frozen_cohorts.mat'),'cohorts'); c=s.cohorts{1};
    out=helm_output_path('outputs','halfspace_residual_repair_20260922'); rows=table();
    names={'shifted','global','dd','global_repair','dd_repair'};
    opts=struct('tol',c.tol,'restart',40,'maxit',20,'reorthogonalize',true);
    ilu_opts=struct('solver','ilu','ilu_droptol',1e-3);
    for qi=1:numel(c.parameters)
        mu=c.parameters(qi); alpha=mu^(-2)-1; A=c.A_base+alpha*c.A_right; b=c.b; nb=norm(b);
        mass=c.omega^2*ones(numel(b),1)+alpha*full(diag(c.A_right));
        for rep=0:5
            for mi=circshift(1:5,[0,mod(qi+rep,5)])
                tt=tic; setup_s=0; repair_s=0; iters=0; accepted=false;
                if mi==1, u=zeros(size(b)); else
                    idx=1+ismember(mi,[3,5]); [u,~]=prefix(A,b,mu,c.models{idx},c.track_tol);
                end
                r=b-A*u; rho=norm(r)/nb; before=rho;
                if mi>=4 && rho>c.tol
                    t=tic; ar=A*r; candidate=u+r*((ar'*r)/(ar'*ar));
                    rr=b-A*candidate; newrho=norm(rr)/nb;
                    if isfinite(newrho) && newrho<rho, u=candidate; rho=newrho; end
                    accepted=rho<=c.tol; repair_s=toc(t);
                end
                if rho>c.tol
                    t=tic; Ashift=A+1i*.2*spdiags(mass,0,numel(b),numel(b));
                    [M,~]=build_shifted_laplacian_preconditioner(Ashift,ilu_opts); setup_s=toc(t);
                    [u,info]=shifted_laplacian_gmres(@(v)A*v,M,b,u,opts); iters=info.iter_count;
                    assert(info.flag==0); clear M Ashift;
                end
                final=norm(b-A*u)/nb; total_s=toc(tt);
                assert(final<=c.tol); err=norm(u-c.references(:,qi))/norm(c.references(:,qi));
                if rep>0
                    method=string(names{mi}); rows=[rows;table(qi,mu,rep,method,total_s,setup_s,repair_s,iters,before,final,err,accepted)]; %#ok<AGROW>
                end
            end
        end
        fprintf('BENCH query=%d/23 passed\n',qi);
        writetable(rows,fullfile(out,'timings.csv'));
    end
    save(fullfile(out,'timings.mat'),'rows','opts','ilu_opts');
end

function [u,history]=prefix(A,b,mu,model,track_tol)
    n=numel(b); depth=numel(model.layers); nb=norm(b);
    u=zeros(n,1); history=1;
    if model.fixed_pod
        u=apply_inverse(A,b,mu,model.layers{1});
        history=[1;norm(b-A*u)/nb]; return;
    end
    Q=zeros(n,depth+1); Z=zeros(n,depth); H=zeros(depth+1,depth); Q(:,1)=b/nb;
    for j=1:depth
        Z(:,j)=apply_inverse(A,Q(:,j),mu,model.layers{j});
        [Q(:,j+1),H(1:j,j),H(j+1,j),stop]=rom_arnoldi_step(A*Z(:,j),Q(:,1:j));
        g=zeros(j+1,1); g(1)=nb;
        if isfield(model.layers{j},'Ar_base'), y=H(1:j+1,1:j)\g;
        else, y=lsqminnorm(H(1:j+1,1:j),g); end
        u=Z(:,1:j)*y;
        history(end+1,1)=norm(b-A*u)/nb; %#ok<AGROW>
        if stop || history(end)<=track_tol, break; end
    end
end

function z=apply_inverse(A,b,mu,layer)
    sp=layer.space;
    br=rom_reduced_space_utils('project',sp,b);
    if isfield(layer,'Ar_base')
        Ar=layer.Ar_base+(mu^(-2)-1)*layer.Ar_right; coeff=Ar\br;
    else
        Ar=full(rom_reduced_space_utils('assemble_reduced_matrix',A,sp));
        if rcond(Ar)<1e-12, coeff=pinv(Ar)*br; else, coeff=Ar\br; end
    end
    z=rom_reduced_space_utils('lift',sp,coeff);
end
