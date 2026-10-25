function probe_halfspace_residual_repair
% Frozen full 23-query cohort; exploratory residual correction, no retraining.
    maxNumCompThreads(1);
    s=load(helm_output_path('outputs','jcp_shifted_20260914','frozen_cohorts.mat'),'cohorts');
    c=s.cohorts{1}; out=helm_output_path('outputs','halfspace_residual_repair_20260922');
    if ~exist(out,'dir'), mkdir(out); end
    rows=table();
    for qi=1:numel(c.parameters)
        mu=c.parameters(qi); A=c.A_base+(mu^(-2)-1)*c.A_right; b=c.b; nb=norm(b);
        for mi=1:2
            model=c.models{mi}; [u,~]=prefix(A,b,mu,model,c.track_tol);
            rho=norm(b-A*u)/nb;
            for variant=1:6
                v=u; tt=tic; steps=0;
                if rho>c.tol
                    residual=b-A*v;
                    if variant<=3
                        z=apply_inverse(A,residual,mu,model.layers{variant});
                        az=A*z; alpha=(az'*residual)/(az'*az); v=v+alpha*z; steps=1;
                    elseif variant==4
                        [z,h]=prefix(A,residual,mu,model,c.track_tol);
                        az=A*z; alpha=(az'*residual)/(az'*az); v=v+alpha*z; steps=numel(h)-1;
                    else
                        target=0.9*c.tol/rho;
                        if variant==5
                            [z,~,~,~,hist]=gmres(A,residual,40,target,1);
                        else
                            D=spdiags(diag(A),0,numel(b),numel(b));
                            [z,~,~,~,hist]=gmres(A,residual,40,target,1,D);
                        end
                        v=v+z; steps=numel(hist)-1;
                    end
                end
                elapsed=toc(tt); final=norm(b-A*v)/nb;
                err=norm(v-c.references(:,qi))/norm(c.references(:,qi));
                rows=[rows;table(qi,string(c.names{mi}),mu,variant,rho,final,steps,elapsed,err)]; %#ok<AGROW>
                if rho>c.tol, fprintf('REPAIR q=%d %s variant=%d before=%.4e after=%.4e time=%.4g\n',qi,c.names{mi},variant,rho,final,elapsed); end
            end
        end
    end
    writetable(rows,fullfile(out,'probe.csv')); save(fullfile(out,'probe.mat'),'rows');
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
